import torch.nn as nn
import torch
import numpy as np
from scene.camera import Camera
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class NeRF(nn.Module):
    """NeRF model"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.num_embeddings = cfg['num_embeddings']
        self.hidden_dim = cfg['hidden_dim']
        self.positional_encoding = PositionalEncoding(self.num_embeddings)

        input_encoding_size = 6 + 6 * 2 * self.num_embeddings
        pos_encoding_size = 3 + 3 * 2 * self.num_embeddings
        dir_encoding_size = 3 + 3 * 2 * self.num_embeddings

        self.initial_layer = nn.Sequential(nn.Linear(input_encoding_size, self.hidden_dim), nn.ReLU())
        self.hidden_layer = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU())
        self.skip_layer1 = nn.Sequential(nn.Linear(self.hidden_dim  + pos_encoding_size, self.hidden_dim), nn.ReLU())
        self.skip_layer2 = nn.Sequential(nn.Linear(self.hidden_dim  + dir_encoding_size, int(self.hidden_dim/2)), nn.ReLU())
        self.rgb = nn.Sequential(nn.Linear(int(self.hidden_dim/2), 3), nn.Sigmoid())
        self.density = nn.Sequential(nn.Linear(self.hidden_dim, 1), nn.Sigmoid())

    def forward(self, ray_o, ray_d):
        """
        Args:
            ray_o (torch.tensor): ray origin (N, 3)
            ray_d (torch.tensor): ray direction (N, 3)
        """
        samples = Camera.get_samples(ray_o, ray_d, step=self.cfg['step'])     # (N, T, 3)

        position = self.positional_encoding(samples)   # (N, T, 3+3*2*num_embeddings)
        direction = self.positional_encoding(ray_d)    # (N, 3+3*2*num_embeddings)
        # tile to concatenate
        direction = direction.unsqueeze(1)
        direction = direction.expand(-1, position.shape[1], -1)

        x_encoded = torch.cat((position, direction), dim=-1)    # (N, T, 6+2*(3+2*num_embeddings))

        # Architecture from original paper
        x = self.initial_layer(x_encoded)
        for i in range(5):  # Assuming 8 layers in total, skip connection after the 4th
            x = self.hidden_layer(x)
            if i == 3:  # Add skip connection after the 4th layer
                x = torch.cat((x, position), dim=-1)
                x = self.skip_layer1(x)

        density = self.density(x)
        x = torch.cat((x, direction), dim=-1)
        x = self.skip_layer2(x)
        rgb = self.rgb(x)

        return density, rgb
      

class PositionalEncoding(nn.Module):
    """Encoding into corresponding Fourier features.
    Args:
        x (torch.tensor): input (b, 3, 1)
    Returns:
        torch.tensor: positional encoding (b, 6 + 2 * positional_encoding_embeddings)
    """
    def __init__(self, num_embeddings):
        super().__init__()
        self.num_embeddings = num_embeddings

    def forward(self, x):
        embeddings = []
        embeddings.extend([torch.sin(torch.pi * x), 
                          torch.cos(torch.pi * x)])
        for i in range(1, self.num_embeddings):
            embeddings.extend([torch.sin(torch.pi * 2 * i * x),
                                torch.cos(torch.pi * 2 * i * x)])
        embeddings.append(x)
        embeddings = torch.cat(embeddings, dim=-1)
        return embeddings
    

class HashNeRF(nn.Module):
     """Decoding features extracted from the hash table into density and rgb."""
     def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg
        self.ht = HashTable(self.cfg, device=device)
        self.num_embeddings = cfg['num_embeddings']
        self.positional_encoding = PositionalEncoding(self.num_embeddings)
        dir_encoding_size = 3 + 3 * 2 * self.num_embeddings

        self.initial_layer = nn.Sequential(nn.Linear(cfg['features_dim'] * cfg['multigrid_levels'] + dir_encoding_size, cfg['hidden_dim']), nn.ReLU())
        self.hidden_layer = nn.Sequential(nn.Linear(cfg['hidden_dim'], cfg['hidden_dim']), nn.ReLU())
        self.final_layer = nn.Sequential(nn.Linear(cfg['hidden_dim'], 4), nn.Sigmoid())

     def forward(self, ray_o, ray_d):
        
        samples = Camera.get_samples(ray_o, ray_d, step=self.cfg['step'])     # (N, T, 3)

        # Get features from the hash table. This uses trilinear interpolation
        features = self.ht.query(samples)

        direction = self.positional_encoding(ray_d)    # (N, 3+3*2*num_embeddings)
        # tile to concatenate
        direction = direction.unsqueeze(1)
        direction = direction.expand(-1, features.shape[1], -1)

        input_dim = torch.cat((features, direction), dim=-1)  # (N, T, F + 3)

        x = self.initial_layer(input_dim)
        x = self.hidden_layer(x)
        x = self.final_layer(x)

        density = x[..., 0:1]
        rgb = x[..., 1:]

        return density, rgb
     

class HashTable(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg
        self.table_size = eval(cfg['table_size'])   # e.g. 2**16
        self.volume_size = float(self.cfg['volume_size'])  # e.g. 128 for 128x128x128
        self.device = device

        # Initialise constant and random features
        self.precompute_constants()
        # self.initialise_features()

        self.hash_table = nn.ModuleList([
            nn.Embedding(self.table_size, self.cfg['features_dim'], device=self.device) for l in range(self.cfg['multigrid_levels'])])
        self.initialise_features()


    def precompute_constants(self):
        self.primes = [73856093, 19349663, 83492791]

        self.voxel_size = self.cfg['grid_resolution']

        # Offsets to compute voxel's corners
        self.offsets = self.voxel_size * torch.tensor([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]
        ], dtype=torch.float32, device=self.device)[None, None, ...] # (1, 1, 8, 3)

        # Bounding box of the scene
        self.min_corner = torch.tensor([-self.volume_size / 2, -self.volume_size/2, -self.volume_size/2 + 0.5], device=self.device)
        self.max_corner = torch.tensor([+self.volume_size / 2, +self.volume_size/2, +self.volume_size/2 + 0.5], device=self.device)



    def initialise_features(self):
        for l in range(self.cfg['multigrid_levels']):
            """Initialise voxels with random features.
            The side of the voxel grid is halved with every subsequent layer."""
            # Create a grid of coordinates
            x, y, z = torch.meshgrid(torch.arange(-self.volume_size/2, +self.volume_size/2, step=self.voxel_size/(l+1)),
                                    torch.arange(-self.volume_size/2, +self.volume_size/2, step=self.voxel_size/(l+1)),
                                    torch.arange(-self.volume_size/2+1.5, +self.volume_size/2+1.5, step=self.voxel_size/(l+1)),
                                    indexing='xy')
            # Flatten the grid to get a list of coordinates
            coords_in_grid = torch.cat((x.flatten()[:,None], 
                                        y.flatten()[:,None],
                                        z.flatten()[:,None]), dim=1)
            # Initialise the features with random values
            features = torch.rand(
                coords_in_grid.shape[0],
                self.cfg['features_dim'],
                device=self.device)
            self.insert(coords_in_grid, features, l)
  

    # Function to retrieve data from the hash table
    def query(self, coords):
        """Query the hash table for the features corresponding to the given coordinates."""
        features = []
        for l in range(self.cfg['multigrid_levels']):
            # Retrieve the coordinates belonging to the voxel
            voxel_coords, base_coords = self.get_voxel_coords(coords, l)     
            
            # Get the hash for the given 3D point
            h = self._hash(voxel_coords)  # N Rays, T steps, 8 vertices 

            # Retrieve the features from the hash table
            features_level = self.hash_table[l](h)  # N Rays, T steps, 8 vertices, F features

            # Trilinear interpolation
            features_coords= self.trilinear_interpolation(coords, base_coords, l, features_level)  # N Rays, T steps, F features
            
            features.append(features_coords)

        features = torch.cat(features, dim=-1)  # N Rays, T steps, F features * levels

        return features      
    

    def get_voxel_coords(self, coords, l):
        """Get the coordinates of the voxel that contains the point,
        and the weighted distances to the 8 corners of the voxel.
        Args:
            coords (torch.tensor): input (N, T, 3)
        Returns:
            voxel_coords: voxel coordinates (N, T, 8, 3)
            base_coords: coordinates of the base vertices (N, T, 8)
        """
        # Ensure coords are within the bounding box
        coords = torch.clamp(coords, self.min_corner, self.max_corner)

        # Index 0: bottom, back, left corner
        base_indices = torch.floor(coords / (self.voxel_size/(l+1)) ).int()   # shape (N, T, 3), base index per sample

        # Get the 8 corners of the voxel: calculate the coordinates of the base corner of the voxel
        base_coords = base_indices * (self.voxel_size/(l+1))
        voxel_coords = base_coords[...,None,:] + self.offsets  # shape (N, T, 8, 3)

        return voxel_coords, base_coords
    
    
    def trilinear_interpolation(self, coords, base_coords, l, features):
        """Trilinear interpolation.
        Args:
            coords (torch.tensor): input (N, T, 3)
            base_coords (torch.tensor): coordinates of the base vertices (N, T, 3)
            features (torch.tensor): features of the voxel's vertices (N, T, 8, F)"""
        # Compute the weights for each corner
        #  Distance from the base corner for each axis. Distances are normalised
        distances = (coords - base_coords) / (self.voxel_size / (l + 1))   # (N, T, 3)

        # Compute weights for trilinear interpolation
        anti_distances = 1 - distances   # Distances from opposite vertex

        # Compute weights for trilinear interpolation
        # We need to compute the weight for each vertex, which is a combination of distances and anti-distances
        weights = torch.empty(features.shape[:-2] + (8,), device=features.device)

        d_0, d_1, d_2 = distances[..., 0], distances[..., 1], distances[..., 2]
        a_0, a_1, a_2 = anti_distances[..., 0], anti_distances[..., 1], anti_distances[..., 2]
        
        # The weight of a vertex is the product of the distances/anti-distances along each axis.
        # Intuitively, the closer coord is to a corner, the more weight that corner's value has. 
        # E.g. if coord is close to base corner, the weight of base corner (0,0,0) is very high (so we take all the 
        # anti-distances, which are high if coord is close to (0,0,0))
        weights = torch.stack([
            a_0 * a_1 * a_2,   # vertex 0 -> (0,0,0)
            d_0 * a_1 * a_2,   # vertex 1 -> (1,0,0)
            a_0 * d_1 * a_2,   # vertex 2 -> (0,1,0)
            d_0 * d_1 * a_2,   # vertex 3 -> (1,1,0)
            a_0 * a_1 * d_2,   # vertex 4 -> (0,0,1)
            d_0 * a_1 * d_2,   # vertex 5 -> (1,0,1)
            a_0 * d_1 * d_2,   # vertex 6 -> (0,1,1)
            d_0 * d_1 * d_2],  # vertex 7 -> (1,1,1)
        dim=-1)

        # Multiply the features by the weights and sum them to get the interpolated features
        interpolated_features = (features * weights.unsqueeze(-1)).sum(dim=-2)

        return interpolated_features


    def _hash(self, coords):
        """Simple spatial hash function for 3D coordinates.
        Coordinates are multiplied by a number and combined using XOR."""
        # Prime numbers for hashing
        h = coords[..., 0] * self.primes[0] + coords[..., 1] * self.primes[1] + coords[..., 2] * self.primes[2]
        
        # Check the hash is within the bounds of the hash table
        return (h % self.table_size).int()
    
    
    def insert(self, coords, features, l):
        '''Function to insert data into the hash table'''
        # Get the hash for the given 3D point
        h = self._hash(coords)

        # Insert the data into the hash table
        #self.hash_table[l].weight.data[h] = features
        with torch.no_grad():
            self.hash_table[l].weight.data[h] = features