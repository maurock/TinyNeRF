import torch
import torch.nn as nn
from operator import itemgetter

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

        self.hash_table = nn.ModuleList([nn.Embedding(self.table_size, self.cfg['features_dim'], device=self.device)])
        self.initialise_features()

    def precompute_constants(self):
        self.primes = [73856093, 19349663, 83492791]
        self.voxel_size = self.cfg['grid_resolution']
        # Offsets to compute voxel's corners
        self.offsets = self.voxel_size * torch.tensor([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ], dtype=torch.float32, device=self.device)[None, None, ...] # (1, 1, 8, 3)
        self.diag = torch.sqrt(torch.tensor([3.0], dtype=torch.float32, device=self.device)) * self.voxel_size

   
    def initialise_features(self):
        """Initialise voxels with random features."""
        # Create a grid of coordinates
        x, y, z = torch.meshgrid(torch.arange(-self.volume_size/2, +self.volume_size/2, step=self.voxel_size),
                                 torch.arange(-self.volume_size/2, +self.volume_size/2, step=self.voxel_size),
                                 torch.arange(-self.volume_size/2, +self.volume_size/2, step=self.voxel_size),
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
        self.insert(coords_in_grid, features)
  

    # Function to retrieve data from the hash table
    def query(self, coords):
        # Retrieve the coordinates belonging to the voxel
        voxel_coords, weights = self.get_voxel_coords(coords)     
        
        # Get the hash for the given 3D point
        h = self._hash_emb(voxel_coords)  # N Rays, T steps, 8 vertices 

        # Retrieve the data from the hash table

        features = self.hash_table_emb[0](h)  # N Rays, T steps, 8 vertices, F features

        # Trilinear interpolation
        features = torch.sum(features * weights[...,None], dim=2)  # N Rays, T steps, F features

        return features      
    

    def get_voxel_coords(self, coords):
        """Get the coordinates of the voxel that contains the point,
        and the weighted distances to the 8 corners of the voxel.
        Args:
            coords (torch.tensor): input (N, T, 3)
        Returns:
            tuple: voxel indices (N, T, 8, 3)
            weights: normalised weights (N, T, 8)
        """
        # Index 0: bottom, back, left corner
        base_indices = torch.floor(coords / self.voxel_size).int()   # shape (N, T, 3), base index per sample

        # Get the 8 corners of the voxel: calculate the coordinates of the base corner of the voxel
        base_coords = base_indices * self.voxel_size
        voxel_coords = base_coords[...,None,:] + self.offsets  # shape (N, T, 8, 3)

        # Compute the weights for each corner
        weights = torch.norm(coords[...,None,:] - voxel_coords, dim=-1) / self.diag   # divide by voxel diagonal to normalise

        return voxel_coords, weights


    def _hash(self, coords):
        """Simple spatial hash function for 3D coordinates.
        Coordinates are multiplied by a number and combined using XOR."""
        # Prime numbers for hashing
        h = coords[..., 0] * self.primes[0] + coords[..., 1] * self.primes[1] + coords[..., 2] * self.primes[2]
        
        # Check the hash is within the bounds of the hash table
        return (h % self.table_size).int()