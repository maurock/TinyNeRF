import torch.nn as nn
import torch
import numpy as np
from scene.camera import Camera
# from model.representation import HashTable

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
    

# class HashNeRF(nn.Module):
#     def __init__(self, cfg, ht):
#         super().__init__()
#         self.cfg = cfg
#         self.ht = HashTable(self.cfg).to(device)

#     def forward(self, ray_o, ray_d):

#         samples = Camera.get_samples(ray_o, ray_d, step=self.cfg['step'])     # (N, T, 3)

#         # Hash the coordinates to get indices
#         indices = self.ht.query(samples)
#         # Look up the features
#         features = feature_table[indices]
#         # 'Run' the neural network (just a dot product here)
#         predicted = neural_network(features)
#         # Compute the error (just mean squared error here)
#         error = (ground_truth - predicted) ** 2
#         # Update the feature table (gradient descent step, simplified)
#         feature_table[indices] -= 0.1 * error  # This would be more complex with a real NN



#         cfg = dict()
#         cfg['table_size'] = 2**4
#         ht = HashTable(cfg)

#         torch.manual_seed(42)
#         coords = torch.rand(2, 3)
#         features = torch.rand(2, 4)

#         ht.insert(coords, features)
#         values = ht.query(coords)

#         print(values)