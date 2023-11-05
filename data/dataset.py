import data
import torch
import os
import glob
import json
from scene.camera import Camera
from PIL import Image
from utils import utils

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class NeRFDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_name, partition):

        self.dataset_name = dataset_name
        self.partition = partition

        # Set folder contaning the data
        folder = os.path.join(os.path.dirname(data.__file__), self.dataset_name, self.partition)

        # Get all camera poses in the folder
        cameras_path = os.path.join(os.path.dirname(data.__file__), self.dataset_name, f'transforms_{self.partition}.json')
        with open(cameras_path, 'r') as file:
            cameras = json.load(file)

        # Get image size
        image_width, image_height = Camera.get_image_size(dataset_name, partition)

        rays_o_list = []
        rays_d_list = []
        rgb_list = []

        # Pixel coordinates
        u, v = torch.meshgrid(
            torch.arange(image_width, dtype=torch.int, device=device),
            torch.arange(image_height, dtype=torch.int, device=device),
            indexing='xy'
        )
        u, v = u.flatten(), v.flatten()

        # Get the rays in camera coordinate
        coord_cam = Camera.get_directions_cam(u, v, image_width, image_height, cameras['camera_angle_x']) # (width x height, 3)
        coord_cam_homogeneous = utils.to_homogeneous(coord_cam) # (width x height, 4)
        coord_cam_homogeneous = coord_cam_homogeneous[...,None]  # (n, 4, 1)
        
        # Loop over all cameras
        for idx, camera in enumerate(cameras['frames']):

            extr = torch.tensor(camera['transform_matrix'], dtype=torch.float, device=device)[None, ...]  # (1, 4, 4)
            
            rays_o, rays_d = Camera.generate_rays(extr, coord_cam_homogeneous)

            # Get colour
            image_path = os.path.join(folder, f'{camera["file_path"].split(os.sep)[-1]}.png')
            image = Image.open(image_path)
            image = torch.tensor(image.getdata(), device=device).float() / 255.0 # (width x height, 4)
            
            rgb = image[v * image_width + u, :3]  # (width x height, 4)

            # Accumulate (adding to list for efficiency)
            rays_o_list.append(torch.tile(rays_o, (rays_d.shape[0], 1)))
            rays_d_list.append(rays_d)
            rgb_list.append(rgb)

        rays_o_acc = torch.cat(rays_o_list, dim=0)   # (N, 3)
        rays_d_acc = torch.cat(rays_d_list, dim=0)   # (N, 3)
        rgb_acc = torch.cat(rgb_list, dim=0)         # (N, 3)


        # Store samples        
        self.data = dict()
        self.data['rays_o'] = rays_o_acc
        self.data['rays_d'] = rays_d_acc
        self.data['rgb'] = rgb_acc

        return

    def __len__(self):
        return self.data['rgb'].shape[0]

    def __getitem__(self, idx):
        rays_o = self.data['rays_o'][idx, :]      # (N, 3)
        rays_d = self.data['rays_d'][idx, :]      # (N, 3)
        rgb = self.data['rgb'][idx, :]               # (N, 3)

        return rays_o, rays_d, rgb