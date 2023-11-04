import torch
import os
from glob import glob
import data
from PIL import Image

class Camera:
    def __init__( 
            self,
            fov_radians=None, 
            image_width=None,
            image_height=None,
            extr=None
        ):
        self.fov_radians = fov_radians
        self.image_width = image_width
        self.image_height = image_height
        self.extr = extr
        self.intr = Camera.compute_intr(self.fov_radians, self.image_width, self.image_height) if self.fov_radians and self.image_width and self.image_height else None
    
    @staticmethod
    def get_directions_cam(u, v, image_width, image_height, fov_radians, device='cuda:0'):
        """Get directions of the rays through all the pixels of an image. 
        Directions are in camera coordinate.
        Params:
            u: pixel index along the u-axis, int
            v: pixel index along the v-axis, int
            image_width: width in pixels, int
            image_height: height in pixels, int
            fov_radians: horizontal field of view, float
        Returns:
            dirs: directions in camera coordinate, [u.shape, 3]
        """
        fx = image_width / (2 * torch.tan(torch.tensor([fov_radians/2], device=device)))
        dirs = torch.stack([(u - image_width/2 + 0.5)/fx, (image_height/2 - v + 0.5)/fx, -torch.ones_like(u)], -1)
        return dirs

    @staticmethod
    def generate_rays(extr, ray_dir_cam):
        """
        Generate ray origin and ray directions in world coordinate for an image.
        Parameters:
            extr: extrinsics matrix (1, 4, 4)
            ray_dir_cam: ray directions in camera frame (num_rays, 4, 1)
        Returns:
            ray_o: origin in world frame [1, 3]
            ray_d: direction in world frame [num_rays, 3]
        """
        ray_o = extr[..., 3]  # (1, 1, 4)
        ray_o = ray_o[..., None]  # (1, 4, 1)
        
        ray_d = (extr @ ray_dir_cam) - ray_o  

        # Homogeneous to heterogeneous
        ray_o = ray_o.squeeze(2)[:, :3]  # (1, 3)
        ray_d = ray_d.squeeze(2)[:, :3]  # (1, 3)
        
        return ray_o, ray_d
    
    
    @staticmethod
    def get_samples(rays_o, rays_d, near=0.5, far=5, step=0.01, device='cuda:0'):
        """Sample along the rays. Return samples in world coordinates
        Params:
            rays_o: origin in world frame [1, 3]
            rays_d: direction in world frame [n, 3, 1]
            near, far = restrict sampling domain
            n_samples = num samples along the ray within the restricted domain
        Returns:
            samples: samples in world coordinate, [n, len(t), 3]. n is the number of rays, 3 coords, len(t) is num samples
        """
        time_steps = torch.arange(near, far, step, dtype=torch.float, device=device)[None,:,None]  # (1, len(t), 1)
        samples = rays_o[:,None,:] + time_steps @ rays_d[:,None,:]  # (n, 1, 3) - (n, len(t), 1) @ (n, 1, 3) -> (n, len(t), 3)
        return samples
    
    @staticmethod
    def compute_Tn(density, step):
        """Compute the transmittance of each sample along the ray."""
        t_n = torch.exp( - torch.cumsum(density, dim=1) * step) # (N, T, 1)
        return t_n
    
    @staticmethod
    def volume_rendering(density, rgb, step):
        """Use volume rendering to compute the final pixel colour.
        Args:
            density: density, (N, T, 1)
            rgb: rgb, (N, T, 3)
            step: step size, float
        Returns:
            C: final pixel colour, (N, 3)"""
        
        # Calculate transmittance
        T = Camera.compute_Tn(density, step)  # (N, T, 1)

        # Calculate weights for each sample
        alpha = 1 - torch.exp( - density * step)  # (N, T, 1)
        weights = T * alpha # (N, T, 1)

        # Compute final pixel colour
        C = torch.sum(weights * rgb, dim=1)   # (N, 3)

        return C

    @staticmethod
    def compute_intr(fov_radians: float, image_width: int, image_height: int, device='cuda:0'):
        """Compute the intrinsic matrix from the camera parameters."""
        fx = image_width / (2 * torch.tan(torch.tensor([fov_radians/2], device=device)))
        fy = fx
        cx = image_width / 2
        cy = image_height / 2
        intr = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=device)
        return intr
    
    @staticmethod
    def get_image_size(dataset_name: str, partition: str):
        """Get the image size in pixel from dataset name and partition."""
        folder = os.path.join(os.path.dirname(data.__file__), dataset_name, partition)
        image_paths = glob(os.path.join(folder, '*.png'))
        image = Image.open(image_paths[0])
        image_width, image_height = image.size

        return image_width, image_height