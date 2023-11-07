from scene.camera import Camera
import torch 
from utils import utils
from matplotlib import pyplot as plt
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Renderer:
    """Renderer class"""
    def __init__(self, cfg):
        self.cfg = cfg

    def render_scene(self, model, camera):
        """
        Args:
            model (torch.nn.Module): NeRF model to use for rendering
            camera (torch.nn.Module): Camera model to use for rendering

        Returns:
            image (np.array): rendered image
        """

        u, v = torch.meshgrid(
            torch.arange(camera.image_width, dtype=torch.int, device=device),
            torch.arange(camera.image_height, dtype=torch.int, device=device),
            indexing='xy'
        )
        u, v = u.flatten(), v.flatten()

        # Get the rays in camera coordinate
        coord_cam = Camera.get_directions_cam(u, v, camera.image_width, camera.image_height, camera.fov_radians, device=device)
        coord_cam_homogeneous = utils.to_homogeneous(coord_cam)[...,None]  # (n, 4, 1)

        # Generate rays in world coordinate
        rays_o, rays_d = Camera.generate_rays(camera.extr, coord_cam_homogeneous) 

        # batchify rays to avoid CUDA memory overflow
        rays_d_batches = utils.batchify(rays_d)

        density_list, rgb_list = [], []
        for rays_d_batch in rays_d_batches:
            # Get density and rgb
            density, rgb = model(rays_o, rays_d_batch)
            
            density_list.append(density)
            rgb_list.append(rgb)

        density = torch.cat(density_list, dim=0)
        rgb = torch.cat(rgb_list, dim=0)
        
        # Get pixel colours
        pixel_colours = Camera.volume_rendering(density, rgb, self.cfg['step'])

        # Reshape to image
        image = pixel_colours.reshape(camera.image_height, camera.image_width, 3)

        image = image.cpu().numpy()

        return image

    def save_image(self, image, path):
        image_obj = Image.fromarray((image*255).astype('uint8'))

        # Save the image
        image_obj.save(path)
        # plt.imsave(path, image, vmin=0, vmax=1)

    def visualise_image(self, image):
        plt.imshow(image)
        plt.show()
        

    