import torch
import yaml
import configs
import os
import argparse
from model import NeRF
import data.dataset as dataset
import torch.optim as optim
from torch.utils.data import DataLoader
from scene.camera import Camera
from scene.diff_renderer import Renderer
import data
import json

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

class Trainer:
    def __init__(self, args, cfg):
        self.cfg = cfg
        self.args = args
        self.render = Renderer(self.cfg)

    def __call__(self):
        # Instantiate model
        self.model = NeRF(self.cfg).to(device)

        # Get data
        train_loader, val_loader = self.get_loaders()

        # Define optimisers
        self.optim = optim.Adam(self.model.parameters(), lr=self.cfg['lr'], weight_decay=0)

        #######################################3
        # Render first image in the validation set
        # Get all camera poses in the folder
        cameras_path = os.path.join(os.path.dirname(data.__file__), self.args.dataset_name, f'transforms_val.json')
        with open(cameras_path, 'r') as file:
            cameras_json = json.load(file)
        camera_json = cameras_json['frames'][7]       
        image_width, image_height = Camera.get_image_size(self.args.dataset_name, 'val')     
        self.camera_first_val = Camera(
            fov_radians=cameras_json['camera_angle_x'], 
            image_width=image_width,
            image_height=image_height, 
            extr=torch.tensor(
                camera_json['transform_matrix'],
                dtype=torch.float,
                device=device)[None, ...]  # (1, 4, 4)
        )
        #################################################################
        
        best_val_loss = 10000000
        # Iterate through data
        for epoch in range(self.cfg['epochs']):
            print(f'============================ Epoch {epoch} ============================')
            self.epoch = epoch

            avg_train_loss = self.train(train_loader)

            with torch.no_grad():
                avg_val_loss = self.val(val_loader)
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    self.save_model()
      

    def get_loaders(self):
        train_dataset = dataset.NeRFDataset(self.args.dataset_name, 'train')
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg['batch_size'],
            shuffle=True,
            drop_last=False
        )   

        val_dataset = dataset.NeRFDataset(self.args.dataset_name, 'val')
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg['batch_size'],
            shuffle=True,
            drop_last=False
        )   

        return train_loader, val_loader

    def train(self, train_loader):
        """
        Train the model for one epoch.
        batch[0]: rays_o, (N, 4, 1)
        batch[1]: rays_d, (N, 4, 1)
        batch[2]: rgb, (N, 3)
        """
        total_loss = 0
        for batch in train_loader:

            self.optim.zero_grad()    # zero the gradient buffers

            rays_o = batch[0]
            rays_d = batch[1]

            density, rgb = self.model(rays_o, rays_d)  # (N, T, 1), (N, T, 3)

            pixel_rgb = Camera.volume_rendering(density, rgb, self.cfg['step'])   # (N, 3)
            
            loss = torch.mean((pixel_rgb - batch[2])**2)
            loss.backward()
            self.optim.step()

            total_loss += loss.item()           
                
        avg_loss = total_loss / len(train_loader)

        print(f'Average train loss: {avg_loss}')
        return avg_loss
        

    def val(self, val_loader):
        """
        Train the model for one epoch.
        batch[0]: rays_o, (N, 4, 1)
        batch[1]: rays_d, (N, 4, 1)
        batch[2]: rgb, (N, 3)
        """
        total_loss = 0
        for batch in val_loader:

            rays_o = batch[0]
            rays_d = batch[1]

            density, rgb = self.model(rays_o, rays_d)

            pixel_rgb = Camera.volume_rendering(density, rgb, self.cfg['step'])   # (N, 3)
            
            loss = torch.mean((pixel_rgb - batch[2])**2)
            
            total_loss += loss.item()           
                
        avg_loss = total_loss / len(val_loader)

        print(f'Average val loss: {avg_loss}')

        with torch.no_grad():
            image = self.render.render_scene(self.model, self.camera_first_val)
            folder = f'./results/{self.args.dataset_name}'
            if not os.path.exists(folder):
                os.makedirs(folder)
            self.render.save_image(image, f'{folder}/epoch_{self.epoch}.jpg')

        return avg_loss

    def save_model(self):
        folder = f'./results/{self.args.dataset_name}'
        if not os.path.exists(folder):
            os.makedirs(folder)
        torch.save(self.model.state_dict(), f'{folder}/model.pt')



if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='', help='name of the dataset, e.g. lego')
    args = parser.parse_args()

    args.dataset_name = 'lego_small'

    train_cfg_path = os.path.join(os.path.dirname(configs.__file__), f'{args.dataset_name}.yaml')
    with open(train_cfg_path, 'rb') as f:
        train_cfg = yaml.load(f, Loader=yaml.FullLoader)

    trainer = Trainer(args, train_cfg)
    trainer()