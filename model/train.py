import torch
import yaml
import configs
import os
import argparse
from model import NeRF, HashNeRF
import data.dataset as dataset
import torch.optim as optim
from torch.utils.data import DataLoader
from scene.camera import Camera
from scene.diff_renderer import Renderer
import data
import json
import time
from utils import utils
import results

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Trainer:
    def __init__(self, args, cfg):
        self.cfg = cfg
        self.args = args
        self.render = Renderer(self.cfg)

    def __call__(self):
        # Set up folders and paths for saving and logging results
        self.folder_res = os.path.join(
            os.path.dirname(results.__file__),
            f"{self.args.dataset_name}_{self.args.exp_name}",
        )
        if not os.path.exists(self.folder_res):
            os.makedirs(self.folder_res)
        self.log_path = f"{self.folder_res}/log.txt"

        # Save config
        with open(f"{self.folder_res}/config.yaml", "w") as f:
            yaml.dump(self.cfg, f)

        utils.log(self.log_path, f"Device: {device}")

        # Instantiate model
        if self.cfg["model"] == "NeRF":
            self.model = NeRF(self.cfg).to(device)

            self.optim_list = [
                optim.Adam(self.model.parameters(), lr=self.cfg["lr"], weight_decay=0)
            ]

        elif self.cfg["model"] == "HashNeRF":
            self.model = HashNeRF(self.cfg, device).to(device)
            # Define optimisers
            self.optim_list = [
                optim.Adam(self.model.parameters(), lr=self.cfg["lr"], weight_decay=0),
                optim.Adam(self.model.ht.hash_table.parameters(), lr=self.cfg["lr_hash"], weight_decay=0)
            ]

        else:
            raise NotImplementedError

        # Get data
        train_loader, val_loader = self.get_loaders()

        #######################################3
        # Render first image in the validation set
        # Get all camera poses in the folder
        cameras_path = os.path.join(
            os.path.dirname(data.__file__),
            self.args.dataset_name,
            f"transforms_val.json",
        )
        with open(cameras_path, "r") as file:
            cameras_json = json.load(file)
        camera_json = cameras_json["frames"][61]
        image_width, image_height = Camera.get_image_size(self.args.dataset_name, "val")
        self.camera_first_val = Camera(
            fov_radians=cameras_json["camera_angle_x"],
            image_width=image_width,
            image_height=image_height,
            extr=torch.tensor(
                camera_json["transform_matrix"], dtype=torch.float, device=device
            )[None, ...],  # (1, 4, 4)
        )
        #################################################################

        best_val_loss = 10000000
        # Iterate through data
        for epoch in range(self.cfg["epochs"]):
            utils.log(
                self.log_path,
                f"============================ Epoch {epoch} ============================",
            )

            self.epoch = epoch

            start = time.time()

            avg_train_loss = self.train(train_loader)

            with torch.no_grad():
                avg_val_loss = self.val(val_loader)

                utils.log(self.log_path, f'Time elapsed: {time.time() - start}')

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    self.save_model()

    def get_loaders(self):
        train_dataset = dataset.NeRFDataset(self.args.dataset_name, "train")
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg["batch_size"],
            shuffle=True,
            drop_last=False,
        )

        val_dataset = dataset.NeRFDataset(self.args.dataset_name, "val")
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg["batch_size"],
            shuffle=True,
            drop_last=False,
        )

        return train_loader, val_loader

    def zero_grad_optimisers(self):
        for optim in self.optim_list:
            optim.zero_grad()

    def step_optimisers(self):
        for optim in self.optim_list:
            optim.step()

    def train(self, train_loader):
        """
        Train the model for one epoch.
        batch[0]: rays_o, (N, 4, 1)
        batch[1]: rays_d, (N, 4, 1)
        batch[2]: rgb, (N, 3)
        """
        total_loss = 0
        for batch in train_loader:
            self.zero_grad_optimisers()
            # self.optim.zero_grad()  # zero the gradient buffers

            rays_o = batch[0]
            rays_d = batch[1]

            density, rgb = self.model(rays_o, rays_d)  # (N, T, 1), (N, T, 3)

            pixel_rgb = Camera.volume_rendering(
                density, rgb, self.cfg["step"]
            )  # (N, 3)

            loss = torch.mean((pixel_rgb - batch[2]) ** 2)

            loss.backward()

            self.step_optimisers()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        utils.log(self.log_path, f"Average train loss: {avg_loss}")
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

            pixel_rgb = Camera.volume_rendering(
                density, rgb, self.cfg["step"]
            )  # (N, 3)

            loss = torch.mean((pixel_rgb - batch[2]) ** 2)

            total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)

        utils.log(self.log_path, f"Average val loss: {avg_loss}")

        with torch.no_grad():
            image = self.render.render_scene(self.model, self.camera_first_val)
            self.render.save_image(image, f"{self.folder_res}/epoch_{self.epoch}.jpg")

        return avg_loss

    def save_model(self):
        torch.save(self.model.state_dict(), f"{self.folder_res}/model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name", type=str, default="", help="Name of the dataset, e.g. lego"
    )
    parser.add_argument(
        "--exp_name", type=str, default="", help="Name of the experiment"
    )
    args = parser.parse_args()

    train_cfg_path = os.path.join(
        os.path.dirname(configs.__file__), f"{args.dataset_name}.yaml"
    )
    with open(train_cfg_path, "rb") as f:
        train_cfg = yaml.load(f, Loader=yaml.FullLoader)


    trainer = Trainer(args, train_cfg)
    trainer()
