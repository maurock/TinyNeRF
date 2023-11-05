# TinyNeRF

Implementation of NeRF and recent advances in neural radiance fields. The goal of this repository is to provide a simple and intuitive implementation of NeRF as a playground to test ideas and techniques. Step-to-step instructions on data extraction, training, and testing are also provided.

# Content
- [Results](#results)
- [Installation](#installation)
- [Data](#data-making)
- [Training](#training)

# Results
Currently, these features are implemented:
- [x] Vanilla NeRF
- [x] Hash Maps a la InstantNGP - WIP
- [ ] Triplanes

# Installation
These installation instructions are tested on Linux (Ubuntu 22.04 LTS, CentOS 7). 
```
conda create -n tinynerf
conda env update -n tinynerf --file environment.yaml
pip install -e .
```

# Data
Download the data by running:
```
sh download_data.sh
```
This script download the `lego` and `fern` scenes and places them in `data/`.

You can generate a dataset with smaller images by run:
```
python data/compress_data --dataset_name <you_dataset> --partition <your_partition> --width <your_width> --height <your_height>
```
For example, if you want to reduce the images of the `lego` dataset in the training, vlaidation, and test set to 200x200, simply run:
```
python data/compress_data.py --dataset_name lego --partition train --width 200 --height 200
python data/compress_data.py --dataset_name lego --partition val --width 200 --height 200
python data/compress_data.py --dataset_name lego --partition test --width 200 --height 200
```
The generated folder will be under `data/<your_dataset>_compressed/<partition?`. *This will be improved soon*. 

# Training
To train the NeRF model: you first need to set `configs/<dataset_name>.yaml`. This repository provides an example for the dataset `lego`.
Then, simply run:
```
python model/train.py --dataset_name <you_dataset_name> --exp_name <your_experiment_name>
```
Once the model is trained, you can find the stored weights in `results/<dataset_name>_<exp_name>/model.pt`.