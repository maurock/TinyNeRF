#!/bin/bash

wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz -O tiny_nerf_data.npz
unzip tiny_nerf_data.npz
mv tiny_nerf_data/nerf_synthetic/lego data/lego
mv tiny_nerf_data/nerf_llff_data/fern data/fern
rm tiny_nerf_data.npz
rm -rf tiny_nerf_data