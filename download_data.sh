#!/bin/bash

wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip -O nerf_example_data.zip
unzip nerf_example_data.zip
mv nerf_synthetic/lego data/lego
mv nerf_llff_data/fern data/fern
rm nerf_synthetic
rm nerf_llff_data