"""Compress data for smaller images."""
from PIL import Image
import argparse
import data
import os
from glob import glob
import shutil

def main(args):
    # Set folder contianing the data
    folder = os.path.join(os.path.dirname(data.__file__), args.dataset_name, args.partition)

    # Create folder for compressed images
    folder_compressed = os.path.join(os.path.dirname(data.__file__), args.dataset_name + '_compressed', args.partition)
    if not os.path.exists(folder_compressed):
        os.makedirs(folder_compressed)

    # Get all images in the folder
    image_paths = glob(os.path.join(folder, '*.png'))
    
    # Loop over all images
    for image_path in image_paths:

        image = Image.open(image_path)

        image_name = image_path.split('/')[-1]

        # Downsize the image with an ANTIALIAS filter (gives the highest quality)
        image_compressed = image.resize((args.width,args.height), Image.LANCZOS)

        # Save
        image_compressed_path = os.path.join(folder_compressed, image_name)
        image_compressed.save(image_compressed_path, quality=95)

        # Copy corresponding .json file
        json_path = os.path.join(
            os.path.dirname(data.__file__),
            args.dataset_name ,
            f'transforms_{args.partition}.json')
        shutil.copy(json_path,
                    os.path.join(
                        os.path.dirname(data.__file__),
                        args.dataset_name + '_compressed',
                        f'transforms_{args.partition}.json'))       
        

if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--dataset_name', type=str, default='', help='name of the dataset, e.g. lego')
    args.add_argument('--partition', type=str, default='', help='Choose between train, val, test')
    args.add_argument('--width', type=int, default='1', help='Desired width of the image in pixels')
    args.add_argument('--height', type=int, default='1', help='Desired height of the image in pixels')
    args = args.parse_args()

    main(args)

