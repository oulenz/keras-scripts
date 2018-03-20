import argparse

from keras_scripts.keras_scripts import prepare_image_dataset_from_root_folder

parser = argparse.ArgumentParser(description='Create dataset file from images in labeled subdirectories')
parser.add_argument('path', help='Directory with images sorted in subdirectories')
parser.add_argument('name', help='Name to be used for dataset')
parser.add_argument('reshape', type=int, nargs=3, help='Shape to reshape images to')
parser.add_argument('split', type=float, nargs='?', default=0.05, help='Portion to split off for testset')
args = parser.parse_args()

prepare_image_dataset_from_root_folder(args.path, args.name, tuple(args.reshape), args.split)