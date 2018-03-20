import argparse
import keras
import os

from keras_scripts.keras_scripts import save_keras_model_as_saved_model

parser = argparse.ArgumentParser(description='Convert keras model file to tensorflow saved model directory')
parser.add_argument('load_path', help='Path of the keras model')
parser.add_argument('save_path', nargs='?', help='Save dir path, defaults to load path minus extension')
args = parser.parse_args()

model = keras.models.load_model(args.load_path)
save_path = args.save_path or os.path.splitext(args.load_path)[0]
save_keras_model_as_saved_model(model, save_path)
