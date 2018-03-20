import argparse
import keras
import os

from keras_scripts.models import keras_cnn
from keras_scripts.optimisers import adam
from keras_scripts.keras_scripts import get_highest_model_version_nr, load_saved_image_dataset, test_model, train_model

parser = argparse.ArgumentParser(description='Train model with dataset file')
parser.add_argument('dataset_path', help='Path of the dataset without \'_train.json\' and \'_test.json\'')
parser.add_argument('model_dir', help='Path of the directory containing the model')
parser.add_argument('name', help='Base part of the model filename')
parser.add_argument('date', help='Date part of the model filename')
parser.add_argument('epochs', type=int, default=1, nargs='?', help='Number of epochs to train')
parser.add_argument('--load', type=int, help='Model version to load, -1 to start over, defaults to highest')
parser.add_argument('--increment', action='store_true', help='Store result as new version of model')
args = parser.parse_args()

train, test = load_saved_image_dataset(dataset_path=args.dataset_path)
print('dataset loaded')

highest_version_nr = get_highest_model_version_nr(args.model_dir, args.name, args.date)
load_version_nr = args.load if args.load is not None else highest_version_nr

if load_version_nr < 0:
    img_reshape = train.loader.preprocessor.reshape_to
    optimiser = adam()
    model = keras_cnn(img_reshape, train.num_classes, optimiser)
    print('new model compiled')
else:
    old_model_filename = '_'.join([args.name, args.date, str(load_version_nr) + '.h5'])
    model_load_path = os.path.join(args.model_dir, old_model_filename)
    model = keras.models.load_model(model_load_path)
    print('loaded existing model ' + old_model_filename)

save_version_nr = highest_version_nr + 1 if args.increment or load_version_nr < 0 else load_version_nr
new_model_filename = '_'.join([args.name, args.date, str(save_version_nr) + '.h5'])
model_save_path = os.path.join(args.model_dir, new_model_filename)

train_model(model, train, args.epochs, model_save_path)
print('trained and saved model as ' + new_model_filename)
test_model(model, test)