import cv2
import json
import keras
import os
import tensorflow as tf
from typing import Tuple

from dataset.image_dataset import ImageDataset, ImagePreprocessor
from experiment_logger.loggable import ObjectDict
from keras_scripts.os_wrapper import list_images, list_subfolders


def get_x_shape_from_root_folder(root_folder_path: str):
    for subfolder in list_subfolders(root_folder_path):
        for img_path in list_images(os.path.join(root_folder_path, subfolder)):
            return list(cv2.imread(os.path.join(root_folder_path, subfolder, img_path)).shape)


def prepare_image_dataset_from_root_folder(root_folder_path: str, name: str, img_shape: Tuple[int, int, int], test_portion: float):
    datast = ImageDataset.from_root_folder(root_folder_path, name=name)
    datast = datast.upsampled()
    datast = datast.shuffled()
    datast = datast.index_encoded()
    datast = datast.onehot_encoded()
    if img_shape is not None:
        preprocessor = ImagePreprocessor()
        preprocessor.reshape_to = img_shape
        datast.preprocessor = preprocessor
    datast_savepath = os.path.join(root_folder_path, name)
    if test_portion:
        test, datast = datast.split(test_portion, suffixes=('test', 'train'))
        test_savepath = datast_savepath + '_test.json'
        datast_savepath = datast_savepath + '_train.json'
        with open(test_savepath, 'w') as f:
            test_json_dct = test.to_object_dict()
            json.dump(test_json_dct, f, indent=4)
    with open(datast_savepath, 'w') as f:
        datast_json_dct = datast.to_object_dict()
        json.dump(datast_json_dct, f, indent=4)


def load_saved_image_dataset(dataset_path: str):
    train_savepath = dataset_path + '_train.json'
    test_savepath = dataset_path + '_test.json'

    with open(train_savepath, 'r') as f:
        train = ObjectDict(json.load(f)).to_object()
    with open(test_savepath, 'r') as f:
        test = ObjectDict(json.load(f)).to_object()

    return train, test


def get_highest_model_version_nr(root_path: str, dataset_name: str, model_date):
    model_name_date = dataset_name + '_' + str(model_date) + '_'
    versions = set()
    for filename in os.listdir(root_path):
        stem, ext = os.path.splitext(filename)
        if not stem.startswith(model_name_date) or ext != '.h5':
            continue
        maybe_version = stem[len(model_name_date):]
        try:
            version = int(maybe_version)
        except ValueError:
            continue
        versions.add(version)
    return max(versions, default=-1)


def train_model(model, train, epochs: int, save_path: str) -> None:
    model.fit(train.X, train.y, epochs=epochs, batch_size=128, validation_split=0.1)
    model.save(save_path)
    return


def test_model(model, test) -> None:
    score = model.evaluate(test.X, test.y, batch_size=128)
    print(score)
    return


def save_keras_model_as_saved_model(model, pb_model_directory):
    builder = tf.saved_model.builder.SavedModelBuilder(pb_model_directory)
    signature = tf.saved_model.signature_def_utils.predict_signature_def(inputs={'inputs': model.input}, outputs={'outputs': model.output})
    builder.add_meta_graph_and_variables(
        sess=keras.backend.get_session(),
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}
    )
    builder.save()