# Author: Fernando García-García <fegarcia@bcamath.org>

import os

import pandas as pd

import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.image import extract_patches

from experim.ordclass import CONFIG

from experim.ordclass.utils_load import load_2d

TAG_TASK_TYPE = CONFIG.TAG_SEGMENT
TAG_TARGET_TYPE = CONFIG.TAG_ORDIN

TAG_DATASET = 'BACH'

COL_FILE = 'filename_in'
COL_TASK = 'filename_out'

N_CLASSES = 5


def load(*, idx_split):
    # load info
    path_base = os.path.join(CONFIG.PATH_DATA, TAG_TASK_TYPE, TAG_TARGET_TYPE, TAG_DATASET)
    file_data = os.path.join(path_base, CONFIG.FILE_SPLIT)
    if not os.path.isfile(file_data):
        raise RuntimeError

    df_data = pd.read_csv(file_data)

    path_data = os.path.join(path_base, CONFIG.TAG_DATA)
    df_data[COL_FILE] = df_data[COL_FILE].apply(lambda x: os.path.join(path_data, x))
    df_data[COL_TASK] = df_data[COL_TASK].apply(lambda x: os.path.join(path_data, x))

    # split
    tag_split = COL_TASK + CONFIG.TAG_SEP + CONFIG.TAG_SPLIT + str(idx_split)
    df_train = df_data[df_data[tag_split] == CONFIG.TAG_TRAIN]
    df_valid = df_data[df_data[tag_split] == CONFIG.TAG_VALID]
    df_test = df_data[df_data[tag_split] == CONFIG.TAG_TEST]

    data_train = Dataset.from_tensor_slices(dict(df_train))
    data_valid = Dataset.from_tensor_slices(dict(df_valid))
    data_test = Dataset.from_tensor_slices(dict(df_test))

    # load
    img_shape_in = (None, None, CONFIG.IMG_CHANNELS_RGB)
    mask_shape_in = (None, None, 1)
    patch_shape = (CONFIG.IMG_HEIGHT_EFFNET_B5, CONFIG.IMG_WIDTH_EFFNET_B5)

    @tf.function
    def _load(data):
        img = load_2d(data[COL_FILE], shape=img_shape_in)
        mask = load_2d(data[COL_TASK], shape=mask_shape_in)

        # extract patches
        img = extract_patches(tf.expand_dims(img, axis=0),
                              sizes=[1, patch_shape[0], patch_shape[1], 1],
                              strides=[1, patch_shape[0], patch_shape[1], 1],
                              rates=[1, 1, 1, 1],
                              padding='VALID')
        mask = extract_patches(tf.expand_dims(mask, axis=0),
                               sizes=[1, patch_shape[0], patch_shape[1], 1],
                               strides=[1, patch_shape[0], patch_shape[1], 1],
                               rates=[1, 1, 1, 1],
                               padding='VALID')
        img = tf.reshape(img, shape=[-1, patch_shape[0], patch_shape[1], img_shape_in[-1]])
        mask = tf.reshape(mask, shape=[-1, patch_shape[0], patch_shape[1], mask_shape_in[-1]])

        mask = tf.squeeze(mask, axis=-1)

        return img, mask

    data_train = data_train.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    data_valid = data_valid.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    data_test = data_test.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    # concatenate
    l_img_train = []
    l_mask_train = []
    for img_, mask_ in data_train:
        l_img_train.append(img_)
        l_mask_train.append(mask_)
    img_train = tf.concat(l_img_train, axis=0)
    mask_train = tf.concat(l_mask_train, axis=0)
    img_train = tf.data.Dataset.from_tensor_slices(img_train)
    mask_train = tf.data.Dataset.from_tensor_slices(mask_train)
    data_train = tf.data.Dataset.zip((img_train, mask_train))

    l_img_valid = []
    l_mask_valid = []
    for img_, mask_ in data_valid:
        l_img_valid.append(img_)
        l_mask_valid.append(mask_)
    img_valid = tf.concat(l_img_valid, axis=0)
    mask_valid = tf.concat(l_mask_valid, axis=0)
    img_valid = tf.data.Dataset.from_tensor_slices(img_valid)
    mask_valid = tf.data.Dataset.from_tensor_slices(mask_valid)
    data_valid = tf.data.Dataset.zip((img_valid, mask_valid))

    l_img_test = []
    l_mask_test = []
    for img_, mask_ in data_test:
        l_img_test.append(img_)
        l_mask_test.append(mask_)
    img_test = tf.concat(l_img_test, axis=0)
    mask_test = tf.concat(l_mask_test, axis=0)
    img_test = tf.data.Dataset.from_tensor_slices(img_test)
    mask_test = tf.data.Dataset.from_tensor_slices(mask_test)
    data_test = tf.data.Dataset.zip((img_test, mask_test))

    # organize
    img_shape_out = tuple([dim_.value for dim_ in data_test.element_spec[0].shape.dims])
    mask_shape_out = tuple([dim_.value for dim_ in data_test.element_spec[-1].shape.dims])

    dataset = {
        CONFIG.TAG_N_CLASSES: N_CLASSES,
        CONFIG.TAG_TRAIN: data_train,
        CONFIG.TAG_VALID: data_valid,
        CONFIG.TAG_TEST: data_test,
        CONFIG.TAG_SHAPE_X: img_shape_out,
        CONFIG.TAG_SHAPE_Y: mask_shape_out
    }
    return dataset
