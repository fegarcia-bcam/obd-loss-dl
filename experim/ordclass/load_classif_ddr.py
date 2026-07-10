# Author: Fernando García-García <fegarcia@bcamath.org>

import os

import pandas as pd

import tensorflow as tf
from tensorflow.data import Dataset

from experim.ordclass import CONFIG

from experim.ordclass.utils_load import load_2d

TAG_TASK_TYPE = CONFIG.TAG_CLASSIF
TAG_TARGET_TYPE = CONFIG.TAG_ORDIN

TAG_DATASET = 'DDR'

COL_FILE = 'filename'
COL_TASK = 'class'


def load(*, idx_split):
    # load info
    path_base = os.path.join(CONFIG.PATH_DATA, TAG_TASK_TYPE, TAG_TARGET_TYPE, TAG_DATASET)
    file_data = os.path.join(path_base, CONFIG.FILE_SPLIT)
    if not os.path.isfile(file_data):
        raise RuntimeError

    df_data = pd.read_csv(file_data)

    path_data = os.path.join(path_base, CONFIG.TAG_DATA)
    df_data[COL_FILE] = df_data[COL_FILE].apply(lambda x: os.path.join(path_data, x))

    n_classes = df_data[COL_TASK].nunique()

    # split
    tag_split = COL_TASK + CONFIG.TAG_SEP + CONFIG.TAG_SPLIT + str(idx_split)
    df_train = df_data[df_data[tag_split] == CONFIG.TAG_TRAIN]
    df_valid = df_data[df_data[tag_split] == CONFIG.TAG_VALID]
    df_test = df_data[df_data[tag_split] == CONFIG.TAG_TEST]

    data_train = Dataset.from_tensor_slices(dict(df_train))
    data_valid = Dataset.from_tensor_slices(dict(df_valid))
    data_test = Dataset.from_tensor_slices(dict(df_test))

    # load
    img_shape = (CONFIG.IMG_HEIGHT_EFFNET_B5, CONFIG.IMG_WIDTH_EFFNET_B5, CONFIG.IMG_CHANNELS_RGB)

    @tf.function
    def _load(data):
        img = load_2d(data[COL_FILE], shape=img_shape)
        lbl = tf.cast(data[COL_TASK], dtype=tf.float32)
        return img, lbl

    data_train = data_train.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    data_valid = data_valid.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    data_test = data_test.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    # organize
    dataset = {
        CONFIG.TAG_N_CLASSES: n_classes,
        CONFIG.TAG_TRAIN: data_train,
        CONFIG.TAG_VALID: data_valid,
        CONFIG.TAG_TEST: data_test,
        CONFIG.TAG_SHAPE_X: img_shape,
        CONFIG.TAG_SHAPE_Y: (CONFIG.N_DIMS_BATCH,)
    }
    return dataset
