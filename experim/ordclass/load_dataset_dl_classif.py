import os

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.data import Dataset


# paths and files
PATH_DATA = 'data/Ordinal_DL'

TAG_CLASSIF = 'Classification'

TAG_ORDIN = 'Ordinal'
TAG_DISCR = 'Discretized'

TAG_SPLIT = 'split'
TAG_TRAIN = 'train'
TAG_VALID = 'valid'
TAG_TEST = 'test'
TAG_TARGET = 'target'

EXT_DATA_CSV = '.csv'

# image properties
IMG_HEIGHT_B0 = 224
IMG_HEIGHT_B5 = 456
IMG_WIDTH_B0 = 224
IMG_WIDTH_B5 = 456
IMG_CHANNELS_RGB = 3


@tf.function
def load_image_2d(filename, shape):
    # read from disk
    image = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(image)
    # cast as float
    image = tf.cast(image, dtype=tf.float32)

    # ensure shape
    if shape is not None:
        image = tf.ensure_shape(image, shape)

    return image


def load_utk(task):
    if task not in ['gr5', 'gr10', 'thresh']:
        raise ValueError

    folder = os.path.join(PATH_DATA, TAG_CLASSIF, TAG_DISCR, 'Facial_age-UTK')
    file_info = os.path.join(folder, 'split' + EXT_DATA_CSV)
    df_info = pd.read_csv(file_info)

    tag_file = 'file_name'
    col_target = 'age_{}'.format(task)
    cols_sel = [TAG_SPLIT, tag_file, col_target]
    df_info = df_info[cols_sel]
    df_info = df_info.rename(columns={col_target: TAG_TARGET})

    n_classes = df_info[TAG_TARGET].nunique()
    if (df_info[TAG_TARGET].min() != 0) or (df_info[TAG_TARGET].max() != n_classes - 1):
        raise RuntimeError

    df_info[tag_file] = df_info.apply(lambda x: os.path.join(folder, x[TAG_SPLIT], x[tag_file]),
                                      axis='columns')

    df_train = df_info.loc[df_info[TAG_SPLIT] == TAG_TRAIN]
    df_valid = df_info.loc[df_info[TAG_SPLIT] == TAG_VALID]
    df_test = df_info.loc[df_info[TAG_SPLIT] == TAG_TEST]

    n_train = len(df_train.index)
    n_valid = len(df_valid.index)
    n_test = len(df_test.index)
    n_total = n_train + n_valid + n_test
    ratio_train = n_train / n_total
    ratio_valid = n_valid / n_total
    ratio_test = n_test / n_total

    counts_train = np.zeros(shape=(n_classes,), dtype=np.uint64)
    counts_valid = np.zeros(shape=(n_classes,), dtype=np.uint64)
    counts_test = np.zeros(shape=(n_classes,), dtype=np.uint64)
    for c in range(n_classes):
        counts_train[c] = (df_train[TAG_TARGET] == c).sum()
        counts_valid[c] = (df_valid[TAG_TARGET] == c).sum()
        counts_test[c] = (df_test[TAG_TARGET] == c).sum()
    priors_train = counts_train / n_train
    priors_valid = counts_valid / n_valid
    priors_test = counts_test / n_test

    data_train = Dataset.from_tensor_slices(dict(df_train))
    data_valid = Dataset.from_tensor_slices(dict(df_valid))
    data_test = Dataset.from_tensor_slices(dict(df_test))

    img_shape = (IMG_HEIGHT_B0, IMG_WIDTH_B0, IMG_CHANNELS_RGB)

    def _load(data):
        img = load_image_2d(data[tag_file], shape=img_shape)
        lbl = tf.cast(data[TAG_TARGET], dtype=tf.float32)
        return img, lbl

    data_train = data_train.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    data_valid = data_valid.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    data_test = data_test.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    info = {'name': 'UTK@{}'.format(task.capitalize()),
            'n_total': n_total, 'n_train': n_train, 'n_valid': n_valid, 'n_test': n_test,
            'ratio_train': ratio_train, 'ratio_valid': ratio_valid, 'ratio_test': ratio_test,
            'n_classes': n_classes,
            'counts_train': counts_train, 'counts_valid': counts_valid, 'counts_test': counts_test,
            'priors_train': priors_train, 'priors_valid': priors_valid, 'priors_test': priors_test,
            'X_shape': img_shape, 'y_shape': (1,),
            'data_train': data_train, 'data_valid': data_valid, 'data_test': data_test}
    return info


def load_ddr():
    folder = os.path.join(PATH_DATA, TAG_CLASSIF, TAG_ORDIN, 'Retinopathy-DDR')
    file_info = os.path.join(folder, 'split-all' + EXT_DATA_CSV)
    df_info = pd.read_csv(file_info)

    tag_file = 'filename'
    col_target = 'dr_group'
    cols_sel = [TAG_SPLIT, tag_file, col_target]
    df_info = df_info[cols_sel]
    df_info = df_info.rename(columns={col_target: TAG_TARGET})

    n_classes = df_info[TAG_TARGET].nunique()
    if (df_info[TAG_TARGET].min() != 0) or (df_info[TAG_TARGET].max() != n_classes - 1):
        raise RuntimeError

    df_info[tag_file] = df_info.apply(lambda x: os.path.join(folder, x[TAG_SPLIT], str(x[TAG_TARGET]), x[tag_file]),
                                      axis='columns')

    df_train = df_info.loc[df_info[TAG_SPLIT] == TAG_TRAIN]
    df_valid = df_info.loc[df_info[TAG_SPLIT] == TAG_VALID]
    df_test = df_info.loc[df_info[TAG_SPLIT] == TAG_TEST]

    n_train = len(df_train.index)
    n_valid = len(df_valid.index)
    n_test = len(df_test.index)
    n_total = n_train + n_valid + n_test
    ratio_train = n_train / n_total
    ratio_valid = n_valid / n_total
    ratio_test = n_test / n_total

    counts_train = np.zeros(shape=(n_classes,), dtype=np.uint64)
    counts_valid = np.zeros(shape=(n_classes,), dtype=np.uint64)
    counts_test = np.zeros(shape=(n_classes,), dtype=np.uint64)
    for c in range(n_classes):
        counts_train[c] = (df_train[TAG_TARGET] == c).sum()
        counts_valid[c] = (df_valid[TAG_TARGET] == c).sum()
        counts_test[c] = (df_test[TAG_TARGET] == c).sum()
    priors_train = counts_train / n_train
    priors_valid = counts_valid / n_valid
    priors_test = counts_test / n_test

    data_train = Dataset.from_tensor_slices(dict(df_train))
    data_valid = Dataset.from_tensor_slices(dict(df_valid))
    data_test = Dataset.from_tensor_slices(dict(df_test))

    img_shape = (IMG_HEIGHT_B5, IMG_WIDTH_B5, IMG_CHANNELS_RGB)

    def _load(data):
        img = load_image_2d(data[tag_file], shape=img_shape)
        lbl = tf.cast(data[TAG_TARGET], dtype=tf.float32)
        return img, lbl

    data_train = data_train.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    data_valid = data_valid.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    data_test = data_test.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    info = {'name': 'DDR',
            'n_total': n_total, 'n_train': n_train, 'n_valid': n_valid, 'n_test': n_test,
            'ratio_train': ratio_train, 'ratio_valid': ratio_valid, 'ratio_test': ratio_test,
            'n_classes': n_classes,
            'counts_train': counts_train, 'counts_valid': counts_valid, 'counts_test': counts_test,
            'priors_train': priors_train, 'priors_valid': priors_valid, 'priors_test': priors_test,
            'X_shape': img_shape, 'y_shape': (1,),
            'data_train': data_train, 'data_valid': data_valid, 'data_test': data_test}
    return info
