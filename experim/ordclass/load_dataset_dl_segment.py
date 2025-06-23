import os

import pandas as pd

import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.image import extract_patches


# paths and files
PATH_DATA = 'data/Ordinal_DL'

TAG_SEGMENT = 'Segmentation'

TAG_ORDIN = 'Ordinal'
TAG_DISCR = 'Discretized'

TAG_SPLIT = 'split'
TAG_TRAIN = 'train'
TAG_VALID = 'valid'
TAG_TEST = 'test'
TAG_TARGET = 'target'

EXT_DATA_CSV = '.csv'

EXT_IMG_PNG = '.png'

# image properties
IMG_HEIGHT_B0 = 224
IMG_HEIGHT_B5 = 456
IMG_WIDTH_B0 = 224
IMG_WIDTH_B5 = 456
IMG_CHANNELS_RGB = 3

MASK_CHANNELS = 1


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


def load_bach():
    folder = os.path.join(PATH_DATA, TAG_SEGMENT, TAG_ORDIN, 'Histology-ICIAR_BACH')
    file_info = os.path.join(folder, 'info_split' + EXT_DATA_CSV)
    df_info = pd.read_csv(file_info)

    tag_file = 'file_name'
    cols_sel = [TAG_SPLIT, tag_file]
    df_info = df_info[cols_sel]

    n_classes = 5

    tag_file_img = 'filename_image'
    tag_file_mask = 'filename_mask'
    df_info[tag_file_img] = df_info.apply(lambda x: os.path.join(folder, x[TAG_SPLIT], x[tag_file] + '_image' + EXT_IMG_PNG),
                                          axis='columns')
    df_info[tag_file_mask] = df_info.apply(lambda x: os.path.join(folder, x[TAG_SPLIT], x[tag_file] + '_mask' + EXT_IMG_PNG),
                                           axis='columns')

    df_train = df_info.loc[df_info[TAG_SPLIT] == TAG_TRAIN]
    df_valid = df_info.loc[df_info[TAG_SPLIT] == TAG_VALID]
    df_test = df_info.loc[df_info[TAG_SPLIT] == TAG_TEST]

    data_train = Dataset.from_tensor_slices(dict(df_train))
    data_valid = Dataset.from_tensor_slices(dict(df_valid))
    data_test = Dataset.from_tensor_slices(dict(df_test))

    img_shape_in = (None, None, IMG_CHANNELS_RGB)
    mask_shape_in = (None, None, MASK_CHANNELS)
    patch_shape = (IMG_HEIGHT_B5, IMG_WIDTH_B5)

    @tf.function
    def _load(data):
        img = load_image_2d(data[tag_file_img], shape=img_shape_in)
        mask = load_image_2d(data[tag_file_mask], shape=mask_shape_in)

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

    # final info
    n_train = int(data_train.cardinality().numpy())
    n_valid = int(data_valid.cardinality().numpy())
    n_test = int(data_test.cardinality().numpy())
    n_total = n_train + n_valid + n_test
    ratio_train = n_train / n_total
    ratio_valid = n_valid / n_total
    ratio_test = n_test / n_total

    img_shape_out = tuple([dim_.value for dim_ in data_train.element_spec[0].shape.dims])
    mask_shape_out = tuple([dim_.value for dim_ in data_train.element_spec[-1].shape.dims])

    info = {'name': 'ICIAR-BACH Segment',
            'n_total': n_total, 'n_train': n_train, 'n_valid': n_valid, 'n_test': n_test,
            'ratio_train': ratio_train, 'ratio_valid': ratio_valid, 'ratio_test': ratio_test,
            'n_classes': n_classes,
            'X_shape': img_shape_out, 'y_shape': mask_shape_out,
            'data_train': data_train, 'data_valid': data_valid, 'data_test': data_test}
    return info
