# Author: Fernando García-García <fegarcia@bcamath.org>

import tensorflow as tf


@tf.function
def load_2d(filename, *, shape):
    # read from disk
    image = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(image)

    # cast as float
    image = tf.cast(image, dtype=tf.float32)

    # ensure shape
    if shape is not None:
        image = tf.ensure_shape(image, shape)

    return image
