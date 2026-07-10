# Author: Fernando García-García <fegarcia@bcamath.org>

import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Layer

from tensorflow.experimental.numpy import shape as np_shape

_MIN_CLASSES = 3
_MAX_CLASSES = 2 ** 8

_N_DIMS_BATCH = 1  # batch

_AX_FIRST = 1  # ignores batch dim 0
_AX_LAST = -1


class OrdinalOHE(Layer):
    def __init__(
            self,
            n_classes,
            *,
            data_format='channels_last',
            **kwargs
    ):
        super().__init__(**kwargs)

        if (not isinstance(n_classes, int)) or (n_classes < _MIN_CLASSES) or (n_classes > _MAX_CLASSES):
            raise ValueError
        self.n_classes = n_classes

        if data_format not in ['channels_last', 'channels_first']:
            raise ValueError
        self.data_format = data_format

    def get_config(self):
        base_config = super().get_config()
        config = {'n_classes': self.n_classes,
                  'data_format': self.data_format}

        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        n_classes = config.pop('n_classes')
        data_format = config.pop('data_format')

        return cls(n_classes=n_classes,
                   data_format=data_format,
                   **config)

    def build(self, input_shape):
        return

    def call(self, inputs, labels_in, training=True):
        n_dims_input = len(np_shape(inputs))

        # move input channels to the last position
        perm_axes = list(range(n_dims_input))
        if self.data_format == 'channels_first':
            perm_axes[_AX_FIRST], perm_axes[_AX_LAST] = perm_axes[_AX_LAST], perm_axes[_AX_FIRST]
            inputs = tf.transpose(inputs, perm=perm_axes)

        # guarantee order and consistency in the shapes
        n_dims_lbl_in = len(np_shape(labels_in))

        labels_classif = (n_dims_lbl_in == _N_DIMS_BATCH)

        labels_in = tf.expand_dims(labels_in, axis=_AX_LAST)
        if not labels_classif:
            # move label channels to the last position
            if self.data_format == 'channels_first':
                labels_in = tf.transpose(labels_in, perm=perm_axes)

        input_shape = np.asarray(np_shape(inputs))
        labels_in_shape = np.asarray(np_shape(labels_in))

        if not labels_classif:
            if input_shape.size != labels_in_shape.size:
                raise ValueError
            if np.any(input_shape[_AX_FIRST:_AX_LAST] != labels_in_shape[_AX_FIRST:_AX_LAST]):
                raise ValueError

        # one-hot encode labels
        dtype = labels_in.dtype
        l_classes = tf.constant(list(range(self.n_classes)), dtype=dtype)

        labels_out = tf.math.equal(labels_in[..., :], l_classes)
        labels_out = tf.cast(labels_out, dtype=dtype)

        # ensure output shape
        labels_out_shape = labels_in_shape
        if self.data_format == 'channels_first':
            labels_out_shape[_AX_FIRST] = self.n_classes
        else:
            labels_out_shape[_AX_LAST] = self.n_classes
        labels_out_shape = tuple(labels_out_shape)
        labels_out = tf.ensure_shape(labels_out, labels_out_shape)

        # leave inputs untouched
        outputs = inputs

        # move channels again to its original position
        if self.data_format == 'channels_first':
            outputs = tf.transpose(outputs, perm=perm_axes)
            if not labels_classif:
                labels_out = tf.transpose(labels_out, perm=perm_axes)

        return outputs, labels_out
