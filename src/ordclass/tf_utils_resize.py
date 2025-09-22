# Authors:  Fernando García-García <fegarcia@bcamath.org>

import tensorflow as tf
from tensorflow.keras.layers import Layer

from tensorflow.experimental.numpy import shape as np_shape


_N_DIMS_BASE_2D = 2  # height, width

_N_DIMS_MAIN_2D = 3  # height, width, channels

_N_DIMS_BATCH_2D = 4  # batch, height, width, channels

_AX_FIRST = 1  # ignores batch dim 0
_AX_LAST = -1


class Resize2D(Layer):
    def __init__(
            self,
            height,
            width,

            method='bilinear',
            antialias=False,

            data_format='channels_last',

            **kwargs
    ):
        super().__init__(**kwargs)
        self.height = height
        self.width = width

        self.method = method
        self.antialias = antialias

        if data_format not in ['channels_last', 'channels_first']:
            raise ValueError
        self.data_format = data_format

    def get_config(self):
        base_config = super().get_config()
        config = {'height': self.height,
                  'width': self.width,
                  'method': self.method,
                  'antialias': self.antialias,
                  'data_format': self.data_format}

        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        height = config.pop('height')
        width = config.pop('width')
        method = config.pop('method')
        antialias = config.pop('antialias')
        data_format = config.pop('data_format')

        return cls(height=height, width=width,
                   method=method, antialias=antialias,
                   data_format=data_format,
                   **config)

    def call(self, inputs):
        n_dims_input = len(np_shape(inputs))

        # check shape consistency
        if n_dims_input not in [_N_DIMS_MAIN_2D, _N_DIMS_BATCH_2D]:
            raise ValueError

        is_batched = (n_dims_input == _N_DIMS_BATCH_2D)
        if not is_batched:
            inputs = tf.expand_dims(inputs, axis=0)

        # move channels to the last position
        perm_axes = list(range(_N_DIMS_BATCH_2D))
        if self.data_format == 'channels_first':
            perm_axes[_AX_FIRST], perm_axes[_AX_LAST] = perm_axes[_AX_LAST], perm_axes[_AX_FIRST]
            inputs = tf.transpose(inputs, perm=perm_axes)

        # resize
        outputs = tf.image.resize(inputs,
                                  size=(self.height, self.width),
                                  preserve_aspect_ratio=False,
                                  method=self.method,
                                  antialias=self.antialias)

        # move channels again to its original position
        if self.data_format == 'channels_first':
            outputs = tf.transpose(outputs, perm=perm_axes)

        if not is_batched:
            outputs = tf.squeeze(outputs, axis=0)

        return outputs
