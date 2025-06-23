# Authors:  Fernando García-García <fegarcia@bcamath.org>

import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Cropping2D

from tensorflow.experimental.numpy import shape as np_shape


_N_CHANNELS_BW = 1  # luminance, grayscale
_N_CHANNELS_COL = 3  # color coding, e.g. RGB

_N_DIMS_FLAT_2D = 2  # height, width

_N_DIMS_IMAGE_2D = 3  # height, width, channels

_N_DIMS_BATCH_2D = 4  # batch, height, width, channels

_N_CHANNELS_LABEL_IDX = 1

_N_DIMS_BATCH = 1  # batch
_N_DIMS_LABELS = 2  # batch, numeric or one-hot encoded label


class CropCenter2D(Layer):
    def __init__(
            self,
            height,
            width,

            data_format='channels_last',

            **kwargs
    ):
        super().__init__(**kwargs)

        if ((not isinstance(height, (int, np.int64))) or (height <= 0)
                or (not isinstance(width, (int, np.int64))) or (width <= 0)):
            raise ValueError
        self.height = height
        self.width = width

        if data_format not in ['channels_last', 'channels_first']:
            raise ValueError
        self.data_format = data_format

    def get_config(self):
        base_config = super().get_config()
        config = {'height': self.height, 'width': self.width,
                  'data_format': self.data_format}

        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        height = config.pop('height')
        width = config.pop('width')
        data_format = config.pop('data_format')

        return cls(height=height, width=width,
                   data_format=data_format,
                   **config)

    def call(self, inputs):
        input_shape = np.asarray(np_shape(inputs))
        n_dims_input = input_shape.size

        # check shape consistency
        if n_dims_input not in [_N_DIMS_IMAGE_2D, _N_DIMS_BATCH_2D]:
            raise ValueError

        is_batched = (n_dims_input == _N_DIMS_BATCH_2D)
        if not is_batched:
            inputs = tf.expand_dims(inputs, axis=0)

        if self.data_format == 'channels_last':
            ax_h, ax_w = 1, 2
        else:  # 'channels_first'
            ax_h, ax_w = 2, 3
        size_h, size_w = input_shape[ax_h], input_shape[ax_w]

        crop_ht = (size_h - self.height) // 2  # height: top
        crop_hb = size_h - (self.height + crop_ht)  # height: bottom
        crop_wl = (size_w - self.width) // 2  # width: left
        crop_wr = size_w - (self.width + crop_wl)  # width: right

        layer_crop_2d = Cropping2D(cropping=((crop_ht, crop_hb), (crop_wl, crop_wr)),
                                   data_format=self.data_format)

        outputs = layer_crop_2d(inputs)
        if not is_batched:
            outputs = tf.squeeze(outputs, axis=0)

        return outputs
