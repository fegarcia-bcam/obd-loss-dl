# Authors:  Fernando García-García <fegarcia@bcamath.org>

import tensorflow as tf
from tensorflow.keras.layers import Layer


class RescaleWithLabels(Layer):
    def __init__(
            self,
            scale,
            offset=0.0,

            **kwargs
    ):
        super().__init__(**kwargs)
        self.scale = scale
        self.offset = offset

    def get_config(self):
        base_config = super().get_config()
        config = {'scale': self.scale,
                  'offset': self.offset}

        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        scale = config.pop('scale')
        offset = config.pop('offset')

        return cls(scale=scale, offset=offset,
                   **config)

    def call(self, inputs, labels_in, training=True):
        dtype = inputs.dtype
        scale = tf.cast(self.scale, dtype=dtype)
        offset = tf.cast(self.offset, dtype=dtype)

        outputs = inputs * scale + offset
        labels_out = labels_in

        return outputs, labels_out


class PipelineWithLabels(Layer):
    def __init__(
            self,
            steps,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.steps = steps

    def get_config(self):
        base_config = super().get_config()
        config = {'steps': tf.keras.utils.serialize_keras_object(self.steps)}

        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        steps = config.pop('steps')
        steps = tf.keras.utils.deserialize_keras_object(steps)

        return cls(steps, **config)

    def call(self, inputs, labels_in, training=True):
        outputs, labels_out = inputs, labels_in
        for step in self.steps:
            outputs, labels_out = step.call(outputs, labels_out, training=training)

        return outputs, labels_out
