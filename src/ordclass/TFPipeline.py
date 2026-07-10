# Author: Fernando García-García <fegarcia@bcamath.org>

import tensorflow as tf
from tensorflow.keras.layers import Layer


class Pipeline(Layer):
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

        return cls(steps=steps, **config)

    def build(self, input_shape):
        return

    def call(self, inputs, labels_in, training=True):
        # single input, single output
        if isinstance(inputs, tf.Tensor):
            outputs, labels_out = inputs, labels_in
            for step in self.steps:
                outputs, labels_out = step.call(outputs, labels_out, training=training)

        # multiple inputs
        elif isinstance(inputs, dict):
            # single output
            if isinstance(labels_in, tf.Tensor):
                outputs = {}
                labels_out = None
                for in_key, in_val in inputs.items():
                    out_ = in_val
                    lbl_ = labels_in
                    for step in self.steps:
                        out_, lbl_ = step.call(out_, lbl_, training=training)
                    outputs[in_key] = out_

                    if labels_out is None:
                        labels_out = lbl_  # here it is assumed that the steps agree on the same output labels

            # multiple outputs
            elif isinstance(labels_in, dict):
                outputs = {}
                labels_out = {}
                for (in_key, in_val), (lbl_key, lbl_val) in zip(inputs.items(), labels_in.items()):
                    out_ = in_val
                    lbl_ = lbl_val
                    for step in self.steps:
                        out_, lbl_ = step.call(out_, lbl_, training=training)
                    outputs[in_key] = out_
                    labels_out[lbl_key] = lbl_
            else:
                raise ValueError

        else:
            raise ValueError

        return outputs, labels_out
