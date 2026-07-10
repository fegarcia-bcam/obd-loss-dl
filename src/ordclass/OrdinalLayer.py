# Author: Fernando García-García <fegarcia@bcamath.org>

import tensorflow as tf
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.layers import Layer

from tensorflow.experimental.numpy import shape as np_shape

_MIN_CLASSES = 3
_MAX_CLASSES = 2 ** 8

_AX_FIRST = 1  # ignores batch dim 0
_AX_LAST = -1


class Regression(Layer):
    def __init__(
            self,
            n_classes,
            *,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
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

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def get_config(self):
        base_config = super().get_config()
        config = {'n_classes': self.n_classes,
                  'kernel_initializer': tf.keras.utils.serialize_keras_object(self.kernel_initializer),
                  'bias_initializer': tf.keras.utils.serialize_keras_object(self.bias_initializer),
                  'kernel_regularizer': tf.keras.utils.serialize_keras_object(self.kernel_regularizer),
                  'bias_regularizer': tf.keras.utils.serialize_keras_object(self.bias_regularizer),
                  'kernel_constraint': tf.keras.utils.serialize_keras_object(self.kernel_constraint),
                  'bias_constraint': tf.keras.utils.serialize_keras_object(self.bias_constraint),
                  'data_format': self.data_format}

        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        n_classes = config.pop('n_classes')
        kernel_initializer = config.pop('kernel_initializer')
        bias_initializer = config.pop('bias_initializer')
        kernel_regularizer = config.pop('kernel_regularizer')
        bias_regularizer = config.pop('bias_regularizer')
        kernel_constraint = config.pop('kernel_constraint')
        bias_constraint = config.pop('bias_constraint')
        data_format = config.pop('data_format')

        kernel_initializer = tf.keras.utils.deserialize_keras_object(kernel_initializer)
        bias_initializer = tf.keras.utils.deserialize_keras_object(bias_initializer)
        kernel_regularizer = tf.keras.utils.deserialize_keras_object(kernel_regularizer)
        bias_regularizer = tf.keras.utils.deserialize_keras_object(bias_regularizer)
        kernel_constraint = tf.keras.utils.deserialize_keras_object(kernel_constraint)
        bias_constraint = tf.keras.utils.deserialize_keras_object(bias_constraint)

        return cls(n_classes=n_classes,
                   kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                   kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                   kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                   data_format=data_format,
                   **config)

    def build(self, input_shape):
        if self.data_format == 'channels_last':
            ax_c = _AX_LAST
        else:  # 'channels_first'
            ax_c = _AX_FIRST

        self.w = self.add_weight(shape=(input_shape[ax_c],),
                                 initializer=self.kernel_initializer,
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint,
                                 name='kernel',
                                 trainable=True)
        self.b = self.add_weight(shape=(1,),
                                 initializer=self.bias_initializer,
                                 regularizer=self.bias_regularizer,
                                 constraint=self.bias_constraint,
                                 name='bias',
                                 trainable=True)

    def call(self, inputs):
        n_dims_input = len(np_shape(inputs))

        # move channels to the last position
        perm_axes = list(range(n_dims_input))
        if self.data_format == 'channels_first':
            perm_axes[_AX_FIRST], perm_axes[_AX_LAST] = perm_axes[_AX_LAST], perm_axes[_AX_FIRST]
            inputs = tf.transpose(inputs, perm=perm_axes)

        # outputs
        outputs = tf.linalg.matvec(inputs, self.w) + self.b
        outputs = tf.clip_by_value(outputs,
                                   clip_value_min=0,
                                   clip_value_max=self.n_classes - 1)

        # move channels again to its original position
        if self.data_format == 'channels_first':
            outputs = tf.transpose(outputs, perm=perm_axes)

        return outputs


class Nominal(Layer):
    def __init__(
            self,
            n_classes,
            *,
            to_logits=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            data_format='channels_last',
            **kwargs
    ):
        super().__init__(**kwargs)

        if (not isinstance(n_classes, int)) or (n_classes < _MIN_CLASSES) or (n_classes > _MAX_CLASSES):
            raise ValueError
        self.n_classes = n_classes

        if not isinstance(to_logits, bool):
            raise ValueError
        self.to_logits = to_logits

        if data_format not in ['channels_last', 'channels_first']:
            raise ValueError
        self.data_format = data_format

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def get_config(self):
        base_config = super().get_config()
        config = {'n_classes': self.n_classes, 'to_logits': self.to_logits,
                  'kernel_initializer': tf.keras.utils.serialize_keras_object(self.kernel_initializer),
                  'bias_initializer': tf.keras.utils.serialize_keras_object(self.bias_initializer),
                  'kernel_regularizer': tf.keras.utils.serialize_keras_object(self.kernel_regularizer),
                  'bias_regularizer': tf.keras.utils.serialize_keras_object(self.bias_regularizer),
                  'kernel_constraint': tf.keras.utils.serialize_keras_object(self.kernel_constraint),
                  'bias_constraint': tf.keras.utils.serialize_keras_object(self.bias_constraint),
                  'data_format': self.data_format}

        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        n_classes = config.pop('n_classes')
        to_logits = config.pop('to_logits')
        kernel_initializer = config.pop('kernel_initializer')
        bias_initializer = config.pop('bias_initializer')
        kernel_regularizer = config.pop('kernel_regularizer')
        bias_regularizer = config.pop('bias_regularizer')
        kernel_constraint = config.pop('kernel_constraint')
        bias_constraint = config.pop('bias_constraint')
        data_format = config.pop('data_format')

        kernel_initializer = tf.keras.utils.deserialize_keras_object(kernel_initializer)
        bias_initializer = tf.keras.utils.deserialize_keras_object(bias_initializer)
        kernel_regularizer = tf.keras.utils.deserialize_keras_object(kernel_regularizer)
        bias_regularizer = tf.keras.utils.deserialize_keras_object(bias_regularizer)
        kernel_constraint = tf.keras.utils.deserialize_keras_object(kernel_constraint)
        bias_constraint = tf.keras.utils.deserialize_keras_object(bias_constraint)

        return cls(n_classes=n_classes, to_logits=to_logits,
                   kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                   kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                   kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                   data_format=data_format,
                   **config)

    def build(self, input_shape):
        if self.data_format == 'channels_last':
            ax_c = _AX_LAST
        else:  # 'channels_first'
            ax_c = _AX_FIRST

        self.w = self.add_weight(shape=(input_shape[ax_c], self.n_classes),
                                 initializer=self.kernel_initializer,
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint,
                                 name='kernel',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.n_classes,),
                                 initializer=self.bias_initializer,
                                 regularizer=self.bias_regularizer,
                                 constraint=self.bias_constraint,
                                 name='bias',
                                 trainable=True)

    def call(self, inputs):
        n_dims_input = len(np_shape(inputs))

        # move channels to the last position
        perm_axes = list(range(n_dims_input))
        if self.data_format == 'channels_first':
            perm_axes[_AX_FIRST], perm_axes[_AX_LAST] = perm_axes[_AX_LAST], perm_axes[_AX_FIRST]
            inputs = tf.transpose(inputs, perm=perm_axes)

        # outputs
        outputs = tf.linalg.matmul(inputs, self.w) + self.b
        if not self.to_logits:
            outputs = tf.nn.softmax(outputs, axis=_AX_LAST)
            outputs = tf.clip_by_value(outputs,
                                       clip_value_min=tf.keras.backend.epsilon(),
                                       clip_value_max=1.0 - tf.keras.backend.epsilon())

        # move channels again to its original position
        if self.data_format == 'channels_first':
            outputs = tf.transpose(outputs, perm=perm_axes)

        return outputs
