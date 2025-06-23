# Authors:  Fernando García-García <fegarcia@bcamath.org>

import numpy as np

import tensorflow as tf
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.layers import Layer

from tensorflow.experimental.numpy import shape as np_shape


_MIN_CLASSES = 3
_MAX_CLASSES = 2 ** 8

_N_DIMS_BATCH = 1  # batch
_N_DIMS_LABELS = 2  # batch, numeric or one-hot encoded label

_AX_FIRST = 1
_AX_LAST = -1


class OrdinalOneHotEncoder(Layer):
    def __init__(
            self,
            n_classes,

            data_format='channels_last',

            **kwargs
    ):
        super().__init__(**kwargs)

        if (not isinstance(n_classes, (int, np.int64))) or (n_classes < _MIN_CLASSES) or (n_classes > _MAX_CLASSES):
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

    def call(self, inputs, labels_in, training=True):
        n_dims_input = len(np_shape(inputs))

        # move input channels to the last position
        perm_axes = list(range(n_dims_input))
        if self.data_format != 'channels_last':
            perm_axes[_AX_FIRST], perm_axes[_AX_LAST] = perm_axes[_AX_LAST], perm_axes[_AX_FIRST]
            inputs = tf.transpose(inputs, perm=perm_axes)

        # guarantee order and consistency in the labels' shape
        n_dims_lbl_in = len(np_shape(labels_in))

        labels_num = (n_dims_lbl_in in [_N_DIMS_LABELS - _N_DIMS_BATCH, _N_DIMS_LABELS])
        labels_num_flat = labels_num and (n_dims_lbl_in == _N_DIMS_LABELS - _N_DIMS_BATCH)
        labels_mask = (not labels_num) and (n_dims_lbl_in in [n_dims_input - _N_DIMS_BATCH, n_dims_input])
        labels_mask_flat = labels_mask and (n_dims_lbl_in == n_dims_input - _N_DIMS_BATCH)

        if not (labels_num or labels_mask):
            raise ValueError
        
        if labels_num_flat:
            labels_in = tf.expand_dims(labels_in, axis=_AX_LAST)

        if labels_mask:
            # move labels channels to the last position
            if self.data_format == 'channels_first':
                if labels_mask_flat:
                    labels_in = tf.expand_dims(labels_in, axis=_AX_FIRST)
                labels_in = tf.transpose(labels_in, perm=perm_axes)
            else:
                if labels_mask_flat:
                    labels_in = tf.expand_dims(labels_in, axis=_AX_LAST)

        input_shape = np.asarray(np_shape(inputs))
        labels_in_shape = np.asarray(np_shape(labels_in))
        if labels_mask:
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
        if self.data_format != 'channels_last':
            outputs = tf.transpose(outputs, perm=perm_axes)
            if labels_mask:
                labels_out = tf.transpose(labels_out, perm=perm_axes)
                if labels_mask_flat:
                    labels_out = tf.squeeze(labels_in, axis=_AX_FIRST)

        return outputs, labels_out


class RegressionOutput(Layer):
    def __init__(
            self,
            n_classes,

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

        if (not isinstance(n_classes, (int, np.int64))) or (n_classes < _MIN_CLASSES) or (n_classes > _MAX_CLASSES):
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
        if self.data_format != 'channels_last':
            perm_axes[_AX_FIRST], perm_axes[_AX_LAST] = perm_axes[_AX_LAST], perm_axes[_AX_FIRST]
            inputs = tf.transpose(inputs, perm=perm_axes)

        # outputs
        outputs = tf.linalg.matvec(inputs, self.w) + self.b
        outputs = tf.clip_by_value(outputs,
                                   clip_value_min=0,
                                   clip_value_max=self.n_classes - 1)

        # move channels again to its original position
        if self.data_format != 'channels_last':
            outputs = tf.transpose(outputs, perm=perm_axes)

        return outputs


class NominalOutput(Layer):
    def __init__(
            self,
            n_classes,
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

        if (not isinstance(n_classes, (int, np.int64))) or (n_classes < _MIN_CLASSES) or (n_classes > _MAX_CLASSES):
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
        if self.data_format != 'channels_last':
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
        if self.data_format != 'channels_last':
            outputs = tf.transpose(outputs, perm=perm_axes)

        return outputs
