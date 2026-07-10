# Author: Fernando García-García <fegarcia@bcamath.org>

from numbers import Integral, Real

import numpy as np

from sklearn.utils import check_array, column_or_1d
from sklearn.utils._param_validation import validate_params, Interval

import tensorflow as tf

_MIN_CLASSES = 2  # conversely to the ordinal, the nominal don't require at least 3 classes
_MAX_CLASSES = 2 ** 8


@validate_params(
    parameter_constraints={'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')],
                           'from_logits': ['boolean'],
                           'gamma': [Interval(Real, 0.0, None, closed='left')],
                           'class_weight': ['array-like', None]},
    prefer_skip_nested_validation=True
)
def focal_loss(n_classes, *, from_logits=True, gamma=2.0, class_weight=None):
    # check class weights
    if class_weight is None:
        class_weight = np.ones(shape=(n_classes,), dtype=np.float32)
    else:
        class_weight = column_or_1d(class_weight)
        class_weight = check_array(class_weight,
                                   dtype=np.float32,
                                   ensure_2d=False,
                                   ensure_all_finite=True,
                                   ensure_non_negative=True)
        if class_weight.size != n_classes:
            raise ValueError
    class_weight = tf.convert_to_tensor(class_weight, dtype=np.float32)

    @tf.function
    def loss_fn(y_true, y_proba):
        if from_logits:
            y_proba = tf.nn.softmax(y_proba, axis=-1)
        else:
            y_total = tf.math.reduce_sum(y_proba, axis=-1, keepdims=True)
            y_proba = tf.math.divide_no_nan(y_proba, y_total)

        # prevent issues with numerical precision
        y_proba = tf.clip_by_value(y_proba,
                                   clip_value_min=tf.keras.backend.epsilon(),
                                   clip_value_max=1.0 - tf.keras.backend.epsilon())

        # focal loss
        losses_pow = tf.math.pow(1.0 - y_proba, gamma)
        losses_log = -1.0 * tf.math.log(y_proba)
        losses = tf.math.multiply(losses_pow, losses_log)
        losses = tf.math.multiply(y_true, losses)
        losses = tf.math.reduce_sum(losses, axis=-1)

        # apply class weights
        sample_weight = tf.linalg.matvec(y_true, class_weight)
        losses = tf.math.multiply(sample_weight, losses)

        # reduce to mean
        loss = tf.math.reduce_mean(losses, axis=None)
        return loss

    return loss_fn
