# Author: Fernando García-García <fegarcia@bcamath.org>

from numbers import Integral, Real

import numpy as np

import tensorflow as tf

from sklearn.utils import check_array, column_or_1d
from sklearn.utils._param_validation import validate_params, Interval

from src.ordclass.unimodal import get_regularization
from src.ordclass.OWK import check_owk_weight
from src.ordclass.OBD import check_obd_weight


_MIN_CLASSES = 3
_MAX_CLASSES = 2 ** 8


@validate_params(
    parameter_constraints={'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')],
                           'from_logits': ['boolean'],
                           'class_weight': ['array-like', None]},
    prefer_skip_nested_validation=True
)
def ur_ce_loss(n_classes, *,
               from_logits=True,
               ur_type='beta', ur_eta=0.0, ur_delta=1.0,
               class_weight=None):
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

    # obtain regularization terms
    ur_terms = get_regularization(n_classes, type=ur_type, delta=ur_delta)
    ur_terms = tf.convert_to_tensor(ur_terms, dtype=np.float32)

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

        # regularized ground truth labels
        y_regul = (1.0 - ur_eta) * y_true + ur_eta * tf.linalg.matmul(y_true, tf.transpose(ur_terms))

        # cross-entropy loss
        losses = -1.0 * tf.math.log(y_proba)
        losses = tf.math.multiply(y_regul, losses)
        losses = tf.math.reduce_sum(losses, axis=-1)

        # apply class weights
        sample_weight = tf.linalg.matvec(y_true, class_weight)
        losses = tf.math.multiply(sample_weight, losses)

        # reduce to mean
        loss = tf.math.reduce_mean(losses, axis=None)
        return loss

    return loss_fn


@validate_params(
    parameter_constraints={'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')],
                           'from_logits': ['boolean'],
                           'gamma': [Interval(Real, 0.0, None, closed='left')],
                           'class_weight': ['array-like', None]},
    prefer_skip_nested_validation=True
)
def ur_focal_loss(n_classes, *,
                  from_logits=True,
                  gamma=2.0,
                  ur_type='beta', ur_eta=0.0, ur_delta=1.0,
                  class_weight=None):
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

    # obtain regularization terms
    ur_terms = get_regularization(n_classes, type=ur_type, delta=ur_delta)
    ur_terms = tf.convert_to_tensor(ur_terms, dtype=np.float32)

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

        # regularized ground truth labels
        y_regul = (1.0 - ur_eta) * y_true + ur_eta * tf.linalg.matmul(y_true, tf.transpose(ur_terms))

        # focal loss
        losses_log = -1.0 * tf.math.log(y_proba)
        losses_pow = tf.math.pow(1.0 - y_proba, gamma)
        losses = tf.math.multiply(losses_log, losses_pow)
        losses = tf.math.multiply(y_regul, losses)
        losses = tf.math.reduce_sum(losses, axis=-1)

        # apply class weights
        sample_weight = tf.linalg.matvec(y_true, class_weight)
        losses = tf.math.multiply(sample_weight, losses)

        # reduce to mean
        loss = tf.math.reduce_mean(losses, axis=None)
        return loss

    return loss_fn


@validate_params(
    parameter_constraints={'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')],
                           'from_logits': ['boolean'],
                           'class_priors': ['array-like', None],
                           'class_weight': ['array-like', None]},
    prefer_skip_nested_validation=True
)
def owk_loss(n_classes, *,
             from_logits=True,
             owk_weight=None, class_priors=None,
             class_weight=None):
    # check OWK weights
    owk_weight = check_owk_weight(owk_weight)
    owk_weight = tf.convert_to_tensor(owk_weight, dtype=np.float32)

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

    # check class priors
    if class_priors is None:
        class_priors = np.ones(shape=(n_classes,), dtype=np.float32)
    else:
        class_priors = column_or_1d(class_priors)
        class_priors = check_array(class_priors,
                                   dtype=np.float32,
                                   ensure_2d=False,
                                   ensure_all_finite=True,
                                   ensure_non_negative=True)
        if class_priors.size != n_classes:
            raise ValueError
        if ~np.isclose(np.sum(class_priors), 1.0):
            raise ValueError
    class_priors = tf.convert_to_tensor(class_priors, dtype=np.float32)

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

        # weighted kappa calculations
        w_true = tf.linalg.matmul(y_true, owk_weight)
        kappa_num = tf.math.multiply(w_true, y_proba)
        kappa_num = tf.math.reduce_sum(kappa_num, axis=-1)

        kappa_den = tf.zeros_like(kappa_num)
        for c in range(n_classes):
            w_temp = owk_weight[c, :]
            kappa_den_temp = tf.math.multiply(w_temp, y_proba)
            kappa_den_temp = tf.math.reduce_sum(kappa_den_temp, axis=-1)
            kappa_den += class_priors[c] * kappa_den_temp

        losses = tf.math.divide_no_nan(kappa_num, kappa_den)
        # losses = tf.math.log(losses)
        # we do not use the logarithm here
        # npt only to avoid issues with log(0) - which could be solved by clipping
        # but also because convergence seems to be easier

        # apply class weights
        sample_weight = tf.linalg.matvec(y_true, class_weight)
        losses = tf.math.multiply(sample_weight, losses)

        # reduce to mean
        loss = tf.math.reduce_mean(losses, axis=None)
        return loss

    return loss_fn


@validate_params(
    parameter_constraints={'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')],
                           'from_logits': ['boolean'],
                           'class_weight': ['array-like', None]},
    prefer_skip_nested_validation=True
)
def obd_ce_loss(n_classes, *,
                from_logits=True,
                obd_weight=None,
                class_weight=None):
    # check OBD weights
    obd_weight = check_obd_weight(obd_weight)
    for d_, w_ in obd_weight.items():
        obd_weight[d_] = tf.convert_to_tensor(w_, dtype=np.float32)

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

        # nominal, multi-class cross-entropy
        losses_nom = tf.math.multiply(y_true, -1.0 * tf.math.log(y_proba))
        losses_nom = tf.math.reduce_sum(losses_nom, axis=-1)
        losses_nom = tf.math.multiply(obd_weight['Nom'], losses_nom)
        losses = losses_nom

        # ordinal binary decompositions
        for c in range(n_classes - 1):
            # OvN
            y_true_neg_ovn = y_true[..., c]
            y_true_pos_ovn = y_true[..., c + 1]
            y_proba_neg_ovn = y_proba[..., c]
            y_proba_pos_ovn = y_proba[..., c + 1]
            y_proba_tot_ovn = y_proba_neg_ovn + y_proba_pos_ovn
            y_proba_neg_ovn = tf.math.divide_no_nan(y_proba_neg_ovn, y_proba_tot_ovn)
            y_proba_pos_ovn = tf.math.divide_no_nan(y_proba_pos_ovn, y_proba_tot_ovn)
            y_proba_neg_ovn = tf.clip_by_value(y_proba_neg_ovn,
                                               clip_value_min=tf.keras.backend.epsilon(),
                                               clip_value_max=1.0 - tf.keras.backend.epsilon())
            y_proba_pos_ovn = tf.clip_by_value(y_proba_pos_ovn,
                                               clip_value_min=tf.keras.backend.epsilon(),
                                               clip_value_max=1.0 - tf.keras.backend.epsilon())
            losses_bce_neg_ovn = tf.math.multiply(y_true_neg_ovn, -1.0 * tf.math.log(y_proba_neg_ovn))
            losses_bce_pos_ovn = tf.math.multiply(y_true_pos_ovn, -1.0 * tf.math.log(y_proba_pos_ovn))
            losses_bce_tot_ovn = losses_bce_neg_ovn + losses_bce_pos_ovn
            losses_bce_tot_ovn = tf.math.multiply(obd_weight['OvN'][c], losses_bce_tot_ovn)
            losses = losses + losses_bce_tot_ovn

            # OvS
            y_true_neg_ovs = y_true[..., c]
            y_true_pos_ovs = tf.math.reduce_sum(y_true[..., (c + 1):], axis=-1)
            y_proba_neg_ovs = y_proba[..., c]
            y_proba_pos_ovs = tf.math.reduce_sum(y_proba[..., (c + 1):], axis=-1)
            y_proba_tot_ovs = y_proba_neg_ovs + y_proba_pos_ovs
            y_proba_neg_ovs = tf.math.divide_no_nan(y_proba_neg_ovs, y_proba_tot_ovs)
            y_proba_pos_ovs = tf.math.divide_no_nan(y_proba_pos_ovs, y_proba_tot_ovs)
            y_proba_neg_ovs = tf.clip_by_value(y_proba_neg_ovs,
                                               clip_value_min=tf.keras.backend.epsilon(),
                                               clip_value_max=1.0 - tf.keras.backend.epsilon())
            y_proba_pos_ovs = tf.clip_by_value(y_proba_pos_ovs,
                                               clip_value_min=tf.keras.backend.epsilon(),
                                               clip_value_max=1.0 - tf.keras.backend.epsilon())
            losses_bce_neg_ovs = tf.math.multiply(y_true_neg_ovs, -1.0 * tf.math.log(y_proba_neg_ovs))
            losses_bce_pos_ovs = tf.math.multiply(y_true_pos_ovs, -1.0 * tf.math.log(y_proba_pos_ovs))
            losses_bce_tot_ovs = losses_bce_neg_ovs + losses_bce_pos_ovs
            losses_bce_tot_ovs = tf.math.multiply(obd_weight['OvS'][c], losses_bce_tot_ovs)
            losses = losses + losses_bce_tot_ovs

            # OvP
            y_true_neg_ovp = tf.math.reduce_sum(y_true[..., :(c + 1)], axis=-1)
            y_true_pos_ovp = y_true[..., c + 1]
            y_proba_neg_ovp = tf.math.reduce_sum(y_proba[..., :(c + 1)], axis=-1)
            y_proba_pos_ovp = y_proba[..., c + 1]
            y_proba_tot_ovp = y_proba_neg_ovp + y_proba_pos_ovp
            y_proba_neg_ovp = tf.math.divide_no_nan(y_proba_neg_ovp, y_proba_tot_ovp)
            y_proba_pos_ovp = tf.math.divide_no_nan(y_proba_pos_ovp, y_proba_tot_ovp)
            y_proba_neg_ovp = tf.clip_by_value(y_proba_neg_ovp,
                                               clip_value_min=tf.keras.backend.epsilon(),
                                               clip_value_max=1.0 - tf.keras.backend.epsilon())
            y_proba_pos_ovp = tf.clip_by_value(y_proba_pos_ovp,
                                               clip_value_min=tf.keras.backend.epsilon(),
                                               clip_value_max=1.0 - tf.keras.backend.epsilon())
            losses_bce_neg_ovp = tf.math.multiply(y_true_neg_ovp, -1.0 * tf.math.log(y_proba_neg_ovp))
            losses_bce_pos_ovp = tf.math.multiply(y_true_pos_ovp, -1.0 * tf.math.log(y_proba_pos_ovp))
            losses_bce_tot_ovp = losses_bce_neg_ovp + losses_bce_pos_ovp
            losses_bce_tot_ovp = tf.math.multiply(obd_weight['OvP'][c], losses_bce_tot_ovp)
            losses = losses + losses_bce_tot_ovp

            # OrdP
            y_true_neg_ordp = tf.math.reduce_sum(y_true[..., :(c + 1)], axis=-1)
            y_true_pos_ordp = tf.math.reduce_sum(y_true[..., (c + 1):], axis=-1)
            y_proba_neg_ordp = tf.math.reduce_sum(y_proba[..., :(c + 1)], axis=-1)
            y_proba_pos_ordp = tf.math.reduce_sum(y_proba[..., (c + 1):], axis=-1)
            y_proba_tot_ordp = y_proba_neg_ordp + y_proba_pos_ordp
            y_proba_neg_ordp = tf.math.divide_no_nan(y_proba_neg_ordp, y_proba_tot_ordp)
            y_proba_pos_ordp = tf.math.divide_no_nan(y_proba_pos_ordp, y_proba_tot_ordp)
            y_proba_neg_ordp = tf.clip_by_value(y_proba_neg_ordp,
                                                clip_value_min=tf.keras.backend.epsilon(),
                                                clip_value_max=1.0 - tf.keras.backend.epsilon())
            y_proba_pos_ordp = tf.clip_by_value(y_proba_pos_ordp,
                                                clip_value_min=tf.keras.backend.epsilon(),
                                                clip_value_max=1.0 - tf.keras.backend.epsilon())
            losses_bce_neg_ordp = tf.math.multiply(y_true_neg_ordp, -1.0 * tf.math.log(y_proba_neg_ordp))
            losses_bce_pos_ordp = tf.math.multiply(y_true_pos_ordp, -1.0 * tf.math.log(y_proba_pos_ordp))
            losses_bce_tot_ordp = losses_bce_neg_ordp + losses_bce_pos_ordp
            losses_bce_tot_ordp = tf.math.multiply(obd_weight['OrdP'][c], losses_bce_tot_ordp)
            losses = losses + losses_bce_tot_ordp

        # apply class weights
        sample_weight = tf.linalg.matvec(y_true, class_weight)
        losses = tf.math.multiply(sample_weight, losses)

        # reduce to mean
        loss = tf.math.reduce_mean(losses, axis=None)
        return loss

    return loss_fn


@validate_params(
    parameter_constraints={'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')],
                           'from_logits': ['boolean'],
                           'gamma': [Interval(Real, 0.0, None, closed='left')],
                           'class_weight': ['array-like', None]},
    prefer_skip_nested_validation=True
)
def obd_focal_loss(n_classes, *,
                   from_logits=True,
                   gamma=2.0,
                   obd_weight=None,
                   class_weight=None):
    # check OBD weights
    obd_weight = check_obd_weight(obd_weight)
    for d_, w_ in obd_weight.items():
        obd_weight[d_] = tf.convert_to_tensor(w_, dtype=np.float32)

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

        # nominal, multi-class focal
        losses_pow_nom = tf.math.pow(1.0 - y_proba, gamma)
        losses_log_nom = -1.0 * tf.math.log(y_proba)
        losses_nom = tf.math.multiply(losses_pow_nom, losses_log_nom)
        losses_nom = tf.math.multiply(y_true, losses_nom)
        losses_nom = tf.math.reduce_sum(losses_nom, axis=-1)
        losses_nom = tf.math.multiply(obd_weight['Nom'], losses_nom)
        losses = losses_nom

        # ordinal binary decompositions
        for c in range(n_classes - 1):
            # OvN
            y_true_neg_ovn = y_true[..., c]
            y_true_pos_ovn = y_true[..., c + 1]
            y_proba_neg_ovn = y_proba[..., c]
            y_proba_pos_ovn = y_proba[..., c + 1]
            y_proba_tot_ovn = y_proba_neg_ovn + y_proba_pos_ovn
            y_proba_neg_ovn = tf.math.divide_no_nan(y_proba_neg_ovn, y_proba_tot_ovn)
            y_proba_pos_ovn = tf.math.divide_no_nan(y_proba_pos_ovn, y_proba_tot_ovn)
            y_proba_neg_ovn = tf.clip_by_value(y_proba_neg_ovn,
                                               clip_value_min=tf.keras.backend.epsilon(),
                                               clip_value_max=1.0 - tf.keras.backend.epsilon())
            y_proba_pos_ovn = tf.clip_by_value(y_proba_pos_ovn,
                                               clip_value_min=tf.keras.backend.epsilon(),
                                               clip_value_max=1.0 - tf.keras.backend.epsilon())
            losses_pow_neg_ovn = tf.math.pow(1.0 - y_proba_neg_ovn, gamma)
            losses_pow_pos_ovn = tf.math.pow(1.0 - y_proba_pos_ovn, gamma)
            losses_log_neg_ovn = -1.0 * tf.math.log(y_proba_neg_ovn)
            losses_log_pos_ovn = -1.0 * tf.math.log(y_proba_pos_ovn)
            losses_bfoc_neg_ovn = tf.math.multiply(losses_pow_neg_ovn, losses_log_neg_ovn)
            losses_bfoc_pos_ovn = tf.math.multiply(losses_pow_pos_ovn, losses_log_pos_ovn)
            losses_bfoc_neg_ovn = tf.math.multiply(y_true_neg_ovn, losses_bfoc_neg_ovn)
            losses_bfoc_pos_ovn = tf.math.multiply(y_true_pos_ovn, losses_bfoc_pos_ovn)
            losses_bfoc_tot_ovn = losses_bfoc_neg_ovn + losses_bfoc_pos_ovn
            losses_bfoc_tot_ovn = tf.math.multiply(obd_weight['OvN'][c], losses_bfoc_tot_ovn)
            losses = losses + losses_bfoc_tot_ovn

            # OvS
            y_true_neg_ovs = y_true[..., c]
            y_true_pos_ovs = tf.math.reduce_sum(y_true[..., (c + 1):], axis=-1)
            y_proba_neg_ovs = y_proba[..., c]
            y_proba_pos_ovs = tf.math.reduce_sum(y_proba[..., (c + 1):], axis=-1)
            y_proba_tot_ovs = y_proba_neg_ovs + y_proba_pos_ovs
            y_proba_neg_ovs = tf.math.divide_no_nan(y_proba_neg_ovs, y_proba_tot_ovs)
            y_proba_pos_ovs = tf.math.divide_no_nan(y_proba_pos_ovs, y_proba_tot_ovs)
            y_proba_neg_ovs = tf.clip_by_value(y_proba_neg_ovs,
                                               clip_value_min=tf.keras.backend.epsilon(),
                                               clip_value_max=1.0 - tf.keras.backend.epsilon())
            y_proba_pos_ovs = tf.clip_by_value(y_proba_pos_ovs,
                                               clip_value_min=tf.keras.backend.epsilon(),
                                               clip_value_max=1.0 - tf.keras.backend.epsilon())
            losses_pow_neg_ovs = tf.math.pow(1.0 - y_proba_neg_ovs, gamma)
            losses_pow_pos_ovs = tf.math.pow(1.0 - y_proba_pos_ovs, gamma)
            losses_log_neg_ovs = -1.0 * tf.math.log(y_proba_neg_ovs)
            losses_log_pos_ovs = -1.0 * tf.math.log(y_proba_pos_ovs)
            losses_bfoc_neg_ovs = tf.math.multiply(losses_pow_neg_ovs, losses_log_neg_ovs)
            losses_bfoc_pos_ovs = tf.math.multiply(losses_pow_pos_ovs, losses_log_pos_ovs)
            losses_bfoc_neg_ovs = tf.math.multiply(y_true_neg_ovs, losses_bfoc_neg_ovs)
            losses_bfoc_pos_ovs = tf.math.multiply(y_true_pos_ovs, losses_bfoc_pos_ovs)
            losses_bfoc_tot_ovs = losses_bfoc_neg_ovs + losses_bfoc_pos_ovs
            losses_bfoc_tot_ovs = tf.math.multiply(obd_weight['OvS'][c], losses_bfoc_tot_ovs)
            losses = losses + losses_bfoc_tot_ovs

            # OvP
            y_true_neg_ovp = tf.math.reduce_sum(y_true[..., :(c + 1)], axis=-1)
            y_true_pos_ovp = y_true[..., c + 1]
            y_proba_neg_ovp = tf.math.reduce_sum(y_proba[..., :(c + 1)], axis=-1)
            y_proba_pos_ovp = y_proba[..., c + 1]
            y_proba_tot_ovp = y_proba_neg_ovp + y_proba_pos_ovp
            y_proba_neg_ovp = tf.math.divide_no_nan(y_proba_neg_ovp, y_proba_tot_ovp)
            y_proba_pos_ovp = tf.math.divide_no_nan(y_proba_pos_ovp, y_proba_tot_ovp)
            y_proba_neg_ovp = tf.clip_by_value(y_proba_neg_ovp,
                                               clip_value_min=tf.keras.backend.epsilon(),
                                               clip_value_max=1.0 - tf.keras.backend.epsilon())
            y_proba_pos_ovp = tf.clip_by_value(y_proba_pos_ovp,
                                               clip_value_min=tf.keras.backend.epsilon(),
                                               clip_value_max=1.0 - tf.keras.backend.epsilon())
            losses_pow_neg_ovp = tf.math.pow(1.0 - y_proba_neg_ovp, gamma)
            losses_pow_pos_ovp = tf.math.pow(1.0 - y_proba_pos_ovp, gamma)
            losses_log_neg_ovp = -1.0 * tf.math.log(y_proba_neg_ovp)
            losses_log_pos_ovp = -1.0 * tf.math.log(y_proba_pos_ovp)
            losses_bfoc_neg_ovp = tf.math.multiply(losses_pow_neg_ovp, losses_log_neg_ovp)
            losses_bfoc_pos_ovp = tf.math.multiply(losses_pow_pos_ovp, losses_log_pos_ovp)
            losses_bfoc_neg_ovp = tf.math.multiply(y_true_neg_ovp, losses_bfoc_neg_ovp)
            losses_bfoc_pos_ovp = tf.math.multiply(y_true_pos_ovp, losses_bfoc_pos_ovp)
            losses_bfoc_tot_ovp = losses_bfoc_neg_ovp + losses_bfoc_pos_ovp
            losses_bfoc_tot_ovp = tf.math.multiply(obd_weight['OvP'][c], losses_bfoc_tot_ovp)
            losses = losses + losses_bfoc_tot_ovp

            # OrdP
            y_true_neg_ordp = tf.math.reduce_sum(y_true[..., :(c + 1)], axis=-1)
            y_true_pos_ordp = tf.math.reduce_sum(y_true[..., (c + 1):], axis=-1)
            y_proba_neg_ordp = tf.math.reduce_sum(y_proba[..., :(c + 1)], axis=-1)
            y_proba_pos_ordp = tf.math.reduce_sum(y_proba[..., (c + 1):], axis=-1)
            y_proba_tot_ordp = y_proba_neg_ordp + y_proba_pos_ordp
            y_proba_neg_ordp = tf.math.divide_no_nan(y_proba_neg_ordp, y_proba_tot_ordp)
            y_proba_pos_ordp = tf.math.divide_no_nan(y_proba_pos_ordp, y_proba_tot_ordp)
            y_proba_neg_ordp = tf.clip_by_value(y_proba_neg_ordp,
                                                clip_value_min=tf.keras.backend.epsilon(),
                                                clip_value_max=1.0 - tf.keras.backend.epsilon())
            y_proba_pos_ordp = tf.clip_by_value(y_proba_pos_ordp,
                                                clip_value_min=tf.keras.backend.epsilon(),
                                                clip_value_max=1.0 - tf.keras.backend.epsilon())
            losses_pow_neg_ordp = tf.math.pow(1.0 - y_proba_neg_ordp, gamma)
            losses_pow_pos_ordp = tf.math.pow(1.0 - y_proba_pos_ordp, gamma)
            losses_log_neg_ordp = -1.0 * tf.math.log(y_proba_neg_ordp)
            losses_log_pos_ordp = -1.0 * tf.math.log(y_proba_pos_ordp)
            losses_bfoc_neg_ordp = tf.math.multiply(losses_pow_neg_ordp, losses_log_neg_ordp)
            losses_bfoc_pos_ordp = tf.math.multiply(losses_pow_pos_ordp, losses_log_pos_ordp)
            losses_bfoc_neg_ordp = tf.math.multiply(y_true_neg_ordp, losses_bfoc_neg_ordp)
            losses_bfoc_pos_ordp = tf.math.multiply(y_true_pos_ordp, losses_bfoc_pos_ordp)
            losses_bfoc_tot_ordp = losses_bfoc_neg_ordp + losses_bfoc_pos_ordp
            losses_bfoc_tot_ordp = tf.math.multiply(obd_weight['OrdP'][c], losses_bfoc_tot_ordp)
            losses = losses + losses_bfoc_tot_ordp

        # apply class weights
        sample_weight = tf.linalg.matvec(y_true, class_weight)
        losses = tf.math.multiply(sample_weight, losses)

        # reduce to mean
        loss = tf.math.reduce_mean(losses, axis=None)
        return loss

    return loss_fn
