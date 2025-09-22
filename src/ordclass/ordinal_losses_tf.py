# Authors:  Fernando García-García <fegarcia@bcamath.org>

from numbers import Integral, Real

import numpy as np
from scipy.stats import beta
import scipy.special as scp

from sklearn.utils import check_array
from sklearn.utils._param_validation import validate_params, Interval, StrOptions

import tensorflow as tf


_MIN_CLASSES = 3
_MAX_CLASSES = 2 ** 8


@validate_params({'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')],
                  'from_logits': ['boolean'],
                  'regul_type': [StrOptions({'beta'}), None],
                  'regul_eta': [Interval(Real, 0.0, 1.0, closed='both'), None],
                  'regul_delta': [Interval(Real, 0.0, None, closed='neither'), None],
                  'class_weight': ['array-like', None]},
                 prefer_skip_nested_validation=True)
def or_cross_entropy_loss(n_classes, from_logits, regul_type, regul_eta, regul_delta, class_weight=None):
    # obtain regularization terms
    regul_terms, regul_eta = _get_unimodal_regul(n_classes, regul_type, regul_eta, regul_delta)
    regul_terms = tf.convert_to_tensor(regul_terms, dtype=np.float32)

    # obtain class weights
    if class_weight is None:
        class_weight = np.ones(shape=(n_classes,), dtype=np.float32)
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

        # regularized ground truth labels
        y_regul = (1.0 - regul_eta) * y_true + regul_eta * tf.linalg.matmul(y_true, tf.transpose(regul_terms))

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


@validate_params({'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')],
                  'from_logits': ['boolean'],
                  'regul_type': [StrOptions({'beta'}), None],
                  'regul_eta': [Interval(Real, 0.0, 1.0, closed='both'), None],
                  'regul_delta': [Interval(Real, 0.0, None, closed='neither'), None],
                  'focal_gamma': [Interval(Real, 0.0, None, closed='left')],
                  'class_weight': ['array-like', None]},
                 prefer_skip_nested_validation=True)
def or_focal_loss(n_classes, from_logits, regul_type, regul_eta, regul_delta, focal_gamma=2.0, class_weight=None):
    # obtain regularization terms
    regul_terms, regul_eta = _get_unimodal_regul(n_classes, regul_type, regul_eta, regul_delta)
    regul_terms = tf.convert_to_tensor(regul_terms, dtype=np.float32)

    # obtain class weights
    if class_weight is None:
        class_weight = np.ones(shape=(n_classes,), dtype=np.float32)
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

        # regularized ground truth labels
        y_regul = (1.0 - regul_eta) * y_true + regul_eta * tf.linalg.matmul(y_true, tf.transpose(regul_terms))

        # focal loss
        losses_log = -1.0 * tf.math.log(y_proba)
        losses_pow = tf.math.pow(1.0 - y_proba, focal_gamma)
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


@validate_params({'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')],
                  'regul_type': [StrOptions({'beta'}), None],
                  'regul_eta': [Interval(Real, 0.0, 1.0, closed='both'), None],
                  'regul_delta': [Interval(Real, 0.0, None, closed='neither'), None]},
                 prefer_skip_nested_validation=True)
def _get_unimodal_regul(n_classes, regul_type, regul_eta, regul_delta):
    if regul_type is None:  # no regularization
        regul_terms = np.eye(n_classes, dtype=np.float32)
    elif regul_type == 'beta':
        regul_terms = _get_unimodal_regul_beta(n_classes, delta=regul_delta)
    else:
        raise ValueError

    if regul_eta is None:  # no regularization
        regul_eta = 0.0

    return regul_terms, regul_eta


_DEFAULT_DELTA = 1.0


@validate_params({'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')],
                  'delta': [Interval(Real, 0.0, None, closed='neither'), None]},
                 prefer_skip_nested_validation=True)
def _get_unimodal_regul_beta(n_classes, delta=_DEFAULT_DELTA):
    if delta is None:
        delta = _DEFAULT_DELTA

    reg_terms = np.zeros(shape=(n_classes, n_classes), dtype=np.float32)

    for c_true in range(n_classes):
        a_beta = ((2 * n_classes - 2 * c_true - 1) * ((2 * c_true + 1) ** 2) * (delta ** 2) - 2 * c_true - 1) / (2 * n_classes)
        b_beta = (2 * n_classes - 2 * c_true - 1) / (2 * c_true + 1) * a_beta

        beta_cdf = lambda x: beta.cdf(x, a=a_beta, b=b_beta)
        for c in range(n_classes):
            x_lo = c / n_classes
            x_hi = (c + 1) / n_classes
            reg = beta_cdf(x_hi) - beta_cdf(x_lo)
            reg_terms[c, c_true] = reg

    return reg_terms


@validate_params({'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')],
                  'from_logits': ['boolean'],
                  'weights': [StrOptions({'nominal', 'linear', 'quadratic'}), 'array-like'],
                  'class_priors': ['array-like'],
                  'class_weight': [StrOptions({'balanced'}), 'array-like', None]},
                 prefer_skip_nested_validation=True)
def owk_loss(n_classes, from_logits, weights, class_priors, class_weight=None):
    # obtain kappa weights matrix
    w = _ordinal_kappa_weights(weights=weights, n_classes=n_classes)

    # obtain class weights
    if isinstance(class_weight, str) and (class_weight == 'balanced'):
        class_weight = 1.0 / (class_priors * n_classes)
    elif class_weight is None:
        class_weight = np.ones(shape=(n_classes,), dtype=np.float32)
    elif not isinstance(class_weight, np.ndarray) or np.any(class_weight < 0.0):
        raise ValueError

    w = tf.convert_to_tensor(w, dtype=np.float32)
    class_priors = tf.convert_to_tensor(class_priors, dtype=np.float32)
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

        # weighted kappa calculations
        w_true = tf.linalg.matmul(y_true, w)
        kappa_num = tf.math.multiply(w_true, y_proba)
        kappa_num = tf.math.reduce_sum(kappa_num, axis=-1)

        kappa_den = tf.zeros_like(kappa_num)
        for c in range(n_classes):
            w_temp = w[c, :]
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


# auxiliary for ordinal weighted kappa
@validate_params({'weights': [StrOptions({'nominal', 'linear', 'quadratic'}), 'array-like'],
                  'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')]},
                 prefer_skip_nested_validation=True)
def _ordinal_kappa_weights(weights, n_classes):
    # weights matrix
    w = np.zeros(shape=(n_classes, n_classes), dtype=np.float32)
    if weights == 'nominal':
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                w[i, j] = 1.0
                w[j, i] = w[i, j]
    elif weights == 'linear':
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                w[i, j] = (j - i) / (n_classes - 1)
                w[j, i] = w[i, j]
    elif weights == 'quadratic':
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                w[i, j] = (j - i) ** 2 / (n_classes - 1) ** 2
                w[j, i] = w[i, j]
    else:
        # check that the weights matrix makes sense
        w = check_array(weights, ensure_2d=True, force_all_finite=True)
        if w.shape != (n_classes, n_classes):  # square matrix and with correct dimensions
            raise ValueError
        if np.any(w < 0.0, axis=None):  # all non-negative weights
            raise ValueError
        if np.any(np.diag(w) != 0.0):  # all zero-valued weights in the main diagonal
            raise ValueError
        for i in range(n_classes):  # all positive weights outside the main diagonal
            for j in range(i + 1, n_classes):
                if (w[i, j] <= 0.0) or (w[j, i] <= 0.0):
                    raise ValueError

    return w


DECOMP_NOM = 'Nom'
DECOMP_ORD = ['OvN', 'OvS', 'OvP', 'OrdP']
DECOMP_ALL = [DECOMP_NOM] + DECOMP_ORD


@validate_params({'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')],
                  'from_logits': ['boolean'],
                  'ordin_decomp_weight': [StrOptions({'balanced'}), dict],
                  'class_counts': ['array-like'],
                  'class_weight': [StrOptions({'balanced'}), 'array-like', None]},
                 prefer_skip_nested_validation=True)
def obd_cross_entropy_loss(n_classes, from_logits, ordin_decomp_weight, class_counts, class_weight=None):
    # verify decomposition weights
    ordin_decomp_weight = _verify_decomposition_weights(ordin_decomp_weight)

    # obtain class weights
    if isinstance(class_weight, str) and (class_weight == 'balanced'):
        if np.all(class_counts == 0):
            # if class counts are not provided, set equal priors
            class_priors = np.ones(shape=(n_classes,), dtype=np.float32) / n_classes
        else:
            class_priors = class_counts / np.sum(class_counts)
        class_weight = 1.0 / (class_priors * n_classes)
    elif class_weight is None:
        class_weight = np.ones(shape=(n_classes,), dtype=np.float32)
    elif not isinstance(class_weight, np.ndarray) or np.any(class_weight < 0.0):
        raise ValueError

    # compute overall contributions to loss from decomposition weights and differential entropies
    diff_entropy_decomp = _get_diff_entropy_from_class_counts(class_counts)
    ordin_contrib_weight = dict()
    for dec_ in DECOMP_ALL:
        ordin_contrib_weight[dec_] = ordin_decomp_weight[dec_] * diff_entropy_decomp[dec_]

    # convert types for TensorFlow
    class_weight = tf.convert_to_tensor(class_weight, dtype=np.float32)
    for dec_, contrib_ in ordin_contrib_weight.items():
        ordin_contrib_weight[dec_] = tf.convert_to_tensor(contrib_, dtype=np.float32)

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
        losses_nom = tf.math.multiply(ordin_contrib_weight['Nom'], losses_nom)
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
            losses_bce_tot_ovn = tf.math.multiply(ordin_contrib_weight['OvN'][c], losses_bce_tot_ovn)
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
            losses_bce_tot_ovs = tf.math.multiply(ordin_contrib_weight['OvS'][c], losses_bce_tot_ovs)
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
            losses_bce_tot_ovp = tf.math.multiply(ordin_contrib_weight['OvP'][c], losses_bce_tot_ovp)
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
            losses_bce_tot_ordp = tf.math.multiply(ordin_contrib_weight['OrdP'][c], losses_bce_tot_ordp)
            losses = losses + losses_bce_tot_ordp

        # apply class weights
        sample_weight = tf.linalg.matvec(y_true, class_weight)
        losses = tf.math.multiply(sample_weight, losses)

        # reduce to mean
        loss = tf.math.reduce_mean(losses, axis=None)
        return loss

    return loss_fn


@validate_params({'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')],
                  'from_logits': ['boolean'],
                  'ordin_decomp_weight': [StrOptions({'balanced'}), dict],
                  'class_counts': ['array-like'],
                  'focal_gamma': [Interval(Real, 0.0, None, closed='left')],
                  'class_weight': [StrOptions({'balanced'}), 'array-like', None]},
                 prefer_skip_nested_validation=True)
def obd_focal_loss(n_classes, from_logits, ordin_decomp_weight, class_counts, focal_gamma=2.0, class_weight=None):
    # verify decomposition weights
    ordin_decomp_weight = _verify_decomposition_weights(ordin_decomp_weight)

    # obtain class weights
    if isinstance(class_weight, str) and (class_weight == 'balanced'):
        if np.all(class_counts == 0):
            # if class counts are not provided, set equal priors
            class_priors = np.ones(shape=(n_classes,), dtype=np.float32) / n_classes
        else:
            class_priors = class_counts / np.sum(class_counts)
        class_weight = 1.0 / (class_priors * n_classes)
    elif class_weight is None:
        class_weight = np.ones(shape=(n_classes,), dtype=np.float32)
    elif not isinstance(class_weight, np.ndarray) or np.any(class_weight < 0.0):
        raise ValueError

    # compute overall contributions to loss from decomposition weights and differential entropies
    diff_entropy_decomp = _get_diff_entropy_from_class_counts(class_counts)
    ordin_contrib_weight = dict()
    for dec_ in DECOMP_ALL:
        ordin_contrib_weight[dec_] = ordin_decomp_weight[dec_] * diff_entropy_decomp[dec_]

    # convert types for TensorFlow
    class_weight = tf.convert_to_tensor(class_weight, dtype=np.float32)
    for dec_, contrib_ in ordin_contrib_weight.items():
        ordin_contrib_weight[dec_] = tf.convert_to_tensor(contrib_, dtype=np.float32)

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
        losses_pow_nom = tf.math.pow(1.0 - y_proba, focal_gamma)
        losses_log_nom = -1.0 * tf.math.log(y_proba)
        losses_nom = tf.math.multiply(losses_pow_nom, losses_log_nom)
        losses_nom = tf.math.multiply(y_true, losses_nom)
        losses_nom = tf.math.reduce_sum(losses_nom, axis=-1)
        losses_nom = tf.math.multiply(ordin_contrib_weight['Nom'], losses_nom)
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
            losses_pow_neg_ovn = tf.math.pow(1.0 - y_proba_neg_ovn, focal_gamma)
            losses_pow_pos_ovn = tf.math.pow(1.0 - y_proba_pos_ovn, focal_gamma)
            losses_log_neg_ovn = -1.0 * tf.math.log(y_proba_neg_ovn)
            losses_log_pos_ovn = -1.0 * tf.math.log(y_proba_pos_ovn)
            losses_bce_neg_ovn = tf.math.multiply(losses_pow_neg_ovn, losses_log_neg_ovn)
            losses_bce_pos_ovn = tf.math.multiply(losses_pow_pos_ovn, losses_log_pos_ovn)
            losses_bce_neg_ovn = tf.math.multiply(y_true_neg_ovn, losses_bce_neg_ovn)
            losses_bce_pos_ovn = tf.math.multiply(y_true_pos_ovn, losses_bce_pos_ovn)
            losses_bce_tot_ovn = losses_bce_neg_ovn + losses_bce_pos_ovn
            losses_bce_tot_ovn = tf.math.multiply(ordin_contrib_weight['OvN'][c], losses_bce_tot_ovn)
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
            losses_pow_neg_ovs = tf.math.pow(1.0 - y_proba_neg_ovs, focal_gamma)
            losses_pow_pos_ovs = tf.math.pow(1.0 - y_proba_pos_ovs, focal_gamma)
            losses_log_neg_ovs = -1.0 * tf.math.log(y_proba_neg_ovs)
            losses_log_pos_ovs = -1.0 * tf.math.log(y_proba_pos_ovs)
            losses_bce_neg_ovs = tf.math.multiply(losses_pow_neg_ovs, losses_log_neg_ovs)
            losses_bce_pos_ovs = tf.math.multiply(losses_pow_pos_ovs, losses_log_pos_ovs)
            losses_bce_neg_ovs = tf.math.multiply(y_true_neg_ovs, losses_bce_neg_ovs)
            losses_bce_pos_ovs = tf.math.multiply(y_true_pos_ovs, losses_bce_pos_ovs)
            losses_bce_tot_ovs = losses_bce_neg_ovs + losses_bce_pos_ovs
            losses_bce_tot_ovs = tf.math.multiply(ordin_contrib_weight['OvS'][c], losses_bce_tot_ovs)
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
            losses_pow_neg_ovp = tf.math.pow(1.0 - y_proba_neg_ovp, focal_gamma)
            losses_pow_pos_ovp = tf.math.pow(1.0 - y_proba_pos_ovp, focal_gamma)
            losses_log_neg_ovp = -1.0 * tf.math.log(y_proba_neg_ovp)
            losses_log_pos_ovp = -1.0 * tf.math.log(y_proba_pos_ovp)
            losses_bce_neg_ovp = tf.math.multiply(losses_pow_neg_ovp, losses_log_neg_ovp)
            losses_bce_pos_ovp = tf.math.multiply(losses_pow_pos_ovp, losses_log_pos_ovp)
            losses_bce_neg_ovp = tf.math.multiply(y_true_neg_ovp, losses_bce_neg_ovp)
            losses_bce_pos_ovp = tf.math.multiply(y_true_pos_ovp, losses_bce_pos_ovp)
            losses_bce_tot_ovp = losses_bce_neg_ovp + losses_bce_pos_ovp
            losses_bce_tot_ovp = tf.math.multiply(ordin_contrib_weight['OvP'][c], losses_bce_tot_ovp)
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
            losses_pow_neg_ordp = tf.math.pow(1.0 - y_proba_neg_ordp, focal_gamma)
            losses_pow_pos_ordp = tf.math.pow(1.0 - y_proba_pos_ordp, focal_gamma)
            losses_log_neg_ordp = -1.0 * tf.math.log(y_proba_neg_ordp)
            losses_log_pos_ordp = -1.0 * tf.math.log(y_proba_pos_ordp)
            losses_bce_neg_ordp = tf.math.multiply(losses_pow_neg_ordp, losses_log_neg_ordp)
            losses_bce_pos_ordp = tf.math.multiply(losses_pow_pos_ordp, losses_log_pos_ordp)
            losses_bce_neg_ordp = tf.math.multiply(y_true_neg_ordp, losses_bce_neg_ordp)
            losses_bce_pos_ordp = tf.math.multiply(y_true_pos_ordp, losses_bce_pos_ordp)
            losses_bce_tot_ordp = losses_bce_neg_ordp + losses_bce_pos_ordp
            losses_bce_tot_ordp = tf.math.multiply(ordin_contrib_weight['OrdP'][c], losses_bce_tot_ordp)
            losses = losses + losses_bce_tot_ordp

        # apply class weights
        sample_weight = tf.linalg.matvec(y_true, class_weight)
        losses = tf.math.multiply(sample_weight, losses)

        # reduce to mean
        loss = tf.math.reduce_mean(losses, axis=None)
        return loss

    return loss_fn


def _verify_decomposition_weights(ordin_decomp_weight):
    _DEFAULT_WEIGHT = 1.0
    _MINIMUM_WEIGHT = 0.0

    # verify the weights of the different decomposition strategies towards the overall loss
    if isinstance(ordin_decomp_weight, str) and (ordin_decomp_weight == 'balanced'):
        ordin_decomp_weight = dict()

        n_decomp_ord = len(DECOMP_ORD)
        for dec_ in DECOMP_ALL:
            if dec_ == DECOMP_NOM:
                ordin_decomp_weight[dec_] = _DEFAULT_WEIGHT
            else:
                ordin_decomp_weight[dec_] = _DEFAULT_WEIGHT / n_decomp_ord

    elif isinstance(ordin_decomp_weight, dict):
        if not set(ordin_decomp_weight.keys()).issubset(set(DECOMP_ALL)):
            raise ValueError
        for dec_ in DECOMP_ALL:
            if dec_ in ordin_decomp_weight:
                if ordin_decomp_weight[dec_] < _MINIMUM_WEIGHT:
                    raise ValueError
            else:
                ordin_decomp_weight[dec_] = _DEFAULT_WEIGHT

    else:
        raise ValueError

    return ordin_decomp_weight


def _get_diff_entropy_from_class_counts(class_counts):

    def h_dirichlet_fn(alpha):
        alpha_dim = alpha.size
        alpha_tot = np.sum(alpha)

        h = 0.0
        for d in range(alpha_dim):
            h += scp.gammaln(alpha[d])
        h -= scp.gammaln(alpha_tot)

        h += (alpha_tot - alpha_dim) * scp.digamma(alpha_tot)
        for d in range(alpha_dim):
            h -= (alpha[d] - 1) * scp.digamma(alpha[d])

        return h

    # compute differential entropies from class counts
    n_classes = class_counts.size
    diff_entropy_decomp = {dec_: np.zeros(shape=(n_classes - 1,), dtype=np.float32) for dec_ in DECOMP_ORD}

    h_max_nom = h_dirichlet_fn(alpha=np.ones(shape=(n_classes,)))
    h_obs_nom = h_dirichlet_fn(alpha=class_counts + 1)
    diff_entropy_decomp[DECOMP_NOM] = h_max_nom - h_obs_nom

    h_max_bin = 0.0
    for c in range(n_classes - 1):
        # OvN
        count_neg_ovn = class_counts[c]
        count_pos_ovn = class_counts[c + 1]
        counts_ovn = np.asarray([count_neg_ovn, count_pos_ovn])
        h_obs_ovn = h_dirichlet_fn(alpha=counts_ovn + 1)
        diff_entropy_decomp['OvN'][c] = (h_max_bin - h_obs_ovn) / (n_classes - 1)

        # OvS
        count_neg_ovs = class_counts[c]
        count_pos_ovs = np.sum(class_counts[(c + 1):])
        counts_ovs = np.asarray([count_neg_ovs, count_pos_ovs])
        h_obs_ovs = h_dirichlet_fn(alpha=counts_ovs + 1)
        diff_entropy_decomp['OvS'][c] = (h_max_bin - h_obs_ovs) / (n_classes - 1)

        # OvP
        count_neg_ovp = np.sum(class_counts[:(c + 1)])
        count_pos_ovp = class_counts[c + 1]
        counts_ovp = np.asarray([count_neg_ovp, count_pos_ovp])
        h_obs_ovp = h_dirichlet_fn(alpha=counts_ovp + 1)
        diff_entropy_decomp['OvP'][c] = (h_max_bin - h_obs_ovp) / (n_classes - 1)

        # OrdP
        count_neg_ordp = np.sum(class_counts[:(c + 1)])
        count_pos_ordp = np.sum(class_counts[(c + 1):])
        counts_ordp = np.asarray([count_neg_ordp, count_pos_ordp])
        h_obs_ordp = h_dirichlet_fn(alpha=counts_ordp + 1)
        diff_entropy_decomp['OrdP'][c] = (h_max_bin - h_obs_ordp) / (n_classes - 1)

    return diff_entropy_decomp
