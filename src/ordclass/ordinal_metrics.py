# Authors:  Fernando García-García <fegarcia@bcamath.org>

from numbers import Integral

import numpy as np
from scipy.stats import gmean, hmean
from scipy.stats import spearmanr, kendalltau

import sklearn.metrics as sklm
from sklearn.utils._param_validation import validate_params, Interval, StrOptions


_MIN_CLASSES = 3
_MAX_CLASSES = 2 ** 8


# classification metrics: accuracy
@validate_params({'y_true': ['array-like'],
                  'y_pred': ['array-like'],
                  'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')]},
                 prefer_skip_nested_validation=True)
def accuracy_score(y_true, y_pred, n_classes):
    if np.any(y_true < 0) or np.any(y_true > n_classes - 1) or (np.unique(y_true).size > n_classes):
        raise ValueError
    if np.any(y_pred < 0) or np.any(y_pred > n_classes - 1) or (np.unique(y_pred).size > n_classes):
        raise ValueError

    score = sklm.accuracy_score(y_true, y_pred)
    return score


# classification metrics: mean zero error (MZE)
@validate_params({'y_true': ['array-like'],
                  'y_pred': ['array-like'],
                  'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')]},
                 prefer_skip_nested_validation=True)
def mean_zero_error(y_true, y_pred, n_classes):
    if np.any(y_true < 0) or np.any(y_true > n_classes - 1) or (np.unique(y_true).size > n_classes):
        raise ValueError
    if np.any(y_pred < 0) or np.any(y_pred > n_classes - 1) or (np.unique(y_pred).size > n_classes):
        raise ValueError

    score = 1.0 - sklm.accuracy_score(y_true, y_pred)
    return score


# classification metrics: balanced accuracy, i.e. macro-averaged per-class sensitivity
@validate_params({'y_true': ['array-like'],
                  'y_pred': ['array-like'],
                  'adjusted': ['boolean'],
                  'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')]},
                 prefer_skip_nested_validation=True)
def balanced_accuracy_score(y_true, y_pred, adjusted, n_classes):
    if np.any(y_true < 0) or np.any(y_true > n_classes - 1) or (np.unique(y_true).size > n_classes):
        raise ValueError
    if np.any(y_pred < 0) or np.any(y_pred > n_classes - 1) or (np.unique(y_pred).size > n_classes):
        raise ValueError

    score = sklm.balanced_accuracy_score(y_true, y_pred, adjusted=adjusted)
    return score


# classification metrics: precision, macro-averaged per class
@validate_params({'y_true': ['array-like'],
                  'y_pred': ['array-like'],
                  'macro_average': [StrOptions({'mean', 'geo', 'harm', 'min'})],
                  'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')]},
                 prefer_skip_nested_validation=True)
def avg_precision_score(y_true, y_pred, macro_average, n_classes):
    if np.any(y_true < 0) or np.any(y_true > n_classes - 1) or (np.unique(y_true).size > n_classes):
        raise ValueError
    if np.any(y_pred < 0) or np.any(y_pred > n_classes - 1) or (np.unique(y_pred).size > n_classes):
        raise ValueError

    precis, _, _, _ = sklm.precision_recall_fscore_support(y_true, y_pred, labels=range(n_classes),
                                                           average=None, zero_division=0.0)

    if macro_average == 'mean':
        score = np.mean(precis)
    elif macro_average == 'geo':
        score = gmean(precis)
    elif macro_average == 'harm':
        if np.any(np.isclose(precis, 0.0)):
            score = 0.0
        else:
            score = hmean(precis)
    elif macro_average == 'min':
        score = np.min(precis)
    else:
        raise ValueError
    return score


# classification metrics: sensitivity, macro-averaged per class
@validate_params({'y_true': ['array-like'],
                  'y_pred': ['array-like'],
                  'macro_average': [StrOptions({'mean', 'geo', 'harm', 'min'})],
                  'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')]},
                 prefer_skip_nested_validation=True)
def avg_sensitivity_score(y_true, y_pred, macro_average, n_classes):
    if np.any(y_true < 0) or np.any(y_true > n_classes - 1) or (np.unique(y_true).size > n_classes):
        raise ValueError
    if np.any(y_pred < 0) or np.any(y_pred > n_classes - 1) or (np.unique(y_pred).size > n_classes):
        raise ValueError

    _, recall, _, _ = sklm.precision_recall_fscore_support(y_true, y_pred, labels=range(n_classes),
                                                           average=None, zero_division=0.0)
    sensit = recall  # sensitivity is always identical to recall, by definition!

    if macro_average == 'mean':
        score = np.mean(sensit)
    elif macro_average == 'geo':
        score = gmean(sensit)
    elif macro_average == 'harm':
        if np.any(np.isclose(sensit, 0.0)):
            score = 0.0
        else:
            score = hmean(sensit)
    elif macro_average == 'min':
        score = np.min(sensit)
    else:
        raise ValueError
    return score


# classification metrics: F1 score, macro-averaged per class
@validate_params({'y_true': ['array-like'],
                  'y_pred': ['array-like'],
                  'macro_average': [StrOptions({'mean', 'geo', 'harm', 'min'})],
                  'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')]},
                 prefer_skip_nested_validation=True)
def avg_f1_score(y_true, y_pred, macro_average, n_classes):
    if np.any(y_true < 0) or np.any(y_true > n_classes - 1) or (np.unique(y_true).size > n_classes):
        raise ValueError
    if np.any(y_pred < 0) or np.any(y_pred > n_classes - 1) or (np.unique(y_pred).size > n_classes):
        raise ValueError

    f1 = sklm.f1_score(y_true, y_pred, labels=range(n_classes),
                       average=None, zero_division=0.0)

    if macro_average == 'mean':
        score = np.mean(f1)
    elif macro_average == 'geo':
        score = gmean(f1)
    elif macro_average == 'harm':
        if np.any(np.isclose(f1, 0.0)):
            score = 0.0
        else:
            score = hmean(f1)
    elif macro_average == 'min':
        score = np.min(f1)
    else:
        raise ValueError
    return score


# classification metrics: Matthews correlation coefficient (MCC)
@validate_params({'y_true': ['array-like'],
                  'y_pred': ['array-like'],
                  'normalize': ['boolean'],
                  'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')]},
                 prefer_skip_nested_validation=True)
def matthews_corrcoef_score(y_true, y_pred, normalize, n_classes):
    if np.any(y_true < 0) or np.any(y_true > n_classes - 1) or (np.unique(y_true).size > n_classes):
        raise ValueError
    if np.any(y_pred < 0) or np.any(y_pred > n_classes - 1) or (np.unique(y_pred).size > n_classes):
        raise ValueError

    score = sklm.matthews_corrcoef(y_true, y_pred)
    if normalize:
        score = (1.0 + score) / 2.0
    return score


# classification metrics: ROC AUC score, macro-averaged per class
@validate_params({'y_true': ['array-like'],
                  'y_proba': ['array-like'],
                  'macro_average': [StrOptions({'mean', 'geo', 'harm', 'min'})],
                  'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')]},
                 prefer_skip_nested_validation=True)
def avg_roc_auc_score(y_true, y_proba, macro_average, n_classes):
    if np.any(y_true < 0) or np.any(y_true > n_classes - 1) or (np.unique(y_true).size > n_classes):
        raise ValueError
    if y_proba.shape[-1] != n_classes:
        raise ValueError

    roc_auc = sklm.roc_auc_score(y_true, y_proba, labels=range(n_classes),
                                 average=None, multi_class='ovr')
    
    if macro_average == 'mean':
        score = np.mean(roc_auc)
    elif macro_average == 'geo':
        score = gmean(roc_auc)
    elif macro_average == 'harm':
        if np.any(np.isclose(roc_auc, 0.0)):
            score = 0.0
        else:
            score = hmean(roc_auc)
    elif macro_average == 'min':
        score = np.min(roc_auc)
    else:
        raise ValueError
    return score


# classification metrics: PRC AUC score, macro-averaged per class
@validate_params({'y_true': ['array-like'],
                  'y_proba': ['array-like'],
                  'macro_average': [StrOptions({'mean', 'geo', 'harm', 'min'})],
                  'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')]},
                 prefer_skip_nested_validation=True)
def avg_prc_auc_score(y_true, y_proba, macro_average, n_classes):
    if np.any(y_true < 0) or np.any(y_true > n_classes - 1) or (np.unique(y_true).size > n_classes):
        raise ValueError
    if y_proba.shape[-1] != n_classes:
        raise ValueError

    prc_auc = sklm.average_precision_score(y_true, y_proba, average=None)

    if macro_average == 'mean':
        score = np.mean(prc_auc)
    elif macro_average == 'geo':
        score = gmean(prc_auc)
    elif macro_average == 'harm':
        if np.any(np.isclose(prc_auc, 0.0)):
            score = 0.0
        else:
            score = hmean(prc_auc)
    elif macro_average == 'min':
        score = np.min(prc_auc)
    else:
        raise ValueError
    return score


# classification metrics: Brier loss
@validate_params({'y_true': ['array-like'],
                  'y_proba': ['array-like'],
                  'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')]},
                 prefer_skip_nested_validation=True)
def brier_score_loss(y_true, y_proba, n_classes):
    if np.any(y_true < 0) or np.any(y_true > n_classes - 1) or (np.unique(y_true).size > n_classes):
        raise ValueError
    if y_proba.shape[-1] != n_classes:
        raise ValueError

    brier = []
    for c in range(n_classes):
        y_true_ = (y_true == c).astype(np.uint8)
        y_proba_ = y_proba[:, c]
        
        brier_ = sklm.brier_score_loss(y_true_, y_proba_)
        brier.append(brier_)
    score = np.mean(brier)
    return score


# ordinal classification metric: ranked probability score loss
@validate_params({'y_true': ['array-like'],
                  'y_proba': ['array-like'],
                  'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')]},
                 prefer_skip_nested_validation=True)
def ranked_prob_score_loss(y_true, y_proba, n_classes):
    if np.any(y_true < 0) or np.any(y_true > n_classes - 1) or (np.unique(y_true).size > n_classes):
        raise ValueError
    if y_proba.shape[-1] != n_classes:
        raise ValueError

    rps = []
    for c in range(n_classes - 1):
        # ordered partitions
        y_true_op = (y_true > c).astype(np.uint8)
        y_proba_op = np.sum(y_proba[:, (c + 1):], axis=1)

        # deal with issues due to numerical precision
        y_proba_op[y_proba_op < 0.0] = 0.0
        y_proba_op[y_proba_op > 1.0] = 1.0

        rps_op = sklm.brier_score_loss(y_true_op, y_proba_op)
        rps.append(rps_op)
    score = np.mean(rps)
    return score


# association metrics: Spearman's rank correlation
@validate_params({'y_true': ['array-like'],
                  'y_pred': ['array-like'],
                  'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')]},
                 prefer_skip_nested_validation=True)
def spearman_r_score(y_true, y_pred, n_classes):
    if np.any(y_true < 0) or np.any(y_true > n_classes - 1) or (np.unique(y_true).size > n_classes):
        raise ValueError
    if np.any(y_pred < 0) or np.any(y_pred > n_classes - 1) or (np.unique(y_pred).size > n_classes):
        raise ValueError

    score = spearmanr(y_true, y_pred).statistic
    return score


# association metrics: Kendall's tau
@validate_params({'y_true': ['array-like'],
                  'y_pred': ['array-like'],
                  'variant': [StrOptions({'b', 'c'})],
                  'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')]},
                 prefer_skip_nested_validation=True)
def kendall_tau_score(y_true, y_pred, variant, n_classes):
    if np.any(y_true < 0) or np.any(y_true > n_classes - 1) or (np.unique(y_true).size > n_classes):
        raise ValueError
    if np.any(y_pred < 0) or np.any(y_pred > n_classes - 1) or (np.unique(y_pred).size > n_classes):
        raise ValueError

    score = kendalltau(y_true, y_pred, variant=variant).statistic
    return score


# association metrics: Cohen's kappa
@validate_params({'y_true': ['array-like'],
                  'y_pred': ['array-like'],
                  'weights': [StrOptions({'linear', 'quadratic'}), None],
                  'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')]},
                 prefer_skip_nested_validation=True)
def cohen_kappa_score(y_true, y_pred, weights, n_classes):
    if np.any(y_true < 0) or np.any(y_true > n_classes - 1) or (np.unique(y_true).size > n_classes):
        raise ValueError
    if np.any(y_pred < 0) or np.any(y_pred > n_classes - 1) or (np.unique(y_pred).size > n_classes):
        raise ValueError

    score = sklm.cohen_kappa_score(y_true, y_pred, labels=range(n_classes), weights=weights)
    return score


# regression metrics: mean absolute error (MAE)
@validate_params({'y_true': ['array-like'],
                  'y_pred': ['array-like'],
                  'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')]},
                 prefer_skip_nested_validation=True)
def mean_absolute_error(y_true, y_pred, n_classes):
    if np.any(y_true < 0) or np.any(y_true > n_classes - 1) or (np.unique(y_true).size > n_classes):
        raise ValueError
    if np.any(y_pred < 0) or np.any(y_pred > n_classes - 1) or (np.unique(y_pred).size > n_classes):
        raise ValueError

    score = sklm.mean_absolute_error(y_true, y_pred)
    return score


# regression metrics: mean absolute error (MAE), macro-averaged per class
@validate_params({'y_true': ['array-like'],
                  'y_pred': ['array-like'],
                  'macro_average': [StrOptions({'mean', 'max'})],
                  'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')]},
                 prefer_skip_nested_validation=True)
def avg_mae(y_true, y_pred, macro_average, n_classes):
    if np.any(y_true < 0) or np.any(y_true > n_classes - 1) or (np.unique(y_true).size > n_classes):
        raise ValueError
    if np.any(y_pred < 0) or np.any(y_pred > n_classes - 1) or (np.unique(y_pred).size > n_classes):
        raise ValueError

    mae = []
    for c in range(n_classes):
        idx_ = (y_true == c)
        y_true_ = y_true[idx_]
        y_pred_ = y_pred[idx_]
        mae_ = sklm.mean_absolute_error(y_true_, y_pred_)
        mae.append(mae_)

    if macro_average == 'mean':
        score = np.mean(mae)
    elif macro_average == 'max':
        score = np.max(mae)
    else:
        raise ValueError
    return score


# regression metrics: mean squared error (MSE)
@validate_params({'y_true': ['array-like'],
                  'y_pred': ['array-like'],
                  'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')]},
                 prefer_skip_nested_validation=True)
def mean_squared_error(y_true, y_pred, n_classes):
    if np.any(y_true < 0) or np.any(y_true > n_classes - 1) or (np.unique(y_true).size > n_classes):
        raise ValueError
    if np.any(y_pred < 0) or np.any(y_pred > n_classes - 1) or (np.unique(y_pred).size > n_classes):
        raise ValueError

    score = sklm.mean_squared_error(y_true, y_pred)
    return score


# regression metrics: mean squared error (MSE), macro-averaged per class
@validate_params({'y_true': ['array-like'],
                  'y_pred': ['array-like'],
                  'macro_average': [StrOptions({'mean', 'max'})],
                  'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')]},
                 prefer_skip_nested_validation=True)
def avg_mse(y_true, y_pred, macro_average, n_classes):
    if np.any(y_true < 0) or np.any(y_true > n_classes - 1) or (np.unique(y_true).size > n_classes):
        raise ValueError
    if np.any(y_pred < 0) or np.any(y_pred > n_classes - 1) or (np.unique(y_pred).size > n_classes):
        raise ValueError

    mse = []
    for c in range(n_classes):
        idx_ = (y_true == c)
        y_true_ = y_true[idx_]
        y_pred_ = y_pred[idx_]
        mse_ = sklm.mean_squared_error(y_true_, y_pred_)
        mse.append(mse_)

    if macro_average == 'mean':
        score = np.mean(mse)
    elif macro_average == 'max':
        score = np.max(mse)
    else:
        raise ValueError
    return score


# regression metrics: root mean squared error (RMSE)
@validate_params({'y_true': ['array-like'],
                  'y_pred': ['array-like'],
                  'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')]},
                 prefer_skip_nested_validation=True)
def root_mean_squared_error(y_true, y_pred, n_classes):
    if np.any(y_true < 0) or np.any(y_true > n_classes - 1) or (np.unique(y_true).size > n_classes):
        raise ValueError
    if np.any(y_pred < 0) or np.any(y_pred > n_classes - 1) or (np.unique(y_pred).size > n_classes):
        raise ValueError

    score = sklm.root_mean_squared_error(y_true, y_pred)
    return score


# regression metrics: root mean squared error (RMSE), macro-averaged per class
@validate_params({'y_true': ['array-like'],
                  'y_pred': ['array-like'],
                  'macro_average': [StrOptions({'mean', 'max'})],
                  'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')]},
                 prefer_skip_nested_validation=True)
def avg_rmse(y_true, y_pred, macro_average, n_classes):
    if np.any(y_true < 0) or np.any(y_true > n_classes - 1) or (np.unique(y_true).size > n_classes):
        raise ValueError
    if np.any(y_pred < 0) or np.any(y_pred > n_classes - 1) or (np.unique(y_pred).size > n_classes):
        raise ValueError

    rmse = []
    for c in range(n_classes):
        idx_ = (y_true == c)
        y_true_ = y_true[idx_]
        y_pred_ = y_pred[idx_]
        rmse_ = sklm.root_mean_squared_error(y_true_, y_pred_)
        rmse.append(rmse_)

    if macro_average == 'mean':
        score = np.mean(rmse)
    elif macro_average == 'max':
        score = np.max(rmse)
    else:
        raise ValueError
    return score
