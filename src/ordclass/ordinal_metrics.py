# Author: Fernando García-García <fegarcia@bcamath.org>

from numbers import Integral

import numpy as np
import scipy as sp

import sklearn.metrics as sklm
from sklearn.utils._param_validation import validate_params, Interval, StrOptions

_MIN_CLASSES = 3
_MAX_CLASSES = 2 ** 8


# classification report: confusion matrix
@validate_params(
    parameter_constraints={'y_true': ['array-like'],
                           'y_pred': ['array-like'],
                           'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')]},
    prefer_skip_nested_validation=True
)
def confusion_matrix(y_true, y_pred, *, n_classes):
    set_classes = set(np.arange(n_classes))
    set_y_true = set(y_true.flatten())
    set_y_pred = set(y_true.flatten())
    if (not set_y_true.issubset(set_classes)) or (not set_y_pred.issubset(set_classes)):
        raise ValueError

    cf_mtx = sklm.confusion_matrix(y_true, y_pred, labels=np.arange(n_classes), normalize=None)
    return cf_mtx


# classification metrics: accuracy
@validate_params(
    parameter_constraints={'y_true': ['array-like'],
                           'y_pred': ['array-like'],
                           'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')]},
    prefer_skip_nested_validation=True
)
def accuracy_score(y_true, y_pred, *, n_classes):
    set_classes = set(np.arange(n_classes))
    set_y_true = set(y_true.flatten())
    set_y_pred = set(y_true.flatten())
    if (not set_y_true.issubset(set_classes)) or (not set_y_pred.issubset(set_classes)):
        raise ValueError

    score = sklm.accuracy_score(y_true, y_pred)
    return score


# classification metrics: balanced accuracy, i.e. macro-averaged per-class sensitivity
@validate_params(
    parameter_constraints={'y_true': ['array-like'],
                           'y_pred': ['array-like'],
                           'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')],
                           'adjusted': ['boolean']},
    prefer_skip_nested_validation=True
)
def balanced_accuracy_score(y_true, y_pred, *, n_classes, adjusted=False):
    set_classes = set(np.arange(n_classes))
    set_y_true = set(y_true.flatten())
    set_y_pred = set(y_true.flatten())
    if (not set_y_true.issubset(set_classes)) or (not set_y_pred.issubset(set_classes)):
        raise ValueError

    score = sklm.balanced_accuracy_score(y_true, y_pred, adjusted=adjusted)
    return score


# classification metrics: precision, macro-averaged per class
@validate_params(
    parameter_constraints={'y_true': ['array-like'],
                           'y_pred': ['array-like'],
                           'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')],
                           'macro_average': [StrOptions({'mean', 'geo', 'harm', 'min'})]},
    prefer_skip_nested_validation=True
)
def avg_precision_score(y_true, y_pred, *, n_classes, macro_average='mean'):
    set_classes = set(np.arange(n_classes))
    set_y_true = set(y_true.flatten())
    set_y_pred = set(y_true.flatten())
    if (not set_y_true.issubset(set_classes)) or (not set_y_pred.issubset(set_classes)):
        raise ValueError

    precis, _, _, _ = sklm.precision_recall_fscore_support(y_true, y_pred, labels=range(n_classes),
                                                           average=None, zero_division=0.0)

    if macro_average == 'mean':
        score = np.mean(precis)
    elif macro_average == 'geo':
        score = sp.stats.gmean(precis)
    elif macro_average == 'harm':
        if np.any(np.isclose(precis, 0.0)):
            score = 0.0
        else:
            score = sp.stats.hmean(precis)
    elif macro_average == 'min':
        score = np.min(precis)
    else:
        raise ValueError
    return score


# classification metrics: sensitivity, macro-averaged per class
@validate_params(
    parameter_constraints={'y_true': ['array-like'],
                           'y_pred': ['array-like'],
                           'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')],
                           'macro_average': [StrOptions({'mean', 'geo', 'harm', 'min'})]},
    prefer_skip_nested_validation=True
)
def avg_sensitivity_score(y_true, y_pred, *, n_classes, macro_average='mean'):
    set_classes = set(np.arange(n_classes))
    set_y_true = set(y_true.flatten())
    set_y_pred = set(y_true.flatten())
    if (not set_y_true.issubset(set_classes)) or (not set_y_pred.issubset(set_classes)):
        raise ValueError

    _, recall, _, _ = sklm.precision_recall_fscore_support(y_true, y_pred, labels=range(n_classes),
                                                           average=None, zero_division=0.0)
    sensit = recall  # sensitivity is always identical to recall, by definition!

    if macro_average == 'mean':
        score = np.mean(sensit)
    elif macro_average == 'geo':
        score = sp.stats.gmean(sensit)
    elif macro_average == 'harm':
        if np.any(np.isclose(sensit, 0.0)):
            score = 0.0
        else:
            score = sp.stats.hmean(sensit)
    elif macro_average == 'min':
        score = np.min(sensit)
    else:
        raise ValueError
    return score


# classification metrics: F1 score, macro-averaged per class
@validate_params(
    parameter_constraints={'y_true': ['array-like'],
                           'y_pred': ['array-like'],
                           'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')],
                           'macro_average': [StrOptions({'mean', 'geo', 'harm', 'min'})]},
    prefer_skip_nested_validation=True
)
def avg_f1_score(y_true, y_pred, *, n_classes, macro_average='mean'):
    set_classes = set(np.arange(n_classes))
    set_y_true = set(y_true.flatten())
    set_y_pred = set(y_true.flatten())
    if (not set_y_true.issubset(set_classes)) or (not set_y_pred.issubset(set_classes)):
        raise ValueError

    f1 = sklm.f1_score(y_true, y_pred, labels=range(n_classes),
                       average=None, zero_division=0.0)

    if macro_average == 'mean':
        score = np.mean(f1)
    elif macro_average == 'geo':
        score = sp.stats.gmean(f1)
    elif macro_average == 'harm':
        if np.any(np.isclose(f1, 0.0)):
            score = 0.0
        else:
            score = sp.stats.hmean(f1)
    elif macro_average == 'min':
        score = np.min(f1)
    else:
        raise ValueError
    return score


# classification metrics: ROC AUC score, macro-averaged per class
@validate_params(
    parameter_constraints={'y_true': ['array-like'],
                           'y_proba': ['array-like'],
                           'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')],
                           'macro_average': [StrOptions({'mean', 'geo', 'harm', 'min'})]},
    prefer_skip_nested_validation=True
)
def avg_roc_auc_score(y_true, y_proba, *, n_classes, macro_average='mean'):
    set_classes = set(np.arange(n_classes))
    set_y_true = set(y_true.flatten())
    if not set_y_true.issubset(set_classes):
        raise ValueError

    if y_proba.shape[-1] != n_classes:
        raise ValueError
    if np.any(y_proba < 0.0, axis=None) or np.any(y_proba > 1.0, axis=None):
        raise ValueError

    roc_auc = sklm.roc_auc_score(y_true, y_proba, labels=range(n_classes),
                                 average=None, multi_class='ovr')

    if macro_average == 'mean':
        score = np.mean(roc_auc)
    elif macro_average == 'geo':
        score = sp.stats.gmean(roc_auc)
    elif macro_average == 'harm':
        if np.any(np.isclose(roc_auc, 0.0)):
            score = 0.0
        else:
            score = sp.stats.hmean(roc_auc)
    elif macro_average == 'min':
        score = np.min(roc_auc)
    else:
        raise ValueError
    return score


# classification metrics: PRC AUC score, macro-averaged per class
@validate_params(
    parameter_constraints={'y_true': ['array-like'],
                           'y_proba': ['array-like'],
                           'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')],
                           'macro_average': [StrOptions({'mean', 'geo', 'harm', 'min'})]},
    prefer_skip_nested_validation=True
)
def avg_prc_auc_score(y_true, y_proba, *, n_classes, macro_average='mean'):
    set_classes = set(np.arange(n_classes))
    set_y_true = set(y_true.flatten())
    if not set_y_true.issubset(set_classes):
        raise ValueError

    if y_proba.shape[-1] != n_classes:
        raise ValueError
    if np.any(y_proba < 0.0, axis=None) or np.any(y_proba > 1.0, axis=None):
        raise ValueError

    prc_auc = sklm.average_precision_score(y_true, y_proba, average=None)

    if macro_average == 'mean':
        score = np.mean(prc_auc)
    elif macro_average == 'geo':
        score = sp.stats.gmean(prc_auc)
    elif macro_average == 'harm':
        if np.any(np.isclose(prc_auc, 0.0)):
            score = 0.0
        else:
            score = sp.stats.hmean(prc_auc)
    elif macro_average == 'min':
        score = np.min(prc_auc)
    else:
        raise ValueError
    return score


# classification metrics: Brier loss
@validate_params(
    parameter_constraints={'y_true': ['array-like'],
                           'y_proba': ['array-like'],
                           'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')]},
    prefer_skip_nested_validation=True
)
def brier_score_loss(y_true, y_proba, *, n_classes):
    set_classes = set(np.arange(n_classes))
    set_y_true = set(y_true.flatten())
    if not set_y_true.issubset(set_classes):
        raise ValueError

    if y_proba.shape[-1] != n_classes:
        raise ValueError
    if np.any(y_proba < 0.0, axis=None) or np.any(y_proba > 1.0, axis=None):
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
@validate_params(
    parameter_constraints={'y_true': ['array-like'],
                           'y_proba': ['array-like'],
                           'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')]},
    prefer_skip_nested_validation=True
)
def ranked_prob_score_loss(y_true, y_proba, *, n_classes):
    set_classes = set(np.arange(n_classes))
    set_y_true = set(y_true.flatten())
    if not set_y_true.issubset(set_classes):
        raise ValueError

    if y_proba.shape[-1] != n_classes:
        raise ValueError
    if np.any(y_proba < 0.0, axis=None) or np.any(y_proba > 1.0, axis=None):
        raise ValueError

    rps = []
    for c in range(n_classes - 1):
        # cumulative probability
        y_true_cum = (y_true > c).astype(np.uint8)
        y_proba_cum = np.sum(y_proba[:, (c + 1):], axis=1)

        # deal with issues due to numerical precision
        y_proba_cum[y_proba_cum < 0.0] = 0.0
        y_proba_cum[y_proba_cum > 1.0] = 1.0

        rps_ = sklm.brier_score_loss(y_true_cum, y_proba_cum)
        rps.append(rps_)
    score = np.mean(rps)
    return score


# association metrics: Matthews' correlation coefficient (MCC)
@validate_params(
    parameter_constraints={'y_true': ['array-like'],
                           'y_pred': ['array-like'],
                           'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')],
                           'normalize': ['boolean']},
    prefer_skip_nested_validation=True
)
def matthews_corrcoef_score(y_true, y_pred, *, n_classes, normalize=True):
    set_classes = set(np.arange(n_classes))
    set_y_true = set(y_true.flatten())
    set_y_pred = set(y_true.flatten())
    if (not set_y_true.issubset(set_classes)) or (not set_y_pred.issubset(set_classes)):
        raise ValueError

    score = sklm.matthews_corrcoef(y_true, y_pred)
    if normalize:
        score = (1.0 + score) / 2.0
    return score


# association metrics: Cohen's kappa
@validate_params(
    parameter_constraints={'y_true': ['array-like'],
                           'y_pred': ['array-like'],
                           'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')],
                           'weights': [StrOptions({'linear', 'quadratic'}), None]},
    prefer_skip_nested_validation=True
)
def cohen_kappa_score(y_true, y_pred, *, n_classes, weights=None):
    set_classes = set(np.arange(n_classes))
    set_y_true = set(y_true.flatten())
    set_y_pred = set(y_true.flatten())
    if (not set_y_true.issubset(set_classes)) or (not set_y_pred.issubset(set_classes)):
        raise ValueError

    score = sklm.cohen_kappa_score(y_true, y_pred, labels=range(n_classes), weights=weights)
    return score


# association metrics: Spearman's rank correlation
@validate_params(
    parameter_constraints={'y_true': ['array-like'],
                           'y_pred': ['array-like'],
                           'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')]},
    prefer_skip_nested_validation=True
)
def spearman_r_score(y_true, y_pred, *, n_classes):
    set_classes = set(np.arange(n_classes))
    set_y_true = set(y_true.flatten())
    set_y_pred = set(y_true.flatten())
    if (not set_y_true.issubset(set_classes)) or (not set_y_pred.issubset(set_classes)):
        raise ValueError

    score = sp.stats.spearmanr(y_true, y_pred).statistic
    return score


# association metrics: Kendall's tau
@validate_params(
    parameter_constraints={'y_true': ['array-like'],
                           'y_pred': ['array-like'],
                           'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')]},
    prefer_skip_nested_validation=True
)
def kendall_tau_score(y_true, y_pred, *, n_classes):
    set_classes = set(np.arange(n_classes))
    set_y_true = set(y_true.flatten())
    set_y_pred = set(y_true.flatten())
    if (not set_y_true.issubset(set_classes)) or (not set_y_pred.issubset(set_classes)):
        raise ValueError

    score = sp.stats.kendalltau(y_true, y_pred).statistic
    return score


# regression metrics: mean absolute error (MAE)
@validate_params(
    parameter_constraints={'y_true': ['array-like'],
                           'y_pred': ['array-like'],
                           'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')]},
    prefer_skip_nested_validation=True
)
def mean_absolute_error(y_true, y_pred, *, n_classes):
    set_classes = set(np.arange(n_classes))
    set_y_true = set(y_true.flatten())
    set_y_pred = set(y_true.flatten())
    if (not set_y_true.issubset(set_classes)) or (not set_y_pred.issubset(set_classes)):
        raise ValueError

    score = sklm.mean_absolute_error(y_true, y_pred)
    return score


# regression metrics: mean absolute error (MAE), macro-averaged per class
@validate_params(
    parameter_constraints={'y_true': ['array-like'],
                           'y_pred': ['array-like'],
                           'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')],
                           'macro_average': [StrOptions({'mean', 'max'})]},
    prefer_skip_nested_validation=True
)
def avg_mae(y_true, y_pred, *, n_classes, macro_average='mean'):
    set_classes = set(np.arange(n_classes))
    set_y_true = set(y_true.flatten())
    set_y_pred = set(y_true.flatten())
    if (not set_y_true.issubset(set_classes)) or (not set_y_pred.issubset(set_classes)):
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
@validate_params(
    parameter_constraints={'y_true': ['array-like'],
                           'y_pred': ['array-like'],
                           'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')]},
    prefer_skip_nested_validation=True
)
def mean_squared_error(y_true, y_pred, *, n_classes):
    set_classes = set(np.arange(n_classes))
    set_y_true = set(y_true.flatten())
    set_y_pred = set(y_true.flatten())
    if (not set_y_true.issubset(set_classes)) or (not set_y_pred.issubset(set_classes)):
        raise ValueError

    score = sklm.mean_squared_error(y_true, y_pred)
    return score


# regression metrics: mean squared error (MSE), macro-averaged per class
@validate_params(
    parameter_constraints={'y_true': ['array-like'],
                           'y_pred': ['array-like'],
                           'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')],
                           'macro_average': [StrOptions({'mean', 'max'})]},
    prefer_skip_nested_validation=True
)
def avg_mse(y_true, y_pred, *, n_classes, macro_average='mean'):
    set_classes = set(np.arange(n_classes))
    set_y_true = set(y_true.flatten())
    set_y_pred = set(y_true.flatten())
    if (not set_y_true.issubset(set_classes)) or (not set_y_pred.issubset(set_classes)):
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
@validate_params(
    parameter_constraints={'y_true': ['array-like'],
                           'y_pred': ['array-like'],
                           'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')]},
    prefer_skip_nested_validation=True
)
def root_mean_squared_error(y_true, y_pred, *, n_classes):
    set_classes = set(np.arange(n_classes))
    set_y_true = set(y_true.flatten())
    set_y_pred = set(y_true.flatten())
    if (not set_y_true.issubset(set_classes)) or (not set_y_pred.issubset(set_classes)):
        raise ValueError

    score = sklm.root_mean_squared_error(y_true, y_pred)
    return score


# regression metrics: root mean squared error (RMSE), macro-averaged per class
@validate_params(
    parameter_constraints={'y_true': ['array-like'],
                           'y_pred': ['array-like'],
                           'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')],
                           'macro_average': [StrOptions({'mean', 'max'})]},
    prefer_skip_nested_validation=True
)
def avg_rmse(y_true, y_pred, *, n_classes, macro_average='mean'):
    set_classes = set(np.arange(n_classes))
    set_y_true = set(y_true.flatten())
    set_y_pred = set(y_true.flatten())
    if (not set_y_true.issubset(set_classes)) or (not set_y_pred.issubset(set_classes)):
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
