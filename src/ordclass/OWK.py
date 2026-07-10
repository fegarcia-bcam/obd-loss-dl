# Author: Fernando García-García <fegarcia@bcamath.org>

from numbers import Integral

import numpy as np

from sklearn.utils import check_array
from sklearn.utils._param_validation import validate_params, Interval, StrOptions

_MIN_CLASSES = 3
_MAX_CLASSES = 2 ** 8


@validate_params(
    parameter_constraints={'owk_weight': ['array-like']},
    prefer_skip_nested_validation=True
)
def check_owk_weight(owk_weight):
    # check that the weights matrix makes sense
    owk_weight = check_array(owk_weight,
                             dtype=np.float32,
                             ensure_2d=True,
                             ensure_all_finite=True,
                             ensure_non_negative=True)

    if owk_weight.shape[0] != owk_weight.shape[1]:  # square matrix
        raise ValueError

    if np.any(np.diag(owk_weight) != 0.0):  # all zero-valued weights in the main diagonal
        raise ValueError

    for i in range(owk_weight.shape[0]):  # all positive weights outside the main diagonal
        for j in range(i + 1, owk_weight.shape[0]):
            if (owk_weight[i, j] <= 0.0) or (owk_weight[j, i] <= 0.0):
                raise ValueError

    return owk_weight


@validate_params(
    parameter_constraints={'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')],
                           'penalty': [StrOptions({'nominal', 'linear', 'quadratic'})]},
    prefer_skip_nested_validation=True
)
def compute_owk_weight(n_classes, *, penalty):
    # weights matrix
    owk_weight = np.zeros(shape=(n_classes, n_classes), dtype=np.float32)
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            if penalty == 'nominal':
                w_ = 1.0
            elif penalty == 'linear':
                w_ = (j - i) / (n_classes - 1)
            elif penalty == 'quadratic':
                w_ = (j - i) ** 2 / (n_classes - 1) ** 2
            else:
                raise ValueError
            owk_weight[i, j] = w_
            owk_weight[j, i] = w_

    return owk_weight
