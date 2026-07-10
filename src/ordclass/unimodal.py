# Author: Fernando García-García <fegarcia@bcamath.org>

from numbers import Integral, Real

import numpy as np
import scipy as sp

from sklearn.utils._param_validation import validate_params, Interval, StrOptions

_MIN_CLASSES = 3
_MAX_CLASSES = 2 ** 8

REGULARIZATION_OPTS = ['beta']


@validate_params(
    parameter_constraints={'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')],
                           'type': [StrOptions({*REGULARIZATION_OPTS}), None],
                           'delta': [Interval(Real, 0.0, None, closed='neither'), None]},
    prefer_skip_nested_validation=True
)
def get_regularization(n_classes, *, type, delta):
    if type is None:
        # no regularization
        ur_terms = np.eye(n_classes, dtype=np.float32)
    elif type == 'beta':
        if delta is None:
            raise ValueError
        ur_terms = _regularize_beta(n_classes, delta=delta)
    else:
        raise NotImplementedError

    return ur_terms

def _regularize_beta(n_classes, *, delta=1.0):
    # delta controls how many standard deviations are expected in each interval
    # in the original publication, delta is implicitly assumed to be one and never changed

    reg_terms = np.zeros(shape=(n_classes, n_classes), dtype=np.float32)

    for c_true in range(n_classes):
        beta_a = ((2 * n_classes - 2 * c_true - 1) * (2 * c_true + 1) * (delta ** 2) - 1) * (2 * c_true + 1) / (2 * n_classes)
        beta_b = (2 * n_classes - 2 * c_true - 1) / (2 * c_true + 1) * beta_a

        beta_cdf = lambda x: sp.stats.beta.cdf(x, a=beta_a, b=beta_b)
        for c_aux in range(n_classes):
            x_lo = c_aux / n_classes
            x_hi = (c_aux + 1) / n_classes
            reg = beta_cdf(x_hi) - beta_cdf(x_lo)
            reg_terms[c_aux, c_true] = reg

    return reg_terms
