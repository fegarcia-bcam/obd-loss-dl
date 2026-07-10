# Author: Fernando García-García <fegarcia@bcamath.org>

import copy
import itertools

_OUT_TYPES = ['regress', 'nominal']

_EXPERIM = []
_experim = {
    'loss': 'reg_mae'
}
_EXPERIM.append(_experim)

_experim = {
    'loss': 'reg_mse'
}
_EXPERIM.append(_experim)

_experim = {
    'loss': 'nom_ce'
}
_EXPERIM.append(_experim)

_experim = {
    'loss': 'nom_focal',
    'gamma': 2.0
}
_EXPERIM.append(_experim)

_experim = {
    'loss': 'ur_ce',
    'ur_type': 'beta',
    'ur_eta': 0.5,
    'ur_delta': 1.0
}
_EXPERIM.append(_experim)

_experim = {
    'loss': 'ur_focal',
    'gamma': 2.0,
    'ur_type': 'beta',
    'ur_eta': 0.5,
    'ur_delta': 1.0
}
_EXPERIM.append(_experim)

_experim = {
    'loss': 'owk',
    'owk_penalty': 'linear'
}
_EXPERIM.append(_experim)

_experim = {
    'loss': 'owk',
    'owk_penalty': 'quadratic'
}
_EXPERIM.append(_experim)

_experim = {
    'loss': 'obd_ce',
    'obd_w_method': 'frequency'
}
_EXPERIM.append(_experim)

_experim = {
    'loss': 'obd_ce',
    'obd_w_method': 'diff_entropy'
}
_EXPERIM.append(_experim)

_experim = {
    'loss': 'obd_focal',
    'gamma': 2.0,
    'obd_w_method': 'frequency'
}
_EXPERIM.append(_experim)

_experim = {
    'loss': 'obd_focal',
    'gamma': 2.0,
    'obd_w_method': 'diff_entropy'
}
_EXPERIM.append(_experim)

EXPERIM = []
for experim, out_type in itertools.product(_EXPERIM, _OUT_TYPES):
    if out_type == 'regress':
        if not experim['loss'].startswith('reg_'):
            continue
    else:
        if experim['loss'].startswith('reg_'):
            continue

    experim_ = copy.deepcopy(experim)
    experim_.update({'out_type': out_type})
    EXPERIM.append(experim_)
    
NUM_EXPERIM = len(EXPERIM)
