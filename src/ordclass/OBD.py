# Author: Fernando García-García <fegarcia@bcamath.org>

import numpy as np
import scipy as sp

from sklearn.utils import check_array, column_or_1d
from sklearn.utils._param_validation import validate_params, StrOptions

DECOMP_NOM = 'Nom'
DECOMP_ORD = ['OvN', 'OvS', 'OvP', 'OrdP']
DECOMP_ALL = [DECOMP_NOM] + DECOMP_ORD

NUM_DECOMP_ORD = len(DECOMP_ORD)

OBD_WEIGHT_METHODS = ['frequency', 'diff_entropy']


@validate_params(
    parameter_constraints={'obd_weight': [dict]},
    prefer_skip_nested_validation=True
)
def check_obd_weight(obd_weight):
    # check the weights of the different decomposition strategies towards the overall loss
    if set(obd_weight.keys()) != set(DECOMP_ALL):
        raise ValueError
    for dec_ in DECOMP_ALL:
        if np.any(obd_weight[dec_] < 0.0):  # negative weights are not allowed
            raise ValueError

    return obd_weight


@validate_params(
    parameter_constraints={'class_counts': ['array-like'],
                           'method': [StrOptions({*OBD_WEIGHT_METHODS})]},
    prefer_skip_nested_validation=True
)
def compute_obd_weight(class_counts, *, method='diff_entropy'):
    class_counts = column_or_1d(class_counts)
    class_counts = check_array(class_counts,
                               dtype=int,
                               ensure_2d=False,
                               ensure_all_finite=True,
                               ensure_non_negative=True)

    if method == 'frequency':
        obd_weight = _obd_frequency(class_counts)
    elif method == 'diff_entropy':
        obd_weight = _obd_diff_entropy(class_counts)
    else:
        raise RuntimeError

    return obd_weight


def _obd_frequency(class_counts):
    # compute frequencies from class counts
    n_classes = class_counts.size
    obd_weight = {dec_: np.zeros(shape=(n_classes - 1,), dtype=np.float32) for dec_ in DECOMP_ORD}

    n_samples = np.sum(class_counts)

    obd_weight[DECOMP_NOM] = 1.0

    for c in range(n_classes - 1):
        # OvN
        count_neg_ovn = class_counts[c]
        count_pos_ovn = class_counts[c + 1]
        counts_ovn = count_neg_ovn + count_pos_ovn
        obd_weight['OvN'][c] = (counts_ovn / n_samples) / ((n_classes - 1) * NUM_DECOMP_ORD)

        # OvS
        count_neg_ovs = class_counts[c]
        count_pos_ovs = np.sum(class_counts[(c + 1):])
        counts_ovs = count_neg_ovs + count_pos_ovs
        obd_weight['OvS'][c] = (counts_ovs / n_samples) / ((n_classes - 1) * NUM_DECOMP_ORD)

        # OvP
        count_neg_ovp = np.sum(class_counts[:(c + 1)])
        count_pos_ovp = class_counts[c + 1]
        counts_ovp = count_neg_ovp + count_pos_ovp
        obd_weight['OvP'][c] = (counts_ovp / n_samples) / ((n_classes - 1) * NUM_DECOMP_ORD)

        # OrdP
        # count_neg_ordp = np.sum(class_counts[:(c + 1)])
        # count_pos_ordp = np.sum(class_counts[(c + 1):])
        # counts_ordp = count_neg_ordp + count_pos_ordp
        # obd_weight['OrdP'][c] = counts_ordp / n_samples
        obd_weight['OrdP'][c] = 1.0 / ((n_classes - 1) * NUM_DECOMP_ORD)  # by construction of OrdP

    return obd_weight


def _obd_diff_entropy(class_counts):
    # entropy of a Dirichlet distribution
    def _entropy_dirichlet(alpha):
        alpha_dim = alpha.size
        alpha_tot = np.sum(alpha)

        h = 0.0
        for d in range(alpha_dim):
            h += sp.special.gammaln(alpha[d])
        h -= sp.special.gammaln(alpha_tot)

        h += (alpha_tot - alpha_dim) * sp.special.digamma(alpha_tot)
        for d in range(alpha_dim):
            h -= (alpha[d] - 1) * sp.special.digamma(alpha[d])

        return h

    # compute differential entropies from class counts
    n_classes = class_counts.size
    obd_weight = {dec_: np.zeros(shape=(n_classes - 1,), dtype=np.float32) for dec_ in DECOMP_ORD}

    h_max_nom = _entropy_dirichlet(alpha=np.ones(shape=(n_classes,)))
    h_obs_nom = _entropy_dirichlet(alpha=class_counts + 1)
    obd_weight[DECOMP_NOM] = (h_max_nom - h_obs_nom)

    h_max_bin = 0.0
    for c in range(n_classes - 1):
        # OvN
        count_neg_ovn = class_counts[c]
        count_pos_ovn = class_counts[c + 1]
        counts_ovn = np.asarray([count_neg_ovn, count_pos_ovn])
        h_obs_ovn = _entropy_dirichlet(alpha=counts_ovn + 1)
        obd_weight['OvN'][c] = (h_max_bin - h_obs_ovn) / (n_classes - 1)

        # OvS
        count_neg_ovs = class_counts[c]
        count_pos_ovs = np.sum(class_counts[(c + 1):])
        counts_ovs = np.asarray([count_neg_ovs, count_pos_ovs])
        h_obs_ovs = _entropy_dirichlet(alpha=counts_ovs + 1)
        obd_weight['OvS'][c] = (h_max_bin - h_obs_ovs) / (n_classes - 1)

        # OvP
        count_neg_ovp = np.sum(class_counts[:(c + 1)])
        count_pos_ovp = class_counts[c + 1]
        counts_ovp = np.asarray([count_neg_ovp, count_pos_ovp])
        h_obs_ovp = _entropy_dirichlet(alpha=counts_ovp + 1)
        obd_weight['OvP'][c] = (h_max_bin - h_obs_ovp) / (n_classes - 1)

        # OrdP
        count_neg_ordp = np.sum(class_counts[:(c + 1)])
        count_pos_ordp = np.sum(class_counts[(c + 1):])
        counts_ordp = np.asarray([count_neg_ordp, count_pos_ordp])
        h_obs_ordp = _entropy_dirichlet(alpha=counts_ordp + 1)
        obd_weight['OrdP'][c] = (h_max_bin - h_obs_ordp) / (n_classes - 1)

    return obd_weight
