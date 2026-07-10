# Author: Fernando García-García <fegarcia@bcamath.org>

import numpy as np

import src.ordclass.ordinal_metrics as om


def get_performance(y_true, y_pred, proba_pred, *, n_classes):
    has_proba = (proba_pred is not None)

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    if has_proba:
        proba_pred = np.reshape(proba_pred, (-1, n_classes))

    # compute performance metrics
    # classification metrics ('predict')
    cfn_mtx = om.confusion_matrix(y_true, y_pred, n_classes=n_classes)

    accur = om.accuracy_score(y_true, y_pred, n_classes=n_classes)
    bal_acc = om.balanced_accuracy_score(y_true, y_pred, adjusted=True, n_classes=n_classes)
    
    avg_prec = om.avg_precision_score(y_true, y_pred, n_classes=n_classes, macro_average='mean')
    avg_sens = om.avg_sensitivity_score(y_true, y_pred, n_classes=n_classes, macro_average='mean')
    avg_f1 = om.avg_f1_score(y_true, y_pred, n_classes=n_classes, macro_average='mean')

    # classification metrics ('predict_proba')
    if has_proba:
        avg_auroc = om.avg_roc_auc_score(y_true, proba_pred, n_classes=n_classes, macro_average='mean')
        avg_auprc = om.avg_prc_auc_score(y_true, proba_pred, n_classes=n_classes, macro_average='mean')
        brier = om.brier_score_loss(y_true, proba_pred, n_classes=n_classes)
        rps = om.ranked_prob_score_loss(y_true, proba_pred, n_classes=n_classes)
    else:
        avg_auroc, avg_auprc, brier, rps = None, None, None, None

    # association metrics ('predict')
    mcc = om.matthews_corrcoef_score(y_true, y_pred, n_classes=n_classes, normalize=False)
    ckappa = om.cohen_kappa_score(y_true, y_pred, n_classes=n_classes, weights='quadratic')
    spearr = om.spearman_r_score(y_true, y_pred, n_classes=n_classes)
    ktau = om.kendall_tau_score(y_true, y_pred, n_classes=n_classes)

    # regression metrics ('predict')
    mae = om.mean_absolute_error(y_true, y_pred, n_classes=n_classes)
    mse = om.mean_squared_error(y_true, y_pred, n_classes=n_classes)
    rmse = om.root_mean_squared_error(y_true, y_pred, n_classes=n_classes)
    
    avg_mae = om.avg_mae(y_true, y_pred, n_classes=n_classes, macro_average='mean')
    avg_mse = om.avg_mse(y_true, y_pred, n_classes=n_classes, macro_average='mean')
    avg_rmse = om.avg_rmse(y_true, y_pred, n_classes=n_classes, macro_average='mean')

    # store the results of the performance analysis
    report = dict()
    report.update({'cfn_mtx': cfn_mtx, 'accur': accur, 'bal_acc': bal_acc,
                   'avg_prec': avg_prec, 'avg_sens': avg_sens, 'avg_f1': avg_f1})
    report.update({'avg_auroc': avg_auroc, 'avg_auprc': avg_auprc,
                   'brier': brier, 'rps': rps})
    report.update({'mcc': mcc, 'spearr': spearr, 'ktau': ktau, 'ckappa': ckappa})
    report.update({'mae': mae, 'mse': mse, 'rmse': rmse,
                   'avg_mae': avg_mae, 'avg_mse': avg_mse, 'avg_rmse': avg_rmse})

    return report
