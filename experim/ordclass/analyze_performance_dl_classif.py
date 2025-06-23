import os
import json

from tqdm import tqdm

import numpy as np
import pandas as pd

import src.ordclass.ordinal_metrics as om


PATH_RESULTS = 'results/ordclass'

TAG_DL_CLASSIF = 'DL Classif'

TAG_INFO = '_summary.json'
TAG_Y_TRUE = '_y-true-test.npy'
TAG_Y_PRED = '_y-pred-test.npy'
TAG_PROBA_PRED = '_proba-pred-test.npy'
TAG_ANALYSIS = '_analysis.json'

EXT_ANALYSIS = '.csv'


def analyze(path, file_info):
    # files
    file_y_true = file_info.replace(TAG_INFO, TAG_Y_TRUE)
    file_y_pred = file_info.replace(TAG_INFO, TAG_Y_PRED)
    file_proba_pred = file_info.replace(TAG_INFO, TAG_PROBA_PRED)
    file_analysis = file_info.replace(TAG_INFO, TAG_ANALYSIS)

    # load basic info
    with open(os.path.join(path, file_info), 'r') as f_info:
        info = json.load(f_info)

    n_classes = info['n_classes']

    # load ground truth and predictions
    y_true = np.load(os.path.join(path, file_y_true))
    y_pred = np.load(os.path.join(path, file_y_pred))
    has_proba = os.path.exists(os.path.join(path, file_proba_pred))
    if has_proba:
        proba_pred = np.load(os.path.join(path, file_proba_pred))

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    if has_proba:
        proba_pred = np.reshape(proba_pred, (-1, n_classes))

    # compute performance metrics
    # classification metrics ('predict')
    accur = om.accuracy_score(y_true, y_pred, n_classes=n_classes)
    bal_acc = om.balanced_accuracy_score(y_true, y_pred, adjusted=True, n_classes=n_classes)
    avg_prec = om.avg_precision_score(y_true, y_pred, macro_average='mean', n_classes=n_classes)
    avg_sens = om.avg_sensitivity_score(y_true, y_pred, macro_average='mean', n_classes=n_classes)
    avg_f1 = om.avg_f1_score(y_true, y_pred, macro_average='mean', n_classes=n_classes)
    mcc = om.matthews_corrcoef_score(y_true, y_pred, normalize=False, n_classes=n_classes)

    # classification metrics ('predict_proba')
    if has_proba:
        avg_roc = om.avg_roc_auc_score(y_true, proba_pred, macro_average='mean', n_classes=n_classes)
        avg_prc = om.avg_prc_auc_score(y_true, proba_pred, macro_average='mean', n_classes=n_classes)
        brier = om.brier_score_loss(y_true, proba_pred, n_classes=n_classes)
        rps = om.ranked_prob_score_loss(y_true, proba_pred, n_classes=n_classes)

    # association metrics ('predict')
    spearr = om.spearman_r_score(y_true, y_pred, n_classes=n_classes)
    ktau = om.kendall_tau_score(y_true, y_pred, variant='b', n_classes=n_classes)
    ckappa = om.cohen_kappa_score(y_true, y_pred, weights='quadratic', n_classes=n_classes)

    # regression metrics ('predict')
    mae = om.mean_absolute_error(y_true, y_pred, n_classes=n_classes)
    avg_mae = om.avg_mae(y_true, y_pred, macro_average='mean', n_classes=n_classes)
    mse = om.mean_squared_error(y_true, y_pred, n_classes=n_classes)
    avg_mse = om.avg_mse(y_true, y_pred, macro_average='mean', n_classes=n_classes)
    rmse = om.root_mean_squared_error(y_true, y_pred, n_classes=n_classes)
    avg_rmse = om.avg_rmse(y_true, y_pred, macro_average='mean', n_classes=n_classes)

    # store the results of the analysis in terms of performance
    analysis = dict()
    analysis.update({'accur': accur, 'bal_acc': bal_acc,
                     'avg_prec': avg_prec, 'avg_sens': avg_sens,
                     'avg_f1': avg_f1, 'mcc': mcc})
    if has_proba:
        analysis.update({'avg_roc': avg_roc, 'avg_prc': avg_prc,
                         'brier': brier, 'rps': rps})
    analysis.update({'spearr': spearr, 'ktau': ktau, 'ckappa': ckappa})
    analysis.update({'mae': mae, 'avg_mae': avg_mae, 'mse': mse, 'avg_mse': avg_mse, 'rmse': rmse, 'avg_rmse': avg_rmse})

    for key, val in analysis.items():
        if np.isnan(val):
            analysis[key] = 'NaN'

    with open(os.path.join(path, file_analysis), 'w') as f_analys:
        json.dump(analysis, f_analys)
    
    return analysis


if __name__ == '__main__':
    # search for all information files: '_summary.json'
    l_files_info = []
    for file_ in os.listdir(PATH_RESULTS):
        if file_.startswith(TAG_DL_CLASSIF) and file_.endswith(TAG_INFO):
            l_files_info.append(file_)
    l_files_info = sorted(l_files_info)
    n_files_info = len(l_files_info)

    l_analysis = []
    for file_info_ in tqdm(l_files_info, total=n_files_info, desc='Analyzing performance - DL Classif'):
        analysis = analyze(path=PATH_RESULTS, file_info=file_info_)
        analysis['file'] = file_info_
        l_analysis.append(analysis)

    df_analysis = pd.DataFrame.from_dict(l_analysis)
    df_file = df_analysis['file']
    df_analysis = df_analysis.drop(columns=['file'])
    df_analysis.insert(0, 'file', df_file)
    df_analysis.to_csv(os.path.join(PATH_RESULTS, TAG_DL_CLASSIF + EXT_ANALYSIS), index=False)

    print('Done!')
