import os
import json

import numpy as np
import pandas as pd

from tqdm import tqdm

from baycomp import HierarchicalTest

from experim.ordclass import CONFIG


RESULTS_EXT = '.json'

# RESULTS_INFO = ['dataset', 'task_type', 'task_name', 'n_classes', 'exprm_idx', 'exprm_total', 'split_idx', 'split_total', 'early_stop']

COL_NUM_CLASSES = 'n_classes'

COL_DATA_IN_1 = 'dataset'
COL_DATA_IN_2 = 'task_type'
COL_DATA_OUT = 'data_task'

COL_EXPRM = 'exprm_idx'
COL_SPLIT = 'split_idx'

RESULTS_INFO = [COL_DATA_IN_1, COL_DATA_IN_2, COL_EXPRM, COL_SPLIT]

RESULTS_PERFORM = 'performance_test'
RESULTS_EXCLUDE = 'cfn_mtx'

TAG_PERFORM = 'performance_'

PERFORM_METRICS = {
    'avg_f1': {'dir': 'max', 'rope': 0.01},
    'mcc': {'dir': 'max', 'rope': 0.01},
    'avg_mae': {'dir': 'min', 'rope': 0.01},
    'avg_rmse': {'dir': 'min', 'rope': 0.01},
    'ckappa': {'dir': 'max', 'rope': 0.01},
    'avg_auroc': {'dir': 'max', 'rope': 0.005},  # inherently more stable
    'avg_auprc': {'dir': 'max', 'rope': 0.005},  # inherently more stable
    'brier': {'dir': 'min', 'rope': 0.01},
    'rps': {'dir': 'min', 'rope': 0.01}
}

FILE_RESULTS_PERFORM = 'Results_Analysis-Performance summary.json'
FILE_RESULTS_BAYES = 'Results_Analysis-Bayesian comparison.json'


if __name__ == '__main__':
    # gather information about all the experimental results
    l_results = []
    for filename in sorted(os.listdir(CONFIG.PATH_RESULTS)):
        if filename.endswith(RESULTS_EXT):
            with open(os.path.join(CONFIG.PATH_RESULTS, filename), 'r') as f_results:
                results_all = json.load(f_results)

                results_sel = {}
                for info in RESULTS_INFO:
                    results_sel[info] = results_all.get(info, None)

                results_perf = results_all[RESULTS_PERFORM]
                results_perf.pop(RESULTS_EXCLUDE)

                results_perf = {TAG_PERFORM + key: val for key, val in results_perf.items()}

                # IMPORTANT: normalize regression error metrics (i.e. MAE, MSE, RMSE) attending to the number of classes
                n_classes = results_all[COL_NUM_CLASSES]
                results_perf[TAG_PERFORM + 'mae'] /= (n_classes - 1)
                results_perf[TAG_PERFORM + 'rmse'] /= (n_classes - 1)
                results_perf[TAG_PERFORM + 'mse'] /= ((n_classes - 1) ** 2)
                results_perf[TAG_PERFORM + 'avg_mae'] /= (n_classes - 1)
                results_perf[TAG_PERFORM + 'avg_rmse'] /= (n_classes - 1)
                results_perf[TAG_PERFORM + 'avg_mse'] /= ((n_classes - 1) ** 2)

                results_sel = results_sel | results_perf

                l_results.append(results_sel)

    df_results = pd.DataFrame(l_results)

    df_results.insert(0, COL_DATA_OUT, df_results[COL_DATA_IN_1] + '-' + df_results[COL_DATA_IN_2])
    df_results = df_results.drop(columns=[COL_DATA_IN_1, COL_DATA_IN_2])

    # describe statistics
    df_perform_stats = df_results.groupby([COL_DATA_OUT, COL_EXPRM])

    df_perform_avg = df_perform_stats.mean().drop(columns=[COL_SPLIT])
    df_perform_std = df_perform_stats.std().drop(columns=[COL_SPLIT])

    df_perform_med = df_perform_stats.median().drop(columns=[COL_SPLIT])
    df_perform_q1 = df_perform_stats.quantile(0.25).drop(columns=[COL_SPLIT])
    df_perform_q3 = df_perform_stats.quantile(0.75).drop(columns=[COL_SPLIT])
    df_perform_iqr = df_perform_q3 - df_perform_q1

    perform_stats = dict()
    for metric_ in PERFORM_METRICS.keys():
        col_metric_ = TAG_PERFORM + metric_
        
        df_avg_ = df_perform_avg[col_metric_].unstack()
        df_std_ = df_perform_std[col_metric_].unstack()

        df_med_ = df_perform_med[col_metric_].unstack()
        df_iqr_ = df_perform_iqr[col_metric_].unstack()

        perform_stats[metric_] = {'avg': df_avg_, 'std': df_std_, 'med': df_med_, 'iqr': df_iqr_}

    # save to file
    perform_json = dict()
    for metric_, perform_dict_ in perform_stats.items():
        perform_item_ = dict()
        for perform_key_, perform_val_ in perform_dict_.items():
            perform_val_ = perform_val_.astype(object).where(pd.notnull(perform_val_), None)  # replace NaNs for JSON serialization
            perform_item_[perform_key_] = perform_val_.to_dict()
        perform_json[metric_] = perform_item_

    with open(os.path.join(CONFIG.PATH_RESULTS, FILE_RESULTS_PERFORM), 'w') as f_perform:
        json.dump(perform_json, f_perform)

    # conduct Bayesian statistical comparisons
    idx_exprm = sorted(df_results[COL_EXPRM].unique())
    num_exprm = len(idx_exprm)
    num_perform = len(PERFORM_METRICS)
    num_comp = int(num_perform * num_exprm * (num_exprm - 1) / 2)

    bayes_comp = dict()
    with tqdm(total=num_comp) as t_comp:
        for metric_, perform_dict_ in PERFORM_METRICS.items():
            col_metric_ = TAG_PERFORM + metric_
            rope_ = perform_dict_['rope']
            dir_ = perform_dict_['dir']

            prob_win_ = pd.DataFrame(np.nan, index=idx_exprm, columns=idx_exprm)
            prob_draw_ = pd.DataFrame(np.nan, index=idx_exprm, columns=idx_exprm)
            prob_loss_ = pd.DataFrame(np.nan, index=idx_exprm, columns=idx_exprm)

            for i in range(num_exprm):
                exp_i = idx_exprm[i]
                df_res_i = df_results[df_results[COL_EXPRM] == exp_i]

                for j in range(i + 1, num_exprm):
                    exp_j = idx_exprm[j]
                    df_res_j = df_results[df_results[COL_EXPRM] == exp_j]

                    df_res_i_ = df_res_i[[COL_DATA_OUT, COL_SPLIT, col_metric_]]
                    df_res_j_ = df_res_j[[COL_DATA_OUT, COL_SPLIT, col_metric_]]
                    df_res_i_ = df_res_i_.set_index([COL_DATA_OUT, COL_SPLIT])
                    df_res_j_ = df_res_j_.set_index([COL_DATA_OUT, COL_SPLIT])
                    df_res_i_ = df_res_i_.sort_index(level=[COL_DATA_OUT, COL_SPLIT])
                    df_res_j_ = df_res_j_.sort_index(level=[COL_DATA_OUT, COL_SPLIT])
                    df_res_i_ = df_res_i_.unstack()
                    df_res_j_ = df_res_j_.unstack()
                    df_res_i_ = df_res_i_.to_numpy()
                    df_res_j_ = df_res_j_.to_numpy()

                    if np.isnan(df_res_i_).any() or np.isnan(df_res_j_).any():
                        prob_left_, prob_rope_, prob_right_ = np.nan, np.nan, np.nan
                    else:
                        # hierarchical Bayesian test
                        test_res_ = HierarchicalTest(x=df_res_i_, y=df_res_j_, rope=rope_)
                        prob_left_, prob_rope_, prob_right_ = test_res_.probs()
                    t_comp.update()

                    if dir_ == 'max':
                        prob_win_.loc[exp_i, exp_j] = prob_left_
                        prob_win_.loc[exp_j, exp_i] = prob_right_
                        prob_draw_.loc[exp_i, exp_j] = prob_rope_
                        prob_draw_.loc[exp_j, exp_i] = prob_rope_
                        prob_loss_.loc[exp_i, exp_j] = prob_right_
                        prob_loss_.loc[exp_j, exp_i] = prob_left_
                    elif dir_ == 'min':
                        prob_win_.loc[exp_i, exp_j] = prob_right_
                        prob_win_.loc[exp_j, exp_i] = prob_left_
                        prob_draw_.loc[exp_i, exp_j] = prob_rope_
                        prob_draw_.loc[exp_j, exp_i] = prob_rope_
                        prob_loss_.loc[exp_i, exp_j] = prob_left_
                        prob_loss_.loc[exp_j, exp_i] = prob_right_
                    else:
                        raise ValueError

            bayes_comp[metric_] = {'win': prob_win_, 'draw': prob_draw_, 'loss': prob_loss_}

    # save to file
    bayes_json = dict()
    for metric_, bayes_dict_ in bayes_comp.items():
        bayes_item_ = dict()
        for bayes_key_, bayes_val_ in bayes_dict_.items():
            bayes_val_ = bayes_val_.astype(object).where(pd.notnull(bayes_val_), None)  # replace NaNs for JSON serialization
            bayes_item_[bayes_key_] = bayes_val_.to_dict()
        bayes_json[metric_] = bayes_item_

    with open(os.path.join(CONFIG.PATH_RESULTS, FILE_RESULTS_BAYES), 'w') as f_bayes:
        json.dump(bayes_json, f_bayes)

    print('Done!')
