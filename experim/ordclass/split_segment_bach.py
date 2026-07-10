# Author: Fernando García-García <fegarcia@bcamath.org>

import os
import itertools

import numpy as np
import pandas as pd
import scipy as sp

from experim.ordclass import CONFIG

TAG_TASK_TYPE = CONFIG.TAG_SEGMENT
TAG_TARGET_TYPE = CONFIG.TAG_ORDIN

TAG_DATASET = 'BACH'

COL_FILE = 'filename_in'
COL_TASK = 'filename_out'

N_CLASSES = 5
COL_CLASS_COUNTS = ['class_count_{}'.format(c) for c in range(N_CLASSES)]
COL_CLASS_RATIOS = ['class_ratio_{}'.format(c) for c in range(N_CLASSES)]

DIMS = ['height', 'width']


def split_with_constraints(df_in):
    df_out = df_in.copy()

    num_total = len(df_in.index)
    num_train = int(np.round((1.0 - CONFIG.RATIO_SIZE_TEST) * (1.0 - CONFIG.RATIO_SIZE_VALID) * num_total))
    num_valid = int(np.round((1.0 - CONFIG.RATIO_SIZE_TEST) * CONFIG.RATIO_SIZE_VALID * num_total))
    num_test = int(np.round(CONFIG.RATIO_SIZE_TEST * num_total))

    num_diff = num_total - (num_train + num_valid + num_test)
    if num_diff != 0:
        # adjust for rounding effects
        num_train += num_diff

    class_counts = df_in[COL_CLASS_COUNTS].sum()

    class_ratios = class_counts.astype(np.float)
    class_ratios /= class_ratios.sum()
    class_ratios.index = COL_CLASS_RATIOS

    # generate all possible combinations of training-validation-test splits
    idx_total = list(range(num_total))
    l_comb = []
    for idx_train in itertools.combinations(idx_total, r=num_train):  # training versus remaining (validation & test)
        idx_train = list(idx_train)
        idx_remain = [idx_r for idx_r in idx_total if idx_r not in idx_train]
        for idx_valid in itertools.combinations(idx_remain, r=num_valid):  # validation versus test
            idx_valid = list(idx_valid)
            idx_test = [idx_t for idx_t in idx_remain if idx_t not in idx_valid]

            df_train = df_in.iloc[idx_train]
            df_valid = df_in.iloc[idx_valid]
            df_test = df_in.iloc[idx_test]

            df_train.insert(0, CONFIG.TAG_SPLIT, CONFIG.TAG_TRAIN)
            df_valid.insert(0, CONFIG.TAG_SPLIT, CONFIG.TAG_VALID)
            df_test.insert(0, CONFIG.TAG_SPLIT, CONFIG.TAG_TEST)

            df_split = pd.concat([df_train, df_valid, df_test])

            # check the distribution of counts
            class_splits = df_split[[CONFIG.TAG_SPLIT] + COL_CLASS_COUNTS].groupby(CONFIG.TAG_SPLIT).sum()
            class_splits = class_splits.reindex(CONFIG.LIST_SPLITS)

            stats = {}
            for class_c_ in COL_CLASS_COUNTS:
                class_count_ = class_counts[class_c_]

                n_exp_train = (1.0 - CONFIG.RATIO_SIZE_TEST) * (1.0 - CONFIG.RATIO_SIZE_VALID) * class_count_
                n_exp_valid = (1.0 - CONFIG.RATIO_SIZE_TEST) * CONFIG.RATIO_SIZE_VALID * class_count_
                n_exp_test = CONFIG.RATIO_SIZE_TEST * class_count_

                freq_obs = class_splits[class_c_].to_numpy()
                freq_exp = np.asarray([n_exp_train, n_exp_valid, n_exp_test])

                chi_sq_test_freq = sp.stats.chisquare(f_obs=freq_obs, f_exp=freq_exp)
                chi_sq_omega_sq = chi_sq_test_freq.statistic / class_count_
                stats[class_c_ + CONFIG.TAG_SEP + 'Omega2'] = chi_sq_omega_sq

            # check the distribution of frequency ratios
            for class_r_ in COL_CLASS_RATIOS:
                idx_train_ = (df_split[CONFIG.TAG_SPLIT] == CONFIG.TAG_TRAIN)
                idx_valid_ = (df_split[CONFIG.TAG_SPLIT] == CONFIG.TAG_VALID)
                idx_test_ = (df_split[CONFIG.TAG_SPLIT] == CONFIG.TAG_TEST)

                data_train = df_split.loc[idx_train_, class_r_].to_numpy()
                data_valid = df_split.loc[idx_valid_, class_r_].to_numpy()
                data_test = df_split.loc[idx_test_, class_r_].to_numpy()

                num_splits = len(CONFIG.LIST_SPLITS)

                kruskal_test_extra = sp.stats.kruskal(data_train, data_valid, data_test)
                kruskal_eta_sq = (kruskal_test_extra.statistic - num_splits + 1) / (num_total - num_splits)
                stats[class_r_ + CONFIG.TAG_SEP + 'Eta2'] = kruskal_eta_sq

            comb_ = {'train': idx_train, 'valid': idx_valid, 'test': idx_test}
            comb_.update(stats)
            l_comb.append(comb_)

    df_comb = pd.DataFrame(l_comb)

    # check effect sizes statistics per combination
    # heuristic procedure: starting from the most minority class, select those cases with effect sizes below the median
    df_stats_all = df_comb[list(stats.keys())]
    df_stats_med = df_stats_all.median()

    class_c_sort = class_counts.sort_values(ascending=True)
    class_r_sort = class_ratios.sort_values(ascending=True)

    sel_comb = pd.Series(True, index=df_stats_all.index)
    for class_c_, class_r_ in zip(class_c_sort.keys(), class_r_sort.keys()):
        class_c_stat_ = class_c_ + CONFIG.TAG_SEP + 'Omega2'
        class_r_stat_ = class_r_ + CONFIG.TAG_SEP + 'Eta2'
        sel_c_ = (df_stats_all[class_c_stat_] < df_stats_med[class_c_stat_])
        sel_r_ = (df_stats_all[class_r_stat_] < df_stats_med[class_r_stat_])
        sel_ = sel_c_ & sel_r_

        sel_next = sel_comb & sel_
        if sel_next.sum() >= CONFIG.NUM_SPLITS:
            sel_comb = sel_next
        else:
            break

    df_comb_sel = df_comb[sel_comb]

    # final sort: heuristic, ascending effect size in ratios with respect to the minority class
    comb_ranker = class_r_sort.index[0] + CONFIG.TAG_SEP + 'Eta2'
    df_comb_sel = df_comb_sel.sort_values(by=comb_ranker, ascending=True).iloc[:CONFIG.NUM_SPLITS, :]

    # final shuffle
    df_comb_sel = df_comb_sel.sample(frac=1.0, replace=False)
    df_comb_sel = df_comb_sel.reset_index(drop=True)

    # add split info
    for idx_comb, row_comb in df_comb_sel.iterrows():
        idx_train = row_comb[CONFIG.TAG_TRAIN]
        idx_valid = row_comb[CONFIG.TAG_VALID]
        idx_test = row_comb[CONFIG.TAG_TEST]

        split_str = COL_TASK + CONFIG.TAG_SEP + CONFIG.TAG_SPLIT + str(idx_comb)

        f_split = lambda x: CONFIG.TAG_TRAIN if (x in idx_train) else (CONFIG.TAG_VALID if (x in idx_valid) else CONFIG.TAG_TEST)
        df_out[split_str] = pd.Series(data=df_out.index.map(f_split))

    return df_out


if __name__ == '__main__':
    # load information
    path_info = os.path.join(CONFIG.PATH_DATA, TAG_TASK_TYPE, TAG_TARGET_TYPE, TAG_DATASET)
    file_info = os.path.join(path_info, CONFIG.FILE_INFO)
    if not os.path.isfile(file_info):
        raise RuntimeError

    df_in = pd.read_csv(file_info)

    # perform split
    df_split = split_with_constraints(df_in)

    # save splits
    df_split.to_csv(os.path.join(path_info, CONFIG.FILE_SPLIT), index=False)

    print('Done!')
