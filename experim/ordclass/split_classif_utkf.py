# Author: Fernando García-García <fegarcia@bcamath.org>

import os

import numpy as np
import pandas as pd
import scipy as sp

from sklearn.model_selection import train_test_split

from experim.ordclass import CONFIG

TAG_TASK_TYPE = CONFIG.TAG_CLASSIF
TAG_TARGET_TYPE = CONFIG.TAG_DISCR

TAG_DATASET = 'UTKF'

COL_FILE = 'filename'
COL_TASK = 'class_thr'

COL_EXTRA = ['sex', 'ethnicity']


def split_with_constraints(df_in):
    df_out = df_in.copy()

    # we must ensure that all effect sizes regarding extra variables are minimal
    for idx_s in range(CONFIG.NUM_SPLITS):
        converged = False
        while not converged:
            skip_rest = False

            id_all = df_in[COL_FILE]
            cl_all = df_in[COL_TASK]

            # stratified split
            id_trval, id_test, cl_trval, cl_test = train_test_split(id_all, cl_all,
                                                                    test_size=CONFIG.RATIO_SIZE_TEST,
                                                                    stratify=cl_all)
            id_train, id_valid, cl_train, cl_valid = train_test_split(id_trval, cl_trval,
                                                                      test_size=CONFIG.RATIO_SIZE_VALID,
                                                                      stratify=cl_trval)

            df_train = df_in[df_in[COL_FILE].isin(id_train)]
            df_valid = df_in[df_in[COL_FILE].isin(id_valid)]
            df_test = df_in[df_in[COL_FILE].isin(id_test)]

            df_train.insert(0, CONFIG.TAG_SPLIT, CONFIG.TAG_TRAIN)
            df_valid.insert(0, CONFIG.TAG_SPLIT, CONFIG.TAG_VALID)
            df_test.insert(0, CONFIG.TAG_SPLIT, CONFIG.TAG_TEST)

            df_split = pd.concat([df_train, df_valid, df_test])
            num_total = len(df_split.index)

            # check the distribution of extra variables (categorical)
            for extra in COL_EXTRA:
                conting_mtx = pd.crosstab(df_split[CONFIG.TAG_SPLIT], df_split[extra])

                chi_sq_test_extra = sp.stats.chi2_contingency(conting_mtx)
                chi_sq_dof = min(conting_mtx.shape) - 1
                chi_sq_cramer_v = np.sqrt(chi_sq_test_extra.statistic / (num_total * chi_sq_dof))
                if chi_sq_cramer_v >= CONFIG.THRESH_CRAMER_V[chi_sq_dof]['S']:
                    # effect size unacceptably high
                    skip_rest = True
                    break

            if not skip_rest:
                # acceptable split
                converged = True

        # add split info
        df_split = df_split[[COL_FILE, CONFIG.TAG_SPLIT]]
        df_split = df_split.rename(columns={CONFIG.TAG_SPLIT: COL_TASK + CONFIG.TAG_SEP + CONFIG.TAG_SPLIT + str(idx_s)})

        df_out = pd.merge(df_out, df_split, how='inner', on=COL_FILE)

    return df_out


if __name__ == '__main__':
    # load information
    path_info = os.path.join(CONFIG.PATH_DATA, TAG_TASK_TYPE, TAG_TARGET_TYPE, TAG_DATASET)
    file_info = os.path.join(path_info, CONFIG.FILE_INFO)
    if not os.path.isfile(file_info):
        raise RuntimeError

    df_info = pd.read_csv(file_info)

    # perform split
    df_split = split_with_constraints(df_info)

    # save splits
    df_split.to_csv(os.path.join(path_info, CONFIG.FILE_SPLIT), index=False)

    print('Done!')
