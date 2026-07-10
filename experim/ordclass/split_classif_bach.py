# Author: Fernando García-García <fegarcia@bcamath.org>

import os

import pandas as pd

from sklearn.model_selection import train_test_split

from experim.ordclass import CONFIG

TAG_TASK_TYPE = CONFIG.TAG_CLASSIF
TAG_TARGET_TYPE = CONFIG.TAG_ORDIN

TAG_DATASET = 'BACH'

COL_FILE = 'filename'
COL_TASK = 'class'


def split_simple(df_in):
    df_out = df_in.copy()

    for idx_s in range(CONFIG.NUM_SPLITS):
        id_all = df_in[COL_FILE]
        cl_all = df_in[COL_TASK]

        # stratified split
        id_trval, id_test, cl_trval, cl_test = train_test_split(id_all, cl_all,
                                                                test_size=CONFIG.RATIO_SIZE_TEST,
                                                                stratify=cl_all)
        id_train, id_valid, cl_train, cl_valid = train_test_split(id_trval, cl_trval,
                                                                  test_size=CONFIG.RATIO_SIZE_VALID,
                                                                  stratify=cl_trval)

        # add split info
        df_train = id_train.to_frame()
        df_valid = id_valid.to_frame()
        df_test = id_test.to_frame()

        df_train.insert(0, CONFIG.TAG_SPLIT, CONFIG.TAG_TRAIN)
        df_valid.insert(0, CONFIG.TAG_SPLIT, CONFIG.TAG_VALID)
        df_test.insert(0, CONFIG.TAG_SPLIT, CONFIG.TAG_TEST)

        df_split = pd.concat([df_train, df_valid, df_test])

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
    df_split = split_simple(df_info)

    # save splits
    df_split.to_csv(os.path.join(path_info, CONFIG.FILE_SPLIT), index=False)

    print('Done!')
