import os
import sys
import json
import time

from argparse import ArgumentParser

import numpy as np

from sklearn.base import clone

import tensorflow as tf

from optuna import create_study
from optuna.samplers import TPESampler
from optuna.artifacts import FileSystemArtifactStore, upload_artifact, download_artifact, get_all_artifact_meta

from src.ordclass.ordinal_metrics import avg_f1_score, matthews_corrcoef_score, avg_mae, avg_mse

from src.ordclass.ordinal_layers_tf import OrdinalOneHotEncoder
from src.ordclass.tf_utils_preprocess import PipelineWithLabels
from src.ordclass.tf_utils_augment import AugmentImage2D

from src.ordclass.OrdinalUNet import OrdinalUNet2DEffB5

from experim.ordclass.load_dataset_dl_segment import load_bach
from experim.ordclass.config_experim_dl_segment import EXPERIMENTS  # import experimental scenarios from an auxiliary file


# path and files
PATH_RESULTS = 'results/ordclass'
FILE_REFERENCE = 'DL Segment-BACH_Experim {0:02d}_Scoring {1}'
FILE_SUMMARY = FILE_REFERENCE + '_summary.json'
FILE_Y_TRUE_TEST = FILE_REFERENCE + '_y-true-test.memmap'
FILE_Y_PRED_TEST = FILE_REFERENCE + '_y-pred-test.memmap'
FILE_PROBA_PRED_TEST = FILE_REFERENCE + '_proba-pred-test.memmap'

# Optuna artifacts for storing results
PATH_OPTUNA = os.path.join(PATH_RESULTS, 'optuna')
os.makedirs(PATH_OPTUNA, exist_ok=True)

PATH_OPTUNA_TEMP_FILES = os.path.join(PATH_OPTUNA, 'tmp')
os.makedirs(PATH_OPTUNA_TEMP_FILES, exist_ok=True)

PATH_OPTUNA_ARTIFACTS = os.path.join(PATH_OPTUNA, 'artifacts')
os.makedirs(PATH_OPTUNA_ARTIFACTS, exist_ok=True)

TEMPLATE_Y_TR_VALID = 'y-true-valid_trial-{0}_{1:%Y-%m-%d_%H-%M-%S,%f}.memmap'
TEMPLATE_Y_PR_VALID = 'y-pred-valid_trial-{0}_{1:%Y-%m-%d_%H-%M-%S,%f}.memmap'
TEMPLATE_P_PR_VALID = 'proba-pred-valid_trial-{0}_{1:%Y-%m-%d_%H-%M-%S,%f}.memmap'
TEMPLATE_Y_TR_TEST = 'y-true-test_trial-{0}_{1:%Y-%m-%d_%H-%M-%S,%f}.memmap'
TEMPLATE_Y_PR_TEST = 'y-pred-test_trial-{0}_{1:%Y-%m-%d_%H-%M-%S,%f}.memmap'
TEMPLATE_P_PR_TEST = 'proba-pred-test_trial-{0}_{1:%Y-%m-%d_%H-%M-%S,%f}.memmap'

TAG_Y_TR = 'artifact_id_y_tr'
TAG_Y_PR = 'artifact_id_y_pr'
TAG_P_PR = 'artifact_id_p_pr'

# scoring and direction for hyperparameter tuning
SCORING = ['avgF1', 'avgMAE']
DIRECTION = ['maximize', 'minimize']

# problem specification
CLASS_COUNTS_BACH = np.asarray([92_316_718, 34_545_954, 5_062_670, 1_052_410, 40_648_808], dtype=np.uint64)  # pre-computed
CLASS_PRIORS_BACH = np.asarray([0.531697, 0.198967, 0.02915838, 0.00606134, 0.2341163], dtype=np.float32)  # pre-computed
BUFFER_SIZE_BACH = 835  # the size of the training set

# algorithmic choices
BATCH_SIZE_BACH = 32
N_EPOCHS_BACH = 30

# tuning choices
N_STARTUP_BACH = 8
N_TRIALS_BACH = 16

# data formatting
IMG_HEIGHT_B5 = 456
IMG_WIDTH_B5 = 456
IMG_CHANNELS_RGB = 3

# data augmentation
DISPLACE_RANGE = 0.125  # ratio
ROTATION_RANGE = 15.0  # degrees
SHEAR_RANGE = 10.0  # degrees
ZOOM_RANGE = 0.20  # 1 +/- ratio
HEIGHT_FLIP = False
WIDTH_FLIP = True
SHARPNESS_RANGE = 0.25  # 1 +/- ratio
BRIGHTNESS_RANGE = 0.25  # 1 +/- ratio
CONTRAST_RANGE = 0.25  # 1 +/- ratio
COLOR_RANGE = 0.25  # 1 +/- ratio

RANDOM_SEED = None


if __name__ == '__main__':
    # read extra arguments provided to the script
    parser = ArgumentParser()
    parser.add_argument('--idx')

    args = parser.parse_args()
    idx = int(args.idx)
    if idx < 0:
        sys.exit()

    # select experiment -- i.e. algorithmic set-up scenario
    idx_exper = idx // len(SCORING)
    if idx_exper >= len(EXPERIMENTS):
        sys.exit()
    experiment = EXPERIMENTS[idx_exper]

    # select scoring metric for hyperparameter tuning
    idx_score = idx % len(SCORING)
    scoring = SCORING[idx_score]
    direction = DIRECTION[idx_score]

    # hyperparameters: fixed and tunable
    hyparams_fixed = dict()
    hyparams_tunable = []
    for hyparam in experiment['hyperparams']:
        if hyparam['fixed']:
            hyparams_fixed[hyparam['name']] = hyparam['value']
        else:
            hyparams_tunable.append(hyparam)

    hyparams_fixed['input_height'] = IMG_HEIGHT_B5
    hyparams_fixed['input_width'] = IMG_WIDTH_B5
    hyparams_fixed['input_channels'] = IMG_CHANNELS_RGB

    hyparams_fixed['epochs'] = N_EPOCHS_BACH

    # load data
    data_info = load_bach()
    data_name = data_info['name']
    n_classes = data_info['n_classes']
    X_train, X_valid, X_test = data_info['data_train'], data_info['data_valid'], data_info['data_test']
    n_train, n_valid, n_test = data_info['n_train'], data_info['n_valid'], data_info['n_test']

    y_shape = list(data_info['y_shape'])
    proba_shape = y_shape[:-1] + [n_classes]

    # pre-initialize the segmentation model
    model = OrdinalUNet2DEffB5(n_classes=n_classes)

    # get class counts and class priors
    # class_counts = model._get_class_counts(X_train, ohe=False)
    class_counts = CLASS_COUNTS_BACH  # pre-computed for speed
    hyparams_fixed['class_counts'] = class_counts
    # class_priors = class_counts / np.sum(class_counts)
    # class_priors = model._get_class_priors(X_train, ohe=False)
    class_priors = CLASS_PRIORS_BACH  # pre-computed for speed
    hyparams_fixed['class_priors'] = class_priors

    # configure the pre-processing steps
    ordinal_encoder = OrdinalOneHotEncoder(n_classes=n_classes)

    augmenter2d = AugmentImage2D(displace_range=DISPLACE_RANGE,
                                 rotation_range=ROTATION_RANGE, shear_range=SHEAR_RANGE,
                                 zoom_range=ZOOM_RANGE,
                                 height_flip=HEIGHT_FLIP, width_flip=WIDTH_FLIP,
                                 sharpness_range=SHARPNESS_RANGE, brightness_range=BRIGHTNESS_RANGE,
                                 contrast_range=CONTRAST_RANGE, color_range=COLOR_RANGE,
                                 seed=RANDOM_SEED)

    out_type = hyparams_fixed['out_type']
    if out_type == 'regression':
        l_preproc_steps_train = [augmenter2d]
        l_preproc_steps_valid = []
        l_preproc_steps_test = []
    else:
        l_preproc_steps_train = [ordinal_encoder, augmenter2d]
        l_preproc_steps_valid = [ordinal_encoder]
        l_preproc_steps_test = [ordinal_encoder]
    preprocessor_train = PipelineWithLabels(steps=l_preproc_steps_train)
    preprocessor_valid = PipelineWithLabels(steps=l_preproc_steps_valid)
    preprocessor_test = PipelineWithLabels(steps=l_preproc_steps_test)

    # apply pre-processing
    hyparams_fixed['batch_size'] = BATCH_SIZE_BACH

    X_train = (X_train
               .cache()
               .shuffle(BUFFER_SIZE_BACH, reshuffle_each_iteration=True)
               .batch(BATCH_SIZE_BACH)
               .map(preprocessor_train)
               .prefetch(buffer_size=tf.data.AUTOTUNE))

    X_valid = (X_valid
               .cache()
               .batch(BATCH_SIZE_BACH)
               .map(preprocessor_valid)
               .prefetch(buffer_size=tf.data.AUTOTUNE))

    X_test = (X_test
              .cache()
              .batch(BATCH_SIZE_BACH)
              .map(preprocessor_test)
              .prefetch(buffer_size=tf.data.AUTOTUNE))

    # Optuna artifacts for storing results
    artifact_store = FileSystemArtifactStore(base_path=PATH_OPTUNA_ARTIFACTS)

    # Optuna optimization objective
    def optim_objective(trial):
        # hyperparameters: to tune, search space
        hyparams_tune = dict()
        for hyparam in hyparams_tunable:
            if hyparam['type'] == 'categorical':
                hyparams_tune[hyparam['name']] = trial.suggest_categorical(name=hyparam['name'],
                                                                           choices=hyparam['choices'])
            elif hyparam['type'] == 'int':
                hyparams_tune[hyparam['name']] = trial.suggest_int(name=hyparam['name'],
                                                                   low=hyparam['low'], high=hyparam['high'],
                                                                   log=hyparam['log'])
            elif hyparam['type'] == 'float':
                hyparams_tune[hyparam['name']] = trial.suggest_float(name=hyparam['name'],
                                                                     low=hyparam['low'], high=hyparam['high'],
                                                                     log=hyparam['log'])
            else:
                raise ValueError

        # fit on the training set, accounting also for the validation set -- just for early stopping
        model_ = clone(model)
        model_.set_params(**hyparams_fixed)
        model_.set_params(**hyparams_tune)

        model_.fit_valid(X_train, X_valid)

        # batched prediction on the validation set
        y_shape_valid = tuple([n_valid] + y_shape)
        # proba_shape_valid = tuple([n_valid] + proba_shape)

        file_y_true_valid = TEMPLATE_Y_TR_VALID.format(trial.number, trial.datetime_start)
        file_y_pred_valid = TEMPLATE_Y_PR_VALID.format(trial.number, trial.datetime_start)
        # if out_type != 'regression':
        #     file_proba_pred_valid = TEMPLATE_P_PR_VALID.format(trial.number, trial.datetime_start)

        y_true_valid = np.memmap(os.path.join(PATH_OPTUNA_TEMP_FILES, file_y_true_valid),
                                 shape=y_shape_valid, dtype=np.uint8, mode='w+')
        y_pred_valid = np.memmap(os.path.join(PATH_OPTUNA_TEMP_FILES, file_y_pred_valid),
                                 shape=y_shape_valid, dtype=np.uint8, mode='w+')
        # if out_type != 'regression':
        #     proba_pred_valid = np.memmap(os.path.join(PATH_OPTUNA_TEMP_FILES, file_proba_pred_valid),
        #                                  shape=proba_shape_valid, dtype=np.float32, mode='w+')

        count_valid = 0
        for X_valid_, y_true_valid_ in X_valid:
            batch_valid_ = int(tf.shape(X_valid_).numpy()[0])

            if out_type != 'regression':
                y_true_valid_ = tf.math.argmax(y_true_valid_, axis=-1)  # revert one-hot encoding
                y_true_valid_ = tf.expand_dims(y_true_valid_, axis=-1)
            y_true_valid_ = tf.cast(y_true_valid_, dtype=tf.uint8)
            y_true_valid_ = y_true_valid_.numpy()
            y_true_valid[count_valid:(count_valid + batch_valid_), ...] = y_true_valid_  # save to disk (memmap)

            y_pred_valid_ = model_._predict(X_valid_)
            y_pred_valid_ = y_pred_valid_.numpy()
            y_pred_valid[count_valid:(count_valid + batch_valid_), ...] = y_pred_valid_  # save to disk (memmap)

            # if out_type != 'regression':
            #     proba_pred_valid_ = model_._predict_proba(X_valid_)
            #     proba_pred_valid_ = proba_pred_valid_.numpy()
            #     proba_pred_valid[count_valid:(count_valid + batch_valid_), ...] = proba_pred_valid_  # save to disk (memmap)

            count_valid += batch_valid_

        # force saving validation data to disk (memmap)
        y_true_valid.flush()
        y_pred_valid.flush()
        # if out_type != 'regression':
        #     proba_pred_valid.flush()

        # evaluate performance on the validation set
        y_true_valid = y_true_valid.flatten()
        y_pred_valid = y_pred_valid.flatten()
        # if out_type != 'regression':
        #     proba_pred_valid = np.reshape(proba_pred_valid, (-1, n_classes))

        if scoring == 'avgF1':
            score_valid = avg_f1_score(y_true_valid, y_pred_valid, macro_average='mean', n_classes=n_classes)
        elif scoring == 'MCC':
            score_valid = matthews_corrcoef_score(y_true_valid, y_pred_valid, n_classes=n_classes)
        elif scoring == 'avgMAE':
            score_valid = avg_mae(y_true_valid, y_pred_valid, macro_average='mean', n_classes=n_classes)
        elif scoring == 'avgMSE':
            score_valid = avg_mse(y_true_valid, y_pred_valid, macro_average='mean', n_classes=n_classes)
        else:
            raise RuntimeError

        # delete temporary validation files
        os.remove(os.path.join(PATH_OPTUNA_TEMP_FILES,file_y_true_valid))
        os.remove(os.path.join(PATH_OPTUNA_TEMP_FILES, file_y_pred_valid))
        # if out_type != 'regression':
        #     os.remove(os.path.join(PATH_OPTUNA_TEMP_FILES, file_proba_pred_valid))

        # batched prediction on the test set
        y_shape_test = tuple([n_test] + y_shape)
        proba_shape_test = tuple([n_test] + proba_shape)

        file_y_true_test = TEMPLATE_Y_TR_TEST.format(trial.number, trial.datetime_start)
        file_y_pred_test = TEMPLATE_Y_PR_TEST.format(trial.number, trial.datetime_start)
        if out_type != 'regression':
            file_proba_pred_test = TEMPLATE_P_PR_TEST.format(trial.number, trial.datetime_start)

        y_true_test = np.memmap(os.path.join(PATH_OPTUNA_TEMP_FILES, file_y_true_test),
                                shape=y_shape_test, dtype=np.uint8, mode='w+')
        y_pred_test = np.memmap(os.path.join(PATH_OPTUNA_TEMP_FILES, file_y_pred_test),
                                shape=y_shape_test, dtype=np.uint8, mode='w+')
        if out_type != 'regression':
            proba_pred_test = np.memmap(os.path.join(PATH_OPTUNA_TEMP_FILES, file_proba_pred_test),
                                        shape=proba_shape_test, dtype=np.float32, mode='w+')

        count_test = 0
        for X_test_, y_true_test_ in X_test:
            batch_test_ = int(tf.shape(X_test_).numpy()[0])

            if out_type != 'regression':
                y_true_test_ = tf.math.argmax(y_true_test_, axis=-1)  # revert one-hot encoding
                y_true_test_ = tf.expand_dims(y_true_test_, axis=-1)
            y_true_test_ = tf.cast(y_true_test_, dtype=tf.uint8)
            y_true_test_ = y_true_test_.numpy()
            y_true_test[count_test:(count_test + batch_test_), ...] = y_true_test_  # save to disk (memmap)

            y_pred_test_ = model_._predict(X_test_)
            y_pred_test_ = y_pred_test_.numpy()
            y_pred_test[count_test:(count_test + batch_test_), ...] = y_pred_test_  # save to disk (memmap)

            if out_type != 'regression':
                proba_pred_test_ = model_._predict_proba(X_test_)
                proba_pred_test_ = proba_pred_test_.numpy()
                proba_pred_test[count_test:(count_test + batch_test_), ...] = proba_pred_test_  # save to disk (memmap)

            count_test += batch_test_

        # force saving test data to disk (memmap)
        y_true_test.flush()
        y_pred_test.flush()
        if out_type != 'regression':
            proba_pred_test.flush()

        # save the results on the test set as Optuna artifacts
        id_y_tr = upload_artifact(artifact_store=artifact_store,
                                  file_path=os.path.join(PATH_OPTUNA_TEMP_FILES, file_y_true_test),
                                  study_or_trial=trial)  # returns the artifact ID
        id_y_pr = upload_artifact(artifact_store=artifact_store,
                                  file_path=os.path.join(PATH_OPTUNA_TEMP_FILES, file_y_pred_test),
                                  study_or_trial=trial)  # returns the artifact ID
        if out_type != 'regression':
            id_p_tr = upload_artifact(artifact_store=artifact_store,
                                      file_path=os.path.join(PATH_OPTUNA_TEMP_FILES, file_proba_pred_test),
                                      study_or_trial=trial)  # returns the artifact ID

        trial.set_user_attr(TAG_Y_TR, id_y_tr)  # save the ID in RDB so that it can be referenced later
        trial.set_user_attr(TAG_Y_PR, id_y_pr)  # save the ID in RDB so that it can be referenced later
        if out_type != 'regression':
            trial.set_user_attr(TAG_P_PR, id_p_tr)  # save the ID in RDB so that it can be referenced later

        # clean-up temporary files, once uploaded as artifacts
        os.remove(os.path.join(PATH_OPTUNA_TEMP_FILES, file_y_true_test))
        os.remove(os.path.join(PATH_OPTUNA_TEMP_FILES, file_y_pred_test))
        if out_type != 'regression':
            os.remove(os.path.join(PATH_OPTUNA_TEMP_FILES, file_proba_pred_test))

        return score_valid

    # optimize with Optuna
    optim_study = create_study(sampler=TPESampler(n_startup_trials=N_STARTUP_BACH),
                               direction=direction)

    # optimize and measure the time involved
    time_start_optim = time.process_time()

    optim_study.optimize(optim_objective, n_trials=N_TRIALS_BACH, gc_after_trial=True)

    time_stop_optim = time.process_time()
    time_optim = time_stop_optim - time_start_optim

    score_optim_best = optim_study.best_value
    hyparams_best = optim_study.best_trial.params

    # for the best trial, recover its results on the test set
    best_trial = optim_study.best_trial
    best_id_y_tr = best_trial.user_attrs[TAG_Y_TR]
    best_id_y_pr = best_trial.user_attrs[TAG_Y_PR]
    if out_type != 'regression':
        best_id_p_pr = best_trial.user_attrs[TAG_P_PR]

    file_y_true_test = FILE_Y_TRUE_TEST.format(idx_exper, scoring)
    file_y_pred_test = FILE_Y_PRED_TEST.format(idx_exper, scoring)
    if out_type != 'regression':
        file_proba_pred_test = FILE_PROBA_PRED_TEST.format(idx_exper, scoring)

    download_artifact(artifact_store=artifact_store, artifact_id=best_id_y_tr,
                      file_path=os.path.join(PATH_RESULTS, file_y_true_test))
    download_artifact(artifact_store=artifact_store, artifact_id=best_id_y_pr,
                      file_path=os.path.join(PATH_RESULTS, file_y_pred_test))
    if out_type != 'regression':
        download_artifact(artifact_store=artifact_store, artifact_id=best_id_p_pr,
                          file_path=os.path.join(PATH_RESULTS, file_proba_pred_test))

    # remove Optuna artifacts
    # NOTE: ``artifact_store.remove`` is discouraged to use because it is an internal feature
    storage = optim_study._storage
    for trial in optim_study.trials:
        for artifact_meta in get_all_artifact_meta(trial, storage=storage):
            # for each trial, remove the artifacts uploaded to ``base_path``
            artifact_store.remove(artifact_meta.artifact_id)

    for artifact_meta in get_all_artifact_meta(optim_study):
        # remove the artifacts uploaded to ``base_path``
        artifact_store.remove(artifact_meta.artifact_id)

    # evaluate performance on the test set
    y_shape_test = tuple([n_test] + y_shape)
    proba_shape_test = tuple([n_test] + proba_shape)

    y_true_test = np.memmap(os.path.join(PATH_RESULTS, file_y_true_test),
                            shape=y_shape_test, mode='r', dtype=np.uint8)
    y_pred_test = np.memmap(os.path.join(PATH_RESULTS, file_y_pred_test),
                            shape=y_shape_test, mode='r', dtype=np.uint8)
    # if out_type != 'regression':
    #     proba_pred_test = np.memmap(os.path.join(PATH_RESULTS, file_proba_pred_test),
    #                                 shape=proba_shape_test, mode='r', dtype=np.float32)

    y_true_test = y_true_test.flatten()
    y_pred_test = y_pred_test.flatten()
    # if out_type != 'regression':
    #     proba_pred_test = np.reshape(proba_pred_test, (-1, n_classes))

    if scoring == 'avgF1':
        score_test = avg_f1_score(y_true_test, y_pred_test, macro_average='mean', n_classes=n_classes)
    elif scoring == 'MCC':
        score_test = matthews_corrcoef_score(y_true_test, y_pred_test, n_classes=n_classes)
    elif scoring == 'avgMAE':
        score_test = avg_mae(y_true_test, y_pred_test, macro_average='mean', n_classes=n_classes)
    elif scoring == 'avgMSE':
        score_test = avg_mse(y_true_test, y_pred_test, macro_average='mean', n_classes=n_classes)
    else:
        raise RuntimeError

    # save the set-up and the summary of results
    # make everything JSON serializable
    hyparams_fixed_ = json.dumps(hyparams_fixed, default=lambda x: x.tolist())
    hyparams_tunable_ = json.dumps(hyparams_tunable, default=lambda x: x.tolist())
    hyparams_best_ = json.dumps(hyparams_best, default=lambda x: x.tolist())

    # remove irrelevant trials information
    trials_info = optim_study.trials_dataframe()
    names_drop = ['datetime', 'user_attrs', 'system_attrs']
    cols_drop = []
    for col_name in trials_info.columns:
        for col_drop in names_drop:
            if col_name.startswith(col_drop):
                cols_drop.append(col_name)
                break
    trials_info = trials_info.drop(columns=cols_drop)
    trials_info = json.loads(trials_info.to_json())

    dict_summary = {'dataset': data_name,
                    'n_classes': n_classes,
                    'n_train': n_train, 'n_valid': n_valid, 'n_test': n_test,
                    'y_shape': y_shape, 'proba_shape': proba_shape,
                    'scoring': scoring, 'direction': direction,
                    'hyparams_fixed': hyparams_fixed_,
                    'hyparams_tunable': hyparams_tunable_,
                    'hyparams_best': hyparams_best_,
                    'n_trials': N_TRIALS_BACH, 'n_startup': N_STARTUP_BACH,
                    'trials_info': trials_info,
                    'time_optim': time_optim,
                    'score_optim_best': score_optim_best,
                    'score_test': score_test}

    file_summary = FILE_SUMMARY.format(idx_exper, scoring)
    with open(os.path.join(PATH_RESULTS, file_summary), 'w') as f_summ:
        json.dump(dict_summary, f_summ)

    print('Done!')
