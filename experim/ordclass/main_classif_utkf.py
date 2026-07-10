# Author: Fernando García-García <fegarcia@bcamath.org>

import os
import json

from argparse import ArgumentParser

import numpy as np

import tensorflow as tf

from experim.ordclass import CONFIG
from experim.ordclass import HYPARAMS
from experim.ordclass import EXPERIM

from src.ordclass.OrdinalOHE import OrdinalOHE
from src.ordclass.TFPipeline import Pipeline
from src.ordclass.TFAugment import Augment2D

from src.ordclass.OrdinalCNN import OrdinalCNN2DEffB0

from experim.ordclass.evaluate import get_performance

from experim.ordclass.load_classif_utkf import load

TAG_TASK_TYPE = CONFIG.TAG_CLASSIF
TAG_TARGET_TYPE = CONFIG.TAG_DISCR

TAG_DATASET = 'UTKF'

TASK = 'class_thr'


if __name__ == '__main__':
    # read extra arguments provided to the script
    parser = ArgumentParser()
    parser.add_argument('--idx')

    args = parser.parse_args()
    idx = int(args.idx)
    if idx < 0:
        raise RuntimeError

    # select experiment
    idx_exprm = idx // CONFIG.NUM_SPLITS
    if idx_exprm >= EXPERIM.NUM_EXPERIM:
        raise RuntimeError
    experim = EXPERIM.EXPERIM[idx_exprm]

    # select split
    idx_split = idx % CONFIG.NUM_SPLITS

    # load dataset
    dataset = load(idx_split=idx_split)

    n_classes = dataset[CONFIG.TAG_N_CLASSES]

    X_train, X_valid, X_test = dataset[CONFIG.TAG_TRAIN], dataset[CONFIG.TAG_VALID], dataset[CONFIG.TAG_TEST]

    n_train = int(tf.data.experimental.cardinality(X_train).numpy())
    n_valid = int(tf.data.experimental.cardinality(X_valid).numpy())
    n_test = int(tf.data.experimental.cardinality(X_test).numpy())
    print('{} cardinality - Train: {} | Valid: {} | Test: {}'.format(TAG_DATASET, n_train, n_valid, n_test), flush=True)

    # hyperparameters
    hyparams = HYPARAMS.HYPARAMS_COMMON.copy()

    hyparams_ = HYPARAMS.HYPARAMS_CLASSIF.copy()
    hyparams.update(hyparams_)

    hyparams_ = HYPARAMS.HYPARAMS_CLASSIF_2D.copy()
    hyparams.update(hyparams_)

    h, w, ch = dataset[CONFIG.TAG_SHAPE_X]
    hyparams['input_height'] = h
    hyparams['input_width'] = w
    hyparams['input_channels'] = ch

    hyparams_ = HYPARAMS.HYPARAMS_DATASET[TAG_TASK_TYPE][TAG_TARGET_TYPE][TAG_DATASET].copy()
    hyparams.update(hyparams_)

    hyparams.update(experim)

    out_type = hyparams['out_type']
    batch_size = hyparams['batch_size']

    # initialize the model
    model = OrdinalCNN2DEffB0(n_classes=n_classes)
    model.set_params(**hyparams)

    # configure the pre-processing steps
    ordinal_ohe = OrdinalOHE(n_classes=n_classes)

    augmenter_train = Augment2D(displace_range=CONFIG.DISPLACE_RANGE,
                                rotation_range=CONFIG.ROTATION_RANGE,
                                shear_range=CONFIG.SHEAR_RANGE,
                                zoom_range=CONFIG.ZOOM_RANGE,
                                height_flip=CONFIG.HEIGHT_FLIP,
                                width_flip=CONFIG.WIDTH_FLIP,
                                sharpness_range=CONFIG.SHARPNESS_RANGE,
                                brightness_range=CONFIG.BRIGHTNESS_RANGE,
                                contrast_range=CONFIG.CONTRAST_RANGE,
                                color_range=CONFIG.COLOR_RANGE,
                                seed=None)
    augmenter_valid = None
    augmenter_test = None

    if out_type == 'regress':
        l_preproc_steps_train = [augmenter_train]
        l_preproc_steps_valid = []
        l_preproc_steps_test = []
    else:
        l_preproc_steps_train = [augmenter_train, ordinal_ohe]
        l_preproc_steps_valid = [ordinal_ohe]
        l_preproc_steps_test = [ordinal_ohe]
    preprocessor_train = Pipeline(steps=l_preproc_steps_train)
    preprocessor_valid = Pipeline(steps=l_preproc_steps_valid)
    preprocessor_test = Pipeline(steps=l_preproc_steps_test)

    # apply pre-processing
    X_train = (X_train
               .cache()
               .shuffle(n_train, reshuffle_each_iteration=True)  # perfect shuffling
               .batch(batch_size)
               .map(preprocessor_train)
               .prefetch(buffer_size=tf.data.AUTOTUNE))

    X_valid = (X_valid
               .cache()
               .batch(batch_size)
               .map(preprocessor_valid)
               .prefetch(buffer_size=tf.data.AUTOTUNE))

    X_test = (X_test
              .cache()
              .batch(batch_size)
              .map(preprocessor_test)
              .prefetch(buffer_size=tf.data.AUTOTUNE))

    # fit with validation data also: for early stopping, when generalization performance degrades
    model.fit_valid(X_train, X_valid)

    # recover information about the training procedure
    class_counts_train = model.class_counts_

    history = model.history_
    losses_train = history.history['loss']
    losses_valid = history.history['val_loss']

    n_epochs_max = hyparams['epochs']
    n_epochs_run = len(losses_valid)
    n_epochs_best = int(np.argmin(losses_valid)) + 1
    early_stop = (n_epochs_run < n_epochs_max)

    # batched prediction on the test set
    y_true_test = np.zeros(shape=(n_test,), dtype=np.uint8)
    y_pred_test = np.zeros(shape=(n_test,), dtype=np.uint8)
    if out_type == 'regress':
        proba_pred_test = None
    else:
        proba_pred_test = np.zeros(shape=(n_test, n_classes), dtype=np.float32)

    cum_test = 0
    for X_test_, y_true_test_ in X_test:
        batch_size_ = int(tf.shape(X_test_).numpy()[0])

        if out_type != 'regress':
            y_true_test_ = tf.math.argmax(y_true_test_, axis=-1)  # revert one-hot encoding
            y_true_test_ = tf.expand_dims(y_true_test_, axis=-1)
        y_true_test_ = tf.cast(y_true_test_, dtype=tf.uint8)
        y_true_test_ = y_true_test_.numpy().flatten()
        y_true_test[cum_test:(cum_test + batch_size_)] = y_true_test_

        y_pred_test_ = model._predict(X_test_)
        y_pred_test_ = y_pred_test_.numpy().flatten()
        y_pred_test[cum_test:(cum_test + batch_size_)] = y_pred_test_

        if out_type != 'regress':
            proba_pred_test_ = model._predict_proba(X_test_)
            proba_pred_test_ = proba_pred_test_.numpy()
            proba_pred_test[cum_test:(cum_test + batch_size_), :] = proba_pred_test_

        cum_test += batch_size_

    # obtain performance
    perform_test = get_performance(y_true=y_true_test, y_pred=y_pred_test, proba_pred=proba_pred_test, n_classes=n_classes)
    perform_test['cfn_mtx'] = perform_test['cfn_mtx'].astype(int).tolist()  # facilitate JSON serialization

    # collect info
    results = {
        'dataset': TAG_DATASET,
        'target_type': TAG_TARGET_TYPE,
        'task_type': TAG_TASK_TYPE, 'task_name': TASK,
        'exprm_idx': idx_exprm + 1, 'exprm_total': EXPERIM.NUM_EXPERIM,
        'split_idx': idx_split + 1, 'split_total': CONFIG.NUM_SPLITS,
        'n_train': n_train, 'n_valid': n_valid, 'n_test': n_test,
        'n_classes': n_classes,
        'class_counts_train': class_counts_train,
        'backbone_model': 'CNN(2D)|EfficientNetB0',
        'hyperparameters': hyparams,
        'n_epochs_max': n_epochs_max, 'n_epochs_run': n_epochs_run, 'n_epochs_best': n_epochs_best, 'early_stop': early_stop,
        'losses_train': losses_train, 'losses_valid': losses_valid,
        'performance_test': perform_test
    }

    file_results = CONFIG.FILE_RESULTS.format(TAG_TASK_TYPE, TAG_DATASET,
                                              idx_exprm + 1, EXPERIM.NUM_EXPERIM,
                                              idx_split + 1, CONFIG.NUM_SPLITS)
    with open(os.path.join(CONFIG.PATH_RESULTS, file_results), 'w') as f_res:
        json.dump(results, f_res)

    print('Done!')
