# Authors:  Fernando García-García <fegarcia@bcamath.org>

import sys

from numbers import Integral, Real

import numpy as np

from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval, Options, StrOptions, InvalidParameterError
from sklearn.utils.validation import check_is_fitted

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout

from tensorflow.keras.initializers import GlorotUniform, Constant

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.losses import CategoricalCrossentropy

from tensorflow.keras.applications import EfficientNetB0, EfficientNetB5

from .ordinal_layers_tf import RegressionOutput, NominalOutput

from .nominal_losses_tf import focal_loss

from .ordinal_losses_tf import or_cross_entropy_loss
from .ordinal_losses_tf import or_focal_loss
from .ordinal_losses_tf import owk_loss
from .ordinal_losses_tf import obd_cross_entropy_loss
from .ordinal_losses_tf import obd_focal_loss


_MIN_CLASSES = 3
_MAX_CLASSES = 2 ** 8

ORDINAL_OUTPUT_TYPES = ['regression', 'nominal']
ORDINAL_LOSSES_UNET = ['reg_mae', 'reg_mse',
                       'nom_cross_entropy', 'nom_focal',
                       'or_cross_entropy', 'or_focal',
                       'owk',
                       'obd_cross_entropy', 'obd_focal']


class OrdinalCNN2DBase(ClassifierMixin, BaseEstimator):

    def __init__(
            self,
            n_classes,
            out_type='nominal',
            to_logits=True,
            loss='or_cross_entropy',
            regul_type=None,
            regul_eta=0.0,
            regul_delta=1.0,
            kappa_weights='quadratic',
            focal_gamma=2.0,
            ordin_decomp_weight='balanced',
            class_counts=None,
            class_priors=None,
            class_weight='balanced',

            input_height=224,
            input_width=224,
            input_channels=3,

            weights='imagenet',
            pooling='avg',
            n_dense=(256, 64),
            activation='swish',
            dropout=0.1,
            freeze_blocks='none',

            epochs=100,
            batch_size=16,
            solver='adam',
            learning_rate=0.001,
            patience=10,
            min_delta=0.0,
            beta_1=0.900,
            beta_2=0.999,
            epsilon=1.0e-07,
            weight_decay=None,

            verbose='auto',
            random_state=None
    ):
        self.n_classes = n_classes
        self.out_type = out_type
        self.to_logits = to_logits
        self.loss = loss
        if (self.out_type == 'regression') and (self.loss not in ['reg_mae', 'reg_mse']):
            raise InvalidParameterError
        if (self.out_type != 'regression') and (self.loss in ['reg_mae', 'reg_mse']):
            raise InvalidParameterError
        self.regul_type = regul_type
        self.regul_eta = regul_eta
        self.regul_delta = regul_delta
        self.kappa_weights = kappa_weights
        self.focal_gamma = focal_gamma
        self.ordin_decomp_weight = ordin_decomp_weight
        self.class_counts = class_counts
        self.class_priors = class_priors
        self.class_weight = class_weight

        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels

        self.weights = weights
        self.pooling = pooling

        for n_d in n_dense:
            if (n_d <= 0) or (not isinstance(n_d, int)):
                raise InvalidParameterError
        self.n_dense = n_dense

        self.activation = activation
        self.dropout = dropout
        self.freeze_blocks = freeze_blocks

        self.epochs = epochs
        self.batch_size = batch_size
        self.solver = solver
        self.learning_rate = learning_rate
        self.patience = patience
        self.min_delta = min_delta
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.weight_decay = weight_decay

        self.verbose = verbose
        self.random_state = random_state

    def _build(self):
        self.model_ = None
        raise NotImplementedError

    def _get_class_counts(self, X, ohe=True):
        class_counts = np.zeros(shape=(self.n_classes,), dtype=np.uint64)
        for _, lbl in X:
            if ohe:
                lbl = tf.math.argmax(lbl, axis=-1)  # revert one-hot encoding
            else:
                lbl = tf.squeeze(lbl, axis=-1)

            for c in range(self.n_classes):
                class_equal = tf.equal(lbl, c)
                class_equal = tf.cast(class_equal, dtype=tf.uint64)
                class_counts[c] += tf.math.reduce_sum(class_equal, axis=None).numpy()

        return class_counts

    def _get_class_priors(self, X, ohe=True):
        class_counts =  self._get_class_counts(X, ohe=ohe)
        class_priors = class_counts / np.sum(class_counts)
        class_priors = class_priors.astype(np.float32)
        return class_priors

    def _fit(self, X):
        # check parameters
        self._validate_params()

        # compute class counts, priors and weights, if necessary
        if self.class_counts is not None:
            self.class_counts_ = self.class_counts
        else:
            ohe = (self.out_type != 'regression')
            self.class_counts_ = self._get_class_counts(X, ohe=ohe)

        if self.class_priors is not None:
            self.class_priors_ = self.class_priors
        else:
            ohe = (self.out_type != 'regression')
            self.class_priors_ = self._get_class_priors(X, ohe=ohe)

        if self.class_weight is None:
            self.class_weight_ = np.ones(shape=(self.n_classes,), dtype=np.float32)
        elif isinstance(self.class_weight, str) and (self.class_weight == 'balanced'):
            self.class_weight_ = 1.0 / (self.class_priors_ * self.n_classes)
        elif not isinstance(self.class_weight, np.ndarray) or np.any(self.class_weight < 0.0):
            raise ValueError
        else:
            self.class_weight_ = self.class_weight

        # early stopping
        callback_early_stop = EarlyStopping(monitor='loss', mode='min',
                                            patience=self.patience, min_delta=self.min_delta,
                                            restore_best_weights=True)

        # build the CNN model
        self._build()

        # run the training
        self.history_ = self.model_.fit(x=X, y=None,
                                        epochs=self.epochs,
                                        callbacks=[callback_early_stop],
                                        verbose=self.verbose)
        return self

    def fit(self, X, y):
        return self._fit(X)

    def _fit_valid(self, X, X_val):
        # check parameters
        self._validate_params()

        # compute class counts, priors and weights, if necessary
        if self.class_counts is not None:
            self.class_counts_ = self.class_counts
        else:
            ohe = (self.out_type != 'regression')
            self.class_counts_ = self._get_class_counts(X, ohe=ohe)

        if self.class_priors is not None:
            self.class_priors_ = self.class_priors
        else:
            ohe = (self.out_type != 'regression')
            self.class_priors_ = self._get_class_priors(X, ohe=ohe)

        if self.class_weight is None:
            self.class_weight_ = np.ones(shape=(self.n_classes,), dtype=np.float32)
        elif isinstance(self.class_weight, str) and (self.class_weight == 'balanced'):
            self.class_weight_ = 1.0 / (self.class_priors_ * self.n_classes)
        elif not isinstance(self.class_weight, np.ndarray) or np.any(self.class_weight < 0.0):
            raise ValueError
        else:
            self.class_weight_ = self.class_weight

        # early stopping
        callback_early_stop = EarlyStopping(monitor='val_loss', mode='min',
                                            patience=self.patience, min_delta=self.min_delta,
                                            restore_best_weights=True)

        # build the CNN model
        self._build()

        # run the training
        self.history_ = self.model_.fit(x=X,
                                        validation_data=X_val,
                                        epochs=self.epochs,
                                        callbacks=[callback_early_stop],
                                        verbose=self.verbose)
        return self

    def fit_valid(self, X, X_val):
        return self._fit_valid(X, X_val)

    def _predict(self, X):
        if self.out_type == 'regression':
            y_pred = self.model_(X)
            y_pred = tf.round(y_pred)
            y_pred = tf.clip_by_value(y_pred, 0, self.n_classes - 1)
        else:
            proba_pred = self.model_(X)
            y_pred = tf.math.argmax(proba_pred, axis=-1)  # no need to revert logits

        y_pred = tf.expand_dims(y_pred, axis=-1)
        y_pred = tf.cast(y_pred, dtype=tf.uint8)
        return y_pred

    def _predict_proba(self, X):
        if self.out_type == 'regression':
            raise NotImplementedError

        proba_pred = self.model_(X)
        if self.to_logits:
            proba_pred = tf.nn.softmax(proba_pred, axis=-1)

        # prevent issues with numerical precision
        proba_pred = tf.clip_by_value(proba_pred,
                                      clip_value_min=tf.keras.backend.epsilon(),
                                      clip_value_max=1.0 - tf.keras.backend.epsilon())

        return proba_pred

    def predict(self, X):
        check_is_fitted(self)

        y_pred = []
        for X_, _ in X:
            y_pred_ = self._predict(X_)
            y_pred.append(y_pred_.numpy())
        y_pred = np.concatenate(y_pred)
        y_pred = tf.convert_to_tensor(y_pred)

        return y_pred

    def predict_proba(self, X):
        check_is_fitted(self)

        proba_pred = []
        for X_, _ in X:
            proba_pred_ = self._predict_proba(X_)
            proba_pred.append(proba_pred_.numpy())
        proba_pred = np.concatenate(proba_pred)
        proba_pred = tf.convert_to_tensor(proba_pred)

        return proba_pred

    def predict_log_proba(self, X):
        check_is_fitted(self)

        log_proba_pred = []
        for X_, _ in X:
            proba_pred_ = self._predict_proba(X_)
            log_proba_pred_= tf.math.log(proba_pred_)
            log_proba_pred.append(log_proba_pred_.numpy())
        log_proba_pred = np.concatenate(log_proba_pred)
        log_proba_pred = tf.convert_to_tensor(log_proba_pred)

        return log_proba_pred

    def score(self, X, y, sample_weight=None):
        check_is_fitted(self)

        scores = []
        for X_, y_ in X:
            if self.out_type == 'regression':
                y_true_ = tf.squeeze(y_, axis=-1)
            else:
                y_true_ = tf.math.argmax(y_, axis=-1)  # revert one-hot encoding

            # predict
            y_pred_ = self._predict(X_)

            # compare true versus predicted
            y_equal_ = tf.math.equal(y_true_, y_pred_)
            y_equal_ = tf.cast(y_equal_, dtype=tf.float32)

            # compute mean accuracy score, within the sample
            axes_reduce = tf.range(1, tf.rank(y_equal_))
            score_ = tf.math.reduce_mean(y_equal_, axis=axes_reduce)
            scores.append(score_.numpy())
        scores = np.concatenate(scores)

        # apply sample weights
        if sample_weight is not None:
            scores = scores * sample_weight

        score = np.mean(scores)
        return score

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.non_deterministic = True
        return tags


class OrdinalCNN2DEffB0(OrdinalCNN2DBase):
    _N_BLOCKS_EFFB0 = 7
    _PREPROC_LAYERS_EFFB0 = ['input', 'rescaling', 'normalization', 'stem']
    _BLOCK_NAME_EFFB0 = 'block{}'

    _parameter_constraints: dict = {
        'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')],
        'out_type': [StrOptions({*ORDINAL_OUTPUT_TYPES})],
        'to_logits': ['boolean'],
        'loss': [StrOptions({*ORDINAL_LOSSES_UNET})],
        'regul_type': [StrOptions({'beta'}), None],
        'regul_eta': [Interval(Real, 0.0, 1.0, closed='both'), None],
        'regul_delta': [Interval(Real, 0.0, None, closed='neither'), None],
        'kappa_weights': [StrOptions({'nominal', 'linear', 'quadratic'}), 'array-like'],
        'focal_gamma': [Interval(Real, 0.0, None, closed='left')],
        'ordin_decomp_weight': [StrOptions({'balanced'}), dict, None],
        'class_counts': ['array-like', None],
        'class_priors': ['array-like', None],
        'class_weight': [StrOptions({'balanced'}), 'array-like', None],

        'weights': [StrOptions({'imagenet'}), None],
        'pooling': [StrOptions({'avg', 'max'}), None],

        'dropout': [Interval(Real, 0.0, 1.0, closed='both'), None],
        'freeze_blocks': [Interval(Integral, 0, _N_BLOCKS_EFFB0, closed='both'), StrOptions({'none', 'all'})],

        'epochs': [Interval(Integral, 1, None, closed='left')],
        'batch_size': [Interval(Integral, 1, None, closed='left')],
        'learning_rate': [Interval(Real, 0.0, None, closed='neither')],
        'patience': [Interval(Integral, 1, None, closed='left'), Options(Real, {np.inf})],
        'min_delta': [Interval(Real, 0.0, None, closed='left')],
        'beta_1': [Interval(Real, 0.0, 1.0, closed='left')],
        'beta_2': [Interval(Real, 0.0, 1.0, closed='left')],
        'epsilon': [Interval(Real, 0.0, None, closed='neither')],

        'verbose': [Interval(Integral, 0, 2, closed='both'), StrOptions({'auto'})],
        'random_state': ['random_state']
    }

    def __init__(
            self,
            n_classes,
            out_type='nominal',
            to_logits=True,
            loss='or_cross_entropy',
            regul_type=None,
            regul_eta=0.0,
            regul_delta=1.0,
            kappa_weights='quadratic',
            ordin_decomp_weight='balanced',
            focal_gamma=2.0,
            class_counts=None,
            class_priors=None,
            class_weight='balanced',

            input_height=224,
            input_width=224,
            input_channels=3,

            weights='imagenet',
            pooling='avg',

            n_dense=(256, 64),
            activation='swish',
            dropout=0.1,
            freeze_blocks='none',

            epochs=100,
            batch_size=16,
            solver='adam',
            learning_rate=0.001,
            patience=10,
            min_delta=0.0,
            beta_1=0.900,
            beta_2=0.999,
            epsilon=1.0e-07,
            weight_decay=None,

            verbose='auto',
            random_state=None
    ):
        super().__init__(
            n_classes=n_classes,
            out_type=out_type,
            to_logits=to_logits,
            loss=loss,
            regul_type=regul_type,
            regul_eta=regul_eta,
            regul_delta=regul_delta,
            kappa_weights=kappa_weights,
            focal_gamma=focal_gamma,
            ordin_decomp_weight=ordin_decomp_weight,
            class_counts=class_counts,
            class_priors=class_priors,
            class_weight=class_weight,

            input_height=input_height,
            input_width=input_width,
            input_channels=input_channels,

            weights=weights,
            pooling=pooling,
            n_dense=n_dense,
            activation=activation,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            solver=solver,

            learning_rate=learning_rate,
            patience=patience,
            min_delta=min_delta,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            weight_decay=weight_decay,

            verbose=verbose,
            random_state=random_state
        )

        self.freeze_blocks = freeze_blocks

        self._validate_params()
        self._random_state = check_random_state(self.random_state)

    def _build(self):
        # input layer
        shape_in = (self.input_height, self.input_width, self.input_channels)
        x_in = Input(shape=shape_in, name='input_cnn')

        # backbone EfficientNetB0 without top dense layers
        model_backbone = EfficientNetB0(input_shape=shape_in,
                                        weights=self.weights,
                                        pooling=self.pooling,
                                        include_top=False)

        # freeze blocks
        if (self.freeze_blocks == 0) or (self.freeze_blocks == 'none'):  # do not freeze any layers
            model_backbone.trainable = True

        elif (self.freeze_blocks == OrdinalCNN2DEffB0._N_BLOCKS_EFFB0) or (self.freeze_blocks == 'all'):  # freeze all
            model_backbone.trainable = False

        else:  # freeze some blocks
            for layer in model_backbone.layers:
                freeze_layer = False
                for preproc_name in OrdinalCNN2DEffB0._PREPROC_LAYERS_EFFB0:
                    if layer.name.startswith(preproc_name):
                        freeze_layer = True
                        break

                if not freeze_layer:
                    if isinstance(self.freeze_blocks, int):  # should always be the case here
                        for idx_block in range(self.freeze_blocks):
                            block_name = OrdinalCNN2DEffB0._BLOCK_NAME_EFFB0.format(idx_block + 1)
                            if layer.name.startswith(block_name):
                                freeze_layer = True
                                break

                layer.trainable = not freeze_layer

        # set backbone architecture
        x = x_in
        x = model_backbone(x)

        # flatten output
        layer_flat = Flatten()
        x = layer_flat(x)

        # add top dense layers
        for idx, n_d in enumerate(self.n_dense):
            if self.dropout > 0.0:
                seed_drop = self._random_state.randint(sys.maxsize)
                layer_drop = Dropout(rate=self.dropout,
                                     seed=seed_drop,
                                     name=f"dropout_{idx}")
                x = layer_drop(x)

            seed_dense = self._random_state.randint(sys.maxsize)
            kernel_init_dense = GlorotUniform(seed=seed_dense)  # initialization as in Glorot et al.

            layer_dense = Dense(units=n_d,
                                activation=self.activation,
                                use_bias=True,
                                kernel_initializer=kernel_init_dense,
                                bias_initializer='zeros',
                                name=f"dense_{idx}")
            x = layer_dense(x)

        # ordinal output
        seed_out = self._random_state.randint(sys.maxsize)
        kernel_init_out = GlorotUniform(seed=seed_out)  # initialization as in Glorot et al.

        if self.out_type == 'regression':
            # initial guess for bias
            idx_classes = np.arange(self.n_classes)
            bias_init_out = np.dot(idx_classes, self.class_priors_)
            bias_init_out = Constant(bias_init_out)

            layer_ord_out = RegressionOutput(n_classes=self.n_classes,
                                             kernel_initializer=kernel_init_out,
                                             bias_initializer=bias_init_out,
                                             name='output')

        elif self.out_type == 'nominal':
            # initial guess for bias
            bias_init_out = np.log(self.class_priors_)
            bias_init_out = Constant(bias_init_out)

            layer_ord_out = NominalOutput(n_classes=self.n_classes,
                                          to_logits=self.to_logits,
                                          kernel_initializer=kernel_init_out,
                                          bias_initializer=bias_init_out,
                                          name='output')

        else:
            raise ValueError
        x = layer_ord_out(x)

        x_out = x

        # full model
        self.model_ = Model(x_in, x_out,
                            name='OrdinalCNN2DEffB0')
        if self.verbose:
            self.model_.summary(expand_nested=True, show_trainable=True)

        # prepare the optimization solver
        if self.solver == 'adam':
            self._optimizer = Adam(learning_rate=self.learning_rate,
                                   beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.epsilon,
                                   weight_decay=self.weight_decay)
        else:
            raise NotImplementedError

        # prepare the loss function
        if self.loss == 'reg_mae':
            loss_ = 'mae'
        elif self.loss == 'reg_mse':
            loss_ = 'mse'
        elif self.loss == 'nom_cross_entropy':
            loss_ = CategoricalCrossentropy(from_logits=self.to_logits)
        elif self.loss == 'nom_focal':
            loss_ = focal_loss(n_classes=self.n_classes,
                               from_logits=self.to_logits,
                               gamma=self.focal_gamma,
                               class_weight=self.class_weight_)
        elif self.loss == 'or_cross_entropy':
            loss_ = or_cross_entropy_loss(n_classes=self.n_classes,
                                          from_logits=self.to_logits,
                                          regul_type=self.regul_type,
                                          regul_eta=self.regul_eta,
                                          regul_delta=self.regul_delta,
                                          class_weight=self.class_weight_)
        elif self.loss == 'or_focal':
            loss_ = or_focal_loss(n_classes=self.n_classes,
                                  from_logits=self.to_logits,
                                  regul_type=self.regul_type,
                                  regul_eta=self.regul_eta,
                                  regul_delta=self.regul_delta,
                                  focal_gamma=self.focal_gamma,
                                  class_weight=self.class_weight_)
        elif self.loss == 'owk':
            loss_ = owk_loss(n_classes=self.n_classes,
                             from_logits=self.to_logits,
                             weights=self.kappa_weights,
                             class_priors=self.class_priors_,
                             class_weight=self.class_weight_)
        elif self.loss == 'obd_cross_entropy':
            loss_ = obd_cross_entropy_loss(n_classes=self.n_classes,
                                           from_logits=self.to_logits,
                                           ordin_decomp_weight=self.ordin_decomp_weight,
                                           class_counts=self.class_counts_,
                                           class_weight=self.class_weight_)
        elif self.loss == 'obd_focal':
            loss_ = obd_focal_loss(n_classes=self.n_classes,
                                   from_logits=self.to_logits,
                                   ordin_decomp_weight=self.ordin_decomp_weight,
                                   class_counts=self.class_counts_,
                                   focal_gamma=self.focal_gamma,
                                   class_weight=self.class_weight_)
        else:
            raise ValueError
        self.loss_ = loss_

        self.model_.compile(optimizer=self._optimizer, loss=self.loss_)


class OrdinalCNN2DEffB5(OrdinalCNN2DBase):
    _N_BLOCKS_EFFB5 = 7
    _PREPROC_LAYERS_EFFB5 = ['input', 'rescaling', 'normalization', 'stem']
    _BLOCK_NAME_EFFB5 = 'block{}'

    _parameter_constraints: dict = {
        'n_classes': [Interval(Integral, _MIN_CLASSES, _MAX_CLASSES, closed='both')],
        'out_type': [StrOptions({*ORDINAL_OUTPUT_TYPES})],
        'to_logits': ['boolean'],
        'loss': [StrOptions({*ORDINAL_LOSSES_UNET})],
        'regul_type': [StrOptions({'beta'}), None],
        'regul_eta': [Interval(Real, 0.0, 1.0, closed='both'), None],
        'regul_delta': [Interval(Real, 0.0, None, closed='neither'), None],
        'kappa_weights': [StrOptions({'nominal', 'linear', 'quadratic'}), 'array-like'],
        'focal_gamma': [Interval(Real, 0.0, None, closed='left')],
        'ordin_decomp_weight': [StrOptions({'balanced'}), dict, None],
        'class_counts': ['array-like', None],
        'class_priors': ['array-like', None],
        'class_weight': [StrOptions({'balanced'}), 'array-like', None],

        'weights': [StrOptions({'imagenet'}), None],
        'pooling': [StrOptions({'avg', 'max'}), None],

        'dropout': [Interval(Real, 0.0, 1.0, closed='both'), None],
        'freeze_blocks': [Interval(Integral, 0, _N_BLOCKS_EFFB5, closed='both'), StrOptions({'none', 'all'})],

        'epochs': [Interval(Integral, 1, None, closed='left')],
        'batch_size': [Interval(Integral, 1, None, closed='left')],
        'learning_rate': [Interval(Real, 0.0, None, closed='neither')],
        'patience': [Interval(Integral, 1, None, closed='left'), Options(Real, {np.inf})],
        'min_delta': [Interval(Real, 0.0, None, closed='left')],
        'beta_1': [Interval(Real, 0.0, 1.0, closed='left')],
        'beta_2': [Interval(Real, 0.0, 1.0, closed='left')],
        'epsilon': [Interval(Real, 0.0, None, closed='neither')],

        'verbose': [Interval(Integral, 0, 2, closed='both'), StrOptions({'auto'})],
        'random_state': ['random_state']
    }

    def __init__(
            self,
            n_classes,
            out_type='nominal',
            to_logits=True,
            loss='or_cross_entropy',
            regul_type=None,
            regul_eta=0.0,
            regul_delta=1.0,
            kappa_weights='quadratic',
            focal_gamma=2.0,
            ordin_decomp_weight='balanced',
            class_counts=None,
            class_priors=None,
            class_weight='balanced',

            input_height=456,
            input_width=456,
            input_channels=3,

            weights='imagenet',
            pooling='avg',
            n_dense=(256, 64),
            activation='swish',
            dropout=0.1,
            freeze_blocks='none',

            epochs=100,
            batch_size=16,
            solver='adam',
            learning_rate=0.001,
            patience=10,
            min_delta=0.0,
            beta_1=0.900,
            beta_2=0.999,
            epsilon=1.0e-07,
            weight_decay=None,

            verbose='auto',
            random_state=None
    ):
        super().__init__(
            n_classes=n_classes,
            out_type=out_type,
            to_logits=to_logits,
            loss=loss,
            regul_type=regul_type,
            regul_eta=regul_eta,
            regul_delta=regul_delta,
            kappa_weights=kappa_weights,
            focal_gamma=focal_gamma,
            ordin_decomp_weight=ordin_decomp_weight,
            class_counts=class_counts,
            class_priors=class_priors,
            class_weight=class_weight,

            input_height=input_height,
            input_width=input_width,
            input_channels=input_channels,

            weights=weights,
            pooling=pooling,
            n_dense=n_dense,
            activation=activation,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            solver=solver,

            learning_rate=learning_rate,
            patience=patience,
            min_delta=min_delta,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            weight_decay=weight_decay,

            verbose=verbose,
            random_state=random_state
        )

        self.freeze_blocks = freeze_blocks

        self._validate_params()
        self._random_state = check_random_state(self.random_state)

    def _build(self):
        # input layer
        shape_in = (self.input_height, self.input_width, self.input_channels)
        x_in = Input(shape=shape_in, name='input_cnn')

        # backbone EfficientNetB5 without top dense layers
        model_backbone = EfficientNetB5(input_shape=shape_in,
                                        weights=self.weights,
                                        pooling=self.pooling,
                                        include_top=False)

        # freeze blocks
        if (self.freeze_blocks == 0) or (self.freeze_blocks == 'none'):  # do not freeze any layers
            model_backbone.trainable = True

        elif (self.freeze_blocks == OrdinalCNN2DEffB5._N_BLOCKS_EFFB5) or (self.freeze_blocks == 'all'):  # freeze all
            model_backbone.trainable = False

        else:  # freeze some blocks
            for layer in model_backbone.layers:
                freeze_layer = False
                for preproc_name in OrdinalCNN2DEffB5._PREPROC_LAYERS_EFFB5:
                    if layer.name.startswith(preproc_name):
                        freeze_layer = True
                        break

                if not freeze_layer:
                    if isinstance(self.freeze_blocks, int):  # should always be the case here
                        for idx_block in range(self.freeze_blocks):
                            block_name = OrdinalCNN2DEffB5._BLOCK_NAME_EFFB5.format(idx_block + 1)
                            if layer.name.startswith(block_name):
                                freeze_layer = True
                                break

                layer.trainable = not freeze_layer

        # set backbone architecture
        x = x_in
        x = model_backbone(x)

        # flatten output
        layer_flat = Flatten()
        x = layer_flat(x)

        # add top dense layers
        for idx, n_d in enumerate(self.n_dense):
            if self.dropout > 0.0:
                seed_drop = self._random_state.randint(sys.maxsize)
                layer_drop = Dropout(rate=self.dropout,
                                     seed=seed_drop,
                                     name=f"dropout_{idx}")
                x = layer_drop(x)

            seed_dense = self._random_state.randint(sys.maxsize)
            kernel_init_dense = GlorotUniform(seed=seed_dense)  # initialization as in Glorot et al.

            layer_dense = Dense(units=n_d,
                                activation=self.activation,
                                use_bias=True,
                                kernel_initializer=kernel_init_dense,
                                bias_initializer='zeros',
                                name=f"dense_{idx}")
            x = layer_dense(x)

        # ordinal output
        seed_out = self._random_state.randint(sys.maxsize)
        kernel_init_out = GlorotUniform(seed=seed_out)  # initialization as in Glorot et al.

        if self.out_type == 'regression':
            # initial guess for bias
            idx_classes = np.arange(self.n_classes)
            bias_init_out = np.dot(idx_classes, self.class_priors_)
            bias_init_out = Constant(bias_init_out)

            layer_ord_out = RegressionOutput(n_classes=self.n_classes,
                                             kernel_initializer=kernel_init_out,
                                             bias_initializer=bias_init_out,
                                             name='output')

        elif self.out_type == 'nominal':
            # initial guess for bias
            bias_init_out = np.log(self.class_priors_)
            bias_init_out = Constant(bias_init_out)

            layer_ord_out = NominalOutput(n_classes=self.n_classes,
                                          to_logits=self.to_logits,
                                          kernel_initializer=kernel_init_out,
                                          bias_initializer=bias_init_out,
                                          name='output')

        else:
            raise ValueError
        x = layer_ord_out(x)

        x_out = x

        # full model
        self.model_ = Model(x_in, x_out,
                            name='OrdinalCNN2DEffB0')
        if self.verbose:
            self.model_.summary(expand_nested=True, show_trainable=True)

        # prepare the optimization solver
        if self.solver == 'adam':
            self._optimizer = Adam(learning_rate=self.learning_rate,
                                   beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.epsilon,
                                   weight_decay=self.weight_decay)
        else:
            raise NotImplementedError

        # prepare the loss function
        if self.loss == 'reg_mae':
            loss_ = 'mae'
        elif self.loss == 'reg_mse':
            loss_ = 'mse'
        elif self.loss == 'nom_cross_entropy':
            loss_ = CategoricalCrossentropy(from_logits=self.to_logits)
        elif self.loss == 'nom_focal':
            loss_ = focal_loss(n_classes=self.n_classes,
                               from_logits=self.to_logits,
                               gamma=self.focal_gamma,
                               class_weight=self.class_weight_)
        elif self.loss == 'or_cross_entropy':
            loss_ = or_cross_entropy_loss(n_classes=self.n_classes,
                                          from_logits=self.to_logits,
                                          regul_type=self.regul_type,
                                          regul_eta=self.regul_eta,
                                          regul_delta=self.regul_delta,
                                          class_weight=self.class_weight_)
        elif self.loss == 'or_focal':
            loss_ = or_focal_loss(n_classes=self.n_classes,
                                  from_logits=self.to_logits,
                                  regul_type=self.regul_type,
                                  regul_eta=self.regul_eta,
                                  regul_delta=self.regul_delta,
                                  focal_gamma=self.focal_gamma,
                                  class_weight=self.class_weight_)
        elif self.loss == 'owk':
            loss_ = owk_loss(n_classes=self.n_classes,
                             from_logits=self.to_logits,
                             weights=self.kappa_weights,
                             class_priors=self.class_priors_,
                             class_weight=self.class_weight_)
        elif self.loss == 'obd_cross_entropy':
            loss_ = obd_cross_entropy_loss(n_classes=self.n_classes,
                                           from_logits=self.to_logits,
                                           ordin_decomp_weight=self.ordin_decomp_weight,
                                           class_counts=self.class_counts_,
                                           class_weight=self.class_weight_)
        elif self.loss == 'obd_focal':
            loss_ = obd_focal_loss(n_classes=self.n_classes,
                                   from_logits=self.to_logits,
                                   ordin_decomp_weight=self.ordin_decomp_weight,
                                   class_counts=self.class_counts_,
                                   focal_gamma=self.focal_gamma,
                                   class_weight=self.class_weight_)
        else:
            raise ValueError
        self.loss_ = loss_

        self.model_.compile(optimizer=self._optimizer, loss=self.loss_)
