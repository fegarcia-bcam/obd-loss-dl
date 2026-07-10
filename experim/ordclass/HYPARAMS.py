# Author: Fernando García-García <fegarcia@bcamath.org>

from experim.ordclass import CONFIG

HYPARAMS_COMMON = {
    'to_logits': True,

    'gamma': 2.0,  # relevant only for focal-based losses
    'class_weight': 'balanced',

    'dropout': 0.1,

    'activation': 'swish',

    'batch_size': 32,
    'optimizer': 'AdamW',
    'learning_rate': 3.0e-4,
    'min_delta': 0.0,
    'beta_1': 0.900,
    'beta_2': 0.999,
    'epsilon': 1.0e-07,
    'weight_decay': 1.0e-4,

    # 'verbose': 'auto',
    'verbose': 2,
    'random_state': None
}

HYPARAMS_CLASSIF = {
    'n_dense': (256, 64)
}

HYPARAMS_CLASSIF_2D = {
    'weights': 'imagenet',
    'pooling': 'avg',
    'freeze_blocks': 'all'
}

HYPARAMS_SEGMENT = {
    'adapt': 'resize',

    'kernel_size': 3,
    'stride_conv': 1,
    'padding': 'same',
    'pool_size': 2,
    'stride_pool': 2
}

HYPARAMS_SEGMENT_2D = {
    'weights': 'imagenet',
    'freeze_blocks': 'all'
}

HYPARAMS_DATASET = {  # specific for each dataset (according to its size, etc.)
    CONFIG.TAG_CLASSIF: {
        CONFIG.TAG_ORDIN: {
            'BACH': {'epochs': 1000, 'patience': 30},
            'DDR': {'epochs': 50, 'patience': 10}
        },
        CONFIG.TAG_DISCR: {
            'UTKF': {'epochs': 20, 'patience': 5}
        }
    },
    CONFIG.TAG_SEGMENT: {
        CONFIG.TAG_ORDIN: {
            'BACH': {'epochs': 500, 'patience': 20}
        }
    }
}
