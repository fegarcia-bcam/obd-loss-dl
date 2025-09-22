HYPERPARAMS_COMMON = [
    {'name': 'to_logits', 'fixed': True, 'value': True},

    {'name': 'class_counts', 'fixed': True, 'value': None},  # to override for each dataset
    {'name': 'class_priors', 'fixed': True, 'value': None},  # to override for each dataset
    {'name': 'class_weight', 'fixed': True, 'value': 'balanced'},

    {'name': 'input_height', 'fixed': True, 'value': -1},  # to override for each dataset
    {'name': 'input_width', 'fixed': True, 'value': -1},  # to override for each dataset
    {'name': 'input_channels', 'fixed': True, 'value': -1},  # to override for each dataset

    {'name': 'weights', 'fixed': True, 'value': 'imagenet'},
    {'name': 'freeze_blocks', 'fixed': True, 'value': 'all'},  # freeze all blocks
    # {'name': 'freeze_blocks', 'fixed': True, 'value': 6},  # unfreeze the last block

    {'name': 'n_dense', 'fixed': True, 'value': (256, 64)},

    {'name': 'dropout', 'fixed': True, 'value': 0.1},
    {'name': 'activation', 'fixed': True, 'value': 'swish'},

    {'name': 'epochs', 'fixed': True, 'value': -1},  # to override for each dataset
    {'name': 'batch_size', 'fixed': True, 'value': -1},  # to override for each dataset
    {'name': 'solver', 'fixed': True, 'value': 'adam'},
    {'name': 'learning_rate', 'fixed': False, 'type': 'float', 'low': 1.0e-5, 'high': 1.0e-1, 'log': True},
    {'name': 'patience', 'fixed': True, 'value': 10},
    {'name': 'min_delta', 'fixed': True, 'value': 0.0},
    {'name': 'beta_1', 'fixed': True, 'value': 0.900},
    {'name': 'beta_2', 'fixed': True, 'value': 0.999},
    {'name': 'epsilon', 'fixed': True, 'value': 1.0e-07},
    {'name': 'weight_decay', 'fixed': True, 'value': None},

    {'name': 'verbose', 'fixed': True, 'value': 2},
    {'name': 'random_state', 'fixed': True, 'value': None}
]


EXPERIMENTS = []

experim = {
    'hyperparams': HYPERPARAMS_COMMON + [
        {'name': 'out_type', 'fixed': True, 'value': 'regression'},
        {'name': 'loss', 'fixed': True, 'value': 'reg_mae'}
    ]
}
EXPERIMENTS.append(experim)  # ID: 00

experim = {
    'hyperparams': HYPERPARAMS_COMMON + [
        {'name': 'out_type', 'fixed': True, 'value': 'regression'},
        {'name': 'loss', 'fixed': True, 'value': 'reg_mse'}
    ]
}
EXPERIMENTS.append(experim)  # ID: 01

experim = {
    'hyperparams': HYPERPARAMS_COMMON + [
        {'name': 'out_type', 'fixed': True, 'value': 'nominal'},
        {'name': 'loss', 'fixed': True, 'value': 'nom_cross_entropy'}
    ]
}
EXPERIMENTS.append(experim)  # ID: 02

experim = {
    'hyperparams': HYPERPARAMS_COMMON + [
        {'name': 'out_type', 'fixed': True, 'value': 'nominal'},
        {'name': 'loss', 'fixed': True, 'value': 'nom_focal'},
        {'name': 'focal_gamma', 'fixed': True, 'value': 2.0}
    ]
}
EXPERIMENTS.append(experim)  # ID: 03

experim = {
    'hyperparams': HYPERPARAMS_COMMON + [
        {'name': 'out_type', 'fixed': True, 'value': 'nominal'},
        {'name': 'loss', 'fixed': True, 'value': 'or_cross_entropy'},
        {'name': 'regul_type', 'fixed': True, 'value': 'beta'},
        {'name': 'regul_eta', 'fixed': False, 'type': 'float', 'low': 0.25, 'high': 0.75, 'log': False},
        {'name': 'regul_delta', 'fixed': True, 'value': 1.0}
    ]
}
EXPERIMENTS.append(experim)  # ID: 04

experim = {
    'hyperparams': HYPERPARAMS_COMMON + [
        {'name': 'out_type', 'fixed': True, 'value': 'nominal'},
        {'name': 'loss', 'fixed': True, 'value': 'or_focal'},
        {'name': 'focal_gamma', 'fixed': True, 'value': 2.0},
        {'name': 'regul_type', 'fixed': True, 'value': 'beta'},
        {'name': 'regul_eta', 'fixed': False, 'type': 'float', 'low': 0.25, 'high': 0.75, 'log': False},
        {'name': 'regul_delta', 'fixed': True, 'value': 1.0}
    ]
}
EXPERIMENTS.append(experim)  # ID: 05

experim = {
    'hyperparams': HYPERPARAMS_COMMON + [
        {'name': 'out_type', 'fixed': True, 'value': 'nominal'},
        {'name': 'loss', 'fixed': True, 'value': 'owk'},
        {'name': 'kappa_weights', 'fixed': True, 'value': 'linear'}
    ]
}
EXPERIMENTS.append(experim)  # ID: 06

experim = {
    'hyperparams': HYPERPARAMS_COMMON + [
        {'name': 'out_type', 'fixed': True, 'value': 'nominal'},
        {'name': 'loss', 'fixed': True, 'value': 'owk'},
        {'name': 'kappa_weights', 'fixed': True, 'value': 'quadratic'}
    ]
}
EXPERIMENTS.append(experim)  # ID: 07

experim = {
    'hyperparams': HYPERPARAMS_COMMON + [
        {'name': 'out_type', 'fixed': True, 'value': 'nominal'},
        {'name': 'loss', 'fixed': True, 'value': 'obd_cross_entropy'},
        {'name': 'obd_weight', 'fixed': True, 'value': 'balanced'}
    ]
}
EXPERIMENTS.append(experim)  # ID: 08

experim = {
    'hyperparams': HYPERPARAMS_COMMON + [
        {'name': 'out_type', 'fixed': True, 'value': 'nominal'},
        {'name': 'loss', 'fixed': True, 'value': 'obd_focal'},
        {'name': 'focal_gamma', 'fixed': True, 'value': 2.0},
        {'name': 'obd_weight', 'fixed': True, 'value': 'balanced'}
    ]
}
EXPERIMENTS.append(experim)  # ID: 09
