# Author: Fernando García-García <fegarcia@bcamath.org>

import os

# naming and tags
TAG_DATA = 'data'
TAG_RESULTS = 'results'

TAG_CLASSIF = 'Classif'
TAG_SEGMENT = 'Segment'

TAG_ORDIN = 'Ordin'
TAG_DISCR = 'Discr'

TAG_SEP = '|'

TAG_SHAPE_X = 'X_shape'
TAG_SHAPE_Y = 'y_shape'

TAG_N_CLASSES = 'n_classes'

# paths and files
PATH_DATA = os.path.join(TAG_DATA, 'DL')
PATH_RESULTS = os.path.join(TAG_RESULTS, 'DL')

FILE_INFO = 'info.csv'
FILE_SPLIT = 'split.csv'

FILE_RESULTS = 'Results_{0}-{1}_Experim-{2:02d}of{3:02d}_Split{4:02d}of{5:02d}.json'

# splitting into train & validation & test
TAG_SPLIT = 'split'

TAG_TRAIN = 'train'
TAG_VALID = 'valid'
TAG_TEST = 'test'
LIST_SPLITS = [TAG_TRAIN, TAG_VALID, TAG_TEST]

NUM_SPLITS = 10

RATIO_SIZE_TEST = 0.20  # with respect to the entire dataset
RATIO_SIZE_VALID = 0.20  # with respect to the train & validation split

# statistical hypothesis testing for suitable splits
# values extracted from https://imaging.mrc-cbu.cam.ac.uk/statswiki/FAQ/effectSize
THRESH_OMEGA_SQ = {'S': 0.01}

THRESH_ETA_SQ = {'S': 0.01}

THRESH_CRAMER_V = {1: {'S': 0.10},
                   2: {'S': 0.07}}  # the maximum number of degrees of freedom is 2, as there are three splits

# general data aspects
N_DIMS_BATCH = 1  # batch

N_DIMS_BATCH_2D = 4  # batch, height, width, channels

# data formats and properties
IMG_CHANNELS_BW = 1
IMG_CHANNELS_RGB = 3

IMG_HEIGHT_EFFNET_B0 = 224
IMG_WIDTH_EFFNET_B0 = 224

IMG_HEIGHT_EFFNET_B5 = 456
IMG_WIDTH_EFFNET_B5 = 456

# data augmentation
DISPLACE_RANGE = 0.10  # ratio
ROTATION_RANGE = 10.0  # degrees
SHEAR_RANGE = 5.0  # degrees
ZOOM_RANGE = 0.10  # 1 +/- ratio
HEIGHT_FLIP = False
WIDTH_FLIP = True
SHARPNESS_RANGE = 0.25  # 1 +/- ratio
BRIGHTNESS_RANGE = 0.25  # 1 +/- ratio
CONTRAST_RANGE = 0.25  # 1 +/- ratio
COLOR_RANGE = 0.25  # 1 +/- ratio
