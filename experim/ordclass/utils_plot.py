# Author: Fernando García-García <fegarcia@bcamath.org>

import matplotlib.pyplot as plt

from PIL import Image

# plotting
FIG_SIZE_BASE = (7.0, 7.0)

COLOR_MAP = 'gray'


def plot_2d_label(image, label, *, n_classes, suptitle='', cmap=COLOR_MAP):
    title = 'Image - Label: {} | Num. classes: {}'.format(label, n_classes)
    img_ = Image.fromarray(image)

    fig_size = FIG_SIZE_BASE
    fig = plt.figure(figsize=fig_size)

    plt.title(title)
    plt.imshow(img_, cmap=cmap)
    plt.axis('off')
    plt.suptitle(suptitle)
    plt.show(block=True)

    return fig


def plot_2d_mask(image, mask, *, suptitle='', cmap=COLOR_MAP):
    l_titles = ['Image', 'Mask']
    l_images = [image, mask]
    n_images = len(l_images)

    fig_size = (n_images * FIG_SIZE_BASE[0], FIG_SIZE_BASE[1])
    fig = plt.figure(figsize=fig_size)
    for idx, (img_, title_) in enumerate(zip(l_images, l_titles)):
        img_ = Image.fromarray(img_)

        plt.subplot(1, n_images, idx + 1)

        plt.title(title_)
        plt.imshow(img_, cmap=cmap)
        plt.axis('off')

    plt.suptitle(suptitle)
    plt.show(block=True)

    return fig
