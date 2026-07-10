# Author: Fernando García-García <fegarcia@bcamath.org>

import numpy as np
import scipy as sp

from PIL import ImageEnhance

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import img_to_array, array_to_img

from tensorflow.experimental.numpy import shape as np_shape

_N_DIMS_BASE_2D = 2  # height, width

_N_DIMS_MAIN_2D = 3  # height, width, channels

_N_DIMS_BATCH_2D = 4  # batch, height, width, channels

_AX_FIRST = 1  # ignores batch dim 0
_AX_LAST = -1

_N_DIMS_BATCH = 1  # batch

_N_CHANNELS_BW = 1  # grayscale
_N_CHANNELS_COL = 3  # color (e.g. RGB)


class Augment2D(Layer):
    def __init__(
            self,
            *,
            displace_range=0.0,
            rotation_range=0.0,
            shear_range=0.0,
            zoom_range=0.0,
            height_flip=False,
            width_flip=False,
            sharpness_range=0.0,
            brightness_range=0.0,
            contrast_range=0.0,
            color_range=0.0,
            fill_mode='nearest',
            data_format='channels_last',
            seed=None,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.displace_range = displace_range
        self.rotation_range = rotation_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.height_flip = height_flip
        self.width_flip = width_flip
        self.sharpness_range = sharpness_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.color_range= color_range
        self.fill_mode = fill_mode

        if data_format not in ['channels_last', 'channels_first']:
            raise ValueError
        self.data_format = data_format

        self.seed = seed
        self.generator = np.random.default_rng(seed=self.seed)

    def get_config(self):
        base_config = super().get_config()
        config = {'displace_range': self.displace_range,
                  'rotation_range': self.rotation_range,
                  'shear_range': self.shear_range,
                  'zoom_range': self.zoom_range,
                  'height_flip': self.height_flip,
                  'width_flip': self.width_flip,
                  'sharpness_range': self.sharpness_range,
                  'brightness_range': self.brightness_range,
                  'contrast_range': self.contrast_range,
                  'color_range': self.color_range,
                  'fill_mode': self.fill_mode,
                  'data_format': self.data_format,
                  'seed': self.seed}

        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        displace_range = config.pop('displace_range')
        rotation_range = config.pop('rotation_range')
        shear_range = config.pop('shear_range')
        zoom_range = config.pop('zoom_range')
        height_flip = config.pop('height_flip')
        width_flip = config.pop('width_flip')
        sharpness_range = config.pop('sharpness_range')
        brightness_range = config.pop('brightness_range')
        contrast_range = config.pop('contrast_range')
        color_range = config.pop('color_range')
        fill_mode = config.pop('fill_mode')
        data_format = config.pop('data_format')
        seed = config.pop('seed')

        return cls(displace_range=displace_range,
                   rotation_range=rotation_range,
                   shear_range=shear_range,
                   zoom_range=zoom_range,
                   height_flip=height_flip,
                   width_flip=width_flip,
                   sharpness_range=sharpness_range,
                   brightness_range=brightness_range,
                   contrast_range=contrast_range,
                   color_range=color_range,
                   fill_mode=fill_mode,
                   data_format=data_format,
                   seed=seed,
                   **config)

    def build(self, input_shape):
        return

    def call(self, inputs, labels_in, training=True):
        if training:
            n_dims_input = len(np_shape(inputs))

            # check shape consistency
            if n_dims_input not in [_N_DIMS_MAIN_2D, _N_DIMS_BATCH_2D]:
                raise ValueError

            is_batched = (n_dims_input == _N_DIMS_BATCH_2D)
            if not is_batched:
                inputs = tf.expand_dims(inputs, axis=0)

            # move input channels to the last position
            perm_axes = list(range(_N_DIMS_BATCH_2D))
            if self.data_format == 'channels_first':
                perm_axes[_AX_FIRST], perm_axes[_AX_LAST] = perm_axes[_AX_LAST], perm_axes[_AX_FIRST]
                inputs = tf.transpose(inputs, perm=perm_axes)

            # guarantee order and consistency in the labels' shape
            n_dims_lbl_in = len(np_shape(labels_in))

            labels_classif = (n_dims_lbl_in == _N_DIMS_BATCH)
            labels_segment = (n_dims_lbl_in == _N_DIMS_MAIN_2D)
            if not (labels_classif or labels_segment):
                raise ValueError

            labels_in = tf.expand_dims(labels_in, axis=_AX_LAST)
            if not labels_classif:
                # move label channels to the last position
                if self.data_format == 'channels_first':
                    labels_in = tf.transpose(labels_in, perm=perm_axes)
            
            input_shape = np.asarray(np_shape(inputs))
            labels_in_shape = np.asarray(np_shape(labels_in))
            if labels_segment:
                if input_shape.size != labels_in_shape.size:
                    raise ValueError
                if np.any(input_shape[_AX_FIRST:_AX_LAST] != labels_in_shape[_AX_FIRST:_AX_LAST]):
                    raise ValueError

            # augment
            if labels_classif:
                # labels stay the same, use an auxiliary NaN tensor
                labels_augm = tf.keras.ops.full_like(inputs, fill_value=np.nan, dtype=tf.float32)
                labels_augm = tf.math.reduce_sum(labels_augm, axis=_AX_LAST, keepdims=True)
            else:
                labels_augm = labels_in

            outputs, labels_out = self._random_augment_2d(inputs, labels_augm)
            if labels_classif:
                labels_out = labels_in

            # move channels again to its original position
            if self.data_format == 'channels_first':
                outputs = tf.transpose(outputs, perm=perm_axes)
                if labels_segment:
                    labels_out = tf.transpose(labels_out, perm=perm_axes)

            labels_out = tf.squeeze(labels_out, axis=_AX_LAST)

        else:
            outputs, labels_out = inputs, labels_in
            
        return outputs, labels_out

    def _random_augment_2d(self, inputs, labels_augm):
        labels_min = tf.math.reduce_min(labels_augm, axis=None)
        labels_max = tf.math.reduce_max(labels_augm, axis=None)

        # define transforms
        def _offset_center_2d(mtx_in, h, w):
            off_h = float(h) / 2 - 0.5
            off_w = float(w) / 2 - 0.5

            mtx_offset = np.array(
                [[1.0, 0.0, +off_h],
                 [0.0, 1.0, +off_w],
                 [0.0, 0.0, 1.0]]
            )
            mtx_reset = np.array(
                [[1.0, 0.0, -off_h],
                 [0.0, 1.0, -off_w],
                 [0.0, 0.0, 1.0]]
            )

            mtx_out = np.dot(np.dot(mtx_offset, mtx_in), mtx_reset)
            return mtx_out

        def _affine_transform_2d(x_in,
                                 displace_x=0.0, displace_y=0.0,
                                 rot_theta=0.0,
                                 shear_xy=0.0, shear_yx=0.0,
                                 zoom_sc=1.0,
                                 fill_mode='nearest'):
            size_h, size_w, size_c = x_in.shape

            mtx_transf = None

            if zoom_sc != 1.0:
                mtx_zoom = np.array(
                    [[zoom_sc, 0.0, 0.0],
                     [0.0, zoom_sc, 0.0],
                     [0.0, 0.0, 1.0]]
                )
                mtx_zoom = _offset_center_2d(mtx_zoom, size_h, size_w)

                if mtx_transf is None:
                    mtx_transf = mtx_zoom
                else:
                    mtx_transf = np.dot(mtx_transf, mtx_zoom)

            if rot_theta != 0.0:
                rot_theta = np.deg2rad(rot_theta)
                mtx_rot = np.array(
                    [[np.cos(rot_theta), -np.sin(rot_theta), 0.0],
                     [np.sin(rot_theta), np.cos(rot_theta), 0.0],
                     [0.0, 0.0, 1.0]]
                )
                mtx_rot = _offset_center_2d(mtx_rot, size_h, size_w)

                if mtx_transf is None:
                    mtx_transf = mtx_rot
                else:
                    mtx_transf = np.dot(mtx_transf, mtx_rot)

            if (shear_xy != 0.0) or (shear_yx != 0.0):
                shear_xy = np.deg2rad(shear_xy)
                shear_yx = np.deg2rad(shear_yx)
                mtx_shear_x = np.array(
                    [[1.0, 0.0, 0.0],
                     [np.tan(shear_xy), 1.0, 0.0],
                     [0.0, 0.0, 1.0]]
                )
                mtx_shear_y = np.array(
                    [[1.0, np.tan(shear_yx), 0.0],
                     [0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0]]
                )
                mtx_shear = np.dot(mtx_shear_x, mtx_shear_y)
                mtx_shear = _offset_center_2d(mtx_shear, size_h, size_w)

                if mtx_transf is None:
                    mtx_transf = mtx_shear
                else:
                    mtx_transf = np.dot(mtx_transf, mtx_shear)

            if (displace_x != 0.0) or (displace_y != 0.0):
                mtx_displace = np.array(
                    [[1.0, 0.0, displace_x],
                     [0.0, 1.0, displace_y],
                     [0.0, 0.0, 1.0]]
                )
                if mtx_transf is None:
                    mtx_transf = mtx_displace
                else:
                    mtx_transf = np.dot(mtx_transf, mtx_displace)

            # apply
            if mtx_transf is None:
                x_out = x_in
            else:
                mtx_affine = mtx_transf[:-1, :-1]
                mtx_offset = mtx_transf[:-1, -1]

                l_x_out = []
                for idx_c in range(size_c):
                    x_in_ = x_in[:, :, idx_c]
                    x_out_ = sp.ndimage.interpolation.affine_transform(input=x_in_,
                                                                       matrix=mtx_affine,
                                                                       offset=mtx_offset,
                                                                       mode=fill_mode)
                    l_x_out.append(x_out_)
                x_out = np.stack(l_x_out, axis=_N_DIMS_BASE_2D)

            return x_out

        def _flip_axis_2d(x_in, axis):
            x_out = x_in.swapaxes(axis, 0)
            x_out = x_out[::-1, ...]
            x_out = x_out.swapaxes(0, axis)
            return x_out

        def _sharpness_shift_2d(x_in, factor_sharp, scale=True):
            _, _, size_c = x_in.shape

            val_min, val_max = np.min(x_in), np.max(x_in)
            local_scale = (val_min < 0) or (val_max > 255)

            if size_c in [_N_CHANNELS_BW, _N_CHANNELS_COL]:
                x_out = array_to_img(x_in, data_format='channels_last', scale=local_scale or scale)
                x_out = ImageEnhance.Sharpness(x_out).enhance(factor_sharp)
                x_out = img_to_array(x_out, data_format='channels_last')

            else:
                l_x_out = []
                for idx_c in range(size_c):
                    x_in_ = x_in[:, :, idx_c]
                    x_out_ = array_to_img(x_in_, data_format='channels_last', scale=local_scale or scale)
                    x_out_ = ImageEnhance.Sharpness(x_out_).enhance(factor_sharp)
                    x_out_ = img_to_array(x_out_, data_format='channels_last')
                    l_x_out.append(x_out_)
                x_out = np.stack(l_x_out, axis=_N_DIMS_BASE_2D)

            if not scale and local_scale:
                x_out = x_out / 255.0 * (val_max - val_min) + val_min

            return x_out

        def _brightness_shift_2d(x_in, factor_bright, scale=True):
            _, _, size_c = x_in.shape

            val_min, val_max = np.min(x_in), np.max(x_in)
            local_scale = (val_min < 0) or (val_max > 255)

            if size_c in [_N_CHANNELS_BW, _N_CHANNELS_COL]:
                x_out = array_to_img(x_in, data_format='channels_last', scale=local_scale or scale)
                x_out = ImageEnhance.Brightness(x_out).enhance(factor_bright)
                x_out = img_to_array(x_out, data_format='channels_last')

            else:
                l_x_out = []
                for idx_c in range(size_c):
                    x_in_ = x_in[:, :, idx_c]
                    x_out_ = array_to_img(x_in_, data_format='channels_last', scale=local_scale or scale)
                    x_out_ = ImageEnhance.Brightness(x_out_).enhance(factor_bright)
                    x_out_ = img_to_array(x_out_, data_format='channels_last')
                    l_x_out.append(x_out_)
                x_out = np.stack(l_x_out, axis=_N_DIMS_BASE_2D)

            if not scale and local_scale:
                x_out = x_out / 255.0 * (val_max - val_min) + val_min

            return x_out

        def _contrast_shift_2d(x_in, factor_contrast, scale=True):
            _, _, size_c = x_in.shape

            val_min, val_max = np.min(x_in), np.max(x_in)
            local_scale = (val_min < 0) or (val_max > 255)

            if size_c in [_N_CHANNELS_BW, _N_CHANNELS_COL]:
                x_out = array_to_img(x_in, data_format='channels_last', scale=local_scale or scale)
                x_out = ImageEnhance.Contrast(x_out).enhance(factor_contrast)
                x_out = img_to_array(x_out, data_format='channels_last')

            else:
                l_x_out = []
                for idx_c in range(size_c):
                    x_in_ = x_in[:, :, idx_c]
                    x_out_ = array_to_img(x_in_, data_format='channels_last', scale=local_scale or scale)
                    x_out_ = ImageEnhance.Contrast(x_out_).enhance(factor_contrast)
                    x_out_ = img_to_array(x_out_, data_format='channels_last')
                    l_x_out.append(x_out_)
                x_out = np.stack(l_x_out, axis=_N_DIMS_BASE_2D)

            if not scale and local_scale:
                x_out = x_out / 255.0 * (val_max - val_min) + val_min

            return x_out

        def _color_shift_2d(x_in, factor_color, scale=True):
            _, _, size_c = x_in.shape

            val_min, val_max = np.min(x_in), np.max(x_in)
            local_scale = (val_min < 0) or (val_max > 255)

            if size_c != _N_CHANNELS_COL:
                x_out = x_in

            else:
                x_out = array_to_img(x_in, data_format='channels_last', scale=local_scale or scale)
                x_out = ImageEnhance.Color(x_out).enhance(factor_color)
                x_out = img_to_array(x_out, data_format='channels_last')

                if not scale and local_scale:
                    x_out = x_out / 255.0 * (val_max - val_min) + val_min

            return x_out

        def _augment_2d(img_in, lbl_in):
            size_h, size_w, size_c = img_in.shape
            lbl_void = np.isnan(lbl_in).all()

            # generate random transforms
            if self.displace_range > 0.0:
                displace_shift = self.generator.uniform(low=-1.0 * self.displace_range,
                                                        high=+1.0 * self.displace_range,
                                                        size=_N_DIMS_BASE_2D)
                displace_h = displace_shift[0] * size_h
                displace_w = displace_shift[1] * size_w
            else:
                displace_h, displace_w = 0.0, 0.0

            if self.rotation_range > 0.0:
                rotation = self.generator.uniform(low=-1.0 * self.rotation_range,
                                                  high=+1.0 * self.rotation_range,
                                                  size=1).item()
            else:
                rotation = 0.0

            if self.shear_range > 0.0:
                shear_shift = self.generator.uniform(low=-1.0 * self.shear_range,
                                                     high=+1.0 * self.shear_range,
                                                     size=_N_DIMS_BASE_2D)
                shear_hw, shear_wh = shear_shift
            else:
                shear_hw, shear_wh = 0.0, 0.0

            if self.zoom_range > 0.0:
                zoom = self.generator.uniform(low=1.0 - self.zoom_range,
                                              high=1.0 + self.zoom_range,
                                              size=1).item()
            else:
                zoom = 1.0

            if self.height_flip:
                flip_h = self.generator.uniform(low=0.0, high=1.0,
                                                size=1).item()
                flip_h = (flip_h < 0.5)
            else:
                flip_h = False

            if self.width_flip:
                flip_w = self.generator.uniform(low=0.0, high=1.0,
                                                size=1).item()
                flip_w = (flip_w < 0.5)
            else:
                flip_w = False

            if self.sharpness_range > 0.0:
                sharp_shift = self.generator.uniform(low=1.0 - self.sharpness_range,
                                                     high=1.0 + self.sharpness_range,
                                                     size=1).item()
            else:
                sharp_shift = 1.0

            if self.brightness_range > 0.0:
                bright_shift = self.generator.uniform(low=1.0 - self.brightness_range,
                                                      high=1.0 + self.brightness_range,
                                                      size=1).item()
            else:
                bright_shift = 1.0

            if self.contrast_range > 0.0:
                contrast_shift = self.generator.uniform(low=1.0 - self.contrast_range,
                                                        high=1.0 + self.contrast_range,
                                                        size=1).item()
            else:
                contrast_shift = 1.0

            if self.color_range > 0.0:
                color_shift = self.generator.uniform(low=1.0 - self.color_range,
                                                     high=1.0 + self.color_range,
                                                     size=1).item()
            else:
                color_shift = 1.0

            # apply those random transformations
            img_out = img_in
            lbl_out = lbl_in

            # affine
            if ((displace_h != 0.0) or (displace_w != 0.0) or (rotation != 0.0)
                    or (shear_hw != 0.0) or (shear_wh != 0.0) or (zoom != 0.0)):
                img_out = _affine_transform_2d(img_out,
                                               displace_x=displace_h, displace_y=displace_w,
                                               rot_theta=rotation,
                                               shear_xy=shear_hw, shear_yx=shear_wh,
                                               zoom_sc=zoom,
                                               fill_mode=self.fill_mode)
                if not lbl_void:
                    lbl_out = _affine_transform_2d(lbl_out,
                                                   displace_x=displace_h, displace_y=displace_w,
                                                   rot_theta=rotation,
                                                   shear_xy=shear_hw, shear_yx=shear_wh,
                                                   zoom_sc=zoom,
                                                   fill_mode=self.fill_mode)

            # flip
            if flip_h:
                img_out = _flip_axis_2d(img_out, axis=0)
                if not lbl_void:
                    lbl_out = _flip_axis_2d(lbl_out, axis=0)
            if flip_w:
                img_out = _flip_axis_2d(img_out, axis=1)
                if not lbl_void:
                    lbl_out = _flip_axis_2d(lbl_out, axis=1)

            # sharpness, brightness, contrast and color shifts
            # these transformations have no effect on the labels
            if sharp_shift != 1.0:
                img_out = _sharpness_shift_2d(img_out, factor_sharp=sharp_shift, scale=False)
            if bright_shift != 1.0:
                img_out = _brightness_shift_2d(img_out, factor_bright=bright_shift, scale=False)
            if contrast_shift != 1.0:
                img_out = _contrast_shift_2d(img_out, factor_contrast=contrast_shift, scale=False)
            if (size_c == _N_CHANNELS_COL) and (color_shift != 1.0):
                img_out = _color_shift_2d(img_out, factor_color=color_shift, scale=False)

            # hack, concatenate image and labels
            img_out = np.expand_dims(img_out, axis=0)
            lbl_out = np.expand_dims(lbl_out, axis=0)
            data_out = np.concatenate([img_out, lbl_out], axis=0)

            return data_out

        # apply those transforms
        @tf.function
        def _augment_2d_tf(input_tf):
            def _augment_2d_np(input_np):
                img_in, lbl_in = input_np
                res_out = _augment_2d(img_in, lbl_in)
                return res_out
            data_out = tf.numpy_function(_augment_2d_np, [input_tf], tf.float32)
            return data_out

        # hack, concatenate image and labels
        input_shape = np.asarray(np_shape(inputs))
        labels_aux = tf.concat([labels_augm] * input_shape[_AX_LAST], axis=_AX_LAST)

        output_shape = input_shape.tolist()
        output_shape = output_shape[1:]
        output_shape.insert(0, 2)
        output_shape = tuple(output_shape)
        out_signature = tf.TensorSpec(shape=output_shape, dtype=tf.float32)

        data_out = tf.map_fn(fn=_augment_2d_tf,
                             elems=(inputs, labels_aux),  # inputs image, as well as labels
                             fn_output_signature=out_signature)

        # hack, separate image from labels
        outputs = data_out[:, 0, ...]
        labels_out = data_out[:, 1, ..., 0]

        # round and clip labels values
        # prevent numerical issues due to the interpolations within the affine transforms
        labels_out = tf.round(labels_out)
        labels_out = tf.clip_by_value(labels_out, labels_min, labels_max)
        labels_out = tf.expand_dims(labels_out, axis=-1)

        return outputs, labels_out
