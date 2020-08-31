""" Copyright 2020 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os, sys
from functools import partial
import random
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

import third_party.augment_ops as augment_ops
import third_party.rand_augment as randaug
import third_party.data_util as simclr_ops


def apply_augment(image, ops_list=None):
    """ Apply Augmentation Sequence.

    Args:
        image: 3D tensor of (height, width, channel)
        opt_list: list of augmentation operation returned by compose_augment_seq
    """
    if ops_list is None:
        return (image,)
    if not isinstance(ops_list, (tuple, list)):
        ops_list = [ops_list]
    def _apply_augment(image, ops):
        for op in ops:
            image = op(image)
        return image
    return tuple([_apply_augment(image, ops) for ops in ops_list])

def compose_augment_seq(aug_list, is_training=False):
    """ Compose Augmentation Sequence.

    Args:
        aug_list: List of tuples (aug_type, kwargs)
    """
    return [generate_augment_ops(aug_type, is_training=is_training, **kwargs) for aug_type, kwargs in aug_list]

def generate_augment_ops(aug_type, is_training=False, **kwargs):
    """ Generate Augmentation Operators.

    Args:
        aug_type: Augmentation type
        is_training: Boolea
    """
    registered_ops = ['resize', 'crop', 'hflip', 'blur',
                      'jitter', 'cutout', 'erase', 'randaug',
                      'rotate90', 'rotate180', 'rotate270',
                      'gaussian', 'translation', 'affine', 'intensity_flip', 'intensity_scaling', 'intensity_offset',
                      'identity'
    ]
    assert aug_type.lower() in registered_ops

    if aug_type.lower() == 'erase':
        scale = kwargs['scale'] if 'scale' in kwargs else (0.02, 0.3)
        ratio = kwargs['ratio'] if 'ratio' in kwargs else 3.3
        value = kwargs['value'] if 'value' in kwargs else 0.5
        tx_op = RandomErase(scale=scale, ratio=ratio, value=value)

    elif aug_type.lower() == 'randaug':
        num_layers = kwargs['num_layers'] if 'num_layers' in kwargs else 2
        prob_to_apply = kwargs['prob_to_apply'] if 'prob_to_apply' in kwargs else 0.5
        magnitude = kwargs['magnitude'] if 'magnitude' in kwargs else None
        num_levels = kwargs['num_levels'] if 'num_levels' in kwargs else None
        mode = kwargs['mode'] if 'mode' in kwargs else 'all'
        size = kwargs['size'] if 'size' in kwargs else None
        tx_op = RandAugment(num_layers=num_layers,
            prob_to_apply=prob_to_apply,
            magnitude=magnitude,
            num_levels=num_levels,
            size=size, mode=mode)

    elif aug_type.lower() == 'cutout':
        scale = kwargs['scale'] if 'scale' in kwargs else 0.5
        tx_op = CutOut(scale=scale)

    elif aug_type.lower() == 'jitter':
        brightness = kwargs['brightness'] if 'brightness' in kwargs else 0.125
        contrast = kwargs['contrast'] if 'contrast' in kwargs else 0.4
        saturation = kwargs['saturation'] if 'saturation' in kwargs else 0.4
        hue = kwargs['hue'] if 'hue' in kwargs else 0
        tx_op = ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    elif aug_type.lower() == 'crop':
        size = kwargs['size'] if 'size' in kwargs else 224
        tx_op = RandomCrop(size=size)

    elif aug_type.lower() == 'hflip':
        tx_op = RandomFlipLeftRight()

    elif aug_type.lower() == 'resize':
        size = kwargs['size'] if 'size' in kwargs else 256
        tx_op = Resize(size)

    elif aug_type.lower() == 'rotate90':
        tx_op = Rotate90()

    elif aug_type.lower() == 'rotate180':
        tx_op = Rotate180()

    elif aug_type.lower() == 'rotate270':
        tx_op = Rotate270()

    elif aug_type.lower() == 'blur':
        prob = kwargs['prob'] if 'prob' in kwargs else 0.5
        tx_op = RandomBlur(prob=prob)
        
    ## augmentation from self-ensembling
    elif aug_type.lower() == 'gaussian':
        tx_op = GaussianNoise(kwargs['std'])

    elif aug_type.lower() == 'translation':
        tx_op = RandomTranslation(kwargs['x_max'], kwargs['y_max'])
        
    elif aug_type.lower() == 'affine':
        tx_op = RandomAffine(kwargs['std'])
        
    elif aug_type.lower() == 'intensity_flip':
        tx_op = RandomIntensityFlip()
        
    elif aug_type.lower() == 'intensity_scaling':
        tx_op = RandomIntensityScaling(kwargs['min'], kwargs['max'])
        
    elif aug_type.lower() == 'intensity_offset':
        tx_op = RandomIntensityOffset(kwargs['min'], kwargs['max'])
    elif aug_type.lower() == 'identity':
        tx_op = Identity()
    else:
        raise NotImplementedError

    return partial(tx_op, is_training=is_training)

def retrieve_augment(aug_list, **kwargs):
    """ Retrieve Augmentation Sequences.

    Args:
        aug_list: List of list of tuples (aug_type, kwargs)
                : +cutout
    """

    def _get_augment_from_shortcut(aug_name):
        if aug_name == 'd':
            return ''
        elif aug_name == 'x':
            # no train time augmentation
            return 'x'
        elif aug_name == 'dh':
            return 'hflip'
        elif aug_name == 'dhc':
            return 'hflip+cutout0.5'
        elif aug_name == 'dhra':
            return 'hflip+randaug2-0.5---all'
        elif aug_name == 'dhrb':
            return 'hflip+randaug2-0.5---cutout'
        elif aug_name == 'dhra1':
            return 'hflip+randaug1-0.5---all'
        elif aug_name == 'dhra3':
            return 'hflip+randaug3-0.5---all'
        elif aug_name == 'dhrc':
            return 'hflip+randaug2-0.5---color'
        elif aug_name == 'dhrg':
            return 'hflip+randaug2-0.5---geo'
        elif aug_name == 'dhrac':
            return 'hflip+randaug2-0.5---all+cutout0.5'
        elif aug_name == 'dhrbc':
            return 'hflip+randaug2-0.5---cutout+cutout0.5'
        elif aug_name == 'dhra1c':
            return 'hflip+randaug1-0.5---all+cutout0.5'
        elif aug_name == 'dhra3c':
            return 'hflip+randaug3-0.5---all+cutout0.5'
        elif aug_name == 'dhrcc':
            return 'hflip+randaug2-0.5---color+cutout0.5'
        elif aug_name == 'dhrgc':
            return 'hflip+randaug2-0.5---geo+cutout0.5'
        elif aug_name == 'dhwj':
            return 'hflip+jitter_b0.125_c0.4_s0.4_h0'
        elif aug_name == 'dhsj':
            return 'hflip+jitter_b0.4_c0.4_s0.4_h0.4'
        elif aug_name == 'dhwjb':
            return 'hflip+jitter_b0.125_c0.4_s0.4_h0+blur0.5'
        elif aug_name == 'dhsjb':
            return 'hflip+jitter_b0.4_c0.4_s0.4_h0.4+blur0.5'
        elif aug_name == 'dhsjcb':
            return 'hflip+jitter_b0.4_c0.4_s0.4_h0.4+cutout0.5+blur0.5'
        else:
            return aug_name

    def _get_augment_args(aug_name, **kwargs):
        aug_args = kwargs
        if aug_name.startswith('cutout'):
            scale = aug_name.replace('cutout', '')
            if len(scale) > 0:
                aug_args['scale'] = float(scale)
            aug_name = 'cutout'
        elif aug_name.startswith('randaug'):
            # [num_layers]-[prob_to_apply]-[magnitude]-[num_layers]-[mode]
            args = aug_name.replace('randaug', '').split('-')
            aug_args['num_layers'] = int(args[0]) if len(args[0]) > 0 else 2
            aug_args['prob_to_apply'] = float(args[1]) if len(args[1]) > 0 else 0.5
            aug_args['magnitude'] = float(args[2]) if len(args[2]) > 0 else None
            aug_args['num_levels'] = int(args[3]) if len(args[3]) > 0 else None
            aug_args['mode'] = args[4] if len(args[4]) > 0 else 'all'
            aug_name = 'randaug'
        elif aug_name.startswith('blur'):
            prob = aug_name.replace('blur', '')
            if len(prob) > 0:
                aug_args['prob'] = float(prob)
            aug_name = 'blur'
        elif aug_name.startswith('jitter'):
            augs = aug_name.replace('jitter', '')
            if len(augs) > 0:
                augs_list = filter(None, augs.split('_'))
                for _aug in augs_list:
                    if   _aug[0] == 'b':
                        aug_args['brightness'] = float(_aug[1:])
                    elif _aug[0] == 'c':
                        aug_args['contrast'] = float(_aug[1:])
                    elif _aug[0] == 's':
                        aug_args['saturation'] = float(_aug[1:])
                    elif _aug[0] == 'h':
                        aug_args['hue'] = float(_aug[1:])
            aug_name = 'jitter'
        return aug_name, aug_args

    def _retrieve_augment(aug_name, is_training=True):
        registered_ops = {
                'base': partial(_base_augment, is_training=is_training),
                'hflip': partial(_hflip_augment, is_training=is_training),
                'jitter': partial(_jitter_augment, is_training=is_training),
                'blur': partial(_blur_augment, is_training=is_training),
                'cutout': partial(_cutout_augment, is_training=is_training),
                'erase': partial(_erase_augment, is_training=is_training),
                'rotate90':  partial(_rotate90_augment,  is_training=is_training),
                'rotate180': partial(_rotate180_augment, is_training=is_training),
                'rotate270': partial(_rotate270_augment, is_training=is_training),
                'randaug': partial(_randaugment, is_training=is_training),
                }
        if aug_name in registered_ops:
            return registered_ops[aug_name]
        else:
            raise NotImplementedError

    def _base_augment(is_training=True, **kwargs):
        size, pad_size = kwargs['size'], int(0.125 * kwargs['size'])
        return [('resize', {'size': size}),
                ('crop', {'size': pad_size}),] if is_training else [('resize', {'size': size})]

    def _jitter_augment(aug=None, is_training=True, **kwargs):
        if aug is None:
            aug = []
        if is_training:
            brightness = kwargs['brightness'] if 'brightness' in kwargs else 0.125
            contrast = kwargs['contrast'] if 'contrast' in kwargs else 0.4
            saturation = kwargs['saturation'] if 'saturation' in kwargs else 0.4
            hue = kwargs['hue'] if 'hue' in kwargs else 0
            return aug + [('jitter', {'brightness': brightness,
                                      'contrast': contrast,
                                      'saturation': saturation,
                                      'hue': hue})]
        return aug

    def _cutout_augment(aug=None, is_training=True, **kwargs):
        if aug is None:
            aug = []
        if is_training:
            scale = kwargs['scale'] if 'scale' in kwargs else 0.5
            return aug + [('cutout', {'scale': scale})]
        return aug

    def _erase_augment(aug=None, is_training=True, **kwargs):
        if aug is None:
            aug = []
        if is_training:
            scale = kwargs['scale'] if 'scale' in kwargs else 0.3
            return aug + [('erase', {'scale': (scale, scale), 'ratio': 1.0})]
        return aug

    def _hflip_augment(aug=None, is_training=True, **kwargs):
        if aug is None:
            aug = []
        if is_training:
            return aug + [('hflip', {})]
        return aug

    def _rotate90_augment(aug=None, is_training=True, **kwargs):
        if aug is None:
            aug = []
        if is_training:
            return aug + [('rotate90', {})]
        return aug

    def _rotate180_augment(aug=None, is_training=True, **kwargs):
        if aug is None:
            aug = []
        if is_training:
            return aug + [('rotate180', {})]
        return aug

    def _rotate270_augment(aug=None, is_training=True, **kwargs):
        if aug is None:
            aug = []
        if is_training:
            return aug + [('rotate270', {})]
        return aug

    def _blur_augment(aug=None, is_training=True, **kwargs):
        if aug is None:
            aug = []
        if is_training:
            prob = kwargs['prob'] if 'prob' in kwargs else 0.5
            return aug + [('blur', {'prob': prob})]
        return aug

    def _randaugment(aug=None, is_training=True, **kwargs):
        if aug is None:
            aug = []
        if is_training:
            num_layers = kwargs['num_layers'] if 'num_layers' in kwargs else 2
            prob_to_apply = kwargs['prob_to_apply'] if 'prob_to_apply' in kwargs else 0.5
            magnitude = kwargs['magnitude'] if 'magnitude' in kwargs else None
            num_levels = kwargs['num_levels'] if 'num_levels' in kwargs else None
            mode = kwargs['mode'] if 'mode' in kwargs else 'all'
            size = kwargs['size'] if 'size' in kwargs else None
            return aug + [('randaug', {'num_layers': num_layers,
                'prob_to_apply': prob_to_apply,
                'magnitude': magnitude,
                'num_levels': num_levels,
                'size': size,
                'mode': mode})]
        return aug

    # Retrieve augmentation ops
    aug_fn_list = []
    for aug_names in aug_list:
        aug_names = _get_augment_from_shortcut(aug_names)
        # chaining from the base augmentation
        if aug_names == 'x':
            # no augmentation
            aug_fn = _retrieve_augment('base', is_training=False)(**kwargs)
        else:
            aug_fn = _retrieve_augment('base', is_training=True)(**kwargs)
            _aug_names = filter(None, aug_names.split('+'))
            for _aug in _aug_names:
                _aug, aug_args = _get_augment_args(_aug, **kwargs)
                aug_fn = _retrieve_augment(_aug, is_training=True)(aug=aug_fn, **aug_args)
        aug_fn_list.append(aug_fn)
        if len(aug_fn_list) == 1:
            # generate test augmentation
            if aug_names == 'x':
                # no augmentation
                test_aug_fn = _retrieve_augment('base', is_training=False)(**kwargs)
            else:
                _aug_names = filter(None, aug_names.split('+'))
                test_aug_fn = _retrieve_augment('base', is_training=False)(**kwargs)
                for _aug in _aug_names:
                    _aug, aug_args = _get_augment_args(_aug, **kwargs)
                    test_aug_fn = _retrieve_augment(_aug, is_training=False)(aug=test_aug_fn, **aug_args)
    return aug_fn_list + [test_aug_fn]


class CutOut(object):
    def __init__(self, scale=0.5, random_scale=False):
        self.scale = scale
        self.random_scale = random_scale

    @staticmethod
    def cutout(image, scale=0.5):
        img_shape = tf.shape(image)
        img_height, img_width, img_depth = img_shape[-3], img_shape[-2], img_shape[-1]
        img_height = tf.cast(img_height, dtype=tf.float32)
        img_width = tf.cast(img_width, dtype=tf.float32)
        cutout_size = (img_height * scale, img_width * scale)
        cutout_size = (tf.maximum(1.0, cutout_size[0]), tf.maximum(1.0, cutout_size[1]))

        def _create_cutout_mask():
            height_loc = tf.round(
                tf.random.uniform(shape=[], minval=0, maxval=img_height))
            width_loc = tf.round(
                tf.random.uniform(shape=[], minval=0, maxval=img_width))

            upper_coord = (tf.maximum(0.0, height_loc - cutout_size[0]//2),
                tf.maximum(0.0, width_loc - cutout_size[1]//2))
            lower_coord = (tf.minimum(img_height, height_loc + cutout_size[0]//2),
                tf.minimum(img_width, width_loc + cutout_size[1]//2))
            mask_height = lower_coord[0] - upper_coord[0]
            mask_width = lower_coord[1] - upper_coord[1]

            padding_dims = ((upper_coord[0], img_height-lower_coord[0]),
                (upper_coord[1], img_width-lower_coord[1]))
            mask = tf.zeros((mask_height, mask_width), dtype=tf.float32)
            mask = tf.pad(mask, tf.cast(padding_dims, dtype=tf.int32),
                constant_values=1.0)
            return tf.expand_dims(mask, -1)
        return _create_cutout_mask() * image

    def __call__(self, image, is_training=True):
        if is_training:
            if self.random_scale:
                scale = tf.random.uniform(shape=[], minval=0.0, maxval=self.scale)
            else:
                scale = self.scale
        return self.cutout(image, scale) if is_training else image


class RandomErase(object):
    def __init__(self, scale=(0.02, 0.3), ratio=3.3, value=0.0):
        self.scale = scale
        self.ratio = ratio
        self.value = value
        assert self.ratio >= 1

    @staticmethod
    def cutout(image, scale=(0.02, 0.3), ratio=3.3, value=0.0):
        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]
        image_depth = tf.shape(image)[2]
        # Sample the center location in the image where the zero mask will be
        # applied.

        def _cutout(img):
            area = tf.cast(image_height*image_width, tf.float32)
            erase_area = tf.random.uniform(shape=[], minval=scale[0], maxval=scale[1])*area
            aspect_ratio = tf.random.uniform(shape=[], minval=1, maxval=ratio)
            aspect_ratio = tf.cond(tf.random.uniform(shape=[])>0.5, lambda: aspect_ratio, lambda: 1.0/aspect_ratio)
            pad_h = tf.cast(tf.math.round(tf.math.sqrt(erase_area*aspect_ratio)), dtype=tf.int32)
            pad_h = tf.minimum(pad_h, image_height-1)
            pad_w = tf.cast(tf.math.round(tf.math.sqrt(erase_area/aspect_ratio)), dtype=tf.int32)
            pad_w = tf.minimum(pad_w, image_width-1)

            cutout_center_height = tf.random.uniform(shape=[], minval=0, maxval=image_height-pad_h, dtype=tf.int32)
            cutout_center_width =  tf.random.uniform(shape=[], minval=0, maxval=image_width-pad_w, dtype=tf.int32)

            lower_pad = cutout_center_height
            upper_pad = tf.maximum(0, image_height - cutout_center_height - pad_h)
            left_pad =  cutout_center_width
            right_pad = tf.maximum(0, image_width - cutout_center_width - pad_w)

            cutout_shape = [image_height - (lower_pad + upper_pad), image_width - (left_pad + right_pad)]
            padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
            mask = tf.pad(tf.zeros(cutout_shape, dtype=img.dtype), padding_dims, constant_values=1)
            mask = tf.expand_dims(mask, -1)
            mask = tf.tile(mask, [1, 1, image_depth])
            img = tf.where(tf.equal(mask, 0), tf.ones_like(img, dtype=img.dtype) * value, img)
            return img
        return _cutout(image)

    def __call__(self, image, is_training=True):
        return self.cutout(image, self.scale, self.ratio, self.value) if is_training else image


class Resize(object):
    def __init__(self, size, method=tf.image.ResizeMethod.BILINEAR):
        self.size = self._check_input(size)
        self.method = method

    def _check_input(self, size):
        if isinstance(size, int):
            size = (size, size)
        elif isinstance(size, (list, tuple)) and len(size) == 1:
            size = size * 2
        else:
            raise TypeError('size must be an integer or list/tuple of integers')
        return size

    def __call__(self, image, is_training=True):
        return tf.image.resize(image, self.size, method=self.method) if is_training else image


class RandomCrop(object):
    def __init__(self, size):
        self.pad = self._check_input(size)

    def _check_input(self, size):
        if isinstance(size, int):
            size = (size, size)
        elif isinstance(size, (list, tuple)):
            if len(size) == 1:
                size = tuple(size) * 2
            elif len(size) > 2:
                size = tuple(size[:2])
        else:
            raise TypeError('size must be an integer or list/tuple of integers')
        return size

    def __call__(self, image, is_training=True):
        if is_training:
            img_size = image.shape[-3:]
            image = tf.pad(image,
                [[self.pad[0]] * 2, [self.pad[1]] * 2, [0] * 2],
                mode='REFLECT')
            image = tf.image.random_crop(image, img_size)
        return image


class RandomFlipLeftRight(object):
    def __init__(self):
        pass

    def __call__(self, image, is_training=True):
        return tf.image.random_flip_left_right(image) if is_training else image


class ColorJitter(object):
    """ Apply color jittering
        This op is equivalent to the following:
        https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.ColorJitter
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness)
        self.contrast = self._check_input(contrast, center=1)
        self.saturation = self._check_input(saturation, center=1)
        self.hue = self._check_input(hue, bound=0.5)

    def _check_input(self, value, center=None, bound=None):
        if bound is not None:
            value = min(value, bound)
        if center is not None:
            value = [center - value, center + value]
            if value[0] == value[1] == center:
                return None
        elif value == 0:
            return None
        return value

    def _get_transforms(self):
        transforms = []
        if self.brightness is not None:
            transforms.append(
                partial(tf.image.random_brightness, max_delta=self.brightness))
        if self.contrast is not None:
            transforms.append(
                partial(tf.image.random_contrast,
                    lower=self.contrast[0], upper=self.contrast[1]))
        if self.saturation is not None:
            transforms.append(
                partial(tf.image.random_saturation,
                    lower=self.saturation[0], upper=self.saturation[1]))
        if self.hue is not None:
            transforms.append(
                partial(tf.image.random_hue, max_delta=self.hue))
        random.shuffle(transforms)
        return transforms

    def __call__(self, image, is_training=True):
        if not is_training:
            return image
        for transform in self._get_transforms():
            image = transform(image)
        return image

class Rotate90(object):
    def __init__(self):
        pass

    def __call__(self, image, is_training=True):
        return tf.image.rot90(image, k=1) if is_training else image

class Rotate180(object):
    def __init__(self):
        pass

    def __call__(self, image, is_training=True):
        return tf.image.rot90(image, k=2) if is_training else image

class Rotate270(object):
    def __init__(self):
        pass

    def __call__(self, image, is_training=True):
        return tf.image.rot90(image, k=3) if is_training else image

class RandomBlur(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, is_training=True):
        if is_training:
            return image
        return simclr_ops.random_blur(image, image.shape[0], image.shape[1], p=self.prob)

class RandAugment(randaug.RandAugment):
    def __init__(self, num_layers=1, prob_to_apply=None, magnitude=None, num_levels=10, size=32, mode='all'):
        super(RandAugment, self).__init__(
            num_layers=num_layers,
            prob_to_apply=prob_to_apply,
            magnitude=magnitude,
            num_levels=num_levels)

        # override TRANSLATE_CONST
        if size == 32:
            randaug.TRANSLATE_CONST = 10.
        elif size == 96:
            randaug.TRANSLATE_CONST = 30.
        elif size == 128:
            randaug.TRANSLATE_CONST = 40.
        elif size == 256:
            randaug.TRANSLATE_CONST = 100.
        else:
            randaug.TRANSLATE_CONST = int(0.3 * size)

        assert mode.upper() in ['ALL', 'COLOR', 'GEO', 'CUTOUT', 'SHAKE'], 'RandAugment mode should be `All`, `COLOR` or `GEO`'
        self.mode = mode.upper()
        self._register_ops()
        if mode.upper() == 'CUTOUT':
            self.cutout_ops = CutOut(scale=0.5, random_scale=True)

    def _apply_one_layer(self, image):
        """Applies one level of augmentation to the image."""
        level = self._get_level()
        branch_fns = []
        for augment_op_name in self.OPS:
            augment_fn = augment_ops.NAME_TO_FUNC[augment_op_name]
            level_to_args_fn = randaug.LEVEL_TO_ARG[augment_op_name]
            def _branch_fn(image=image,
                augment_fn=augment_fn,
                level_to_args_fn=level_to_args_fn):
                args = [image] + list(level_to_args_fn(level))
                return augment_fn(*args)
            branch_fns.append(_branch_fn)
        branch_index = tf.random.uniform(
            shape=[], maxval=len(branch_fns), dtype=tf.int32)
        aug_image = tf.switch_case(branch_index, branch_fns, default=lambda: image)
        if self.prob_to_apply is not None:
            return tf.cond(
                tf.random.uniform(shape=[], dtype=tf.float32) < self.prob_to_apply,
                lambda: aug_image,
                lambda: image)
        else:
            return aug_image

    def _register_ops(self):
        if self.mode == 'ALL':
            self.OPS = [
                'AutoContrast', 'Equalize', 'Posterize', 'Solarize',
                'Color', 'Contrast', 'Brightness', 'Identity',
                'Invert', 'Sharpness', 'SolarizeAdd',
            ]
            self.OPS += [
                'Rotate', 'ShearX', 'ShearY',
                'TranslateX', 'TranslateY',
            ]
        elif self.mode == 'CUTOUT':
            self.OPS = [
                'AutoContrast', 'Equalize', 'Posterize', 'Solarize',
                'Color', 'Contrast', 'Brightness', 'Identity',
                'Invert', 'Sharpness', 'SolarizeAdd',
            ]
            self.OPS += [
                'Rotate', 'ShearX', 'ShearY',
                'TranslateX', 'TranslateY',
            ]
        elif self.mode == 'COLOR':
            self.OPS = [
                'AutoContrast', 'Equalize', 'Posterize', 'Solarize',
                'Color', 'Contrast', 'Brightness', 'Identity',
                'Invert', 'Sharpness', 'SolarizeAdd',
            ]
        elif self.mode == 'GEO':
            self.OPS = [
                'Rotate', 'ShearX', 'ShearY',
                'TranslateX', 'TranslateY', 'Identity',
            ]
        elif self.mode == 'SHAKE':
            self.OPS = [
                'TranslateX', 'TranslateY',
            ]
        else:
            raise NotImplementedError

    def wrap(self, image):
        image += tf.constant(1.0, image.dtype)
        image *= tf.constant(255.0/2.0, image.dtype)
        image = tf.saturate_cast(image, tf.uint8)
        return image

    def unwrap(self, image):
        image = tf.cast(image, tf.float32)
        image /= tf.constant(255.0/2.0, image.dtype)
        image -= tf.constant(1.0, image.dtype)
        return image

    def _apply_cutout(self, image):
        # cutout assumes pixels are in [-1, 1]
        aug_image = self.unwrap(image)
        aug_image = self.cutout_ops(aug_image)
        aug_image = self.wrap(aug_image)
        if self.prob_to_apply is not None:
            return tf.cond(
                tf.random.uniform(shape=[], dtype=tf.float32) < self.prob_to_apply,
                lambda: aug_image,
                lambda: image)
        else:
            return aug_image

    def __call__(self, image, y, is_training=True):
        if not is_training:
            return image, y

        ##TODO:
        if len(image.shape) == 4:
            image0, y = self(image[0], y, is_training)
            image1, y = self(image[1], y, is_training)
            return [image0, image1], y
            
        image = self.wrap(image)
        if self.mode == 'CUTOUT':
            for _ in range(self.num_layers):
                # made an exception for cutout
                image = tf.cond(
                    tf.random.uniform(shape=[], dtype=tf.float32) < tf.divide(
                        tf.constant(1.0), tf.cast(len(self.OPS) + 1, dtype=tf.float32)),
                    lambda: self._apply_cutout(image),
                    lambda: self._apply_one_layer(image))
            image = self.unwrap(image)
            #return self.unwrap(image), y
        else:
            for _ in range(self.num_layers):
                image = self._apply_one_layer(image)
            #return self.unwrap(image), y
            image = self.unwrap(image)
        return image, y


class DataAugmentation:
    
    def __call__(self, x, y, is_training=True):
        if len(x.shape) == 4:
            assert(x.shape[0] == 2)
            x = [self._apply(x[0], is_training=is_training), self._apply(x[1], is_training=is_training)]
        else:
            x = self._apply(x, is_training=is_training)
        return x, y

    

class GaussianNoise(DataAugmentation):
    def __init__(self, std=0.0):
        super().__init__()
        self.std = std

    def _apply(self, image, is_training=True):
        x = image+tf.random.normal(image.shape, stddev=self.std) if is_training else image
        return x
    

class RandomIntensityScaling(DataAugmentation):
    def __init__(self, mn=1.0, mx=1.0):
        super().__init__()
        
        self.mn = mn
        self.mx = mx

    def _apply(self, image, is_training=True):
        assert(len(image.shape) == 2 or len(image.shape) == 3)
        scaling_rnd = tf.random.uniform([1]*len(image.shape), minval=self.mn, maxval=self.mx)
        x = image * scaling_rnd if is_training else image
        return x
    

class RandomIntensityOffset(DataAugmentation):
    def __init__(self, mn=0.0, mx=0.0):
        super().__init__()
        
        self.mn = mn
        self.mx = mx

    def _apply(self, image, is_training=True):
        assert(len(image.shape) == 2 or len(image.shape) == 3)
        offset_rnd = tf.random.uniform([1]*len(image.shape), minval=self.mn, maxval=self.mx)
        x = image + offset_rnd if is_training else image
        return x


class RandomIntensityFlip(DataAugmentation):
    def __init__(self):
        super().__init__()
        
    def _apply(self, image, is_training=True):
        assert(len(image.shape) == 2 or len(image.shape) == 3)
        flip_rnd = tf.cast(tf.random.uniform([1]*len(image.shape), minval=0, maxval=2, dtype=tf.int32)*2 - 1, tf.float32)
        x = image*flip_rnd if is_training else image
        return x


class RandomTranslation(DataAugmentation):
    def __init__(self, x=0.0, y=0.0):
        super().__init__()
        
        self.x = x
        self.y = y

    def _apply(self, image, is_training=True):
        tform = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        tform[2] += tf.random.uniform((), minval=-self.x, maxval=self.x)
        tform[5] += tf.random.uniform((), minval=-self.y, maxval=self.y)
        x = tfa.image.transform(image, tform, interpolation='BILINEAR') if is_training else image
        return x

    
class RandomAffine(DataAugmentation):
    def __init__(self, std=0.0):
        super().__init__()
        
        self.std = std

    def _apply(self, image, is_training=True):
        tform = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        tform[0] += tf.random.normal((), stddev=self.std)
        tform[1] += tf.random.normal((), stddev=self.std)
        tform[3] += tf.random.normal((), stddev=self.std)
        tform[4] += tf.random.normal((), stddev=self.std)
        x = tfa.image.transform(image, tform, interpolation='BILINEAR') if is_training else image
        return x

    
    
if __name__== "__main__":
    # aug_list = [('resize',{'size': 256}),
    #             ('crop', {'size': 224}),
    #             ('jitter', {'brightness': 0.4, 'contrast': 0.4, 'saturation': 0.4}),
    #             ('hflip', {}),
    #             ('cutout', {'scale': (0.02, 0.15)}),]
    # aug_ops = compose_augment_seq(aug_list, is_training=False)


    aug_list = [('gaussian', {'std': 0.1}),
                ('translation', {'x_max': 0.0, 'y_max': 5.0}),
                ('affine', {'std': 0.5}),
                ('intensity_flip', {}),
                ('intensity_scaling', {'min': 0.25, 'max': 1.5}),
                ('intensity_offset', {'min': -0.5, 'max': 0.5})
    ]

    from data import MNIST
    tmp_root = 'tmp'
    for aug in aug_list:
        dsld = MNIST(batch_size=10, aug_list=[aug], train_shuffle=False)
        dsld_ori = MNIST(batch_size=10, train_shuffle=False)
                
        for (x_ori, _), (x, _) in zip(dsld_ori.train, dsld.train):
            for i, (x_ori_b, x_b) in enumerate(zip(x_ori, x)):
                x_concat = tf.concat((x_ori_b, x_b), 1) 
                x_png = tf.image.encode_png(tf.cast(x_concat*255, tf.uint8))
                tf.io.write_file(os.path.join(tmp_root, aug[0], "%d.png"%(i)), x_png)
                
            break

class Identity:
    def __call__(self, x, y, is_training=True):
        return x, y
