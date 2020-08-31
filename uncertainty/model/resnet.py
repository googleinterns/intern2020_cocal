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

import sys
import numpy as np
from functools import partial
from model.resnet_util import (ResNet, WideResNet,
    basic_stack1, basic_stack2, basic_stack3,
    bottleneck_stack1, bottleneck_stack2, bottleneck_stack3)


__all__ = ['ResNet10', 'ResNet18', 'ResNet50', 'ResNet101', 'ResNet152',
           'ResNet10V2', 'ResNet18V2', 'ResNet50V2', 'ResNet101V2', 'ResNet152V2',
           'WRN28', 'WRN36']

BLOCK = {
    'ResNet10': [1, 1, 1, 1],
    'ResNet18': [2, 2, 2, 2],
    'ResNet50': [3, 4, 6, 3],
    'ResNet101': [3, 4, 23, 3],
    'ResNet152': [3, 8, 36, 3],
    'ResNet10v2': [1, 1, 1, 1],
    'ResNet18v2': [2, 2, 2, 2],
    'ResNet50v2': [3, 4, 6, 3],
    'ResNet101v2': [3, 4, 23, 3],
    'ResNet152v2': [3, 8, 36, 3],
    'WRN28': [4, 4, 4],
    'WRN36': [4, 4, 4, 4],
}

EXPANSION = {
    'ResNet10': 1,
    'ResNet18': 1,
    'ResNet50': 4,
    'ResNet101':4,
    'ResNet152': 4,
    'ResNet10v2': 1,
    'ResNet18v2': 1,
    'ResNet50v2': 4,
    'ResNet101v2': 4,
    'ResNet152v2': 4,
    'WRN28': 1,
    'WRN36': 1,
}

STACK = {
    'ResNet10': basic_stack1,
    'ResNet18': basic_stack1,
    'ResNet50': bottleneck_stack1,
    'ResNet101': bottleneck_stack1,
    'ResNet152': bottleneck_stack1,
    'ResNet10v2': basic_stack2,
    'ResNet18v2': basic_stack2,
    'ResNet50v2': bottleneck_stack2,
    'ResNet101v2': bottleneck_stack2,
    'ResNet152v2': bottleneck_stack2,
    'WRN28': basic_stack3,
    'WRN36': basic_stack3,
}


def ResNetV1(
    arch='ResNet18',
    head='linear',
    head_mlp=None,
    input_shape=None,
    num_class=1000,
    pooling=None,
    activation='relu',
    bn_sync=False,
    width=1.0):
    """Instantiates the ResNet architecture."""

    def stack_fn(x, arch, width=1.0):
        block, stack, expansion = BLOCK[arch], STACK[arch], EXPANSION[arch]
        x = stack(x, int(64*width), block[0],  expansion=expansion,
            stride1=1,
            activation=activation, bn_sync=bn_sync, name='conv2')
        x = stack(x, int(128*width), block[1], expansion=expansion,
            activation=activation, bn_sync=bn_sync, name='conv3')
        x = stack(x, int(256*width), block[2], expansion=expansion,
            activation=activation, bn_sync=bn_sync, name='conv4')
        return stack(x, int(512*width), block[3], expansion=expansion,
            activation=activation, bn_sync=bn_sync, name='conv5')

    return ResNet(
        stack_fn=partial(stack_fn, arch=arch, width=width),
        preact=False,
        model_name='{}_width{:g}_{}'.format(arch, width, activation),
        head=head,
        head_mlp=head_mlp,
        input_shape=input_shape,
        pooling=pooling,
        activation=activation,
        bn_sync=bn_sync,
        num_class=num_class)


def ResNetV2(
    arch='ResNet18',
    head='linear',
    head_mlp='',
    input_shape=None,
    num_class=1000,
    pooling=None,
    activation='relu',
    bn_sync=False,
    width=1.0):
    """Instantiates the ResNet architecture."""

    def stack_fn(x, arch, width=1.0):
        block, stack, expansion = BLOCK[arch], STACK[arch], EXPANSION[arch]
        x = stack(x, int(64*width), block[0], expansion=expansion,
            activation=activation, bn_sync=bn_sync, name='conv2')
        x = stack(x, int(128*width), block[1], expansion=expansion,
            activation=activation, bn_sync=bn_sync, name='conv3')
        x = stack(x, int(256*width), block[2], expansion=expansion,
            activation=activation, bn_sync=bn_sync, name='conv4')
        return stack(x, int(512*width), block[3], expansion=expansion,
            stride1=1,
            activation=activation, bn_sync=bn_sync, name='conv5')

    return ResNet(
        stack_fn=partial(stack_fn, arch=arch, width=width),
        preact=True,
        model_name='{}_width{:g}_{}'.format(arch, width, activation),
        head=head,
        head_mlp=head_mlp,
        input_shape=input_shape,
        pooling=pooling,
        activation=activation,
        bn_sync=bn_sync,
        num_class=num_class)


def WRN(
    arch='WRN28',
    head='linear',
    head_mlp='',
    input_shape=None,
    num_class=1000,
    pooling=None,
    activation='leaky_relu',
    bn_sync=False,
    width=1.0):
    """Instantiates the ResNet architecture."""

    def stack_fn(x, arch, width=1.0):
        block, stack, expansion = BLOCK[arch], STACK[arch], EXPANSION[arch]
        for i, b in enumerate(block):
            x = stack(x, int((16 << i)*width), b,
                expansion=expansion,
                stride1=2 if i > 0 else 1,
                activation=activation,
                bn_sync=bn_sync,
                name='conv%d' %(i+2))
        return x

    return WideResNet(
        stack_fn=partial(stack_fn, arch=arch, width=width),
        preact=True,
        model_name='{}_width{:g}_{}'.format(arch, width, activation),
        head=head,
        head_mlp=head_mlp,
        input_shape=input_shape,
        pooling=pooling,
        activation=activation,
        bn_sync=bn_sync,
        num_class=num_class)


def ResNet10(**kwargs):
    return ResNetV1(arch='ResNet10', **kwargs)


def ResNet18(**kwargs):
    return ResNetV1(arch='ResNet18', **kwargs)


def ResNet50(**kwargs):
    return ResNetV1(arch='ResNet50', **kwargs)


def ResNet101(**kwargs):
    return ResNetV1(arch='ResNet101', **kwargs)


def ResNet152(**kwargs):
    return ResNetV2(arch='ResNet152', **kwargs)


def ResNet10V2(**kwargs):
    return ResNetV2(arch='ResNet10v2', **kwargs)


def ResNet18V2(**kwargs):
    return ResNetV2(arch='ResNet18v2', **kwargs)


def ResNet50V2(**kwargs):
    return ResNetV2(arch='ResNet50v2', **kwargs)


def ResNet101V2(**kwargs):
    return ResNetV2(arch='ResNet101v2', **kwargs)


def ResNet152V2(**kwargs):
    return ResNetV2(arch='ResNet152v2', **kwargs)


def WRN28(**kwargs):
    return WRN(arch='WRN28', **kwargs)


def WRN36(**kwargs):
    return WRN(arch='WRN36', **kwargs)
