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
""" ResNet models for Keras.

Reference paper:
    - [Deep Residual Learning for Image Recognition]
        (https://arxiv.org/abs/1512.03385) (CVPR 2015)
"""

import os

import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.keras import layers
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine import training


BN_MOM = 0.9
BN_EPS = 1e-05
BN_AXIS = 3 if backend.image_data_format() == 'channels_last' else 1
BN_SYNC=True

# Utils
def convnxn(x, filters=64, kernel_size=3,
    strides=1, use_bias=False,
    kernel_initializer=initializers.VarianceScaling(
        scale=2.0, mode='fan_out'),
    name=None):
    return layers.Conv2D(
        filters=filters, kernel_size=kernel_size,
        strides=strides, padding='SAME',
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        name=name)(x)


def conv3x3(x, filters=64,
    strides=1, use_bias=False,
    kernel_initializer=initializers.VarianceScaling(
        scale=2.0, mode='fan_out'),
    name=None):
    return convnxn(x, filters, 3, strides, use_bias,
        kernel_initializer, name)


def conv1x1(x, filters=64,
    strides=1, use_bias=False,
    kernel_initializer=initializers.VarianceScaling(
        scale=2.0, mode='fan_out'),
    name=None):
    return convnxn(x, filters, 1, strides, use_bias,
        kernel_initializer, name)


def batchnorm(x, bn_axis=BN_AXIS, bn_mom=BN_MOM, bn_eps=BN_EPS, bn_sync=False, name=None):
    if bn_sync is True:
        return layers.SyncBatchNormalization(
            axis=bn_axis, momentum=bn_mom, epsilon=bn_eps, name=name)(x)
    else:
        return layers.BatchNormalization(
            axis=bn_axis, momentum=bn_mom, epsilon=bn_eps, name=name)(x)


def nonlinearity(x, layer_activation='relu', name=None):
    if layer_activation == 'relu':
        return layers.Activation('relu', name=name)(x)
    elif layer_activation == 'leaky_relu':
        return layers.LeakyReLU(alpha=0.1, name=name)(x)
    else:
        return x

def get_head(x, arch, dims=None, num_class=2,
    bn_mom=BN_MOM, bn_eps=BN_EPS, bn_sync=BN_SYNC,
    classifier_activation='linear'):
    # No head
    share = x
    if arch in [None, False]:
        return x
    # Single or dual heads
    registered_arch = [
        'linear',
        'one_head', 'one_head_bn', 'one_head_norm',
        'two_head', 'two_head_bn', 'two_head_old_bn',
        'two_head_cls', 'two_head_cls_bn'
    ]
    assert arch in registered_arch
    if arch == 'linear':
        embeds = x
        logits = layers.Dense(
            units=num_class, activation=classifier_activation,
            name='predictions')(embeds)
    elif arch in ['one_head', 'one_head_bn']:
        embeds = x
        if dims is not None:
            for i, d in enumerate(dims[:-1]):
                embeds = layers.Dense(
                units=d, name='embeds_%d' %i)(embeds)
                if arch.endswith('bn'):
                    embeds = batchnorm(embeds,
                        bn_axis=-1,
                        bn_mom=bn_mom, bn_eps=bn_eps,
                        bn_sync=bn_sync,
                        name='embeds_%d_bn' %i)
                embeds = layers.Activation('relu',
                    name='embeds_%d_relu' %i)(embeds)
            embeds = layers.Dense(
                units=dims[-1], activation='linear',
                use_bias=num_class > 0,
                name='embeds')(embeds)
        if num_class > 0:
            logits = layers.Dense(
                units=num_class, activation=classifier_activation,
                name='predictions')(embeds)
        else:
            logits = embeds
    elif arch == 'one_head_norm':
        embeds = x
        if dims is not None:
            for i, d in enumerate(dims[:-1]):
                embeds = layers.Dense(
                    units=d, activation=backend.relu,
                    name='embeds_%d' %i)(embeds)
            embeds = layers.Dense(
                units=dims[-1], activation='linear',
                use_bias=num_class > 0,
                name='embeds')(embeds)
            embeds = backend.l2_normalize(embeds, axis=1)
        if num_class > 0:
            logits = layers.Dense(
                units=num_class, activation=classifier_activation,
                name='predictions')(embeds)
        else:
            logits = embeds
    elif arch in ['two_head', 'two_head_bn']:
        assert dims is not None
        assert num_class > 0
        logits = embeds = x
        for i, d in enumerate(dims[:-1]):
            embeds = layers.Dense(
                units=d, name='embeds_%d' %i)(embeds)
            if arch.endswith('bn'):
                embeds = batchnorm(embeds,
                    bn_axis=-1,
                    bn_mom=bn_mom, bn_eps=bn_eps,
                    bn_sync=bn_sync,
                    name='embeds_%d_bn' %i)
            embeds = layers.Activation('relu',
                name='embeds_%d_relu' %i)(embeds)
        embeds = layers.Dense(
            units=dims[-1], activation='linear',
            use_bias=False,
            name='embeds')(embeds)
        logits = layers.Dense(
            units=num_class, activation=classifier_activation,
            name='predictions')(logits)
    elif arch in ['two_head_cls', 'two_head_cls_bn']:
        assert dims is not None
        assert num_class > 0
        logits = embeds = x
        for i, d in enumerate(dims):
            embeds = layers.Dense(
                units=d, name='embeds_%d' %i)(embeds)
            if arch.endswith('bn'):
                embeds = batchnorm(embeds,
                    bn_axis=-1,
                    bn_mom=bn_mom, bn_eps=bn_eps,
                    bn_sync=bn_sync,
                    name='embeds_%d_bn' %i)
            embeds = layers.Activation('relu',
                name='embeds_%d_relu' %i)(embeds)
        embeds = layers.Dense(
            units=num_class, activation=classifier_activation,
            name='embeds')(embeds)
        logits = layers.Dense(
            units=num_class, activation=classifier_activation,
            name='predictions')(logits)
    elif arch in ['two_head_old_bn']:
        assert dims is not None
        assert num_class > 0
        logits = embeds = x
        for i, d in enumerate(dims[:-1]):
            embeds = layers.Dense(
                units=d, name='embeds_%d' %i)(embeds)
            if arch.endswith('bn'):
                embeds = batchnorm(embeds,
                    bn_axis=-1,
                    bn_mom=BN_MOM, bn_eps=BN_EPS,
                    bn_sync=bn_sync,
                    name='embeds_%d_bn' %i)
            embeds = layers.Activation('relu',
                name='embeds_%d_relu' %i)(embeds)
        embeds = layers.Dense(
            units=dims[-1], activation='linear',
            use_bias=False,
            name='embeds')(embeds)
        logits = layers.Dense(
            units=num_class, activation=classifier_activation,
            name='predictions')(logits)
    elif arch in ['two_head_cls_old_bn']:
        assert dims is not None
        assert num_class > 0
        logits = embeds = x
        for i, d in enumerate(dims):
            embeds = layers.Dense(
                units=d, name='embeds_%d' %i)(embeds)
            if arch.endswith('bn'):
                embeds = batchnorm(embeds,
                    bn_axis=-1,
                    bn_mom=BN_MOM, bn_eps=BN_EPS,
                    bn_sync=bn_sync,
                    name='embeds_%d_bn' %i)
            embeds = layers.Activation('relu',
                name='embeds_%d_relu' %i)(embeds)
        embeds = layers.Dense(
            units=num_class, activation=classifier_activation,
            name='embeds')(embeds)
        logits = layers.Dense(
            units=num_class, activation=classifier_activation,
            name='predictions')(logits)
    return {'logits': logits, 'embeds': embeds, 'share': share, 'y_pred': tf.math.argmax(logits, -1)}


def ResNet(
    stack_fn,
    preact,
    model_name='resnet',
    head='one_head',
    head_mlp='',
    input_shape=None,
    pooling=None,
    activation='relu',
    bn_sync=BN_SYNC,
    num_class=1000,
    **kwargs):
    """Instantiates the ResNet, ResNetV2, and ResNeXt architecture.

    Reference paper:
    - [Deep Residual Learning for Image Recognition]
        (https://arxiv.org/abs/1512.03385) (CVPR 2015)

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    Caution: Be sure to properly pre-process your inputs to the application.
    Please see `applications.resnet.preprocess_input` for an example.

    Arguments:
        stack_fn: a function that returns output tensor for the
            stacked residual blocks.
        preact: whether to use pre-activation or not
            (True for ResNetV2, False for ResNet and ResNeXt).
        model_name: string, model name.
        head: whether to include the fully-connected
            layer at the top of the network.
        input_shape: optional shape tuple, only to be specified
            if `head` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
        pooling: optional pooling mode for feature extraction
            when `head` is `False`.
            - `None` means that the output of the model will be
                    the 4D tensor output of the
                    last convolutional layer.
            - `avg` means that global average pooling
                    will be applied to the output of the
                    last convolutional layer, and thus
                    the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                    be applied.
        num_class: optional number of classes to classify images
            into, only to be specified if `head` is True, and
            if no `weights` argument is specified.
        **kwargs: For backwards compatibility only.
    Returns:
        A `keras.Model` instance.

    Raises:
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    # Input layer.
    inputs = img_input = layers.Input(shape=input_shape)

    # Conv1 block.
    # CIFAR (32x32) or STL-10 (96x96) use small input; otherwise use large input
    if input_shape[0] == 256:
        kernel_size, stride, maxpool = 7, 2, True
    elif input_shape[0] in [96, 128]:
        kernel_size, stride, maxpool = 5, 1, True
    elif input_shape[0] == 32:
        kernel_size, stride, maxpool = 3, 1, False
    else:
        raise NotImplementedError
    x = convnxn(img_input, filters=64, kernel_size=kernel_size,
        strides=stride, use_bias=preact,
        name='conv1_conv')
    if not preact:
        x = batchnorm(x, bn_sync=bn_sync, name='conv1_bn')
        x = nonlinearity(x, layer_activation=activation, name='conv1_'+activation)
    if maxpool:
        x = layers.MaxPooling2D(3, strides=2, padding='SAME',
            name='pool1_pool')(x)

    # Conv2 to Conv5 blocks
    x = stack_fn(x)
    if preact:
        x = batchnorm(x, bn_sync=bn_sync, name='post_bn')
        x = nonlinearity(x, layer_activation=activation, name='post_'+activation)

    # Pooling layer
    if pooling in ['avg', None]:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPooling2D(name='max_pool')(x)

    # # Dropout layer
    # x = layers.Dropout(0.5)(x)
    # print("!! dropout layer")
        
    # Classifier and/or embedding head
    outputs = get_head(
        x, arch=head, dims=head_mlp, num_class=num_class,
        bn_sync=bn_sync,
        classifier_activation='linear')

    # Create model.
    if bn_sync:
        model_name += '_sbn'
    if not head in ['linear', None, False]:
        model_name += '_{}'.format(head)
        if head_mlp is not None and len(head_mlp) > 0:
            model_name += '_mlp' + '_'.join(['%d'%d for d in head_mlp])

    model = training.Model(inputs, outputs, name=model_name)
    res = model(tf.zeros((1, *input_shape)), training=False)
    model.dim_feat = tf.reshape(res['embeds'], (-1)).shape[0]
    
    return model


def WideResNet(
    stack_fn,
    preact=True,
    model_name='resnet',
    head='one_head',
    head_mlp='',
    input_shape=None,
    pooling=None,
    activation='leaky_relu',
    bn_sync=BN_SYNC,
    num_class=1000,
    **kwargs):

    # Input layer.
    inputs = img_input = layers.Input(shape=input_shape)

    # Conv1 block.
    # CIFAR (32x32) or STL-10 (96x96) use small input; otherwise use large input
    if input_shape[0] == 256:
        kernel_size, stride, maxpool = 7, 2, True
    elif input_shape[0] in [96, 128]:
        kernel_size, stride, maxpool = 3, 1, False
    elif input_shape[0] == 32:
        kernel_size, stride, maxpool = 3, 1, False
    else:
        raise NotImplementedError
    x = convnxn(img_input, filters=16, kernel_size=kernel_size,
        strides=stride, use_bias=False,
        name='conv1_conv')
    if not preact:
        x = batchnorm(x, bn_mom=0.999, bn_eps=0.001, bn_sync=bn_sync, name='conv1_bn')
        x = nonlinearity(x, layer_activation=activation, name='conv1_'+activation)
    if maxpool:
        x = layers.MaxPooling2D(3, strides=2, padding='SAME',
            name='pool1_pool')(x)

    # Conv2 to Conv5 blocks
    x = stack_fn(x)
    if preact:
        x = batchnorm(x, bn_mom=0.999, bn_eps=0.001, bn_sync=bn_sync, name='post_bn')
        x = nonlinearity(x, layer_activation=activation, name='post_'+activation)

    # Pooling layer
    if pooling in ['avg', None]:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPooling2D(name='max_pool')(x)

    # Classifier and/or embedding head
    outputs = get_head(
        x, arch=head, dims=head_mlp, num_class=num_class,
        bn_mom=0.999, bn_eps=0.001, bn_sync=bn_sync,
        classifier_activation='linear')

    # Create model.
    if bn_sync:
        model_name += '_sbn'
    if not head in ['linear', None, False]:
        model_name += '_{}'.format(head)
        if head_mlp is not None and len(head_mlp) > 0:
            model_name += '_mlp' + '_'.join(['%d'%d for d in head_mlp])
    return training.Model(inputs, outputs, name=model_name)


def block1(x,
    filters,
    bottleneck=False,
    stride=1,
    expansion=1,
    activation='relu',
    bn_sync=BN_SYNC,
    name=None):
    """A basic residual block.

    Arguments:
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default True, use convolution shortcut if True,
                otherwise identity shortcut.
        name: string, block label.

    Returns:
        Output tensor for the residual block.
    """
    conv_shortcut = (stride != 1) or (expansion*filters != x.shape[3])
    if conv_shortcut:
        shortcut = conv1x1(x, filters=expansion*filters, strides=stride,
            name=name+'_0_conv')
        shortcut = batchnorm(shortcut, bn_sync=bn_sync, name=name+'_0_bn')
    else:
        shortcut = x
    # First conv.
    if bottleneck:
        x = conv1x1(x, filters=filters, strides=1, name=name+'_1_conv')
        x = batchnorm(x, bn_sync=bn_sync, name=name+'_1_bn')
        x = nonlinearity(x, layer_activation=activation, name=name+'_1_'+activation)
    # Second conv.
    idx = 2 if bottleneck else 1
    x = conv3x3(x, filters=filters, strides=stride, name=name+'_%d_conv' %idx)
    x = batchnorm(x, bn_sync=bn_sync, name=name+'_%d_bn' %idx)
    x = nonlinearity(x, layer_activation=activation, name=name+'_%d_%s' %(idx, activation))
    # Last conv.
    last_conv = conv1x1 if bottleneck else conv3x3
    x = last_conv(x, filters=expansion*filters, strides=1, name=name+'_%d_conv' %(idx+1))
    x = batchnorm(x, bn_sync=bn_sync, name=name+'_%d_bn' %(idx+1))
    # Skip connection.
    x = layers.Add(name=name+'_add')([shortcut, x])
    x = nonlinearity(x, layer_activation=activation, name=name+'_out_'+activation)
    return x


def block2(x,
    filters,
    bottleneck=False,
    stride=1,
    expansion=4,
    activation='relu',
    bn_sync=BN_SYNC,
    conv_shortcut=False,
    name=None):
    """A residual block.

    Arguments:
            x: input tensor.
            filters: integer, filters of the bottleneck layer.
            stride: default 1, stride of the first layer.
            conv_shortcut: default False, use convolution shortcut if True,
                otherwise identity shortcut.
            name: string, block label.

    Returns:
        Output tensor for the residual block.
    """
    x = batchnorm(x, bn_sync=bn_sync, name=name+'_preact_bn')
    x = nonlinearity(x, layer_activation=activation, name=name+'_preact_'+activation)
    if conv_shortcut:
        shortcut = conv1x1(x, filters=expansion*filters, strides=stride,
            name=name+'_0_conv')
    else:
        shortcut = layers.MaxPooling2D(1, strides=stride)(x) if stride > 1 else x
    # First conv.
    if bottleneck:
        x = conv1x1(x, filters=filters, strides=1, name=name+'_1_conv')
        x = batchnorm(x, bn_sync=bn_sync, name=name+'_1_bn')
        x = nonlinearity(x, layer_activation=activation, name=name+'_1_'+activation)
    # Second conv.
    idx = 2 if bottleneck else 1
    x = conv3x3(x, filters=filters, strides=stride, name=name+'_%d_conv' %idx)
    x = batchnorm(x, bn_sync=bn_sync, name=name+'_%d_bn' %idx)
    x = nonlinearity(x, layer_activation=activation, name=name+'_%d_%s' %(idx, activation))
    # Last conv
    last_conv = conv1x1 if bottleneck else conv3x3
    x = last_conv(x, filters=expansion*filters, strides=1, name=name+'_%d_conv' %(idx+1))
    # Skip connection.
    x = layers.Add(name=name+'_add')([shortcut, x])
    return x


def block3(x,
    filters,
    bottleneck=False,
    stride=1,
    expansion=1,
    activation='leaky_relu',
    bn_sync=BN_SYNC,
    conv_shortcut=False,
    use_bias=False,
    name=None):
    """ A residual block.

    """
    preact = batchnorm(x, bn_mom=0.999, bn_eps=0.001, bn_sync=bn_sync, name=name+'_preact_bn')
    preact = nonlinearity(preact, layer_activation=activation,
        name=name+'_preact_'+activation)
    shortcut = preact if conv_shortcut else x
    if shortcut.shape[3] != expansion*filters:
        shortcut = conv1x1(shortcut, filters=expansion*filters, strides=stride,
            use_bias=use_bias, name=name+'_0_conv')
    # First conv.
    if bottleneck:
        x = conv1x1(preact, filters=filters, strides=1,
            use_bias=use_bias, name=name+'_1_conv')
        x = batchnorm(x, bn_mom=0.999, bn_eps=0.001, bn_sync=bn_sync, name=name+'_1_bn')
        x = nonlinearity(x, layer_activation=activation, name=name+'_1_'+activation)
    # Second conv.
    idx = 2 if bottleneck else 1
    x = conv3x3(x if bottleneck else preact, filters=filters, strides=stride,
        use_bias=use_bias, name=name+'_%d_conv' %idx)
    x = batchnorm(x, bn_mom=0.999, bn_eps=0.001, bn_sync=bn_sync, name=name+'_%d_bn' %idx)
    x = nonlinearity(x, layer_activation=activation, name=name+'_%d_%s' %(idx, activation))
    # Second conv.
    last_conv = conv1x1 if bottleneck else conv3x3
    x = last_conv(x, filters=expansion*filters, strides=1,
        use_bias=use_bias, name=name+'_%d_conv' %(idx+1))
    # Skip connection.
    x = layers.Add(name=name+'_add')([shortcut, x])
    return x


def stack_v1(x,
    filters,
    blocks,
    bottleneck=False,
    stride1=2,
    expansion=4,
    activation='relu',
    bn_sync=BN_SYNC,
    name=None):
    """A set of stacked residual blocks.

    Arguments:
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        name: string, stack label.

    Returns:
        Output tensor for the stacked blocks.
    """
    x = block1(x, filters,
        bottleneck=bottleneck,
        stride=stride1,
        expansion=expansion,
        activation=activation,
        bn_sync=bn_sync,
        name=name+'_block1')
    for i in range(1, blocks):
        x = block1(x, filters,
            bottleneck=bottleneck,
            expansion=expansion,
            activation=activation,
            bn_sync=bn_sync,
            name=name+'_block'+str(i+1))
    return x


def stack_v2(x,
    filters,
    blocks,
    bottleneck=False,
    stride1=2,
    expansion=4,
    activation='relu',
    bn_sync=BN_SYNC,
    name=None):
    """A set of stacked residual blocks.

    Arguments:
            x: input tensor.
            filters: integer, filters of the bottleneck layer in a block.
            blocks: integer, blocks in the stacked blocks.
            stride1: default 2, stride of the first layer in the first block.
            name: string, stack label.

    Returns:
            Output tensor for the stacked blocks.
    """
    if blocks == 1:
        x = block2(x, filters,
            bottleneck=bottleneck,
            stride=stride1,
            expansion=expansion,
            activation=activation,
            bn_sync=bn_sync,
            conv_shortcut=True and (stride1 != 1),
            name=name+'_block1')
        return x
    else:
        x = block2(x, filters,
            bottleneck=bottleneck,
            expansion=expansion,
            activation=activation,
            bn_sync=bn_sync,
            conv_shortcut=True and (stride1 != 1),
            name=name+'_block1')
        for i in range(1, blocks-1):
            x = block2(x, filters,
                bottleneck=bottleneck,
                expansion=expansion,
                activation=activation,
                bn_sync=bn_sync,
                name=name+'_block' + str(i+1))
        x = block2(x, filters,
            bottleneck=bottleneck,
            stride=stride1,
            expansion=expansion,
            activation=activation,
            bn_sync=bn_sync,
            name=name+'_block' + str(blocks))
        return x


def stack_v3(x,
    filters,
    blocks,
    bottleneck=False,
    stride1=2,
    expansion=4,
    activation='leaky_relu',
    bn_sync=BN_SYNC,
    name=None):
    """A set of stacked residual blocks.

    Arguments:
            x: input tensor.
            filters: integer, filters of the bottleneck layer in a block.
            blocks: integer, blocks in the stacked blocks.
            stride1: default 2, stride of the first layer in the first block.
            name: string, stack label.

    Returns:
            Output tensor for the stacked blocks.
    """
    if blocks == 1:
        x = block3(x, filters,
            bottleneck=bottleneck,
            stride=stride1,
            expansion=expansion,
            activation=activation,
            bn_sync=bn_sync,
            conv_shortcut=True,
            name=name+'_block1')
        return x
    else:
        x = block3(x, filters,
            bottleneck=bottleneck,
            stride=stride1,
            expansion=expansion,
            activation=activation,
            bn_sync=bn_sync,
            conv_shortcut=True,
            name=name+'_block1')
        for i in range(1, blocks):
            x = block3(x, filters,
                bottleneck=bottleneck,
                expansion=expansion,
                activation=activation,
                bn_sync=bn_sync,
                name=name+'_block' + str(i+1))
        return x


def basic_stack1(x, filters, blocks, **kwargs):
    return stack_v1(x, filters, blocks, bottleneck=False, **kwargs)


def bottleneck_stack1(x, filters, blocks, **kwargs):
    return stack_v1(x, filters, blocks, bottleneck=True, **kwargs)


def basic_stack2(x, filters, blocks, **kwargs):
    return stack_v2(x, filters, blocks, bottleneck=False, **kwargs)


def bottleneck_stack2(x, filters, blocks, **kwargs):
    return stack_v2(x, filters, blocks, bottleneck=True, **kwargs)


def basic_stack3(x, filters, blocks, **kwargs):
    return stack_v3(x, filters, blocks, bottleneck=False, **kwargs)


def bottleneck_stack3(x, filters, blocks, **kwargs):
    return stack_v3(x, filters, blocks, bottleneck=True, **kwargs)
