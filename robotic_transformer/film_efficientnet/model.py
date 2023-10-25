# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------
# Robotic-transformer-torch
# build the efficientnet from the modification.
# Copyright (c) 2023 andyoung. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from
# """model.py - Model and module class for EfficientNet.
#    They are built to mirror those in the official TensorFlow implementation.
# ------------------------------------------------------------------------
"""
model.py - Model and module class for EfficientNet.
They are built to mirror those in the official TensorFlow implementation.
"""

import torch
from torch import nn
from torch.nn import functional as F

# Author: lukemelas (github username)
# Github repo: https://github.com/lukemelas/EfficientNet-PyTorch
# With adjustments and added comments by workingcoder (github username).
from robotic_transformer.film_efficientnet.utils import (
    MemoryEfficientSwish, Swish, calculate_output_image_size, drop_connect,
    efficientnet_params, get_model_params, get_same_padding_conv2d,
    load_pretrained_weights, round_filters, round_repeats)

VALID_MODELS = (
    'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
    'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
    'efficientnet-b8',

    # Support the construction of 'efficientnet-l2' without pretrained weights
    'efficientnet-l2',
)


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block (MBConvBlock) is a fundamental
    building block used in mobile and efficient convolutional neural networks.
    It is responsible for efficiently processing features in a convolutional
    neural network.

    Args:
        block_args (namedtuple): BlockArgs, defined in utils.py.
        global_params (namedtuple): GlobalParam, defined in utils.py.
        image_size (tuple or list): [image_height, image_width].

    Example usage:
    ```python
    from efficientnet_utils import BlockArgs, GlobalParams
    block_args = BlockArgs(
        kernel_size=3, num_repeat=1, input_filters=32, output_filters=16,
        expand_ratio=1, id_skip=True, se_ratio=0.25, stride=1,
    )
    global_params = GlobalParams(
        batch_norm_momentum=0.99, batch_norm_epsilon=0.001, dropout_rate=0.2,
        num_classes=1000, width_coefficient=1.0, depth_coefficient=1.0,
        depth_divisor=8, min_depth=None,
    )

    mbconv_block = MBConvBlock(block_args, global_params)
    input_data = torch.randn(32, 32, 128, 128)  # Batch of input data
    output_data = mbconv_block(input_data)
    ```

    The `MBConvBlock` class implements a block used in MobileNet-like architectures.
    It takes input data and applies a series of depthwise and pointwise convolutions.

    References:
    [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
    [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
    [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
    """

    def __init__(self, block_args, global_params, image_size=None):
        """Initialize a new MBConvBlock instance.

        Args:
            block_args (namedtuple): BlockArgs, defined in utils.py.
            global_params (namedtuple): GlobalParam, defined in utils.py.
            image_size (tuple or list): [image_height, image_width].
        """
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (block_args.se_ratio is not None) and (
            0 < block_args.se_ratio <= 1
        )
        self.id_skip = block_args.id_skip
        inp = block_args.input_filters
        oup = block_args.input_filters * block_args.expand_ratio
        if block_args.expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(
                in_channels=inp, out_channels=oup, kernel_size=1, bias=False,
            )
            self._bn0 = nn.BatchNorm2d(
                num_features=oup, momentum=self._bn_mom, eps=self._bn_eps,
            )
        k = block_args.kernel_size
        s = block_args.stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,
            kernel_size=k, stride=s, bias=False,
        )
        self._bn1 = nn.BatchNorm2d(
            num_features=oup, momentum=self._bn_mom, eps=self._bn_eps,
        )
        image_size = calculate_output_image_size(image_size, s)
        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1, 1))
            num_squeezed_channels = max(
                1, int(block_args.input_filters * block_args.se_ratio),
            )
            self._se_reduce = Conv2d(
                in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1,
            )
            self._se_expand = Conv2d(
                in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1,
            )
        final_oup = block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(
            in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False,
        )
        self._bn2 = nn.BatchNorm2d(
            num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps,
        )
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """MBConvBlock's forward function.

        Args:
            inputs (tensor): Input tensor.
            drop_connect_rate (bool): Drop connect rate (float, between 0 and 1).

        Returns:
            Output of this block after processing.
        """
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)
        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x
        x = self._project_conv(x)
        x = self._bn2(x)
        input_filters, output_filters = self._block_args.input_filters, \
            self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(
                    x, p=drop_connect_rate,
                    training=self.training,
                )
            x = x + inputs
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard
        (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class EfficientNet(nn.Module):
    """EfficientNet model. Most easily loaded with the .from_name or
    .from_pretrained methods.

    Args:
        blocks_args (list[namedtuple]): A list of BlockArgs to construct blocks.
        global_params (namedtuple): A set of GlobalParams shared between blocks.

    References:
    [1] https://arxiv.org/abs/1905.11946 (EfficientNet)

    Example:
    >>> import torch
    >>> from efficientnet.model import EfficientNet
    >>> inputs = torch.rand(1, 3, 224, 224)
    >>> model = EfficientNet.from_pretrained('efficientnet-b0')
    >>> model.eval()
    >>> outputs = model(inputs)
    """

    def __init__(self, blocks_args=None, global_params=None):
        """Initialize a new EfficientNet instance.

        Args:
            blocks_args (list[namedtuple]): A list of BlockArgs to construct blocks.
            global_params (namedtuple): A set of GlobalParams shared between blocks.
        """

        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args
        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon
        # Get stem static or dynamic convolution depending on image size
        image_size = global_params.image_size
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        # Stem
        in_channels = 3  # rgb
        # number of output channels
        out_channels = round_filters(32, self._global_params)
        self._conv_stem = Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, bias=False,
        )
        self._bn0 = nn.BatchNorm2d(
            num_features=out_channels, momentum=bn_mom, eps=bn_eps,
        )
        image_size = calculate_output_image_size(image_size, 2)
        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:
            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(
                    block_args.input_filters, self._global_params,
                ),
                output_filters=round_filters(
                    block_args.output_filters, self._global_params,
                ),
                num_repeat=round_repeats(
                    block_args.num_repeat, self._global_params,
                ),
            )
            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(
                MBConvBlock(
                    block_args, self._global_params, image_size=image_size,
                ),
            )
            image_size = calculate_output_image_size(
                image_size, block_args.stride,
            )
            if block_args.num_repeat > 1:  # modify block_args to keep same output size
                block_args = block_args._replace(
                    input_filters=block_args.output_filters, stride=1,
                )
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(
                    MBConvBlock(
                        block_args, self._global_params, image_size=image_size,
                    ),
                )
                # image_size = calculate_output_image_size(image_size, block_args.stride)  # stride = 1  # NOQA
        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._conv_head = Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False,
        )
        self._bn1 = nn.BatchNorm2d(
            num_features=out_channels, momentum=bn_mom, eps=bn_eps,
        )
        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        if self._global_params.include_top:
            self._dropout = nn.Dropout(self._global_params.dropout_rate)
            self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        # set activation to memory efficient swish by default
        self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard
        (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """

        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_endpoints(self, inputs):
        """Use convolution layer to extract features from reduction levels i in
        [1, 2, 3, 4, 5].

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Dictionary of last intermediate features with reduction levels i in [1, 2, 3, 4, 5].
            Example:
                >>> import torch
                >>> from efficientnet.model import EfficientNet
                >>> inputs = torch.rand(1, 3, 224, 224)
                >>> model = EfficientNet.from_pretrained('efficientnet-b0')
                >>> endpoints = model.extract_endpoints(inputs)
                >>> print(endpoints['reduction_1'].shape)  # torch.Size([1, 16, 112, 112])
                >>> print(endpoints['reduction_2'].shape)  # torch.Size([1, 24, 56, 56])
                >>> print(endpoints['reduction_3'].shape)  # torch.Size([1, 40, 28, 28])
                >>> print(endpoints['reduction_4'].shape)  # torch.Size([1, 112, 14, 14])
                >>> print(endpoints['reduction_5'].shape)  # torch.Size([1, 320, 7, 7])
                >>> print(endpoints['reduction_6'].shape)  # torch.Size([1, 1280, 7, 7])
        """
        endpoints = {}
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        prev_x = x
        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                # scale drop connect_rate
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints[f'reduction_{len(endpoints) + 1}'] = prev_x
            elif idx == len(self._blocks) - 1:
                endpoints[f'reduction_{len(endpoints) + 1}'] = x
            prev_x = x
        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        endpoints[f'reduction_{len(endpoints) + 1}'] = x

        return endpoints

    def extract_features(self, inputs):
        """Use convolution layer to extract feature.

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of the final convolution layer in the efficientnet model.
        """
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        print('After stem:', x.shape)
        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                # scale drop connect_rate
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            # x = block(x)
            print(f'the {idx + 1} block of MBConv shape is {x.shape}!')
        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        print('After head:', x.shape)
        return x

    def forward(self, inputs):
        """EfficientNet's forward function. Calls extract_features to extract
        features, applies the final linear layer, and returns logits.

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of this model after processing.
        """
        # Convolution layers
        x = self.extract_features(inputs)
        # Pooling and final linear layer
        x = self._avg_pooling(x)
        if self._global_params.include_top:
            x = x.flatten(start_dim=1)
            x = self._dropout(x)
            x = self._fc(x)
        return x

    @classmethod
    def from_name(cls, model_name, in_channels=3, **override_params):
        """Create an EfficientNet model according to its name.

        Args:
            model_name (str): Name for EfficientNet.
            in_channels (int): Input data's channel number.
            override_params (other keyword parameters):
                Params to override the model's global parameters.
                Optional keys:
                    'width_coefficient', 'depth_coefficient', 'image_size', 'dropout_rate',
                    'num_classes', 'batch_norm_momentum', 'batch_norm_epsilon',
                    'drop_connect_rate', 'depth_divisor', 'min_depth'

        Returns:
            An EfficientNet model.
        """
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(
            model_name, override_params,
        )
        model = cls(blocks_args, global_params)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def from_pretrained(
        cls, model_name, weights_path=None, advprop=False,
        in_channels=3, num_classes=1000, **override_params,
    ):
        """Create an EfficientNet model according to its name and load
        pretrained weights.

        Args:
            model_name (str): Name for EfficientNet.
            weights_path (None or str):
                str: path to pretrained weights file on the local disk.
                None: use pretrained weights downloaded from the Internet.
            advprop (bool):
                Whether to load pretrained weights trained with AdvProp (valid when weights_path
                        is None).
            in_channels (int): Input data's channel number.
            num_classes (int):
                Number of categories for classification.
                It controls the output size for the final linear layer.
            override_params (other keyword parameters):
                Params to override the model's global parameters.
                Optional keys:
                    'width_coefficient', 'depth_coefficient', 'image_size', 'dropout_rate',
                    'batch_norm_momentum', 'batch_norm_epsilon', 'drop_connect_rate',
                    'depth_divisor', 'min_depth'

        Returns:
            A pretrained EfficientNet model.
        """
        model = cls.from_name(
            model_name, num_classes=num_classes, **override_params,
        )
        load_pretrained_weights(
            model, model_name, weights_path=weights_path,
            load_fc=(num_classes == 1000), advprop=advprop,
        )
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        """Get the input image size for a given EfficientNet model.

        Args:
            model_name (str): Name for EfficientNet.

        Returns:
            Input image size (resolution).
        """
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """Validates the model name.

        Args:
            model_name (str): Name for EfficientNet.
        """
        if model_name not in VALID_MODELS:
            raise ValueError(
                'model_name should be one of: ' + # NOQA
                ', '.join(VALID_MODELS),
            )

    def _change_in_channels(self, in_channels):
        """Adjust the model's first convolution layer to in_channels, if
        in_channels is not equal to 3.

        Args:
            in_channels (int): Input data's channel number.
        """
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(
                image_size=self._global_params.image_size,
            )
            out_channels = round_filters(32, self._global_params)
            self._conv_stem = Conv2d(
                in_channels, out_channels, kernel_size=3, stride=2, bias=False,
            )
