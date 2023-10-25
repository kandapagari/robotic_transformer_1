# -*- coding: utf-8 -*-
"""
utils.py - Helper functions for building the model and for loading model parameters.
These helper functions are built to mirror those in the official TensorFlow implementation.
"""

# Author: lukemelas (github username)
# Github repo: https://github.com/lukemelas/EfficientNet-PyTorch
# With adjustments and added comments by workingcoder (github username).

import collections
import math
import re
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo

################################################################################
# Help functions for model architecture
################################################################################

# GlobalParams and BlockArgs: Two namedtuples
# Swish and MemoryEfficientSwish: Two implementations of the method
# round_filters and round_repeats:
#     Functions to calculate params for scaling model width and depth ! ! !
# get_width_and_height_from_size and calculate_output_image_size
# drop_connect: A structural design
# get_same_padding_conv2d:
#     Conv2dDynamicSamePadding
#     Conv2dStaticSamePadding
# get_same_padding_maxPool2d:
#     MaxPool2dDynamicSamePadding
#     MaxPool2dStaticSamePadding
#     It's an additional function, not used in EfficientNet,
#     but can be used in other model (such as EfficientDet).

# Parameters for the entire model (stem, all blocks, and head)
GlobalParams = collections.namedtuple(
    'GlobalParams', [
        'width_coefficient', 'depth_coefficient', 'image_size', 'dropout_rate',
        'num_classes', 'batch_norm_momentum', 'batch_norm_epsilon',
        'drop_connect_rate', 'depth_divisor', 'min_depth', 'include_top',
    ],
)

# Parameters for an individual model block
BlockArgs = collections.namedtuple(
    'BlockArgs', [
        'num_repeat', 'kernel_size', 'stride', 'expand_ratio',
        'input_filters', 'output_filters', 'se_ratio', 'id_skip',
    ],
)

# Set GlobalParams and BlockArgs's defaults
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

# Swish activation function


def get_swish():
    """Get the Swish activation function.

    This function checks if the `SiLU` activation function is available in PyTorch's `nn` module.
    If it is available, it returns `nn.SiLU`, otherwise, it defines a custom
    Swish activation class.

    Returns:
        Swish (class): Swish activation function class.
    """
    if hasattr(nn, 'SiLU'):
        Swish = nn.SiLU
    else:
        # For compatibility with old PyTorch versions
        class Swish(nn.Module):
            def forward(self, x):
                return x * torch.sigmoid(x)
    return Swish


Swish = get_swish()

# A memory-efficient implementation of Swish function


class SwishImplementation(torch.autograd.Function):
    """Custom autograd function for the Swish activation function.

    Swish is defined as Swish(x) = x * sigmoid(x).

    This class defines both the forward and backward passes for Swish.

    Args:
        ctx: Context object to store intermediate results.
        i: Input tensor.

    Returns:
        result: The Swish activation result.
    """
    @staticmethod
    def forward(ctx, i):
        """Forward pass for the Swish activation.

        Args:
            ctx (Context): Context object.
            i (Tensor): Input tensor.

        Returns:
            result (Tensor): The Swish activation result.
        """
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass for the Swish activation.

        Args:
            ctx (Context): Context object.
            grad_output (Tensor): Gradient of the loss with respect to the output.

        Returns:
            grad_input (Tensor): Gradient of the loss with respect to the input.
        """
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    """Memory-efficient implementation of the Swish activation function.

    Swish is a non-linear activation function that can be computationally expensive due to the
    use of the sigmoid function. This implementation optimizes the memory usage by
    utilizing a custom autograd function.

    Example:
        >>> swish = MemoryEfficientSwish()
        >>> output = swish(input)

    Attributes:
        None

    Methods:
        forward(x): Apply the Swish activation function to the input tensor.
    """

    def forward(self, x):
        """Apply the Swish activation function to the input tensor.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The tensor after applying the Swish activation.
        """
        return SwishImplementation.apply(x)


def round_filters(filters, global_params):
    """Calculate and round the number of filters based on the specified width
    multiplier. This function uses the width_coefficient, depth_divisor, and
    min_depth from global_params to perform the calculation and rounding.

    Args:
        filters (int): The initial number of filters to be calculated.
        global_params (namedtuple): Global parameters of the model that include width-related
        settings.

    Returns:
        new_filters (int): The new number of filters after calculating and rounding.
    """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters

    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth

    filters *= multiplier

    min_depth = min_depth or divisor  # Ensure min_depth is not None

    # Calculate the new number of filters based on the formula transferred from the official
    # TensorFlow implementation
    new_filters = max(
        min_depth, int(filters + divisor / 2) // divisor * divisor,
    )

    # Ensure that rounding does not reduce the number of filters by more than 10%
    if new_filters < 0.9 * filters:
        new_filters += divisor

    return int(new_filters)


def round_repeats(repeats, global_params):
    """Calculate the number of times a module within a block should be repeated
    based on the specified depth multiplier. This function uses the
    depth_coefficient from global_params to perform the calculation.

    Args:
        repeats (int): The initial number of times the module should be repeated (num_repeat).
        global_params (namedtuple): Global parameters of the model that include depth-related
        settings.

    Returns:
        new_repeat (int): The new number of times the module should be repeated after calculating.
    """
    multiplier = global_params.depth_coefficient

    # Calculate the new number of repeats based on the depth multiplier
    return int(math.ceil(multiplier * repeats)) if multiplier else repeats


def drop_connect(inputs, p, training):
    """Apply drop connection to the inputs.

    Args:
        inputs (tensor: BCWH): Input tensor with shape (Batch, Channel, Width, Height).
        p (float: 0.0~1.0): Probability of drop connection for each element.
        training (bool): Boolean flag to indicate whether the model is in training mode.

    Returns:
        output: Output tensor after applying drop connection.

    Notes:
        - If the model is not in training mode (i.e., training = False), the inputs
            remain unchanged.
        - During training, random elements of the input tensor are set to zero with
            a probability of 'p'.
    """
    assert 0 <= p <= 1, 'p must be in the range of [0, 1]'

    if not training:
        return inputs

    batch_size = inputs.shape[0]
    keep_prob = 1 - p

    # Generate a binary tensor mask according to the drop probability
    random_tensor = keep_prob
    random_tensor += torch.rand(
        [batch_size, 1, 1, 1],
        dtype=inputs.dtype,
        device=inputs.device,
    )
    binary_tensor = torch.floor(random_tensor)

    # Apply the drop connection: set elements to zero with probability 'p'
    return inputs / keep_prob * binary_tensor


def get_width_and_height_from_size(x):
    """Extract height and width from the input data size.

    Args:
        x (int, tuple or list): Data size.

    Returns:
        size: A tuple or list (H, W) representing the height and width.

    Raises:
        TypeError: If 'x' is not of type int, tuple, or list.
    """
    if isinstance(x, int):
        return x, x
    if isinstance(x, (list, tuple)):
        return x
    else:
        raise TypeError("Input 'x' should be an integer, a tuple, or a list.")


def calculate_output_image_size(input_image_size, stride):
    """Calculate the output image size when using Conv2dSamePadding with a
    specified stride.

    Args:
        input_image_size (int, tuple, or list): Size of the input image.
        stride (int or tuple): Conv2d operation's stride.

    Returns:
        output_image_size: A list [H, W] representing the height and width of the output image.

    Raises:
        TypeError: If 'input_image_size' is not an integer, tuple, or list.
    """
    if input_image_size is None:
        return None
    image_height, image_width = get_width_and_height_from_size(
        input_image_size)
    stride = stride if isinstance(stride, int) else stride[0]
    image_height = int(math.ceil(image_height / stride))
    image_width = int(math.ceil(image_width / stride))
    return [image_height, image_width]


# Note:
# The following 'SamePadding' functions make output size equal ceil(input size/stride).
# Only when stride equals 1, can the output size be the same as input size.
# Don't be confused by their function names ! ! !

def get_same_padding_conv2d(image_size=None):
    """Choose between static padding (with specified image size) or dynamic
    padding based on the requirements.

    Args:
        image_size (int, tuple, or None): Size of the image. If None, dynamic padding is used;
        otherwise, static padding is chosen.

    Returns:
        Conv2dDynamicSamePadding or Conv2dStaticSamePadding: The appropriate convolutional layer
        class.

    Note:
    - Dynamic padding is used when the image size is not specified.
    - Static padding is necessary for ONNX exporting of models when image size is specified.
    - When using dynamic padding, the output size may not be the same as the input size.
    """
    if image_size is None:
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dStaticSamePadding, image_size=image_size)


class Conv2dDynamicSamePadding(nn.Conv2d):
    """2D Convolution with dynamic padding for 'SAME' mode, accommodating
    variable input image sizes.

    The padding is performed within the forward function, allowing
    dynamic adjustments based on input size.
    """

    # Tips for 'SAME' mode padding.
    #     Given the following:
    #         i: width or height
    #         s: stride
    #         k: kernel size
    #         d: dilation
    #         p: padding
    #     Output after Conv2d:
    #         o = floor((i+p-((k-1)*d+1))/s+1)
    # If o equals i, i = floor((i+p-((k-1)*d+1))/s+1),
    # => p = (i-1)*s+((k-1)*d+1)-i

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, dilation=1,
        groups=1, bias=True,
    ):
        """Initialize a dynamic 2D convolutional layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int or tuple): Size of the convolutional kernel.
            stride (int or tuple): The stride for the convolution operation.
            dilation (int or tuple): The dilation rate for the convolution operation.
            groups (int): Number of groups for grouped convolution.
            bias (bool): Whether to include a bias term in the convolution.

        Note:
        - The padding is determined dynamically during the forward pass.
        - This layer helps achieve 'SAME' mode padding regardless of input size.
        """
        super().__init__(
            in_channels, out_channels,
            kernel_size, stride, 0, dilation, groups, bias,
        )
        self.stride = self.stride if len(self.stride) == 2 else [
            self.stride[0],
        ] * 2

    def forward(self, x):
        """Forward pass of the dynamic convolutional layer.

        Args:
            x (tensor): Input tensor to be convolved.

        Returns:
            tensor: The output of the convolution operation with dynamic padding.
        """
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        # change the output size according to stride ! ! !
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max(
            (oh - 1) * self.stride[0] + # NOQA
            (kh - 1) * self.dilation[0] + 1 - ih, 0,
        )
        pad_w = max(
            (ow - 1) * self.stride[1] + # NOQA
            (kw - 1) * self.dilation[1] + 1 - iw, 0,
        )
        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [
                    pad_w // 2, pad_w - pad_w // # NOQA
                    2, pad_h // 2, pad_h - pad_h // 2,
                ],
            )
        return F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation,
            self.groups,
        )


class Conv2dStaticSamePadding(nn.Conv2d):
    """2D Convolutions with 'SAME' mode padding, using a predefined input image
    size.

    The padding is calculated during construction based on the specified
    input image size, then applied in the forward pass.
    """

    # With the same calculation as Conv2dDynamicSamePadding

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1,
        image_size=None, **kwargs,
    ):
        """Initialize a 2D convolutional layer with static 'SAME' mode padding.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int or tuple): Size of the convolutional kernel.
            stride (int or tuple): The stride for the convolution operation.
            image_size (int or tuple): Size of the input image to determine padding.
            **kwargs: Additional keyword arguments for the convolutional layer.

        Note:
        - The padding is calculated during construction based on the input image size.
        - This layer ensures consistent 'SAME' mode padding behavior for the specified input
            image size.
        """
        super().__init__(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [
            self.stride[0],
        ] * 2
        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(
            image_size, int,
        ) else image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max(
            (oh - 1) * self.stride[0] + # NOQA
            (kh - 1) * self.dilation[0] + 1 - ih, 0,
        )
        pad_w = max(
            (ow - 1) * self.stride[1] + # NOQA
            (kw - 1) * self.dilation[1] + 1 - iw, 0,
        )
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((
                pad_w // 2, pad_w - pad_w // 2,
                pad_h // 2, pad_h - pad_h // 2,
            ))
        else:
            self.static_padding = nn.Identity()

    def forward(self, x):
        """Forward pass of the static convolutional layer with 'SAME' mode
        padding.

        Args:
            x (tensor): Input tensor to be convolved.

        Returns:
            tensor: The output of the convolution operation with static 'SAME' mode padding.
        """
        x = self.static_padding(x)
        x = F.conv2d(
            x, self.weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups,
        )
        return x


def get_same_padding_maxPool2d(image_size=None):
    """Choose between dynamic and static padding for 2D max-pooling operations
    based on input image size.

    Args:
        image_size (int or tuple): Size of the image.

    Returns:
        MaxPool2dDynamicSamePadding or MaxPool2dStaticSamePadding: The appropriate 2D max-pooling
        layer class.
    """
    if image_size is None:
        return MaxPool2dDynamicSamePadding
    else:
        return partial(MaxPool2dStaticSamePadding, image_size=image_size)


class MaxPool2dDynamicSamePadding(nn.MaxPool2d):
    """2D MaxPooling with dynamic padding behavior similar to TensorFlow's
    'SAME' mode, adjusting to dynamic image sizes.

    Args:
        kernel_size (int or tuple): The size of the max-pooling window.
        stride (int or tuple): The stride for the max-pooling operation.
        padding (int, optional): Additional padding applied to both sides of the input.
                                    Default is 0.
        dilation (int or tuple, optional): Dilation rate. Default is 1.
        return_indices (bool, optional): Whether to return the indices in the output.
                                            Default is False.
        ceil_mode (bool, optional): When True, uses the ceil function to compute output size.
                                        Default is False.
    """

    def __init__(
        self, kernel_size, stride, padding=0, dilation=1, return_indices=False,
        ceil_mode=False,
    ):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        self.stride = [self.stride] * \
            2 if isinstance(self.stride, int) else self.stride
        self.kernel_size = [self.kernel_size] * \
            2 if isinstance(self.kernel_size, int) else self.kernel_size
        self.dilation = [self.dilation] * \
            2 if isinstance(self.dilation, int) else self.dilation

    def forward(self, x):
        """Forward pass for 2D MaxPooling with dynamic padding behavior similar
        to TensorFlow's 'SAME' mode, adjusting to dynamic image sizes.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Max-pooled output tensor.

        Note:
        - This method dynamically adjusts the padding based on the input size, kernel size,
                stride, and dilation.
        - The max-pooling operation is performed with the specified kernel size, stride, padding,
                dilation, and ceil mode.
        - Padding is applied to ensure that the output size matches the desired behavior similar
                to 'SAME' mode in TensorFlow.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Max-pooled output tensor.
        """
        ih, iw = x.size()[-2:]
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max(
            (oh - 1) * self.stride[0] + # NOQA
            (kh - 1) * self.dilation[0] + 1 - ih, 0,
        )
        pad_w = max(
            (ow - 1) * self.stride[1] + # NOQA
            (kw - 1) * self.dilation[1] + 1 - iw, 0,
        )
        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [
                    pad_w // 2, pad_w - pad_w // # NOQA
                    2, pad_h // 2, pad_h - pad_h // 2,
                ],
            )
        return F.max_pool2d(
            x, self.kernel_size, self.stride, self.padding,
            self.dilation, self.ceil_mode, self.return_indices,
        )


class MaxPool2dStaticSamePadding(nn.MaxPool2d):
    """2D MaxPooling like TensorFlow's 'SAME' mode, with the given input image
    size.

    The padding module is calculated in the construction function, then used in the forward pass.

    Args:
        kernel_size (int or tuple): The size of the max-pooling window.
        stride (int or tuple): The stride for the max-pooling operation.
        image_size (int or tuple, optional): The size of the input image. Default is None.
        **kwargs: Additional keyword arguments to be passed to the parent class constructor.

    Attributes:
        static_padding (nn.Module): A padding module that ensures the output size matches the
        desired behavior similar to 'SAME' mode in TensorFlow.
    """

    def __init__(self, kernel_size, stride, image_size=None, **kwargs):
        """Initialize the MaxPool2dStaticSamePadding layer with the given
        parameters.

        Args:
            kernel_size (int or tuple): The size of the max-pooling window.
            stride (int or tuple): The stride for the max-pooling operation.
            image_size (int or tuple, optional): The size of the input image. Default is None.
            **kwargs: Additional keyword arguments to be passed to the parent class constructor.
        """
        super().__init__(kernel_size, stride, **kwargs)
        self.stride = [self.stride] * \
            2 if isinstance(self.stride, int) else self.stride
        self.kernel_size = [self.kernel_size] * \
            2 if isinstance(self.kernel_size, int) else self.kernel_size
        self.dilation = [self.dilation] * \
            2 if isinstance(self.dilation, int) else self.dilation
        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(
            image_size, int,
        ) else image_size
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max(
            (oh - 1) * self.stride[0] + # NOQA
            (kh - 1) * self.dilation[0] + 1 - ih, 0,
        )
        pad_w = max(
            (ow - 1) * self.stride[1] + # NOQA
            (kw - 1) * self.dilation[1] + 1 - iw, 0,
        )
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d(
                (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
            )
        else:
            self.static_padding = nn.Identity()

    def forward(self, x):
        """Forward pass for 2D MaxPooling with static padding behavior similar
        to TensorFlow's 'SAME' mode.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Max-pooled output tensor.

        Note:
        - This method uses pre-calculated static padding based on the given image size,
            kernel size, stride, and dilation.
        - The max-pooling operation is performed with the specified kernel size, stride, padding,
            dilation, and ceil mode.
        - Padding is applied to ensure that the output size matches the desired behavior similar
            to 'SAME' mode in TensorFlow.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Max-pooled output tensor.
        """
        x = self.static_padding(x)
        x = F.max_pool2d(
            x, self.kernel_size, self.stride, self.padding,
            self.dilation, self.ceil_mode, self.return_indices,
        )
        return x


################################################################################
# Helper functions for loading model params
################################################################################

# BlockDecoder: A Class for encoding and decoding BlockArgs
# efficientnet_params: A function to query compound coefficient
# get_model_params and efficientnet:
#     Functions to get BlockArgs and GlobalParams for efficientnet
# url_map and url_map_advprop: Dicts of url_map for pretrained weights
# load_pretrained_weights: A function to load pretrained weights

class BlockDecoder:
    """Block Decoder for readability, straight from the official TensorFlow
    repository.

    This class provides methods to decode and encode block
    configurations using string notations.
    """

    @staticmethod
    def _decode_block_string(block_string):
        """Get a block through a string notation of arguments.

        Args:
            block_string (str): A string notation of arguments.
                Examples: 'r1_k3_s11_e1_i32_o16_se0.25_noskip'.

        Returns:
            BlockArgs: The namedtuple defined at the top of this file.
        """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value
        # Check stride
        assert (('s' in options and len(options['s']) == 1) or # NOQA
                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))
        return BlockArgs(
            num_repeat=int(options['r']),
            kernel_size=int(options['k']),
            stride=[int(options['s'][0])],
            expand_ratio=int(options['e']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            se_ratio=float(options['se']) if 'se' in options else None,
            id_skip=('noskip' not in block_string),
        )

    @staticmethod
    def _encode_block_string(block):
        """Encode a block to a string.

        Args:
            block (namedtuple): A BlockArgs type argument.

        Returns:
            block_string: A String form of BlockArgs.
        """
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            f'e{block.expand_ratio}',
            'i%d' % block.input_filters,
            'o%d' % block.output_filters,
        ]
        if 0 < block.se_ratio <= 1:
            args.append(f'se{block.se_ratio}')
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def decode(string_list):
        """Decode a list of string notations to specify blocks inside the
        network.

        Args:
            string_list (list[str]): A list of strings, each string is a notation of block.

        Returns:
            blocks_args: A list of BlockArgs namedtuples of block args.
        """
        assert isinstance(string_list, list)
        return [
            BlockDecoder._decode_block_string(block_string)
            for block_string in string_list
        ]

    @staticmethod
    def encode(blocks_args):
        """Encode a list of BlockArgs to a list of strings.

        Args:
            blocks_args (list[namedtuples]): A list of BlockArgs namedtuples of block args.

        Returns:
            block_strings: A list of strings, each string is a notation of block.
        """
        return [BlockDecoder._encode_block_string(block) for block in blocks_args]


def efficientnet_params(model_name):
    """Map EfficientNet model name to parameter coefficients.

    Args:
        model_name (str): Model name to be queried.

    Returns:
        params_dict[model_name]: A (width, depth, res, dropout) tuple. These values
        correspond to the width multiplier, depth multiplier, input image resolution,
        and dropout rate for the specified EfficientNet model.
    """
    params_dict = {
        # Coefficients:   width, depth, res, dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        'efficientnet-b8': (2.2, 3.6, 672, 0.5),
        'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    }
    return params_dict[model_name]


def efficientnet(
    width_coefficient=None, depth_coefficient=None, image_size=None,
    dropout_rate=0.2, drop_connect_rate=0.2, num_classes=1000, include_top=True,
):
    """Create BlockArgs and GlobalParams for an EfficientNet model.

    Args:
        width_coefficient (float): Width multiplier for the model.
        depth_coefficient (float): Depth multiplier for the model.
        image_size (int): Input image size.
        dropout_rate (float): Dropout rate for the model.
        drop_connect_rate (float): Drop connect rate for the model.
        num_classes (int): Number of output classes.
        include_top (bool): Whether to include the fully connected top layer.

    Returns:
        blocks_args, global_params: A tuple containing the block arguments (blocks_args) and
        global parameters (global_params) for the EfficientNet model. These parameters are
        used to configure the model's architecture and behavior.
    """

    # Blocks args for the whole model (efficientnet-b0 by default)
    # It will be modified in the construction of the EfficientNet Class according to the model
    blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25',
        'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25',
        'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25',
        'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    blocks_args = BlockDecoder.decode(blocks_args)

    global_params = GlobalParams(
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        image_size=image_size,
        dropout_rate=dropout_rate,
        num_classes=num_classes,
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        drop_connect_rate=drop_connect_rate,
        depth_divisor=8,
        min_depth=None,
        include_top=include_top,
    )

    return blocks_args, global_params


def get_model_params(model_name, override_params):
    """Get the block arguments and global parameters for a given model name.

    Args:
        model_name (str): The name of the model.
        override_params (dict): A dictionary to modify global parameters.

    Returns:
        blocks_args, global_params: A tuple containing the block arguments (blocks_args) and
        global parameters (global_params) for the specified model. These parameters define
        the architecture and behavior of the model.

    Raises:
        NotImplementedError: If the provided model name is not pre-defined.
    """
    if not model_name.startswith('efficientnet'):
        raise NotImplementedError(
            f'model name is not pre-defined: {model_name}',
        )
    w, d, s, p = efficientnet_params(model_name)
    # note: all models have drop connect rate = 0.2
    blocks_args, global_params = efficientnet(
        width_coefficient=w, depth_coefficient=d, dropout_rate=p, image_size=s,
    )
    if override_params:
        # ValueError will be raised here if override_params has fields not included in
        # global_params.
        global_params = global_params._replace(**override_params)
    return blocks_args, global_params


# train with Standard methods
# check more details in paper(EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks)  # NOQA
url_map = {
    'efficientnet-b0': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth',  # NOQA
    'efficientnet-b1': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth',  # NOQA
    'efficientnet-b2': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth',  # NOQA
    'efficientnet-b3': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth',  # NOQA
    'efficientnet-b4': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth',  # NOQA
    'efficientnet-b5': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth',  # NOQA
    'efficientnet-b6': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth',  # NOQA
    'efficientnet-b7': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth',  # NOQA
}

# train with Adversarial Examples(AdvProp)
# check more details in paper(Adversarial Examples Improve Image Recognition)
url_map_advprop = {
    'efficientnet-b0': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b0-b64d5a18.pth',  # NOQA
    'efficientnet-b1': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b1-0f3ce85a.pth',  # NOQA
    'efficientnet-b2': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b2-6e9d97e5.pth',  # NOQA
    'efficientnet-b3': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b3-cdd7c0f4.pth',  # NOQA
    'efficientnet-b4': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b4-44fb3a87.pth',  # NOQA
    'efficientnet-b5': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b5-86493f6b.pth',  # NOQA
    'efficientnet-b6': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b6-ac80338e.pth',  # NOQA
    'efficientnet-b7': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b7-4652b6dd.pth',  # NOQA
    'efficientnet-b8': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b8-22a8fe65.pth',  # NOQA
}

# TODO: add the pretrained weights url map of 'efficientnet-l2'


def load_pretrained_weights(
    model, model_name, weights_path=None, load_fc=True, advprop=False,
    verbose=True,
):
    """Loads pretrained weights from a local path or downloads them from the
    Internet for an EfficientNet model.

    Args:
        model (Module): The entire EfficientNet model.
        model_name (str): The model name of the EfficientNet (e.g., 'efficientnet-b0').
        weights_path (str or None):
            - str: The local path to a pretrained weights file on the disk.
            - None: Download pretrained weights from the Internet.
        load_fc (bool): Whether to load pretrained weights for the fully connected (fc) layer at
                        the end of the model.
        advprop (bool): Whether to load pretrained weights trained with advprop (valid when
                        weights_path is None).
        verbose (bool): Whether to print a message indicating that pretrained weights have been
                        loaded.

    Raises:
        AssertionError: If there are missing or unexpected keys when loading pretrained weights.

    Notes:
        This function loads pretrained weights for an EfficientNet model from a local file or
            downloads them
        from the Internet. It is possible to load only the weights up to the last convolutional
            layer (excluding
        the fully connected layer) by setting load_fc to False. If using weights trained with
            "advprop," set
        advprop to True. The function raises an AssertionError if there are any missing or
            unexpected keys when loading the weights.
    """
    if isinstance(weights_path, str):
        state_dict = torch.load(weights_path)
    else:
        # AutoAugment or Advprop (different preprocessing)
        url_map_ = url_map_advprop if advprop else url_map
        state_dict = model_zoo.load_url(url_map_[model_name])
    if load_fc:
        ret = model.load_state_dict(state_dict, strict=False)
        assert (
            not ret.missing_keys
        ), f'Missing keys when loading pretrained weights: {ret.missing_keys}'
    else:
        state_dict.pop('_fc.weight')
        state_dict.pop('_fc.bias')
        ret = model.load_state_dict(state_dict, strict=False)
        assert set(ret.missing_keys) == {
            '_fc.weight',
            '_fc.bias',
        }, f'Missing keys when loading pretrained weights: {ret.missing_keys}'
    assert (
        not ret.unexpected_keys
    ), f'Missing keys when loading pretrained weights: {ret.unexpected_keys}'
    if verbose:
        print(f'Loaded pretrained weights for {model_name}')
