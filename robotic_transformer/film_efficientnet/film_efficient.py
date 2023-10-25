# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.init as init

from robotic_transformer.film_efficientnet.model import (EfficientNet,
                                                         MBConvBlock)
from robotic_transformer.film_efficientnet.utils import get_model_params
from robotic_transformer.models.tokenlearner import TokenLearner
from robotic_transformer.models.USE import USEncoder


class FiLM(nn.Module):
    """FiLM (Feature-wise Linear Modulation) is a PyTorch module that applies
    feature-wise linear modulation to input data based on context information.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.

    Attributes:
        gamma (nn.Linear): Linear layer for modulating the scaling of features.
        beta (nn.Linear): Linear layer for modulating the bias of features.

    Example usage:
    ```python
    film_layer = FiLM(in_channels=256, out_channels=512)
    input_data = torch.randn(32, 256, 16, 16)  # Batch of input data
    context = torch.randn(32, 64)  # Context information
    output_data = film_layer(input_data, context)
    ```

    The `FiLM` class applies feature-wise linear modulation to input data, allowing
    the modulation of feature scaling and bias based on context information.
    """

    def __init__(self, in_channels, out_channels):
        """Initialize a new FiLM instance.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
        """
        super().__init__()
        self.gamma = nn.Linear(in_channels, out_channels)
        self.beta = nn.Linear(in_channels, out_channels)

        # Initialize weights and biases to zero
        init.zeros_(self.gamma.weight)
        init.zeros_(self.gamma.bias)
        init.zeros_(self.beta.weight)
        init.zeros_(self.beta.bias)

    def forward(self, x, context):
        """Apply feature-wise linear modulation to the input data based on
        context.

        Args:
            x (torch.Tensor): Input data with shape (B, in_channels, H, W).
            context (torch.Tensor): Context information with shape (B, in_channels).

        Returns:
            torch.Tensor: Modulated data with the same shape as the input.

        Example usage:
        ```python
        film_layer = FiLM(in_channels=256, out_channels=512)
        input_data = torch.randn(32, 256, 16, 16)  # Batch of input data
        context = torch.randn(32, 64)  # Context information
        output_data = film_layer(input_data, context)
        ```
        """
        gamma = self.gamma(context).unsqueeze(2).unsqueeze(3)
        beta = self.beta(context).unsqueeze(2).unsqueeze(3)
        return x * (1 + gamma) + beta


class FiLMEfficientNet(nn.Module):
    """FiLMEfficientNet is a PyTorch module that combines the EfficientNet
    backbone with FiLM (Feature-wise Linear Modulation) for image
    classification using contextual information.

    Args:
        USEncoder (nn.Module): Universal Sentence Encoder for context embedding.
        backbone (EfficientNet): EfficientNet backbone with pre-trained weights.
        tokenlearner (TokenLearner): Token Learner for processing image features.
        num_classes (int): Number of output classes for classification.

    Example usage:
    ```python
    from efficientnet_pytorch import EfficientNet

    # Load pre-trained EfficientNet backbone
    backbone = EfficientNet.from_pretrained('efficientnet-b3')

    film_efficient_net = FiLMEfficientNet(
        USEncoder=use_encoder,
        backbone=backbone,
        tokenlearner=token_learner,
        num_classes=1000,
    )
    input_image = torch.randn(32, 3, 224, 224)  # Batch of input images
    context_sentences = ["Context sentence 1", "Context sentence 2"]
    output_data = film_efficient_net(input_image, context_sentences)
    ```

    The `FiLMEfficientNet` class combines an EfficientNet backbone with FiLM to enable
    context-based image classification.
    """

    def __init__(
        self,
        USEncoder,
        backbone,
        tokenlearner,
        num_classes=1000,
    ):
        """Initialize a new FiLMEfficientNet instance.

        Args:
            USEncoder (nn.Module): Universal Sentence Encoder for context embedding.
            backbone (EfficientNet): EfficientNet backbone with pre-trained weights.
            tokenlearner (TokenLearner): Token Learner for processing image features.
            num_classes (int): Number of output classes for classification.
        """
        super().__init()

        # Define the USEncoder
        self.USEncoder = USEncoder

        # Load EfficientNet backbone with pre-trained weights,
        # copy the weight to the backbone model
        self.backbone = backbone
        self.backbone._blocks_args, self.backbone._global_params = get_model_params(
            'efficientnet-b3', None,
        )
        self.backbone = EfficientNet(
            blocks_args=self.backbone._blocks_args, global_params=self.backbone._global_params,
        )
        self.backbone.load_state_dict(pretrained_backbone.state_dict())

        self.backbone_with_film = []
        # Replace or append MBConvBlock with FiLMBlock and FiLM modules
        for block in self.backbone._blocks:
            self.backbone_with_film.append(block)
            self.backbone_with_film.append(
                FiLM(self.USEncoder._hidden_size, block._bn2.num_features),
            )
        self.backbone_with_film = nn.ModuleList(self.backbone_with_film)

        self.Linear_1b1_conv = nn.Conv2d(1536, 512, 1)
        self.tokenlearner = tokenlearner

        # Replace the last linear layer with a new one
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x, context_sentences):
        """Forward pass of the FiLMEfficientNet model.

        Args:
            x (torch.Tensor): Input image data with shape (B, C, H, W).
            context_sentences (list): List of context sentences.

        Returns:
            torch.Tensor: Model output for image classification.

        Example usage:
        ```python
        film_efficient_net = FiLMEfficientNet(...)
        input_image = torch.randn(32, 3, 224, 224)  # Batch of input images
        context_sentences = ["Context sentence 1", "Context sentence 2"]
        output_data = film_efficient_net(input_image, context_sentences)
        ```
        """
        # Embed context sentences using USEncoder
        context = self.USEncoder(context_sentences)

        # Process the input image with the EfficientNet backbone
        inputs = x
        x = self.backbone._swish(self.backbone._bn0(
            self.backbone._conv_stem(inputs)))
        for block in self.backbone_with_film:
            if isinstance(block, MBConvBlock):
                x = block(x)
            elif isinstance(block, FiLM):
                x = block(x, context)
        x = self.backbone._swish(
            self.backbone._bn1(self.backbone._conv_head(x)))

        # Add a convolution module for channel conversion
        x = self.Linear_1b1_conv(x)

        # Permute for token learner
        x = x.permute(0, 2, 3, 1)

        # Process image features with the token learner
        x = self.tokenlearner(x)

        return x
# This class is not used yet


class FiLMBlock(nn.Module):
    """FiLMBlock is a PyTorch module that extends an existing base block with
    FiLM (Feature-wise Linear Modulation) for image feature processing.

    Args:
        base_block (nn.Module): An existing base block, such as a MobileNetV2 block.

    Example usage:
    ```python
    base_block = SomeExistingBlock()
    film_block = FiLMBlock(base_block)
    input_data = torch.randn(32, 256, 64, 64)  # Batch of input data
    context = torch.randn(32, 384)  # Context information
    output_data = film_block(input_data, context)
    ```

    The `FiLMBlock` class enhances an existing base block by adding FiLM modulation
    to the feature processing pipeline.
    """

    def __init__(self, base_block):
        """Initialize a new FiLMBlock instance.

        Args:
            base_block (nn.Module): An existing base block, such as a MobileNetV2 block.
        """
        super().__init__()
        self.base_block = base_block

        # Initialize FiLM with the correct input and output channels
        # For example, when context is encoded as a 384-dimensional vector
        self.film = FiLM(384, base_block._bn2.num_features)

    def forward(self, x, context):
        """Forward pass of the FiLMBlock model.

        Args:
            x (torch.Tensor): Input data with shape (B, in_channels, H, W).
            context (torch.Tensor): Context information with shape (B, 384).

        Returns:
            torch.Tensor: Modulated feature data with the same shape as the input.

        Example usage:
        ```python
        base_block = SomeExistingBlock()
        film_block = FiLMBlock(base_block)
        input_data = torch.randn(32, 256, 64, 64)  # Batch of input data
        context = torch.randn(32, 384)  # Context information
        output_data = film_block(input_data, context)
        ```
        """
        # Process the input with the base block
        x = self.base_block(x)

        # Apply FiLM modulation based on context information
        x = self.film(x, context)

        return x


if __name__ == '__main__':

    USEncoder_model = USEncoder()
    pretrained_backbone = EfficientNet.from_pretrained('efficientnet-b3')
    tklr = TokenLearner
    model = FiLMEfficientNet(USEncoder_model, pretrained_backbone, tklr)
    # blocks_args, global_params = get_model_params('efficientnet-b3', None)
    # print(blocks_args)
    # print('test')
    # print(global_params)
    # pdb.set_trace()
    # model = EfficientNet(blocks_args=blocks_args, global_params=global_params)
    # with open('output.txt', 'w') as f:
    #     # Redirect standard output to a file object
    #     # print('Hello, world!', file=f)
    #     # print('This is a test.', file=f)
    #     for name, module in model.named_modules():
    #         print(name, module, file=f)
    input_tensor = torch.randn(6, 3, 300, 300)
    sentences = ["Pick apple from top drawer and place on counter."]
    # USE_model = USEncoder()
    # embeddings = USE_model(sentences)
    # context_embed = torch.randn(2,3)
    # output_tensor = model(input_tensor,context_embed)
    output_tensor = model(input_tensor, sentences)
    # # for idx, block in enumerate(model._blocks):
    # #     print(f"Block {idx+1} output shape: {block._project_conv.weight.shape}")
    # print((output_tensor.shape))
    # print(model.extract_features(input_tensor).shape)
    # # print(model.extract_endpoints(input_tensor))
    # print(type(model.extract_features(input_tensor)))
    # # print(model.extract_endpoints(input_tensor))
    # # print(model.extract_features(input_tensor))
    print(output_tensor.shape)
