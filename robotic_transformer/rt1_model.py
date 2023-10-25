# -*- coding: utf-8 -*-
"""RT-1 model architecture class."""
import torch
import torch.nn as nn

from robotic_transformer.film_efficientnet import (EfficientNet, FiLM,
                                                   MBConvBlock,
                                                   get_model_params)
from robotic_transformer.models import (TokenLearner, Transformers_Decoder,
                                        USEncoder)


class RT1(nn.Module):
    """RT1 is a custom neural network architecture that combines an
    EfficientNet backbone with a Transformers Decoder for a specific task. It
    takes an image as input and generates a sequence of instructions as output.

    Args:
        backbone_name (str): Name of the EfficientNet backbone architecture
                (default: 'efficientnet-b3').
        d_model (int): Dimension of the model (default: 512).
        depth (int): Number of layers in the Transformers Decoder (default: 8).
        dim (int): Dimension of the model for each token (default: 512).
        dim_head (int): Dimension of the model for each attention head (default: 64).
        dropout (float): Dropout probability (default: 0.0).
        heads (int): Number of attention heads (default: 8).
        max_seq_len (int): Maximum sequence length (default: 48).
        mlp_dim (int): Dimension of the feedforward network within the decoder (default: 512).
        num_actions (int): Number of possible actions (default: 11).
        num_classes (int): Number of output classes (default: 1000).
        token_learner_layers (int): Number of layers in the Token Learner (default: 8).
        vocab_size (int): Size of the vocabulary for the decoder (default: 256).

    Attributes:
        USEncoder: An instance of the USEncoder used for processing context sentences.
        pretrained_backbone: The pretrained EfficientNet backbone.
        backbone: The EfficientNet backbone used for feature extraction.
        backbone_with_film: A modified backbone with FiLM (Feature-wise Linear Modulation) blocks.
        Linear_1b1_conv: A 1x1 convolution layer for channel conversion.
        tokenlearner: The Token Learner module.
        transformers_decoder: The Transformers Decoder module.
        classifier: The final linear layer for classification.

    Methods:
        to(device): Move the model and its components to the specified device.
        forward(x, context_sentences): Forward pass of the model, taking an image tensor 'x' and
                context sentences.

    Note:
        RT1 combines an EfficientNet backbone with a Transformers Decoder to perform a specific
        task, which may require fine-tuning and additional configuration based on the problem
        domain.
    """

    def __init__(
        self,
        backbone_name: str = 'efficientnet-b3',
        d_model: int = 512,
        depth: int = 8,
        dim: int = 512,
        dim_head: int = 64,
        dropout: float = 0.,
        heads: int = 8,
        max_seq_len: int = 48,
        mlp_dim: int = 512,
        num_actions: int = 11,
        num_classes: int = 1000,
        token_learner_layers: int = 8,
        vocab_size: int = 256,
    ):
        super().__init__()
        # define the USEncoder
        self.USEncoder = USEncoder()
        # Load EfficientNet backbone with pre-trained weights,
        # copy the weight to the backbone model
        self.pretrained_backbone = EfficientNet.from_pretrained(
            backbone_name,
        )
        self.backbone = self.pretrained_backbone
        self.backbone._blocks_args, self.backbone._global_params = get_model_params(
            backbone_name, None,
        )
        self.backbone = EfficientNet(
            blocks_args=self.backbone._blocks_args, global_params=self.backbone._global_params,
        )
        self.backbone.load_state_dict(self.pretrained_backbone.state_dict())
        # self._swish = MemoryEfficientSwish()
        # self.film = FiLM(self.USEncoder._hidden_size, out_channels)
        self.backbone_with_film = []
        # Replace or append MBConvBlock with FiLMBlock, 添加到一个新的modulelist中去
        for block in self.backbone._blocks:
            # self.backbone._blocks[idx] = FiLMBlock(block)
            self.backbone_with_film.append(block)
            self.backbone_with_film.append(
                FiLM(self.USEncoder._hidden_size, block._bn2.num_features),
            )
        self.backbone_with_film = nn.ModuleList(self.backbone_with_film)
        self.Linear_1b1_conv = nn.Conv2d(1536, 512, 1)
        self.tokenlearner = TokenLearner(S=token_learner_layers)
        self.transformers_decoder = Transformers_Decoder(
            d_model=d_model,
            depth=depth,
            dim=dim,
            dim_head=dim_head,
            dropout=dropout,
            heads=heads,
            max_seq_len=max_seq_len,
            mlp_dim=mlp_dim,
            num_actions=num_actions,
            vocab_size=vocab_size,
        )
        # Replace the last linear layer with a new one
        self.classifier = nn.Linear(1280, num_classes)
        """Initializes the RT1 model with the specified configuration.

        Args:
            backbone_name (str, optional): Name of the EfficientNet backbone architecture
                    (default: 'efficientnet-b3').
            d_model (int, optional): Dimension of the model (default: 512).
            depth (int, optional): Number of layers in the Transformers Decoder (default: 8).
            dim (int, optional): Dimension of the model for each token (default: 512).
            dim_head (int, optional): Dimension of the model for each attention head (default: 64).
            dropout (float, optional): Dropout probability (default: 0.0).
            heads (int, optional): Number of attention heads (default: 8).
            max_seq_len (int, optional): Maximum sequence length (default: 48).
            mlp_dim (int, optional): Dimension of the feedforward network within the decoder
                    (default: 512).
            num_actions (int, optional): Number of possible actions (default: 11).
            num_classes (int, optional): Number of output classes (default: 1000).
            token_learner_layers (int, optional): Number of layers in the Token Learner
                    (default: 8).
            vocab_size (int, optional): Size of the vocabulary for the decoder (default: 256).

        Note:
            The RT1 model is a custom neural network architecture designed for a specific task.
            It combines an EfficientNet backbone with a Transformers Decoder for
            image-to-instruction generation. The provided constructor arguments allow for
            customization of various aspects of the model's architecture. During initialization,
            it also sets up various components including the USEncoder, EfficientNet backbone,
            FiLM blocks, Token Learner, Transformers Decoder, and the final linear classifier.
        """

    def to(self, device):
        """Move the RT1 model and its components to the specified device.

        Args:
            device (torch.device): The target device (e.g., 'cpu' or 'cuda').

        Returns:
            RT1: The RT1 model with all components moved to the specified device.
        """
        self.USEncoder.to(device)
        self.backbone.to(device)
        self.backbone_with_film.to(device)
        self.Linear_1b1_conv.to(device)
        self.tokenlearner.to(device)
        self.transformers_decoder.to(device)
        return self

    def forward(self, x, context_sentences):
        """Forward pass of the RT1 model.

        Args:
            x (torch.Tensor): The input image tensor.
            context_sentences (list): A list of context sentences for instruction generation.

        Returns:
            torch.Tensor: The model's output tensor, representing generated instructions.

        Notes:
            The forward method processes an input image and context sentences to generate
            instructions. The specific processing steps are defined in the method's implementation.
        """
        # Change the context to sentences, and then put the relevant processing
        # into the forward function!
        context = self.USEncoder(context_sentences)
        # Stem
        # x = inputs
        inputs = x
        x = self.backbone._swish(
            self.backbone._bn0(
                self.backbone._conv_stem(inputs),
            ),
        )
        for block in self.backbone_with_film:
            if isinstance(block, MBConvBlock):
                # if 'MBConv' in block.name:
                x = block(x)
            elif isinstance(block, FiLM):
                # elif 'FiLM' in block.name:
                x = block(x, context)
        x = self.backbone._swish(
            self.backbone._bn1(self.backbone._conv_head(x)),
        )
        # Add a convolution module for channel conversion, converting the last 1536 channels
        # of efficientnet to 512 channels
        x = self.Linear_1b1_conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.tokenlearner(x)
        x = self.transformers_decoder(x)
        # The tail processing of the original model is commented out, mainly to extract the
        # features of the fusion instruction.
        # x = self.backbone.extract_features(x)  # Get features from the backbone
        # x = self.backbone._avg_pooling(x)  # Global average pooling
        # x = x.flatten(start_dim=1)  # Flatten
        # x = self.classifier(x)  # Classification layer
        return x


if __name__ == '__main__':
    input_tensor = torch.randn(6, 3, 224, 224)
    sentences = ["Pick apple from top drawer and place on counter."]
    model = RT1(num_actions=15, backbone_name='efficientnet-b0')
    output_tensor = model(input_tensor, sentences)
    print(output_tensor.shape)
