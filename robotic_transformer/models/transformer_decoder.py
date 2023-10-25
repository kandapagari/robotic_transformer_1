# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------
# Robotic-transformer-torch
# Copyright (c) 2023 andyoung. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from vit-pytorch (https://github.com/lucidrains/vit-pytorch)
# ------------------------------------------------------------------------
"""RT1 decoder transformer implemented with pytorch."""
import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn

# helpers


def pair(t):
    """Ensure that the input is a tuple, converting it to a tuple if it's not.

    Args:
        t: A value or tuple.

    Returns:
        tuple: If the input is already a tuple, it's returned as is. If not, it's converted into a
        tuple with two elements, where both elements have the same value as the input.

    Example usage:
    ```python
    value = 64
    result = pair(value)  # Converts the value into (64, 64)

    values = (32, 48)
    result = pair(values)  # Returns the values tuple as is
    ```
    """
    return t if isinstance(t, tuple) else (t, t)

# classes


class LayerNorm(nn.Module):
    """LayerNorm is a PyTorch module that implements layer normalization.

    Args:
        dim (int): The input dimension.

    Attributes:
        gamma (nn.Parameter): A learnable scale parameter.
        beta (torch.Tensor): A zero-initialized bias parameter.

    Example usage:
    ```python
    layer_norm = LayerNorm(dim=512)
    input_data = torch.randn(32, 64, 512)  # Batch of input data
    output_data = layer_norm(input_data)
    ```

    The `LayerNorm` class applies layer normalization to input data, which is a common
    technique used in neural networks for stabilizing training and improving performance.
    """

    def __init__(self, dim):
        """Initialize a new LayerNorm instance.

        Args:
            dim (int): The input dimension.
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        """Apply layer normalization to the input data.

        Args:
            x (torch.Tensor): Input data with shape (B, N, dim).

        Returns:
            torch.Tensor: Normalized data with the same shape as the input.

        Example usage:
        ```python
        layer_norm = LayerNorm(dim=512)
        input_data = torch.randn(32, 64, 512)  # Batch of input data
        output_data = layer_norm(input_data)
        ```
        """
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class PreNorm(nn.Module):
    """PreNorm is a PyTorch module that implements layer normalization followed
    by a transformation function for input data in a transformer model.

    Args:
        dim (int): The input dimension.
        fn (callable): The transformation function to be applied.

    Attributes:
        norm (nn.LayerNorm): Layer normalization module.
        fn (callable): The transformation function.

    Example usage:
    ```python
    prenorm_layer = PreNorm(dim=512, fn=Attention(dim=512, heads=8))
    input_data = torch.randn(32, 64, 512)  # Batch of input data
    output_data = prenorm_layer(input_data)
    ```

    The `PreNorm` class applies layer normalization to input data followed by a
    transformation function, which is a common practice in transformer architectures.
    """

    def __init__(self, dim, fn):
        """Initialize a new PreNorm instance.

        Args:
            dim (int): The input dimension.
            fn (callable): The transformation function to be applied.
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        """Apply layer normalization followed by the transformation function to
        the input data.

        Args:
            x (torch.Tensor): Input data with shape (B, N, dim).
            **kwargs: Additional keyword arguments to be passed to the transformation function.

        Returns:
            torch.Tensor: Transformed data with the same shape as the input.

        Example usage:
        ```python
        prenorm_layer = PreNorm(dim=512, fn=Attention(dim=512, heads=8))
        input_data = torch.randn(32, 64, 512)  # Batch of input data
        output_data = prenorm_layer(input_data)
        ```
        """
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """FeedForward is a PyTorch module that implements a feedforward network
    for transforming input data in a transformer model.

    Args:
        dim (int): The input dimension.
        hidden_dim (int): The dimension of the hidden layer.
        dropout (float): Dropout rate applied to the feedforward network.

    Attributes:
        net (nn.Sequential): The feedforward network.

    Example usage:
    ```python
    feedforward = FeedForward(dim=512, hidden_dim=2048, dropout=0.1)
    input_data = torch.randn(32, 64, 512)  # Batch of input data
    output_data = feedforward(input_data)
    ```

    The `FeedForward` class implements a feedforward network that is commonly used
    in transformer architectures to transform input data.
    """

    def __init__(self, dim, hidden_dim, dropout=0.):
        """Initialize a new FeedForward instance.

        Args:
            dim (int): The input dimension.
            hidden_dim (int): The dimension of the hidden layer.
            dropout (float): Dropout rate applied to the feedforward network.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """Apply the feedforward network to the input data.

        Args:
            x (torch.Tensor): Input data with shape (B, N, dim).

        Returns:
            torch.Tensor: Transformed data with the same shape as the input.

        Example usage:
        ```python
        feedforward = FeedForward(dim=512, hidden_dim=2048, dropout=0.1)
        input_data = torch.randn(32, 64, 512)  # Batch of input data
        output_data = feedforward(input_data)
        ```
        """
        return self.net(x)

# PositionalEncoding before transformer decoder


class PositionalEncoding(torch.nn.Module):
    """PositionalEncoding is a PyTorch module that adds positional encodings to
    input sequences.

    Args:
        d_model (int): The model dimension.
        max_seq_len (int): The maximum sequence length.

    Attributes:
        d_model (int): The model dimension.
        max_seq_len (int): The maximum sequence length.
        positional_encoding (torch.Tensor): The positional encoding matrix.

    Example usage:
    ```python
    positional_encoder = PositionalEncoding(d_model=512, max_seq_len=100)
    input_data = torch.randn(32, 100, 512)  # Batch of input sequences
    output_data = positional_encoder(input_data)
    ```

    The `PositionalEncoding` class adds positional encodings to input sequences, allowing
    the model to capture sequential information.
    """

    def __init__(self, d_model, max_seq_len):
        """Initialize a new PositionalEncoding instance.

        Args:
            d_model (int): The model dimension.
            max_seq_len (int): The maximum sequence length.
        """
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.positional_encoding = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(
                0, d_model, 2,
            ).float() * (-math.log(10000.0) / d_model),
        )
        self.positional_encoding[:, 0::2] = torch.sin(pos * div_term)
        self.positional_encoding[:, 1::2] = torch.cos(pos * div_term)
        self.positional_encoding = self.positional_encoding.unsqueeze(0)

    def forward(self, x):
        """Add positional encodings to the input sequences.

        Args:
            x (torch.Tensor): Input sequences with shape (B, seq_len, d_model).

        Returns:
            torch.Tensor: Sequences with added positional encodings.

        Example usage:
        ```python
        positional_encoder = PositionalEncoding(d_model=512, max_seq_len=100)
        input_data = torch.randn(32, 100, 512)  # Batch of input sequences
        output_data = positional_encoder(input_data)
        ```
        """
        batch_size, seq_len, _ = x.size()
        self.positional_encoding = self.positional_encoding.to(x.device)
        x = x + self.positional_encoding[:, :seq_len, :]
        return x

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_seq_len):
#         super().__init__()
#         pe = torch.zeros(max_seq_len, d_model)
#         position = torch.arange(0, max_seq_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.register_buffer("pe", pe)

#     def forward(self, x):
#         seq_len = x.size(1)
#         x = x + self.pe[:seq_len, :]
#         return x


class Attention(nn.Module):
    """Attention is a PyTorch module that implements scaled dot-product self-
    attention.

    Args:
        dim (int): The model dimension.
        heads (int): The number of self-attention heads.
        dim_head (int): The dimension of each self-attention head.
        dropout (float): Dropout rate applied to attention weights.

    Attributes:
        heads (int): The number of self-attention heads.
        scale (float): The scaling factor for attention scores.
        attend (nn.Softmax): Softmax function to compute attention weights.
        dropout (nn.Dropout): Dropout layer.
        to_qkv (nn.Linear): Linear projection for queries, keys, and values.
        to_out (nn.Sequential or nn.Identity): Output projection.

    Example usage:
    ```python
    attention = Attention(dim=512, heads=8, dim_head=64, dropout=0.1)
    input_data = torch.randn(32, 64, 512)  # Batch of input data
    output_data = attention(input_data)
    ```

    The `Attention` class implements scaled dot-product self-attention, allowing
    you to compute attention weights and apply self-attention to input data.
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        """Initialize a new Attention instance.

        Args:
            dim (int): The model dimension.
            heads (int): The number of self-attention heads.
            dim_head (int): The dimension of each self-attention head.
            dropout (float): Dropout rate applied to attention weights.
        """
        super().__init__()
        inner_dim = dim_head * heads
        project_out = heads != 1 or dim_head != dim

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, x):
        """Apply scaled dot-product self-attention to the input data.

        Args:
            x (torch.Tensor): Input data with shape (B, N, dim).

        Returns:
            torch.Tensor: Output data with the same shape as the input.

        Example usage:
        ```python
        attention = Attention(dim=512, heads=8, dim_head=64, dropout=0.1)
        input_data = torch.randn(32, 64, 512)  # Batch of input data
        output_data = attention(input_data)
        ```
        """
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(
                t, 'b n (h d) -> b h n d', h=self.heads,
            ), qkv,
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    """Transformer is a PyTorch module that implements a standard transformer
    architecture.

    Args:
        dim (int): The model dimension.
        depth (int): The depth (number of layers) of the transformer.
        heads (int): The number of self-attention heads.
        dim_head (int): The dimension of each self-attention head.
        mlp_dim (int): The dimension of the feedforward network in the transformer.
        dropout (float): Dropout rate applied to the transformer.

    Attributes:
        layers (nn.ModuleList): A list of transformer layers.

    Example usage:
    ```python
    transformer = Transformer(dim=512, depth=6, heads=8, dim_head=64, mlp_dim=1024)
    input_data = torch.randn(32, 64, 512)  # Batch of input data
    output_data = transformer(input_data)
    ```

    The `Transformer` class implements a standard transformer architecture, allowing
    you to apply self-attention and feedforward operations to input data.
    """

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        """Initialize a new Transformer instance.

        Args:
            dim (int): The model dimension.
            depth (int): The depth (number of layers) of the transformer.
            heads (int): The number of self-attention heads.
            dim_head (int): The dimension of each self-attention head.
            mlp_dim (int): The dimension of the feedforward network in the transformer.
            dropout (float): Dropout rate applied to the transformer.
        """
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNorm(
                        dim, Attention(
                            dim, heads=heads,
                            dim_head=dim_head, dropout=dropout,
                        ),
                    ),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                ]),
            )

    def forward(self, x):
        """Apply the transformer to the input data.

        Args:
            x (torch.Tensor): Input data with shape (B, N, dim).

        Returns:
            torch.Tensor: Transformed data with the same shape as the input.

        Example usage:
        ```python
        transformer = Transformer(dim=512, depth=6, heads=8, dim_head=64, mlp_dim=1024)
        input_data = torch.randn(32, 64, 512)  # Batch of input data
        output_data = transformer(input_data)
        ```
        """
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    """ViT (Vision Transformer) is a PyTorch module that implements a vision
    transformer for image classification tasks.

    Args:
        image_size (int or tuple): The size of the input image (height, width).
        patch_size (int or tuple): The size of image patches (height, width).
        num_classes (int): The number of output classes.
        dim (int): The model dimension.
        depth (int): The depth (number of layers) of the transformer.
        heads (int): The number of self-attention heads.
        mlp_dim (int): The dimension of the feedforward network in the transformer.
        pool (str): Pooling type, either 'cls' for cls token or 'mean' for mean pooling.
        channels (int): The number of image channels (default is 3 for RGB images).
        dim_head (int): The dimension of each self-attention head.
        dropout (float): Dropout rate applied to the transformer.
        emb_dropout (float): Dropout rate applied to the input embeddings.

    Attributes:
        to_patch_embedding (nn.Sequential): A sequence of layers to convert image patches
        to embeddings.
        pos_embedding (nn.Parameter): Learnable positional embeddings.
        cls_token (nn.Parameter): Learnable classification token.
        dropout (nn.Dropout): Dropout layer.
        transformer (Transformer): The transformer model.
        pool (str): Pooling type, either 'cls' for cls token or 'mean' for mean pooling.
        to_latent (nn.Identity): An identity function.
        mlp_head (nn.Sequential): A sequential module for classification.

    Example usage:
    ```python
    model = ViT(image_size=(224, 224), patch_size=16, num_classes=1000, dim=768, depth=12,
                heads=12, mlp_dim=3072)
    input_image = torch.randn(32, 3, 224, 224)  # Batch of input images
    output_logits = model(input_image)
    ```

    The `ViT` class implements a vision transformer for image classification tasks,
    allowing you to convert input images into class logits.
    """

    def __init__(
        self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
        pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.,
    ):
        """Initialize a new ViT instance.

        Args:
            image_size (int or tuple): The size of the input image (height, width).
            patch_size (int or tuple): The size of image patches (height, width).
            num_classes (int): The number of output classes.
            dim (int): The model dimension.
            depth (int): The depth (number of layers) of the transformer.
            heads (int): The number of self-attention heads.
            mlp_dim (int): The dimension of the feedforward network in the transformer.
            pool (str): Pooling type, either 'cls' for cls token or 'mean' for mean pooling.
            channels (int): The number of image channels (default is 3 for RGB images).
            dim_head (int): The dimension of each self-attention head.
            dropout (float): Dropout rate applied to the transformer.
            emb_dropout (float): Dropout rate applied to the input embeddings.
        """
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * \
            (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {
            'cls', 'mean',
        }, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                p1=patch_height, p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout,
        )
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
        )

    def forward(self, img):
        """Forward pass to classify input images into classes.

        Args:
            img (torch.Tensor): Input images with shape (B, C, H, W).

        Returns:
            torch.Tensor: Logits for classification.

        Example usage:
        ```python
        model = ViT(image_size=(224, 224), patch_size=16, num_classes=1000, dim=768, depth=12,
                    heads=12, mlp_dim=3072)
        input_image = torch.randn(32, 3, 224, 224)  # Batch of input images
        output_logits = model(input_image)
        ```
        """
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)


class Transformers_Decoder(nn.Module):
    """Transformers_Decoder is a PyTorch module that uses a transformer
    architecture to decode sequences of tokens into a probability distribution
    over possible actions.

    Args:
        dim (int): The model dimension.
        depth (int): The depth (number of layers) of the transformer.
        heads (int): The number of self-attention heads.
        dim_head (int): The dimension of each self-attention head.
        mlp_dim (int): The dimension of the feedforward network in the transformer.
        dropout (float): Dropout rate applied to the input embeddings.
        d_model (int): The dimension of the model, which defaults to 512.
        max_seq_len (int): The maximum sequence length, which defaults to 48.
        num_actions (int): The number of possible actions.
        vocab_size (int): The size of the token vocabulary.

    Attributes:
        positionalencoding (PositionalEncoding): A positional encoding layer.
        dropout (nn.Dropout): Dropout layer.
        transformer (Transformer): The transformer model.
        to_logits (nn.Sequential): Sequential layers to produce action logits.

    Example usage:
    ```python
    decoder = Transformers_Decoder(dim=512, depth=6, heads=8, dim_head=64, mlp_dim=1024)
    input_tokens = torch.randint(0, 256, (32, 48))  # Batch of input tokens
    action_logits = decoder(input_tokens)
    ```

    The `Transformers_Decoder` class decodes sequences of tokens into action logits
    using a transformer architecture.
    """

    def __init__(
        self, dim, depth, heads, dim_head, mlp_dim, dropout=0., d_model=512,
        max_seq_len=48, num_actions=11, vocab_size=256,
    ):
        """Initialize a new Transformers_Decoder instance.

        Args:
            dim (int): The model dimension.
            depth (int): The depth (number of layers) of the transformer.
            heads (int): The number of self-attention heads.
            dim_head (int): The dimension of each self-attention head.
            mlp_dim (int): The dimension of the feedforward network in the transformer.
            dropout (float): Dropout rate applied to the input embeddings.
            d_model (int): The dimension of the model, which defaults to 512.
            max_seq_len (int): The maximum sequence length, which defaults to 48.
            num_actions (int): The number of possible actions.
            vocab_size (int): The size of the token vocabulary.
        """
        super().__in
        super().__init__()
        self.positionalencoding = PositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.to_logits = nn.Sequential(
            LayerNorm(d_model),
            nn.Linear(d_model, num_actions * vocab_size),
            Rearrange('... (a b) -> ... a b', b=vocab_size),
        )
        # self.output_tokens = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """Decode input tokens into action logits.

        Args:
            x (torch.Tensor): Input tokens with shape (B, N).

        Returns:
            torch.Tensor: Action logits with shape (B, N, num_actions, vocab_size).

        Example usage:
        ```python
        decoder = Transformers_Decoder(dim=512, depth=6, heads=8, dim_head=64, mlp_dim=1024)
        input_tokens = torch.randint(0, 256, (32, 48))  # Batch of input tokens
        action_logits = decoder(input_tokens)
        ```
        """
        B, N, C = x.shape
        x += self.positionalencoding(x)
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.to_logits(x)
        return x


if __name__ == '__main__':
    # The current output vector form is correct, but I am not sure where：
    # 1.Transformer(dim, depth, heads, dim_head, mlp_dim)中dim = 512 or 4096,
    # The code is inconsistent with the paper description
    # 2.Regarding the action tokenizer step, is it necessary to initialize
    # the action space according to the code in tensorflow?
    x = torch.randn(48, 512).unsqueeze(0)
    # model = Transformers_Decoder(dim=4096, depth=8, heads=8, dim_head=512, mlp_dim=512,
    # dropout = 0., d_model = 512, max_seq_len = 48, num_actions = 11, vocab_size = 256)
    model = Transformers_Decoder(
        dim=512, depth=8, heads=8, dim_head=64, mlp_dim=512,
        dropout=0., d_model=512, max_seq_len=48, num_actions=11,
        vocab_size=256,
    )
    logits_outputs = model(x)
    print(logits_outputs.shape)
