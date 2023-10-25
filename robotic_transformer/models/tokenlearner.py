# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------
# Robotic-transformer-torch
# Copyright (c) 2023 andyoung. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from tokenlearner-pytorch (https://github.com/rish-16/tokenlearner-pytorch)
# ------------------------------------------------------------------------
import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    """SpatialAttention is a PyTorch module that calculates spatial attention
    weights for input features, based on the maximum and average values of
    feature maps.

    Example usage:
    ```python
    attention = SpatialAttention()
    input_features = torch.rand(32, 16, 16, 64)  # Batch of input features
    weighted_features, attention_map = attention(input_features)
    ```

    The `SpatialAttention` class computes spatial attention weights and produces
    weighted features for the input data.
    """

    def __init__(self) -> None:
        """Initialize a new SpatialAttention instance."""
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=(1, 1), stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )

        self.sgap = nn.AvgPool2d(2)

    def forward(self, x):
        """Calculate spatial attention weights for input features and produce
        weighted features.

        Args:
            x (torch.Tensor): Input features with shape (B, H, W, C).

        Returns:
            torch.Tensor: Weighted features with shape (B, C).
            torch.Tensor: Attention map with shape (B, H, W, C).

        Example usage:
        ```python
        attention = SpatialAttention()
        input_features = torch.rand(32, 16, 16, 64)  # Batch of input features
        weighted_features, attention_map = attention(input_features)
        ```
        """
        B, H, W, C = x.shape
        # x = x.view(B, C, H, W)
        # There is a problem with the previous code, please modify it as follows
        x = x.permute(0, 3, 1, 2)
        # Get the maximum value of x in the first dimension (channel) and add dimension 1
        # (the original channel dimension)，mx.shape = (B, 1, H, W)
        mx = torch.max(x, 1)[0].unsqueeze(1)
        # Calculate the average on the 1st dimension of input x and add dimension 1
        # (the original channel dimension), avg.shape = (B, 1, H, W)
        avg = torch.mean(x, 1).unsqueeze(1)
        # Feature splicing, including maximum value information and average information,
        # is spliced in dimension 1 (original channel dimension)
        combined = torch.cat([mx, avg], dim=1)  # combined.shape = (B, 2, H, W)
        fmap = self.conv(combined)  # fmap.shape = (B, 1, H, W)
        weight_map = torch.sigmoid(fmap)
        # out.shape = (B, C), x * weight_map = (B, C, H, W)
        out = (x * weight_map).mean(dim=(-2, -1))
        return out, x * weight_map


class TokenLearner(nn.Module):
    """TokenLearner is a PyTorch module that learns token representations from
    spatial features using a set of SpatialAttention modules.

    Args:
        S (int): The number of token representations to learn.

    Attributes:
        tokenizers (nn.ModuleList): A list of SpatialAttention modules for token learning.
        S (int): The number of token representations.

    Example usage:
    ```python
    learner = TokenLearner(S=8)
    spatial_features = torch.rand(32, 16, 16, 64)  # Batch of spatial features
    token_representations = learner(spatial_features)
    ```

    The `TokenLearner` class employs multiple SpatialAttention modules to learn token
    representations from spatial features, producing a set of token representations.
    """

    def __init__(self, S) -> None:
        """Initialize a new TokenLearner instance.

        Args:
            S (int): The number of token representations to learn.
        """
        super().__init__()
        self.S = S
        self.tokenizers = nn.ModuleList([SpatialAttention() for _ in range(S)])

    def forward(self, x):
        """Learn token representations from spatial features.

        Args:
            x (torch.Tensor): Spatial features with shape (B, H, W, C).

        Returns:
            torch.Tensor: Learned token representations with shape (B, S, C).

        Example usage:
        ```python
        learner = TokenLearner(S=8)
        spatial_features = torch.rand(32, 16, 16, 64)  # Batch of spatial features
        token_representations = learner(spatial_features)
        ```
        """
        B, _, _, C = x.shape
        return torch.stack(
            [self.tokenizers[i](x)[0] for i in range(self.S)], dim=1,
        )

# For RT-1-torch, there is actually no use of the following parts


class TokenFuser(nn.Module):
    """TokenFuser is a PyTorch module for fusing token embeddings with spatial
    features using learned weights and spatial attention.

    Args:
        H (int): The height of the spatial features.
        W (int): The width of the spatial features.
        C (int): The number of channels in the spatial features.
        S (int): The dimension of the token embeddings.

    Attributes:
        projection (nn.Linear): Linear projection layer for token embeddings.
        Bi (nn.Linear): Linear transformation for spatial attention weights.
        spatial_attn (SpatialAttention): A spatial attention module.
        S (int): The dimension of the token embeddings.

    Example usage:
    ```python
    fuser = TokenFuser(H=16, W=16, C=64, S=256)
    token_embeddings = torch.rand(32, 256, 64)  # Batch of token embeddings
    spatial_features = torch.rand(32, 16, 16, 64)  # Batch of spatial features
    fused_features = fuser(token_embeddings, spatial_features)
    ```

    The `TokenFuser` class combines token embeddings with spatial features using
    learned weights and spatial attention to produce fused features.
    """

    def __init__(self, H, W, C, S) -> None:
        """Initialize a new TokenFuser instance.

        Args:
            H (int): The height of the spatial features.
            W (int): The width of the spatial features.
            C (int): The number of channels in the spatial features.
            S (int): The dimension of the token embeddings.
        """
        super().__init__()
        self.projection = nn.Linear(S, S, bias=False)
        self.Bi = nn.Linear(C, S)
        self.spatial_attn = SpatialAttention()
        self.S = S

    def forward(self, y, x):
        """Fuse token embeddings with spatial features.

        Args:
            y (torch.Tensor): Token embeddings with shape (B, S, C).
            x (torch.Tensor): Spatial features with shape (B, H, W, C).

        Returns:
            torch.Tensor: Fused features with shape (B, H, W, C).

        Example usage:
        ```python
        fuser = TokenFuser(H=16, W=16, C=64, S=256)
        token_embeddings = torch.rand(32, 256, 64)  # Batch of token embeddings
        spatial_features = torch.rand(32, 16, 16, 64)  # Batch of spatial features
        fused_features = fuser(token_embeddings, spatial_features)
        ```
        """
        B, S, C = y.shape
        B, H, W, C = x.shape
        Y = self.projection(y.view(B, C, S)).view(B, S, C)
        Bw = torch.sigmoid(self.Bi(x)).view(B, H * W, S)  # [B, HW, S]
        BwY = torch.matmul(Bw, Y)
        _, xj = self.spatial_attn(x)
        xj = xj.view(B, H * W, C)
        return (BwY + xj).view(B, H, W, C)


if __name__ == '__main__':
    tklr = TokenLearner(S=8)
    x = torch.rand(6, 32, 32, 512)
    y = tklr(x)  # torch.Size([6, 8, 512])
    print(y.shape)
