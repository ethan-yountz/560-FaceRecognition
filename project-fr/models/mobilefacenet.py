#!/usr/bin/env python3
"""Paper-style MobileFaceNet backbone for face recognition."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, linear=False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = None if linear else nn.PReLU(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.prelu is not None:
            x = self.prelu(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, stride):
        super().__init__()
        hidden_channels = in_channels * expansion
        self.use_residual = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            ConvBlock(in_channels, hidden_channels, kernel_size=1),
            ConvBlock(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=hidden_channels,
            ),
            ConvBlock(hidden_channels, out_channels, kernel_size=1, linear=True),
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


def make_stage(in_channels, out_channels, expansion, repeats, stride):
    blocks = [Bottleneck(in_channels, out_channels, expansion=expansion, stride=stride)]
    blocks.extend(Bottleneck(out_channels, out_channels, expansion=expansion, stride=1) for _ in range(repeats - 1))
    return nn.Sequential(*blocks)


class MobileFaceNet(nn.Module):
    """
    MobileFaceNet backbone following the paper's table-style architecture for 112x112 inputs.

    The paper architecture uses a 128-dimensional embedding after the linear GDConv head.
    """

    def __init__(self, embedding_dim=128):
        super().__init__()
        if embedding_dim != 128:
            raise ValueError("Exact MobileFaceNet uses a 128-dimensional embedding; set --embedding_dim 128.")

        self.embedding_dim = embedding_dim
        self.stem = nn.Sequential(
            ConvBlock(3, 64, kernel_size=3, stride=2, padding=1),
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1, groups=64),
        )
        self.body = nn.Sequential(
            make_stage(64, 64, expansion=2, repeats=5, stride=2),
            make_stage(64, 128, expansion=4, repeats=1, stride=2),
            make_stage(128, 128, expansion=2, repeats=6, stride=1),
            make_stage(128, 128, expansion=4, repeats=1, stride=2),
            make_stage(128, 128, expansion=2, repeats=2, stride=1),
        )
        self.conv_sep = ConvBlock(128, 512, kernel_size=1, stride=1, padding=0)
        self.gdconv = ConvBlock(512, 512, kernel_size=7, stride=1, padding=0, groups=512, linear=True)
        self.proj = ConvBlock(512, embedding_dim, kernel_size=1, stride=1, padding=0, linear=True)

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        x = self.conv_sep(x)
        x = self.gdconv(x)
        x = self.proj(x)
        return torch.flatten(x, 1)

    def encode(self, images):
        return F.normalize(self.forward(images), p=2, dim=1)
