#!/usr/bin/python3

import torch
import torch.nn as nn
from math import sqrt
from torch.nn import functional as F


class CausalConv1d(nn.Module):
    def __init__(self, in_channels,
                 out_channels, kernel_size, dilation, bias=True):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=self.padding, dilation=dilation, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        if self.padding != 0:
            x = x[:, :, :-self.padding]
        return x


class ResidualBlock(nn.Module):
    def __init__(self, residual_channels, dilation):
        super(ResidualBlock, self).__init__()
        # TODO: replace with learnable dilation and padding 1D convolution
        self.dilated_conv = nn.Conv1d(
            residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.conv1x1 = nn.Conv1d(residual_channels, 2 * residual_channels, 1)
        self.norm = nn.BatchNorm1d(residual_channels)

    def forward(self, x):
        residual = x
        x = self.dilated_conv(x)

        gate, filter = torch.chunk(x, 2, dim=1)
        x = torch.sigmoid(gate) * torch.tanh(filter)

        x = self.conv1x1(x)
        x, skip = torch.chunk(x, 2, dim=1)

        return self.norm(x + residual), skip


class WaveNet(nn.Module):
    def __init__(self, input_channels=2,
                 residual_channels=512,
                 residual_layers=30, dilation_cycle_length=10):
        super(WaveNet, self).__init__()
        self.input_projection = CausalConv1d(
            input_channels, residual_channels, 1, 1)

        self.residual_layers = nn.ModuleList([
            ResidualBlock(residual_channels, 2**(i % dilation_cycle_length))
            for i in range(residual_layers)
        ])

        self.skip_projection = nn.Conv1d(
            residual_channels, residual_channels, 1)
        self.output_projection = nn.Conv1d(
            residual_channels, input_channels, 1)

    def forward(self, x):
        x = self.input_projection(x)
        skip = None
        for layer in self.residual_layers:
            x, s = layer(x)
            skip = s if skip is None else skip + s

        # normalize with sqrt(num_residual_layers)
        x = skip / sqrt(len(self.residual_layers))
        x = F.relu(skip)
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x
