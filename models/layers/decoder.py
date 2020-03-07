#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import Conv2dBatchNormActiv


__all__ = ["DecoderPSPFeature", "AuxiliaryPSPlayers"]


class DecoderPSPFeature(nn.Module):
    def __init__(self, height, width, num_classes, activ_func="relu"):
        super(DecoderPSPFeature, self).__init__()
        self.height = height
        self.width = width

        self.cbr = Conv2dBatchNormActiv(
            in_channels=4096,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=False,
            activ_func=activ_func,
        )

        self.dropout = nn.Dropout2d(p=0.1)
        self.classification = nn.Conv2d(
            in_channels=512,
            out_channels=num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        x = self.cbr(x)
        x = self.dropout(x)
        x = self.classification(x)
        output = F.interpolate(
            x,
            size=(self.height, self.width),
            mode="bilinear",
            align_corners=True,
        )

        return output


class AuxiliaryPSPlayers(nn.Module):
    def __init__(
        self, in_channels, height, width, num_classes, activ_func="relu"
    ):
        super(AuxiliaryPSPlayers, self).__init__()
        self.height = height
        self.width = width

        self.cbr = Conv2dBatchNormActiv(
            in_channels=in_channels,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=False,
            activ_func=activ_func,
        )

        self.dropout = nn.Dropout2d(p=0.1)
        self.classification = nn.Conv2d(
            in_channels=256,
            out_channels=num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        x = self.cbr(x)
        x = self.dropout(x)
        x = self.classification(x)
        output = F.interpolate(
            x,
            size=(self.height, self.width),
            mode="bilinear",
            align_corners=True,
        )

        return output
