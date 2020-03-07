#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "Conv2dBatchNormActiv",
    "FeatureMapConvolution",
    "ResidualBlockPSP",
    "PyramidPooling",
]


def activation_func(activation):
    return nn.ModuleDict(
        [
            ["relu", nn.ReLU(inplace=True)],
            ["leaky_relu", nn.LeakyReLU(negative_slope=0.01, inplace=True)],
            ["selu", nn.SELU(inplace=True)],
            ["none", nn.Identity()],
        ]
    )[activation]


class Conv2dBatchNormActiv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        activ_func="relu",
        bias=False,
    ):
        super(Conv2dBatchNormActiv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            bias=bias,
        )
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activ = activation_func(activ_func)

    def forward(self, x):
        x = self.conv(x)
        try:
            x = self.batchnorm(x)
        except:
            # error on doing batch norm on single chanel input
            x = nn.Identity()(x)

        outputs = self.activ(x)

        return outputs


class FeatureMapConvolution(nn.Module):
    def __init__(self, activ_func="relu"):
        super(FeatureMapConvolution, self).__init__()
        (
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            bias,
        ) = (3, 64, 3, 2, 1, 1, False)
        self.cbnr_1 = Conv2dBatchNormActiv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            activ_func=activ_func,
            bias=bias,
        )
        (
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            bias,
        ) = (64, 64, 3, 1, 1, 1, False)
        self.cbnr_2 = Conv2dBatchNormActiv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            activ_func=activ_func,
            bias=bias,
        )
        (
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            bias,
        ) = (64, 128, 3, 1, 1, 1, False)
        self.cbnr_3 = Conv2dBatchNormActiv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            activ_func=activ_func,
            bias=bias,
        )
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.cbnr_1(x)
        x = self.cbnr_2(x)
        x = self.cbnr_3(x)
        outputs = self.max_pool(x)

        return outputs


class Conv2dBatchNorm(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        bias,
    ):
        super(Conv2dBatchNorm, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            bias=bias,
        )
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        outputs = self.batchnorm(x)

        return outputs


class BottleNeckPSP(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        stride,
        dilation,
        activ_func="relu",
    ):
        super(BottleNeckPSP, self).__init__()
        self.cbr_1 = Conv2dBatchNormActiv(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            activ_func=activ_func,
            bias=False,
        )
        self.cbr_2 = Conv2dBatchNormActiv(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            activ_func=activ_func,
            bias=False,
        )
        self.cb_3 = Conv2dBatchNormActiv(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            activ_func=activ_func,
            bias=False,
        )

        self.cb_residual = Conv2dBatchNorm(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            dilation=1,
            bias=False,
        )

        self.activ = activation_func(activ_func)

    def forward(self, x):
        conv = self.cb_3(self.cbr_2(self.cbr_1(x)))
        residual = self.cb_residual(x)

        return self.activ(conv + residual)


class BottleNeckIdentityPSP(nn.Module):
    def __init__(
        self, in_channels, mid_channels, dilation, activ_func="relu",
    ):
        super(BottleNeckIdentityPSP, self).__init__()
        self.cbr_1 = Conv2dBatchNormActiv(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            activ_func=activ_func,
            bias=False,
        )
        self.cbr_2 = Conv2dBatchNormActiv(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            activ_func=activ_func,
            bias=False,
        )
        self.cb_3 = Conv2dBatchNormActiv(
            in_channels=mid_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            activ_func=activ_func,
            bias=False,
        )

        self.cb_residual = nn.Identity()

        self.activ = activation_func(activ_func)

    def forward(self, x):
        conv = self.cb_3(self.cbr_2(self.cbr_1(x)))
        residual = self.cb_residual(x)

        return self.activ(conv + residual)


class ResidualBlockPSP(nn.Sequential):
    def __init__(
        self,
        n_blocks,
        in_channels,
        mid_channels,
        out_channels,
        stride,
        dilation,
        activ_func="relu",
    ):
        super(ResidualBlockPSP, self).__init__()
        self.add_module(
            "block1",
            BottleNeckPSP(
                in_channels,
                mid_channels,
                out_channels,
                stride,
                dilation,
                activ_func=activ_func,
            ),
        )

        for i in range(n_blocks - 1):
            self.add_module(
                "block" + str(i + 2),
                BottleNeckIdentityPSP(
                    out_channels, mid_channels, dilation, activ_func=activ_func,
                ),
            )


class PyramidPooling(nn.Module):
    def __init__(
        self, in_channels, pool_sizes, height, width, activ_func="relu"
    ):
        super(PyramidPooling, self).__init__()
        self.height = height
        self.width = width
        self.pool_sizes = pool_sizes

        out_channels = int(in_channels / len(pool_sizes))
        self.layer1 = self._make_layer(
            pool_sizes[0],
            in_channels=in_channels,
            out_channels=out_channels,
            activ_func=activ_func,
        )
        self.layer2 = self._make_layer(
            pool_sizes[1],
            in_channels=in_channels,
            out_channels=out_channels,
            activ_func=activ_func,
        )
        self.layer3 = self._make_layer(
            pool_sizes[2],
            in_channels=in_channels,
            out_channels=out_channels,
            activ_func=activ_func,
        )
        self.layer4 = self._make_layer(
            pool_sizes[3],
            in_channels=in_channels,
            out_channels=out_channels,
            activ_func=activ_func,
        )

    def _make_layer(self, pool_size, in_channels, out_channels, activ_func):
        layers = [nn.AdaptiveAvgPool2d(output_size=pool_size)]
        layers.append(
            Conv2dBatchNormActiv(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                bias=False,
                activ_func=activ_func,
            )
        )

        return nn.Sequential(*layers)

    def forward(self, x):
        outs = []
        for i in range(len(self.pool_sizes)):
            out = getattr(self, "layer{}".format(i + 1))(x)
            out = F.interpolate(
                out,
                size=(self.height, self.width),
                mode="bilinear",
                align_corners=True,
            )
            outs.append(out)

        output = torch.cat([x, outs[0], outs[1], outs[2], outs[3]], dim=1)

        return output
