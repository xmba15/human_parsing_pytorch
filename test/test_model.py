#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import torch


_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
try:
    sys.path.append(os.path.join(_CURRENT_DIR, ".."))
    from models import PSPNet
except Exception as e:
    print(e)
    exit(1)


def main():
    model = PSPNet(num_classes=12)
    input = torch.randn(1, 3, 475, 475)
    output, output_aux = model(input)
    print(output.shape)
    print(output_aux.shape)


if __name__ == "__main__":
    main()
