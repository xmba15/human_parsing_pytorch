#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["PSPLoss"]


class PSPLoss(nn.Module):
    def __init__(self, aux_weight=0.4):
        super(PSPLoss, self).__init__()
        self.aux_weight = aux_weight

    def forward(self, outputs, targets):
        output, output_aux = outputs
        loss = F.cross_entropy(output, targets, reduction="mean")
        loss_aux = F.cross_entropy(output_aux, targets, reduction="mean")

        return loss + loss_aux * self.aux_weight
