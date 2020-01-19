#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import glob
import cv2
import numpy as np
from .data_loader_base import BaseDataset, BaseDatasetConfig


class LipDatasetConfig(BaseDatasetConfig):
    def __init__(self):
        super(LipDatasetConfig, self).__init__()
        self.CLASSES = [
            "Void",
            "Hat",
            "Hair",
            "Glove",
            "Sunglasses",
            "UpperClothes",
            "Dress",
            "Coat",
            "Socks",
            "Pants",
            "Jumpsuits",
            "Scarf",
            "Skirt",
            "Face",
            "Left-arm",
            "Right-arm",
            "Left-leg",
            "Right-leg",
            "Left-shoe",
            "Right-shoe",
        ]

        self.COLORS = [
            (0, 0, 0),
            (244, 35, 232),
            (255, 255, 0),
            (70, 70, 70),
            (102, 102, 156),
            (190, 153, 153),
            (153, 153, 153),
            (250, 170, 30),
            (220, 220, 0),
            (107, 142, 35),
            (152, 251, 152),
            (70, 130, 180),
            (220, 20, 60),
            (255, 0, 0),
            (0, 0, 142),
            (51, 204, 51),
            (0, 60, 100),
            (0, 80, 100),
            (0, 0, 230),
            (119, 11, 32),
        ]


_lip_dataset_config = LipDatasetConfig()


class LipDataset(BaseDataset):
    def __init__(self, data_path, phase="test", transform=None):
        super(LipDataset, self).__init__(
            data_path,
            phase=phase,
            classes=_lip_dataset_config.CLASSES,
            colors=_lip_dataset_config.COLORS,
            transform=transform,
        )

        _lip_data_path = os.path.join(self._data_path, "LIP")
        _image_data_paths = os.path.join(_lip_data_path, "{}_images".format(phase))

        self._image_paths = glob.glob(os.path.join(_image_data_paths, "*.jpg"))
        self._image_paths.sort(key=BaseDataset.human_sort)

        if self._phase != "test":
            _gt_data_paths = os.path.join(_lip_data_path, "{}_segmentations".format(phase))
            self._gt_paths = glob.glob(os.path.join(_gt_data_paths, "*.png"))
            self._gt_paths.sort(key=BaseDataset.human_sort)

        self._color_idx_dict = BaseDataset.color_to_color_idx_dict(self._colors)

    def weighted_class(self):
        assert "Void" in self._classes
        class_idx_dict = BaseDataset.class_to_class_idx_dict(self._classes)
        weighted = super(LipDataset, self).weighted_class()
        weighted[class_idx_dict["Void"]] = 0

        return weighted
