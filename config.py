#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from data_loader import LipDataset


_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


class Config(object):
    def __init__(self):
        self.CURRENT_DIR = _CURRENT_DIR

        self.DATA_PATH = os.path.abspath(os.path.join(_CURRENT_DIR, "data"))

        self.DATASETS = {"lip_dataset": LipDataset}

        self.SAVED_MODEL_PATH = os.path.join(self.CURRENT_DIR, "saved_models")
        if not os.path.isdir(self.SAVED_MODEL_PATH):
            os.system("mkdir -p {}".format(self.SAVED_MODEL_PATH))

        self.LOG_PATH = os.path.join(self.CURRENT_DIR, "logs")
        if not os.path.isdir(self.LOG_PATH):
            os.system("mkdir -p {}".format(self.LOG_PATH))

    def display(self):
        """
        Display Configuration values.
        """
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
