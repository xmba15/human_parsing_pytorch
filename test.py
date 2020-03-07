#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import torch
import argparse
from config import Config
from models import PSPNet, PSPLoss


parser = argparse.ArgumentParser()
parser.add_argument("--snapshot", type=str)
parser.add_argument(
    "--dataset",
    type=str,
    default="lip_dataset",
    help="name of the dataset to use",
)
parser.add_argument("--image_path", type=str, help="path to the test image")
parser.add_argument(
    "--alpha", type=float, default=0.7, help="overlay parameter"
)
parsed_args = parser.parse_args()


def test_one_image(args, dt_config, dataset_class):
    input_size = (475, 475)
    model_path = args.snapshot
    dataset_instance = dataset_class(data_path=dt_config.DATA_PATH)
    num_classes = dataset_instance.num_classes
    model = PSPNet(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path)["state_dict"])
    model.eval()

    img = cv2.imread(args.image_path)
    processed_img = cv2.resize(img, input_size)
    overlay = np.copy(processed_img)
    processed_img = processed_img / 255.0
    processed_img = torch.tensor(
        processed_img.transpose(2, 0, 1)[np.newaxis, :]
    ).float()
    if torch.cuda.is_available():
        model = model.cuda()
        processed_img = processed_img.cuda()
    output = model(processed_img)[0]
    mask = output.data.max(1)[1].cpu().numpy().reshape(475, 475)
    color_mask = np.array(dataset_instance.colors)[mask]
    alpha = args.alpha
    overlay = (((1 - alpha) * overlay) + (alpha * color_mask)).astype("uint8")
    overlay = cv2.resize(overlay, (img.shape[1], img.shape[0]))
    cv2.imwrite("result.jpg", overlay)


def main(args):
    dt_config = Config()
    if not args.dataset or args.dataset not in dt_config.DATASETS.keys():
        raise Exception(
            "specify one of the datasets to use in {}".format(
                list(dt_config.DATASETS.keys())
            )
        )
    if not args.snapshot or not os.path.isfile(args.snapshot):
        raise Exception("invalid snapshot")
    if not args.image_path or not os.path.isfile(args.image_path):
        raise Exception("invalid image path")

    dataset = args.dataset
    dataset_class = dt_config.DATASETS[dataset]
    test_one_image(args, dt_config, dataset_class)


if __name__ == "__main__":
    main(parsed_args)
