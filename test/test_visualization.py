#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import cv2
import sys
import argparse


_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
try:
    sys.path.append(os.path.join(_CURRENT_DIR, ".."))
    from config import Config
except:
    print("cannot load module")
    exit(1)


parser = argparse.ArgumentParser()
parser.add_argument("--idx", type=int, default=0, help="index of the images")
parser.add_argument(
    "--dataset",
    type=str,
    default="lip_dataset",
    help="name of the dataset to use",
)
parsed_args = parser.parse_args()


def main(args):
    dt_config = Config()
    if not args.dataset or args.dataset not in dt_config.DATASETS.keys():
        raise Exception(
            "specify one of the datasets to use in {}".format(
                list(dt_config.DATASETS.keys())
            )
        )

    DatasetClass = dt_config.DATASETS[args.dataset]

    dataset = DatasetClass(data_path=dt_config.DATA_PATH, phase="val")

    assert args.idx < len(dataset)
    img = dataset.get_overlay_image(idx=args.idx)
    cv2.imshow("visualized_bboxes", cv2.resize(img, (1000, 1000)))
    cv2.waitKey(0)
    cv2.imwrite("visualized_bboxes.png", img)


if __name__ == "__main__":
    main(parsed_args)
