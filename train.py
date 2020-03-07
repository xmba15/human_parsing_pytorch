#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch
from tensorboardX import SummaryWriter
from trainer import Trainer
from config import Config
from data_loader import LipDataTransform
from models import PSPNet, PSPLoss


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--num_epoch", type=int, default=30)
parser.add_argument("--save_period", type=int, default=5)
parser.add_argument("--snapshot", type=str)
parser.add_argument("--batch_multiplier", default=6, type=int)
parser.add_argument("--dataset", type=str, help="name of the dataset to use")
parsed_args = parser.parse_args()


def train_process(args, dt_config, dataset_class, data_transform_class):
    # input_size = (params["img_h"], params["img_w"])
    input_size = (475, 475)
    num_classes = 20

    # transforms = [
    #     OneOf([IAAAdditiveGaussianNoise(), GaussNoise()], p=0.5),
    #     # OneOf(
    #     #     [
    #     #         MedianBlur(blur_limit=3),
    #     #         GaussianBlur(blur_limit=3),
    #     #         MotionBlur(blur_limit=3),
    #     #     ],
    #     #     p=0.1,
    #     # ),
    #     RandomGamma(gamma_limit=(80, 120), p=0.5),
    #     RandomBrightnessContrast(p=0.5),
    #     HueSaturationValue(
    #         hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=10, p=0.5
    #     ),
    #     ChannelShuffle(p=0.5),
    #     HorizontalFlip(p=0.5),
    #     Cutout(num_holes=2, max_w_size=40, max_h_size=40, p=0.5),
    #     Rotate(limit=20, p=0.5, border_mode=0),
    # ]

    data_transform = data_transform_class(
        num_classes=num_classes, input_size=input_size
    )
    train_dataset = dataset_class(
        data_path=dt_config.DATA_PATH, phase="train", transform=data_transform,
    )

    val_dataset = dataset_class(
        data_path=dt_config.DATA_PATH, phase="val", transform=data_transform,
    )

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    val_data_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=True,
    )
    data_loaders_dict = {"train": train_data_loader, "val": val_data_loader}
    tblogger = SummaryWriter(dt_config.LOG_PATH)

    model = PSPNet(num_classes=num_classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = PSPLoss()
    optimizer = torch.optim.SGD(
        [
            {"params": model.feature_conv.parameters(), "lr": 1e-3},
            {"params": model.feature_res_1.parameters(), "lr": 1e-3},
            {"params": model.feature_res_2.parameters(), "lr": 1e-3},
            {"params": model.feature_dilated_res_1.parameters(), "lr": 1e-3},
            {"params": model.feature_dilated_res_2.parameters(), "lr": 1e-3},
            {"params": model.pyramid_pooling.parameters(), "lr": 1e-3},
            {"params": model.decode_feature.parameters(), "lr": 1e-2},
            {"params": model.aux.parameters(), "lr": 1e-2},
        ],
        momentum=0.9,
        weight_decay=0.0001,
    )

    def _lambda_epoch(epoch):
        import math

        max_epoch = args.num_epoch
        return math.pow((1 - epoch / max_epoch), 0.9)

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=_lambda_epoch)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        metric_func=None,
        optimizer=optimizer,
        num_epochs=args.num_epoch,
        save_period=args.save_period,
        config=dt_config,
        data_loaders_dict=data_loaders_dict,
        scheduler=scheduler,
        device=device,
        dataset_name_base=train_dataset.__name__,
        batch_multiplier=args.batch_multiplier,
        logger=tblogger,
    )

    if args.snapshot and os.path.isfile(args.snapshot):
        trainer.resume_checkpoint(args.snapshot)

    with torch.autograd.set_detect_anomaly(True):
        trainer.train()

    tblogger.close()


def main(args):
    dt_config = Config()
    dt_config.display()
    if not args.dataset or args.dataset not in dt_config.DATASETS.keys():
        raise Exception(
            "specify one of the datasets to use in {}".format(
                list(dt_config.DATASETS.keys())
            )
        )

    dataset = args.dataset
    dataset_class = dt_config.DATASETS[dataset]
    data_transform_class = LipDataTransform
    train_process(args, dt_config, dataset_class, data_transform_class)


if __name__ == "__main__":
    main(parsed_args)
