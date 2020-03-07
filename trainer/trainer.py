#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import torch
import tqdm
from .trainer_base import BaseTrainer


_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_CURRENT_DIR, ".."))
try:
    from utils import inf_loop
except Exception as e:
    print(e)
    sys.exit(-1)


class Trainer(BaseTrainer):
    def __init__(
        self,
        model,
        criterion,
        metric_func,
        optimizer,
        num_epochs,
        save_period,
        config,
        data_loaders_dict,
        scheduler=None,
        device=None,
        len_epoch=None,
        dataset_name_base="",
        batch_multiplier=1,
        logger=None,
    ):
        super(Trainer, self).__init__(
            model,
            criterion,
            metric_func,
            optimizer,
            num_epochs,
            save_period,
            config,
            device,
            dataset_name_base,
            batch_multiplier,
            logger,
        )

        self.train_data_loader = data_loaders_dict["train"]
        self.val_data_loader = data_loaders_dict["val"]

        self.num_train_imgs = len(self.train_data_loader.dataset)
        self.num_val_imgs = len(self.val_data_loader.dataset)

        if len_epoch is None:
            self._len_epoch = len(self.train_data_loader)
        else:
            self.train_data_loader = inf_loop(self.train_data_loader)
            self._len_epoch = len_epoch

        self._do_validation = self.val_data_loader is not None
        self._scheduler = scheduler

    def _train_epoch(self, epoch):
        self._model.train()

        batch_size = self.train_data_loader.batch_size

        epoch_train_loss = 0.0
        count = self._batch_multiplier
        for batch_idx, (data, target) in enumerate(self.train_data_loader):
            data, target = data.to(self._device), target.to(self._device)

            if count == 0:
                self._optimizer.step()
                self._optimizer.zero_grad()
                count = self._batch_multiplier

            with torch.set_grad_enabled(True):
                output = self._model(data)
                train_loss = (
                    self._criterion(output, target.long())
                    / self._batch_multiplier
                )
                train_loss.backward()
                count -= 1

                if batch_idx % 100 == 0:
                    print(
                        "\n epoch: {} || iter: {} || total_loss: {}".format(
                            epoch,
                            batch_idx,
                            train_loss.item()
                            / batch_size
                            * self._batch_multiplier,
                        )
                    )

                epoch_train_loss += train_loss * self._batch_multiplier
            if batch_idx == self._len_epoch:
                break

        if self._do_validation:
            epoch_val_loss = self._valid_epoch(epoch)

        if self._scheduler is not None:
            self._scheduler.step()

        return (
            epoch_train_loss / self.num_train_imgs,
            epoch_val_loss / self.num_val_imgs,
        )

    def _valid_epoch(self, epoch):
        print("start validation...")
        self._model.eval()

        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_data_loader):
                data, target = data.to(self._device), target.to(self._device)

                output = self._model(data)
                val_loss = self._criterion(output, target.long())
                epoch_val_loss += val_loss

        return epoch_val_loss
