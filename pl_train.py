import warnings
import time
import numpy as np
import cv2 as cv

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from albumentations.core.serialization import from_dict
from pytorch_lightning.loggers import CometLogger

import pytorch_lightning as pl
from pytorch_toolbelt.losses import JaccardLoss, BinaryFocalLoss

import hydra
from omegaconf import DictConfig
conf_path = './hydra_conf/conf.yaml'

from SINet_ECA_resize_deconv import SINet

from dataset import *
from utils import *
from metrics import *

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class SegmentPeople(pl.LightningModule):
    def __init__(self, cfg, model=None, verbose=False):
        super(SegmentPeople, self).__init__()
        self.cfg = cfg
        config = [[[3, 1], [5, 1]], [[3, 1], [3, 1]],
              [[3, 1], [5, 1]], [[3, 1], [3, 1]], [[5, 1], [3, 2]], [[5, 2], [3, 4]],
              [[3, 1], [3, 1]], [[5, 1], [5, 1]], [[3, 2], [3, 4]], [[3, 1], [5, 2]]]
        self.model = SINet(classes=1, p=2, q=8, config=config, chnn=1)

        # self.loss = BinaryFocalLoss()
        # self.losses = [
        #     ("jaccard", 0.2, JaccardLoss(mode="binary", from_logits=True)),
        #     ("focal", 0.8, BinaryFocalLoss()),
        # ]
        self.loss = torch.nn.BCEWithLogitsLoss()

    def setup(self, stage=0):
        samples = get_eg1800_paths('EG1800')
        test_samples = get_eg1800_paths('EG1800')

        # num_train = int((1 - self.cfg.val_split) * len(samples))

        # self.train_samples = samples[:num_train]
        # self.val_samples = samples[num_train:]

        self.train_samples = samples
        self.val_samples = test_samples

        print("Len train samples = ", len(self.train_samples))
        print("Len val samples = ", len(self.val_samples))

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images = batch['features']
        masks = batch['masks']
        images = images.float()
        masks = masks.float()
        outputs = self(images)

        # total_loss = 0
        logs = {}
        # for loss_name, weight, loss in self.losses:
        #     ls_mask = loss(outputs, masks)
        #     total_loss += weight * ls_mask
        #     logs[f"train_mask_{loss_name}"] = ls_mask
        loss = self.loss(outputs, masks)
        logs["train_loss"] = loss
        logs["lr"] = self._get_current_lr()
        logs["iou"] = binary_mean_iou(outputs, masks)
        metrics = eval_metrics(masks.bool().cpu(), (outputs>0.5).cpu(), 2)
        metrics["lr"] = self._get_current_lr()
        return {"loss": loss, 
                "progress_bar": logs,
                "log": {**logs, **metrics}
                }

    def validation_step(self, batch, batch_idx):
        images = batch['features']
        masks = batch['masks']
        images = images.float()
        masks = masks.float()
        outputs = self(images)
        
        result = {}
        # for loss_name, _, loss in self.losses:
        #     result[f"val_mask_{loss_name}"] = loss(outputs, masks)
        loss = self.loss(outputs, masks)
        result["val_loss"] = loss
        result["val_iou"] = binary_mean_iou(outputs, masks)

        metrics = eval_metrics(masks.bool().cpu(), (outputs>0.5).cpu(), 2)

        return {**metrics, **result}

    def validation_epoch_end(self, outputs):
        avg_val_iou = find_average(outputs, "val_iou")
        loss_val = sum(output['val_loss'] for output in outputs) / len(outputs)
        overall_acc = sum(output['overall_acc'] for output in outputs) / len(outputs)
        avg_per_class_acc = sum(output['avg_per_class_acc'] for output in outputs) / len(outputs)
        avg_jacc = sum(output['avg_jacc'] for output in outputs) / len(outputs)
        avg_dice = sum(output['avg_dice'] for output in outputs) / len(outputs)
        mean_metrics = {
            'val_loss': loss_val,
            'val_overall_acc': overall_acc,
            'val_avg_per_class_acc': avg_per_class_acc,
            'val_avg_jacc': avg_jacc,
            'val_avg_dice': avg_dice,
            'val_iou': avg_val_iou
            }
            
        return {'val_iou': avg_val_iou, 
                'progress_bar': mean_metrics, 
                'log': mean_metrics
                }

    def configure_optimizers(self):
        params = filter(lambda x: x.requires_grad, self.model.parameters())
        self.optimizer = object_from_dict(self.cfg.optimizer, params=params)
        self.scheduler = object_from_dict(self.cfg.scheduler, optimizer=self.optimizer)
        # return [self.optimizer], [self.scheduler], "monitor": "metric_to_track"
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler, "monitor": "val_iou"}

    def train_dataloader(self):
        train_sampler = None
        # train_aug = from_dict(self.cfg.train_aug)
        train_aug = get_transform(self.cfg.train_aug)

        result = DataLoader(
            SegmentationDataset(self.train_samples, train_aug, length=None),
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )
        print("Train dataloader = ", len(result))
        return result

    def val_dataloader(self):
        train_sampler = None
        # val_aug = from_dict(dict(self.cfg.val_aug))
        val_aug = get_transform(self.cfg.val_aug)
        # val_aug = {}
        result = DataLoader(
            SegmentationDataset(self.val_samples, val_aug, length=None),
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

        print("Val dataloader = ", len(result))

        return result

    def _get_current_lr(self):
        return list(map(lambda group: group["lr"], self.optimizer.param_groups))[0]
    




@hydra.main(config_path=conf_path)
def main(cfg : DictConfig):
    if cfg.seed:
        set_determenistic(cfg.seed)
    pipeline = SegmentPeople(cfg)
    print(cfg.pretty())

    trainer = object_from_dict(
        cfg.trainer,
        # logger=CometLogger(
        #     api_key="", 
        #     project_name="person-segmentation", 
        #     workspace="", 
        #     experiment_name=cfg.experiment_name),
        checkpoint_callback=object_from_dict(cfg.checkpoint_callback),
    )

    trainer.fit(pipeline)



# @hydra.main(config_path=conf_path)
# def kfold_main(cfg):
#     if cfg.seed:
#         set_determenistic(cfg.seed)

#     for fold in range(cfg.dataset.kfold):
#         logger = TensorBoardLogger(f"tb_logs", name=f'model_{fold}')
#         setattr(cfg.dataset, 'nfold', fold)
#         pipeline = UnetPipeline(cfg)

#         trainer = pl.Trainer(
#             weights_summary=None,
#             max_epochs=cfg.epochs,
#             show_progress_bar=True,
#             early_stop_callback=object_from_dict(cfg.earlystop),
#             logger=logger,
#         )
#         trainer.fit(pipeline)



if __name__ == "__main__":
    main()
    # kfold_main()
