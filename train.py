#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 16:16:23 2021

@author: Bruce Muller
"""

import os
import torch

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from model.cnn_geometric_model import CNNGeometric
from model.loss import RelativePoseLoss , VGGPerceptualLoss
from util.homography import GroundPlaneRepToHomography
from options.options import ArgumentParser
from loader.loader_kitti_raw import KittiPairLoader_raw
from util.torch_util import save_checkpoint, warp_image
from pytorch_lightning.loggers.neptune import NeptuneLogger
import sys
sys.path.append('core')

# torch.autograd.set_detect_anomaly(True)
#NCCL_DEBUG=INFO

def uppercut(img):
    h = int(img.size()[2] / 2)
    img = img.clone()
    img[:,:,0:h,:] = 0.0
    return img

def uppercut_m(mask):
    h = int(mask.size()[2] / 2)
    mask = mask.clone()
    mask[:,:,0:h,:] = True
    return mask








class RelativePoseModel(pl.LightningModule):

    def __init__(self, hparam , args, arg_groups, expname, len_train_loader,len_val_loader):
        super().__init__()

        self.hparam = hparam
        self.args = args

        self.model = CNNGeometric(use_cuda=True,output_dim=hparam.num_params, train_fe = args.train_fe, **arg_groups['model'])
        self.ploss = VGGPerceptualLoss(mask_norm=args.mask_norm)

        self.groundPlaneRepToHomography = GroundPlaneRepToHomography(fix_priors=False)

        self.checkpoint_path = os.path.join(args.log_dir, expname ,expname + '.pth.tar')
        self.epoch = 1
        self.optimizer = False
        self.scheduler = False


    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparam.lr)   # self.hparam.lr
        self.optimizer = optimizer
        if self.args.lr_scheduler:
            print("USING SCHEDULER")
#            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, threshold = 0.0001, verbose=True)   # optimizer, 'min', patience=10, threshold = 0.0001, verbose=True
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 1000)   # optimizer, 'min', patience=10, threshold = 0.0001, verbose=True
            self.scheduler = scheduler
#            scheduler = {'scheduler': scheduler , 'monitor': 'epoch_avg_loss','interval': 'epoch'}
            scheduler = {'scheduler': scheduler ,'interval': 'step'}
            return [optimizer] , [scheduler]
        else:
            return optimizer

    def app_loss(self, batch, H, H_inv):
        app_loss = 0
        for idx in range(self.hparam.num_scales):
            if self.hparam.segment_lossInput:
                source = batch['source_images_masked'][idx]
                target = batch['target_images_masked'][idx]
            else:
                source = batch['source_scales'][idx]
                target = batch['target_scales'][idx]

            if self.hparam.symmetric:
                warped_src_zeros_right = warp_image(source, H, border_padding=False)
                mask_zeros_right = ~ warped_src_zeros_right.type(torch.cuda.BoolTensor)
                if self.hparam.upperCut:
                    app_loss_right = self.ploss(uppercut(target), uppercut(warped_src_zeros_right), uppercut_m(mask_zeros_right))
                else:
                    app_loss_right = self.ploss(target, warped_src_zeros_right, mask_zeros_right)
            else:
                app_loss_right = 0

            warped_src_zeros_left = warp_image(target, H_inv, border_padding=False)
            mask_zeros_left = ~ warped_src_zeros_left.type(torch.cuda.BoolTensor)
            if self.hparam.upperCut:
                app_loss_left = self.ploss(uppercut(source), uppercut(warped_src_zeros_left), uppercut_m(mask_zeros_left))
            else:
                app_loss_left = self.ploss(source, warped_src_zeros_left, mask_zeros_left)

            app_loss = app_loss_right + app_loss_left + app_loss
        return app_loss

    def training_step(self, batch, batch_idx):
        theta, loss = self.inferenceAndLosses(batch)

        loss_R, loss_t = RelativePoseLoss.forward(theta, batch['t_1to2_tar'], batch['R_1to2_tar'])

        # Logging
        # self.log('total_loss', loss, on_step=False, on_epoch=True, sync_dist=True, logger=True)
        #
        # if self.hparam.appLoss:
        #     self.log('train_app_loss_left', app_loss, on_step=False, on_epoch=True, sync_dist=True, logger=True)
        #
        # self.log('training_loss_t', loss_t, on_step=False, on_epoch=True, sync_dist=True, logger=True)
        # self.log('training_loss_R', loss_R, on_step=False, on_epoch=True, sync_dist=True, logger=True)

        return loss
           
        

    def validation_step(self, batch, batch_idx):

        theta, loss = self.inferenceAndLosses(batch)

        loss_R, loss_t = RelativePoseLoss.forward(theta, batch['t_1to2_tar'], batch['R_1to2_tar'])
        # self.log('total_loss_val', loss, on_step=False, on_epoch=True, sync_dist=True, logger=True)
        # self.log('training_loss_t_val', loss_t, on_step=False, on_epoch=True, sync_dist=True, logger=True)
        # self.log('training_loss_R_val', loss_R, on_step=False, on_epoch=True, sync_dist=True, logger=True)

        return loss  # {'loss': loss , 'myHLoss': H_loss}

    def inferenceAndLosses(self, batch):
        theta = self(batch)

        H = self.groundPlaneRepToHomography(theta, batch['Ki'], batch['Kj'], batch['T_1241_1']).flatten(1)
        H_inv = torch.inverse(H.reshape(-1, 3, 3)).flatten(1)

        if self.hparam.appLoss:
            app_loss = self.app_loss(batch, H, H_inv)

            loss_cz = torch.mean((theta[:, 0] - 1.65) ** 2)
            loss_roll = torch.mean((theta[:, 1] ** 2))
            loss_pitch = torch.mean((theta[:, 2]) ** 2)
            loss_cz_cam2 = torch.mean((theta[:, 5] - 1.65) ** 2)
            loss_roll_cam2 = torch.mean((theta[:, 6] ** 2))
            loss_pitch_cam2 = torch.mean((theta[:, 7]) ** 2)
            priors = loss_cz + loss_roll + loss_pitch + loss_cz_cam2 + loss_roll_cam2 + loss_pitch_cam2
            loss = (100.0 * priors) + app_loss

        return theta, loss



    def training_epoch_end(self, training_step_output_result):

        if not self.args.debug:
            if (self.epoch % 5) == 0:
                if self.args.lr_scheduler:
                    state = {'epoch': self.epoch + 1, 'args': args, 'hparam': self.hparam, 'state_dict': self.model.state_dict(),
                             'optimizer': self.optimizer.state_dict(),
                             'scheduler': self.scheduler.state_dict()}
                else:
                    state = {'epoch': self.epoch + 1, 'args': args, 'hparam': self.hparam, 'state_dict': self.model.state_dict(),
                             'optimizer': self.optimizer.state_dict()}
                save_checkpoint(state, False, self.checkpoint_path, epoch_save=True, epoch=self.epoch)

        self.epoch += 1

        return None

if __name__ == '__main__':

    argparser =  ArgumentParser(mode='train')
    args,arg_groups = argparser.parse()
    hyperparams = argparser.hyperparams
    num_exps = 7
    gpu = '1'

    for hparam_trial in hyperparams.trials(num_exps):
        
        dataset = KittiPairLoader_raw(...)      # Provide dataloaders.
        dataset_val = KittiPairLoader_raw(...)
        train_loader = DataLoader(dataset, batch_size=hparam_trial.batch_size,shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(dataset_val, batch_size=hparam_trial.batch_size,shuffle=False, num_workers=7)

        expname = 'exp1'
        
        neptune_logger = NeptuneLogger(
                offline_mode=True,
                api_key="",
                project_name="brm512/AppearanceLoss",
                close_after_fit=False,
                experiment_name=expname,  # Optional,
                params={**vars(args) , **vars(hparam_trial)},
                upload_stdout=False,
                upload_stderr=False,
                 )


        
        checkpoint_path = os.path.join(args.log_dir, expname)
        model = RelativePoseModel(hparam_trial , args, arg_groups, expname, len(train_loader) , len(val_loader))
        trainer = pl.Trainer(gpus=gpu, default_root_dir=checkpoint_path  , checkpoint_callback=False, logger=neptune_logger, num_sanity_val_steps=0,max_epochs=100, limit_train_batches=4 ,limit_val_batches=4 )
        trainer.fit(model, train_loader, val_loader)