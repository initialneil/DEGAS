import os
import numpy as np
import torch
import torch.nn.functional as thf
from utils.general_utils import make_expon_lr_func
from tqdm import tqdm
from .loss_base import LossBase
from .optim_base import OptimizerBase

# standard 3dgs
class StandardGaussOptimizer(LossBase, OptimizerBase):
    def __init__(self, gs_model, optimizer_config=None) -> None:
        LossBase.__init__(self, gs_model, optimizer_config=optimizer_config)
        OptimizerBase.__init__(self, gs_model, optimizer_config=optimizer_config)

    ##################################################
    def collect_loss(self, iteration, gt_image, render, gt_alpha_mask=None, image_weights=None, **render_pkg):
        # loss
        ret = super().collect_loss(gt_image, render, gt_alpha_mask=gt_alpha_mask, 
                                   image_weights=image_weights,
                                   iteration=iteration)
        loss = ret['loss']

        # opacity
        if self.optimizer_config.get('lambda_alpha', 0) > 0:
            alpha = render_pkg['alpha']
            loss += self.optimizer_config.lambda_alpha * thf.l1_loss(alpha, gt_alpha_mask)

        ret['loss'] = loss
        return ret
    
    ##################################################
    def save_checkpoint(self, model_path, iteration):
        pc_dir = os.path.join(model_path, f'point_cloud/iteration_{iteration}')
        os.makedirs(pc_dir, exist_ok=True)

        self.gs_model.save_ply(os.path.join(pc_dir, 'point_cloud.ply'))
        return pc_dir


