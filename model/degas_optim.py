import os
import torch
import torch.nn as nn
import torch.nn.functional as thf
from utils.loss_utils import (
    l1_loss,
    predicted_normal_loss, delta_normal_loss, zero_one_loss, 
    nonsaturating_loss, logistic_loss
)
from .loss_base import LossBase
from .optim_base import OptimizerBase, requires_grad

# DEGAS
class DEGASOptimizer(LossBase, OptimizerBase):
    def __init__(self, gs_model, smplx_optim=None, optimizer_config=None) -> None:
        LossBase.__init__(self, gs_model, optimizer_config=optimizer_config)
        OptimizerBase.__init__(self, gs_model, smplx_optim=smplx_optim, optimizer_config=optimizer_config)

    def setup_optimizer(self, optimizer_config):
        OptimizerBase.setup_optimizer(self, optimizer_config)

    ##################################################
    def collect_loss(self, iteration, batch, **render_pkg):
        ret = super().collect_loss(iteration=iteration, **render_pkg)
        loss = ret['loss']

        gs_model = self.gs_model

        # zero or one
        if self.optimizer_config.get('lambda_zero_one', 0) > 0:
            loss += self.optimizer_config.lambda_zero_one * zero_one_loss(render_pkg['alpha'])

        # normal depth consistency
        if self.optimizer_config.get('lambda_predicted_normal', 0) > 0:
            loss += self.optimizer_config.lambda_predicted_normal * predicted_normal_loss(render_pkg["normal"], render_pkg["normal_ref"], render_pkg["alpha"])
        
        # normal delta reg
        if self.optimizer_config.get('lambda_delta_reg', 0) > 0:
            loss += self.optimizer_config.lambda_delta_reg * delta_normal_loss(render_pkg["delta_normal_norm"], render_pkg["alpha"])

        # gt depth
        if self.optimizer_config.get('lambda_depth', 0) > 0:
            loss += self.optimizer_config.lambda_depth * l1_loss(render_pkg["depth"][0], render_pkg["gt_depth"])

        # corrections
        correction_warpup_iter = self.optimizer_config.get('correction_warpup_iter', 0)
        if iteration < correction_warpup_iter:
            lambda_corrections = 100
        else:
            lambda_corrections = self.optimizer_config.get('lambda_corrections', 0)

        if lambda_corrections > 0:
            for key in render_pkg['corrections']:
                loss += lambda_corrections * thf.mse_loss(render_pkg['corrections'][key],
                                                          torch.zeros_like(render_pkg['corrections'][key]))
                
        if self.optimizer_config.get('lambda_corrections_xyz', 0) > 0:
            corr_xyz_mean = render_pkg['corrections']['_xyz'].norm(dim=-1).mean()
            # sanity check
            if corr_xyz_mean > 0.1:
                loss = corr_xyz_mean
                print(f'[WARNING] sanity check: corr_xyz_mean = {corr_xyz_mean}')
            else:
                loss += self.optimizer_config.lambda_corrections_xyz * corr_xyz_mean
                
        # scaling over base_scaling
        if self.optimizer_config.get('lambda_base_scaling', 0) > 0:
            base_scaling_thresh = self.optimizer_config.get('base_scaling_thresh', 10.0)
            threshed = gs_model.scaling_activation(gs_model.base_scaling) * base_scaling_thresh
            thresh_idxs = gs_model.get_scaling > threshed
            if thresh_idxs.sum() > 0:
                loss += self.optimizer_config.lambda_base_scaling * self.gs_model.get_scaling[thresh_idxs].mean()

        ret['loss'] = loss
        return ret

    def grad_loss_step(self, iteration, loss, render, **render_pkg):
        loss.backward()

        max_norm = torch.max(torch.tensor([p.grad.norm() if p.grad is not None else 0 for p in self.gs_model.parameters()]))
        if max_norm > 1.0:
            print(f'[DEGASOptimizer] max_norm = {max_norm}, clip to 0.1')
            nn.utils.clip_grad_norm_(self.gs_model.parameters(), max_norm=0.1, norm_type=2)
            
        self.adaptive_density_control(render_pkg, iteration)
        self.step(iteration)

        requires_grad(self.gs_model, flag=True)
        self.zero_grad(set_to_none=True)


