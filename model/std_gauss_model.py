import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from model import libcore
from utils.sh_utils import eval_sh, RGB2SH
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, repeat_on_dim
from .gauss_base import GaussianBase, to_abs_path, to_cache_path
from gaussian_renderer import render

# standard 3dgs
class StandardGaussModel(GaussianBase):
    def __init__(self, config,
                 device=torch.device('cuda'),
                 verbose=False):
        super().__init__()
        self.config = config
        self.device = device
        self.verbose = verbose

        # per instance attributes
        self.instance_attributes = {
            '_xyz': 3, 
            '_features_dc': 3, 
            '_features_rest': 0,
            '_scaling': 3, 
            '_rotation': 4, 
            '_opacity': 1,
        }

        for key in self.instance_attributes.keys():
            setattr(self, key, torch.Tensor(0))

        self.setup_config(config)

    ##################################################
    @property
    def num_gauss(self):
        return self._xyz.shape[0]

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_params(self, device='cpu'):
        return {
            '_xyz': self._xyz.detach().to(device),
            '_rotation': self._rotation.detach().to(device),
            '_scaling': self._scaling.detach().to(device),
            '_features_dc': self._features_dc.detach().to(device),
            '_features_rest': self._features_rest.detach().to(device),
            '_opacity': self._opacity.detach().to(device),
        }
    
    def set_params(self, params):
        if '_xyz' in params:
            self._xyz = params['_xyz'].to(self.device)
        if '_rotation' in params:
            self._rotation = params['_rotation'].to(self.device)
        if '_scaling' in params:
            self._scaling = params['_scaling'].to(self.device)
        if '_features_dc' in params:
            self._features_dc = params['_features_dc'].to(self.device)
        if '_features_rest' in params:
            self._features_rest = params['_features_rest'].to(self.device)
        if '_opacity' in params:
            self._opacity = params['_opacity'].to(self.device)
    
    def get_colors_precomp(self, viewpoint_camera=None):
        return self.color_activation(self._color)
    
    def get_colors_precomp(self, viewpoint_camera=None):
        shs_view = self.get_features.transpose(1, 2).view(-1, 3, (self.max_sh_degree+1)**2)
        if viewpoint_camera is not None:
            dir_pp = (self.get_xyz - viewpoint_camera.camera_center.repeat(self.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        else:
            dir_pp_normalized = torch.zeros_like(self._xyz)
        sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        return colors_precomp

    ##################################################
    def setup_config(self, config):
        self.config = config
        self.max_sh_degree = config.get('sh_degree', 0)
        self.active_sh_degree = self.max_sh_degree

    ##################################################
    # render
    def render_to_camera(self, viewpoint_cam, pipe, background=None, 
                         scaling_modifer=1.0, render_validation=False):
        bg_clr = self.get_color(background)
        out = render(viewpoint_cam, self, pipe, bg_clr, scaling_modifer)

        # extras
        if self.config.get('render_alpha', False):
            bg_extra = self.get_color('black')
            override_color = torch.ones_like(self._xyz)
            alpha = render(viewpoint_cam, self, pipe, bg_extra, scaling_modifer, 
                           override_color=override_color)['render']
            out['alpha'] = alpha

        self.update_to_result(viewpoint_cam, bg_clr, out)
        return out
    
    