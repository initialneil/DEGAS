
import numpy as np
import torch
from .gauss_base import GaussianBase
from model.bone_deformer import smplx_utils

class AvatarBase(GaussianBase):
    def __init__(self):
        super().__init__()
        self.name = 'AvatarBase'

    ##################################################
    def set_pose(self, params):
        posed_params = {}
        for key in params.keys():
            if isinstance(params[key], torch.Tensor):
                posed_params[key] = params[key].to(self.device)
            else:
                posed_params[key] = params[key]

        # ignore keys
        ignore_keys = ['betas']
        if 'exp' in self.config.get('smplx_nofacial', ''):
            ignore_keys.append('expression')
        if 'jaw' in self.config.get('smplx_nofacial', ''):
            ignore_keys.append('jaw_pose')
        
        for key in ignore_keys:
            if key in posed_params:
                posed_params.pop(key)

        self.smplx_deformer.update(**posed_params)

        # conditions: full_pose, posed smplx vertices
        self.full_pose = smplx_utils.get_smplx_full_pose(posed_params, self.smplx_deformer.smplx_model)
        self.full_pose = self.full_pose.detach()

    ##################################################
    # pose pca
    def transform_pca_pose_v1(self, posed_params):
        full_pose = smplx_utils.get_smplx_full_pose(posed_params)
        global_orient, full_pose = full_pose[:, :3], full_pose[:, 3:]
        pose_pca = self.pca['pose_pca']
        lowdim_pose_conds = pose_pca.transform(full_pose.detach().cpu().numpy())
        std = np.sqrt(pose_pca.explained_variance_)

        num_comps = self.config.get('num_pca_comps', 20)
        sigma_pca = self.config.get('sigma_pca_pose', 10.0)
        lowdim_pose_conds[..., num_comps:] = 0

        lowdim_pose_conds = np.maximum(lowdim_pose_conds, -sigma_pca * std)
        lowdim_pose_conds = np.minimum(lowdim_pose_conds, sigma_pca * std)
        new_pose_conds = pose_pca.inverse_transform(lowdim_pose_conds)
        new_pose_conds = torch.tensor(new_pose_conds).to(full_pose)

        new_pose_conds = torch.concat([global_orient, new_pose_conds], dim=-1)
        posed_params = smplx_utils.set_full_pose_to_params(new_pose_conds, posed_params)
        return posed_params

    def transform_pca_pose_v2(self, posed_params):
        full_pose = smplx_utils.get_smplx_full_pose(posed_params)
        global_orient, full_pose = full_pose[:, :3], full_pose[:, 3:]
        pose_pca = self.pca['pose_pca']
        lowdim_pose_conds = pose_pca.transform(full_pose.detach().cpu().numpy())
        std = np.sqrt(pose_pca.explained_variance_)

        num_comps = self.config.get('num_pca_comps', 20)
        
        scale = abs(lowdim_pose_conds[0]) / std
        mag = np.exp(-(scale - 1.0) / 1.0)
        fix_conds = mag * lowdim_pose_conds
        lowdim_pose_conds = fix_conds

        # import matplotlib.pyplot as plt
        # plt.plot(lowdim_pose_conds[0])
        # plt.plot(fix_conds[0])
        # plt.show()

        new_pose_conds = pose_pca.inverse_transform(lowdim_pose_conds)
        new_pose_conds = torch.tensor(new_pose_conds).to(full_pose)

        new_pose_conds = torch.concat([global_orient, new_pose_conds], dim=-1)
        posed_params = smplx_utils.set_full_pose_to_params(new_pose_conds, posed_params)
        return posed_params

    def transform_pca_pose_v3(self, posed_params):
        full_pose = smplx_utils.get_smplx_full_pose(posed_params)
        global_orient, full_pose = full_pose[:, :3], full_pose[:, 3:]
        pose_pca = self.pca['pose_pca']
        lowdim_pose_conds = pose_pca.transform(full_pose.detach().cpu().numpy())
        std = np.sqrt(pose_pca.explained_variance_)

        num_comps = self.config.get('num_pca_comps', 20)

        scale = abs(lowdim_pose_conds[0]) / std
        mag = np.exp(-(scale - 1.0) / 1.0)
        fix_conds = mag * lowdim_pose_conds
        lowdim_pose_conds = fix_conds

        _new_pose = pose_pca.inverse_transform(lowdim_pose_conds)
        _new_pose = torch.tensor(_new_pose).to(full_pose)

        # smoothing
        if hasattr(self, 'last_new_pose'):
            _new_pose = _new_pose * 0.1 + self.last_new_pose * 0.9
        self.last_new_pose = _new_pose

        # fix shoulder and legs
        new_pose = full_pose.detach().clone()
        # 0:6: left_hip, right_hip
        # 9:15: left_knee, right_knee
        # 18:24: left_ankle, right_ankle
        # 27:33: left_foot, right_foot
        # 36:42: left_collar, right_collar
        # 45:51: left_shoulder, right_shoulder
        jnt_idxs = np.concatenate([#np.arange(0, 6), 
                                   #np.arange(9, 15), 
                                   np.arange(18, 24),
                                   np.arange(27, 33), 
                                   np.arange(36, 42), 
                                   #np.arange(45, 51)
                                  ])
        new_pose[..., jnt_idxs] = _new_pose[..., jnt_idxs]

        new_pose_conds = torch.concat([global_orient, new_pose], dim=-1)
        posed_params = smplx_utils.set_full_pose_to_params(new_pose_conds, posed_params)
        return posed_params

    def transform_pca_pose(self, posed_params):
        return self.transform_pca_pose_v3(posed_params)

    # exp pca
    def transform_pca_exp(self, face_embs):
        sigma_pca = self.config.get('sigma_pca_exp', 4.0)
        exp_pca = self.pca['exp_pca']
        lowdim_exp_conds = exp_pca.transform(face_embs)
        
        std = np.sqrt(exp_pca.explained_variance_)
        lowdim_exp_conds = np.maximum(lowdim_exp_conds, -sigma_pca * std)
        lowdim_exp_conds = np.minimum(lowdim_exp_conds, sigma_pca * std)
        new_exp_conds = exp_pca.inverse_transform(lowdim_exp_conds)
        face_embs = torch.tensor(new_exp_conds).to(face_embs)
        return face_embs
    
    # tpose
    def transform_by_tpose(self, posed_params, tpose_params):
        full_pose = smplx_utils.get_smplx_full_pose(posed_params)
        global_orient, full_pose = full_pose[:, :3], full_pose[:, 3:]
        data_tpose = smplx_utils.get_smplx_full_pose(tpose_params)[..., 3:]
        full_pose_shift = full_pose - data_tpose

        model_tpose = smplx_utils.get_smplx_full_pose(self.tpose_params)[..., 3:]
        _new_pose = model_tpose + full_pose_shift

        new_pose = full_pose.detach().clone()
        # 0:6: left_hip, right_hip
        # 9:15: left_knee, right_knee
        # 18:24: left_ankle, right_ankle
        # 27:33: left_foot, right_foot
        # 36:42: left_collar, right_collar
        # 45:51: left_shoulder, right_shoulder
        jnt_idxs = np.concatenate([#np.arange(0, 6), 
                                   np.arange(9, 15), 
                                   np.arange(18, 24),
                                   np.arange(27, 33), 
                                   #np.arange(36, 42), 
                                   #np.arange(45, 51)
                                  ])
        # new_pose[..., jnt_idxs] = _new_pose[..., jnt_idxs]
        new_pose[..., jnt_idxs] = model_tpose[..., jnt_idxs]
        # new_pose = _new_pose

        new_pose_conds = torch.concat([global_orient, new_pose], dim=-1)
        posed_params = smplx_utils.set_full_pose_to_params(new_pose_conds, posed_params)
        return posed_params

