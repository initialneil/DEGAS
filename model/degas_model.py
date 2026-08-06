import os
import torch
import torch.nn.functional as thf
import numpy as np
import pytorch3d.transforms as py3d_trans
import pytorch3d.structures.meshes as py3d_meshes
from scene.dataset_readers import convert_from_scene_camera
from model import libcore
from model.bone_deformer import smplx_deformer, smplx_utils
from utils import general_utils, graphics_utils
from utils.data_utils import construct_quaternion_triangle
from utils.general_utils import inverse_sigmoid, repeat_on_dim, build_rotation
from utils.map import face_attr_to_vertex
from simple_knn._C import distCUDA2
import nvdiffrast.torch as dr
from .avatar_base import AvatarBase
from .degas_vae_driver import PoseVAE, PoseMapper
from gaussian_renderer import render

def _render_normal(viewpoint_cam, depth, bg_color, alpha):
    # depth: (H, W), bg_color: (3), alpha: (H, W)
    # normal_ref: (3, H, W)
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf()

    normal_ref = graphics_utils.normal_from_depth_image(depth, intrinsic_matrix.to(depth.device), extrinsic_matrix.to(depth.device))
    background = bg_color[None,None,...]
    normal_ref = normal_ref*alpha[...,None] + background*(1. - alpha[...,None])
    normal_ref = normal_ref.permute(2,0,1)

    return normal_ref

def _normalize_normal_inplace(normal, alpha):
    # normal: (3, H, W), alpha: (H, W)
    fg_mask = (alpha[None,...]>0.).repeat(3, 1, 1)
    normal = torch.where(fg_mask, torch.nn.functional.normalize(normal, p=2, dim=0), normal)

##################################################
# DEGAS
class DEGASModel(AvatarBase):
    def __init__(self, config, render_config,
                 device=torch.device('cuda'),
                 verbose=False):
        super().__init__()
        self.config = config
        self.device = device
        self.verbose = verbose
        self.name = 'DEGASModel'

        # # per instance attributes
        # self.instance_attributes = {
        #     '_xyz': 3, 
        #     '_features_dc': 3, 
        #     '_features_rest': 0,
        #     '_scaling': 3, 
        #     '_rotation': 4, 
        #     '_opacity': 1,
        #     '_normal1': 3, 
        #     '_normal2': 3, 
        #     '_specular': 3, 
        #     '_roughness': 1,
        # }

        # for key in self.instance_attributes.keys():
        #     setattr(self, key, torch.Tensor(0))

        self.setup_config(config)
        self.render_config = render_config

    ##################################################
    @property
    def num_gauss(self):
        return self.base_xyz.shape[0]
    
    @property
    def get_xyz(self):
        return self.xyz_posed

    @property
    def get_rotation(self):
        return self.rotation_activation(self.rotation_posed)

    @property
    def get_scaling_cano(self):
        return self.scaling_activation(self.base_scaling)

    @property
    def get_scaling(self):
        return self.scaling_activation(self.scaling_corrected)

    @property
    def get_features_dc(self):
        return self.features_dc_corrected
    
    @property
    def get_features_rest(self):
        return torch.zeros(self.base_xyz.shape[0], 0, 3).to(self.base_xyz)

    @property
    def get_features(self):
        return self.features_dc_corrected
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self.opacity_corrected)

    ####################
    @property
    def get_minimum_axis(self):
        # # get_rotation in posed space
        # # return normal (minimum axis) in posed space
        # return general_utils.get_minimum_axis(self.get_scaling_cano, self.get_rotation)

        # choose z-axis
        R = build_rotation(self.get_rotation)
        # https://github.com/Asparagus15/GaussianShader/issues/19
        # original code seems wrong???
        z_axis = R[:, 2, :] # normalized by defaut
        # z_axis = R[:, :, 2] # normalized by defaut
        return z_axis

    def get_normal(self):
        normal_axis = self.get_minimum_axis
        return normal_axis

    ##################################################
    def setup_config(self, config):
        self.config = config
        self.max_sh_degree = config.get('sh_degree', 0)
        self.active_sh_degree = self.max_sh_degree

        # correction from pose driver
        self.corrections = {}

        self.offset_scale = config.get('offset_scale', 0.05)

    def setup_canonical(self, cano_params, cano_verts, cano_norms, cano_faces):
        self.cano_verts = cano_verts.detach().clone()
        self.cano_norms = cano_norms.detach().clone()
        self.cano_faces = cano_faces.detach().clone()

        self.cano_params = {}
        for key in cano_params:
            if isinstance(cano_params[key], torch.Tensor):
                self.cano_params[key] = cano_params[key].detach().clone()
            else:
                self.cano_params[key] = cano_params[key]

        # smplx deformer
        self.smplx_deformer = smplx_deformer.SMPLXDeformer(**self.cano_params).to(self.device)

        # uv condition mask
        bbox_min = cano_verts.min(0)[0]
        bbox_max = cano_verts.max(0)[0]

        if self.config.get('with_vae_driver', False):
            print('[DEGASModel] pose driver = PoseVAE')
            self._pose_driver = PoseVAE(self.config.pose_driver, bbox_min, bbox_max, self.cano_params).to(self.device)

    def create_from_canonical(self, cano_params, cano_mesh, sample_fidxs=None, sample_bary=None):
        self.canonical = {
            'cano_params': cano_params,
            'cano_mesh': cano_mesh,
        }

        if isinstance(cano_mesh, py3d_meshes.Meshes):
            cano_verts = cano_mesh.verts_packed().float().to(self.device)
            cano_norms = cano_mesh.verts_normals_packed().float().to(self.device)
            cano_faces = cano_mesh.faces_packed().long().to(self.device)
        else:
            cano_verts = cano_mesh['mesh_verts'].float().to(self.device)
            cano_norms = cano_mesh['mesh_norms'].float().to(self.device)
            cano_faces = cano_mesh['mesh_faces'].long().to(self.device)

        self.setup_canonical(cano_params, cano_verts, cano_norms, cano_faces)
        
        # render base info
        smplx_model = self.smplx_deformer.smplx_model
        mesh = smplx_utils.convert_smplx_to_meshcpu(smplx_model, self.smplx_deformer.smplx_verts_c[0])
        self.base_mapper = PoseMapper(self.config.base_mapper, mesh)

        rlt = self.base_mapper.rasterize_base_info()
        mask = (rlt['mask'].squeeze() != 0)
        self.uv_xyz = rlt['vertices'].squeeze(0)
        self.uv_mask = mask
        self.base_xyz = self.uv_xyz[self.uv_mask]

        # align rotation with mesh
        face_quats = construct_quaternion_triangle(cano_verts, cano_norms, cano_faces)
        # face_R = build_rotation(face_quats)
        # libcore.savePointsToPly('e:/dummy/verts.ply', cano_verts[cano_faces].mean(dim=1), face_R[:, :, 2])

        vert_quats = face_attr_to_vertex(cano_verts, cano_faces, face_quats)
        rast_out = rlt['rast_out']
        self.uv_rotation, _ = dr.interpolate(vert_quats, rast_out, cano_faces.to(torch.int32))
        self.uv_rotation = thf.normalize(self.uv_rotation[0], dim=-1)

        # scaling
        dist2 = torch.clamp_min(distCUDA2(self.base_xyz), 0.0000001)
        scales = torch.sqrt(dist2)[...,None].repeat(1, 3)
        scales[:, 2] = 1e-4
        self.uv_scaling = torch.zeros_like(self.uv_xyz)
        self.uv_scaling[self.uv_mask] = self.scaling_inverse_activation(scales)
        self.base_scaling = self.uv_scaling[self.uv_mask]
    
    @staticmethod
    def create_from_checkpoint(ckpt, config, render_config):
        cano_params = ckpt['cano']['cano_params']
        cano_mesh = ckpt['cano']['cano_mesh']

        gs_model = DEGASModel(config, render_config)
        gs_model.create_from_canonical(cano_params, cano_mesh)

        rlt = gs_model.load_state_dict(ckpt['model']['state_dict'], strict=False)
        for key in rlt.unexpected_keys:
            print(f'[DEGASModel] set unexpected key from state_dict: {key}')
            setattr(gs_model, key, ckpt['model']['state_dict'][key])
        return gs_model

    ##################################################
    def update_to_pose(self, posed_params=None, face_embs=None):
        if posed_params is not None:
            self.set_pose(posed_params)
        self.face_embs = face_embs

        # predict corrections
        self.corrections = self._pose_driver.forward(self.full_pose, face_embs)
        self.corrections['_xyz'] = self.corrections['_xyz'] * self.offset_scale
        self.corrections['_rotation'] = self.corrections['_rotation'] * self.offset_scale

        # lbs weights from uncorrected _xyz
        # warp corrected _xyz
        uv_xyz = self.uv_xyz + self.corrections['_xyz']
        xyz_corrected = uv_xyz[self.uv_mask]
        self.xyz_posed, tfs, _ = self.smplx_deformer.forward(xyz_corrected, knn_x=self.base_xyz.detach(), with_tfs=True)
        self.xyz_posed = self.xyz_posed.squeeze(0)

        # rotation is saved in transpose
        # (R * R0)^-1 = R0^-1 * R^-1
        pose_rot = py3d_trans.matrix_to_quaternion(tfs[0, :, :3, :3])
        uv_rot = self.uv_rotation + self.corrections['_rotation']
        rot_corrected = uv_rot[self.uv_mask]
        # self.rotation_posed = py3d_trans.quaternion_multiply(rot_corrected, py3d_trans.quaternion_invert(pose_rot))
        self.rotation_posed = py3d_trans.quaternion_multiply(rot_corrected, py3d_trans.quaternion_invert(pose_rot))

        uv_scaling = self.uv_scaling + self.corrections['_scaling']
        self.scaling_corrected = uv_scaling[self.uv_mask]
        
        uv_features_dc = self.corrections['_features_dc']
        features_dc = uv_features_dc[self.uv_mask]
        self.features_dc_corrected = features_dc[:, None, :]

        uv_opacity = self.corrections['_opacity']
        self.opacity_corrected = uv_opacity[self.uv_mask]

        uv_normal = self.corrections['_normal']
        normal_corrected = uv_normal[self.uv_mask]
        self.delta_normal_posed = py3d_trans.quaternion_apply(pose_rot, normal_corrected)

        uv_specular = self.corrections['_specular']
        self.specular_corrected = uv_specular[self.uv_mask]

        uv_roughness = self.corrections['_roughness']
        self.roughness_corrected = uv_roughness[self.uv_mask]

    def update_to_cano_mesh(self):
        self.update_to_pose(self.cano_params)

    ##################################################    
    # pre-render
    def pre_render(self, batch):
        # pose
        if self.render_config['mesh_from'] == 'batch':
            posed_params = batch['smplx_params']
        else:
            smplx_optim = self.render_config['smplx_optim']
            frm_idx = batch['frm_idx']
            if frm_idx in smplx_optim.frm_list:
                idx = np.where(np.array(smplx_optim.frm_list) == frm_idx)[0].item()
                posed_params = smplx_optim.get_smplx_params(idx)
            else:
                posed_params = batch['smplx_params']

        # fix flat hand
        smplx_model = self._pose_driver.smplx_model
        posed_params = smplx_utils.fix_hand_mean(smplx_model, posed_params)

        # facial expression code
        if 'exp_code' in batch:
            face_embs = batch['exp_code']
        else:
            face_embs = None

        # pca
        if batch.get('enable_pca', False):
            posed_params = self.transform_pca_pose(posed_params)
            # face_embs = self.transform_pca_exp(face_embs)

        # tpose
        if batch.get('enable_tpose', False):
            posed_params = self.transform_by_tpose(posed_params, batch['tpose_params'])

        self.update_to_pose(posed_params, face_embs)

    def post_optim(self):
        pass

    def post_densify(self):
        self.update_to_pose()
    
    def get_view_depth(self, viewpoint_cam):
        p_hom = torch.cat([self.get_xyz, torch.ones_like(self.get_xyz[...,:1])], -1).unsqueeze(-1)
        p_view = torch.matmul(viewpoint_cam.world_view_transform.transpose(0,1), p_hom)
        p_view = p_view[...,:3,:]
        depth = p_view.squeeze()[...,2:3]
        depth = depth.repeat(1,3)
        return depth

    # update gt_image to out
    def update_to_result(self, viewpoint_cam, bg_clr, out):
        super().update_to_result(viewpoint_cam, bg_clr, out)
    
    # render
    def render_to_camera(self, viewpoint_cam, pipe, background=None, 
                         scaling_modifer=1.0, render_validation=False):
        bg_clr = self.get_color(background)

        ##################################################
        out = render(viewpoint_cam, self, pipe, bg_clr, scaling_modifer)
        out['corrections'] = self.corrections
        self.update_to_result(viewpoint_cam, bg_clr, out)
        return out

    ##################################################
    # overwrite this function is needed
    # sh 0: f_rest_dims=0
    # sh 3: f_rest_dims=45
    def prepare_to_write(self, f_rest_dims=0):
        out = super().prepare_to_write(f_rest_dims=f_rest_dims)
        out.update({
            'normals': self.get_normal().detach().cpu().numpy(),
        })
        return out
    
    # update to checkpoint dict
    def update_to_checkpoint(self, checkpoint):
        super().update_to_checkpoint(checkpoint)

        checkpoint.update({
            'cano': self.canonical,
        })

