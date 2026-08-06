import os
import copy
import numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement
import torch
import torch.nn as nn
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.general_utils import inverse_sigmoid, repeat_on_dim, build_rotation
from utils.sh_utils import eval_sh, RGB2SH
from utils.graphics_utils import BasicPointCloud
from simple_knn._C import distCUDA2
from gaussian_renderer import render

def to_abs_path(fn, dir):
    if not os.path.isabs(fn):
        fn = os.path.join(dir, fn)
    return fn

def to_cache_path(dir):
    cache_dir = os.path.join(dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def find_checkpoint(model_path, cli_configs=[]):
    print('--------------------------------------------------')
    print(f'[model_path] {model_path}')
    eval_dirs = [dir for dir in os.listdir(os.path.join(model_path, 'point_cloud')) if dir.startswith('iteration_')]
    iters = [int(dir.split('iteration_')[1]) for dir in eval_dirs]
    last_i = np.argsort(iters)[-1]
    last_dir = eval_dirs[last_i]
    ckpt_fn = os.path.join(model_path, f'point_cloud/{last_dir}/checkpoint.pt').replace('\\', '/')
    print(f'Found checkpoint: {ckpt_fn}')

    config_fn = os.path.join(model_path, 'config.yaml').replace('\\', '/')
    configs = copy.deepcopy(cli_configs)
    if os.path.isfile(config_fn):
        configs.append(config_fn)
        print(f'Found config: {config_fn}')

    # pca
    pca_fn = os.path.join(model_path, 'pca.pt')
    if os.path.isfile(pca_fn):
        print(f'Found pca: {pca_fn}')
    else:
        pca_fn = None

    # refined smplx
    smplx_fn = os.path.abspath(os.path.join(ckpt_fn, '../smplx_refined.pt'))
    if os.path.isfile(smplx_fn):
        print(f'Found smplx: {smplx_fn}')
    else:
        smplx_fn = None
    
    ckpt_info = {
        'configs': configs,
        'ckpt_fn': ckpt_fn,
        'pca_fn': pca_fn,
        'smplx_fn': smplx_fn,
        'iteration': iters[last_i],
    }
    print('--------------------------------------------------')
    return ckpt_info

class GaussianBase(nn.Module):
    def __init__(self, sh_degree=0) -> None:
        super().__init__()
        self.active_sh_degree = sh_degree
        self.max_sh_degree = sh_degree
        self.setup_functions()

        self.instance_attributes = {}

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self.get_rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
    
    ##################################################
    @property
    def num_gauss(self):
        return self._xyz.shape[0]
    
    @property
    def get_xyz_cano(self):
        return self._xyz

    @property
    def get_rotation_cano(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_scaling_cano(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_opacity_cano(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_features_dc(self):
        return self._features_dc

    @property
    def get_features_rest(self):
        return self._features_rest

    ##################################################
    def init_gauss(self, xyz, features_dc, features_extra, opacities, scales, rots):
        self._xyz = torch.tensor(xyz, dtype=torch.float, device='cuda')
        self._features_dc = torch.tensor(features_dc, dtype=torch.float, device='cuda').transpose(1, 2).contiguous()
        self._features_rest = torch.tensor(features_extra, dtype=torch.float, device='cuda').transpose(1, 2).contiguous()
        self._opacity = torch.tensor(opacities, dtype=torch.float, device='cuda')
        self._scaling = torch.tensor(scales, dtype=torch.float, device='cuda')
        self._rotation = torch.tensor(rots, dtype=torch.float, device='cuda')

        self.max_radii2D = torch.zeros((self._opacity.shape[0]), device='cuda')
        self.active_sh_degree = self.max_sh_degree
        
    def create_from_pcd(self, pcd : BasicPointCloud):
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().to(self.device)
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().to(self.device))
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().to(self.device)
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print('Number of points at initialisation : ', fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().to(self.device)), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device='cuda')
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device='cuda'))
        self.init_gauss(fused_point_cloud, features[:,:,0:1], features[:,:,1:], 
                        opacities, scales, rots)

    def create_to_size(self, num_gauss):
        pcd = BasicPointCloud(np.random.rand(num_gauss, 3), np.zeros((num_gauss, 3)), np.zeros((num_gauss, 3)))
        self.create_from_pcd(pcd)

    ##################################################
    def pre_render(self, batch=None):
        pass

    def post_optim(self):
        pass

    def post_densify(self):
        pass

    # get bg color
    def get_color(self, key):
        if key == 'white':
            color = torch.tensor([1, 1, 1], dtype=torch.float32, device='cuda')
        elif key == 'black':
            color = torch.tensor([0, 0, 0], dtype=torch.float32, device='cuda')
        else:
            color = torch.rand((3,), dtype=torch.float32, device='cuda')
        return color

    # update gt_image to out
    def update_to_result(self, viewpoint_cam, bg_clr, out):
        if hasattr(viewpoint_cam, 'original_image'):
            if hasattr(viewpoint_cam, 'gt_alpha_mask'):
                gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()
                gt_image = viewpoint_cam.original_image.cuda()
                gt_image = gt_image * gt_alpha_mask + bg_clr[:, None, None] * (1 - gt_alpha_mask)

                out.update({
                    'gt_image': gt_image,
                    'gt_alpha_mask': gt_alpha_mask,
                })
            else:
                gt_image = viewpoint_cam.original_image.cuda()
                out.update({
                    'gt_image': gt_image,
                })

        if hasattr(viewpoint_cam, 'image_weights'):
            out['image_weights'] = viewpoint_cam.image_weights.cuda()

        if hasattr(viewpoint_cam, 'image_depths'):
            out['gt_depth'] = viewpoint_cam.image_depths.cuda()

    # render
    def render_to_camera(self, viewpoint_cam, pipe, background=None, 
                         scaling_modifer=1.0, render_validation=False):
        bg_clr = self.get_color(background)
        out = render(viewpoint_cam, self, pipe, bg_clr, scaling_modifer)

        self.update_to_result(viewpoint_cam, bg_clr, out)
        return out
    
    ##################################################
    def prune_points(self, valid_points_mask, optimizable_tensors):
        for key in self.instance_attributes:
            if key in optimizable_tensors:
                setattr(self, key, optimizable_tensors[key])
            else:
                setattr(self, key, getattr(self, key)[valid_points_mask])

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.post_densify()

    def densification_postfix(self, optimizable_tensors, densify_out):
        for key in self.instance_attributes:
            if key in optimizable_tensors:
                setattr(self, key, optimizable_tensors[key])
            else:
                setattr(self, key, torch.cat([getattr(self, key), densify_out[f'new{key}']], dim=0))

        self.xyz_gradient_accum = torch.zeros((self.num_gauss, 1), device='cuda')
        self.denom = torch.zeros((self.num_gauss, 1), device='cuda')
        self.max_radii2D = torch.zeros((self.num_gauss), device='cuda')
        self.post_densify()

    def prepare_densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.num_gauss
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device='cuda')
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)

        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling_cano, dim=1).values > self.percent_dense * scene_extent)
    
        if self.config.get('force_scaling_split', False):
            aspect_mask = (torch.max(self.get_scaling_cano, dim=-1).values / self.get_scaling_cano.mean(dim=-1)) > 2.0
            force_mask = torch.max(self.get_scaling_cano, dim=-1).values > self.percent_dense * scene_extent * 1
            force_mask = torch.logical_and(force_mask, aspect_mask)
            selected_pts_mask = torch.logical_or(selected_pts_mask, force_mask)

        stds = repeat_on_dim(self.get_scaling_cano[selected_pts_mask], N, dim=0)
        means = torch.zeros((stds.size(0), 3),device='cuda')
        samples = torch.normal(mean=means, std=stds)
        rots = repeat_on_dim(build_rotation(self._rotation[selected_pts_mask]), N, dim=0)
        
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + repeat_on_dim(self.get_xyz_cano[selected_pts_mask], N, dim=0)

        return selected_pts_mask, new_xyz
 
    def prepare_split_selected_to_new_xyz(self, selected_pts_mask, new_xyz, N):
        splitout = {}
        for key in self.instance_attributes:
            if key == '_xyz':
                splitout['new_xyz'] = new_xyz
            elif key == '_rotation':
                new_rotation = repeat_on_dim(self._rotation[selected_pts_mask], N, dim=0)
                splitout['new_rotation'] = new_rotation
            elif key == '_scaling':
                new_scaling = self.scaling_inverse_activation(
                    repeat_on_dim(self.get_scaling_cano[selected_pts_mask], N, dim=0) / (0.8*N))
                splitout['new_scaling'] = new_scaling
            else:
                splitout[f'new{key}'] = repeat_on_dim(getattr(self, key)[selected_pts_mask], N, dim=0)

        return splitout

    def prepare_densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling_cano, dim=1).values <= self.percent_dense*scene_extent)
        
        cloneout = {}
        for key in self.instance_attributes:
            cloneout[f'new{key}'] = getattr(self, key)[selected_pts_mask]

        return cloneout

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    ##################################################
    # save
    def construct_list_of_attributes(self, f_dc, f_rest, scale, rotation):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']

        for i in range(f_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        for i in range(f_rest.shape[1]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(scale.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotation.shape[1]):
            l.append('rot_{}'.format(i))

        return l
    
    # overwrite this function is needed
    # sh 0: f_rest_dims=0
    # sh 3: f_rest_dims=45
    def prepare_to_write(self, f_rest_dims=0):
        xyz = self.get_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)

        f_dc = self.get_features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self.get_features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        if f_rest_dims > f_rest.shape[-1]:
            zero_dims = f_rest_dims - f_rest.shape[-1]
            f_rest = np.concatenate([f_rest, np.zeros(xyz.shape[0], zero_dims)], axis=-1).astype(np.float32)
        
        opacities = self.inverse_opacity_activation(self.get_opacity.detach()).cpu().numpy()

        scale = self.scaling_inverse_activation(self.get_scaling.detach()).cpu().numpy()
        rotation = self.get_rotation.detach().cpu().numpy()

        return {
            'xyz': xyz,
            'normals': normals,
            'f_dc': f_dc,
            'f_rest': f_rest,
            'opacities': opacities,
            'scale': scale,
            'rotation': rotation,
        }
    
    def save_ply(self, path, f_rest_dims=0):
        print(f'[3DGS] save_ply to {path}')
        os.makedirs(Path(path).parent, exist_ok=True)

        contents = self.prepare_to_write(f_rest_dims=f_rest_dims)
        xyz = contents['xyz']
        normals = contents['normals']
        f_dc = contents['f_dc']
        f_rest = contents['f_rest']
        opacities = contents['opacities']
        scale = contents['scale']
        rotation = contents['rotation']

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(f_dc, f_rest, scale, rotation)]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    # load
    def load_ply(self, path):
        print(f'[3DGS] load_ply from {path}')
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]['x']),
                        np.asarray(plydata.elements[0]['y']),
                        np.asarray(plydata.elements[0]['z'])),  axis=1)
        opacities = np.asarray(plydata.elements[0]['opacity'])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]['f_dc_0'])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]['f_dc_1'])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]['f_dc_2'])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith('f_rest_')]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith('scale_')]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith('rot')]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        self.init_gauss(xyz, features_dc, features_extra, opacities, scales, rots)

    # update to checkpoint dict
    def update_to_checkpoint(self, checkpoint):
        checkpoint.update({
            'model': {
                'state_dict': self.state_dict(),
                'config': self.config,
            },
        })
