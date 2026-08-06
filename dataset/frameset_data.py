import os
import random
import numpy as np
import torch
from scene.dataset_readers import convert_to_scene_cameras
from utils.graphics_utils import BasicPointCloud
from model import libcore
from model.bone_deformer import smplx_utils
from .dataset_utils import (
    read_frame_list, sample_pcd, touch_split_txt, 
    set_weights_to_scene_cams, match_smplx_file,
    reset_smplx_facial,
    AvatarDataset
)

##################################################
class FramesetDataset(AvatarDataset):
    def __init__(self, config, split='train', frm_list=None):
        self.config = config
        self.split = split

        if frm_list is None:
            self.frm_list = config.get(split, {}).get('frm_list', None)
            if isinstance(self.frm_list, str):
                self.frm_list = eval(self.frm_list)
            elif self.frm_list is None:
                split_fn = config.get(f'split_{split}_fn', f'{split}.txt')
                if not os.path.exists(os.path.join(config.dat_dir, split_fn)):
                    if split == 'train':
                        touch_split_txt(config.dat_dir, split_fn, step=1)
                    elif split == 'val':
                        touch_split_txt(config.dat_dir, split_fn, total=1)
                    else:
                        touch_split_txt(config.dat_dir, split_fn, total=100)

                self.frm_list = read_frame_list(os.path.join(config.dat_dir, split_fn), to_int=True)
        else:
            self.frm_list = frm_list

        self.num_frames = len(self.frm_list)
        print(f'[FramesetDataset][{split}] num_frames = {self.num_frames}')

        self.cam_select = config.get(split, {}).get('cam_select', None)
        if isinstance(self.cam_select, str):
            self.cam_select = eval(self.cam_select)

        self.dat_dir = config.dat_dir
        self.verbose_timer = config.get('verbose_timer', False)
        self.frameset_type = config.get('frameset_type', 'color_frames')
        self.smplx_type = self.config.get('smplx_type', None)
        self.smplx_forward_transl = config.get('smplx_forward_transl', False)
        self.cameras_extent = config.get('cameras_extent', 2.0)
        self.mini_batch = config.get('mini_batch', 0)

        self.load_all_smplx(config.get('all_smplx_fn', self.smplx_type))
        if config.get('with_face_code', None):
            if 'dpe' in config.with_face_code.lower():
                self.load_face_dpe(config.with_face_code)
            elif 'DAD' in config.with_face_code.upper():
                self.load_face_DAD(config.with_face_code)
            elif 'deca' in config.with_face_code.lower():
                self.load_face_deca(config.with_face_code)
            else:
                raise NotImplementedError
        else:
            self.exp_codes = None

        self.pca = None
        if split == 'train' and config.get('constrct_pca', True):
            self.construct_pca()

    ##################################################
    def __len__(self):
        return len(self.frm_list)

    def __getitem__(self, idx=None, cam_idxs=None):
        if idx is None:
            idx = torch.randint(0, len(self.frm_list), (1,)).item()

        frm_idx = self.frm_list[idx]
        batch = {
            'idx': idx,
            'frm_idx': frm_idx,
        }

        ##########
        if self.verbose_timer:
            libcore.startCpuTimer('[FramesetDataset] readPromethInfo')

        img_dir = os.path.join(self.dat_dir, f'{frm_idx:06d}/{self.frameset_type}')
        color_frames = libcore.loadDataVecFromFolder(img_dir, with_frames=False)

        # focal select for ActorsHQ
        if self.config.get('select_focal_lt', 0) > 0:
            t_focal = self.config.select_focal_lt
            cam_select = [i for i in range(len(color_frames.cams)) if color_frames.cams[i].fx < t_focal]
            color_frames = color_frames.toSubSet(cam_select)

        # manual select or mini batch
        if cam_idxs is None:
            if self.cam_select is not None:
                cam_idxs = self.cam_select
            else:
                cam_idxs = np.arange(color_frames.size).tolist()

            if self.mini_batch > 0 and self.mini_batch < len(cam_idxs):
                cam_idxs = np.array(cam_idxs)
                np.random.shuffle(cam_idxs)
                cam_idxs = cam_idxs[:self.mini_batch].tolist()
        
        color_frames = color_frames.toSubSet(cam_idxs)

        # load images
        color_frames.load_images_parallel(max_workers=4)
        scene_cameras = convert_to_scene_cameras(color_frames, self.config)

        # rgb weights
        if self.config.get('image_weights', None):
            weight_dir = os.path.join(self.dat_dir, f'{frm_idx:06d}/{self.config.image_weights}')
            weights_vec = libcore.loadDataVecFromFolder(weight_dir)
            weights_vec = weights_vec.toSubSet(cam_idxs)
            set_weights_to_scene_cams(weights_vec, scene_cameras, attr_key='image_weights')

        # depths
        if self.config.get('with_depths', None):
            depth_dir = os.path.join(self.dat_dir, f'{frm_idx:06d}/{self.config.with_depths}')
            depths_vec = libcore.loadDataVecFromFolder(depth_dir, with_frames=False)
            depths_vec = depths_vec.toSubSet(cam_idxs)
            depths_vec.load_images_parallel(max_workers=4)
            set_weights_to_scene_cams(depths_vec, scene_cameras, attr_key='image_depths')

        
        batch.update({
            'cam_idxs': cam_idxs,
            'color_frames': color_frames,
            'scene_cameras': scene_cameras,
            'cameras_extent': self.cameras_extent,
        })

        if self.verbose_timer:
            libcore.stopCpuTimer('[FramesetDataset] cameraList_from_camInfos')

        # load smplx if needed
        smplx_params = self.load_smplx_params(idx)
        if smplx_params is not None:
            batch.update({
                'smplx_params': smplx_params,
            })

        # face dpe
        if self.exp_codes is not None and self.exp_codes[idx] is not None:
            # random combination of exp codes
            _codes = self.exp_codes[idx]
            w = torch.rand((_codes.shape[0],))
            w = w / w.sum()
            code = torch.einsum('i,ij->j', w, _codes)[None, ...]
            batch.update({
                'exp_code': code,
            })

        return batch

    def load_smplx_params(self, idx, with_mesh=False):
        if self.smplx_params is not None:
            smplx_params = {}
            for key in self.smplx_params:
                if not isinstance(self.smplx_params[key], torch.Tensor):
                    smplx_params[key] = self.smplx_params[key]
                else:
                    if key == 'betas':
                        smplx_params[key] = self.smplx_params[key].detach().clone()
                    elif key == 'v_template':
                        if len(self.smplx_params[key]) == 2 or self.smplx_params[key].shape[0] == 1:
                            smplx_params[key] = self.smplx_params[key].detach().clone().squeeze(0)
                        else:
                            smplx_params[key] = self.smplx_params[key][idx].detach().clone()
                    else:
                        smplx_params[key] = self.smplx_params[key][idx].detach().clone()
                        if len(smplx_params[key].shape) == 1:
                            smplx_params[key] = smplx_params[key].unsqueeze(0)
        else:
            smplx_type = self.smplx_type
            if smplx_type is None:
                smplx_params = None
            else:
                frm_idx = self.frm_list[idx]

                # per frame smplx fn' naming style shoud be:
                # - xxxx/smplx-000000.pt
                if smplx_type == 'nvdiffsmplx_out':
                    smplx_fn = os.path.join(self.config.dat_dir, f'nvdiffsmplx_out/{frm_idx:06d}', 'smplx.pt')
                else:
                    smplx_fn = match_smplx_file(self.config.dat_dir, smplx_type, frm_idx)

                smplx_params = smplx_utils.load_and_detach(smplx_fn) if os.path.exists(smplx_fn) else None
        
                if self.config.get('smplx_nofacial', None):
                    reset_smplx_facial(smplx_params, self.config.smplx_nofacial)
        
        return smplx_params
    
    def sample_pcd(self):
        batch = self.__getitem__(0)
        if 'smplx_params' in batch:
            smplx_model = smplx_utils.create_smplx_model(**batch['smplx_params'])
            with torch.no_grad():
                out = smplx_model(**batch['smplx_params'])

            pcd = BasicPointCloud(out['vertices'][0], 
                                  torch.full_like(out['vertices'][0], 0.1),
                                  torch.zeros_like(out['vertices'][0]))
        
            cam_pos = np.stack([cam.c for cam in batch['color_frames'].cams])
            bbox_min = cam_pos.min(axis=0)
            bbox_max = cam_pos.max(axis=0)
        else:
            pcd, bbox_min, bbox_max = sample_pcd(batch['color_frames'].cams, 10000)

        return pcd, bbox_min, bbox_max
