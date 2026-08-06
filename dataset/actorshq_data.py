import os
from copy import deepcopy
import torch
import numpy as np
from scene.dataset_readers import convert_to_scene_cameras
from model import libcore
import cv2
import csv
from .dataset_utils import (
    AvatarDataset
)

def read_ActorsHQ_cameras(source_path, scale):
    with open(os.path.join(source_path, scale, 'calibration.csv'), 'r') as fp:
        csv_reader = csv.DictReader(fp)
        line_count = 0
        cams = []
        for row in csv_reader:
            if line_count == 0:
                print(f'[ActorsHQ] Column names are {", ".join(row)}')
                line_count += 1

            rvec = np.array([float(row['rx']), float(row['ry']), float(row['rz'])])
            c = np.array([float(row['tx']), float(row['ty']), float(row['tz'])])
            
            cam = libcore.Camera()
            cam.R = cv2.Rodrigues(rvec)[0].T
            cam.c = c
            cam.w = int(row['w'])
            cam.h = int(row['h'])
            cam.fx = float(row['fx']) * cam.w
            cam.fy = float(row['fy']) * cam.h
            cam.cx = float(row['px']) * cam.w
            cam.cy = float(row['py']) * cam.h
            cams.append(cam)

            line_count += 1
        print(f'[ActorsHQ] Processed {line_count} lines.')
        libcore.saveCamerasToPly(os.path.join(source_path, 'cams.ply'), cams)
    return cams

def read_ActorsHQ_frameset(img_dir, frm_idx, cams, cam_select=None, mini_batch=0):
    color_frames, color_masks = libcore.DataVec(), libcore.DataVec()
    color_frames.cams = []
    color_frames.images_path = []
    color_masks.images_path = []
    for cam_id in range(0, len(cams)):
        cam_sn = 'Cam%03d' % (cam_id + 1)
        img_fpath = os.path.join(img_dir, 'rgbs/%s/%s_rgb%06d.jpg' % (cam_sn, cam_sn, frm_idx))
        msk_fpath = os.path.join(img_dir, 'masks/%s/%s_mask%06d.png' % (cam_sn, cam_sn, frm_idx))
        # if os.path.exists(img_fpath) and os.path.exists(msk_fpath):
        color_frames.cams.append(cams[cam_id])
        color_frames.images_path.append(img_fpath)
        color_masks.images_path.append(msk_fpath)
    color_masks.cams = color_frames.cams

    if cam_select is None:
        cam_idxs = np.arange(color_frames.size).tolist()
    else:
        cam_idxs = deepcopy(cam_select)

    if mini_batch > 0:
        np.random.shuffle(cam_idxs)
        cam_idxs = cam_idxs[:mini_batch]

    color_frames = color_frames.toSubSet(cam_idxs)
    color_masks = color_masks.toSubSet(cam_idxs)

    if color_frames.size >= 4:
        color_frames.load_images_parallel(max_workers=4)
        color_masks.load_images_parallel(max_workers=4)
    else:
        color_frames.load_images(silent=True)
        color_masks.load_images(silent=True)

    for i in range(color_frames.size):
        color = color_frames.frames[i]
        mask = color_masks.frames[i]
        img = np.concatenate([color, mask[:, :, None]], axis=-1)
        color_frames.frames[i] = img
    color_frames.image_formats = ['RGB' for i in range(color_frames.size)]

    return color_frames, cam_idxs

class ActorsHQDataset(AvatarDataset):
    def __init__(self, config, split='train', frm_list=None):
        self.config = config
        self.split = split
        self.scale = config.get('scale', '4x')

        if frm_list is None:
            self.frm_list = config.get(split, {}).get('frm_list', None)
            if isinstance(self.frm_list, str):
                self.frm_list = eval(self.frm_list)
            if self.frm_list is None:
                fns = [fn for fn in os.listdir(os.path.join(config.dat_dir, f'{config.scale}/rgbs/Cam001')) if fn.endswith('.jpg')]
                self.frm_list = [int(fn.split('rgb')[-1].split('.')[0]) for fn in fns]
        else:
            self.frm_list = frm_list

        self.cam_select = config.get(split, {}).get('cam_select', None)
        if isinstance(self.cam_select, str):
            self.cam_select = eval(self.cam_select)

        # actorshq's camera starts from 001
        if self.cam_select is not None:
            self.cam_select = (np.array(self.cam_select) - 1).tolist()

        self.mini_batch = config.get(split, {}).get('mini_batch', 0)

        self.num_frames = len(self.frm_list)
        print(f'[ActorsHQDataset] num_frames = {self.num_frames}')

        self.dat_dir = config.dat_dir
        self.frameset_type = config.get('frameset_type', 'actorshq')
        self.smplx_type = self.config.get('smplx_type', None)
        self.smplx_forward_transl = config.get('smplx_forward_transl', False)
        self.cameras_extent = config.get('cameras_extent', 2.0)

        self.load_camera_data()
        # self.load_body_model()

        self.load_all_smplx(config.get('all_smplx_fn', self.smplx_type))
        if config.get('with_face_dpe', None):
            self.load_face_dpe(config.with_face_dpe)
        else:
            self.exp_codes = None

        self.pca = None
        if split == 'train':
            if config.get('num_pca_comps', 0) > 0:
                self.construct_pca(config.num_pca_comps)

    ##################################################
    # load camera data
    def load_camera_data(self):
        self.cams = read_ActorsHQ_cameras(self.dat_dir, self.scale)

        if self.config.get('select_focal_lt', 0) > 0:
            t_focal = self.config.select_focal_lt
            self.cam_select = [i for i in range(len(self.cams)) if self.cams[i].fx < t_focal]

    # ##################################################
    # # load smpl data
    # def load_body_model(self):
    #     from easymocap.smplmodel import load_model
    #     from model.bone_deformer.smplx_utils import get_smplx_model_path
    #     self.body_model = load_model(model_path=get_smplx_model_path(), gender='male', model_type='smplx').cpu()

    ##################################################
    def __len__(self):
        return len(self.frm_list)

    def __getitem__(self, idx):
        if idx is None:
            idx = torch.randint(0, len(self.frm_list), (1,)).item()

        frm_idx = int(self.frm_list[idx])

        ##########            
        img_dir = os.path.join(self.dat_dir, self.scale)
        color_frames, cam_idxs = read_ActorsHQ_frameset(img_dir, frm_idx, self.cams, self.cam_select, self.mini_batch)
        scene_cameras = convert_to_scene_cameras(color_frames, self.config)
        
        batch = {
            'idx': idx,
            'frm_idx': frm_idx,
            'cam_idxs': cam_idxs,
            'scene_cameras': scene_cameras,
            'cameras_extent': self.cameras_extent,
        }

        # load smplx if needed
        smplx_params = self.load_smplx_params(idx)
        if smplx_params is not None:
            batch.update({
                'smplx_params': smplx_params,
            })

        # face dpe, not in use for ActorsHQ
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
