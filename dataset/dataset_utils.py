import os
import numpy as np
import cv2
import torch
from utils.graphics_utils import BasicPointCloud
from utils.sh_utils import SH2RGB
import pytorch3d.structures.meshes as py3d_meshes
from sklearn.decomposition import PCA
from tqdm import tqdm
from model import libcore
from model.bone_deformer import smplx_utils
import json
import fs

def read_frame_list(fn, to_int=True):
    if fn is None or not os.path.exists(fn):
        return None 
    
    frm_list = []
    with open(fn) as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            if len(line) > 0:
                frm_list.append(line)
    
    if to_int:
        frm_list = [int(i) for i in frm_list]
    return frm_list

def sample_pcd(cams, num_pts):
    cam_pos = np.stack([cam.c for cam in cams])
    bbox_min = cam_pos.min(axis=0)
    bbox_max = cam_pos.max(axis=0)

    xyz = np.random.random((num_pts, 3)) * (bbox_max - bbox_min) + bbox_min
    shs = np.random.random((num_pts, 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
    return pcd, bbox_min, bbox_max

def load_frameset(frm_dir, thresh_focal=None, resolution=None):
    color_frames = libcore.loadDataVecFromFolder(frm_dir, with_frames=False)
    if thresh_focal is not None:
        cam_select = [i for i in range(len(color_frames.cams)) if color_frames.cams[i].fx < thresh_focal]
        color_frames = color_frames.toSubSet(cam_select)
    color_frames.load_images_parallel(4)

    if resolution is not None:
        for i in range(color_frames.size):
            cam = color_frames.cams[i]
            img = color_frames.frames[i]
            if 1:
                cam.scaleIntrinsics(cam.w // resolution, cam.h // resolution)
                img = cv2.resize(img, (cam.w, cam.h), interpolation=cv2.INTER_CUBIC)
            else:
                for j in range(int(np.log(resolution) // np.log(2))):
                    img = cv2.pyrDown(img)
                cam.scaleIntrinsics(img.shape[1], img.shape[0])

            color_frames.cams[i] = cam
            color_frames.frames[i] = img

    return color_frames

##################################################
def reset_smplx_facial(smplx_params, smplx_nofacial='exp'):
    if smplx_params is None:
        return
    
    if 'jaw' in smplx_nofacial:
        print('[nofacial] reset jaw')
        if 'jaw_pose' in smplx_params:
            smplx_params['jaw_pose'] = torch.zeros_like(smplx_params['jaw_pose'])
    if 'exp' in smplx_nofacial:
        print('[nofacial] reset exp')
        if 'expression' in smplx_params:
            smplx_params['expression'] = torch.zeros_like(smplx_params['expression'])

def touch_split_txt(dat_dir, split_fn, step=0, total=0):
    if step <= 0 and total <= 0:
        step = 1

    frm_list = [i for i in os.listdir(dat_dir) if len(i) == 6 and i.isdigit()]
    if step <= 0:
        step = max(int(len(frm_list) // total), 1)
        
    with open(os.path.join(dat_dir, split_fn), 'w') as fp:
        for frm_idx in frm_list[::step]:
            fp.write(f'{frm_idx}\n')
    
def load_smplx_pt(fn, frm_list):
    smplx_params = torch.load(fn, map_location='cpu')

    idxs = [int(idx) for idx in frm_list]
    for k in smplx_params:
        if isinstance(smplx_params[k], torch.Tensor):
            smplx_params[k] = smplx_params[k].detach().clone().cpu()
            if smplx_params[k].shape[0] > 1:
                smplx_params[k] = smplx_params[k][idxs]
    return smplx_params
    
# added for Animatable Gaussians' smplx on ActorsHQ
# https://github.com/lizhe00/AnimatableGaussians?tab=readme-ov-file#avatarrex-actorshq-or-thuman40-dataset
def load_smplx_npz(fn, frm_list):
    smplx_params = dict(np.load(fn))

    if 'gender' not in smplx_params:
        smplx_params['gender'] = 'neutral'
    if 'num_betas' not in smplx_params and 'betas' in smplx_params:
        smplx_params['num_betas'] = smplx_params['betas'].shape[-1]
    if 'use_pca' not in smplx_params:
        smplx_params['use_pca'] = False

    # # dirty fix: jaw_pose seems weird
    # smplx_params.pop('jaw_pose')

    for key in smplx_params:
        if isinstance(smplx_params[key], np.ndarray):
            smplx_params[key] = torch.tensor(smplx_params[key])

    # dirty fix for AnimatableGaussians
    # https://github.com/lizhe00/AnimatableGaussians/blob/master/utils/smpl_util.py#L83
    if 'flat_hand_mean' not in smplx_params:
        smplx_params['flat_hand_mean'] = True

    # ##########
    # model = smplx_utils.create_smplx_model(**smplx_params, 
    #                                        batch_size=smplx_params['global_orient'].shape[0])
    # out = model(**smplx_params)
    # libcore.save_mesh_to_obj('e:/dummy/mesh.obj', out['vertices'][1500], model.faces)
    # ##########

    idxs = [int(idx) for idx in frm_list]
    for k in smplx_params:
        if isinstance(smplx_params[k], torch.Tensor):
            smplx_params[k] = smplx_params[k].detach().clone().cpu()
            if smplx_params[k].shape[0] > 1:
                smplx_params[k] = smplx_params[k][idxs]
    return smplx_params

class AvatarDataset(torch.utils.data.Dataset):
    def __init__(self):
        pass

    ##################################################
    def load_all_smplx(self, fn=None):
        if fn is None:
            fn = os.path.join(self.config.dat_dir, 'all_smplx.pt')

        if not os.path.isabs(fn):
            fn = os.path.join(self.config.dat_dir, fn)

        if os.path.isfile(fn):
            if fn.endswith('.pt'):
                self.smplx_params = load_smplx_pt(fn, self.frm_list)
            elif fn.endswith('.npz'):
                self.smplx_params = load_smplx_npz(fn, self.frm_list)

            if 'smplx_forward_transl' in self.smplx_params:
                self.smplx_forward_transl = self.smplx_params['smplx_forward_transl']
        else:
            self.smplx_params = None

        if self.config.get('smplx_nofacial', False):
            reset_smplx_facial(self.smplx_params, self.config.smplx_nofacial)

    def load_face_dpe(self, dpe_path):
        if not os.path.isabs(dpe_path):
            dpe_path = os.path.join(self.dat_dir, dpe_path)
        if os.path.isdir(dpe_path):
            dpe_path = os.path.join(dpe_path, 'dpe-multi-faces.zip')

        if dpe_path.endswith('.pt'):
            all_codes = torch.load(dpe_path, map_location='cpu')
            exp_codes = [cc['exp'] for cc in all_codes]
        elif dpe_path.endswith('.zip'):
            mem_fs = fs.open_fs('mem://')
            mem_fs.makedirs('dpe')
            print(f'[AvatarDataset] unzip {dpe_path}')
            with mem_fs.opendir('dpe') as dpe_fs:
                with fs.open_fs(f'zip://{dpe_path}') as zip_fs:
                    fs.copy.copy_fs(zip_fs, dpe_fs)

            fns = [fn for fn in mem_fs.listdir('dpe') if fn.endswith('.pt')]
            exp_codes = []
            for frm_idx in tqdm(self.frm_list, desc='[AvatarDataset] load dpe'):
                _fns = [fn for fn in fns if fn.startswith(f'dpe-{frm_idx:06d}')]
                if len(_fns) > 0:
                    _codes = []
                    for fn in _fns:
                        with mem_fs.open(f'dpe/{fn}', 'rb') as fp:
                            cc = torch.load(fp, map_location='cpu')
                        _codes.append(cc['exp'])
                    exp_codes.append(torch.concat(_codes, dim=0))
                else:
                    exp_codes.append(None)

        else:
            raise not NotImplementedError

        self.exp_codes = exp_codes

    def load_face_DAD(self, DAD_path):
        if not os.path.isabs(DAD_path):
            DAD_path = os.path.join(self.dat_dir, DAD_path)

        if os.path.isfile(DAD_path):
            all_codes = torch.load(DAD_path, map_location='cpu')
            exp_codes = [cc['expression'] for cc in all_codes]
        else:
            fns = [fn for fn in os.listdir(DAD_path) if fn.endswith('.json')]

            exp_codes = []
            for idx, frm_idx in tqdm(enumerate(self.frm_list), total=len(self.frm_list), desc='[AvatarDataset] load DAD-3DHeads'):
                _fns = [fn for fn in fns if fn.startswith(f'{frm_idx:06d}') and fn.endswith('_flame_params.json')]
                if len(_fns) > 0:
                    _codes = []
                    for fn in _fns:
                        with open(os.path.join(DAD_path, fn), 'r') as fp:
                            cc = json.load(fp)
                        _codes.append(torch.tensor(cc['expression']).float().unsqueeze(0))
                    exp_codes.append(torch.concat(_codes, dim=0))

                    # replace jaw
                    if self.smplx_params is not None:
                        self.smplx_params['jaw_pose'][idx] = torch.tensor(cc['jaw']).float()

                else:
                    exp_codes.append(None)

        self.exp_codes = exp_codes

    def load_face_deca(self, deca_path):
        if not os.path.isabs(deca_path):
            deca_path = os.path.join(self.dat_dir, deca_path)

        with open(deca_path, 'r') as fp:
            cc = json.load(fp)

        exp_codes = []
        for idx, frm_idx in tqdm(enumerate(self.frm_list), total=len(self.frm_list), desc='[AvatarDataset] load deca'):
            key = f'{frm_idx:06d}'
            if key in cc:
                exp_codes.append(torch.tensor(cc[key]['exp']).float())

                # replace jaw
                """
                - This is the IMAvatar/DECA version of FLAME
                - Originally from: https://github.com/zhengyuf/IMavatar/tree/main/code/flame
                - What's changed from normal FLAME:
                - There's a `factor=4` in the `flame.py`, making the output mesh 4 times larger
                - The input `full_pose` is [Nx15], which is a combination of different pose components
                - In a standard FLAME model, there is `pose_params`[Nx6], `neck_pose`[Nx3], `eye_pose`[Nx6].
                    To convert to `full_pose`:
                    ```
                    # [3] global orient
                    # [3] neck
                    # [3] jaw
                    # [6] eye
                    full_pose = torch.concat([pose_params[:, :3], neck_pose, pose_params[:, 3:], eye_pose], dim=-1)
                    ```
                """
                if self.smplx_params is not None:
                    self.smplx_params['jaw_pose'][idx] = torch.tensor(cc[key]['pose'])[:, 3:6].float()

            else:
                exp_codes.append(None)

        self.exp_codes = exp_codes

    def load_smplx_params(self, idx):
        smplx_params = {}
        for key in self.smplx_params:
            if not isinstance(self.smplx_params[key], torch.Tensor):
                smplx_params[key] = self.smplx_params[key]
            else:
                if key == 'betas':
                    smplx_params[key] = self.smplx_params[key].detach().clone()
                else:
                    smplx_params[key] = self.smplx_params[key][idx].detach().clone()
                    if len(smplx_params[key].shape) == 1:
                        smplx_params[key] = smplx_params[key].unsqueeze(0)
        
        return smplx_params

    ##################################################
    def construct_pca(self):
        num_frames = self.num_frames
        smplx_params_list = []
        for idx in range(num_frames):
            params = self.load_smplx_params(idx)
            smplx_params_list.append(params)

        smplx_params = smplx_utils.concate_smplx_list(smplx_params_list)
        full_pose = smplx_utils.get_smplx_full_pose(smplx_params)

        # full pose without global orient
        # jaw_pose should already be zeros if set nofacial
        full_pose = full_pose[:, 3:]

        # pca
        pose_pca = PCA(n_components=full_pose.shape[-1])
        pose_pca.fit(full_pose)
        self.pca = {
            'pose_pca': pose_pca,
        }

        if hasattr(self, 'exp_codes'):
            codes = []
            for cc in self.exp_codes:
                if cc is not None:
                    codes.append(cc)
            codes = torch.concat(codes, dim=0)

            exp_pca = PCA(n_components=codes.shape[-1])
            exp_pca.fit(codes)
            self.pca.update({
                'exp_pca': exp_pca,
            })



