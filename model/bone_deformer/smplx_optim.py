import os
import torch
from torch import nn
from tqdm import tqdm
import pytorch3d.structures.meshes as py3d_meshes
import pytorch3d.renderer.mesh.textures as py3d_tex
from model import libcore
from model.bone_deformer import smplx_utils

def face_label_from_vert(num_verts, vt_ids, faces):
    vert_labels = torch.zeros((num_verts,), dtype=torch.bool)
    vert_labels[vt_ids] = True

    return vert_labels[faces].any(dim=-1)


class SMPLXOptimizer(nn.Module):
    def __init__(self, 
                 smplx_forward_transl=False, 
                 optim_betas=False,
                 optim_v_template=False,
                 optim_skip=[],
                 device=torch.device('cuda'),
                 **kwargs):
        super().__init__()
        self.smplx_forward_transl = smplx_forward_transl
        self.optim_betas = optim_betas
        self.optim_v_template = optim_v_template
        self.optim_skip = optim_skip
        self.device = device

        self.init_keys = ['gender', 'model_type', 'model_path',
                          'flat_hand_mean', 'use_pca', 'num_pca_comps',
                          'betas', 'v_template',
                          'num_betas', 'num_expression_coeffs']
            
        self.condition_keys = ['body_pose', 'left_hand_pose', 'right_hand_pose', 'jaw_pose']
        self.cur_idx = None
        self.optimizer = None

    def setup_smplx_params_list(self, smplx_params_list):
        num_frames = len(smplx_params_list)
        self.num_frames = num_frames

        smplx_params = smplx_params_list[0]
        if 'betas' in smplx_params:
            if smplx_params['betas'].shape[0] > 1:
                smplx_params['betas'] = smplx_params['betas'].mean(dim=0, keepdim=True)

            if self.optim_betas:
                self.betas = nn.Parameter(smplx_params['betas'].detach().clone().to(self.device).requires_grad_(True))
            else:
                self.betas = smplx_params['betas'].detach().clone().to(self.device)
        else:
            self.betas = None

        if 'v_template' in smplx_params:
            self.v_template = smplx_params['v_template'].detach().clone().to(self.device)
        else:
            self.v_template = None

        ##############################
        # setup model and tensor
        self.init_params = { k: smplx_params[k] for k in self.init_keys if k in smplx_params }

        # self.params_keys = [ k for k in self.params_keys if k in smplx_params ]
        # self.forward_keys = [ k for k in self.forward_keys if (k in smplx_params or k == 'betas') ]
        self.params_keys = []
        self.forward_keys = []
        for key in smplx_params:
            if isinstance(smplx_params[key], torch.Tensor):
                if key == 'betas' or key == 'v_template':
                    self.forward_keys.append(key)
                else:
                    if key == 'transl':
                        self.params_keys.append(key)
                        if self.smplx_forward_transl:
                            self.forward_keys.append(key)
                    elif key in self.optim_skip:
                        print(f'[SMPLXOptimizer] optim_skip: {key}')
                        self.forward_keys.append(key)
                    else:
                        self.forward_keys.append(key)
                        self.params_keys.append(key)
                        
                    setattr(self, key, torch.zeros((num_frames, *smplx_params[key].shape[1:]), dtype=torch.float))

        ##############################
        # load tensor init value
        print('[SMPLXOptimizer] setup_smplx_params_list')
        for idx in tqdm(range(num_frames)):
            smplx_params = smplx_params_list[idx]
            for key in self.params_keys:
                getattr(self, key)[idx:idx+1, :] = smplx_params[key].detach().clone()

        # tensor to parameters
        for key in self.params_keys:
            setattr(self, key, nn.Parameter(getattr(self, key).to(self.device).requires_grad_(True)))
            # setattr(self, key, getattr(self, key).to(self.device))

        # base model
        self.smplx_model = smplx_utils.create_smplx_model(**self.init_params).to(self.device)
        for key in self.init_params:
            if isinstance(self.init_params[key], torch.Tensor):
                self.init_params[key] = self.init_params[key].to(self.device)

        # base mesh
        if self.v_template is None:
            self.v_template = self.smplx_model.v_template.float().detach().clone().to(self.device)

        if len(self.v_template.shape) == 2:
            self.v_template = self.v_template[None, ...].to(self.device)

        if self.optim_v_template:
            self.v_template = nn.Parameter(self.v_template.requires_grad_(True))

        tex = py3d_tex.TexturesUV(torch.zeros(1, 0, 0, 0).to(self.device), 
                                       torch.from_numpy(self.smplx_model.tc_faces[None, ...].astype(int)).to(self.device), 
                                       torch.from_numpy(self.smplx_model.tc[None, ...]).float().to(self.device))
        self.mesh_py3d = py3d_meshes.Meshes(self.v_template, 
                                            torch.from_numpy(self.smplx_model.faces[None, ...].astype(int)).to(self.device),
                                            textures=tex)

    def setup_smplx_params_frameset(self, frameset):
        num_frames = frameset.num_frames
        smplx_params_list = []

        print('[SMPLXOptimizer] load_smplx_params')
        for idx in tqdm(range(num_frames)):
            params = frameset.load_smplx_params(idx)
            smplx_params_list.append(params)
            
        self.frm_list = frameset.frm_list
        self.setup_smplx_params_list(smplx_params_list)

    def setup_optimizer(self, lr=4e-3):
        l = [
            {
                'params': [getattr(self, key)], 
                'name': key
            } for key in self.params_keys
        ]

        if self.optim_betas:
            l.append({
                'params': [self.betas], 
                'name': 'betas',
            })

        if self.optim_v_template:
            l.append({
                'params': [self.v_template], 
                'name': 'v_template',
            })

        self.optimizer = torch.optim.Adam(l, lr=lr, eps=1e-15)
        return self.optimizer
    
    def get_tpose_params(self):
        smplx_params = self.get_smplx_params(0)
        tpose_keys = ['body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'left_hand_pose', 'right_hand_pose']
        for key in tpose_keys:
            if key in smplx_params:
                smplx_params[key] = smplx_params[key].detach().clone()
                smplx_params[key][:] = 0
        return smplx_params

    def get_tpose_mesh(self):
        tpose_params = self.get_tpose_params()
        out = self.smplx_model(**tpose_params)
        frame_mesh = self.mesh_py3d.update_padded(out['vertices'])
        return frame_mesh

    def load_part_labels(self):
        ##############################
        # # load label for hand
        # faces = torch.tensor(self.smplx_model.faces.astype(int)).long()
        # verts_ids = smplx_utils.load_smplx_part_labels(gender=smplx_params['gender'])
        # hand_ids = torch.concat([
        #     torch.tensor(verts_ids['left_hand']), 
        #     torch.tensor(verts_ids['right_hand']),
        #     torch.tensor(verts_ids['eyes_mouth']),
        # ]).long()
        # self.hand_labels = face_label_from_vert(self.smplx_model.v_template.shape[0], hand_ids, faces).to(self.device)
        pass

    def step(self):
        if self.optimizer is not None:
            self.optimizer.step()

    def zero_grad(self):
        if self.optimizer is not None:
            self.optimizer.zero_grad()

    def smplx_state_dict(self):
        return {
            **self.state_dict(),
            'optimizer': {**self.optimizer.state_dict()},
        }

    def forward(self, idx=None):
        if idx is None:
            params = { k: getattr(self, k) for k in self.forward_keys }
        else:
            if idx < 0 or idx >= self.num_frames:
                return None

            params = {}
            for k in self.forward_keys:
                if k == 'betas':
                    params.update({ k: getattr(self, k) })
                else:
                    params.update({ k: getattr(self, k)[idx:idx+1] })

        out = self.smplx_model(**params)
        return out
    
    def update_current(self, idx=None):
        if idx is None:
            return self.cur_idx
        else:
            self.cur_idx = idx
            return idx

    def get_frame_mesh(self, idx=None):
        idx = self.update_current(idx)

        out = self.forward(idx)
        verts = out['vertices'][0]
        if not self.smplx_forward_transl:
            verts += self.transl[idx]

        frame_mesh = self.mesh_py3d.update_padded(verts[None, ...])
        return frame_mesh
    
    def get_frame_mesh_info(self, idx=None):
        frame_mesh = self.get_frame_mesh(idx)
        mesh_info = {
            'mesh_verts': frame_mesh.verts_packed(),
            'mesh_norms': frame_mesh.verts_normals_packed(),
            'mesh_faces': frame_mesh.faces_packed(),
        }
        return mesh_info

    def get_smplx_params(self, idx):
        smplx_params = {}
        for key in self.init_params:
            smplx_params.update({ key: self.init_params[key] })

        smplx_params.update({ 'betas': self.betas })

        for key in self.params_keys:
            smplx_params.update({ key: getattr(self, key)[idx:idx+1] })
        return smplx_params
    
    def get_forward_params(self):
        params = {}
        for k in self.forward_keys:
            params.update({ k: getattr(self, k) })
        return params
    
    def get_all_params(self):
        smplx_params = {}
        for key in self.init_params:
            smplx_params.update({ key: self.init_params[key] })

        smplx_params.update({ 'betas': self.betas })
        if self.v_template is not None:
            smplx_params.update({ 'v_template': self.v_template })

        for key in self.params_keys:
            smplx_params.update({ key: getattr(self, key) })

        for key in self.forward_keys:
            smplx_params.update({ key: getattr(self, key) })

        return smplx_params

    def save(self, pc_dir, frm_i):
        frame_mesh = self.get_frame_mesh(frm_i)
        mesh = libcore.MeshCpu()
        mesh.V = frame_mesh.verts_packed().detach().cpu().numpy()
        mesh.N = frame_mesh.verts_normals_packed().detach().cpu().numpy()
        mesh.F = frame_mesh.faces_packed().detach().cpu().numpy()
        mesh.save_to_obj(os.path.join(pc_dir, f'mesh_{self.frm_list[frm_i]}.obj'))

        smplx_dir = os.path.join(pc_dir, 'smplx')
        os.makedirs(smplx_dir, exist_ok=True)
        for idx, frm_idx in enumerate(self.frm_list):
            smplx_params = self.get_smplx_params(idx)
            smplx_fn = os.path.join(smplx_dir, f'smplx_{frm_idx}.pt')
            torch.save(smplx_params, smplx_fn)

    def save_refined_to(self, frame_dir, frm_i):
        frame_mesh = self.get_frame_mesh(frm_i)
        mesh = libcore.MeshCpu()
        mesh.V = frame_mesh.verts_packed().detach().cpu().numpy()
        mesh.N = frame_mesh.verts_normals_packed().detach().cpu().numpy()
        mesh.F = frame_mesh.faces_packed().detach().cpu().numpy()
        mesh.save_to_obj(os.path.join(frame_dir, f'smplx-refined.obj'))

        smplx_params = self.get_smplx_params(frm_i)
        smplx_fn = os.path.join(frame_dir, f'smplx-refined.pt')
        torch.save(smplx_params, smplx_fn)


