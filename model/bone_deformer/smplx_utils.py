import os
import copy
import torch
import cv2
import numpy as np
from .. import libcore
from . import smplx

def get_smplx_model_path(model_type='smplx', fn=None):
    if fn is None:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/smplx'))
    else:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/smplx', model_type, fn))

def create_smplx_model(model_path=None, gender='neutral', model_type='smplx', ext='npz',
                       skip_betas=False, skip_v_template=False, skip_poses=True,
                       **smplx_params):
    if model_path is None:
        model_path = get_smplx_model_path()
    elif not os.path.exists(model_path):
        model_path = get_smplx_model_path(model_type=model_type, fn=model_path)

    if skip_betas:
        if 'betas' in smplx_params:
            smplx_params.pop('betas')

    if skip_v_template:
        if 'v_template' in smplx_params:
            smplx_params.pop('v_template')
            
    if skip_poses:
        keys = [key for key in smplx_params]
        for key in keys:
            if isinstance(smplx_params[key], torch.Tensor):
                if key != 'betas' and key != 'v_template':
                    smplx_params.pop(key)

    for key in smplx_params:
        if isinstance(smplx_params[key], torch.Tensor):
            smplx_params[key] = smplx_params[key].cpu()

    smplx_model = smplx.create(model_path, gender=gender, model_type=model_type, ext=ext,
                               **smplx_params)
    
    # by default disable inner grads
    [p.requires_grad_(False) for p in smplx_model.parameters()]

    # # replace left elbow J_regressor
    # if 'lite' in gender:
    #     X_regressor = load_smplx_J_regressor_body25_smplx_lite()
    # else:
    #     X_regressor = load_smplx_J_regressor_body25_smplx()
    # smplx_model.J_regressor[18, :] = X_regressor[6, :]

    return smplx_model

def create_smplx_lite_model(model_path=None, gender='male', model_type='smplx-lite', ext='pkl',
                       **smplx_params):
    if model_path is None:
        model_path = get_smplx_model_path()
    elif not os.path.exists(model_path):
        model_path = get_smplx_model_path(model_type=model_type, fn=model_path)

    model_path = os.path.join(model_path, model_type, 'SMPLX-LITE_{}.{ext}'.format(gender.upper(), ext=ext))

    smplx_model = smplx.create(model_path, gender=gender, model_type=model_type, ext=ext,
                               **smplx_params)
    return smplx_model

def load_regressor(regressor_path):
    if regressor_path.endswith('.npy'):
        X_regressor = torch.tensor(np.load(regressor_path)).float()
    elif regressor_path.endswith('.txt'):
        data = np.loadtxt(regressor_path)
        with open(regressor_path, 'r') as f:
            shape = f.readline().split()[1:]
        reg = np.zeros((int(shape[0]), int(shape[1])))
        for i, j, v in data:
            reg[int(i), int(j)] = v
        X_regressor = torch.tensor(reg).float()
    else:
        import ipdb; ipdb.set_trace()
    return X_regressor

def load_smplx_J_regressor_body25_smplx(fn='J_regressor_body25_smplx.txt', dtype=torch.float, device='cpu'):
    if not os.path.isabs(fn):
        fn = get_smplx_model_path(model_type='', fn=fn)
    return load_regressor(fn).to(dtype=dtype, device=device)

def load_smplx_J_regressor_body25_smplx_lite(fn='J_regressor_body25_smplx_lite.txt', dtype=torch.float, device='cpu'):
    if not os.path.isabs(fn):
        fn = get_smplx_model_path(model_type='', fn=fn)
    return load_regressor(fn).to(dtype=dtype, device=device)

def write_J_regressor(fn, J_regressor):
    with open(fn, 'w') as fp:
        fp.write(f'# {J_regressor.shape[0]} {J_regressor.shape[1]}\n')
        for i in range(J_regressor.shape[0]):
            for j in range(J_regressor.shape[1]):
                if J_regressor[i, j] != 0:
                    fp.write(f'{i} {j} {J_regressor[i, j]}\n')

def regress_joints(X_regressor, verts):
    if len(verts.shape) == 2:
        return torch.einsum('ik,kj->ij', X_regressor, verts)
    else:
        return torch.einsum('ik,bkj->bij', X_regressor, verts)

def save_joints_to_ply(fn, joints, parents):
    if len(joints.shape) == 3:
        joints = joints[0]

    joints = joints.detach().cpu()
    parents = parents.detach().cpu()

    ply_writer = libcore.PlyWriter()
    for i in range(joints.shape[0]):
        ply_writer.addVertex(joints[i])
        if parents[i] >= 0:
            ply_writer.addEdgeByIdx(parents[i], i)
    ply_writer.writeToPly(fn)

def load_smplx_part_labels(gender='male', model_type='smplx', model_path=None):
    assert gender in ['male', 'female']

    if model_path is None:
        model_path = get_smplx_model_path()

    # https://github.com/Skype-line/X-Avatar#quick-demo
    # load part labels
    import pickle as pkl
    verts_ids = pkl.load(open(os.path.join(model_path, model_type, f'non_watertight_{gender}_vertex_labels.pkl'), 'rb'), 
                         encoding='latin1')
    return verts_ids        

def convert_smplx_to_meshcpu(smplx_model, V=None):
    if V is None:
        V = smplx_model.v_template.squeeze(0)
    if isinstance(V, torch.Tensor):
        V = V.detach().cpu().numpy()
    
    mesh = libcore.MeshCpu()
    mesh.V = V
    mesh.F = smplx_model.faces.astype(int)
    mesh.update_per_vertex_normals()
    if hasattr(smplx_model, 'tc'):
        mesh.TC = smplx_model.tc
        mesh.FTC = smplx_model.tc_faces
    return mesh

def save_smplx_to_obj(fn, smplx_model, V=None):
    mesh = convert_smplx_to_meshcpu(smplx_model, V=V)
    mesh.save_to_obj(fn)

def write_smplx_objs(smplx_dir, frm_list, smplx_model, out, max_workers=8, skip_existed=False):
    def _write_smplx_obj(idx):
        frm_idx = frm_list[idx]
        mesh = convert_smplx_to_meshcpu(smplx_model, V=out['vertices'][idx])
        obj_fn = os.path.join(smplx_dir, f'smplx_{frm_idx:06d}.obj')
        if skip_existed and os.path.isfile(obj_fn):
            return
        mesh.save_to_obj(obj_fn)

    num_frames = len(frm_list)
    idxs = [i for i in range(num_frames)]

    import concurrent.futures
    from tqdm import tqdm
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executrer:
        for res in tqdm(executrer.map(_write_smplx_obj, idxs), total=num_frames):
            pass

def convert_smplx_params_cv2gl(smplx_params):
    # create model
    smplx_model = create_smplx_model(**smplx_params)

    # target verts in gl
    out = smplx_model(**smplx_params)
    verts_gl = out['vertices'].detach().clone()
    verts_gl[0, :, 1:3] = -verts_gl[0, :, 1:3]

    # flip R
    rvec = smplx_params['poses'][0, :3].numpy()
    smplx_R = cv2.Rodrigues(rvec)[0]
    smplx_R_gl = copy.deepcopy(smplx_R)
    smplx_R_gl[1:3, :] = -smplx_R_gl[1:3, :]
    rvec_gl = cv2.Rodrigues(smplx_R_gl)[0]

    # calc transl to target
    smplx_params_gl = copy.deepcopy(smplx_params)
    smplx_params_gl['poses'][0, :3] = torch.from_numpy(rvec_gl).view(-1)
    smplx_params_gl['transl'] = torch.zeros_like(smplx_params_gl['transl'])
    out = smplx_model(**smplx_params_gl)
    
    smplx_t_gl = (verts_gl - out['vertices']).mean(dim=-2)
    smplx_params_gl['transl'] = smplx_t_gl

    return smplx_params_gl

def get_smplx_full_pose(smplx_params, smplx_model=None):
    # Concatenate all pose vectors
    # 1 + 21 + 1 + 1 + 1 + 15 + 15 = 55 joints
    # 55 * 3 = 165
    keys = ['global_orient', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'left_hand_pose', 'right_hand_pose']

    full_pose = []
    for key in keys:
        if key in smplx_params:
            full_pose.append(smplx_params[key])
        elif smplx_model is not None:
            full_pose.append(getattr(smplx_model, key))
        else:
            if key in ['jaw_pose', 'leye_pose', 'reye_pose']:
                full_pose.append(torch.zeros_like(smplx_params['global_orient']))

    full_pose = torch.cat(full_pose, dim=-1)
    return full_pose

def set_full_pose_to_params(full_pose, smplx_params={}):
    smplx_params['global_orient'] = full_pose[..., :3]
    smplx_params['body_pose'] = full_pose[..., 3:66]
    smplx_params['jaw_pose'] = full_pose[..., 66:69]
    smplx_params['leye_pose'] = full_pose[..., 69:72]
    smplx_params['reye_pose'] = full_pose[..., 72:75]
    smplx_params['left_hand_pose'] = full_pose[..., 75:120]
    smplx_params['right_hand_pose'] = full_pose[..., 120:165]
    return smplx_params

def get_smplx_Apose_params(device='cuda'):
     body_pose = torch.zeros(1, 63).to(device)
     body_pose[:, 2] = np.pi / 6
     body_pose[:, 5] = -np.pi / 6

     jaw_pose = torch.zeros(1, 3).to(device)
     jaw_pose[:, 0] = 0.2

     pose_params = {
         'body_pose': body_pose,
         'jaw_pose': jaw_pose,
     }
     return pose_params

def set_smplx_to_Apose(smplx_params):
    pose_params = { key: smplx_params[key] for key in smplx_params }

    for key in ['global_orient', 'body_pose', 'jaw_pose', 
                'leye_pose', 'reye_pose', 'left_hand_pose', 'right_hand_pose',
                'transl', 'expression']:
        if key in pose_params:
            pose_params[key] = torch.zeros_like(pose_params[key])

    pose_params['body_pose'][:, 2] = np.pi / 6
    pose_params['body_pose'][:, 5] = -np.pi / 6

    pose_params['jaw_pose'][:, 0] = 0.2
    return pose_params

def reset_smplx_poses(smplx_params):
    pose_params = { key: smplx_params[key] for key in smplx_params }

    for key in ['global_orient', 'body_pose', 'jaw_pose', 
                'leye_pose', 'reye_pose', 'left_hand_pose', 'right_hand_pose',
                'transl', 'expression']:
        if key in pose_params:
            pose_params[key] = torch.zeros_like(pose_params[key])

    return pose_params

def concate_smplx_list(smplx_list, mean_betas=True):
    all_params = {}

    for smplx_params in smplx_list:
        for key in smplx_params:
            if isinstance(smplx_params[key], torch.Tensor):
                if key in all_params:
                    all_params[key].append(smplx_params[key])
                else:
                    all_params[key] = [smplx_params[key]]
            else:
                all_params[key] = smplx_params[key]
        
    for key in all_params:
        if isinstance(all_params[key], list):
            all_params[key] = torch.concat(all_params[key], dim=0)
    
    if mean_betas:
        all_params['betas'] = all_params['betas'].mean(dim=0, keepdims=True)

    return all_params

def get_smplx_params_by_idx(all_params, idx):
    smplx_params = {}
    for key in all_params:
        if isinstance(all_params[key], torch.Tensor):
            # if key == 'betas' or key == 'v_template':
            if all_params[key].shape[0] == 1:
                smplx_params[key] = all_params[key]
            else:
                if isinstance(idx, int):
                    smplx_params[key] = all_params[key][idx:idx+1]
                else:
                    smplx_params[key] = all_params[key][idx]
        else:
            smplx_params[key] = all_params[key]
    return smplx_params

def fix_hand_mean(smplx_model, smplx_params):
    if smplx_model.flat_hand_mean == smplx_params.get('flat_hand_mean', True):
        return smplx_params
    
    params = copy.deepcopy(smplx_params)
    hands_meanl = smplx_model.hands_meanl.to(smplx_params['left_hand_pose'])
    hands_meanr = smplx_model.hands_meanr.to(smplx_params['right_hand_pose'])

    if smplx_model.flat_hand_mean:
        params['left_hand_pose'] = params['left_hand_pose'] - hands_meanl
        params['right_hand_pose'] = params['right_hand_pose'] - hands_meanr
    else:
        params['left_hand_pose'] = params['left_hand_pose'] + hands_meanl
        params['right_hand_pose'] = params['right_hand_pose'] + hands_meanr
    return params

def fix_hand_pca_comps(smplx_params, use_pca=False, left_hand_components=None, right_hand_components=None, 
                       **kwargs):
    if use_pca == smplx_params.get('use_pca', True):
        return smplx_params
    
    params = { key: smplx_params[key] for key in smplx_params }
    left_hand_pose = smplx_params['left_hand_pose']
    right_hand_pose = smplx_params['right_hand_pose']
    
    if not use_pca:
        if left_hand_components is None or right_hand_components is None:
            tmp_model = create_smplx_model(**smplx_params)
            left_hand_components = tmp_model.left_hand_components.to(left_hand_pose)
            right_hand_components = tmp_model.right_hand_components.to(right_hand_pose)

        params['left_hand_pose'] = torch.einsum(
            'bi,ij->bj', [left_hand_pose, left_hand_components])
        params['right_hand_pose'] = torch.einsum(
            'bi,ij->bj', [right_hand_pose, right_hand_components])
        
        if 'NUM_HANDJOINTS' in params:
            params.pop('NUM_HANDJOINTS')
    else:
        raise NotImplementedError

    params['use_pca'] = use_pca
    return params

##################################################
def load_and_detach(fn, map_location='cpu'):
    smplx_params = torch.load(fn, map_location=map_location)
    for key in smplx_params:
        if isinstance(smplx_params[key], torch.Tensor):
            smplx_params[key] = smplx_params[key].detach()

            if len(smplx_params[key].shape) == 1:
                smplx_params[key] = smplx_params[key].unsqueeze(0)

    if 'use_pca' not in smplx_params:
        smplx_params['use_pca'] = True
    if 'flat_hand_mean' not in smplx_params:
        smplx_params['flat_hand_mean'] = True
    if 'num_betas' not in smplx_params and 'betas' in smplx_params:
        smplx_params['num_betas'] = smplx_params['betas'].shape[-1]
    if 'num_expression_coeffs' not in smplx_params and 'expression' in smplx_params:
        smplx_params['num_expression_coeffs'] = smplx_params['expression'].shape[-1]
    if 'gender' not in smplx_params:
        smplx_params['gender'] = 'male'
    if 'model_type' not in smplx_params:
        smplx_params['model_type'] = 'smplx'

    return smplx_params

