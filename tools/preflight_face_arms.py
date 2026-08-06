#!/usr/bin/env python
"""Prove that arm B's expression/jaw actually reach the SMPL-X mesh and arm A's do not.

Three settings have to agree for B to work (reader, model, optimiser). Any one of them
left at the DEGAS default silently discards the expression -- and the run would still
train happily for three days and produce a face identical to A. So check it in 30 seconds
instead.

    CUDA_VISIBLE_DEVICES=5 python preflight_face_arms.py \
        --dat_dir .../data/P1C1 --frames 944 903 750
"""
import argparse
import sys

import numpy as np
import torch

# These tools live in tools/ but import the packages at the repo root, and Python
# puts the SCRIPT's directory on sys.path, not the caller's. Add the root so
# `python tools/<name>.py` works from anywhere.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from model.libcore.omegaconf_utils import load_from_config
from dataset.dataset_helper import make_frameset_data
from model.bone_deformer.smplx_optim import SMPLXOptimizer

BASE = ['configs/degas_config.yaml', 'configs/degas_vae_driver.yaml',
        'configs/dreams/p1_train_base.yaml']


def build(arm, dat_dir, frames):
    cfg = load_from_config(BASE + [f'configs/dreams/p1_face_{arm}.yaml'])
    cfg.dataset.dat_dir = dat_dir
    cfg.dataset.build_cache = False          # no decode needed: we only touch SMPL-X
    cfg.dataset.train.frm_list = list(frames)
    cfg.dataset.train.cam_select = [3]

    print(f'\n--- arm {arm} ---')
    print(f'  dataset.smplx_nofacial = {cfg.dataset.get("smplx_nofacial", None)!r}')
    print(f'  model.smplx_nofacial   = {cfg.model.get("smplx_nofacial", None)!r}')
    print(f'  optim_skip             = {list(cfg.optim.smplx_optim.get("optim_skip", []))}')

    ds = make_frameset_data(cfg.dataset, split='train')
    p = ds.load_smplx_params(0)
    print(f'  reader   |expression| = {p["expression"].abs().max():.4f}   '
          f'|jaw_pose| = {p["jaw_pose"].abs().max():.4f}')

    optim = SMPLXOptimizer(**cfg.optim.smplx_optim)
    optim.setup_smplx_params_frameset(ds)
    print(f'  optimiser forward_keys has expression: '
          f'{"expression" in optim.forward_keys}, params_keys has expression: '
          f'{"expression" in optim.params_keys}')
    print(f'  optimiser tensor |expression| = {optim.expression.abs().max().item():.4f}   '
          f'|jaw_pose| = {optim.jaw_pose.abs().max().item():.4f}')

    # Mirror the TRAINING path exactly, not SMPLXOptimizer.forward():
    #   degas_model.pre_render -> smplx_optim.get_smplx_params(idx)   (params_keys only)
    #   -> avatar_base.set_pose, which POPS expression/jaw_pose when model.smplx_nofacial
    #      mentions them
    #   -> smplx_deformer.update(**posed_params)
    # (SMPLXOptimizer.forward is never called during training, and would in fact throw a
    # device error here: optim_skip tensors are allocated on CPU and only params_keys are
    # moved to the GPU. Harmless in practice, but don't build the check on it.)
    from model.bone_deformer import smplx_utils

    nofacial = cfg.model.get('smplx_nofacial', '') or ''
    verts = []
    for i in range(len(frames)):
        posed = dict(optim.get_smplx_params(i))
        popped = ['betas']
        if 'exp' in nofacial:
            popped.append('expression')
        if 'jaw' in nofacial:
            popped.append('jaw_pose')
        for k in popped:
            posed.pop(k, None)
        present = [k for k in ('expression', 'jaw_pose') if k in posed]
        if i == 0:
            print(f'  reaches the mesh: {present if present else "NEITHER"}')
        init = {k: v for k, v in posed.items() if not isinstance(v, torch.Tensor)}
        model = smplx_utils.create_smplx_model(betas=optim.betas.cpu(), **init)
        fwd = {k: v.detach().cpu() for k, v in posed.items()
               if isinstance(v, torch.Tensor)}
        fwd['betas'] = optim.betas.detach().cpu()
        with torch.no_grad():
            verts.append(model(**fwd)['vertices'][0].numpy())
    return cfg, np.stack(verts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dat_dir', required=True)
    ap.add_argument('--frames', type=int, nargs='*', default=[944, 903, 750])
    a = ap.parse_args()

    _, va = build('A', a.dat_dir, a.frames)
    _, vb = build('B', a.dat_dir, a.frames)

    d = np.linalg.norm(va - vb, axis=-1)                 # (F, V)

    # The face region straight out of the model, not a height heuristic: a raised hand is
    # the highest thing in the mesh at frame 944, so "top 10% by y" picks the wrong
    # vertices. Expression acts through shapedirs[..., 300:400]; jaw acts through the LBS
    # weight of joint 22. Their union is exactly the geometry these two arms can differ on.
    from model.bone_deformer import smplx_utils
    m = smplx_utils.create_smplx_model(gender='neutral', model_type='smplx',
                                       num_betas=300, num_expression_coeffs=100,
                                       use_pca=False, flat_hand_mean=False)
    expr_moved = m.expr_dirs.detach().cpu().numpy().reshape(m.expr_dirs.shape[0], -1)
    face = (np.linalg.norm(expr_moved, axis=1) > 1e-6) | \
           (m.lbs_weights.detach().cpu().numpy()[:, 22] > 1e-4)
    print(f'\nface region = {int(face.sum())} / {len(face)} vertices '
          f'(expression blendshapes + jaw LBS)')

    print('=== arm A mesh vs arm B mesh ===')
    print(f'  all vertices    : mean {d.mean()*1000:7.3f} mm   max {d.max()*1000:7.3f} mm')
    print(f'  FACE            : mean {d[:, face].mean()*1000:7.3f} mm   '
          f'max {d[:, face].max()*1000:7.3f} mm')
    print(f'  everywhere else : mean {d[:, ~face].mean()*1000:7.3f} mm   '
          f'max {d[:, ~face].max()*1000:7.3f} mm')

    # within-arm motion across the probe frames: does the face MOVE in B but not in A?
    for name, v in (('A', va), ('B', vb)):
        mm = np.linalg.norm(v - v.mean(0, keepdims=True), axis=-1)
        print(f'  arm {name}: face motion across the {len(a.frames)} probe frames = '
              f'{mm[:, face].mean()*1000:7.3f} mm mean, {mm[:, face].max()*1000:7.3f} mm max')

    ok = (d[:, face].max() > 1.0e-3          # B's face differs from A's by > 1 mm
          and d[:, ~face].max() < 1.0e-3)    # and nothing outside the face moved
    print('\n' + ('PASS -- B moves the face and ONLY the face; A has no facial signal'
                  if ok else
                  'FAIL -- expression is being dropped, or it is moving the body too'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
