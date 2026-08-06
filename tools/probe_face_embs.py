#!/usr/bin/env python
"""Prove the DPE face branch is actually live -- by hooking the layer that consumes it.

`degas_vae_driver.py` silently substitutes zeros when no code arrives:

    if face_embs is None:
        face_embs = torch.zeros((1, self.n_face_embs))

which is exactly how the first arm A ended up a static-face strawman while looking like a
healthy 44 h run. So don't infer it from the config: put a forward hook on `face_embs_fc`,
run real batches through the real model, and read what the layer actually received.

    cd <degas repo> && CUDA_VISIBLE_DEVICES=5 python probe_face_embs.py \
        --dat_dir .../P1C1 --configs configs/...,configs/dreams/p1_face_A_dpe.yaml
"""
from __future__ import annotations

import argparse

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
from model.degas_model import DEGASModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dat_dir', required=True)
    ap.add_argument('--configs', required=True)
    ap.add_argument('--n', type=int, default=16)
    a = ap.parse_args()

    cfg = load_from_config(a.configs.split(','), dat_dir=a.dat_dir)
    cfg.dataset.dat_dir = a.dat_dir

    train = make_frameset_data(cfg.dataset, split='train')
    print(f'\ndataset exp_codes present: {train.exp_codes is not None}')
    if train.exp_codes is not None:
        n_ok = sum(c is not None for c in train.exp_codes)
        print(f'  frames with a code: {n_ok}/{len(train.exp_codes)}')
        print(f'  per-frame code tensor: {tuple(train.exp_codes[0].shape)}')

    smplx_optim = SMPLXOptimizer(**cfg.optim.smplx_optim)
    smplx_optim.setup_smplx_params_frameset(train)
    cano_params = smplx_optim.get_tpose_params()
    cano_mesh = smplx_optim.get_tpose_mesh().detach().clone()

    gs = DEGASModel(cfg.model, {'mesh_from': 'smplx_optim', 'smplx_optim': smplx_optim},
                    verbose=False)
    gs.create_from_canonical(cano_params, cano_mesh)
    gs.update_to_pose(cano_params)

    seen = []
    hooked = []
    for name, mod in gs.named_modules():
        if name.endswith('face_embs_fc'):
            hooked.append(name)
            mod.register_forward_hook(
                lambda m, i, o: seen.append(i[0].detach().float().cpu().numpy()))
    print(f'hooked modules: {hooked}')
    if not hooked:
        print('FAIL: no face_embs_fc found -- n_face_embs is 0, the branch does not exist')
        return 1

    for k in range(a.n):
        batch = train[k % len(train)]
        gs.pre_render(batch)

    if not seen:
        print('FAIL: face_embs_fc never ran')
        return 1

    X = np.concatenate(seen, axis=0)
    zero_rows = int((np.abs(X).sum(1) == 0).sum())
    print(f'\ncaptured {X.shape[0]} face_embs vectors of dim {X.shape[1]}')
    print(f'  all-zero vectors: {zero_rows} / {X.shape[0]}')
    print(f'  |value| mean {np.abs(X).mean():.5f}   max {np.abs(X).max():.5f}')
    print(f'  across-batch variation (per-dim std, mean) {X.std(0).mean():.5f}')

    ok = zero_rows == 0 and X.shape[1] == 512 and X.std(0).mean() > 1e-4
    print('\n' + ('PASS -- the branch receives real, frame-varying 512-d DPE codes'
                  if ok else
                  'FAIL -- zeros, wrong width, or constant across frames'))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
