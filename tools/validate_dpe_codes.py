#!/usr/bin/env python
"""Checkpoint validation for the generated DPE codes, before committing 40 h of GPU.

Four questions, in order of how badly a wrong answer would hurt:
  1. Are the codes non-degenerate (not constant, not NaN)?
  2. Do they TRACK EXPRESSION -- do open-mouth and closed-mouth frames separate?
  3. Are they expression and not just HEAD POSE in disguise? DPE claims disentanglement;
     if the code correlates better with global orientation than with jaw opening, the
     "expression" branch is contaminated and arm A would be a pose baseline, not a face one.
  4. Does DEGAS's own `load_face_dpe` parse the zip with no reshaping?

    cd <degas repo> && python validate_dpe_codes.py --capture .../P1C1 --codes .../P1C1
"""
from __future__ import annotations

import argparse
import io
import json
import os
import zipfile

import numpy as np
import torch


def load_zip_codes(zip_fn, cam):
    """{frame: (512,)} for one camera, straight out of the zip."""
    out = {}
    with zipfile.ZipFile(zip_fn) as zf:
        for n in zf.namelist():
            if not n.endswith(f'-cam{cam:02d}.pt'):
                continue
            f = int(n.split('-')[1])
            d = torch.load(io.BytesIO(zf.read(n)), map_location='cpu', weights_only=False)
            out[f] = d['exp'].numpy()[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--capture', required=True)
    ap.add_argument('--codes', required=True)
    ap.add_argument('--cam', type=int, default=7)
    a = ap.parse_args()

    zip_fn = os.path.join(a.codes, 'dpe-multi-faces.zip')
    codes = load_zip_codes(zip_fn, a.cam)
    z = np.load(os.path.join(a.capture, 'smplx.npz'), allow_pickle=False)
    frames = [int(f) for f in z['frames'].astype(int) if int(f) in codes]
    C = np.stack([codes[f] for f in frames])                      # (T,512)
    row = {int(f): i for i, f in enumerate(z['frames'].astype(int))}
    idx = [row[f] for f in frames]

    jaw = np.linalg.norm(z['jaw_pose'][idx], axis=1)              # mouth opening
    expr = np.linalg.norm(z['expression'][idx], axis=1)
    orient = z['global_orient'][idx]                              # head/body yaw etc.

    print(f'=== 1. non-degenerate?  cam{a.cam:02d}, {len(frames)} frames ===')
    print(f'  shape {C.shape}   dtype {C.dtype}   NaN {np.isnan(C).any()}   '
          f'Inf {np.isinf(C).any()}')
    print(f'  per-dim std: mean {C.std(0).mean():.5f}  min {C.std(0).min():.5f}  '
          f'max {C.std(0).max():.5f}')
    print(f'  dims that are effectively constant (std<1e-6): '
          f'{int((C.std(0) < 1e-6).sum())} / {C.shape[1]}')

    Cc = C - C.mean(0)
    U, S, Vt = np.linalg.svd(Cc, full_matrices=False)
    ev = S ** 2 / (S ** 2).sum()
    print(f'  variance in first 5 PCs: {ev[:5].round(4)}  (sum {ev[:5].sum():.3f})')

    print('\n=== 2. do the codes separate open-mouth from closed-mouth? ===')
    k = max(20, len(frames) // 20)
    o = np.argsort(-jaw)[:k]        # most open
    c = np.argsort(jaw)[:k]         # most closed
    mo, mc = C[o].mean(0), C[c].mean(0)
    within = (np.linalg.norm(C[o] - mo, axis=1).mean() +
              np.linalg.norm(C[c] - mc, axis=1).mean()) / 2
    between = np.linalg.norm(mo - mc)
    print(f'  top/bottom {k} frames by SMPL-X jaw opening')
    print(f'  |mean_open - mean_closed| = {between:.4f}')
    print(f'  mean within-group spread   = {within:.4f}')
    print(f'  separation ratio between/within = {between/within:.3f}   '
          f'({"SEPARATED" if between/within > 0.5 else "NOT separated"})')

    # projection on the open-vs-closed axis, correlated with jaw over ALL frames
    axis = (mo - mc) / (np.linalg.norm(mo - mc) + 1e-12)
    proj = Cc @ axis
    r_jaw = float(np.corrcoef(proj, jaw)[0, 1])
    r_expr = float(np.corrcoef(proj, expr)[0, 1])
    print(f'  corr(code projected on open-closed axis, jaw opening) = {r_jaw:+.3f}')
    print(f'  corr(same projection, |SMPL-X expression|)            = {r_expr:+.3f}')

    print('\n=== 3. expression, or head pose in disguise? ===')
    def best_r(x, Y):
        return max(abs(float(np.corrcoef(x, Y[:, j])[0, 1])) for j in range(Y.shape[1]))
    r_pc_jaw = max(abs(float(np.corrcoef(U[:, i] * S[i], jaw)[0, 1])) for i in range(5))
    r_pc_or = max(best_r(U[:, i] * S[i], orient) for i in range(5))
    print(f'  best |corr| of a top-5 code PC with jaw opening      = {r_pc_jaw:.3f}')
    print(f'  best |corr| of a top-5 code PC with global_orient    = {r_pc_or:.3f}')
    print('  -> ' + ('expression signal is present' if r_jaw > 0.2 or r_pc_jaw > 0.2
                     else 'WEAK expression signal, investigate'))
    if r_pc_or > max(r_pc_jaw, abs(r_jaw)) + 0.25:
        print('  !! WARNING: the code tracks global orientation notably better than jaw. '
              'DPE pose/expression disentanglement may be leaking for this capture.')

    print('\n=== 4. DEGAS load_face_dpe parses it, no reshape hacks ===')
    try:
        from dataset.dataset_utils import AvatarDataset
        ds = AvatarDataset.__new__(AvatarDataset)
        ds.dat_dir = a.capture
        ds.frm_list = frames[:200]
        ds.smplx_params = None
        AvatarDataset.load_face_dpe(ds, zip_fn)
        got = [c for c in ds.exp_codes if c is not None]
        print(f'  parsed {len(got)}/{len(ds.frm_list)} frames')
        print(f'  per-frame tensor shape {tuple(got[0].shape)}  '
              f'(expect (n_faces, 512) -- n_faces = cameras extracted)')
        w = torch.rand((got[0].shape[0],)); w = w / w.sum()
        code = torch.einsum('i,ij->j', w, got[0])[None, ...]
        print(f'  batch["exp_code"] would be {tuple(code.shape)}, '
              f'|code| = {float(code.abs().max()):.4f}, nonzero = {bool(code.abs().sum() > 0)}')
        ok = code.shape == (1, 512) and float(code.abs().sum()) > 0
        print('  ' + ('OK: matches n_face_embs=512 exactly, no reshape' if ok else 'MISMATCH'))
    except Exception as e:
        print(f'  FAILED: {type(e).__name__}: {e}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
