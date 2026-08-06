#!/usr/bin/env python
"""Extract DPE expression codes for a DREAMS-AVATAR capture, in the format DEGAS eats.

WHAT THE CODE IS  (traced through OpenTalker/DPE, not guessed)
    Generator(size=256, style_dim=512, motion_dim=20)   # run_demo.py defaults
        wa_t         = gen.enc.net_app(img)[0]          # (1,512)  per-IMAGE appearance latent
        alpha        = gen.mlp(wa_t)                    # (1,20)   motion code
        directions   = gen.dir(alpha)                   # (1,512)  Direction: QR of a (512,20) basis
        exp_code     = gen.mlp_exp(directions)          # (1,512)  <-- this is DPE's expression latent
    `dec_exp` consumes exactly `wa + exp_code`, so this is the tensor that carries expression
    and nothing else. Its width, 512, is exactly DEGAS's `n_face_embs` -- no reshaping, no
    padding, no projection anywhere in this script.

    IMPORTANT: `EncoderApp.forward(x)` takes ONE image. `wa_t` therefore depends only on the
    frame being encoded, so the expression code is an ABSOLUTE per-frame function -- there is
    no source/reference frame to choose and no risk of a hidden convention mismatch. (The
    `img_source` argument of `Encoder.forward` only exists to encode a second image in the
    same call; it does not enter `wa_t`.)

PREPROCESSING  (matches DPE's own crop_video.py + run_demo.py)
    * face box from S3FD (`face_detection.FaceAlignment`), expanded by `--pad` px per side
      (DPE hardcodes 50 px), computed ONCE on a reference frame and then held FIXED for the
      whole sequence -- DPE does exactly this (`crop_video.py` breaks after the first frame
      and reuses that box for every frame).
    * crop -> RGB -> resize 256x256 -> /255 -> (x-0.5)*2  => [-1,1]

OUTPUT  (`dpe-multi-faces.zip`, the frame-id-keyed form)
    members `dpe-{frame:06d}-cam{cc:02d}.pt`, each a dict {'exp': FloatTensor(1,512)}.
    `AvatarDataset.load_face_dpe` globs `dpe-{frm_idx:06d}*`, concatenates the per-camera
    codes to (N_faces,512) and samples a random convex combination each iteration.
    Chosen over the flat `dpe-codes.pt` deliberately: that form is a LIST indexed by POSITION
    in `frm_list`, so it silently mis-pairs codes with frames whenever the split changes,
    and it cannot hold more than one face per frame. The zip is keyed by frame id and is
    what `configs/degas_config.yaml`'s `with_face_code: dpe_face` convention expects
    (`load_face_dpe` appends `dpe-multi-faces.zip` when handed a directory).

    python extract_dpe_codes.py --capture .../data/P1C1 --cams 7 30 --out .../P1C1/dpe_face
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zipfile

import cv2
import numpy as np
import torch

# Point DPE_ROOT at your clone of https://github.com/OpenTalker/DPE (or pass --dpe-root).
# `networks.generator` and `face_detection` both live inside that repo.
DPE_ROOT = os.environ.get('DPE_ROOT', os.path.expanduser('~/DPE'))
if '--dpe-root' in sys.argv:
    DPE_ROOT = sys.argv[sys.argv.index('--dpe-root') + 1]
if not os.path.isdir(DPE_ROOT):
    raise SystemExit(
        f'[FATAL] DPE repo not found at {DPE_ROOT!r}.\n'
        '        git clone https://github.com/OpenTalker/DPE\n'
        '        then set DPE_ROOT=/path/to/DPE (or pass --dpe-root /path/to/DPE).')
sys.path.insert(0, DPE_ROOT)
from networks.generator import Generator          # noqa: E402
import face_detection                             # noqa: E402


def build_generator(ckpt, size=256, style_dim=512, motion_dim=20, ch_mult=1):
    gen = Generator(size, style_dim, motion_dim, ch_mult).cuda()
    w = torch.load(ckpt, map_location=lambda s, l: s, weights_only=False)['gen']
    gen.load_state_dict(w)
    gen.eval()
    return gen


@torch.no_grad()
def exp_code(gen, bgr_crop):
    """BGR uint8 crop -> (1,512) DPE expression latent, exactly as dec_exp consumes it."""
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0).cuda()
    x = (x - 0.5) * 2.0                                   # [-1,1], as img_preprocessing
    wa_t, _ = gen.enc.net_app(x)                          # (1,512)
    alpha = gen.mlp(wa_t)                                 # (1,20)
    directions = gen.dir(alpha)                           # (1,512)
    return gen.mlp_exp(directions).float().cpu()          # (1,512)


def decode_cam(video, out_dir, frames, rgb_w, quality=2):
    """Decode the RGB half of one camera at FULL resolution into out_dir/%08d.jpg."""
    os.makedirs(out_dir, exist_ok=True)
    todo = [f for f in frames if not os.path.exists(os.path.join(out_dir, '%08d.jpg' % f))]
    if not todo:
        return 0
    v = todo
    if len(v) > 1 and all(b - a == v[1] - v[0] for a, b in zip(v, v[1:])):
        step = v[1] - v[0]
        sel = (f'between(n\\,{v[0]}\\,{v[-1]})' if step == 1 else
               f'between(n\\,{v[0]}\\,{v[-1]})*not(mod(n-{v[0]}\\,{step}))')
    else:
        sel = '+'.join(f'eq(n\\,{n})' for n in v)
    tmp = os.path.join(out_dir, '_tmp')
    subprocess.run(['rm', '-rf', tmp], check=False)
    os.makedirs(tmp)
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', video,
                    '-vf', f"select='{sel}',crop={rgb_w}:in_h:0:0", '-vsync', '0',
                    '-q:v', str(quality), os.path.join(tmp, '%08d.jpg')], check=True)
    got = sorted(f for f in os.listdir(tmp) if f.endswith('.jpg'))
    if len(got) != len(todo):
        raise RuntimeError(f'{video}: got {len(got)} frames, wanted {len(todo)}')
    for src, f in zip(got, todo):
        os.replace(os.path.join(tmp, src), os.path.join(out_dir, '%08d.jpg' % f))
    subprocess.run(['rm', '-rf', tmp], check=False)
    return len(todo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--capture', required=True)
    ap.add_argument('--out', required=True, help='dir to hold dpe-multi-faces.zip')
    ap.add_argument('--cams', type=int, nargs='+', default=[7, 30],
                    help='FRONTAL cameras that are in the TRAIN split. Never pass a '
                         'held-out eval camera: its pixels would leak into the face '
                         'conditioning and the evaluation would flatter itself.')
    ap.add_argument('--dpe-root', default=DPE_ROOT,
                    help='clone of https://github.com/OpenTalker/DPE (or set $DPE_ROOT)')
    ap.add_argument('--dpe-ckpt', default=os.path.join(DPE_ROOT, 'checkpoints/dpe.pt'))
    ap.add_argument('--scratch', default=None,
                    help='scratch dir for decoded face frames (default: <out>/_frames)')
    ap.add_argument('--pad', type=int, default=50, help="DPE's crop_video.py uses 50 px")
    ap.add_argument('--ref-frame', type=int, default=None,
                    help='frame the fixed face box is detected on (default: mid-sequence)')
    ap.add_argument('--stride', type=int, default=1)
    a = ap.parse_args()
    if a.scratch is None:
        a.scratch = os.path.join(a.out, '_frames')

    cap_name = os.path.basename(os.path.normpath(a.capture))
    z = np.load(os.path.join(a.capture, 'smplx.npz'), allow_pickle=False)
    frames = [int(f) for f in z['frames'].astype(int)][::a.stride]
    card = json.load(open(os.path.join(a.capture, 'capture.json')))
    rgb_w = int(card['video']['rgb_width'])
    ref = a.ref_frame if a.ref_frame is not None else frames[len(frames) // 2]

    gen = build_generator(a.dpe_ckpt)
    det = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                       flip_input=False, device='cuda')

    os.makedirs(a.out, exist_ok=True)
    members, stats = [], {}
    for cam in a.cams:
        frm_dir = os.path.join(a.scratch, cap_name, f'cam{cam:02d}')
        n_new = decode_cam(os.path.join(a.capture, 'videos', f'cam{cam:02d}.mp4'),
                           frm_dir, frames, rgb_w)
        print(f'[{cap_name} cam{cam:02d}] decoded {n_new} new frames -> {frm_dir}', flush=True)

        # --- fixed face box from the reference frame, DPE-style
        ref_img = cv2.imread(os.path.join(frm_dir, '%08d.jpg' % ref))
        pred = det.get_detections_for_batch(np.array([ref_img[:, :, ::-1]]))
        if pred[0] is None:
            print(f'[{cap_name} cam{cam:02d}] NO FACE on ref frame {ref}; skipping camera')
            continue
        x1, y1, x2, y2 = pred[0]
        H, W = ref_img.shape[:2]
        x1, y1 = max(0, x1 - a.pad), max(0, y1 - a.pad)
        x2, y2 = min(W, x2 + a.pad), min(H, y2 + a.pad)
        print(f'[{cap_name} cam{cam:02d}] fixed box from frame {ref}: '
              f'({x1},{y1})-({x2},{y2})  {x2-x1}x{y2-y1} px', flush=True)

        codes = []
        for f in frames:
            img = cv2.imread(os.path.join(frm_dir, '%08d.jpg' % f))
            if img is None:
                continue
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            c = exp_code(gen, crop)
            fn = os.path.join(a.out, f'dpe-{f:06d}-cam{cam:02d}.pt')
            torch.save({'exp': c}, fn)
            members.append(fn)
            codes.append(c.numpy()[0])
        codes = np.stack(codes)
        stats[f'cam{cam:02d}'] = {
            'box': [int(x1), int(y1), int(x2), int(y2)],
            'n_frames': len(codes),
            'dim': int(codes.shape[1]),
            'per_dim_std_mean': float(codes.std(0).mean()),
            'per_dim_std_max': float(codes.std(0).max()),
            'code_norm_mean': float(np.linalg.norm(codes, axis=1).mean()),
        }
        print(f'[{cap_name} cam{cam:02d}] {len(codes)} codes, dim {codes.shape[1]}, '
              f'per-dim std mean {codes.std(0).mean():.5f}', flush=True)

    zip_fn = os.path.join(a.out, 'dpe-multi-faces.zip')
    with zipfile.ZipFile(zip_fn, 'w', zipfile.ZIP_STORED) as zf:
        for m in members:
            zf.write(m, os.path.basename(m))
    for m in members:
        os.remove(m)
    with open(os.path.join(a.out, 'dpe_meta.json'), 'w') as fp:
        json.dump({'capture': cap_name, 'cams': a.cams, 'ref_frame': ref, 'pad': a.pad,
                   'dim': 512, 'source': 'OpenTalker/DPE mlp_exp(dir(mlp(enc(img))))',
                   'stats': stats}, fp, indent=2)
    print(f'\n[done] {zip_fn}  ({len(members)} members)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
