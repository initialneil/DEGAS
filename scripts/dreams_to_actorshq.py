#!/usr/bin/env python
"""Convert one DREAMS-AVATAR capture into an on-disk **ActorsHQ-format** tree.

Goal: an UNMODIFIED ActorsHQ reader -- DEGAS's own `dataset/actorshq_data.py`, or
Synthesia's `actorshq.dataset.camera_data`, or anything else that speaks the
ActorsHQ layout -- can open the result and train, with no knowledge that the data
came from a video bundle.

    huggingface-cli download initialneil/DREAMS-AVATAR --repo-type dataset \
        --local-dir DREAMS-AVATAR
    cd DREAMS-AVATAR
    python scripts/dreams_to_actorshq.py --capture-dir data/P1C1 --out actorshq/P1C1

Needs only numpy + ffmpeg on PATH (torch too, but only for the optional `.pt`).
Subsetting, for a quick look or a smoke train:

    python scripts/dreams_to_actorshq.py --capture-dir data/P1C1 --out actorshq/P1C1 \
        --scale 2x --stride 4 --cams 0 3 6 9 12 15 18 21 --workers 8

OUTPUT TREE
    <out>/
      <scale>/                       e.g. 1x, 2x, 4x  (ActorsHQ calls this the "scale" dir)
        calibration.csv              ActorsHQ calibration, exact column order
        rgbs/Cam001/Cam001_rgb000000.jpg   ...
        masks/Cam001/Cam001_mask000000.png ...
      smplx_dreams.pt                LOSSLESS SMPL-X, full metadata  <- use this one
      smpl_params.npz                float-array-only compat file    <- see CAVEAT below
      dreams_meta.json               what was emitted, and with which conventions

============================== THE CALIBRATION ==============================
ActorsHQ `calibration.csv` (verified against synthesiaresearch/humanrf
`actorshq/dataset/camera_data.py`) is:

    name,w,h,rx,ry,rz,tx,ty,tz,fx,fy,px,py

  * (rx,ry,rz)  axis-angle of the **camera-to-world** rotation.
                  world = R_c2w @ cam + t          (their docstring, verbatim)
  * (tx,ty,tz)  the **camera centre in world space** -- NOT the w2c translation.
  * (fx,fy)     focal length **normalised**: fx_pixels = fx * w, fy_pixels = fy * h
  * (px,py)     principal point **normalised**: cx_pixels = px * w, cy = py * h

DREAMS-AVATAR ships the DEGAS-native rig calibration instead:
`cameras.json` -> rigs[i].cameras[0] with fx,fy,cx,cy in pixels, `R`, and the camera
centre `c`, in a **Y-down** world.  `capture.json["camera_convention"]` fixes the
mapping to the Y-up world the SMPL-X fit lives in:

    A     = diag(1, -1, -1)                    # world_flip, Y-down -> Y-up
    R_w2c = R_json @ A
    t_w2c = -R_json @ c_json

so the ActorsHQ row for camera i is

    R_c2w = R_w2c.T = A @ R_json.T             (A is symmetric and A@A = I)
    rx,ry,rz = Rodrigues(R_c2w)
    tx,ty,tz = -R_w2c.T @ t_w2c = A @ c_json   # centre, moved into the Y-up world
    fx = fx_px / w ;  fy = fy_px / h ;  px = cx_px / w ;  py = cy_px / h

Because the focal/principal are normalised, the SAME csv row is valid at every
`--scale`; only `w` and `h` change.  That is why the scale dir owns its own csv.

Round-trip sanity: DEGAS's reader does `cam.R = Rodrigues(rvec).T ; cam.c = t`, and
its `libcore.Camera.t` property is `-R @ c`.  So it recovers exactly (R_w2c, t_w2c).
`--verify` re-projects the SMPL-X joints through both paths and asserts they agree.

============================== THE IMAGES ==============================
Each `videos/camNN.mp4` is 4096x1500 and holds two things side by side:
LEFT 2048 = matted RGB, RIGHT 2048 = the alpha matte (binary silhouette).  One
ffmpeg pass per camera per half.  ActorsHQ camera `Cam%03d` is 1-based, so DREAMS
`cam00` becomes `Cam001`.  File numbering is the GT frame id, `%06d`, which is the
same number that indexes `smplx.npz` (frame offset d = 0).

============================== THE SMPL-X ==============================
ActorsHQ itself ships **no** SMPL-X; DEGAS reads a registration file next to the
scale dir, chosen by `dataset.smplx_type`, and supports two formats:

  `.pt`   torch.load -> dict.  Non-tensor entries pass through untouched, so the
          model metadata survives.  **This is the lossless path.**  We write
          `smplx_dreams.pt` with gender/model_type/use_pca/flat_hand_mean/
          num_betas(300)/num_expression_coeffs(100) plus every per-frame tensor.
          Config:  `smplx_type: smplx_dreams.pt`

  `.npz`  `dataset/dataset_utils.py:load_smplx_npz` turns EVERY array into a
          tensor and then indexes anything with shape[0] > 1.  A 0-d scalar or a
          string in the npz therefore crashes it, so an npz can only carry the
          per-frame float arrays.  Two consequences, both handled here:
            1. `flat_hand_mean` defaults to True in that loader, but the DREAMS fit
               is flat_hand_mean=False.  We bake the difference in:
                   hand_pose_npz = hand_pose_dreams + hands_mean
               (smplx does `full_pose += pose_mean`, and pose_mean carries
               hands_meanl/r only when flat_hand_mean=False -- so adding the mean
               makes a flat_hand_mean=True model reproduce the same hands).
               Needs `--smplx-model-dir` (reads hands_meanl/hands_meanr straight
               out of SMPLX_NEUTRAL.npz with numpy; the smplx package is not
               imported).
            2. `num_expression_coeffs` cannot be expressed, so the loader would
               build a 10-coefficient model and choke on our (N,100) expression.
               By default `expression` is OMITTED from the npz -- which is also
               what DEGAS's own ActorsHQ configs effectively do
               (`smplx_nofacial: exp+jaw`).  `--npz-expression-coeffs 10` writes
               the leading 10 instead.  Either way the npz is a COMPAT artifact:
               for full fidelity use the .pt.

Row index == frame id.  `load_smplx_npz` indexes the arrays with the raw frame
numbers from `frm_list`, so the arrays are emitted at full length
(max(frame)+1 rows) and rows for frames without a fit are filled with the nearest
preceding fit.  `dreams_meta.json["smplx"]["valid_rows"]` records which are real.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------
# Self-contained copies of the two conventions, so this script can be dropped into
# the published dataset repo and run with nothing but numpy + ffmpeg on PATH.
WORLD_FLIP = np.diag([1.0, -1.0, -1.0])   # cameras.json is Y-down; SMPL-X world is Y-up
VIDEO_FRAME_OFFSET = 0                    # measured per capture by verify_alignment.py

PARAM_KEYS = ("global_orient", "body_pose", "jaw_pose", "leye_pose", "reye_pose",
              "left_hand_pose", "right_hand_pose", "betas", "expression", "transl")

# per-frame arrays that go into the ActorsHQ-compat npz (expression handled separately)
NPZ_KEYS = ("global_orient", "body_pose", "jaw_pose", "leye_pose", "reye_pose",
            "left_hand_pose", "right_hand_pose", "betas", "transl")

CALIB_HEADER = ["name", "w", "h", "rx", "ry", "rz", "tx", "ty", "tz", "fx", "fy", "px", "py"]

SCALES = {"1x": 1, "2x": 2, "4x": 4}


# --------------------------------------------------------------------------- cameras
def load_dreams_cameras(cameras_json: Path) -> list[dict]:
    """cameras.json -> [{K, R_w2c, t_w2c, w, h, info}, ...]; index i == videos/cam{i:02d}.mp4."""
    d = json.loads(cameras_json.read_text())
    out = []
    for rig in d["rigs"]:
        c = rig["cameras"][0]
        K = np.array([[c["fx"], 0.0, c["cx"]],
                      [0.0, c["fy"], c["cy"]],
                      [0.0, 0.0, 1.0]], np.float64)
        R = np.asarray(c["R"], np.float64).reshape(3, 3)
        C = np.asarray(c["c"], np.float64).reshape(3)
        out.append({
            "K": K,
            "R_w2c": R @ WORLD_FLIP,
            "t_w2c": -R @ C,
            "w": int(c["w"]),
            "h": int(c["h"]),
            "info": str(c.get("info", "")),
        })
    return out


def orthonormalize(R: np.ndarray) -> np.ndarray:
    """Nearest true rotation to R (SVD). cameras.json rows are ~1e-7 off orthonormal, and
    axis-angle can only represent an exact rotation, so project before converting."""
    U, _, Vt = np.linalg.svd(R)
    S = np.eye(3)
    S[2, 2] = np.sign(np.linalg.det(U @ Vt))
    return U @ S @ Vt


def rodrigues(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> axis-angle vector, via a quaternion.

    NOT the textbook `axis = (R - R.T) / (2 sin th)`: that divides by sin(th), so it
    bleeds precision as th approaches pi, and several of these rigs sit there. Measured
    on P1C1 the naive form round-tripped to only 4e-3 (worst camera, and cv2.Rodrigues
    is no better); Shepperd's branch-on-the-largest-diagonal quaternion form round-trips
    to ~1e-15 everywhere.
    """
    R = orthonormalize(np.asarray(R, np.float64))
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw, qx, qy, qz = 0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw, qx, qy, qz = (m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw, qx, qy, qz = (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw, qx, qy, qz = (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s

    q = np.array([qw, qx, qy, qz], np.float64)
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q                      # shortest arc, so |theta| <= pi
    v = q[1:]
    nv = float(np.linalg.norm(v))
    if nv < 1e-15:
        return np.zeros(3)
    theta = 2.0 * np.arctan2(nv, q[0])
    return v / nv * theta


def rodrigues_inv(rvec: np.ndarray) -> np.ndarray:
    """Axis-angle -> rotation matrix (only used by --verify)."""
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-12:
        return np.eye(3)
    k = rvec / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def actorshq_rows(cams: list[dict], cam_ids: list[int], down: int) -> list[list]:
    """ActorsHQ calibration.csv rows, at 1/`down` resolution.

    ALWAYS every camera, even when only a subset is decoded. ActorsHQ readers derive the
    image folder from the ROW INDEX -- DEGAS's is literally
    `cam_sn = 'Cam%03d' % (cam_id + 1)` over `range(len(cams))` -- and ignore the `name`
    column. A csv holding only the decoded cameras would therefore silently shift every
    camera onto the wrong images. Keeping it dense makes row i == DREAMS cam{i:02d} ==
    Cam{i+1:03d}, so a config's 1-based `cam_select` means what it looks like it means.
    Rows for cameras that were not decoded are still correct calibration; the reader only
    touches the ones `cam_select` asks for.
    """
    rows = []
    for i in range(len(cams)):
        c = cams[i]
        R_c2w = c["R_w2c"].T
        centre = -c["R_w2c"].T @ c["t_w2c"]
        rvec = rodrigues(R_c2w)
        w, h = c["w"] // down, c["h"] // down
        rows.append([
            "Cam%03d" % (i + 1), w, h,
            rvec[0], rvec[1], rvec[2],
            centre[0], centre[1], centre[2],
            # normalised -- scale-invariant, so `down` never enters here
            c["K"][0, 0] / c["w"], c["K"][1, 1] / c["h"],
            c["K"][0, 2] / c["w"], c["K"][1, 2] / c["h"],
        ])
    return rows


def write_calibration(rows: list[list], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(CALIB_HEADER)
        for r in rows:
            w.writerow(r)
    print(f"[calib] {path} ({len(rows)} cameras)", flush=True)


def verify_calibration(cams: list[dict], rows: list[list],
                       joints: np.ndarray, down: int) -> float:
    """Re-project `joints` through cameras.json and through the emitted csv; max |du,dv|."""
    worst = 0.0
    for i, row in enumerate(rows):
        c = cams[i]
        # native path
        x = joints @ c["R_w2c"].T + c["t_w2c"]
        uv_native = (x @ c["K"].T)[:, :2] / x[:, 2:3] / down
        # ActorsHQ path, replayed exactly as DEGAS's reader does it
        rvec = np.array(row[3:6], float)
        R_w2c = rodrigues_inv(rvec).T
        centre = np.array(row[6:9], float)
        t_w2c = -R_w2c @ centre
        w, h = int(row[1]), int(row[2])
        K = np.array([[row[9] * w, 0, row[11] * w],
                      [0, row[10] * h, row[12] * h],
                      [0, 0, 1.0]])
        x2 = joints @ R_w2c.T + t_w2c
        uv_ahq = (x2 @ K.T)[:, :2] / x2[:, 2:3]
        worst = max(worst, float(np.abs(uv_native - uv_ahq).max()))
    return worst


# ---------------------------------------------------------------------------- images
def _select_expr(frames: list[int]) -> str:
    """ffmpeg select expression for a video-frame list; compact for arithmetic runs."""
    v = [f + VIDEO_FRAME_OFFSET for f in frames]
    if len(v) > 1:
        step = v[1] - v[0]
        if step > 0 and all(b - a == step for a, b in zip(v, v[1:])):
            if step == 1:
                return f"between(n\\,{v[0]}\\,{v[-1]})"
            return f"between(n\\,{v[0]}\\,{v[-1]})*not(mod(n-{v[0]}\\,{step}))"
    return "+".join(f"eq(n\\,{n})" for n in v)


def _decode_cam(video: Path, out_root: Path, cam_id: int, frames: list[int],
                rgb_w: int, rgb_h: int, down: int, quality: int, masks: bool) -> tuple[str, int]:
    """Two ffmpeg passes for one camera: LEFT half -> rgbs/, RIGHT half -> masks/."""
    sn = "Cam%03d" % (cam_id + 1)
    sel = _select_expr(frames)
    w, h = rgb_w // down, rgb_h // down
    scale = "" if down == 1 else f",scale={w}:{h}:flags=area"

    img_dir = out_root / "rgbs" / sn
    img_dir.mkdir(parents=True, exist_ok=True)
    tmp = img_dir / "_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-threads", "1", "-i", str(video),
         "-vf", f"select='{sel}',crop={rgb_w}:in_h:0:0{scale}", "-vsync", "0",
         "-q:v", str(quality), str(tmp / "%08d.jpg")],
        check=True)
    got = sorted(tmp.glob("*.jpg"))
    if len(got) != len(frames):
        raise RuntimeError(f"{sn}: ffmpeg produced {len(got)} rgb frames, wanted {len(frames)}")
    for src, f in zip(got, frames):
        src.rename(img_dir / f"{sn}_rgb{f:06d}.jpg")
    shutil.rmtree(tmp)

    if masks:
        msk_dir = out_root / "masks" / sn
        msk_dir.mkdir(parents=True, exist_ok=True)
        tmp.mkdir()
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-threads", "1", "-i", str(video),
             "-vf", f"select='{sel}',crop={rgb_w}:in_h:{rgb_w}:0{scale},format=gray",
             "-vsync", "0", "-pix_fmt", "gray", str(tmp / "%08d.png")],
            check=True)
        got = sorted(tmp.glob("*.png"))
        if len(got) != len(frames):
            raise RuntimeError(f"{sn}: ffmpeg produced {len(got)} masks, wanted {len(frames)}")
        for src, f in zip(got, frames):
            src.rename(msk_dir / f"{sn}_mask{f:06d}.png")
        shutil.rmtree(tmp)

    return sn, len(frames)


# ---------------------------------------------------------------------------- smplx
def _dense(arr: np.ndarray, frames: np.ndarray, n_rows: int) -> np.ndarray:
    """Scatter (T,...) fit rows to (n_rows,...) indexed by frame id; gaps hold-forward."""
    out = np.zeros((n_rows, *arr.shape[1:]), arr.dtype)
    filled = np.zeros(n_rows, bool)
    out[frames] = arr
    filled[frames] = True
    last = None
    for i in range(n_rows):
        if filled[i]:
            last = i
        elif last is not None:
            out[i] = out[last]
    if last is None:
        return out
    first = int(frames.min())
    out[:first] = out[first]
    return out


def read_hands_mean(model_dir: Path, gender: str = "neutral") -> tuple[np.ndarray, np.ndarray]:
    """(hands_meanl, hands_meanr) straight out of SMPLX_<GENDER>.npz -- no smplx import."""
    fn = model_dir / f"SMPLX_{gender.upper()}.npz"
    if not fn.exists():
        fn = model_dir / "smplx" / f"SMPLX_{gender.upper()}.npz"
    if not fn.exists():
        raise FileNotFoundError(f"no SMPLX_{gender.upper()}.npz under {model_dir}")
    z = np.load(fn, allow_pickle=True)
    return (np.asarray(z["hands_meanl"], np.float64).reshape(-1),
            np.asarray(z["hands_meanr"], np.float64).reshape(-1))


def write_smplx(z: dict, frames: np.ndarray, out: Path, kwargs: dict,
                smplx_model_dir: Path | None, npz_expr_coeffs: int) -> dict:
    """Write smplx_dreams.pt (lossless) and smpl_params.npz (ActorsHQ compat)."""
    import torch

    n_rows = int(frames.max()) + 1
    note = {}

    # ---- lossless .pt: tensors + the metadata SMPLXOptimizer.init_keys asks for
    pt = {k: torch.from_numpy(_dense(z[k], frames, n_rows).astype(np.float32))
          for k in PARAM_KEYS if k in z}
    pt.update({
        "gender": str(kwargs.get("gender", "neutral")),
        "model_type": str(kwargs.get("model_type", "smplx")),
        "use_pca": bool(kwargs.get("use_pca", False)),
        "flat_hand_mean": bool(kwargs.get("flat_hand_mean", False)),
        "num_betas": int(z["betas"].shape[-1]),
        "num_expression_coeffs": int(z["expression"].shape[-1]) if "expression" in z else 10,
    })
    pt_fn = out / "smplx_dreams.pt"
    torch.save(pt, pt_fn)
    print(f"[smplx] {pt_fn}  ({n_rows} rows, lossless, flat_hand_mean="
          f"{pt['flat_hand_mean']}, num_betas={pt['num_betas']}, "
          f"num_expression_coeffs={pt['num_expression_coeffs']})", flush=True)
    note["pt"] = pt_fn.name

    # ---- ActorsHQ-compat .npz: float arrays only
    npz = {k: _dense(z[k], frames, n_rows).astype(np.float32) for k in NPZ_KEYS if k in z}

    if not bool(kwargs.get("flat_hand_mean", False)):
        if smplx_model_dir is None:
            note["npz_hands"] = "RAW -- no --smplx-model-dir given, hands are WRONG for a " \
                                "flat_hand_mean=True reader"
            print("[smplx][WARN] no --smplx-model-dir: smpl_params.npz hand poses are NOT "
                  "converted to the flat_hand_mean=True convention that load_smplx_npz "
                  "assumes. Use smplx_dreams.pt, or re-run with --smplx-model-dir.",
                  flush=True)
        else:
            ml, mr = read_hands_mean(Path(smplx_model_dir), str(kwargs.get("gender", "neutral")))
            npz["left_hand_pose"] = (npz["left_hand_pose"] + ml).astype(np.float32)
            npz["right_hand_pose"] = (npz["right_hand_pose"] + mr).astype(np.float32)
            note["npz_hands"] = "hands_mean folded in (flat_hand_mean False -> True)"
    else:
        note["npz_hands"] = "already flat_hand_mean=True"

    if npz_expr_coeffs > 0 and "expression" in z:
        npz["expression"] = _dense(z["expression"], frames, n_rows)[:, :npz_expr_coeffs] \
            .astype(np.float32)
        note["npz_expression"] = f"first {npz_expr_coeffs} coefficients"
    else:
        note["npz_expression"] = "omitted (load_smplx_npz cannot carry " \
                                 "num_expression_coeffs; the model would default to 10)"

    npz_fn = out / "smpl_params.npz"
    np.savez(npz_fn, **npz)
    print(f"[smplx] {npz_fn}  (compat: {note['npz_hands']}; expression: "
          f"{note['npz_expression']})", flush=True)
    note["npz"] = npz_fn.name
    note["n_rows"] = n_rows
    note["valid_rows"] = [int(frames.min()), int(frames.max())]
    return note


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture_dir_pos", type=Path, nargs="?", default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--capture-dir", type=Path, default=None,
                    help="DREAMS-AVATAR capture dir, e.g. data/P1C1 (may also be positional)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scale", default="2x", choices=sorted(SCALES),
                    help="ActorsHQ scale dir; 2x halves 2048x1500 to 1024x750 (default 2x)")
    ap.add_argument("--cams", type=int, nargs="*", default=None, help="0-based DREAMS cam ids")
    ap.add_argument("--frames", type=int, nargs=2, metavar=("START", "END"), default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--quality", type=int, default=2, help="ffmpeg -q:v for the jpgs")
    ap.add_argument("--no-masks", action="store_true")
    ap.add_argument("--no-images", action="store_true", help="calibration + SMPL-X only")
    ap.add_argument("--smplx-model-dir", type=Path, default=None,
                    help="dir holding SMPLX_NEUTRAL.npz; needed to fold hands_mean into "
                         "smpl_params.npz")
    ap.add_argument("--npz-expression-coeffs", type=int, default=0,
                    help="how many expression coefficients to put in smpl_params.npz "
                         "(0 = omit; >10 will break an unmodified reader)")
    ap.add_argument("--verify", action="store_true",
                    help="re-project SMPL-X joints through cameras.json and through the "
                         "emitted csv and report the worst pixel disagreement")
    ap.add_argument("--verify-tol", type=float, default=1e-3,
                    help="px; the floor is cameras.json's ~1e-7 orthonormality defect, "
                         "which axis-angle has to project away (default 1e-3)")
    a = ap.parse_args()

    cap = a.capture_dir if a.capture_dir is not None else a.capture_dir_pos
    if cap is None:
        raise SystemExit("give the capture dir, either positionally or with --capture-dir")
    for need in ("cameras.json", "smplx.npz"):
        if not (cap / need).exists():
            raise SystemExit(f"{cap} is not a DREAMS-AVATAR capture dir (no {need}). "
                             f"After `huggingface-cli download initialneil/DREAMS-AVATAR "
                             f"--repo-type dataset --local-dir DREAMS-AVATAR` the captures "
                             f"live at DREAMS-AVATAR/data/<PxCy>/.")
    down = SCALES[a.scale]
    out = a.out
    scale_dir = out / a.scale
    scale_dir.mkdir(parents=True, exist_ok=True)

    cams = load_dreams_cameras(cap / "cameras.json")
    card = json.loads((cap / "capture.json").read_text()) if (cap / "capture.json").exists() else {}

    z = dict(np.load(cap / "smplx.npz", allow_pickle=False))
    frames_all = z["frames"].astype(int)
    kwargs = json.loads(str(z["smplx_kwargs"])) if "smplx_kwargs" in z else {}

    cam_ids = a.cams if a.cams is not None else list(range(len(cams)))
    bad = [i for i in cam_ids if not (0 <= i < len(cams))]
    if bad:
        raise SystemExit(f"camera ids out of range: {bad} (capture has {len(cams)})")

    frames = frames_all
    if a.frames is not None:
        frames = frames[(frames >= a.frames[0]) & (frames <= a.frames[1])]
    frames = frames[::a.stride]
    frame_list = [int(f) for f in frames]
    if not frame_list:
        raise SystemExit("no frames selected")

    print(f"[dreams->actorshq] {cap.name}: {len(cam_ids)} cams x {len(frame_list)} frames "
          f"-> {out} (scale {a.scale}, {cams[0]['w'] // down}x{cams[0]['h'] // down})",
          flush=True)

    rows = actorshq_rows(cams, cam_ids, down)
    write_calibration(rows, scale_dir / "calibration.csv")

    reproj_err = None
    if a.verify:
        j = z["joints"][int(np.searchsorted(frames_all, frame_list[len(frame_list) // 2]))]
        reproj_err = verify_calibration(cams, rows, j.astype(np.float64), down)
        print(f"[verify] cameras.json vs calibration.csv reprojection: "
              f"max |delta| = {reproj_err:.3e} px  (tol {a.verify_tol:g})", flush=True)
        if reproj_err > a.verify_tol:
            raise SystemExit(f"calibration round-trip failed ({reproj_err:.3e} px)")

    smplx_note = write_smplx(z, frames_all, out, kwargs, a.smplx_model_dir,
                             a.npz_expression_coeffs)

    n_done = 0
    if not a.no_images:
        jobs = []
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for i in cam_ids:
                jobs.append(ex.submit(_decode_cam, cap / "videos" / f"cam{i:02d}.mp4",
                                      scale_dir, i, frame_list,
                                      cams[i]["w"], cams[i]["h"], down,
                                      a.quality, not a.no_masks))
            for fut in as_completed(jobs):
                sn, n = fut.result()
                n_done += 1
                print(f"[decode] {sn}: {n} frames  ({n_done}/{len(cam_ids)})", flush=True)

    meta = {
        "source_capture": str(cap),
        "capture": cap.name,
        "format": "ActorsHQ",
        "scale": a.scale,
        "downscale": down,
        "width": cams[0]["w"] // down,
        "height": cams[0]["h"] // down,
        "n_cameras_in_calibration": len(cams),
        "cam_ids_decoded_dreams": cam_ids,
        "cam_names_decoded_actorshq": ["Cam%03d" % (i + 1) for i in cam_ids],
        "cam_select_hint_1based": [i + 1 for i in cam_ids],
        "frames": {"n": len(frame_list), "first": frame_list[0], "last": frame_list[-1],
                   "stride": a.stride},
        "frame_convention": card.get("frame_convention", {"offset_d": VIDEO_FRAME_OFFSET}),
        "calibration": {
            "file": f"{a.scale}/calibration.csv",
            "columns": CALIB_HEADER,
            "rotation": "axis-angle of R_cam2world",
            "translation": "camera centre in world space",
            "focal": "normalised by (w, h)",
            "principal_point": "normalised by (w, h)",
            "world": "Y-up (cameras.json world_flip diag(1,-1,-1) applied)",
            "reprojection_max_px_vs_cameras_json": reproj_err,
        },
        "images": {
            "rgb": f"{a.scale}/rgbs/Cam%03d/Cam%03d_rgb%06d.jpg",
            "mask": None if a.no_masks else f"{a.scale}/masks/Cam%03d/Cam%03d_mask%06d.png",
            "note": "rgb is the LEFT half of camNN.mp4, mask is the RIGHT half (alpha matte)",
        },
        "smplx": {**smplx_note, "smplx_kwargs": kwargs},
        "degas_config_hint": {
            "frameset_type": "actorshq",
            "scale": a.scale,
            "smplx_type": "smplx_dreams.pt",
            "resolution": 1,
        },
    }
    (out / "dreams_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[done] {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
