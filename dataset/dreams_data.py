"""DREAMS-AVATAR dataset reader -- `frameset_type: dreams`.

Reads the PUBLISHED DREAMS-AVATAR bundle in its native form:

    <dat_dir>/
      cameras.json          DEGAS rig calibration, nested rigs[i].cameras[0], Y-DOWN world
      capture.json          the conventions, written down (camera_convention/frame_convention)
      smplx.npz             every SMPL-X param stacked over time, plus `frames`
      videos/camNN.mp4      4096x1500: LEFT 2048 = matted RGB, RIGHT 2048 = alpha matte

and hands the rest of DEGAS exactly what `dataset/actorshq_data.py` hands it: a
`scene_cameras` list built by `convert_to_scene_cameras`, plus `smplx_params`.
Nothing downstream of `__getitem__` can tell the two apart, so the trainer is
unchanged.

WHY A DEDICATED READER AND NOT JUST THE CONVERTER
    `scripts/dreams_to_actorshq.py` writes a full ActorsHQ tree, which is the right
    answer for other codebases. Inside DEGAS this reader is nicer: no CSV round
    trip (the camera math stays in double precision, straight off cameras.json),
    no lossy SMPL-X compat step (`flat_hand_mean=False` and 300/100 betas/expression
    survive), and the mp4s are decoded lazily -- only the cameras and frames a split
    actually asks for.

DECODE CACHE
    mp4 seeking per sample would be ruinous, so the frames a split needs are decoded
    ONCE, on construction, into

        <cache_dir>/<scale>/rgb/cam00/00000000.jpg
        <cache_dir>/<scale>/mask/cam00/00000000.png

    `cache_dir` defaults to `<dat_dir>/_degas_cache`. Existing files are kept, so a
    second split, a resumed run, or a widened `frm_list` only decodes the delta. Set
    `cache_dir` to a scratch disk if the capture lives on slow storage.

CONVENTIONS (from capture.json, not guessed)
    camera:  K     = [[fx,0,cx],[0,fy,cy],[0,0,1]]   from rigs[i].cameras[0]
             R_w2c = R @ diag(1,-1,-1)               world_flip: cameras.json is Y-down
             t_w2c = -R @ c                          c = centre in the unflipped world
             so libcore.Camera gets R = R_w2c and c = -R_w2c.T @ t_w2c (Y-up centre),
             which is what `framesetToCameraInfo` expects.
    frame:   d = 0. video frame index == smplx.npz["frames"] entry == GT frame id.
             Measured per capture by the dataset's own verify_alignment.py; this
             reader reads `capture.json["frame_convention"]["offset_d"]` and does not
             assume it.

CONFIG
    dataset:
      dat_dir: /path/to/DREAMS-AVATAR/data/P1C1
      frameset_type: dreams
      scale: 2x                 # 1x = 2048x1500, 2x = 1024x750, 4x = 512x375
      resolution: 1             # keep 1: DEGAS's loadCam resizes the image but NOT
                                # the intrinsics, so all downscaling happens here
      smplx_nofacial: exp+jaw   # optional, same semantics as the ActorsHQ path
      train: {frm_list: ..., cam_select: [...], mini_batch: 1}
      val:   {frm_list: [...], cam_select: [...]}
      test:  {frm_list: ..., cam_select: [...]}

    `cam_select` is 0-BASED here (cam00..cam31), unlike the ActorsHQ path which is
    1-based because ActorsHQ names its cameras Cam001..; the config key is the same
    but the numbering follows whatever the format itself uses.
"""
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

from scene.dataset_readers import convert_to_scene_cameras
from model import libcore
from .dataset_utils import AvatarDataset, reset_smplx_facial

WORLD_FLIP = np.diag([1.0, -1.0, -1.0])

PARAM_KEYS = ("global_orient", "body_pose", "jaw_pose", "leye_pose", "reye_pose",
              "left_hand_pose", "right_hand_pose", "betas", "expression", "transl")

SCALES = {'1x': 1, '2x': 2, '4x': 4}


##################################################
# cameras
def read_DREAMS_cameras(cameras_json, down=1):
    """cameras.json -> [libcore.Camera]; index i == videos/cam{i:02d}.mp4.

    The world_flip is applied here, so every camera and the SMPL-X fit share one
    Y-up world and nothing downstream needs to know about the DEGAS calibration quirk.
    """
    with open(cameras_json, 'r') as fp:
        d = json.load(fp)

    cams = []
    for i, rig in enumerate(d['rigs']):
        c = rig['cameras'][0]
        R = np.asarray(c['R'], np.float64).reshape(3, 3)
        C = np.asarray(c['c'], np.float64).reshape(3)

        R_w2c = R @ WORLD_FLIP
        t_w2c = -R @ C

        cam = libcore.Camera()
        cam.info = str(c.get('info', 'cam%02d' % i))
        cam.R = R_w2c
        cam.c = -R_w2c.T @ t_w2c          # camera centre in the Y-up world
        cam.w = int(c['w'])
        cam.h = int(c['h'])
        cam.fx = float(c['fx'])
        cam.fy = float(c['fy'])
        cam.cx = float(c['cx'])
        cam.cy = float(c['cy'])
        if down != 1:
            # scale intrinsics WITH the images: DEGAS's loadCam does not do this for us
            cam.scaleIntrinsics(cam.w // down, cam.h // down)
        cams.append(cam)
    return cams


##################################################
# decode cache
def _select_expr(video_frames):
    """ffmpeg select expression; compact form when the frames are an arithmetic run."""
    v = list(video_frames)
    if len(v) > 1:
        step = v[1] - v[0]
        if step > 0 and all(b - a == step for a, b in zip(v, v[1:])):
            if step == 1:
                return 'between(n\\,%d\\,%d)' % (v[0], v[-1])
            return 'between(n\\,%d\\,%d)*not(mod(n-%d\\,%d))' % (v[0], v[-1], v[0], step)
    return '+'.join('eq(n\\,%d)' % n for n in v)


def _decode_one_cam(args):
    """Decode the missing frames of ONE camera into the cache. Returns (cam, n_new)."""
    (video, rgb_dir, msk_dir, frames, offset, rgb_w, rgb_h, down, quality) = args

    todo = [f for f in frames
            if not (os.path.exists(os.path.join(rgb_dir, '%08d.jpg' % f)) and
                    os.path.exists(os.path.join(msk_dir, '%08d.png' % f)))]
    if not todo:
        return os.path.basename(rgb_dir), 0

    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(msk_dir, exist_ok=True)
    sel = _select_expr([f + offset for f in todo])
    w, h = rgb_w // down, rgb_h // down
    scale = '' if down == 1 else ',scale=%d:%d:flags=area' % (w, h)

    tmp = os.path.join(rgb_dir, '_tmp')
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    subprocess.run(
        ['ffmpeg', '-y', '-v', 'error', '-threads', '1', '-i', video,
         '-vf', "select='%s',crop=%d:in_h:0:0%s" % (sel, rgb_w, scale), '-vsync', '0',
         '-q:v', str(quality), os.path.join(tmp, '%08d.jpg')],
        check=True)
    got = sorted(f for f in os.listdir(tmp) if f.endswith('.jpg'))
    if len(got) != len(todo):
        raise RuntimeError('%s: ffmpeg produced %d rgb frames, wanted %d'
                           % (video, len(got), len(todo)))
    for src, f in zip(got, todo):
        os.replace(os.path.join(tmp, src), os.path.join(rgb_dir, '%08d.jpg' % f))
    shutil.rmtree(tmp, ignore_errors=True)

    os.makedirs(tmp)
    subprocess.run(
        ['ffmpeg', '-y', '-v', 'error', '-threads', '1', '-i', video,
         '-vf', "select='%s',crop=%d:in_h:%d:0%s,format=gray" % (sel, rgb_w, rgb_w, scale),
         '-vsync', '0', '-pix_fmt', 'gray', os.path.join(tmp, '%08d.png')],
        check=True)
    got = sorted(f for f in os.listdir(tmp) if f.endswith('.png'))
    if len(got) != len(todo):
        raise RuntimeError('%s: ffmpeg produced %d masks, wanted %d'
                           % (video, len(got), len(todo)))
    for src, f in zip(got, todo):
        os.replace(os.path.join(tmp, src), os.path.join(msk_dir, '%08d.png' % f))
    shutil.rmtree(tmp, ignore_errors=True)

    return os.path.basename(rgb_dir), len(todo)


##################################################
def read_DREAMS_frameset(cache_dir, frm_idx, cams, cam_ids, cam_select=None, mini_batch=0):
    """One frame, many cameras -> (libcore.DataVec of RGBA images, positions).

    Mirrors `dataset/actorshq_data.py:read_ActorsHQ_frameset` exactly: RGB and mask are
    read separately and concatenated into a 4-channel BGRA image, which
    `convert_to_scene_cameras` then turns into the RGBA tensor DEGAS trains on.

    The second return value is POSITIONS within `cams`/`cam_ids`, because that is what
    `DataVec.toSubSet` needs. `__getitem__` maps them back to capture-level camera ids
    before putting them in the batch -- the ActorsHQ path puts capture-level ids there
    (`loss_base.py:302` uses them to name validation renders), so returning subset
    positions would silently mislabel every render `cam0000`.
    """
    color_frames, color_masks = libcore.DataVec(), libcore.DataVec()
    color_frames.cams = []
    color_frames.images_path = []
    color_masks.images_path = []
    for k, cam_id in enumerate(cam_ids):
        sn = 'cam%02d' % cam_id
        color_frames.cams.append(cams[k])
        color_frames.images_path.append(
            os.path.join(cache_dir, 'rgb', sn, '%08d.jpg' % frm_idx))
        color_masks.images_path.append(
            os.path.join(cache_dir, 'mask', sn, '%08d.png' % frm_idx))
    color_masks.cams = color_frames.cams

    if cam_select is None:
        cam_idxs = np.arange(color_frames.size).tolist()
    else:
        cam_idxs = list(cam_select)

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
        if color is None or mask is None:
            raise RuntimeError('[DREAMS] missing cache file: %s / %s'
                               % (color_frames.images_path[i], color_masks.images_path[i]))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        img = np.concatenate([color, mask[:, :, None]], axis=-1)
        color_frames.frames[i] = img
    color_frames.image_formats = ['RGB' for _ in range(color_frames.size)]

    return color_frames, cam_idxs


##################################################
class DREAMSDataset(AvatarDataset):
    def __init__(self, config, split='train', frm_list=None):
        self.config = config
        self.split = split
        self.dat_dir = config.dat_dir
        self.scale = config.get('scale', '2x')
        if self.scale not in SCALES:
            raise ValueError('[DREAMS] scale must be one of %s, got %r'
                             % (sorted(SCALES), self.scale))
        self.down = SCALES[self.scale]

        # ---------- the capture's own declaration of its conventions
        card_fn = os.path.join(self.dat_dir, 'capture.json')
        self.card = json.load(open(card_fn, 'r')) if os.path.exists(card_fn) else {}
        self.frame_offset = int(self.card.get('frame_convention', {}).get('offset_d', 0))

        z = np.load(os.path.join(self.dat_dir, 'smplx.npz'), allow_pickle=False)
        self.smplx_npz = {k: z[k] for k in z.files}
        self.all_frames = self.smplx_npz['frames'].astype(int)
        self.frame_to_row = {int(f): i for i, f in enumerate(self.all_frames)}
        self.smplx_kwargs = json.loads(str(self.smplx_npz['smplx_kwargs'])) \
            if 'smplx_kwargs' in self.smplx_npz else {}

        # ---------- frames
        if frm_list is None:
            frm_list = config.get(split, {}).get('frm_list', None)
            if isinstance(frm_list, str):
                frm_list = eval(frm_list)
            if frm_list is None:
                frm_list = [int(f) for f in self.all_frames]
        self.frm_list = [int(f) for f in frm_list]
        missing = [f for f in self.frm_list if f not in self.frame_to_row]
        if missing:
            raise KeyError('[DREAMS] no SMPL-X fit for frames %s (have %d..%d)'
                           % (missing[:5], self.all_frames[0], self.all_frames[-1]))
        self.num_frames = len(self.frm_list)

        # ---------- cameras (0-based ids, matching videos/camNN.mp4)
        self.all_cams = read_DREAMS_cameras(os.path.join(self.dat_dir, 'cameras.json'),
                                            down=self.down)
        cam_select = config.get(split, {}).get('cam_select', None)
        if isinstance(cam_select, str):
            cam_select = eval(cam_select)
        self.cam_ids = list(range(len(self.all_cams))) if cam_select is None \
            else [int(i) for i in cam_select]
        bad = [i for i in self.cam_ids if not (0 <= i < len(self.all_cams))]
        if bad:
            raise ValueError('[DREAMS] cam_select out of range: %s (capture has %d cameras; '
                             'note this reader is 0-based)' % (bad, len(self.all_cams)))
        self.cams = [self.all_cams[i] for i in self.cam_ids]

        # after subsetting, __getitem__ selects positionally within self.cams
        self.cam_select = None
        self.mini_batch = config.get(split, {}).get('mini_batch',
                                                    config.get('mini_batch', 0))
        self.cameras_extent = config.get('cameras_extent', 2.0)
        self.smplx_forward_transl = config.get('smplx_forward_transl', False)

        print('[DREAMSDataset][%s] %s: %d cams %s, %d frames, %dx%d (scale %s), d=%+d'
              % (split, os.path.basename(os.path.normpath(self.dat_dir)),
                 len(self.cams), self.cam_ids[:8], self.num_frames,
                 self.cams[0].w, self.cams[0].h, self.scale, self.frame_offset))

        # ---------- decode cache
        self.cache_dir = config.get('cache_dir', None) or \
            os.path.join(self.dat_dir, '_degas_cache')
        if not os.path.isabs(self.cache_dir):
            self.cache_dir = os.path.join(self.dat_dir, self.cache_dir)
        self.cache_dir = os.path.join(self.cache_dir, self.scale)
        self._guard_cache_owner()
        if config.get('build_cache', True):
            self.build_cache(workers=int(config.get('cache_workers', 8)),
                             quality=int(config.get('jpeg_quality', 2)))

        # ---------- smplx
        self.load_all_smplx()

        self.exp_codes = None
        if config.get('with_face_dpe', None):
            self.load_face_dpe(config.with_face_dpe)

        self.pca = None
        if split == 'train' and config.get('num_pca_comps', 0) > 0:
            self.construct_pca()

    ##################################################
    def _guard_cache_owner(self):
        """Refuse a cache dir that was filled from a DIFFERENT capture.

        Cached frames are named `cam%02d/%08d.jpg` -- capture-agnostic. So pointing two
        captures at one `cache_dir` makes P1C2 read P1C1's pixels with no error anywhere:
        the files exist, the shapes match, the metrics come out plausible. It is a very
        easy mistake to make, because DEGAS re-loads a run's saved `config.yaml` (which
        carries the TRAIN capture's cache_dir) when it evaluates a different `--dat_dir`.
        One stamp file turns that silent corruption into a loud failure.
        """
        name = os.path.basename(os.path.normpath(self.dat_dir))
        stamp = os.path.join(self.cache_dir, 'capture.txt')
        if os.path.exists(stamp):
            with open(stamp, 'r') as fp:
                owner = fp.read().strip()
            if owner != name:
                raise RuntimeError(
                    '[DREAMS] cache dir %s was built from capture %r but this dataset is '
                    '%r. Cached frames are named cam%%02d/%%08d.jpg, so reusing it would '
                    'silently feed you the wrong capture. Set dataset.cache_dir to a path '
                    'of its own (e.g. .../%s).' % (self.cache_dir, owner, name, name))
        else:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(stamp, 'w') as fp:
                fp.write(name + '\n')

    ##################################################
    def build_cache(self, workers=8, quality=2):
        """Decode every (camera, frame) this split needs. Idempotent: skips what exists."""
        jobs = []
        for cam_id in self.cam_ids:
            sn = 'cam%02d' % cam_id
            jobs.append((
                os.path.join(self.dat_dir, 'videos', '%s.mp4' % sn),
                os.path.join(self.cache_dir, 'rgb', sn),
                os.path.join(self.cache_dir, 'mask', sn),
                self.frm_list, self.frame_offset,
                self.cams[0].w * self.down, self.cams[0].h * self.down,
                self.down, quality,
            ))

        n_new = 0
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            for sn, n in ex.map(_decode_one_cam, jobs):
                n_new += n
        if n_new:
            print('[DREAMSDataset][%s] decoded %d new frames into %s'
                  % (self.split, n_new, self.cache_dir))
        else:
            print('[DREAMSDataset][%s] decode cache complete (%s)'
                  % (self.split, self.cache_dir))

    ##################################################
    def load_all_smplx(self, fn=None):
        """Build `self.smplx_params` straight from smplx.npz -- no lossy round trip.

        The metadata that `SMPLXOptimizer.init_keys` reads (gender, model_type, use_pca,
        flat_hand_mean, num_betas, num_expression_coeffs) comes from the capture's own
        `smplx_kwargs`, so `flat_hand_mean=False` and the 300/100 betas/expression sizes
        are carried through as fitted rather than defaulted.
        """
        rows = [self.frame_to_row[f] for f in self.frm_list]

        params = {}
        for k in PARAM_KEYS:
            if k in self.smplx_npz:
                params[k] = torch.from_numpy(
                    np.ascontiguousarray(self.smplx_npz[k][rows]).astype(np.float32))

        kw = self.smplx_kwargs
        params['gender'] = str(kw.get('gender', 'neutral'))
        params['model_type'] = str(kw.get('model_type', 'smplx'))
        params['use_pca'] = bool(kw.get('use_pca', False))
        params['flat_hand_mean'] = bool(kw.get('flat_hand_mean', False))
        params['num_betas'] = int(params['betas'].shape[-1])
        if 'expression' in params:
            params['num_expression_coeffs'] = int(params['expression'].shape[-1])
        if kw.get('num_pca_comps', None) is not None:
            params['num_pca_comps'] = int(kw['num_pca_comps'])

        self.smplx_params = params
        if self.config.get('smplx_nofacial', False):
            reset_smplx_facial(self.smplx_params, self.config.smplx_nofacial)

    ##################################################
    def __len__(self):
        return len(self.frm_list)

    def __getitem__(self, idx):
        if idx is None:
            idx = torch.randint(0, len(self.frm_list), (1,)).item()

        frm_idx = int(self.frm_list[idx])

        color_frames, positions = read_DREAMS_frameset(
            self.cache_dir, frm_idx, self.cams, self.cam_ids,
            self.cam_select, self.mini_batch)
        scene_cameras = convert_to_scene_cameras(color_frames, self.config)

        batch = {
            'idx': idx,
            'frm_idx': frm_idx,
            # capture-level camera ids (cam00..cam31), matching what the ActorsHQ path
            # reports -- NOT positions inside cam_select
            'cam_idxs': [self.cam_ids[p] for p in positions],
            'scene_cameras': scene_cameras,
            'cameras_extent': self.cameras_extent,
        }

        smplx_params = self.load_smplx_params(idx)
        if smplx_params is not None:
            batch.update({'smplx_params': smplx_params})

        if self.exp_codes is not None and self.exp_codes[idx] is not None:
            _codes = self.exp_codes[idx]
            w = torch.rand((_codes.shape[0],))
            w = w / w.sum()
            batch.update({'exp_code': torch.einsum('i,ij->j', w, _codes)[None, ...]})

        return batch

    ##################################################
    def project(self, cam_pos, xyz_world):
        """(N,3) world points -> (N,2) pixels, for overlays and sanity checks."""
        cam = self.cams[cam_pos]
        x = np.asarray(xyz_world, np.float64).reshape(-1, 3) @ cam.R.T + cam.t
        uvw = x @ cam.K.T
        return uvw[:, :2] / uvw[:, 2:3]

    def joints(self, frm_idx):
        return self.smplx_npz['joints'][self.frame_to_row[int(frm_idx)]]

    def read_frame(self, cam_pos, frm_idx):
        """(rgb HxWx3 uint8 RGB-order, alpha HxW uint8) out of the decode cache."""
        sn = 'cam%02d' % self.cam_ids[cam_pos]
        bgr = cv2.imread(os.path.join(self.cache_dir, 'rgb', sn, '%08d.jpg' % frm_idx),
                         cv2.IMREAD_UNCHANGED)
        alpha = cv2.imread(os.path.join(self.cache_dir, 'mask', sn, '%08d.png' % frm_idx),
                           cv2.IMREAD_UNCHANGED)
        if alpha is not None and alpha.ndim == 3:
            alpha = alpha[:, :, 0]
        return bgr[:, :, ::-1].copy(), alpha
