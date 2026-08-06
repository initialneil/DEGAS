#!/usr/bin/env python
"""Decode a capture's frames into the cache up front, before training starts.

The `dreams` reader decodes the (camera, frame) pairs a split needs on construction, so
the first training run pays that cost while holding a GPU. Doing it here instead means the
GPU run starts at full speed, and a decode failure surfaces in minutes rather than after
you have queued a 40 h job.

Safe to re-run: existing cache entries are kept, so only the delta is decoded.

    python tools/prebuild_cache.py --dat_dir /path/to/DREAMS-AVATAR/data/P1C1 \
        --configs configs/degas_config.yaml,configs/degas_vae_driver.yaml,configs/dreams/p1_train_base.yaml,configs/dreams/p1_face_B.yaml
"""
import argparse
import os
import sys

# This tool lives in tools/ but imports the packages at the repo root, and Python puts the
# SCRIPT's directory on sys.path, not the caller's. Add the root so it works from anywhere.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model.libcore.omegaconf_utils import load_from_config  # noqa: E402
from dataset.dataset_helper import make_frameset_data       # noqa: E402

DEFAULT_CONFIGS = ("configs/degas_config.yaml,configs/degas_vae_driver.yaml,"
                   "configs/dreams/p1_train_base.yaml,configs/dreams/p1_face_B.yaml")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dat_dir', required=True, help='the capture, e.g. .../data/P1C1')
    ap.add_argument('--configs', default=DEFAULT_CONFIGS,
                    help='comma-separated, same order as degas_train.py')
    ap.add_argument('--splits', default='val,test,train',
                    help='which splits to decode (default: all three)')
    a = ap.parse_args()

    # resolve config paths against the repo root, so any cwd works
    cfgs = [c if os.path.isabs(c) else os.path.join(ROOT, c)
            for c in a.configs.split(',')]
    missing = [c for c in cfgs if not os.path.isfile(c)]
    if missing:
        raise SystemExit(f'[FATAL] config not found: {missing}')

    cfg = load_from_config(cfgs, dat_dir=a.dat_dir)
    cfg.dataset.dat_dir = a.dat_dir

    for split in a.splits.split(','):
        ds = make_frameset_data(cfg.dataset, split=split)
        print('[prebuild]', split, 'ok:', len(ds), 'frames', flush=True)
    print('[prebuild] DONE')


if __name__ == '__main__':
    main()
