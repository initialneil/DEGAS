import sys
from omegaconf import OmegaConf
# These tools live in tools/ but import the packages at the repo root, and Python
# puts the SCRIPT's directory on sys.path, not the caller's. Add the root so
# `python tools/<name>.py` works from anywhere.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from model.libcore.omegaconf_utils import load_from_config
from dataset.dataset_helper import make_frameset_data
cfg = load_from_config(['configs/degas_config.yaml','configs/degas_vae_driver.yaml',
                        'configs/dreams/p1_train_base.yaml','configs/dreams/p1_face_A.yaml'])
cfg.dataset.dat_dir = sys.argv[1]
for split in ['val','test','train']:
    ds = make_frameset_data(cfg.dataset, split=split)
    print('[prebuild]', split, 'ok:', len(ds), 'frames', flush=True)
print('[prebuild] DONE')
