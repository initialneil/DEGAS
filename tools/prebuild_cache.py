import sys
from omegaconf import OmegaConf
from model.libcore.omegaconf_utils import load_from_config
from dataset.dataset_helper import make_frameset_data
cfg = load_from_config(['configs/degas_config.yaml','configs/degas_vae_driver.yaml',
                        'configs/dreams/p1_train_base.yaml','configs/dreams/p1_face_A.yaml'])
cfg.dataset.dat_dir = sys.argv[1]
for split in ['val','test','train']:
    ds = make_frameset_data(cfg.dataset, split=split)
    print('[prebuild]', split, 'ok:', len(ds), 'frames', flush=True)
print('[prebuild] DONE')
