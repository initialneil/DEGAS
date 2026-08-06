"""
For error `OSError: [Errno 24] Too many open files`, use this:
https://discuss.pytorch.org/t/too-many-open-files-when-using-dataloader/9476/6
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
"""
import os
import torch
import copy
import numpy as np
from argparse import ArgumentParser
from gaussian_renderer import network_gui
from model.degas_model import DEGASModel
from model.loss_base import run_testing, run_validation
from dataset.dataset_helper import make_frameset_data, make_dataloader
from model import libcore
from model.libcore.omegaconf_utils import load_from_config
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':
    parser = ArgumentParser(description='DEGAS')
    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--dat_dir', type=str, required=True)
    parser.add_argument('--configs', type=lambda s: [i for i in s.split(',')], 
                        required=True, help='path to config file')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--ckpt_fn', type=str, default=None)
    args, extras = parser.parse_known_args()

    # model path or ckpt
    if args.model_path is None and args.ckpt_fn is None:
        print(f'[QUITING] must provide at least one of "--model_path" or "--ckpt_fn"')
        exit()

    if args.model_path is not None:
        model_path = args.model_path
        if not os.path.isabs(model_path):
            model_path = os.path.join(args.dat_dir, model_path)
            
        print('--------------------------------------------------')
        print(f'[model_path] {model_path}')
        eval_dirs = [dir for dir in os.listdir(os.path.join(model_path, 'point_cloud')) if dir.startswith('iteration_')]
        iters = [int(dir.split('iteration_')[1]) for dir in eval_dirs]
        last_i = np.argsort(iters)[-1]
        last_dir = eval_dirs[last_i]
        ckpt_fn = os.path.join(model_path, f'point_cloud/{last_dir}/checkpoint.pt').replace('\\', '/')
        print(f'Found checkpoint: {ckpt_fn}')
    
        config_fn = os.path.join(model_path, 'config.yaml').replace('\\', '/')
        configs = copy.deepcopy(args.configs)
        if os.path.isfile(config_fn):
            configs.append(config_fn)
            print(f'Found config: {config_fn}')

        # load model and training config
        config = load_from_config(configs, dat_dir=args.dat_dir, cli_args=extras)
        libcore.set_seed(config.get('seed', 9061))
        
        pca_fn = os.path.join(model_path, 'pca.pt')
        if os.path.isfile(pca_fn):
            print(f'Found pca: {pca_fn}')
            pca_fn = pca_fn
        else:
            pca_fn = None
        
        model_cfgs = {
            'config': config,
            'ckpt_fn': ckpt_fn,
            'pca_fn': pca_fn,
            'iteration': iters[last_i],
        }
        print('--------------------------------------------------')
    else:
        # load model and training config
        config = load_from_config(args.configs, cli_args=extras)
        libcore.set_seed(config.get('seed', 9061))

        model_cfgs = {
            'config': config,
            'ckpt_fn': args.ckpt_fn,
            'pca_fn': args.pca_fn,
            'iteration': 0,
        }

    ##################################################
    config.dataset.dat_dir = args.dat_dir
    
    frameset_test = make_frameset_data(config.dataset, split='test')

    # # smplx optimizer
    # smplx_optim = SMPLXOptimizer(**config.optim.smplx_optim)
    # smplx_optim.setup_smplx_params_frameset(frameset_train)
    # cano_params = smplx_optim.get_tpose_params()
    # cano_mesh = smplx_optim.get_tpose_mesh().detach().clone()

    ##################################################
    render_config = {
        'mesh_from': 'batch',
    }

    checkpoint = torch.load(model_cfgs['ckpt_fn'], weights_only=False)
    # config_model = checkpoint['config']
    config_model = config.model
    gs_model = DEGASModel.create_from_checkpoint(checkpoint, config_model, render_config)

    ##################################################
    pipe = config.pipe
    if args.ip != 'none':
        network_gui.init(args.ip, args.port)
        verify = args.dat_dir
    else:
        verify = None

    ##################################################
    iteration = model_cfgs['iteration'] + 1
    run_testing(pipe, frameset_test, gs_model, model_path, iteration, verify=verify)

    ##################################################
    # eval finished. hold on
    network_gui.try_connect()
    while network_gui.conn is not None:
        network_gui.render_to_network(gs_model, pipe, args.dat_dir)

    print('[done]')
