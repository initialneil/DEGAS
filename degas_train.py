"""
For error `OSError: [Errno 24] Too many open files`, use this:
https://discuss.pytorch.org/t/too-many-open-files-when-using-dataloader/9476/6
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
"""
import os
import torch
from pathlib import Path
import numpy as np
from random import randint
from argparse import ArgumentParser
from gaussian_renderer import network_gui
from datetime import datetime
from tqdm import tqdm
from omegaconf import OmegaConf
from model.gauss_base import find_checkpoint
from model.degas_model import DEGASModel
from model.degas_optim import DEGASOptimizer
from model.loss_base import run_testing, run_validation
from dataset.dataset_helper import make_frameset_data, make_dataloader
from model import libcore
from model.bone_deformer.smplx_optim import SMPLXOptimizer
from model.libcore.omegaconf_utils import load_from_config
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from utils.pynvml_utils import is_free_processes
import time

if __name__ == '__main__':
    parser = ArgumentParser(description='DEGAS')
    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--dat_dir', type=str, required=True)
    parser.add_argument('--configs', type=lambda s: [i for i in s.split(',')], 
                        required=True, help='path to config file')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--is_continue', action='store_true')
    parser.add_argument('--no-is_continue', action='store_false', dest='is_continue')
    parser.set_defaults(is_continue=True)
    parser.add_argument('--wait_for_gpu', action='store_true')
    args, extras = parser.parse_known_args()

    # output dir
    if args.model_path is None:
        model_path = f"output-splatting/{datetime.now().strftime('@%Y%m%d-%H%M%S')}"
    else:
        model_path = args.model_path
        
    if not os.path.isabs(model_path):
        model_path = os.path.join(args.dat_dir, model_path)

    # continue training
    if args.is_continue and os.path.exists(os.path.join(model_path, 'point_cloud')):
        ckpt_info = find_checkpoint(model_path, args.configs)
        if ckpt_info['smplx_fn'] is not None:
            extras.insert(0, f'dataset.smplx_type={ckpt_info["smplx_fn"]}')
    else:
        ckpt_info = {
            'configs': args.configs,
            'ckpt_fn': None,
            'pca_fn': None,
            'smplx_fn': None,
            'iteration': 0,
        }
        
    # load model and training config
    config = load_from_config(ckpt_info['configs'], dat_dir=args.dat_dir, cli_args=extras)
    libcore.set_seed(config.get('seed', 9061))

    ##################################################
    # waiting for gpu
    if args.wait_for_gpu:
        gpu_id = int(os.environ['CUDA_VISIBLE_DEVICES']) if 'CUDA_VISIBLE_DEVICES' in os.environ else 0
        while not is_free_processes(gpu_id):
            print('wait 10 seconds')
            time.sleep(10)

    ##################################################
    config.dataset.dat_dir = args.dat_dir
    
    frameset_val = make_frameset_data(config.dataset, split='val')
    frameset_train = make_frameset_data(config.dataset, split='train')
    frameset_test = make_frameset_data(config.dataset, split='test')
    dataloader = make_dataloader(frameset_train, shuffle=True)

    # smplx optimizer
    smplx_optim = SMPLXOptimizer(**config.optim.smplx_optim)
    smplx_optim.setup_smplx_params_frameset(frameset_train)
    cano_params = smplx_optim.get_tpose_params()
    cano_mesh = smplx_optim.get_tpose_mesh().detach().clone()

    ##################################################
    render_config = {
        'mesh_from': 'smplx_optim',
        'smplx_optim': smplx_optim,
    }

    if ckpt_info['ckpt_fn'] is not None:
        print(f'Training continue on iter#{ckpt_info["iteration"]}, {model_path}')
        checkpoint = torch.load(ckpt_info['ckpt_fn'], weights_only=False)
        # config_model = checkpoint['config']
        config_model = config.model
        gs_model = DEGASModel.create_from_checkpoint(checkpoint, config_model, render_config)
        if ckpt_info['pca_fn'] is not None:
            print(f'Load pca... {ckpt_info["pca_fn"]}')
            gs_model.pca = torch.load(ckpt_info['pca_fn'], weights_only=False)
    else:
        print(f'Training from scratch, {model_path}')
        gs_model = DEGASModel(config.model, render_config, verbose=True)
        gs_model.create_from_canonical(cano_params, cano_mesh)
        gs_model.pca = frameset_train.pca

    gs_model.update_to_pose(cano_params)

    gs_optim = DEGASOptimizer(gs_model, smplx_optim, config.optim)

    ##################################################
    pipe = config.pipe
    if args.ip != 'none':
        network_gui.init(args.ip, args.port)
        verify = args.dat_dir
    else:
        verify = None

    ##################################################
    os.makedirs(model_path, exist_ok=True)
    OmegaConf.save(config, os.path.join(model_path, 'config.yaml'))
    print(f'Training start on {model_path}')

    data_iterator = iter(dataloader)
    viewpoint_stack = None
    do_training = True

    total_iteration = config.optim.total_iteration
    save_every_iter = config.optim.get('save_every_iter', 10000)
    validate_every_inter = config.optim.get('validate_every_inter', 1000)
    testing_iterations = config.optim.get('testing_iterations', [total_iteration])

    pbar = tqdm(range(1, total_iteration+1))
    pbar.update(ckpt_info['iteration'])
    iteration = ckpt_info['iteration'] + 1
    while iteration < total_iteration+1:
        gs_optim.update_learning_rate(iteration)

        if not viewpoint_stack:
            try:
                batches = next(data_iterator)
            except:
                data_iterator = iter(dataloader)
                batches = next(data_iterator)
            batch = batches[0]
            frm_idx = batch['frm_idx']
            scene_cameras = batch['scene_cameras']
            viewpoint_stack = scene_cameras.copy()
            
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1)).cuda()
        gt_image = viewpoint_cam.original_image.cuda()

        # pre-render
        gs_model.pre_render(batch)
        
        # send one image to gui (optional)
        if args.ip != 'none':
            do_training = network_gui.render_to_network(gs_model, pipe, verify, gt_image=gt_image)
            while not do_training:
                do_training = network_gui.render_to_network(gs_model, pipe, verify, gt_image=gt_image)

        # render
        render_pkg = gs_model.render_to_camera(viewpoint_cam, pipe)
        render = render_pkg['render']
        gt_image = render_pkg['gt_image']

        # ### debug ###
        # from model import libcore
        # libcore.write_tensor_image(os.path.join('e:/dummy/gt_image.jpg'), gt_image, rgb2bgr=True)
        # libcore.write_tensor_image(os.path.join('e:/dummy/render.jpg'), render, rgb2bgr=True)

        # loss
        loss = gs_optim.collect_loss(iteration, batch, **render_pkg)
        gs_optim.grad_loss_step(iteration, loss['loss'], **render_pkg)

        with torch.no_grad():
            gs_model.post_optim()

            pbar.set_description(f'#{frm_idx:06d}')
            pbar.set_postfix({
                '#gauss': gs_model.num_gauss,
                'loss': loss['loss'].item(),
                'psnr': loss['psnr_full'],
            })

            # save
            if iteration == 10000 or (save_every_iter > 0 and iteration % save_every_iter == 0):
                pc_dir = gs_optim.save_checkpoint(model_path, iteration, config)
                libcore.write_tensor_image(os.path.join(pc_dir, 'gt_image.jpg'), gt_image, rgb2bgr=True)
                libcore.write_tensor_image(os.path.join(pc_dir, 'render.jpg'), render, rgb2bgr=True)

        # report testing
        if iteration in testing_iterations:
            run_testing(pipe, frameset_test, gs_model, model_path, iteration, verify=verify)
        if iteration == 100 or iteration % validate_every_inter == 0:
            run_validation(pipe, frameset_val, gs_model, model_path, iteration, verify=verify)

        iteration += 1
        pbar.update(1)

    ##################################################
    # training finished. hold on
    network_gui.try_connect()
    while network_gui.conn is not None:
        network_gui.render_to_network(gs_model, pipe, args.dat_dir)

    print('[done]')


