import os
import torch
import torch.nn as nn
from utils.general_utils import get_expon_lr_func

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

class OptimizerBase:
    def __init__(self, gs_model, smplx_optim=None, optimizer_config=None) -> None:
        self.gs_model = gs_model
        self.optimizer = None
        self.smplx_optim = smplx_optim
        self.schedulers = []
        # manual learning rate scheduler
        self.lr_schedulers = {}

        if optimizer_config is not None:
            self.setup_optimizer(optimizer_config)

    def make_expon_lr_func(self, scheduler_args):
        return get_expon_lr_func(
            lr_init=scheduler_args.lr_init,
            lr_final=scheduler_args.lr_final,
            lr_delay_mult=scheduler_args.get('lr_delay_mult', 0.1),
            max_steps=scheduler_args.get('lr_max_steps', self.optimizer_config.total_iteration))

    def config_lr(self, keys, optimizer_config):
        model = self.gs_model
        lr_schedulers = {}
        l = []
        for key in keys:
            var_name = key[5:]
            if hasattr(model, var_name):
                optim_cfg = optimizer_config[key]
                if not optim_cfg:
                    print(f'[OptimizerBase] optim{var_name}: off')
                    continue

                variable = getattr(model, var_name)
                if isinstance(variable, torch.Tensor) or isinstance(variable, nn.Module):
                    # single learning rate
                    if not optim_cfg.get('scheduler_args', None):
                        lr = optim_cfg.lr
                        print(f'[OptimizerBase] optim{var_name}, lr = {lr}')
                    # with manual schedular
                    else:
                        lr = optim_cfg.scheduler_args.lr_init
                        lr_init = optim_cfg.scheduler_args.lr_init
                        lr_final = optim_cfg.scheduler_args.lr_final
                        print(f'[OptimizerBase] optim{var_name}, lr_scheduler = {lr_init} -> {lr_final}')
                        lr_schedulers[var_name] = self.make_expon_lr_func(optim_cfg.scheduler_args)

                    # variable is a tensor
                    if isinstance(variable, torch.Tensor):
                        setattr(model, var_name, nn.Parameter(getattr(model, var_name).requires_grad_(True)))
                        l.append({
                            'params': getattr(model, var_name), 
                            'lr': lr, 
                            'name': var_name,
                            'density_control': True,
                        })
                    # variable is a mlp
                    else:
                        l.append({
                            'params': getattr(model, var_name).parameters(), 
                            'lr': lr, 
                            'name': var_name,
                            'density_control': False,
                        })
            # variable not found
            else:
                print(f'[OptimizerBase][WARNING] variable {var_name} not found!')
                # comment if you want to ignore this.
                raise NotImplementedError
            
        return l, lr_schedulers

    def setup_optimizer(self, optimizer_config):
        self.optimizer_config = optimizer_config
        model = self.gs_model

        # optim_xyz, optim_opacity, optim_scaling ...
        keys = [k for k in optimizer_config.keys() if k.startswith('optim_')]
        l, lr_schedulers = self.config_lr(keys, optimizer_config)
        self.lr_schedulers = lr_schedulers

        self.optimizer = torch.optim.Adam(l, lr=5e-4, eps=1e-15)
        model.xyz_gradient_accum = torch.zeros((model.num_gauss, 1), device='cuda')
        model.denom = torch.zeros((model.num_gauss, 1), device='cuda')
        model.percent_dense = optimizer_config.get('percent_dense', 0.01)

        # optimizer scheduler
        if optimizer_config.get('scheduler', None):
            total_iteration = optimizer_config.total_iteration
            milestones = optimizer_config.scheduler.get('milestone', 10000)
            if not isinstance(milestones, list):
                milestones = [i for i in range(1, total_iteration) if i % milestones == 0]
            decay = optimizer_config.get('decay', 0.33)

            self.schedulers.append(torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=milestones,
                gamma=decay,
            ))

        # smplx
        if self.smplx_optim is not None:
            lr = optimizer_config.get('smplx_optim', {}).get('lr', 0)
            self.smplx_optim.lr = lr
            if lr > 0:
                print(f'[OptimizerBase] smplx_optim, lr = {lr}')
                self.smplx_optim.setup_optimizer(lr=lr)
                self.smplx_start_iter = optimizer_config.smplx_optim.get('start_from_iter', 1000)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group['name'] in self.lr_schedulers:
                lr = self.lr_schedulers[param_group['name']](iteration)
                param_group['lr'] = lr
        # return lr

    def step(self, iteration, enable_optim=True, enable_smplx=True):
        if enable_optim:
            self.optimizer.step()

        if enable_smplx and self.smplx_optim is not None and self.smplx_optim.lr > 0:
            if iteration > self.smplx_start_iter:
                self.smplx_optim.step()

        for scheduler in self.schedulers:
            scheduler.step()

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none=set_to_none)

        if self.smplx_optim is not None and self.smplx_optim.lr > 0:
            self.smplx_optim.zero_grad()

    # def smplx_state_dict(self):
    #     state = {
    #         **self.state_dict(),
    #         'optimizer': {**self.optimizer.state_dict()},
    #     }

    #     if self.smplx_optim is not None:
    #         state['smplx_optim'] = {**self.smplx_optim.state_dict()}

    #     return state
            
    def grad_loss_step(self, iteration, loss, render_pkg):
        loss.backward()
        self.adaptive_density_control(render_pkg, iteration)
        self.step(iteration)
        self.zero_grad(set_to_none=True)

    ##################################################
    def save_checkpoint(self, model_path, iteration, full_config, with_optim_state=True):
        pc_dir = os.path.join(model_path, f'point_cloud/iteration_{iteration}')
        os.makedirs(pc_dir, exist_ok=True)

        # point cloud
        model = self.gs_model
        model.save_ply(os.path.join(pc_dir, 'point_cloud.ply'))

        # checkpoint
        checkpoint = {
            'full_config': full_config,
            'optimizer':{
                'config': self.optimizer_config,
            }
        }

        model.update_to_checkpoint(checkpoint)

        if with_optim_state:
            checkpoint['optimizer']['state_dict'] = self.optimizer.state_dict()

        torch.save(checkpoint, os.path.join(pc_dir, 'checkpoint.pt'))

        # pca
        if hasattr(model, 'pca') and model.pca is not None:
            torch.save(model.pca, os.path.join(model_path, 'pca.pt'))

        # smplx
        if self.smplx_optim is not None:
            torch.save(self.smplx_optim.get_all_params(), os.path.join(pc_dir, 'smplx_refined.pt'))
            with open(os.path.join(pc_dir, 'train.txt'), 'w') as fp:
                for frm_idx in self.smplx_optim.frm_list:
                    print(f'{frm_idx:06d}', file=fp)

        return pc_dir

    ##################################################
    def reset_opacity(self):
        model = self.gs_model
        opacities_new = model.inverse_opacity_activation(torch.min(model.get_opacity_cano, torch.ones_like(model.get_opacity_cano)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, '_opacity')
        model._opacity = optimizable_tensors['_opacity']

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group['name'] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state['exp_avg'] = torch.zeros_like(tensor)
                stored_state['exp_avg_sq'] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group['params'][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group['name']] = group['params'][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group['params']) != 1:
                continue
            if not group['density_control']:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)

            if group['name'] != 'xyz_comp':
                if stored_state is not None:
                    stored_state['exp_avg'] = stored_state['exp_avg'][mask]
                    stored_state['exp_avg_sq'] = stored_state['exp_avg_sq'][mask]

                    del self.optimizer.state[group['params'][0]]
                    group['params'][0] = nn.Parameter((group['params'][0][mask].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group['name']] = group['params'][0]
                else:
                    group['params'][0] = nn.Parameter(group['params'][0][mask].requires_grad_(True))
                    optimizable_tensors[group['name']] = group['params'][0]
            else:
                if stored_state is not None:
                    stored_state['exp_avg'] = stored_state['exp_avg'][:, mask]
                    stored_state['exp_avg_sq'] = stored_state['exp_avg_sq'][:, mask]

                    del self.optimizer.state[group['params'][0]]
                    group['params'][0] = nn.Parameter((group['params'][0][:, mask].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group['name']] = group['params'][0]
                else:
                    group['params'][0] = nn.Parameter(group['params'][0][:, mask].requires_grad_(True))
                    optimizable_tensors[group['name']] = group['params'][0]

        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)
        self.gs_model.prune_points(valid_points_mask, optimizable_tensors)

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # assert len(group['params']) == 1
            if len(group['params']) != 1:
                continue

            extension_tensor = tensors_dict.get(group['name'], None)
            if extension_tensor is None:
                continue

            dd = 1 if group['name'] == 'xyz_comp' else 0
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state['exp_avg'] = torch.cat((stored_state['exp_avg'], torch.zeros_like(extension_tensor)), dim=dd)
                stored_state['exp_avg_sq'] = torch.cat((stored_state['exp_avg_sq'], torch.zeros_like(extension_tensor)), dim=dd)

                del self.optimizer.state[group['params'][0]]
                group['params'][0] = nn.Parameter(torch.cat((group['params'][0], extension_tensor), dim=dd).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group['name']] = group['params'][0]
            else:
                group['params'][0] = nn.Parameter(torch.cat((group['params'][0], extension_tensor), dim=dd).requires_grad_(True))
                optimizable_tensors[group['name']] = group['params'][0]

        return optimizable_tensors
    
    def densification_postfix(self, densify_out):
        d = {}
        for key in densify_out.keys():
            if key.startswith('new_'):
                d[key[3:]] = densify_out[key]

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self.gs_model.densification_postfix(optimizable_tensors, densify_out)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        selected_pts_mask, new_xyz = self.gs_model.prepare_densify_and_split(grads, grad_threshold, scene_extent, N=N)
        self.split_selected_to_new_xyz(selected_pts_mask, new_xyz, N)
        
    def split_selected_to_new_xyz(self, selected_pts_mask, new_xyz, N):
        splitout = self.gs_model.prepare_split_selected_to_new_xyz(selected_pts_mask, new_xyz, N)
        self.densification_postfix(splitout)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device='cuda', dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent=2.0):
        cloneout = self.gs_model.prepare_densify_and_clone(grads, grad_threshold, scene_extent)
        self.densification_postfix(cloneout)

    def densify_and_prune(self, max_grad, min_opacity, min_scaling, extent, max_screen_size):
        model = self.gs_model
        grads = model.xyz_gradient_accum / model.denom
        grads[grads.isnan()] = 0.0

        if model.config.get('max_n_gauss', -1) <= 0 or model.num_gauss < model.config.max_n_gauss:
            self.densify_and_clone(grads, max_grad, extent)
            self.densify_and_split(grads, max_grad, extent)

        self.prune(min_opacity, min_scaling, extent, max_screen_size)

    def prune(self, min_opacity, min_scaling, extent, max_screen_size):
        model = self.gs_model

        opacity = model.get_opacity_cano
        scaling_b = model.get_scaling_cano.max(dim=-1).values
        prune_mask = torch.logical_or(
            (opacity < min_opacity).squeeze(),
            (scaling_b < min_scaling))

        if max_screen_size:
            big_points_vs = model.max_radii2D > max_screen_size
            big_points_ws = scaling_b > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.gs_model.add_densification_stats(viewspace_point_tensor, update_filter)
        
    def adaptive_density_control(self, render_pkg, iteration, cameras_extent=2.0):
        gs_model = self.gs_model
        viewspace_point_tensor = render_pkg['viewspace_points']
        visibility_filter = render_pkg['visibility_filter']
        radii = render_pkg['radii']

        opt = self.optimizer_config
        min_opacity = opt.get('min_opacity', 0.0)
        min_scaling = opt.get('min_scaling', 0.0)
        opacity_reset_iter = opt.get('opacity_reset_start_iter', 0)
        opacity_reset_interval = opt.get('opacity_reset_interval', 0)

        # Densification
        if iteration < opt.densify_until_iter:
            # Keep track of max radii in image-space for pruning
            gs_model.max_radii2D[visibility_filter] = torch.max(gs_model.max_radii2D[visibility_filter], radii[visibility_filter])
            self.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if iteration >= opt.densify_from_iter and iteration % opt.densification_interval == 0:
                size_threshold = opt.size_threshold if iteration > opacity_reset_interval else None
                # size_threshold = None
                self.densify_and_prune(opt.densify_grad_threshold, min_opacity, min_scaling,
                                       cameras_extent, size_threshold)
            
            # if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
            if opacity_reset_interval > 0 and (iteration - opacity_reset_iter) % opacity_reset_interval == 0:
                self.reset_opacity()
