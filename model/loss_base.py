import os
import cv2
import numpy as np
import json
import torch
import torch.nn.functional as thf
from utils.loss_utils import l1_loss, ssim, LPIPS
from utils.image_utils import psnr
from utils.metrics import img_mse, img_ssim, img_psnr, perceptual
from dataset.dataset_helper import make_dataloader
from gaussian_renderer import network_gui
from model import libcore
from pathlib import Path
from tqdm import tqdm

# LPIPS is calculated on cropped images
# (PSNR and SSIM are calculated on full images)
# https://github.com/lizhe00/AnimatableGaussians/blob/master/eval/comparison_body_only_avatars.py#L68-L70
# https://github.com/lizhe00/AnimatableGaussians/blob/master/eval/score.py#L23
def crop_image(gt_mask, patch_size, *args):
    """
    :param gt_mask: (H, W)
    :param patch_size: resize the cropped patch to the given patch_size
    :param args: some images with shape of (H, W, C)
    """
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.detach().cpu().numpy().squeeze()

    mask_uv = np.argwhere(gt_mask > 0.)
    min_v, min_u = mask_uv.min(0)
    max_v, max_u = mask_uv.max(0)
    pad_size = 50
    min_v = (min_v - pad_size).clip(0, gt_mask.shape[0])
    min_u = (min_u - pad_size).clip(0, gt_mask.shape[1])
    max_v = (max_v + pad_size).clip(0, gt_mask.shape[0])
    max_u = (max_u + pad_size).clip(0, gt_mask.shape[1])
    len_v = max_v - min_v
    len_u = max_u - min_u
    max_size = max(len_v, len_u)

    cropped_images = []
    for _image in args:
        if _image is None:
            cropped_images.append(None)
        else:
            if isinstance(_image, torch.Tensor):
                image = _image.detach().cpu().permute([1, 2, 0]).numpy()
            elif image.dtype == np.uint8:
                image = _image.astype(np.float32) / 255
            else:
                image = _image

            cropped_image = np.ones((max_size, max_size, 3), dtype=image.dtype)
            if len_v > len_u:
                start_u = (max_size - len_u) // 2
                cropped_image[:, start_u: start_u + len_u] = image[min_v: max_v, min_u: max_u]
            else:
                start_v = (max_size - len_v) // 2
                cropped_image[start_v: start_v + len_v, :] = image[min_v: max_v, min_u: max_u]

            if patch_size > 0:
                cropped_image = cv2.resize(cropped_image, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)

            if isinstance(_image, torch.Tensor):
                cropped_image = torch.tensor(cropped_image).to(_image).permute([2, 0, 1])

            cropped_images.append(cropped_image)

    if len(cropped_images) > 1:
        return cropped_images
    else:
        return cropped_images[0]


class LossBase:
    def __init__(self, gs_model, optimizer_config=None) -> None:
        self.gs_model = gs_model
        self.optimizer_config = optimizer_config

        # lpips for perceptual loss
        if self.optimizer_config.get('lambda_perceptual', 0) > 0:
            self.lpips = LPIPS(eval=False).cuda()

    def collect_loss(self, gt_image, render, gt_alpha_mask=None, image_weights=None, 
                     iteration=None, **render_pkg):
        # image weights anealing
        if 'image_weights_anealing_steps' in self.optimizer_config:
            max_steps = self.optimizer_config['image_weights_anealing_steps']
            if iteration is not None and iteration < max_steps:
                anealing_ratio = (np.cos(iteration / max_steps * np.pi) + 1.0) / 2.0
                anealing_final = self.optimizer_config.image_weights_anealing_final
                anealing = anealing_final + (1.0 - anealing_final) * anealing_ratio
            else:
                anealing = anealing_final
            image_weights = image_weights * (1.0 - anealing) + torch.ones_like(image_weights) * anealing

        # l1 loss
        if image_weights is None:
            Ll1 = l1_loss(gt_image, render)
        else:
            Ll1 = ((gt_image - render) * image_weights).abs().mean()

        # ssim loss
        if self.optimizer_config.get('lambda_ssim', 0) > 0:
            if image_weights is None:
                Lssim = 1.0 - ssim(gt_image, render)
            else:
                Lssim = 1.0 - ssim(gt_image * image_weights, render * image_weights)
            loss = (1.0 - self.optimizer_config.lambda_ssim) * Ll1 + self.optimizer_config.lambda_ssim * Lssim
        else:
            loss = Ll1

        # mse loss
        if self.optimizer_config.get('lambda_rgb_mse', 0) > 0:
            if image_weights is None:
                Ll2 = thf.mse_loss(gt_image, render)
            else:
                Ll2 = (((gt_image - render) * image_weights) ** 2).mean()
            loss += self.optimizer_config.lambda_rgb_mse * Ll2

        # perceptual loss
        if self.optimizer_config.get('lambda_perceptual', 0) > 0:
            if gt_alpha_mask is not None:
                Llpips = self.lpips(gt_image * gt_alpha_mask, render * gt_alpha_mask).squeeze()
            else:
                Llpips = self.lpips(gt_image, render).squeeze()
            loss += self.optimizer_config.lambda_perceptual * Llpips

        # sparsity loss
        if self.optimizer_config.get('lambda_sparsity', 0) > 0:
            loss += self.optimizer_config.lambda_sparsity * self.gs_model.get_opacity.mean()

        # scaling_skew loss
        if self.optimizer_config.get('lambda_scaling_skew', 0) > 0:
            thresh_scaling_max = self.optimizer_config.get('thresh_scaling_max', 0.008)
            thresh_scaling_ratio = self.optimizer_config.get('thresh_scaling_ratio', 10.0)
            max_vals = self.gs_model.get_scaling.max(dim=-1).values
            min_vals = self.gs_model.get_scaling.min(dim=-1).values
            ratio = max_vals / min_vals
            thresh_idxs = (max_vals > thresh_scaling_max) & (ratio > thresh_scaling_ratio)
            if thresh_idxs.sum() > 0:
                loss += self.optimizer_config.lambda_scaling_skew * max_vals[thresh_idxs].mean()

        # scaling mse
        if self.optimizer_config.get('lambda_scaling_mse', 0) > 0:
            scaling = self.gs_model.get_scaling
            loss += self.optimizer_config.lambda_scaling_mse * thf.mse_loss(scaling, torch.zeros_like(scaling))

        # flat gaussian
        if self.optimizer_config.get('lambda_scaling_z', 0) > 0:
            thresh_scaling_z = self.optimizer_config.get('thresh_scaling_z', 0.01)
            thresh_idxs = self.gs_model.get_scaling_cano[..., -1] > thresh_scaling_z
            if thresh_idxs.sum() > 0:
                loss += self.optimizer_config.lambda_scaling_z * self.gs_model.get_scaling_cano[..., -1].mean()

        if self.optimizer_config.get('lambda_normal_z_dot', 0) > 0:
            mesh_normal = self.gs_model.base_normal_cano
            gs_normal = self.gs_model.get_normal_cano
            l_dot = 1.0 - torch.einsum('ni,ni->n', mesh_normal, gs_normal).mean()
            loss += self.optimizer_config.lambda_normal_z_dot * l_dot

        # xyz
        if self.optimizer_config.get('lambda_xyz_z', 0) > 0:
            loss += self.optimizer_config.lambda_xyz_z * self.gs_model._xyz[..., 2].abs().mean()

        # psnr
        psnr_full = psnr(gt_image, render).mean().float().item()

        return {
            'loss': loss,
            'psnr_full': psnr_full,
        }

########## testing routine ##########
def visualize_compare(gt_image, image, psnr, ssim, lpips):
    compare = torch.concat([gt_image, image], dim=2)
    compare = (compare.permute([1, 2, 0]) * 255)[:, :, [2, 1, 0]].detach().cpu().numpy()
    compare = cv2.putText(compare, f'psnr/ssim/lpips', (20, compare.shape[0] - 50), 0, 1, (0, 0, 255))
    compare = cv2.putText(compare, f'{psnr:.4f}/{ssim:.4f}/{lpips:.4f}', 
                            (20, compare.shape[0] - 10), 0, 1, (0, 0, 255))
    
    err = (image - gt_image).abs().max(dim=0)[0].clip(0, 1)     
    from model import libcore       
    err_map = libcore.colorizeWeightsMap(err.detach().cpu().numpy(), min_val=0, max_val=1)
    compare = np.concatenate([compare, err_map], axis=1)
    return compare

# tensor to image
def write_tensor_image(fn, tensor, rgb2bgr=False):
    if len(tensor.shape) == 3:
        if tensor.shape[0] == 3 or tensor.shape[0] == 4:
            tensor = tensor.permute([1, 2, 0])

    if rgb2bgr:
        if tensor.shape[2] == 3:
            tensor = tensor[:, :, [2, 1, 0]]
        else:
            tensor = tensor[:, :, [2, 1, 0, 3]]
    
    os.makedirs(Path(fn).parent, exist_ok=True)
    cv2.imwrite(fn, (tensor.clamp(0, 1) * 255).detach().cpu().numpy().astype(np.uint8))

# testing routine
def testing_routine(pipe, frameset, gs_model, save_dir=None, verify=None):
    if save_dir is not None:
        os.makedirs(os.path.join(save_dir, 'render'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'compare'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'crop_render'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'crop_gt'), exist_ok=True)

    psnr_full = 0
    ssim_full = 0
    lpips_full = 0
    count = 0

    dataloader = make_dataloader(frameset, shuffle=False)
    data_iterator = iter(dataloader)

    with torch.no_grad():
        num_frames = len(frameset)
        pbar = tqdm(range(num_frames))
        for idx in pbar:
            batch = next(data_iterator)[0]
            frm_idx = batch['frm_idx']
            scene_cameras = batch['scene_cameras']

            # update to current posed mesh
            gs_model.pre_render(batch)
                
            # there should be only one camera
            viewpoint_cam = scene_cameras[0].cuda()

            # render
            render_pkg = gs_model.render_to_camera(viewpoint_cam, pipe, background='white')
            image = render_pkg['render']
            gt_image = render_pkg['gt_image']
            gt_mask = render_pkg['gt_alpha_mask']

            _rmse = img_mse(image[None, ...], gt_image[None, ...], mask=None, error_type='rmse', use_mask=False)
            _ssim = img_ssim(image[None, ...], gt_image[None, ...])
            _psnr = img_psnr(image[None, ...], gt_image[None, ...], rmse=_rmse)
            # _lpips = perceptual(image[None, ...], gt_image[None, ...], mask=None, use_mask=False)

            # lpips on cropped images
            image_cropped, gt_image_cropped = crop_image(gt_mask, 512, image, gt_image)
            _lpips = perceptual(image_cropped[None, ...], gt_image_cropped[None, ...], mask=None, use_mask=False)

            #############
            if verify is not None:
                network_gui.send_image_to_network(image, verify)
        
            #############
            if save_dir is not None:
                # render
                write_tensor_image(os.path.join(save_dir, f'render/{frm_idx:05d}.png'), image, rgb2bgr=True)
                # compare
                compare = visualize_compare(gt_image, image, _psnr.item(), _ssim.item(), _lpips.item())
                cv2.imwrite(os.path.join(save_dir, f'compare/{frm_idx:05d}.jpg'), compare)
                # cropped
                write_tensor_image(os.path.join(save_dir, f'crop_render/{frm_idx:05d}.png'), image_cropped, rgb2bgr=True)
                write_tensor_image(os.path.join(save_dir, f'crop_gt/{frm_idx:05d}.png'), gt_image_cropped, rgb2bgr=True)

                # mask
                write_tensor_image(os.path.join(save_dir, f'gt_mask/{frm_idx:05d}.png'), gt_mask.squeeze())
                # gt
                write_tensor_image(os.path.join(save_dir, f'gt_image/{frm_idx:05d}.png'), gt_image.squeeze(), rgb2bgr=True)

                # err_map = cv2.putText(err_map, f'psnr/ssim/lpips', (20, err_map.shape[0] - 50), 0, 1, (255, 255, 255))
                # err_map = cv2.putText(err_map, f'{_psnr.item():.4f}/{_ssim.item():.4f}/{_lpips.item():.4f}', 
                #                       (20, err_map.shape[0] - 10), 0, 1, (255, 255, 255))
                # cv2.imwrite(os.path.join(err_dir, f'{frm_idx:05d}.jpg'), err_map)
            #############

            psnr_full += _psnr.item()
            ssim_full += _ssim.item()
            lpips_full += _lpips.item()
            count += 1

            pbar.set_postfix({
                'psnr': f'{(psnr_full / count):.4f}({_psnr.item():.4f})',
                'ssim': f'{(ssim_full / count):.4f}({_ssim.item():.4f})',
                'lpips': f'{(lpips_full / count):.4f}({_lpips.item():.4f})',
            })

    return {
        'psnr': psnr_full / count,
        'ssim': ssim_full / count,
        'lpips': lpips_full / count,
        'n_gauss': gs_model.num_gauss,
    }

# validation routine
def validation_routine(pipe, frameset, gs_model, iteration, val_dir, verify=None):
    if val_dir is not None:
        os.makedirs(val_dir, exist_ok=True)

    with torch.no_grad():
        num_frames = len(frameset)
        for idx in range(num_frames):
            batch = frameset.__getitem__(idx)
            frm_idx = batch['frm_idx']
            cam_idx = batch['cam_idxs'][0]
            scene_cameras = batch['scene_cameras']

            # update to current posed mesh
            gs_model.pre_render(batch)
                
            # there should be only one camera
            viewpoint_cam = scene_cameras[0].cuda()

            # render
            render_pkg = gs_model.render_to_camera(viewpoint_cam, pipe, background='white', render_validation=True)
            image = render_pkg['render']
            gt_image = render_pkg['gt_image']

            _rmse = img_mse(image[None, ...], gt_image[None, ...], mask=None, error_type='rmse', use_mask=False)
            _ssim = img_ssim(image[None, ...], gt_image[None, ...])
            _psnr = img_psnr(image[None, ...], gt_image[None, ...], rmse=_rmse)
            _lpips = perceptual(image[None, ...], gt_image[None, ...], mask=None, use_mask=False)

            #############
            if verify is not None:
                network_gui.send_image_to_network(image, verify)
        
            #############
            compare = visualize_compare(gt_image, image, _psnr.item(), _ssim.item(), _lpips.item())
            cv2.imwrite(os.path.join(val_dir, f'iter{iteration:06d}_{frm_idx:06d}_{cam_idx:04d}.png'), compare)

            #############
            if 'render_extras' in render_pkg:
                extras = []
                for key in render_pkg['render_extras']:
                    if key == 'depth':
                        img = (render_pkg[key].permute(1, 2, 0) * 255.0).detach().cpu().numpy().astype(np.uint8)
                    else:
                        img = libcore.write_tensor_image(None, render_pkg[key])
                    img = cv2.putText(img.copy(), key, (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    extras.append(img)
                extras = np.concatenate(extras, axis=1)
                cv2.imwrite(os.path.join(val_dir, f'iter{iteration:06d}_{frm_idx:06d}_{cam_idx:04d}_extras.png'), extras)


def run_testing(pipe, frameset_test, gs_model, model_path=None, iteration=None, verify=None):
    if model_path is not None:
        if iteration is not None:
            save_dir = os.path.join(model_path, f'eval_{iteration}')
            stats_fn = os.path.join(model_path, f'eval_{iteration}/stats.json')
        else:
            save_dir = os.path.join(model_path, f'eval')
            stats_fn = os.path.join(model_path, f'eval/stats.json')
    else:
        stats_fn = None

    stats = testing_routine(pipe, frameset_test, gs_model,
                            save_dir=save_dir, 
                            verify=verify)
    
    if stats_fn is not None:
        with open(stats_fn, 'w') as fp:
            json.dump(stats, fp)

def run_validation(pipe, frameset_test, gs_model, model_path, iteration, verify=None):
    val_dir = os.path.join(model_path, f'validation')
    validation_routine(pipe, frameset_test, gs_model, iteration, val_dir, verify=verify)
    