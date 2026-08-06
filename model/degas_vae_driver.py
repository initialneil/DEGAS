import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as thf
from model.bone_deformer import smplx_deformer, smplx_utils
from model.libcore.nvdiffrast_mesh import NvdiffrastMesh
import model.ca_body.nn.layers as la
from model.ca_body.nn.blocks import (
    ConvBlock,
    ConvDownBlock,
    UpConvBlockDeep,
    tile2d,
    weights_initializer,
)

##################################################
class PoseMapper(nn.Module):
    def __init__(self, config, mesh,
                 device='cuda'):
        super().__init__()
        self.device = device

        self.setup_network(config)
        self.setup_nvmesh(mesh)

    def setup_network(self, config):
        self.config = config

    def setup_nvmesh(self, mesh):
        self.nvmesh = NvdiffrastMesh(mesh, backend='cuda', coordinate='opencv')

    def rasterize_verts_cond(self, verts=None, width=None, height=None):
        if verts is not None:
            self.nvmesh.vertices = verts
        if width is None:
            width = self.config.get('resolution', 512)
        if height is None:
            height = self.config.get('resolution', 512)

        # render lbs as attribute to uv
        rlt = self.nvmesh.rasterizeToAtlas(width, height, with_texture=False, with_vertex=True)
        verts_cond = rlt['vertices'].squeeze()
        verts_cond = verts_cond.permute(2, 0, 1)

        # mirror
        if self.config.get('mirror_verts_cond', True):
            verts_cond = torch.concat([
                verts_cond,
                verts_cond.flip(1),
                verts_cond.flip(2),
                verts_cond.flip(2).flip(1),
            ], dim=0)
        return verts_cond
    
    def rasterize_verts_to_cam(self, cam, verts=None, with_attr=None):
        if verts is not None:
            self.nvmesh.vertices = verts
            
        rlt = self.nvmesh.rasterizeToCamera(cam, with_texture=False, with_attr=with_attr)
        verts_cam = rlt['attributes'].squeeze()
        return verts_cam
    
    def rasterize_base_info(self, width=None, height=None):
        if width is None:
            width = self.config.get('resolution', 512)
        if height is None:
            height = self.config.get('resolution', 512)

        rlt = self.nvmesh.rasterizeToAtlas(width, height, with_texture=False, with_vertex=True)
        return rlt

    def forward(self, mesh_verts, width=None, height=None):
        # rasterize to verts cond
        # verts_cond: vertex map 3xHxW
        verts_cond = self.rasterize_verts_cond(mesh_verts, width=width, height=height)
        return verts_cond

##################################################
class Encoder(nn.Module):
    # https://github.com/facebookresearch/ca_body/tree/main/ca_body
    def __init__(self, config, smplx_model,
                 device='cuda'):
        """Fixed-width conv encoder."""
        super().__init__()
        self.device = device

        self.setup_network(config, smplx_model)

    def setup_network(self, config, smplx_model):
        self.config = config
        self.with_pose_net = False
        self.with_latent_net = False
        
        if config.get('n_verts_enc_channels', 0) > 0:
            self.setup_pose_net(config)

        if config.get('n_embs', 0) > 0:
            self.setup_latent_net(config)

    def setup_pose_net(self, config):
        if config.get('mirror_verts_cond', True):
            self.pose_verts_conv = ConvDownBlock(12, 8, 512)
        else:
            self.pose_verts_conv = ConvDownBlock(3, 8, 512)

        self.pose_conv_blocks = nn.Sequential(
            ConvDownBlock(8, 16, 256),
            ConvDownBlock(16, 32, 128),
        )

        self.with_pose_net = True

    def setup_latent_net(self, config):
        self.noise_std = config.get('noise_std', 1.0)
        self.n_embs = config.get('n_embs', 1024)
        self.logvar_scale = config.get('logvar_scale', 0.1)

        if config.get('mirror_verts_cond', True):
            self.latent_verts_conv = ConvDownBlock(12, 8, 512)
        else:
            self.latent_verts_conv = ConvDownBlock(3, 8, 512)

        # mask = torch.as_tensor(mask[np.newaxis, np.newaxis], dtype=torch.float32)
        # mask = thf.interpolate(mask, size=(512, 512), mode='bilinear').to(torch.bool)
        # self.register_buffer("mask", mask)

        self.latent_conv_blocks = nn.Sequential(
            ConvDownBlock(8, 16, 256),
            ConvDownBlock(16, 32, 128),
            ConvDownBlock(32, 32, 64),
            ConvDownBlock(32, 64, 32),
            ConvDownBlock(64, 128, 16),
            ConvDownBlock(128, 128, 8),
            # ConvDownBlock(128, 128, 4),
        )

        # TODO: should we put initializer
        self.mu = la.LinearWN(4 * 4 * 128, self.n_embs)
        self.logvar = la.LinearWN(4 * 4 * 128, self.n_embs)

        self.apply(weights_initializer(0.2))
        self.mu.apply(weights_initializer(1.0))
        self.logvar.apply(weights_initializer(1.0))

        self.with_latent_net = True

    def forward(self, verts_cond, B=1):
        preds = {}
        if self.with_pose_net:
            preds.update(self.predict_verts_code(verts_cond))
        if self.with_latent_net:
            preds.update(self.predict_latent_code(verts_cond, B=B))
        return preds
    
    def predict_verts_code(self, verts_cond):
        preds = {}
        joint_cond = self.pose_verts_conv(verts_cond)
        x = self.pose_conv_blocks(joint_cond)

        preds.update(
            verts_code=x,
        )
        return preds

    def predict_latent_code(self, verts_cond, B=1):
        preds = {}
        # tex_cond = thf.interpolate(tex_avg, size=(512, 512), mode='bilinear') * self.mask
        # tex_cond = self.tex_conv(tex_cond)
        # joint_cond = torch.cat([verts_cond, tex_cond], dim=1)
        # joint_cond = verts_cond
        joint_cond = self.latent_verts_conv(verts_cond)
        
        x = self.latent_conv_blocks(joint_cond)
        x = x.reshape(B, -1)
        embs_mu = self.mu(x)
        embs_logvar = self.logvar_scale * self.logvar(x)

        # NOTE: the noise is only applied to the input-conditioned values
        if self.training:
            noise = torch.randn_like(embs_mu)
            embs = embs_mu + torch.exp(embs_logvar) * noise * self.noise_std
        else:
            embs = embs_mu.clone()

        preds.update(
            embs=embs,
            embs_mu=embs_mu,
            embs_logvar=embs_logvar,
        )
        return preds

##################################################
class Decoder(nn.Module):
    def __init__(self, config, smplx_model,
                 device='cuda'):
        super().__init__()
        self.bulk_size = config.get('bulk_size', 100000)
        self.device = device

        # embedding
        n_embs = config.get('n_embs', 1024)
        if n_embs > 0:
            n_embs_enc_channels = config.get('n_embs_enc_channels', 32)

            self.embs_fc = nn.Sequential(
                la.LinearWN(n_embs, 4 * 4 * 128),
                nn.LeakyReLU(0.2, inplace=True),
            )

            # TODO: should we switch to the basic version?
            self.embs_conv_block = nn.Sequential(
                UpConvBlockDeep(128, 128, 8),
                UpConvBlockDeep(128, 128, 16),
                UpConvBlockDeep(128, 64, 32),
                UpConvBlockDeep(64, n_embs_enc_channels, 64),
            )
        else:
            n_embs_enc_channels = 0

        n_verts_enc_channels = config.get('n_verts_enc_channels', 32)

        # face condition
        n_face_embs = config.get('n_face_embs', 0)
        if n_face_embs > 0:
            self.face_embs_fc = nn.Sequential(
                la.LinearWN(n_face_embs, 4 * 4 * 32),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.face_embs_conv_block = nn.Sequential(
                UpConvBlockDeep(32, 64, 8),
                UpConvBlockDeep(64, 64, 16),
                UpConvBlockDeep(64, 32, 32),
            )

        # pose condition
        uv_size = config.get('uv_size', 512)
        init_uv_size = config.get('init_uv_size', 64)
        n_init_channels = config.get('n_init_channels', 64)
        n_min_channels = config.get('n_min_channels', 4)

        # correction branches
        self.corrections = config.corrections
        self.branches = {}
        for key in config.branches:
            l = [k for k in config.branches[key] if k in self.corrections]
            if len(l) > 0:
                self.branches[key] = l

        n_groups = 1 * len(self.branches)

        # pose condition ignore global_orient
        n_pose_dims = config.get('n_pose_dims', 162)
        n_pose_enc_channels = config.get('n_pose_enc_channels', 16)

        if n_pose_enc_channels > 0:
            # n_pose_dims -> n_pose_enc_channels
            self.local_pose_conv_block = ConvBlock(
                n_pose_dims,
                n_pose_enc_channels,
                init_uv_size,
                kernel_size=1,
                padding=0,
            )

        # pose enc to conv decoder input
        self.joint_conv_block = ConvBlock(
            n_pose_enc_channels + n_embs_enc_channels + n_verts_enc_channels,
            n_init_channels,
            init_uv_size,
        )

        # convolution decoder
        self.n_blocks = int(np.log2(uv_size // init_uv_size))
        self.sizes = [init_uv_size * 2**s for s in range(self.n_blocks + 1)]
        self.n_channels = [
            max(n_init_channels // 2**b, n_min_channels) for b in range(self.n_blocks + 1)
        ]
        print(f'[PoseDriverConvDec] convolution decoder with {self.n_blocks} blocks')
        print(f'[PoseDriverConvDec] groups: {len(self.branches)}x3')
        print(f'[PoseDriverConvDec] channels: {self.n_channels}')
        print(f'[PoseDriverConvDec] size: {self.sizes}')
        for sz, chn in zip(self.sizes, self.n_channels):
            print(f'[PoseDriverConvDec] -> [{len(self.branches)}x3] {chn}x{sz}x{sz}')

        self.conv_blocks = nn.ModuleList([])
        for b in range(self.n_blocks):
            self.conv_blocks.append(
                UpConvBlockDeep(
                    self.n_channels[b] * n_groups,
                    self.n_channels[b + 1] * n_groups,
                    self.sizes[b + 1],
                    groups=n_groups,
                ),
            )

        # init weights
        self.apply(weights_initializer(0.2))

        # pose condition mask on uv
        pose_cond_mask = self.create_pose_cond_mask(config, smplx_model)
        self.register_buffer('pose_cond_mask', pose_cond_mask)

        self.output_nets = nn.ModuleList()
        for key in self.branches:
            n_out_dims = np.sum([self.corrections[k] for k in self.corrections if k in self.branches[key]])
            output_conv = la.Conv2dWNUB(
                in_channels=self.n_channels[-1],
                out_channels=n_out_dims,
                kernel_size=3,
                height=uv_size,
                width=uv_size,
                padding=1,
            )
            output_conv.apply(weights_initializer(1.0))
            self.output_nets.append(output_conv)
            
        # mark params
        self.n_embs = n_embs
        self.n_verts_enc_channels = n_verts_enc_channels
        self.n_pose_enc_channels = n_pose_enc_channels
        self.init_uv_size = init_uv_size
        self.n_groups = n_groups
        self.n_face_embs = n_face_embs

    ##################################################
    # pose condition mask on uv
    def create_pose_cond_mask(self, config, smplx_model):
        init_uv_size = config.get('init_uv_size', 64)

        # pose condition mask on uv
        # discard global_orient
        lbs_weight = smplx_model.lbs_weights[:, 1:].contiguous().float().cuda()

        mesh = smplx_utils.convert_smplx_to_meshcpu(smplx_model)
        nvmesh = NvdiffrastMesh(mesh, backend='cuda', coordinate='opencv')

        # render lbs as attribute to uv
        rlt = nvmesh.rasterizeToAtlas(init_uv_size, init_uv_size, with_texture=False, with_attr=lbs_weight[None, ...])
        uv_lbs = rlt['attributes'].squeeze()

        # to mask
        pose_cond_mask = (uv_lbs > 0).to(int)

        # ##################################################
        # # dump
        # from model import libcore
        # import cv2
        # for i in range(uv_lbs.shape[-1]):
        #     m = uv_lbs[..., i].detach().cpu().numpy()
        #     m = libcore.colorizeWeightsMap(m, min_val=0, max_val=1)
        #     m = cv2.resize(m, (512, 512))
        #     cv2.imwrite(f'e:/dummy/pose_tile2d/tile2d_{i:02d}_lbs.jpg', m)

        # for i in range(pose_cond_mask.shape[-1]):
        #     m = (pose_cond_mask[..., i].detach().cpu() * 255).numpy().astype(np.uint8)
        #     m = cv2.resize(m, (512, 512))
        #     cv2.imwrite(f'e:/dummy/pose_tile2d/tile2d_{i:02d}_mask.jpg', m)
        # ##################################################

        # x3 dims -> 1x162x64x64
        pose_cond_mask = pose_cond_mask.permute(2, 0, 1).repeat(3, 1, 1)[None, ...]
        return pose_cond_mask
    
    def forward(self, full_pose, embs, verts_code, face_embs=None): 
        B = 1
        
        # TODO: decoding properly?
        # Bx32x64x64
        if self.n_embs > 0:
            embs_conv = self.embs_conv_block(self.embs_fc(embs).reshape(B, 128, 4, 4))

        # pose to cond
        if self.n_pose_enc_channels > 0:
            pose_cond = full_pose[..., 3:]
            pose_masked = tile2d(pose_cond, self.init_uv_size) * self.pose_cond_mask    # Bx162x64x64
            pose_conv = self.local_pose_conv_block(pose_masked)                         # Bx16x64x64

        # face embedding from DPE
        if self.n_face_embs > 0:
            if face_embs is None:
                face_embs = torch.zeros((1, self.n_face_embs))
            face_embs = face_embs.float().to(self.device)
            face_conv = self.face_embs_conv_block(self.face_embs_fc(face_embs).reshape(B, 32, 4, 4))

            # top-left for face area
            if self.n_embs > 0:
                embs_conv[:, :, :32, :32] = face_conv

            if self.n_verts_enc_channels > 0:
                verts_code[:, :, :32, :32] = face_conv

            if self.n_pose_enc_channels > 0:
                pose_conv[:, :, :32, :32] = face_conv[:, :16, :, :]

            # img_verts_code = verts_code[0, 0, :, :].detach().cpu().numpy()
            # img_face_conv = face_conv[0, 0, :, :].detach().cpu().numpy()
            # img_embs_conv = embs_conv[0, 0, :, :].detach().cpu().numpy()
            # from model import libcore
            # import cv2
            # cv2.imwrite('e:/dummy/verts_code.jpg', libcore.colorizeWeightsMap(img_verts_code))
            # cv2.imwrite('e:/dummy/embs_conv.jpg', libcore.colorizeWeightsMap(img_embs_conv))
        
        conditions = []
        if self.n_pose_enc_channels > 0:
            conditions.append(pose_conv)
        if self.n_embs > 0:
            conditions.append(embs_conv)
        if self.n_verts_enc_channels > 0:
            conditions.append(verts_code)

        cond_conv = torch.cat(conditions, axis=1)
        x = self.joint_conv_block(cond_conv)                                        # Bx64x64x64

        # to groups
        x = x.repeat(B, self.n_groups, 1, 1)

        # conv decoder
        # -> Bx96(32x3)x128x128
        # -> Bx48(16x3)x256x256
        # -> Bx24(8x3)x512x512 
        # -> Bx12(4x3)x1024x1024
        for b in range(self.n_blocks):
            x = self.conv_blocks[b](x)

        # corrections by branch
        num_branches = len(self.branches)
        x_s = x.split(int(x.shape[1] / num_branches), dim=1)
        corrections = {}

        for branch_i, branch_key in enumerate(self.branches):
            x = x_s[branch_i]
            pred = self.output_nets[branch_i](x)
            pred = pred[0].permute(1, 2, 0)

            # to corrections
            k0 = 0
            for key in self.branches[branch_key]:
                if key in self.corrections:
                    dims = self.corrections[key]
                    k1 = k0 + dims
                    corrections[key] = pred[..., k0:k1]
                    k0 = k1

        return corrections
    
##################################################
# convolution decoder from pose parameter map to triplane
class PoseVAE(nn.Module):
    def __init__(self, config, bbox_min, bbox_max, 
                 smplx_params,
                 device='cuda'):
        super().__init__()

        self.config = config
        self.device = device
        
        self.update_bbox(bbox_min, bbox_max)
        self.setup_network(config, smplx_params)
    
    ##################################################
    def update_bbox(self, bbox_min, bbox_max):
        if not isinstance(bbox_min, torch.Tensor):
            bbox_min = torch.tensor(bbox_min).float()
        if not isinstance(bbox_max, torch.Tensor):
            bbox_max = torch.tensor(bbox_max).float()

        self.bbox_min = bbox_min.to(self.device)
        self.bbox_max = bbox_max.to(self.device)
        self.bbox_center = (self.bbox_min + self.bbox_max) / 2.0
        self.bbox_radius = (self.bbox_max - self.bbox_min).max()
        self.bbox_radius = self.bbox_radius * 1.2

    def normalize_position(self, x):
        x = (x - self.bbox_center) / self.bbox_radius + 0.5
        return x
    
    ##################################################
    def setup_network(self, config, smplx_params):
        smplx_model = smplx_utils.create_smplx_model(**smplx_params, skip_betas=True, 
                                                     skip_v_template=True, skip_poses=True)
        smplx_model = smplx_model.cuda()
        
        if config.get('pose_mapper', False) and config.pose_mapper.get('resolution', 0) > 0:
            mesh = smplx_utils.convert_smplx_to_meshcpu(smplx_model)
            self.pose_maper = PoseMapper(config.pose_mapper, mesh)
            self.encoder = Encoder(config.encoder, smplx_model)
            self.with_encoder = True
        else:
            self.with_encoder = False

        config.decoder.n_pose_dims = (smplx_model.J_regressor.shape[0] - 1) * 3
        self.decoder = Decoder(config.decoder, smplx_model)

        self.smplx_model = smplx_model

    ##################################################    
    def forward(self, full_pose, face_embs=None):
        cond_params = smplx_utils.set_full_pose_to_params(full_pose)
        if 'betas' in cond_params:
            cond_params.pop('betas')
        if 'v_template' in cond_params:
            cond_params.pop('v_template')
        if 'global_orient' in cond_params:
            cond_params.pop('global_orient')
        if 'transl' in cond_params:
            cond_params.pop('transl')

        # rasterize to verts cond
        # verts_cond: vertex map 3xHxW
        if self.with_encoder:
            out = self.smplx_model(**cond_params)
            verts_cond = self.pose_maper(out['vertices'].contiguous())
            enc_preds = self.encoder(verts_cond)
        else:
            enc_preds = {}

        corrections = self.decoder.forward(full_pose, 
                                           enc_preds.get('embs', None), 
                                           enc_preds.get('verts_code', None), 
                                           face_embs)
        return corrections
    

