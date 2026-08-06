# Differentiable Rendering with nvdiffrast.
# https://nvlabs.github.io/nvdiffrast/#mipmaps-and-texture-dimensions
# Contributer(s): Neil Z. Shao
# All rights reserved 2023.
import torch
import torch.nn.functional as thf
import nvdiffrast.torch as dr
from .transform import *

# https://github.com/NVlabs/nvdiffrast/blob/main/samples/torch/earth.py#L23
def _transform_verts(mvp, verts):
    mvp = torch.from_numpy(mvp).cuda() if isinstance(mvp, np.ndarray) else mvp
    verts_homo = torch.cat([verts, torch.ones([*verts.shape[:-1], 1]).cuda()], axis=-1)
    return torch.einsum('bnj,bij->bni', verts_homo, mvp)

def _transform_norms(mv, norms):
    t_mv = torch.from_numpy(mv).cuda() if isinstance(mv, np.ndarray) else mv
    norms_homo = torch.cat([norms, torch.zeros([*norms.shape[:-1], 1]).cuda()], axis=-1)
    return torch.einsum('bnj,bij->bni', norms_homo, t_mv)

# https://github.com/NVlabs/nvdiffrecmc/blob/main/render/util.py#L62
def _pixel_grid(width, height, center_x = 0.5, center_y = 0.5):
    y, x = torch.meshgrid(
            (torch.arange(0, height, dtype=torch.float32, device="cuda") + center_y) / height, 
            (torch.arange(0, width, dtype=torch.float32, device="cuda") + center_x) / width,
            indexing='ij')
    return torch.stack((x, y), dim=-1)

# get bg color
def _get_preset_color(key):
    if key == 'white':
        color = torch.tensor([1, 1, 1], dtype=torch.float32, device='cuda')
    elif key == 'black':
        color = torch.tensor([0, 0, 0], dtype=torch.float32, device='cuda')
    else:
        color = torch.rand((3,), dtype=torch.float32, device='cuda')
    return color

# Differentiable Rendering with nvdiffrast
# !Important: choose the working coordinate first
#       |  ^ z
#       | /
#   --- + --> x
#      /|
#     / v y
# - for OpenCV coordinates:
#   > mesh = libcore.MeshCpu(obj_fn)
#   > atlas = cv2.imread(atlas_fn)
#   > nvmesh = NvdiffrastMesh(mesh, atlas=atlas, coordinate='opencv')
#   > img = nvmesh.rasterizeToCamera(cam_gl)['image']
#
#     y ^ /  
#       |/
#   --- + --> x
#      /|
#   z v |
# - for OpenGL coordinates:
#   > mesh = libcore.MeshCpu(obj_fn)
#   > mesh.flipToOpenGL()
#   > atlas = cv2.imread(atlas_fn)
#   > nvmesh = NvdiffrastMesh(mesh, atlas=atlas, coordinate='opengl')
#   > cam_gl = cam.toOpenGL()
#   > img = nvmesh.rasterizeToCamera(cam_gl)['image']
#
class NvdiffrastMesh:
    def __init__(self, mesh=None, coordinate='opengl',
                 tex=None, atlas=None, 
                 enable_mip=False, enable_jitter=True,
                 seam_edges=None,
                 backend='gl'):
        self.coordinate = coordinate
        self.backend = backend

        if mesh is not None:
            self.set_mesh(mesh)
        
        # use external texture
        self.atlas = atlas
        if tex is not None:
            self.use_ext_tex = True
            self.tex = tex
        else:
            self.use_ext_tex = False
            if self.atlas is not None:
                self.tex = torch.from_numpy(self.atlas.astype(np.float32)).cuda() / 255.0
            else:
                self.tex = None

        # seam edges
        if seam_edges is not None:
            self.seam_edges = torch.from_numpy(seam_edges.astype(np.int32)).cuda()
        
        self.enable_mip = enable_mip
        self.max_mip_level = 5

        self.enable_jitter = enable_jitter
        self.pos_gradient_boost = 1.0

        if self.backend == 'gl':
            self.glctx = dr.RasterizeGLContext()
        else:
            self.glctx = dr.RasterizeCudaContext()
    
    def set_mesh(self, mesh):
        self.mesh = mesh

        self.ori_vertices = torch.from_numpy(self.mesh.V.astype(np.float32)).unsqueeze(0).cuda()
        self.vertices = torch.from_numpy(self.mesh.V.astype(np.float32)).unsqueeze(0).cuda()
        self.normals = torch.from_numpy(self.mesh.N.astype(np.float32)).unsqueeze(0).cuda()
        self.faces = torch.from_numpy(self.mesh.F.astype(np.int32)).cuda()
        if mesh.TC is not None:
            self.uvs = torch.from_numpy(self.mesh.TC.astype(np.float32)).unsqueeze(0).cuda()
            self.uvs[..., -1] = 1.0 - self.uvs[..., -1]
        if self.mesh.FTC is not None:
            self.uv_faces = torch.from_numpy(self.mesh.FTC.astype(np.int32)).cuda()

    # every v corresponds to multiple uv
    # def update_v2uv(self):
    #     self.v2uv = torch.zeros(self.vertices.shape[-2]).long().cuda()
    #     self.v2uv[self.faces.reshape(-1).long()] = self.uv_faces.reshape(-1).long()
    #     v_uv_pair = torch.stack([self.faces.reshape(-1), self.uv_faces.reshape(-1)], dim=-1).long()
    #     self.v_uv_pair = v_uv_pair[v_uv_pair[:, 0].sort()[1]]
    #     self.v_uv_count = self.v_uv_pair[:, 0].unique(return_counts=True)[1]
    
    def update_v2uv(self, n_max=8):
        """Computes mapping from vertex indices to texture indices.

        Args:
            vi: [F, 3], triangles
            vti: [F, 3], texture triangles
            n_max: int, max number of texture locations

        Returns:
            [n_verts, n_max], texture indices
        """
        n_verts = self.vertices.shape[-2]
        vi = self.faces.detach().cpu().numpy()
        vti = self.uv_faces.detach().cpu().numpy()

        v2uv_dict = {}
        for i_v, i_uv in zip(vi.reshape(-1), vti.reshape(-1)):
            v2uv_dict.setdefault(i_v, set()).add(i_uv)
        assert len(v2uv_dict) == n_verts
        v2uv = np.zeros((n_verts, n_max), dtype=np.int32)
        for i in range(n_verts):
            vals = sorted(v2uv_dict[i])[:n_max]
            v2uv[i, :] = vals[0]
            v2uv[i, :len(vals)] = np.array(vals)

        self.v2uv = torch.tensor(v2uv).long().cuda()

    def updateMeshDeform(self, deform_verts):
        self.vertices = self.vertices + deform_verts.unsqueeze(0)
        #vertices_cpu = self.vertices.squeeze(0).detach().cpu().numpy()
        #normals_cpu = igl.per_vertex_normals(vertices_cpu, self.mesh.F)
        #self.normals = torch.from_numpy(normals_cpu.astype(np.float32)).unsqueeze(0).cuda()
        #return vertices_cpu

    @property
    def texture(self):
        if isinstance(self.tex, torch.Tensor):
            return self.tex
        return self.tex.texture

    @property
    def tex_params(self):
        if isinstance(self.tex, torch.Tensor):
            return [self.tex]
        return self.tex.parameters()

    def antialias(self, image, rast_out, verts_clip, verts_faces, bg_clr=0):
        image = dr.antialias(torch.where(rast_out[..., -1:] != 0, image, bg_clr).contiguous(), 
                             rast_out.contiguous(), verts_clip.contiguous(), verts_faces.contiguous(), 
                             pos_gradient_boost=self.pos_gradient_boost)
        return image

    def rasterizeTo(self, width, height, verts_clip, verts_faces,
                    with_texture=True, with_vertex=False,
                    with_normal=False, with_depth=False,
                    with_attr=None, with_attr_uv=None,
                    bg_clr=0):
        
        if isinstance(bg_clr, str):
            bg_clr = _get_preset_color(bg_clr)

        rast_out, rast_out_db = dr.rasterize(self.glctx, verts_clip, verts_faces, resolution=[height, width])

        if with_depth:
            depth_clip = verts_clip[None, :, :, 3:4].detach().squeeze(0).contiguous()

        if with_texture and self.tex is None:
            print('[NvdiffrastMesh][WARNING] with_texture==True but self.tex is None!')

        face_idx = (rast_out[..., -1:] - 1).long()
        mask = (face_idx >= 0)

        # rasterized
        rlt = {
            # 'soft_mask': soft_mask,
            'mask': mask,
            'face_idx': face_idx,
            'rast_out': rast_out,
            'verts_clip': verts_clip,
            'verts_faces': verts_faces,
        }

        # soft mask for mask loss
        soft_mask = dr.antialias(torch.where(rast_out[..., -1:] != 0, 1.0, 0.0), 
                        rast_out, verts_clip, verts_faces, 
                        pos_gradient_boost=self.pos_gradient_boost)
        rlt.update({ 'soft_mask': soft_mask })

        if with_texture:
            # internal texture
            if isinstance(self.tex, torch.Tensor):
                if self.enable_mip or self.enable_jitter:
                    texc, texd = dr.interpolate(self.uvs, rast_out, self.uv_faces, rast_db=rast_out_db, diff_attrs='all')
                else:
                    texc, _ = dr.interpolate(self.uvs, rast_out, self.uv_faces)

                if self.enable_mip:
                    image = dr.texture(self.tex[None, ...], texc, texd, 
                                       filter_mode='linear-mipmap-linear', 
                                       max_mip_level=self.max_mip_level)
                else:
                    image = dr.texture(self.tex[None, ...], texc, filter_mode='linear')
            # external texture
            else:
                texc, texd = dr.interpolate(self.uvs, rast_out, self.uv_faces, rast_db=rast_out_db, diff_attrs='all')
                if self.enable_mip:
                    tex, mips = self.tex.get_texture_mips()
                    image = dr.texture(tex, texc, texd, mip=mips, 
                                       filter_mode='linear-mipmap-linear')
                else:
                    image = dr.texture(tex[None, ...], texc, filter_mode='linear')

            # Important! https://nvlabs.github.io/nvdiffrast/#antialiasing
            # dr.texture has no grad on uv (no grad on vertices). so loss on mask cannot optimize vertices
            # after dr.antialias, image will have has on vertices
            image = dr.antialias(torch.where(rast_out[..., -1:] != 0, image, bg_clr), 
                                 rast_out, verts_clip, verts_faces, 
                                 pos_gradient_boost=self.pos_gradient_boost)

            rlt.update({ 'texture_coords': texc })
            rlt.update({ 'image': image })

            if self.enable_jitter:
                offset = torch.normal(mean=0, std=0.001, size=(1, height, width, 2), device="cuda")

                # offset = torch.normal(mean=0, std=0.05, size=(1, height, width, 2), device="cuda")
                # offset = (offset * texd.abs().mean(dim=-1)[..., None]).detach()

                jitter = (_pixel_grid(width, height)[None, ...] + offset).contiguous()
                mask_tap = dr.texture(soft_mask.contiguous(), jitter, filter_mode='linear', boundary_mode='clamp')
                grad_weight = soft_mask * mask_tap
                
                image_jitter = dr.texture(image.contiguous(), jitter, filter_mode='linear', boundary_mode='clamp')
                image_grad = torch.abs(image_jitter - image) * grad_weight
                rlt.update({ 'image_grad': image_grad })

        if with_vertex:
            vertices, _ = dr.interpolate(self.vertices, rast_out, self.faces)
            rlt.update({ 'vertices': vertices })    

        if with_depth:
            depths, _ = dr.interpolate(depth_clip, rast_out, self.faces)
            rlt.update({ 'depths': depths })

        if with_normal:
            normals, _ = dr.interpolate(self.normals, rast_out, self.faces)
            normals[normals.isnan()] = 0.0
            normals = normals * soft_mask
            normals = torch.nn.functional.normalize(normals, p=2, dim=3)
            rlt.update({ 'normals': normals })

        # additional attribute
        if with_attr is not None:
            attributes, _ = dr.interpolate(with_attr, rast_out, self.faces)
            rlt.update({ 'attributes': attributes })

        # additional texture
        if with_attr_uv is not None:
            texc, _ = dr.interpolate(self.uvs, rast_out, self.uv_faces)
            if isinstance(with_attr_uv, torch.Tensor):
                attributes, _ = dr.texture(with_attr_uv, texc, filter_mode='linear')
                rlt.update({ 'attributes': attributes })
            elif isinstance(with_attr_uv, dict):
                for key in with_attr_uv:
                    attr_uv = with_attr_uv[key]
                    attr_image = dr.texture(attr_uv, texc, filter_mode='linear')
                    rlt.update({ key: attr_image })

        return rlt

    # rasterize to libcore.Camera
    # for coordinate=='opengl':
    #   convert camera to opengl with cam.toOpenGL()
    # https://github.com/NVlabs/nvdiffrast/blob/main/samples/torch/earth.py#L28
    def rasterizeToCamera(self, cam, bg_clr=0, near=0.1, far=100.0,
                          with_texture=True, with_vertex=False,
                          with_normal=False, with_depth=False,
                          with_attr=None, with_attr_uv=None):
        # perspective projection matrix from camera intrinsics
        proj = perspectiveFromCamera(cam, near=near, far=far, coordinate=self.coordinate)
        # model view matrix from camera extrinsics
        mv = makeTransform(cam.R, cam.t)

        mvp = np.matmul(proj, mv).astype(np.float32)
        mvp = torch.from_numpy(mvp).cuda()[None, ...]

        #depth in 4th dimensionality
        verts_clip = _transform_verts(mvp, self.vertices).contiguous()

        height, width = cam.h, cam.w
        if self.backend == 'cuda':
            width = int(np.ceil(width / 8) * 8)
            height = int(np.ceil(height / 8) * 8)
        
        rlt = self.rasterizeTo(width, height, verts_clip, self.faces,
                               with_texture=with_texture, with_vertex=with_vertex,
                               with_normal=with_normal, with_depth=with_depth,
                               with_attr=with_attr, with_attr_uv=with_attr_uv,
                               bg_clr=bg_clr)
        
        if width != cam.w or height != cam.h:
            for key in rlt:
                if isinstance(rlt[key], torch.Tensor) and len(rlt[key].shape) == 4:
                    rlt[key] = rlt[key][:, :cam.h, :cam.w]
        return rlt
    
    def rasterizeToCameras(self, cams, bg_clr=0, near=0.1, far=100.0,
                          with_texture=True, with_vertex=False,
                          with_normal=False, with_depth=False,
                          with_attr=None, with_attr_uv=None):
        mvp_all = []
        for cam in cams:
            # perspective projection matrix from camera intrinsics
            proj = perspectiveFromCamera(cam, near=near, far=far, coordinate=self.coordinate)
            # model view matrix from camera extrinsics
            mv = makeTransform(cam.R, cam.t)

            mvp = np.matmul(proj, mv).astype(np.float32)
            mvp = torch.from_numpy(mvp).cuda()
            mvp_all.append(mvp)
        mvp_all = torch.stack(mvp_all, dim=0)

        #depth in 4th dimensionality
        verts_clip = _transform_verts(mvp_all, self.vertices).contiguous()

        H, W = cams[0].h, cams[0].w
        height, width = H, W
        if self.backend == 'cuda':
            width = int(np.ceil(width / 8) * 8)
            height = int(np.ceil(height / 8) * 8)
        
        rlt = self.rasterizeTo(width, height, verts_clip, self.faces,
                               with_texture=with_texture, with_vertex=with_vertex,
                               with_normal=with_normal, with_depth=with_depth,
                               with_attr=with_attr, with_attr_uv=with_attr_uv,
                               bg_clr=bg_clr)
        
        if width != W or height != H:
            for key in rlt:
                if isinstance(rlt[key], torch.Tensor) and len(rlt[key].shape) == 4:
                    rlt[key] = rlt[key][:, :H, :W]
        return rlt

    def rasterizeToAtlas(self, width, height, with_texture=False, with_vertex=True, 
                          with_normal=False, with_depth=False,
                          with_attr=None):
        verts_clip = self.uvs * 2.0 - 1.0
        verts_clip = torch.concat([verts_clip, torch.zeros_like(verts_clip[..., :1]),
                                   torch.ones_like(verts_clip[..., :1])], dim=-1)
        verts_faces = self.uv_faces
        return self.rasterizeTo(width, height, verts_clip, verts_faces,
                                with_texture=with_texture, with_vertex=with_vertex,
                                with_normal=with_normal, with_depth=with_depth,
                                with_attr=with_attr)

    # sample edge colors in seam edge pairs
    def sample_seam_colors(self, n_seams_pairs):
        idxs = torch.randint(0, self.seam_edges.shape[0], size=(n_seams_pairs,), device=self.seam_edges.device)
        theta = torch.rand_like(idxs.float())

        select_seam_edges = self.seam_edges[idxs].long()

        edge0_uv_idx0 = self.uv_faces[select_seam_edges[:, 0], select_seam_edges[:, 1]].long()
        edge0_uv_idx1 = self.uv_faces[select_seam_edges[:, 0], (select_seam_edges[:, 1] + 1) % 3].long()

        edge1_uv_idx0 = self.uv_faces[select_seam_edges[:, 2], (select_seam_edges[:, 3] + 1) % 3].long()
        edge1_uv_idx1 = self.uv_faces[select_seam_edges[:, 2], select_seam_edges[:, 3]].long()

        edge0_uv = self.uvs[:, edge0_uv_idx0] * theta[None, :, None] + self.uvs[:, edge0_uv_idx1] * (1.0 - theta[None, :, None])
        edge1_uv = self.uvs[:, edge1_uv_idx0] * theta[None, :, None] + self.uvs[:, edge1_uv_idx1] * (1.0 - theta[None, :, None])

        texc = torch.stack([edge0_uv, edge1_uv], dim=2)

        if isinstance(self.tex, torch.Tensor):
            seam_rgbs = dr.texture(self.tex[None, ...], texc, filter_mode='linear')
        else:
            tex, mips = self.tex.get_texture_mips()
            seam_rgbs = [dr.texture(tex, texc, filter_mode='linear')]
            if len(mips) > 0:
                # seam_rgbs.append(dr.texture(mips[0], texc, filter_mode='linear'))
                for i in range(0, len(mips)):
                    texture = mips[i]
                    lvl_seam_rgbs = dr.texture(texture, texc, filter_mode='linear')
                    seam_rgbs.append(lvl_seam_rgbs)
            seam_rgbs = torch.concat(seam_rgbs, dim=0)

        return seam_rgbs

    # sample on uv BHWC or HWC
    def sample_uv(self, value_map, sample_uv2v=True, hwc2chw=True,
                  mode="bilinear", align_corners=True, flip_uvs=False):
        # sample on uv
        uv_coords = self.uvs
        v2uv = self.v2uv

        # check input
        if len(value_map.shape) == 3:
            value_map = value_map[None, ...]
        if hwc2chw:
            value_map = value_map.permute(0, 3, 1, 2)

        # sampling
        batch_size = value_map.shape[0]
        if flip_uvs:
            uv_coords = uv_coords.clone()
            uv_coords[..., 1] = 1.0 - uv_coords[..., 1]

        uv_coords_norm = (uv_coords * 2.0 - 1.0)[:, :, None, :].expand(
            batch_size, -1, -1, -1
        )

        values = (
            thf.grid_sample(value_map, uv_coords_norm, align_corners=align_corners, mode=mode)
            .squeeze(-1)
            .permute((0, 2, 1))
        )

        if sample_uv2v:
            v2uv = self.v2uv
            values_duplicate = values[:, v2uv]
            values = values_duplicate.mean(2)
        return values
