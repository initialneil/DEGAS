import torch
import torch.nn.functional as F
from .deformer_utils import skinning, interpolate_knn_weights, calc_knn_weights
from .smplx_utils import create_smplx_model, convert_smplx_to_meshcpu

class SMPLXDeformer(torch.nn.Module):
    """ Deformer based on SMPLX
    - solvable smplx parameters in smplx_params
        betas: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        body_pose: Optional[Tensor] = None,
        left_hand_pose: Optional[Tensor] = None,
        right_hand_pose: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        expression: Optional[Tensor] = None,
        jaw_pose: Optional[Tensor] = None,
        leye_pose: Optional[Tensor] = None,
        reye_pose: Optional[Tensor] = None,
    """
    def __init__(self, model_path=None, gender='neutral', model_type='smplx', ext='npz',
                 scale=1.0, transl=None,
                 K=1, verbose=False, 
                 **smplx_params):
        super().__init__()

        self.model_type = model_type
        self.smplx_params = smplx_params
        self.K = K
        self.verbose = verbose

        # to cpu for creation
        for key in smplx_params:
            if isinstance(smplx_params[key], torch.Tensor):
                smplx_params[key] = smplx_params[key].cpu()
        if isinstance(transl, torch.Tensor):
            transl = transl.cpu()

        # global
        self.global_scale_c = scale

        # fix transl
        # Important! Must set transl to zeros, other tfs_c_inv will crash
        if transl is not None:
            global_transl_c = transl.detach().clone()
        else:
            global_transl_c = None

        # smplx model
        self.smplx_model = create_smplx_model(model_path, gender=gender, model_type=model_type, ext=ext,
                                              **smplx_params)

        self.register_buffer('smplx_verts', torch.Tensor())
        self.register_buffer('smplx_tfs', torch.Tensor())
        self.register_buffer('smplx_tfs_p', torch.Tensor())
        self.register_buffer('smplx_weights', torch.Tensor())

        # mark posed as canonical
        self.update(absolute=True, scale=scale, transl=transl, **smplx_params)

        self.register_buffer('smplx_verts_c', self.smplx_verts.detach().clone())
        self.register_buffer('smplx_tfs_c', self.smplx_tfs.detach().clone())
        self.register_buffer('smplx_tfs_c_inv', self.smplx_tfs_c.squeeze(0).inverse())

        if global_transl_c is None:
            global_transl_c = torch.zeros((1, 3)).to(self.smplx_verts_c)
        self.register_buffer('global_transl_c', global_transl_c)

        # update again to reset additional tfs
        self.update(absolute=False, scale=scale, transl=transl, **smplx_params)

        # show nn.Parameters in smplx
        if self.verbose:
            for name, param in self.smplx_model.named_parameters():
                if param.requires_grad:
                    print(name, param.data)

    def set_canonical(self, scale=None, transl=None, **smplx_params):
        # mark posed as canonical
        self.update(absolute=True, scale=scale, transl=transl, **smplx_params)

        self.smplx_verts_c = self.smplx_verts
        self.smplx_tfs_c = self.smplx_tfs
        self.smplx_tfs_c_inv = self.smplx_tfs_c.squeeze(0).inverse()

        if transl is not None:
            self.global_transl_c = transl

        # update again to reset additional tfs
        self.update(absolute=False, scale=scale, transl=transl, **smplx_params)

    def update(self, scale=None, transl=None, absolute=False, **smplx_params):
        # pca for hand
        if 'left_hand_pose' in smplx_params and smplx_params['left_hand_pose'].shape[-1] == 45:
            self.smplx_model.use_pca = False
        if 'pose_mean' in smplx_params:
            self.smplx_model.pose_mean = smplx_params['pose_mean']

        # transl is handled in a separated variable
        # smplx_params here must not have transl
        smplx_out = self.smplx_model(**smplx_params)
        self.smplx_out = smplx_out

        # [Warning] care about the defaults
        # if scale is None:
        #     scale = self.global_scale_c
        # if transl is None:
        #     transl = self.global_transl_c if hasattr(self, 'global_transl_c') else torch.tensor(0)
        if scale is None:
            scale = 1.0
        if transl is None:
            transl = torch.zeros_like(self.global_transl_c) if hasattr(self, 'global_transl_c') else torch.tensor(0)
        if len(transl.shape) == 2:
            transl = transl[:, None, :]

        self.smplx_verts = smplx_out['vertices'] * scale + transl.expand_as(smplx_out['vertices'])

        self.smplx_tfs = smplx_out['T']
        self.smplx_tfs[:, :, :3, :] = self.smplx_tfs[:, :, :3, :] * scale
        self.smplx_tfs[:, :, :3, 3] = self.smplx_tfs[:, :, :3, 3] + transl.expand_as(self.smplx_tfs[:, :, :3, 3]) * scale

        # posed tfs without considering canonical
        self.smplx_tfs_p = self.smplx_tfs

        # relative tfs from canonical to pose
        if not absolute:
            self.smplx_tfs = torch.einsum('bnij,njk->bnik', self.smplx_tfs, self.smplx_tfs_c_inv)

        self.smplx_weights = smplx_out['weights']
        self.smplx_joints = smplx_out['joints'] * scale + transl.expand_as(smplx_out['joints'])
                    
    def forward(self, x, knn_x=None, smplx_verts=None, smpl_tfs=None, weights=None, 
                inverse=False, with_tfs=False):
        if smpl_tfs is None:
            smpl_tfs = self.smplx_tfs
        if x.shape[0] == 0: 
            return x
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        if weights is None:
            if smplx_verts is None:
                if inverse:
                    smplx_verts = self.smplx_verts
                else:
                    smplx_verts = self.smplx_verts_c
            
            if knn_x is None:
                knn_x = x

            weights, nearest_dists = self.query_skinning_weights_smpl_multi(knn_x, smplx_verts=smplx_verts, smpl_weights=self.smplx_weights)
        else:
            nearest_dists = None

        if with_tfs:
            x_transformed, tfs = skinning(x, weights, smpl_tfs, inverse=inverse, with_tfs=with_tfs)
            return x_transformed, tfs, nearest_dists
        else:
            return skinning(x, weights, smpl_tfs, inverse=inverse), nearest_dists

    def forward_skinning(self, xc, cond=None, smpl_tfs=None):
        if smpl_tfs is None:
            smpl_tfs = self.smplx_tfs

        weights, _ = self.query_skinning_weights_smpl_multi(xc, smplx_verts=self.smplx_verts_c[0], smpl_weights=self.smplx_weights)
        x_transformed = skinning(xc, weights, smpl_tfs, inverse=False)

        return x_transformed

    def query_skinning_weights_smpl_multi(self, pts, smplx_verts=None, smpl_weights=None):
        if smplx_verts is None:
            smplx_verts = self.smplx_verts_c[0]
        if len(smplx_verts.shape) == 2:
            smplx_verts = smplx_verts.unsqueeze(0)
        if len(pts.shape) == 2:
            pts = pts.unsqueeze(0)
        if smpl_weights is None:
            smpl_weights = self.smplx_weights

        # interpolate weights from knn
        index_batch, weights_conf, distance_batch = calc_knn_weights(pts, smplx_verts, K=self.K, 
                                                                     lambda_d='auto',
                                                                     with_distance=True)

        # knn weights in batch
        weights = interpolate_knn_weights(index_batch, weights_conf, smpl_weights)

        nearest_dists = distance_batch[..., 0]
        return weights, nearest_dists

    def query_weights(self, xc):
        weights = self.forward(xc, None, return_weights=True, inverse=False)
        return weights

    def forward_skinning_normal(self, xc, normal, cond, tfs, inverse=False):
        if normal.ndim == 2:
            normal = normal.unsqueeze(0)
        w = self.query_weights(xc[0], cond)

        p_h = F.pad(normal, (0, 1), value=0)

        if inverse:
            # p:num_point, n:num_bone, i,j: num_dim+1
            tf_w = torch.einsum('bpn,bnij->bpij', w.double(), tfs.double())
            p_h = torch.einsum('bpij,bpj->bpi', tf_w.inverse(), p_h.double()).float()
        else:
            p_h = torch.einsum('bpn, bnij, bpj->bpi', w.double(), tfs.double(), p_h.double()).float()
        
        return p_h[:, :, :3]
    
    # save to obj
    def save_smplx_verts_to_obj(self, obj_fn):
        smplx_model_obj = convert_smplx_to_meshcpu(self.smplx_model, V=self.smplx_verts.squeeze(0))
        smplx_model_obj.save_to_obj(obj_fn)

    def save_smplx_verts_c_to_obj(self, obj_fn):
        smplx_model_obj = convert_smplx_to_meshcpu(self.smplx_model, V=self.smplx_verts_c.squeeze(0))
        smplx_model_obj.save_to_obj(obj_fn)


