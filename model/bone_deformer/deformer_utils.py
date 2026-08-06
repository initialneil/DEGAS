import torch
import torch.nn.functional as thf
from pytorch3d import ops

##################################################
def skinning(x, w, tfs, inverse=False, with_tfs=False):
    """Linear blend skinning
    Args:
        x (tensor): canonical points. shape: [B, N, D]
        w (tensor): conditional input. [B, N, J]
        tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
    Returns:
        x (tensor): skinned points. shape: [B, N, D]
    """
    x_h = thf.pad(x, (0, 1), value=1.0)

    if inverse:
        # p:n_point, n:n_bone, i,k: n_dim+1
        w_tf = torch.einsum("bpn,bnij->bpij", w, tfs)
        x_h = torch.einsum("bpij,bpj->bpi", w_tf.float().inverse(), x_h)
    else:
        if with_tfs:
            w_tf = torch.einsum("bpn,bnij->bpij", w, tfs)
            x_h = torch.einsum("bpij,bpj->bpi", w_tf.float(), x_h)
        else:
            x_h = torch.einsum("bpn,bnij,bpj->bpi", w, tfs, x_h)

    if with_tfs:
        return x_h[:, :, :3], w_tf
    else:
        return x_h[:, :, :3]

##################################################
def batched_index_select(input, index):
    selected = []
    for i in range(input.shape[0]):
        # print(input[i].shape, index[i].shape)
        select = torch.index_select(input[i], 0, index[i].view(-1))
        select = select.view((*index[i].shape, select.shape[-1]))
        # print(select.shape)
        selected.append(select)
    return torch.stack(selected)

def query_knn(query_points, refernce_points, K=7):
    if len(query_points.shape) == 2:
        query_points = query_points[None, ...]
    if len(refernce_points.shape) == 2:
        refernce_points = refernce_points[None, ...]

    distance_sqr_batch, index_batch, neighbor_points = ops.knn_points(query_points, refernce_points,
                                                                      K=K, return_nn=True)
    return distance_sqr_batch, index_batch, neighbor_points

def calc_knn_weights(query_points, refernce_points, K=7, lambda_d='auto', with_distance=False, with_knn_points=False):
    if len(query_points.shape) == 2:
        query_points = query_points[None, ...]
    if len(refernce_points.shape) == 2:
        refernce_points = refernce_points[None, ...]

    distance_batch, index_batch, neighbor_points = ops.knn_points(query_points, refernce_points,
                                                                  K=K, return_nn=True)
    distance_batch = torch.clamp(distance_batch, max=4)
    distance_batch = torch.sqrt(distance_batch)

    if lambda_d == 'auto':
        weights_conf = torch.exp(-distance_batch / (distance_batch.min(dim=-1, keepdim=True)[0] + 1e-15))
    elif isinstance(lambda_d, float):
        weights_conf = torch.exp(-distance_batch / lambda_d)
    else:
        weights_conf = torch.exp(-distance_batch)

    weights_conf = weights_conf / weights_conf.sum(-1, keepdim=True)

    if with_distance and with_knn_points:
        return index_batch, weights_conf, distance_batch, neighbor_points
    elif with_distance:
        return index_batch, weights_conf, distance_batch
    elif with_knn_points:
        return index_batch, weights_conf, neighbor_points

    return index_batch, weights_conf

def interpolate_knn_weights(index_batch, weights_conf, reference_weights):
    _reference_weights = reference_weights.reshape(*reference_weights.shape[:2], -1)

    # knn weights in batch
    interp_weights = batched_index_select(_reference_weights, index_batch)
    interp_weights = torch.sum(interp_weights * weights_conf.unsqueeze(-1), dim=-2).detach()

    interp_weights = interp_weights.reshape(*interp_weights.shape[:2], *reference_weights.shape[2:])
    return interp_weights

def interpolate_knn_values(index_batch, weights_conf, reference_values):
    if len(reference_values.shape) == 2:
        _reference_values = reference_values[None, ...]
    else:
        _reference_values = reference_values

    _interp = interpolate_knn_weights(index_batch, weights_conf, _reference_values)
    if len(reference_values.shape) == 2:
        return _interp[0]
    else:
        return _interp
        
        