import math
import numpy as np
import torch

def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))

def normalize(x, eps: float = 1e-9):
    return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(eps)

def normalize_exp_map(exp_map):
    angle = torch.linalg.vector_norm(exp_map, dim=-1, keepdim=True).clamp_min(1e-9)
    scale = normalize_angle(angle) / angle
    norm_exp_map = exp_map * scale
    return norm_exp_map

def quat_unit(a):
    return normalize(a)

def quat_conjugate(q):
    q_conj = q.clone()
    q_conj[..., :3].neg_()
    return q_conj

def quat_pos(x):
    q = torch.where(x[..., 3:] < 0, -x, x)
    return q


_QUAT_MUL_IDX = (3, 2, 1, 0,
                 2, 3, 0, 1,
                 1, 0, 3, 2,
                 0, 1, 2, 3)
                 
_QUAT_MUL_SIGN = ((1.0, -1.0, 1.0, 1.0),
                  (1.0, 1.0, -1.0, 1.0),
                  (-1.0, 1.0, 1.0, 1.0),
                  (-1.0, -1.0, -1.0, 1.0))

_QUAT_MUL_CONJ_SIGN = ((-1.0, 1.0, -1.0, 1.0),
                       (-1.0, -1.0, 1.0, 1.0),
                       (1.0, -1.0, -1.0, 1.0),
                       (1.0, 1.0, 1.0, 1.0))

def quat_mul(a, b):
    assert a.shape == b.shape
    return _quat_mul(a, b, _QUAT_MUL_SIGN)

def _quat_mul(a, b, sign_table):
    idx = torch.tensor(_QUAT_MUL_IDX, device=a.device, dtype=torch.long)
    sign = torch.tensor(sign_table, device=a.device, dtype=a.dtype)
    mat = a[..., idx].reshape(a.shape[:-1] + (4, 4)) * sign
    quat = (mat * b.unsqueeze(-2)).sum(dim=-1)
    return quat

def quat_rotate(q, v):
    q_v = q[..., :3]
    q_w = q[..., 3:]
    t = torch.linalg.cross(q_v, v)
    return torch.add(v, torch.addcmul(torch.linalg.cross(q_v, t), q_w, t), alpha=2.0)

def axis_or_default(mask, axis):
    # returns axis where mask is true and the default axis [0, 0, 1] elsewhere
    default_axis = torch.tensor((0.0, 0.0, 1.0), device=axis.device, dtype=axis.dtype)
    return torch.where(mask.unsqueeze(-1), axis, default_axis)

def quat_to_axis_angle(q):
    eps = 1e-5
    
    # need to make sure w is not negative to calculate geodesic distance
    q = quat_pos(q)
    q_xyz = q[..., :3]
    length = torch.linalg.vector_norm(q_xyz, dim=-1)
    angle = 2.0 * torch.atan2(length, q[..., 3])

    mask = length > eps
    angle = torch.where(mask, angle, 0.0)
    axis = q_xyz / length.unsqueeze(-1)
    axis = axis_or_default(mask, axis)

    return axis, angle


_QUAT_TO_MATRIX_IDX = (5, 10,  1, 11,  2, 7,
                       1, 11,  0, 10,  6, 3,
                       2, 7,   6, 3,   0, 5)
                       
_QUAT_TO_MATRIX_COEFF = ((-2.0, -2.0), (2.0, -2.0), (2.0, 2.0),
                         (2.0, 2.0), (-2.0, -2.0), (2.0, -2.0),
                         (2.0, -2.0), (2.0, 2.0), (-2.0, -2.0))
                         
_IDENTITY_FLAT = (1.0, 0.0, 0.0,
                  0.0, 1.0, 0.0,
                  0.0, 0.0, 1.0)

def quat_to_matrix(q):
    batch_shape = q.shape[:-1]
    outer = q.unsqueeze(-1) * q.unsqueeze(-2)
    sq_norm = outer.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)

    idx = torch.tensor(_QUAT_TO_MATRIX_IDX, device=q.device, dtype=torch.long)
    products = outer.reshape(batch_shape + (16,))[..., idx].reshape(batch_shape + (9, 2))
    coeff = torch.tensor(_QUAT_TO_MATRIX_COEFF, device=q.device, dtype=q.dtype)
    terms = (products * coeff).sum(dim=-1)

    identity = torch.tensor(_IDENTITY_FLAT, device=q.device, dtype=q.dtype)
    mat = torch.addcdiv(identity, terms, sq_norm)
    return mat.reshape(batch_shape + (3, 3))


_QUAT_TO_EULER_IDX = (12, 6,  14, 1,  0, 5,  5, 10,  13, 8)
_QUAT_TO_EULER_COEFF = ((2.0, 2.0), (2.0, 2.0), (-2.0, -2.0), (-2.0, -2.0), (2.0, -2.0))
_QUAT_TO_EULER_BASE = (0.0, 0.0, 1.0, 1.0, 0.0)

def quat_to_euler_xyz(q):
    batch_shape = q.shape[:-1]
    outer = (q.unsqueeze(-1) * q.unsqueeze(-2)).reshape(batch_shape + (16,))
    idx = torch.tensor(_QUAT_TO_EULER_IDX, device=q.device, dtype=torch.long)
    products = outer[..., idx].reshape(batch_shape + (5, 2))
    coeff = torch.tensor(_QUAT_TO_EULER_COEFF, device=q.device, dtype=q.dtype)
    base = torch.tensor(_QUAT_TO_EULER_BASE, device=q.device, dtype=q.dtype)
    t = (products * coeff).sum(dim=-1) + base

    roll_yaw = torch.atan2(t[..., 0:2], t[..., 2:4])
    pitch = torch.asin(torch.clamp(t[..., 4:5], min=-1.0, max=1.0))

    return torch.cat([roll_yaw[..., 0:1], pitch, roll_yaw[..., 1:2]], dim=-1)

def angle_to_matrix(angle, axis):
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)
    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


_AXIS_ROT_COEFFS = {
    "X": ((1.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0),
          (0.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 1.0),
          (0.0, 0.0, 0.0,  0.0, 0.0, -1.0,  0.0, 1.0, 0.0)),
    "Y": ((0.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 0.0),
          (1.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 1.0),
          (0.0, 0.0, 1.0,  0.0, 0.0, 0.0,  -1.0, 0.0, 0.0)),
    "Z": ((0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 1.0),
          (1.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 0.0),
          (0.0, -1.0, 0.0,  1.0, 0.0, 0.0,  0.0, 0.0, 0.0)),
}

def euler_angle_to_matrix(euler, axis_order):
    euler = euler[None, ...] if len(euler.shape) <= 1 else euler
    num_axes = len(axis_order)

    # build the rotation matrices of all axes at once
    coeffs = [_AXIS_ROT_COEFFS[axis] for axis in axis_order]
    const, cos_coeff, sin_coeff = torch.tensor(coeffs, device=euler.device, dtype=euler.dtype).unbind(1)

    angles = euler[..., :num_axes].unsqueeze(-1)
    R_flat = torch.addcmul(const, torch.cos(angles), cos_coeff)
    R_flat = torch.addcmul(R_flat, torch.sin(angles), sin_coeff)
    R = R_flat.reshape(angles.shape[:-2] + (num_axes, 3, 3))

    # the 3x3 matrix products are computed as a broadcasted multiply and sum,
    # which is much faster on the gpu than torch.matmul for small matrices
    mat = R[..., 0, :, :]
    for i in range(1, num_axes):
        mat = (mat.unsqueeze(-1) * R[..., i, :, :].unsqueeze(-3)).sum(dim=-2)
    return mat

def matrix_to_axis_angle(R):
    trace = R[...,0,0] +  R[...,1,1] +  R[...,2,2]
    cs =  torch.clip((trace - 1) / 2, -1+1e-7, 1-1e-7)
    angle = torch.acos(cs)
    
    skew = R - R.transpose(-1, -2)
    axis = torch.stack([skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]], dim=-1)
    axis = normalize(axis)
    
    return axis, angle

def axis_angle_to_quat(axis, angle):
    # axis is assumed to be unit length
    theta = (angle / 2).unsqueeze(-1)
    xyz = axis * theta.sin()
    w = theta.cos()
    return torch.cat([xyz, w], dim=-1)

def axis_angle_to_exp_map(axis, angle):
    angle_expand = angle.unsqueeze(-1)
    exp_map = angle_expand * axis
    return exp_map

def matrix_to_quat(R):
    axis, angle = matrix_to_axis_angle(R)
    quat = quat_unit(axis_angle_to_quat(axis, angle))
    return quat

def quat_to_exp_map(q):
    eps = 1e-5
    q = quat_pos(q)
    q_xyz = q[..., 0:3]
    length = torch.linalg.vector_norm(q_xyz, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(length, q[..., 3:4])
    exp_map = torch.where(length > eps, q_xyz * (angle / length), 0.0)
    return exp_map


def matrix_to_exp_map(R):
    quat = matrix_to_quat(R)
    exp_map = quat_to_exp_map(quat)
    return exp_map

def quat_to_tan_norm(q):
    ref_axes = torch.zeros((1,) * (q.dim() - 1) + (2, 3), device=q.device, dtype=q.dtype)
    ref_axes[..., 0, 0] = 1.0
    ref_axes[..., 1, 2] = 1.0
    norm_tan = quat_rotate(q.unsqueeze(-2), ref_axes)
    norm_tan = norm_tan.reshape(q.shape[:-1] + (6,))
    return norm_tan

def tan_norm_to_matrix(tan_norm):
    tan = tan_norm[..., 0:3]
    norm = tan_norm[..., 3:6]
    col2 = torch.linalg.cross(norm, tan)
    mat3 = torch.stack([tan, col2, norm], dim=-1)
    return mat3

def tan_norm_to_quat(tan_norm):
    mat = tan_norm_to_matrix(tan_norm)
    quat = matrix_to_quat(mat)
    return quat

def exp_map_to_axis_angle(exp_map):
    min_theta = 1e-5

    angle = torch.linalg.vector_norm(exp_map, dim=-1)
    axis = exp_map / angle.unsqueeze(-1)
    angle = normalize_angle(angle)

    mask = torch.abs(angle) > min_theta
    angle = torch.where(mask, angle, 0.0)
    axis = axis_or_default(mask, axis)

    return axis, angle

def exp_map_to_quat(exp_map):
    axis, angle = exp_map_to_axis_angle(exp_map)
    q = axis_angle_to_quat(axis, angle)
    return q

def quat_diff(q0, q1):
    # quat_mul(q1, quat_conjugate(q0))
    assert q0.shape == q1.shape
    dq = _quat_mul(q1, q0, _QUAT_MUL_CONJ_SIGN)
    return dq

def quat_diff_angle(q0, q1):
    dq = quat_diff(q0, q1)
    length = torch.linalg.vector_norm(dq[..., 0:3], dim=-1)
    angle = 2.0 * torch.atan2(length, torch.abs(dq[..., 3]))
    return angle

def quat_abs(x):
    x = torch.linalg.vector_norm(x, dim=-1)
    return x

def quat_normalize(q):
    q = quat_unit(quat_pos(q))  # normalized to positive and unit quaternion
    return q

def slerp(q0, q1, t):
    assert(len(t.shape) == len(q0.shape) - 1)
    cos_half_theta = torch.sum(q0 * q1, dim=-1)

    neg_mask = cos_half_theta < 0
    q1 = torch.where(neg_mask.unsqueeze(-1), -q1, q1)
    
    cos_half_theta = torch.abs(cos_half_theta)
    cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

    half_theta = torch.acos(cos_half_theta)
    sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

    t = t.unsqueeze(-1)
    ratioA = torch.sin((1 - t) * half_theta) / sin_half_theta
    ratioB = torch.sin(t * half_theta) / sin_half_theta
    
    new_q = ratioA * q0 + ratioB * q1
    
    new_q = torch.where(sin_half_theta < 0.001, (q0 + q1) * 0.5, new_q)
    new_q = torch.where(cos_half_theta >= 1, q0, new_q)

    return new_q

def calc_heading(q):
    yz = q[..., 1:3]
    xw = q[..., 0::3]
    rot_dir_x = 1 - 2 * (yz * yz).sum(dim=-1)
    rot_dir_y = 2 * (xw * yz).sum(dim=-1)

    heading = torch.atan2(rot_dir_y, rot_dir_x)
    return heading

def heading_to_quat(heading):
    theta = (heading / 2).unsqueeze(-1)
    zeros = torch.zeros_like(theta)
    heading_q = torch.cat([zeros, zeros, theta.sin(), theta.cos()], dim=-1)
    return heading_q

def calc_heading_quat(q):
    heading = calc_heading(q)
    heading_q = heading_to_quat(heading)
    return heading_q

def calc_heading_quat_inv(q):
    heading = calc_heading(q)
    heading_q = heading_to_quat(-heading)
    return heading_q


_EULER_TO_QUAT_IDX = ((2, 3, 1), (5, 0, 7),
                      (2, 0, 4), (5, 3, 1),
                      (5, 0, 1), (2, 6, 4),
                      (2, 0, 1), (5, 3, 4))

def euler_xyz_to_quat(roll, pitch, yaw):
    roll, pitch, yaw = torch.broadcast_tensors(roll, pitch, yaw)
    half_angles = torch.stack([roll, pitch, yaw], dim=-1) * 0.5
    sin = torch.sin(half_angles)
    cos_sin = torch.cat([torch.cos(half_angles), sin, -sin], dim=-1)

    idx = torch.tensor(_EULER_TO_QUAT_IDX, device=cos_sin.device, dtype=torch.long)
    terms = cos_sin[..., idx].prod(dim=-1).reshape(roll.shape + (4, 2))
    quat = terms.sum(dim=-1)
    return quat

def euler_xyz_to_exp_map(roll, pitch, yaw):
    q = euler_xyz_to_quat(roll, pitch, yaw)
    exp_map = quat_to_exp_map(q)
    return exp_map

def quat_twist_angle(q, twist_axis):
    q = quat_pos(q)
    twist_sin = torch.sum(q[..., 0:3] * twist_axis, dim=-1)
    twist_cos = q[..., 3]
    angle = 2.0 * torch.atan2(twist_sin, twist_cos)

    # like quat_to_axis_angle(), twists with sin(angle / 2) <= eps are set to zero
    eps = 1e-5
    is_small = torch.abs(angle) <= 2.0 * math.asin(eps)
    angle = torch.where(is_small, 0.0, angle)

    return angle


def add_torch_dict(in_dict, out_dict):
    for k, v in in_dict.items():
        if (v.requires_grad):
            v = v.detach()

        if (k in out_dict):
            out_dict[k] += v
        else:
            out_dict[k] = v
    return
        
def scale_torch_dict(scale, out_dict):
    for k in out_dict.keys():
        out_dict[k] *= scale
    return

def calc_layers_out_size(layers):
    modules = list(layers.modules())
    for m in reversed(modules):
        if hasattr(m, "out_features"):
            out_size = m.out_features
            break
    return out_size

def eval_minibatch(fn, inputs, batch_size):
    if (batch_size > 0):
        x = next(iter(inputs.values()))
        n = x.shape[0]
		
        minibatch_vals = []
        for idx_beg in range(0, n, batch_size):
            idx_end = min(idx_beg + batch_size, n)
            minibatch_input = {k:v[idx_beg:idx_end] for k, v in inputs.items()}
            minibatch_vals.append(fn(**minibatch_input))
        y = torch.cat(minibatch_vals, dim=0)
                
    else:
        y = fn(**inputs)

    return y

def torch_dtype_to_numpy(torch_dtype):
    if (torch_dtype == torch.float32):
        numpy_dtype = np.float32
    elif (torch_dtype == torch.uint8):
        numpy_dtype = np.uint8
    elif (torch_dtype == torch.int64):
        numpy_dtype = np.int64
    else:
        assert(False), "Unsupported type {}".format(torch_dtype)
    return numpy_dtype

def numpy_dtype_to_torch(numpy_dtype):
    if (numpy_dtype == np.float32):
        torch_dtype = torch.float32
    elif (numpy_dtype == np.uint8):
        torch_dtype = torch.uint8
    elif (numpy_dtype == np.int64):
        torch_dtype = torch.int64
    else:
        assert(False), "Unsupported type {}".format(numpy_dtype)
    return torch_dtype

def sample_categorical(weights, n):
    # samples n indices with probability proportional to weights using the inverse cdf,
    # which unlike torch.multinomial also supports a symbolic n under torch.compile
    cdf = torch.cumsum(weights, dim=0)
    u = cdf[-1] * torch.rand([n], device=weights.device)
    idx = torch.searchsorted(cdf, u, right=True)
    idx = torch.clamp_max(idx, cdf.shape[0] - 1)
    return idx

class UInt8ToFloat(torch.nn.Module):
    def forward(self, x):
        float_x = x.type(torch.float32) / 255.0
        return float_x