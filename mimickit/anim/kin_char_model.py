import abc
import enum
import numpy as np
import torch
import xml.etree.ElementTree as ET

import util.torch_util as torch_util

class JointType(enum.Enum):
    ROOT = 0
    HINGE = 1
    SPHERICAL = 2
    FIXED = 3

class Joint():
    def __init__(self, name, joint_type, axis):
        self.name = name
        self.joint_type = joint_type
        self.axis = axis
        return

    def get_dof_dim(self):
        if (self.joint_type == JointType.ROOT):
            dof_dim = 0
        elif (self.joint_type == JointType.HINGE):
            dof_dim = 1
        elif (self.joint_type == JointType.SPHERICAL):
            dof_dim = 3
        elif (self.joint_type == JointType.FIXED):
            dof_dim = 0
        else:
            assert(False), "Unsupported joint type: {:s}".format(self.joint_type)
        return dof_dim


class KinCharModel():
    def __init__(self, device):
        self._device = device
        return

    def init(self, body_names, parent_indices, local_translation, local_rotation, joints):
        num_bodies = len(body_names)
        assert(len(parent_indices) == num_bodies)
        assert(len(local_translation) == num_bodies)
        assert(len(local_rotation) == num_bodies)
        assert(len(joints) == num_bodies)

        self._body_names = body_names
        self._parent_indices = np.array(parent_indices)
        self._local_translation = torch.tensor(np.array(local_translation), device=self._device, dtype=torch.float32)
        self._local_rotation = torch.tensor(np.array(local_rotation), device=self._device, dtype=torch.float32)
        self._joints = joints
        
        self._joint_dof_idx, self._dof_size = self._build_dof_indices(self._joints)
        self._name_body_map = self._build_name_body_map()
        self._build_dof_rot_tensors()
        self._build_fk_tensors()
        return

    @abc.abstractmethod
    def load(self, char_file):
        return
    
    @abc.abstractmethod
    def save(self, output_file):
        return
    
    def get_body_names(self):
        return self._body_names

    def get_parent_id(self, j):
        return self._parent_indices[j]

    def get_dof_size(self):
        return self._dof_size

    def get_num_joints(self):
        return len(self._joints)

    def get_joint(self, j):
        return self._joints[j]

    def get_joint_name(self, j):
        return self._joints[j].name

    def get_joint_type(self, j):
        return self.get_joint(j).joint_type
    
    def get_joint_dof_idx(self, j):
        dof_idx = self._joint_dof_idx[j]
        return dof_idx

    def get_joint_dof_dim(self, j):
        dof_dim = self.get_joint(j).get_dof_dim()
        return dof_dim

    def get_joint_dof(self, j, dof):
        # returns the dofs of joint j from the full dof vector
        dof_idx = self.get_joint_dof_idx(j)
        dof_dim = self.get_joint_dof_dim(j)
        j_dof = dof[..., dof_idx:dof_idx + dof_dim]
        return j_dof

    def set_joint_dof(self, j, j_dof, out_dof):
        # writes the dofs of joint j into the full dof vector out_dof
        dof_idx = self.get_joint_dof_idx(j)
        dof_dim = self.get_joint_dof_dim(j)
        out_dof[..., dof_idx:dof_idx + dof_dim] = j_dof
        return

    def get_body_name(self, body_id):
        return self._body_names[body_id]
    
    def get_body_id(self, body_name):
        assert body_name in self._name_body_map
        return self._name_body_map[body_name]
    
    def get_joint_id(self, body_name):
        assert body_name in self._name_body_map
        return self._name_body_map[body_name] -1 # joint arrays exclude the root
    
    def dof_to_rot(self, dof):
        rot_shape = dof.shape[:-1] + (self.get_num_joints() - 1, 4)
        joint_rot = self._identity_rot.to(dof.dtype).expand(rot_shape).clone()

        hinge_idx = JointType.HINGE.value
        if (self._joint_type_idx[hinge_idx].shape[0] > 0):
            hinge_dof = dof[..., self._joint_type_dof_idx[hinge_idx]]
            hinge_rot = torch_util.axis_angle_to_quat(self._hinge_axis, hinge_dof)
            joint_rot[..., self._joint_type_idx[hinge_idx], :] = hinge_rot

        sph_idx = JointType.SPHERICAL.value
        if (self._joint_type_idx[sph_idx].shape[0] > 0):
            sph_dof = dof[..., self._joint_type_dof_idx[sph_idx]].unflatten(-1, (-1, 3))
            joint_rot[..., self._joint_type_idx[sph_idx], :] = torch_util.exp_map_to_quat(sph_dof)

        return joint_rot

    def rot_to_dof(self, rot):
        # every dof belongs to a hinge or spherical joint, so all entries of dof are written below
        dof_shape = rot.shape[:-2] + (self.get_dof_size(),)
        dof = torch.empty(dof_shape, device=rot.device, dtype=rot.dtype)

        hinge_idx = JointType.HINGE.value
        if (self._joint_type_idx[hinge_idx].shape[0] > 0):
            hinge_rot = rot[..., self._joint_type_idx[hinge_idx], :]
            hinge_dof = torch_util.quat_twist_angle(hinge_rot, self._hinge_axis)
            dof[..., self._joint_type_dof_idx[hinge_idx]] = hinge_dof

        sph_idx = JointType.SPHERICAL.value
        if (self._joint_type_idx[sph_idx].shape[0] > 0):
            sph_rot = rot[..., self._joint_type_idx[sph_idx], :]
            sph_dof = torch_util.quat_to_exp_map(sph_rot).flatten(-2)
            dof[..., self._joint_type_dof_idx[sph_idx]] = sph_dof

        return dof

    def forward_kinematics(self, root_pos, root_rot, joint_rot):
        # Internally rotations use a component-major layout [4, num_bodies + 1, batch], where
        # the extra body slot holds an identity rotation that acts as the root's ancestor.
        num_bodies = self.get_num_joints()
        batch_shape = root_pos.shape[:-1]
        batch_size = batch_shape.numel()

        # one GEMM applies the local rotations (local_rot * joint_rot) and changes the layout
        x = torch.cat([root_rot.reshape(batch_size, 4),
                       joint_rot.reshape(batch_size, 4 * (num_bodies - 1))], dim=-1)
        body_rot = torch.addmm(self._fk_identity_slot, self._fk_input_mat, x.T)
        body_rot = body_rot.view(4, num_bodies + 1, batch_size)

        def fk_quat_mul(a, b):
            outer = (a.unsqueeze(1) * b.unsqueeze(0)).view(16, -1)
            return torch.mm(self._fk_quat_mul_mat, outer).view(a.shape)

        # world rotations by pointer jumping
        for ancestors in self._fk_jump_ancestors:
            body_rot = fk_quat_mul(body_rot.index_select(1, ancestors), body_rot)
        body_rot = body_rot[:, :-1]
        
        rot_outer = (body_rot.unsqueeze(0) * body_rot.unsqueeze(1)).view(16 * num_bodies, batch_size)
        body_pos = torch.addmm(self._fk_pos_offset, rot_outer.T, self._fk_pos_mat)
        body_pos = body_pos.view(batch_shape + (num_bodies, 3)) + root_pos.unsqueeze(-2)

        body_rot = body_rot.permute(2, 1, 0).reshape(batch_shape + (num_bodies, 4))
        return body_pos, body_rot
    
    def compute_frame_dof_vel(self, joint_rot, dt):
        joint_rot0 = joint_rot[:-1, :, :]
        joint_rot1 = joint_rot[1:, :, :]
        dof_vel = self.compute_dof_vel(joint_rot0, joint_rot1, dt)
        
        final_vels = dof_vel[-1:, :]
        dof_vel = torch.cat([dof_vel, final_vels], dim=-2)

        return dof_vel

    def compute_dof_vel(self, joint_rot0, joint_rot1, dt):
        drot = torch_util.quat_mul(torch_util.quat_conjugate(joint_rot0), joint_rot1)
        drot = torch_util.quat_normalize(drot)

        dof_shape = drot.shape[:-2] + (self.get_dof_size(),)
        dof_diff = torch.empty(dof_shape, device=drot.device, dtype=drot.dtype)

        hinge_idx = JointType.HINGE.value
        if (self._joint_type_idx[hinge_idx].shape[0] > 0):
            hinge_drot = drot[..., self._joint_type_idx[hinge_idx], :]
            hinge_exp_map = torch_util.quat_to_exp_map(hinge_drot)
            dof_diff[..., self._joint_type_dof_idx[hinge_idx]] = torch.sum(self._hinge_axis * hinge_exp_map, dim=-1)

        sph_idx = JointType.SPHERICAL.value
        if (self._joint_type_idx[sph_idx].shape[0] > 0):
            sph_drot = drot[..., self._joint_type_idx[sph_idx], :]
            dof_diff[..., self._joint_type_dof_idx[sph_idx]] = torch_util.quat_to_exp_map(sph_drot).flatten(-2)

        dof_vel = dof_diff / dt
        return dof_vel
    
    def _build_name_body_map(self):
        name_body_map = dict()

        for body_id, body_name in enumerate(self._body_names):
            name_body_map[body_name] = body_id

        return name_body_map
    
    def _build_root_joint(self):
        joint = Joint(name="root",
                      joint_type=JointType.ROOT,
                      axis=None)
        return joint
    
    def _build_dof_indices(self, joints):
        # returns the index of each joint's first dof in the dof vector, and the total number of dofs
        joint_dof_idx = []
        dof_size = 0
        for curr_joint in joints:
            joint_dof_idx.append(dof_size)
            if (curr_joint is not None):
                dof_dim = curr_joint.get_dof_dim()
                dof_size += dof_dim
        
        return joint_dof_idx, dof_size

    def _build_name_body_map(self):
        name_body_map = dict()
        for body_id, body_name in enumerate(self._body_names):
            name_body_map[body_name] = body_id

        return name_body_map
    
    def _build_body_children_map(self, parent_indices):
        num_bodies = len(parent_indices)
        body_children = [[] for _ in range(num_bodies)]
        for j in range(num_bodies):
            parent_idx = parent_indices[j]
            if (parent_idx != -1):
                body_children[parent_idx].append(j)
        
        return body_children

    def _build_dof_rot_tensors(self):
        def idx_tensor(x):
            return torch.tensor(x, device=self._device, dtype=torch.long)

        self._identity_rot = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self._device)

        joint_ids = [[] for _ in JointType]
        for j in range(1, self.get_num_joints()):
            joint_type = self.get_joint_type(j)
            joint_ids[joint_type.value].append(j)

        self._joint_type_idx = []
        self._joint_type_dof_idx = []
        for joint_type in JointType:
            ids = joint_ids[joint_type.value]
            joint_idx = [j - 1 for j in ids]
            dof_idx = [self._joint_dof_idx[j] + i for j in ids for i in range(self.get_joint_dof_dim(j))]

            self._joint_type_idx.append(idx_tensor(joint_idx))
            self._joint_type_dof_idx.append(idx_tensor(dof_idx))

        all_dof_idx = torch.cat(self._joint_type_dof_idx)
        assert(sorted(all_dof_idx.tolist()) == list(range(self._dof_size))), \
            "Joint dof indices must cover every dof exactly once"

        hinge_ids = joint_ids[JointType.HINGE.value]
        hinge_axis = torch.zeros([len(hinge_ids), 3], device=self._device)
        for k, j in enumerate(hinge_ids):
            joint = self.get_joint(j)
            hinge_axis[k] = torch_util.normalize(joint.axis.reshape(3))
        self._hinge_axis = hinge_axis
        return

    def _build_fk_tensors(self):
        # precomputes the constant tensors used by forward_kinematics(),
        # assumes the local transforms are not modified after init()
        num_bodies = len(self._parent_indices)
        parents = [int(p) for p in self._parent_indices]
        basis = torch.eye(4, dtype=torch.float64)

        # quat_mul is bilinear: quat_mul(a, b)[c] = sum_mn a[m] b[n] quat_mul_mat[c, 4m + n]
        quat_mul_mat = torch_util.quat_mul(basis.repeat_interleave(4, dim=0), basis.repeat(4, 1)).T
        self._fk_quat_mul_mat = quat_mul_mat.to(self._device, torch.float32)

        local_rot = self._local_rotation.cpu().double()
        input_mat = torch.zeros(4, num_bodies + 1, num_bodies, 4, dtype=torch.float64)
        input_mat[:, 0, 0, :] = basis
        for j in range(1, num_bodies):
            input_mat[:, j, j, :] = torch_util.quat_mul(local_rot[j].expand(4, 4), basis).T
        self._fk_input_mat = input_mat.view(4 * (num_bodies + 1), 4 * num_bodies).to(self._device, torch.float32)

        identity_slot = torch.zeros(4, num_bodies + 1, 1)
        identity_slot[3, num_bodies] = 1.0
        self._fk_identity_slot = identity_slot.view(-1, 1).to(self._device)

        # ancestor tables for pointer jumping, slot num_bodies is the identity
        ancestors = parents[:]
        ancestors[0] = num_bodies
        ancestors.append(num_bodies)
        self._fk_jump_ancestors = []
        while any(a != num_bodies for a in ancestors):
            self._fk_jump_ancestors.append(torch.tensor(ancestors, device=self._device))
            ancestors = [ancestors[a] for a in ancestors]

        # rot_coeffs[m, n, k] = coefficient of q[m] q[n] in quat_rotate(q, local_trans[k]) - local_trans[k],
        # symmetric in m, n and found by polarization of the quadratic form
        local_trans = self._local_translation.cpu().double()
        def quat_rotate_quadratic(q):
            v = local_trans.unsqueeze(0).expand(q.shape[0], -1, -1)
            q = q.unsqueeze(1).expand(-1, num_bodies, -1)
            return torch_util.quat_rotate(q.reshape(-1, 4), v.reshape(-1, 3)).view(v.shape) - v
        pair_rot = quat_rotate_quadratic((basis.unsqueeze(0) + basis.unsqueeze(1)).view(16, 4)).view(4, 4, num_bodies, 3)
        single_rot = quat_rotate_quadratic(basis)
        rot_coeffs = 0.5 * (pair_rot - single_rot.unsqueeze(0) - single_rot.unsqueeze(1))
        rot_coeffs[range(4), range(4)] = single_rot

        # pos_mat[(m, n, i), (j, c)]: for each joint k on the path from the root to body j,
        # adds the coefficients of rotating local_trans[k] by the world rotation of its parent i.
        # pos_offset[j] is the sum of the local translations along the same path.
        pos_mat = torch.zeros(4, 4, num_bodies, num_bodies, 3, dtype=torch.float64)
        pos_offset = torch.zeros(num_bodies, 3, dtype=torch.float64)
        for j in range(1, num_bodies):
            k = j
            while k != 0:
                pos_mat[:, :, parents[k], j, :] += rot_coeffs[:, :, k, :]
                pos_offset[j] += local_trans[k]
                k = parents[k]
        
        self._fk_pos_mat = pos_mat.view(16 * num_bodies, 3 * num_bodies).to(self._device, torch.float32)
        self._fk_pos_offset = pos_offset.view(-1).to(self._device, torch.float32)
        return