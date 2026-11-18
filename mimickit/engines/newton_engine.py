import warp as wp
wp.config.enable_backward = False

import newton
import mujoco_warp
import numpy as np
import os
import torch
import pyglet
from PIL import Image

import engines.engine as engine
import engines.newton_recorder as newton_recorder
from util.logger import Logger

BACKGROUND_COLOR = (0.89, 0.894, 0.898)
GROUND_TEXTURE_FILE = "data/assets/ground_texture.png"
GROUND_TEXTURE_DESATURATION = 0.2
GROUND_TEXTURE_BRIGHTNESS = 0.56
GROUND_TEXTURE_ADD = 0.21
GROUND_FOG_COLOR = (0.487, 0.478, 0.454)
GROUND_TILE_SIZE = 1.0

SOLVER_OVERFLOW_CHECK_STEPS = 1000
SOLVER_OVERFLOW_ERROR_MASK = ~(mujoco_warp.OverflowType.ITERATIONS | mujoco_warp.OverflowType.LS_ITERATIONS)

def str_to_key_code(key_str):
    key_name = key_str.upper()

    if (len(key_name) == 1 and key_name.isdigit()):
        key_name = "_" + key_name
    elif (key_name == "ESC"):
        key_name = "ESCAPE"
    elif (key_name == "RETURN"):
        key_name = "ENTER"
    elif (key_name == "DELETE"):
        key_name = "DEL"
    elif (key_name == "LEFT_SHIFT"):
        key_name = "LSHIFT"
    elif (key_name == "LEFT_CONTROL"):
        key_name = "LCTRL"
    elif (key_name == "LEFT_ALT"):
        key_name = "LALT"
    elif (key_name == "RIGHT_SHIFT"):
        key_name = "RSHIFT"
    elif (key_name == "RIGHT_CONTROL"):
        key_name = "RCTRL"
    elif (key_name == "RIGHT_ALT"):
        key_name = "RALT"
    
    key_code = getattr(pyglet.window.key, key_name)
    return key_code

class SimState:
    def __init__(self, sim_model, num_envs):
        self._sim_model = sim_model
        self.raw_state = self._sim_model.state()
        self.eval_fk()

        self._torch_joint_q = wp.to_torch(self.raw_state.joint_q)
        self._torch_joint_qd = wp.to_torch(self.raw_state.joint_qd)
        self._torch_body_q = wp.to_torch(self.raw_state.body_q)
        self._torch_body_qd = wp.to_torch(self.raw_state.body_qd)
        
        self._wp_body_force = wp.clone(self.raw_state.body_f)
        self._torch_body_force = wp.to_torch(self._wp_body_force)
        self._torch_joint_X_p = wp.to_torch(sim_model.joint_X_p)

        mujoco_state = getattr(self.raw_state, "mujoco", None)
        self._has_actuator_forces = getattr(mujoco_state, "qfrc_actuator", None) is not None
        
        self.root_pos = []
        self.root_rot = []
        self.root_vel = []
        self.root_ang_vel = []
        self.dof_pos = []
        self.dof_vel = []
        self.dof_actuator_force = []

        self.body_pos = []
        self.body_rot = []
        self.body_vel = []
        self.body_ang_vel = []
        self.body_force = []

        articulation_start = sim_model.articulation_start.numpy()
        joint_q_start = sim_model.joint_q_start.numpy()
        joint_qd_start = sim_model.joint_qd_start.numpy()
        joint_types = self._sim_model.joint_type.numpy()

        num_objs = sim_model.articulation_count
        objs_per_env = int(num_objs / num_envs)

        joint_q = self._torch_joint_q.view(num_envs, -1)
        joint_qd = self._torch_joint_qd.view(num_envs, -1)
        body_q = self._torch_body_q.view(num_envs, -1, 7)
        body_qd = self._torch_body_qd.view(num_envs, -1, 6)
        body_force = self._torch_body_force.view(num_envs, -1, 6)
        joint_X_p = self._torch_joint_X_p.view(num_envs, -1, 7)

        self._record_spherical_joints()
        has_sphere_joints = self._has_spherical_joints()

        if (has_sphere_joints):
            self._wp_dof_pos = wp.zeros_like(self.raw_state.joint_qd)
            self._wp_dof_vel = wp.zeros_like(self.raw_state.joint_qd)
            dof_pos = wp.to_torch(self._wp_dof_pos).view(num_envs, -1)
            dof_vel = wp.to_torch(self._wp_dof_vel).view(num_envs, -1)
        else:
            dof_pos = joint_q
            dof_vel = joint_qd

        if (self._has_actuator_forces):
            if (has_sphere_joints):
                self._wp_dof_actuator_force = wp.zeros_like(self.raw_state.mujoco.qfrc_actuator)
            else:
                self._wp_dof_actuator_force = self.raw_state.mujoco.qfrc_actuator
            dof_actuator_force = wp.to_torch(self._wp_dof_actuator_force).view(num_envs, -1)

        for obj_id in range(objs_per_env):
            body_start = articulation_start[obj_id]
            body_end = articulation_start[obj_id + 1]

            q_root_start = joint_q_start[body_start]
            qd_root_start = joint_qd_start[body_start]

            if (has_sphere_joints):
                dof_start = joint_qd_start[body_start + 1]
                dof_end = joint_qd_start[body_end]
            else:
                dof_start = joint_q_start[body_start + 1]
                dof_end = joint_q_start[body_end]
            qd_dof_start = joint_qd_start[body_start + 1]
            qd_dof_end = joint_qd_start[body_end]

            obj_body_q = body_q[:, body_start:body_end, :]
            obj_body_qd = body_qd[:, body_start:body_end, :]
            obj_body_force = body_force[:, body_start:body_end, :]

            root_type = joint_types[body_start]
            fix_root = root_type == newton.JointType.FIXED

            if (fix_root):
                obj_root_pos = joint_X_p[..., body_start, :3]
                obj_root_rot = joint_X_p[..., body_start, 3:7]
                obj_root_vel = obj_body_qd[:, 0, :3]
                obj_root_ang_vel = obj_body_qd[:, 0, 3:6]
            else:
                obj_root_pos = joint_q[:, q_root_start:q_root_start + 3]
                obj_root_rot = joint_q[:, q_root_start + 3:q_root_start + 7]
                obj_root_vel = joint_qd[:, qd_root_start:qd_root_start + 3]
                obj_root_ang_vel = joint_qd[:, qd_root_start + 3:qd_root_start + 6]

            self.root_pos.append(obj_root_pos)
            self.root_rot.append(obj_root_rot)
            self.root_vel.append(obj_root_vel)
            self.root_ang_vel.append(obj_root_ang_vel)

            self.dof_pos.append(dof_pos[:, dof_start:dof_end])
            self.dof_vel.append(dof_vel[:, qd_dof_start:qd_dof_end])
            if (self._has_actuator_forces):
                self.dof_actuator_force.append(dof_actuator_force[:, qd_dof_start:qd_dof_end])

            self.body_pos.append(obj_body_q[..., :3])
            self.body_rot.append(obj_body_q[..., 3:])
            self.body_vel.append(obj_body_qd[..., :3])
            self.body_ang_vel.append(obj_body_qd[..., 3:])
            self.body_force.append(obj_body_force)

        self.post_step_update()
        return
    
    def clear_forces(self):
        self._wp_body_force.zero_()
        return
    
    def copy(self, raw_state):
        wp.copy(self.raw_state.joint_q, raw_state.joint_q)
        wp.copy(self.raw_state.joint_qd, raw_state.joint_qd)
        wp.copy(self.raw_state.body_q, raw_state.body_q)
        wp.copy(self.raw_state.body_qd, raw_state.body_qd)
        wp.copy(self.raw_state.body_f, raw_state.body_f)

        if (self._has_actuator_forces):
            wp.copy(self.raw_state.mujoco.qfrc_actuator, raw_state.mujoco.qfrc_actuator)
        return
    
    def pre_step_update(self):
        if (self._has_spherical_joints()):
            joint_q = self.raw_state.joint_q
            self.dof_pos_to_joint_q(self._wp_dof_pos, joint_q)
            self.convert_dof_vecs(joint_q, self._wp_dof_vel, self.raw_state.joint_qd, to_parent=True)
        return

    def post_step_update(self):
        if (self._has_spherical_joints()):
            joint_q = self.raw_state.joint_q
            self.joint_q_to_dof_pos(joint_q, self._wp_dof_pos)
            self.convert_dof_vecs(joint_q, self.raw_state.joint_qd, self._wp_dof_vel, to_parent=False)

            if (self._has_actuator_forces):
                self.convert_dof_vecs(joint_q, self.raw_state.mujoco.qfrc_actuator, self._wp_dof_actuator_force, 
                                      to_parent=False)
        return
    
    def dof_pos_to_joint_q(self, dof_pos, joint_q):
        if (self._wp_nonsphere_dof_idx.shape[0] > 0):
            wp.launch(kernel=copy_indexed, dim=self._wp_nonsphere_dof_idx.shape[0],
                      inputs=[dof_pos, self._wp_nonsphere_dof_idx, self._wp_nonsphere_q_idx, joint_q])
        if (self._has_spherical_joints()):
            wp.launch(kernel=exp_map_to_quat_indexed, dim=self._wp_sphere_dof_start.shape[0],
                      inputs=[dof_pos, self._wp_sphere_dof_start, self._wp_sphere_q_start, joint_q])
        return
    
    def joint_q_to_dof_pos(self, joint_q, dof_pos):
        if (self._wp_nonsphere_dof_idx.shape[0] > 0):
            wp.launch(kernel=copy_indexed, dim=self._wp_nonsphere_dof_idx.shape[0],
                      inputs=[joint_q, self._wp_nonsphere_q_idx, self._wp_nonsphere_dof_idx, dof_pos])
        if (self._has_spherical_joints()):
            wp.launch(kernel=quat_to_exp_map_indexed, dim=self._wp_sphere_q_start.shape[0],
                      inputs=[joint_q, self._wp_sphere_q_start, self._wp_sphere_dof_start, dof_pos])
        return
    
    def convert_dof_vecs(self, joint_q, vecs_in, vecs_out, to_parent):
        # converts per dof velocities or forces between the child frame and the parent frame of 
        # spherical joints
        if (self._wp_nonsphere_dof_idx.shape[0] > 0):
            wp.launch(kernel=copy_indexed, dim=self._wp_nonsphere_dof_idx.shape[0],
                      inputs=[vecs_in, self._wp_nonsphere_dof_idx, self._wp_nonsphere_dof_idx, vecs_out])
        if (self._has_spherical_joints()):
            wp.launch(kernel=rotate_sphere_vec_indexed, dim=self._wp_sphere_dof_start.shape[0],
                      inputs=[vecs_in, self._wp_sphere_dof_start, self._wp_sphere_q_start, 
                              joint_q, to_parent, vecs_out])
        return

    def eval_fk(self):
        newton.eval_fk(self._sim_model, self.raw_state.joint_q, self.raw_state.joint_qd, self.raw_state)
        return
    
    def _record_spherical_joints(self):
        joint_q_start = self._sim_model.joint_q_start.numpy()
        joint_qd_start = self._sim_model.joint_qd_start.numpy()
        joint_types = self._sim_model.joint_type.numpy()
        joint_parent = self._sim_model.joint_parent.numpy()
        num_joints = self._sim_model.joint_count

        sphere_q_start = []
        sphere_dof_start = []
        nonsphere_q_idx = []
        nonsphere_dof_idx = []

        for j in range(num_joints):
            is_root = joint_parent[j] == -1
            q_start = joint_q_start[j]
            q_end = joint_q_start[j + 1]
            dof_start = joint_qd_start[j]
            dof_end = joint_qd_start[j + 1]

            if (is_root):
                pass
            elif (joint_types[j] == newton.JointType.BALL):
                assert(q_end - q_start == 4)
                assert(dof_end - dof_start == 3)
                sphere_q_start.append(q_start)
                sphere_dof_start.append(dof_start)
            else:
                assert(q_end - q_start == dof_end - dof_start)
                nonsphere_q_idx += list(range(q_start, q_end))
                nonsphere_dof_idx += list(range(dof_start, dof_end))

        device = self._sim_model.device
        self._wp_sphere_q_start = wp.array(sphere_q_start, device=device, dtype=int)
        self._wp_sphere_dof_start = wp.array(sphere_dof_start, device=device, dtype=int)
        self._wp_nonsphere_q_idx = wp.array(nonsphere_q_idx, device=device, dtype=int)
        self._wp_nonsphere_dof_idx = wp.array(nonsphere_dof_idx, device=device, dtype=int)
        return
    
    def _has_spherical_joints(self):
        return self._wp_sphere_q_start.shape[0] > 0


class Controls:
    def __init__(self, sim_model, num_envs, sim_state):
        self.control = sim_model.control()
        self._sim_state = sim_state

        self.wp_dof_target_pos = wp.zeros_like(self.control.joint_target_qd)
        self.wp_dof_target_vel = wp.zeros_like(self.control.joint_target_qd)

        target_pos = wp.to_torch(self.wp_dof_target_pos).view(num_envs, -1)
        target_vel = wp.to_torch(self.wp_dof_target_vel).view(num_envs, -1)
        joint_force = wp.to_torch(self.control.joint_f).view(num_envs, -1)

        articulation_start = sim_model.articulation_start.numpy()
        joint_qd_start = sim_model.joint_qd_start.numpy()

        num_objs = sim_model.articulation_count
        objs_per_env = int(num_objs / num_envs)

        self.target_pos = []
        self.target_vel = []
        self.joint_force = []
        
        for obj_id in range(objs_per_env):
            body_start = articulation_start[obj_id]
            body_end = articulation_start[obj_id + 1]
            qd_start = joint_qd_start[body_start + 1]
            qd_end = joint_qd_start[body_end]

            self.target_pos.append(target_pos[:, qd_start:qd_end])
            self.target_vel.append(target_vel[:, qd_start:qd_end])
            self.joint_force.append(joint_force[:, qd_start:qd_end])

        return

    def update_target_pos(self):
        self._sim_state.dof_pos_to_joint_q(self.wp_dof_target_pos, self.control.joint_target_q)
        return

    def update_target_vel(self, joint_q):
        self._sim_state.convert_dof_vecs(joint_q, self.wp_dof_target_vel, self.control.joint_target_qd,
                                         to_parent=True)
        return


class ObjCfg():
    def __init__(self, asset_file, fix_root, is_visual, enable_self_collisions):
        self.asset_file = asset_file
        self.fix_root = fix_root
        self.is_visual = is_visual
        self.enable_self_collisions = enable_self_collisions
        return
    
    def __hash__(self):
        values = (self.asset_file, self.fix_root, self.is_visual, self.enable_self_collisions)
        obj_hash = hash(values)
        return obj_hash
    
    def __eq__(self, other):
        same = self.asset_file == other.asset_file
        same &= self.fix_root == other.fix_root
        same &= self.is_visual == other.is_visual
        same &= self.enable_self_collisions == other.enable_self_collisions
        return same


class NewtonEngine(engine.Engine):
    def __init__(self, config, num_envs, device, visualize, record_video=False):
        super().__init__(visualize=visualize)

        self._device = device
        wp.set_device(device)

        self._num_envs = num_envs
        self._env_spacing = config["env_spacing"]

        sim_freq = config.get("sim_freq", 60)
        control_freq = config.get("control_freq", 10)
        assert(sim_freq >= control_freq and sim_freq % control_freq == 0), \
            "Simulation frequency must be a multiple of the control frequency"

        self._timestep = 1.0 / control_freq
        self._sim_steps = int(sim_freq / control_freq)
        self._sim_timestep = 1.0 / sim_freq
        self._sim_step_count = 0

        self._scene_builder = self._create_model_builder()
        self._builder_cache = dict()

        self._obj_types = []
        self._obj_fix_root = []
        self._obj_builders = []
        self._obj_start_pos = []
        self._obj_start_rot = []
        self._obj_colors = []
        
        if ("control_mode" in config):
            self._control_mode = engine.ControlMode[config["control_mode"]]
        else:
            self._control_mode = engine.ControlMode.none

        self._build_ground()

        if (visualize):
            self._build_viewer()
            self._keyboard_callbacks = dict()
            self._record_video = False
        else:
            self._viewer = None
            self._record_video = record_video
        
        return
    
    def get_name(self):
        return "newton"
    
    def create_env(self):
        env_id = len(self._obj_builders)
        assert(env_id < self.get_num_envs())

        self._obj_builders.append([])
        self._obj_types.append([])
        self._obj_fix_root.append([])
        self._obj_start_pos.append([])
        self._obj_start_rot.append([])
        self._obj_colors.append([])

        return env_id
    
    def initialize_sim(self):
        self._validate_envs()
        self._build_objs()

        Logger.print("Initializing simulation...")
        self._sim_model = self._scene_builder.finalize(device=self._device, requires_grad=False)

        self._articulation_start = self._sim_model.articulation_start.numpy()
        self._joint_qd_start = self._sim_model.joint_qd_start.numpy()
        
        # the actuator forces are only computed by the solver in modes where the solver computes the torques
        if (self._control_mode in (engine.ControlMode.pos, engine.ControlMode.vel)):
            self._sim_model.request_state_attributes("mujoco:qfrc_actuator")

        num_envs = self.get_num_envs()
        self._sim_state = SimState(self._sim_model, num_envs)
        self._state_swap_buffer = self._sim_model.state()
        self._gravity = self._sim_model.gravity.numpy()[0]

        self._apply_start_xform()
        self._apply_obj_colors()

        self._build_controls()
        self._build_contact_sensors()
        self._build_solver()

        self._build_sim_tensors()
        self._build_contacts()

        if (self._visualize()):
            self.init_viewer(self._viewer)
            self._camera_offsets = self._viewer.world_offsets.numpy()
        
        if self.enabled_record_video():
            self._video_recorder = self._build_video_recorder()
            self._recording = False

        super().initialize_sim()
        
        self._build_graphs()
        return
    
    def step(self):
        self._apply_cmd()

        if (self.enabled_record_video() and self._recording):
            self._video_recorder.capture_frame()

        if (self._graph):
            wp.capture_launch(self._graph)
        else:
            self._simulate()

        self._solver.update_contacts(self._contacts, self._sim_state.raw_state)
        self._update_contact_sensors()
        self._sim_step_count += 1

        if (self._sim_step_count % SOLVER_OVERFLOW_CHECK_STEPS == 0):
            self._check_solver_overflow()
        return
    
    def create_obj(self, env_id, obj_type, asset_file, name, is_visual=False, enable_self_collisions=True, 
                   fix_root=False, start_pos=None, start_rot=None, color=None, disable_motors=False):
        if (start_rot is None):
            start_rot = np.array([0.0, 0.0, 0.0, 1.0])

        if (start_pos is None):
            start_pos = np.array([0.0, 0.0, 0.0])
        
        obj_builder = self._create_obj_builder(asset_file=asset_file, fix_root=fix_root, is_visual=is_visual, 
                                               enable_self_collisions=enable_self_collisions)
        
        obj_id = len(self._obj_builders[env_id])
        self._obj_types[env_id].append(obj_type)
        self._obj_fix_root[env_id].append(fix_root)
        self._obj_builders[env_id].append(obj_builder)
        self._obj_start_pos[env_id].append(start_pos)
        self._obj_start_rot[env_id].append(start_rot)
        self._obj_colors[env_id].append(color)

        return obj_id
    
    def _apply_cmd(self):
        for obj_id in range(self.get_objs_per_env()):
            if (self._has_dof(obj_id)):
                cmd = self._get_obj_dof_cmd(obj_id)
                self._set_sim_cmd(obj_id, cmd)
        return

    def _set_sim_cmd(self, obj_id, cmd):
        if (self._control_mode == engine.ControlMode.none):
            pass
        elif (self._control_mode == engine.ControlMode.pos):
            self._controls.target_pos[obj_id][:] = cmd
        elif (self._control_mode == engine.ControlMode.vel):
            self._controls.target_vel[obj_id][:] = cmd
        elif (self._control_mode == engine.ControlMode.torque):
            self._controls.joint_force[obj_id][:] = cmd
        elif (self._control_mode == engine.ControlMode.pd_explicit):
            self._controls.target_pos[obj_id][:] = cmd
        else:
            assert(False), "Unsupported control mode: {}".format(self._control_mode)
        return
    
    def set_camera_pose(self, pos, look_at):
        dx = look_at[0] - pos[0]
        dy = look_at[1] - pos[1]
        dz = look_at[2] - pos[2]

        pitch = np.arctan2(dz, np.sqrt(dx * dx + dy * dy))
        yaw = np.arctan2(dy, dx)
        cam_pos = pyglet.math.Vec3(pos[0] + self._camera_offsets[0, 0], 
                                   pos[1] + self._camera_offsets[0, 1], 
                                   pos[2] + self._camera_offsets[0, 2])
        self._viewer.set_camera(cam_pos, np.rad2deg(pitch), np.rad2deg(yaw))
        return
    
    def get_camera_pos(self):
        cam_state_pos = self._viewer.camera.pos
        cam_pos = np.array([float(cam_state_pos[0] - self._camera_offsets[0, 0]), 
                            float(cam_state_pos[1] - self._camera_offsets[0, 1]), 
                            float(cam_state_pos[2] - self._camera_offsets[0, 2])])
        return cam_pos
    
    def get_camera_dir(self):
        pitch = float(np.deg2rad(self._viewer.camera.pitch))
        yaw = float(np.deg2rad(self._viewer.camera.yaw))

        dx = np.cos(pitch) * np.cos(yaw)
        dy = np.cos(pitch) * np.sin(yaw)
        dz = np.sin(pitch)

        cam_dir = np.array([dx, dy, dz])
        cam_dir /= np.linalg.norm(cam_dir)
        return cam_dir
    
    def render(self):
        sim_time = self.get_sim_time()

        self._viewer.end_frame()
        self._viewer.begin_frame(sim_time)
        self._viewer.log_state(self._sim_state.raw_state)

        self._draw_line_count = 0
        
        super().render()
        return
    
    def get_timestep(self):
        return self._timestep
    
    def get_sim_timestep(self):
        return self._sim_timestep
    
    def get_sim_time(self):
        time_step = self.get_timestep()
        return time_step * self._sim_step_count
    
    def get_num_envs(self):
        return self._num_envs
    
    def get_gravity(self):
        return self._gravity
    
    def get_objs_per_env(self):
        return len(self._sim_state.root_pos)

    def get_control_mode(self):
        return self._control_mode
    
    def get_obj_type(self, obj_id):
        obj_type = self._obj_types[0][obj_id]
        return obj_type
    
    def get_obj_fix_root(self, obj_id):
        fix_root = self._obj_fix_root[0][obj_id]
        return fix_root
    
    def get_obj_num_bodies(self, obj_id):
        num_bodies = self._sim_state.body_pos[obj_id].shape[-2]
        return num_bodies
    
    def get_obj_num_dofs(self, obj_id):
        num_dofs = self._sim_state.dof_pos[obj_id].shape[-1]
        return num_dofs
    
    def get_obj_body_names(self, obj_id):
        body_start, body_end = self._get_obj_body_range(0, obj_id)
        body_labels = self._sim_model.body_label[body_start:body_end]
        body_names = [os.path.basename(l) for l in body_labels]
        return body_names

    def get_obj_torque_limits(self, env_id, obj_id):
        dof_start, dof_end = self._get_obj_dof_range(env_id, obj_id)
        torque_lim = self._sim_model.joint_effort_limit.numpy()[dof_start:dof_end]
        return torque_lim
    
    def get_obj_dof_limits(self, env_id, obj_id):
        dof_start, dof_end = self._get_obj_dof_range(env_id, obj_id)
        dof_low = self._sim_model.joint_limit_lower.numpy()[dof_start:dof_end]
        dof_high = self._sim_model.joint_limit_upper.numpy()[dof_start:dof_end]
        return dof_low, dof_high
    
    def get_obj_pd_gains(self, env_id, obj_id):
        dof_start, dof_end = self._get_obj_dof_range(env_id, obj_id)

        if (self._control_mode == engine.ControlMode.pd_explicit):
            kp = self._kp_raw.numpy()[dof_start:dof_end]
            kd = self._kd_raw.numpy()[dof_start:dof_end]
        else:
            kp = self._sim_model.joint_target_ke.numpy()[dof_start:dof_end]
            kd = self._sim_model.joint_target_kd.numpy()[dof_start:dof_end]

        return kp, kd
    
    def calc_obj_mass(self, env_id, obj_id):
        body_start, body_end = self._get_obj_body_range(env_id, obj_id)
        masses = self._sim_model.body_mass.numpy()[body_start:body_end]
        total_mass = float(masses.sum())
        return total_mass
    
    def get_root_pos(self, obj_id, env_ids=None):
        env_ids = engine.none_to_env_ids(env_ids)
        root_pos = self._sim_state.root_pos[obj_id][env_ids]
        return root_pos
    
    def get_root_rot(self, obj_id, env_ids=None):
        env_ids = engine.none_to_env_ids(env_ids)
        root_rot = self._sim_state.root_rot[obj_id][env_ids]
        return root_rot
    
    def get_root_vel(self, obj_id, env_ids=None):
        env_ids = engine.none_to_env_ids(env_ids)
        root_vel = self._sim_state.root_vel[obj_id][env_ids]
        return root_vel
    
    def get_root_ang_vel(self, obj_id, env_ids=None):
        env_ids = engine.none_to_env_ids(env_ids)
        root_ang_vel = self._sim_state.root_ang_vel[obj_id][env_ids]
        return root_ang_vel
    
    def get_dof_pos(self, obj_id, env_ids=None):
        env_ids = engine.none_to_env_ids(env_ids)
        dof_pos = self._sim_state.dof_pos[obj_id][env_ids]
        return dof_pos
    
    def get_dof_vel(self, obj_id, env_ids=None):
        env_ids = engine.none_to_env_ids(env_ids)
        dof_vel = self._sim_state.dof_vel[obj_id][env_ids]
        return dof_vel
    
    def get_dof_forces(self, obj_id):
        dof_forces = self._dof_forces[obj_id]
        return dof_forces
    
    def get_body_pos(self, obj_id, env_ids=None):
        env_ids = engine.none_to_env_ids(env_ids)
        body_pos = self._sim_state.body_pos[obj_id][env_ids]
        return body_pos
    
    def get_body_rot(self, obj_id, env_ids=None):
        env_ids = engine.none_to_env_ids(env_ids)
        body_rot = self._sim_state.body_rot[obj_id][env_ids]
        return body_rot
    
    def get_body_vel(self, obj_id, env_ids=None):
        env_ids = engine.none_to_env_ids(env_ids)
        body_vel = self._sim_state.body_vel[obj_id][env_ids]
        return body_vel
    
    def get_body_ang_vel(self, obj_id, env_ids=None):
        env_ids = engine.none_to_env_ids(env_ids)
        body_ang_vel = self._sim_state.body_ang_vel[obj_id][env_ids]
        return body_ang_vel
    
    def get_contact_forces(self, obj_id):
        forces = self._contact_forces[obj_id]
        return forces
    
    def get_ground_contact_forces(self, obj_id):
        forces = self._ground_contact_forces[obj_id]
        return forces
    
    def set_root_pos(self, env_id, obj_id, root_pos):
        env_ids = engine.none_to_env_ids(env_id)
        self._sim_state.root_pos[obj_id][env_ids] = root_pos

        self._flag_obj_needs_reset(env_id, obj_id)
        return
    
    def set_root_rot(self, env_id, obj_id, root_rot):
        env_ids = engine.none_to_env_ids(env_id)
        self._sim_state.root_rot[obj_id][env_ids] = root_rot

        self._flag_obj_needs_reset(env_id, obj_id)
        return
    
    def set_root_vel(self, env_id, obj_id, root_vel):
        env_ids = engine.none_to_env_ids(env_id)
        self._sim_state.root_vel[obj_id][env_ids] = root_vel

        self._flag_obj_needs_reset(env_id, obj_id)
        return
    
    def set_root_ang_vel(self, env_id, obj_id, root_ang_vel):
        env_ids = engine.none_to_env_ids(env_id)
        self._sim_state.root_ang_vel[obj_id][env_ids] = root_ang_vel

        self._flag_obj_needs_reset(env_id, obj_id)
        return
    
    def set_dof_pos(self, env_id, obj_id, dof_pos):
        env_ids = engine.none_to_env_ids(env_id)
        self._sim_state.dof_pos[obj_id][env_ids] = dof_pos

        self._flag_obj_needs_reset(env_id, obj_id)
        return
    
    def set_dof_vel(self, env_id, obj_id, dof_vel):
        env_ids = engine.none_to_env_ids(env_id)
        self._sim_state.dof_vel[obj_id][env_ids] = dof_vel

        self._flag_obj_needs_reset(env_id, obj_id)
        return
    
    def set_body_pos(self, env_id, obj_id, body_pos):
        env_ids = engine.none_to_env_ids(env_id)
        self._sim_state.body_pos[obj_id][env_ids] = body_pos
        return
    
    def set_body_rot(self, env_id, obj_id, body_rot):
        env_ids = engine.none_to_env_ids(env_id)
        self._sim_state.body_rot[obj_id][env_ids] = body_rot
        return
    
    def set_body_vel(self, env_id, obj_id, body_vel):
        env_ids = engine.none_to_env_ids(env_id)
        self._sim_state.body_vel[obj_id][env_ids] = body_vel
        return
    
    def set_body_ang_vel(self, env_id, obj_id, body_ang_vel):
        env_ids = engine.none_to_env_ids(env_id)
        self._sim_state.body_ang_vel[obj_id][env_ids] = body_ang_vel
        return
    
    def set_body_forces(self, env_id, obj_id, body_id, forces):
        if (env_id is None or len(env_id) > 0):
            assert(len(forces.shape) == 2)
            self._has_body_forces.fill_(1)

            env_ids = engine.none_to_env_ids(env_id)
            self._sim_state.body_force[obj_id][env_ids, body_id, :3] = forces
        return
    
    def find_obj_body_id(self, obj_id, body_name):
        body_names = self.get_obj_body_names(obj_id)
        if (body_name in body_names):
            body_id = body_names.index(body_name)
        else:
            body_id = -1
        return body_id

    def get_sim_model(self):
        return self._sim_model
    
    def get_sim_state(self):
        return self._sim_state
    
    def set_body_color(self, env_id, obj_id, body_id, color):
        body_start, _ = self._get_obj_body_range(env_id, obj_id)
        body_shape_ids = list(self._sim_model.body_shapes[body_start + body_id])

        shape_color = wp.to_torch(self._sim_model.shape_color)
        shape_color[body_shape_ids] = torch.tensor(color[:3], dtype=shape_color.dtype, device=shape_color.device)
        return
    
    def init_viewer(self, viewer, visible_env_ids=None):
        env_spacing = self._get_env_spacing()
        viewer.set_model(self._sim_model)

        if (visible_env_ids is not None):
            viewer.set_visible_worlds(visible_env_ids)

        viewer.set_world_offsets([env_spacing, env_spacing, 0.0])

        self._setup_lighting(viewer)
        self._build_ground_mesh(viewer)
        return

    def draw_lines(self, env_id, start_verts, end_verts, cols, line_width):
        cam_offset = self._camera_offsets[env_id]
        start_pts = start_verts.copy()
        end_pts = end_verts.copy()

        start_pts[:, :2] += cam_offset[:2]
        end_pts[:, :2] += cam_offset[:2]
        line_name = "lines{:d}".format(self._draw_line_count)
        self._viewer.log_lines(name=line_name, 
                               starts=wp.array(start_pts, dtype=wp.vec3), 
                               ends=wp.array(end_pts, dtype=wp.vec3), 
                               colors=wp.array(cols[:, :3], dtype=wp.vec3), 
                               width=line_width)
        self._draw_line_count += 1
        return
    
    def register_keyboard_callback(self, key_str, callback_func):
        key_code = str_to_key_code(key_str)
        assert(key_code not in self._keyboard_callbacks)
        self._keyboard_callbacks[key_code] = callback_func
        return
    
    def _validate_envs(self):
        num_envs = self.get_num_envs()
        objs_per_env = len(self._obj_builders[0])

        for i in range(0, num_envs):
            curr_objs = self._obj_builders[i]
            num_objs = len(curr_objs)
            assert(num_objs == objs_per_env), "All envs must have the same number of objects."
        return

    def _build_objs(self):
        num_envs = self.get_num_envs()
        objs_per_env = len(self._obj_builders[0])
        total_objs = num_envs * objs_per_env

        for env_id in range(num_envs):
            self._scene_builder.begin_world()

            for obj_id in range(objs_per_env):
                obj_builder = self._obj_builders[env_id][obj_id]
                self._scene_builder.add_builder(obj_builder)
                
                num_created = env_id * objs_per_env + obj_id + 1
                Logger.print("Building {:d}/{:d} objs".format(num_created, total_objs), end='\r')

            self._scene_builder.end_world()

        Logger.print("")
        return
    
    def _apply_start_xform(self):
        num_envs = self.get_num_envs()
        for obj_id in range(self.get_objs_per_env()):
            start_pos = np.array([self._obj_start_pos[env_id][obj_id] for env_id in range(num_envs)])
            start_rot = np.array([self._obj_start_rot[env_id][obj_id] for env_id in range(num_envs)])
            self.set_root_pos(None, obj_id, torch.tensor(start_pos, dtype=torch.float32, device=self._device))
            self.set_root_rot(None, obj_id, torch.tensor(start_rot, dtype=torch.float32, device=self._device))
        
        self._sim_state.eval_fk()
        return
    
    def _build_controls(self):
        kp = self._sim_model.mujoco.dof_passive_stiffness
        kd = self._sim_model.joint_damping

        if (self._control_mode == engine.ControlMode.none):
            self._sim_model.joint_target_ke.fill_(0.0)
            self._sim_model.joint_target_kd.fill_(0.0)
            self._sim_model.joint_target_mode.fill_(int(newton.JointTargetMode.NONE))

        elif (self._control_mode == engine.ControlMode.pos):
            wp.copy(self._sim_model.joint_target_ke, kp)
            wp.copy(self._sim_model.joint_target_kd, kd)
            self._sim_model.joint_target_mode.fill_(int(newton.JointTargetMode.POSITION))

        elif (self._control_mode == engine.ControlMode.vel):
            self._sim_model.joint_target_ke.fill_(0.0)
            wp.copy(self._sim_model.joint_target_kd, kd)
            self._sim_model.joint_target_mode.fill_(int(newton.JointTargetMode.VELOCITY))

        elif (self._control_mode == engine.ControlMode.torque):
            self._sim_model.joint_target_ke.fill_(0.0)
            self._sim_model.joint_target_kd.fill_(0.0)
            self._sim_model.joint_target_mode.fill_(int(newton.JointTargetMode.EFFORT))

        elif (self._control_mode == engine.ControlMode.pd_explicit):
            self._sim_model.joint_target_ke.fill_(0.0)
            self._sim_model.joint_target_kd.fill_(0.0)
            self._kp_raw = wp.clone(kp)
            self._kd_raw = wp.clone(kd)
            self._dof_q_idx = self._build_dof_q_idx()
            self._sim_model.joint_target_mode.fill_(int(newton.JointTargetMode.EFFORT))

        else:
            assert(False), "Unsupported control mode: {}".format(self._control_mode)
        
        self._torque_lim_raw = wp.clone(self._sim_model.joint_effort_limit)
        self._neg_torque_lim_raw = wp.array(-self._torque_lim_raw.numpy(), device=self._device)
        
        self._sim_model.mujoco.dof_passive_stiffness.fill_(0.0)
        self._sim_model.joint_damping.fill_(0.0)
        self._disable_free_joint_targets()

        num_envs = self.get_num_envs()
        self._controls = Controls(self._sim_model, num_envs, self._sim_state)
        return

    def _disable_free_joint_targets(self):
        # free joints (e.g. the roots of floating objects) cannot be actuated by the solver
        joint_types = self._sim_model.joint_type.numpy()
        joint_qd_start = self._sim_model.joint_qd_start.numpy()
        target_mode = self._sim_model.joint_target_mode.numpy()

        for j, joint_type in enumerate(joint_types):
            if (joint_type == newton.JointType.FREE):
                curr_start = joint_qd_start[j]
                curr_end = joint_qd_start[j + 1]
                target_mode[curr_start:curr_end] = int(newton.JointTargetMode.NONE)

        self._sim_model.joint_target_mode.assign(target_mode)
        return

    def _build_dof_q_idx(self):
        # maps each dof to the index of its coordinate in joint_q, which is needed since joint_q
        # and joint_qd have different sizes, dofs of multi-dimensional joints (e.g. free and ball
        # joints) do not have a corresponding coordinate and are assigned an index of -1
        joint_q_start = self._sim_model.joint_q_start.numpy()
        joint_qd_start = self._sim_model.joint_qd_start.numpy()
        num_joints = self._sim_model.joint_count
        num_dofs = self._sim_model.joint_dof_count

        dof_q_idx = np.full(num_dofs, -1, dtype=np.int32)
        for j in range(num_joints):
            q_start = joint_q_start[j]
            dof_start = joint_qd_start[j]
            num_coords = joint_q_start[j + 1] - q_start
            num_joint_dofs = joint_qd_start[j + 1] - dof_start

            if (num_coords == num_joint_dofs):
                dof_q_idx[dof_start:dof_start + num_joint_dofs] = np.arange(q_start, q_start + num_coords)

        dof_q_idx = wp.array(dof_q_idx, dtype=wp.int32, device=self._device)
        return dof_q_idx

    def _build_solver(self):
        self._solver = newton.solvers.SolverMuJoCo(
            self._sim_model,
            solver="newton",
            njmax=256,
            nconmax=200,
            impratio=10,
            iterations=100,
            ls_iterations=50
        )
        return
    
    def _build_contacts(self):
        max_contacts = self._solver.mjw_data.naconmax
        self._collision_pipeline = newton.CollisionPipeline(self._sim_model,
                                                            broad_phase="explicit",
                                                            rigid_contact_max=max_contacts)
        self._contacts = self._collision_pipeline.contacts()
        return

    def _build_graphs(self):
        self._graph = None
        
        if (wp.get_device(self._device).is_cuda):
            with wp.ScopedCapture() as capture:
                self._simulate()

            self._graph = capture.graph
        return
    
    def _build_sim_tensors(self):
        self._has_body_forces = wp.array([0], dtype=wp.int32, device=self._device)

        if (self._control_mode in (engine.ControlMode.pos, engine.ControlMode.vel)):
            self._dof_forces = self._sim_state.dof_actuator_force
        else:
            self._dof_forces = self._controls.joint_force
        return
    
    def _create_obj_builder(self, asset_file, fix_root, is_visual, enable_self_collisions):
        obj_cfg = ObjCfg(asset_file=asset_file, fix_root=fix_root, is_visual=is_visual, 
                         enable_self_collisions=enable_self_collisions)
        
        if (obj_cfg in self._builder_cache):
            obj_builder = self._builder_cache[obj_cfg]
        else:
            obj_builder = self._create_model_builder()
            
            _, asset_ext = os.path.splitext(asset_file)
            if (asset_ext == ".xml"):
                obj_builder.add_mjcf(
                    asset_file,
                    floating=not fix_root,
                    ignore_inertial_definitions=False,
                    collapse_fixed_joints=False,
                    enable_self_collisions=enable_self_collisions,
                    convert_3d_hinge_to_ball_joints=True
                )
            elif (asset_ext == ".urdf"):
                obj_builder.add_urdf(
                    asset_file,
                    floating=not fix_root,
                    ignore_inertial_definitions=False,
                    collapse_fixed_joints=False,
                    enable_self_collisions=enable_self_collisions,
                    joint_ordering="dfs"
                )
            else:
                assert(False), "Unsupported asset format: {:s}".format(asset_ext)

            if (is_visual):
                for i in range(len(obj_builder.shape_flags)):
                    obj_builder.shape_flags[i] &= ~newton.ShapeFlags.COLLIDE_SHAPES
            
            self._builder_cache[obj_cfg] = obj_builder

        return obj_builder
    
    def _create_model_builder(self):
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_shape_cfg.mu = 1.0
        return builder
    
    def _build_viewer(self):
        self._viewer = newton.viewer.ViewerGL(headless=False)
        self._draw_line_count = 0

        def on_keyboard_event(symbol, modifiers):
            self._on_keyboard_event(symbol, modifiers)
            return
        
        self._viewer.renderer.register_key_press(on_keyboard_event)
        return

    def _setup_lighting(self, viewer):
        renderer = viewer.renderer
        renderer.draw_sky = False
        renderer.sky_upper = BACKGROUND_COLOR
        renderer.sky_lower = GROUND_FOG_COLOR
        
        renderer.ambient_sky = (0.3, 0.31, 0.33)
        renderer.ambient_ground = (0.55, 0.54, 0.52)
        renderer.exposure = 2.5
        
        renderer.spotlight_enabled = False
        renderer.shadow_radius = 6.0
        return

    def _build_ground_mesh(self, viewer):
        viewer.show_ground = False

        world_offsets = viewer.world_offsets.numpy()
        half_size = np.max(np.abs(world_offsets[:, :2])) + 500.0
        verts = np.array([[-half_size, -half_size, 0.0],
                          [half_size, -half_size, 0.0],
                          [half_size, half_size, 0.0],
                          [-half_size, half_size, 0.0]], dtype=np.float32)
        normals = np.array([[0.0, 0.0, 1.0]] * 4, dtype=np.float32)
        uvs = verts[:, :2] / GROUND_TILE_SIZE
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
        texture = self._load_ground_texture()

        viewer.log_mesh("ground_grid_mesh",
                        points=wp.array(verts, dtype=wp.vec3),
                        indices=wp.array(indices, dtype=wp.int32),
                        normals=wp.array(normals, dtype=wp.vec3),
                        uvs=wp.array(uvs, dtype=wp.vec2),
                        texture=texture,
                        hidden=True)

        # materials are specified as (roughness, metallic, checker_enable, texture_enable)
        viewer.log_instances("ground_grid",
                             mesh="ground_grid_mesh",
                             xforms=wp.array([wp.transform_identity()], dtype=wp.transform),
                             scales=wp.array([wp.vec3(1.0, 1.0, 1.0)], dtype=wp.vec3),
                             colors=wp.array([wp.vec3(1.0, 1.0, 1.0)], dtype=wp.vec3),
                             materials=wp.array([wp.vec4(0.6, 0.0, 0.0, 1.0)], dtype=wp.vec4))
        return

    def _load_ground_texture(self):
        texture = np.array(Image.open(GROUND_TEXTURE_FILE).convert("RGB"), dtype=np.float32) / 255.0

        lum = texture @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        texture = (1.0 - GROUND_TEXTURE_DESATURATION) * texture + GROUND_TEXTURE_DESATURATION * lum[..., None]
        texture = GROUND_TEXTURE_BRIGHTNESS * texture + GROUND_TEXTURE_ADD

        texture = np.clip(np.round(255.0 * texture), 0, 255).astype(np.uint8)
        return texture

    def _build_ground(self):
        # MuJoCo uses the max friction of the two shapes in contact, so the ground has (nearly) zero
        # friction, which allows the friction of each object to be different
        shape_cfg = self._scene_builder.ShapeConfig(mu=1e-4, restitution=0)
        self._scene_builder.add_ground_plane(cfg=shape_cfg)
        return
    
    def _build_contact_sensors(self):
        self._build_ground_contact_sensor()
        return

    def _build_ground_contact_sensor(self):
        self._ground_contact_sensor = newton.sensors.SensorContact(self._sim_model,
                                                                   sensing_bodies="*",
                                                                   counterpart_shapes="ground*",
                                                                   measure_total=True,
                                                                   verbose=True)
        num_envs = self.get_num_envs()
        num_objs = self.get_objs_per_env()

        total_forces = wp.to_torch(self._ground_contact_sensor.total_force)
        total_forces = total_forces.view(num_envs, -1, 3)
        ground_forces = wp.to_torch(self._ground_contact_sensor.force_matrix)
        ground_forces = ground_forces.view(num_envs, -1, 3)

        offset_idx = 0
        self._contact_forces = []
        self._ground_contact_forces = []

        for obj_id in range(num_objs):
            num_bodies = self.get_obj_num_bodies(obj_id)
            obj_total_forces = total_forces[:, offset_idx:offset_idx + num_bodies, :]
            obj_ground_forces = ground_forces[:, offset_idx:offset_idx + num_bodies, :]

            self._contact_forces.append(obj_total_forces)
            self._ground_contact_forces.append(obj_ground_forces)
            offset_idx += num_bodies
        return
    
    def _get_env_spacing(self):
        return self._env_spacing
    
    def _simulate(self):
        state0 = self._sim_state.raw_state
        state1 = self._state_swap_buffer
        controls = self._controls

        self._sim_state.pre_step_update()

        for i in range(self._sim_steps):
            self._pre_sim_step(state0, controls)
            self._solver.step(state0, state1, controls.control, self._contacts, self._sim_timestep)
            state0, state1 = state1, state0

        if (self._sim_steps % 2 != 0):
            self._sim_state.copy(self._state_swap_buffer)
        
        self._sim_state.post_step_update()
        
        def clear_body_forces():
            self._sim_state.clear_forces()
            self._has_body_forces.zero_()
            return
        wp.capture_if(self._has_body_forces, on_true=clear_body_forces)

        if (not isinstance(self._solver, newton.solvers.SolverMuJoCo) 
            and not isinstance(self._solver, newton.solvers.SolverFeatherstone)):
            newton.eval_ik(self._sim_model, state0, state0.joint_q, state0.joint_qd)
        return
    
    def _pre_sim_step(self, raw_state, controls):
        raw_state.clear_forces()

        def apply_body_force():
            wp.copy(raw_state.body_f, self._sim_state._wp_body_force)
            return
        wp.capture_if(self._has_body_forces, on_true=apply_body_force)
        
        if (self._visualize()):
            self._viewer.apply_forces(raw_state)

        if (self._control_mode == engine.ControlMode.pos):
            controls.update_target_pos()
        elif (self._control_mode == engine.ControlMode.vel):
            controls.update_target_vel(raw_state.joint_q)
        elif (self._control_mode == engine.ControlMode.torque):
            self._clip_torques(controls)
        elif (self._control_mode == engine.ControlMode.pd_explicit):
            self._apply_pd_explicit_torque(raw_state, controls)
        return
    
    def _update_contact_sensors(self):
        self._ground_contact_sensor.update(self._sim_state.raw_state, self._contacts)
        return
    
    def _visualize(self):
        return self._viewer is not None
    
    def _apply_obj_colors(self):
        body_colors = np.zeros([self._sim_model.body_count, 3], dtype=np.float32)
        body_has_color = np.zeros(self._sim_model.body_count, dtype=bool)

        for env_id in range(self.get_num_envs()):
            for obj_id in range(self.get_objs_per_env()):
                obj_col = self._obj_colors[env_id][obj_id]
                if (obj_col is not None):
                    body_start, body_end = self._get_obj_body_range(env_id, obj_id)
                    body_colors[body_start:body_end] = obj_col[:3]
                    body_has_color[body_start:body_end] = True

        shape_body = self._sim_model.shape_body.numpy()
        shape_has_color = (shape_body >= 0) & body_has_color[shape_body]

        shape_colors = self._sim_model.shape_color.numpy()
        shape_colors[shape_has_color] = body_colors[shape_body[shape_has_color]]
        self._sim_model.shape_color.assign(shape_colors)
        return

    def _get_obj_body_range(self, env_id, obj_id):
        obj_idx = env_id * self.get_objs_per_env() + obj_id
        body_start = int(self._articulation_start[obj_idx])
        body_end = int(self._articulation_start[obj_idx + 1])
        return body_start, body_end

    def _get_obj_dof_range(self, env_id, obj_id):
        body_start, body_end = self._get_obj_body_range(env_id, obj_id)
        dof_start = int(self._joint_qd_start[body_start + 1])
        dof_end = int(self._joint_qd_start[body_end])
        return dof_start, dof_end
    
    def _apply_pd_explicit_torque(self, sim_state, controls):
        joint_f = controls.control.joint_f
        wp.launch(
            kernel=compute_pd_explicit_torque,
            dim=joint_f.shape[0],
            inputs=[sim_state.joint_q, sim_state.joint_qd, controls.wp_dof_target_pos,
                    self._kp_raw, self._kd_raw, self._dof_q_idx, joint_f]
        )
        self._clip_torques(controls)
        return

    def _clip_torques(self, controls):
        joint_f = controls.control.joint_f
        wp.launch(
            kernel=clamp_arrays,
            dim=joint_f.shape[0],
            inputs=[joint_f, self._neg_torque_lim_raw, self._torque_lim_raw]
        )
        return

    def _flag_obj_needs_reset(self, env_id, obj_id):
        return
    
    def _on_keyboard_event(self, symbol, modifiers):
        if (symbol in self._keyboard_callbacks):
            callback = self._keyboard_callbacks[symbol]
            callback()
        return

    def enabled_record_video(self):
        return self._record_video

    def get_video_recording(self):
        return self._video_recorder.get_video()

    def start_video_recording(self):
        self._video_recorder.clear()
        self._recording = True
        return

    def stop_video_recording(self):
        self._recording = False
        return

    def _check_solver_overflow(self):
        overflow = np.bitwise_or.reduce(self._solver.mjw_data.overflow.numpy())
        overflow = int(overflow) & SOLVER_OVERFLOW_ERROR_MASK
        if (overflow != 0):
            raise RuntimeError("MuJoCo-Warp solver buffer overflow: {}, increase njmax or nconmax in the solver".format(
                mujoco_warp.OverflowType(overflow).name))
        return
    
    def _build_video_recorder(self) -> newton_recorder.NewtonVideoRecorder:
        Logger.print("Video recording enabled")
        return newton_recorder.NewtonVideoRecorder(self)


@wp.kernel
def clamp_arrays(x: wp.array(dtype=float),
                 mn: wp.array(dtype=float),
                 mx: wp.array(dtype=float)):
    i = wp.tid()
    x[i] = wp.clamp(x[i], mn[i], mx[i])
    return

@wp.kernel
def compute_pd_explicit_torque(joint_q: wp.array(dtype=float),
                               joint_qd: wp.array(dtype=float),
                               target_pos: wp.array(dtype=float),
                               kp: wp.array(dtype=float),
                               kd: wp.array(dtype=float),
                               dof_q_idx: wp.array(dtype=wp.int32),
                               torque: wp.array(dtype=float)):
    i = wp.tid()
    q_idx = dof_q_idx[i]
    
    tau = 0.0
    if (q_idx >= 0):
        tau = kp[i] * (target_pos[i] - joint_q[q_idx]) - kd[i] * joint_qd[i]

    torque[i] = tau
    return

@wp.kernel
def copy_indexed(input: wp.array(dtype=float),
                 idx_in: wp.array(dtype=int),
                 idx_out: wp.array(dtype=int),
                 output: wp.array(dtype=float)):
    tid = wp.tid()
    value = input[idx_in[tid]]
    output[idx_out[tid]] = value
    return

@wp.kernel
def quat_to_exp_map_indexed(in_q: wp.array(dtype=float),
                 q_start: wp.array(dtype=int),
                 dof_start: wp.array(dtype=int),
                 out_dof: wp.array(dtype=float)):
    tid = wp.tid()
    q_idx = q_start[tid]
    dof_idx = dof_start[tid]

    q = wp.quaternion(in_q[q_idx], in_q[q_idx + 1], in_q[q_idx + 2], in_q[q_idx + 3])
    axis, angle = wp.quat_to_axis_angle(q)
    exp_map = axis * angle

    out_dof[dof_idx] = exp_map[0]
    out_dof[dof_idx + 1] = exp_map[1]
    out_dof[dof_idx + 2] = exp_map[2]
    return

@wp.kernel
def rotate_sphere_vec_indexed(in_vec: wp.array(dtype=float),
                              dof_start: wp.array(dtype=int),
                              q_start: wp.array(dtype=int),
                              joint_q: wp.array(dtype=float),
                              to_parent: bool,
                              out_vec: wp.array(dtype=float)):
    # rotates the angular velocities or torques of spherical joints between the child frame and the parent frame
    tid = wp.tid()
    dof_idx = dof_start[tid]
    q_idx = q_start[tid]

    q = wp.quaternion(joint_q[q_idx], joint_q[q_idx + 1], joint_q[q_idx + 2], joint_q[q_idx + 3])
    vec = wp.vec3f(in_vec[dof_idx], in_vec[dof_idx + 1], in_vec[dof_idx + 2])

    if (to_parent):
        vec = wp.quat_rotate(q, vec)
    else:
        vec = wp.quat_rotate_inv(q, vec)

    out_vec[dof_idx] = vec[0]
    out_vec[dof_idx + 1] = vec[1]
    out_vec[dof_idx + 2] = vec[2]
    return

@wp.kernel
def exp_map_to_quat_indexed(in_dof: wp.array(dtype=float),
                 dof_start: wp.array(dtype=int),
                 q_start: wp.array(dtype=int),
                 out_q: wp.array(dtype=float)):
    min_theta = 1e-5

    tid = wp.tid()
    dof_idx = dof_start[tid]
    q_idx = q_start[tid]

    exp_map = wp.vec3f(in_dof[dof_idx], in_dof[dof_idx + 1], in_dof[dof_idx + 2])

    angle = wp.length(exp_map)
    axis = exp_map / angle
    angle = wp.atan2(wp.sin(angle), wp.cos(angle))

    mask = wp.abs(angle) > min_theta
    angle = angle if mask else 0.0
    axis[0] = axis[0] if mask else 0.0
    axis[1] = axis[1] if mask else 0.0
    axis[2] = axis[2] if mask else 1.0

    q = wp.quat_from_axis_angle(axis, angle)

    out_q[q_idx] = q[0]
    out_q[q_idx + 1] = q[1]
    out_q[q_idx + 2] = q[2]
    out_q[q_idx + 3] = q[3]
    return