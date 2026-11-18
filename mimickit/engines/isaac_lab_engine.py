from isaaclab.app import AppLauncher

import carb

import numpy as np
import os
import torch

import engines.engine as engine
import engines.isaac_lab_recorder as isaac_lab_recorder
from util.logger import Logger
import util.torch_util as torch_util

ENV_PATH_TEMPLATE = "/World/envs/env_{}"
OBJ_PATH_TEMPLATE = "/World/envs/env_{}/obj_{}"
GROUND_PATH = "/World/ground"
LIGHT_PATH = "/World/Light"
COLLISION_ROOT_PATH = "/World/collisions"
CAMERA_PATH = "/OmniverseKit_Persp"
ACTUATOR_GROUP = "actuators"

BACKGROUND_COLOR = [0.85, 0.87, 0.92]
GROUND_TEXTURE_FILE = "data/assets/ground_texture.png"
GROUND_TILE_SIZE = 1.0
DEFAULT_GROUND_TEXTURE_SIZE = 2.0

def str_to_key_code(key_str):
    key_name = key_str.upper()

    if (key_name == "ESC"):
        key_name = "ESCAPE"
    elif (key_name == "RETURN"):
        key_name = "ENTER"
    elif (key_name == "DELETE"):
        key_name = "DEL"
    elif (len(key_name) == 1 and key_name.isdigit()):
        key_name = "KEY_" + key_name
    
    key_code = getattr(carb.input.KeyboardInput, key_name)
    return key_code

def is_scalar(data):
    return np.isscalar(data) or (torch.is_tensor(data) and data.dim() == 0)

class ObjCfg:
    def __init__(self, obj_type, asset_file, is_visual, enable_self_collisions, 
                fix_root, start_pos, start_rot, color, disable_motors):
        self.obj_type = obj_type
        self.asset_file = asset_file
        self.is_visual = is_visual
        self.enable_self_collisions = enable_self_collisions
        self.fix_root = fix_root
        self.start_pos = start_pos
        self.start_rot = start_rot
        self.color = color
        self.disable_motors = disable_motors
        return
    
    def is_valid_clone(self, other):
        is_same = self.obj_type == other.obj_type
        is_same &= self.asset_file == other.asset_file
        is_same &= self.is_visual == other.is_visual
        is_same &= self.enable_self_collisions == other.enable_self_collisions
        is_same &= self.fix_root == other.fix_root
        is_same &= self.disable_motors == other.disable_motors
        return is_same


class IsaacLabEngine(engine.Engine):
    def __init__(self, config, num_envs, device, visualize, record_video=False):
        super().__init__(visualize=visualize)

        self._device = device
        sim_freq = config.get("sim_freq", 60)
        control_freq = config.get("control_freq", 10)
        assert(sim_freq >= control_freq and sim_freq % control_freq == 0), \
            "Simulation frequency must be a multiple of the control frequency"

        self._timestep = 1.0 / control_freq
        self._sim_steps = int(sim_freq / control_freq)
        sim_timestep = 1.0 / sim_freq

        if (visualize):
            record_video = False

        self._create_simulator(sim_timestep, visualize, record_video)

        self._env_spacing = config["env_spacing"]
        self._obj_cfgs = []
        
        if ("control_mode" in config):
            self._control_mode = engine.ControlMode[config["control_mode"]]
        else:
            self._control_mode = engine.ControlMode.none

        self._env_offsets = self._compute_env_offsets(num_envs)
        self._build_ground()
        
        if (visualize or record_video):
            self._build_lights()
            self._build_camera()
            self._build_ground_mesh()

        if (visualize):
            self._build_draw_interface()
            self._setup_keyboard()

        if (record_video):
            self._recording = False
            self._build_video_recorder()
        
        return
    
    def get_name(self):
        return "isaac_lab"
    
    def create_env(self):
        env_id = len(self._obj_cfgs)
        assert(env_id < self.get_num_envs())

        self._obj_cfgs.append([])
        return env_id
    
    def initialize_sim(self):
        self._validate_envs()

        self._build_envs()
        self._build_objs()
        self._build_ground_contact_sensors()
        self._filter_env_collisions()

        if (self.enabled_record_video()):
            self._hide_unrecorded_envs()

        Logger.print("Initializing simulation...")
        self._sim.reset()
        
        self._build_body_order_tensors()
        self._build_sensor_order_tensors()
        self._build_sim_tensors()

        self._gravity = np.array(self._sim.cfg.gravity)
        
        super().initialize_sim()
        return
    
    def step(self):
        self._update_reset_objs()
        self._apply_cmd()

        if (self.enabled_record_video() and self._recording):
            self._video_recorder.capture_frame()
        
        for i in range(self._sim_steps):
            self._pre_sim_step()
            self._sim_step()
            self._post_sim_step()
            
        self._clear_forces()
        return

    def create_obj(self, env_id, obj_type, asset_file, name, is_visual=False, enable_self_collisions=True, 
                   fix_root=False, start_pos=None, start_rot=None, 
                   color=None, disable_motors=False):
        if (start_rot is None):
            start_rot = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            start_rot = start_rot.copy()

        if (start_pos is None):
            start_pos = np.array([0.0, 0.0, 0.0])
        else:
            start_pos = start_pos.copy()
        
        start_pos[0] += self._env_offsets[env_id, 0]
        start_pos[1] += self._env_offsets[env_id, 1]

        obj_cfg = ObjCfg(obj_type=obj_type,
                         asset_file=asset_file, 
                         is_visual=is_visual, 
                         enable_self_collisions=enable_self_collisions,
                         fix_root=fix_root, 
                         start_pos=start_pos, 
                         start_rot=start_rot,
                         color=color, 
                         disable_motors=disable_motors)
        
        obj_id = len(self._obj_cfgs[env_id])
        self._obj_cfgs[env_id].append(obj_cfg)
        
        return obj_id
    
    def _apply_cmd(self):
        for obj_id in range(self.get_objs_per_env()):
            if (self._has_dof(obj_id)):
                cmd = self._get_obj_dof_cmd(obj_id)
                dof_order_common2sim = self._dof_order_common2sim[obj_id]
                sim_cmd = cmd[:, dof_order_common2sim]
                self._set_sim_cmd(obj_id, sim_cmd)
        return
    
    def _set_sim_cmd(self, obj_id, sim_cmd):
        # sim_cmd is in the simulator's dof order
        target_command = self._objs[obj_id].actuators.target_command

        if (self._control_mode == engine.ControlMode.none):
            pass
        elif (self._control_mode == engine.ControlMode.pos):
            target_command.set_position_index(value=sim_cmd)
        elif (self._control_mode == engine.ControlMode.vel):
            target_command.set_velocity_index(value=sim_cmd)
        elif (self._control_mode == engine.ControlMode.torque):
            target_command.set_effort_index(value=sim_cmd)
        elif (self._control_mode == engine.ControlMode.pd_explicit):
            target_command.set_position_index(value=sim_cmd)
        else:
            assert(False), "Unsupported control mode: {}".format(self._control_mode)
        return
    
    def set_camera_pose(self, pos, look_at):
        from isaacsim.core.rendering_manager import ViewportManager

        env_offset = self._env_offsets[0].cpu().numpy()
        cam_pos = pos.copy()
        cam_look_at = look_at.copy()

        cam_pos[:2] += env_offset
        cam_look_at[:2] += env_offset
        ViewportManager.set_camera_view(CAMERA_PATH, eye=cam_pos, target=cam_look_at)
        return
    
    def get_camera_pos(self):
        cam_state_pos = self._camera_state.position_world
        env_offset = self._env_offsets[0].cpu().numpy()
        cam_pos = np.array([cam_state_pos[0] - env_offset[0], 
                            cam_state_pos[1] - env_offset[1], 
                            cam_state_pos[2]])
        return cam_pos
    
    def get_camera_dir(self):
        cam_state_pos = self._camera_state.position_world
        cam_state_target = self._camera_state.target_world
        cam_delta = np.array([cam_state_target[0] - cam_state_pos[0],
                              cam_state_target[1] - cam_state_pos[1],
                              cam_state_target[2] - cam_state_pos[2]])
        cam_dir = cam_delta / np.linalg.norm(cam_delta)
        return cam_dir
    
    def render(self):
        self._sim.render()
        self._draw_interface.clear_lines()
        super().render()
        return
    
    def get_timestep(self):
        return self._timestep
    
    def get_sim_timestep(self):
        dt = self._sim.get_physics_dt()
        return dt
    
    def get_num_envs(self):
        return self._env_offsets.shape[0]
    
    def get_gravity(self):
        return self._gravity
    
    def get_objs_per_env(self):
        return len(self._obj_cfgs[0])

    def get_control_mode(self):
        return self._control_mode
    
    def get_obj_type(self, obj_id):
        obj_type = self._obj_cfgs[0][obj_id].obj_type
        return obj_type
    
    def get_obj_fix_root(self, obj_id):
        fix_root = self._obj_cfgs[0][obj_id].fix_root
        return fix_root
    
    def get_obj_num_bodies(self, obj_id):
        obj = self._objs[obj_id]
        num_bodies = obj.num_bodies
        return num_bodies
    
    def get_obj_num_dofs(self, obj_id):
        obj_type = self.get_obj_type(obj_id)
        if (obj_type == engine.ObjType.articulated):
            obj = self._objs[obj_id]
            num_dofs = obj.num_joints
        else:
            num_dofs = 0
        return num_dofs
    
    def get_obj_body_names(self, obj_id):
        obj = self._objs[obj_id]
        body_names = obj.body_names
        body_order_sim2common = self._body_order_sim2common[obj_id]

        body_names = [body_names[i] for i in body_order_sim2common.tolist()]
        return body_names

    def get_obj_torque_limits(self, env_id, obj_id):
        obj = self._objs[obj_id]
        torque_lim = obj.data.joint_effort_limits.torch[env_id]

        dof_order_sim2common = self._dof_order_sim2common[obj_id]
        torque_lim = torque_lim[dof_order_sim2common]
        return torque_lim.cpu().numpy()
    
    def get_obj_dof_limits(self, env_id, obj_id):
        obj = self._objs[obj_id]
        dof_limits = obj.data.joint_pos_limits.torch[env_id]
        dof_low = dof_limits[:, 0]
        dof_high = dof_limits[:, 1]

        dof_order_sim2common = self._dof_order_sim2common[obj_id]
        dof_low = dof_low[dof_order_sim2common]
        dof_high = dof_high[dof_order_sim2common]

        return dof_low.cpu().numpy(), dof_high.cpu().numpy()
    
    def get_obj_pd_gains(self, env_id, obj_id):
        kp, kd = self._get_sim_pd_gains(obj_id)
        kp = kp[env_id]
        kd = kd[env_id]

        dof_order_sim2common = self._dof_order_sim2common[obj_id]
        kp = kp[dof_order_sim2common]
        kd = kd[dof_order_sim2common]
        kp = kp.cpu().numpy()
        kd = kd.cpu().numpy()

        return kp, kd
    
    def calc_obj_mass(self, env_id, obj_id):
        obj = self._objs[obj_id]
        masses = obj.data.body_mass.torch[env_id]
        total_mass = masses.sum().item()
        return total_mass
    
    def get_root_pos(self, obj_id, env_ids=None):
        env_ids = engine.none_to_env_ids(env_ids)
        obj = self._objs[obj_id]
        root_pos = obj.data.root_link_pose_w.torch[env_ids, :3].clone()
        root_pos[..., 0:2] -= self._env_offsets[env_ids]
        return root_pos
    
    def get_root_rot(self, obj_id, env_ids=None):
        env_ids = engine.none_to_env_ids(env_ids)
        obj = self._objs[obj_id]
        root_rot = obj.data.root_link_pose_w.torch[env_ids, 3:]
        return root_rot
    
    def get_root_vel(self, obj_id, env_ids=None):
        env_ids = engine.none_to_env_ids(env_ids)
        obj = self._objs[obj_id]
        root_vel = obj.data.root_link_vel_w.torch[env_ids, :3]
        return root_vel
    
    def get_root_ang_vel(self, obj_id, env_ids=None):
        env_ids = engine.none_to_env_ids(env_ids)
        obj = self._objs[obj_id]
        root_ang_vel = obj.data.root_link_vel_w.torch[env_ids, 3:]
        return root_ang_vel
    
    def get_dof_pos(self, obj_id, env_ids=None):
        env_ids = engine.none_to_env_ids(env_ids)
        obj = self._objs[obj_id]
        dof_pos = obj.data.joint_pos.torch[env_ids]

        dof_order_sim2common = self._dof_order_sim2common[obj_id]
        dof_pos = dof_pos[..., dof_order_sim2common]
        return dof_pos
    
    def get_dof_vel(self, obj_id, env_ids=None):
        env_ids = engine.none_to_env_ids(env_ids)
        obj = self._objs[obj_id]
        dof_vel = obj.data.joint_vel.torch[env_ids]

        dof_order_sim2common = self._dof_order_sim2common[obj_id]
        dof_vel = dof_vel[..., dof_order_sim2common]
        return dof_vel
    
    def get_dof_forces(self, obj_id):
        obj = self._objs[obj_id]
        dof_forces = obj.actuators.applied_effort.torch

        dof_order_sim2common = self._dof_order_sim2common[obj_id]
        dof_forces = dof_forces[:, dof_order_sim2common]
        return dof_forces
    
    def get_body_pos(self, obj_id, env_ids=None):
        env_ids = engine.none_to_env_ids(env_ids)
        obj = self._objs[obj_id]
        body_pos = obj.data.body_link_pose_w.torch[env_ids, :, :3]

        body_order_sim2common = self._body_order_sim2common[obj_id]
        body_pos = body_pos[..., body_order_sim2common, :]

        env_offsets = self._env_offsets[env_ids].unsqueeze(-2)
        body_pos[..., :, 0:2] -= env_offsets
        return body_pos
    
    def get_body_rot(self, obj_id, env_ids=None):
        env_ids = engine.none_to_env_ids(env_ids)
        obj = self._objs[obj_id]
        body_rot = obj.data.body_link_pose_w.torch[env_ids, :, 3:]

        body_order_sim2common = self._body_order_sim2common[obj_id]
        body_rot = body_rot[..., body_order_sim2common, :]
        return body_rot
    
    def get_body_vel(self, obj_id, env_ids=None):
        env_ids = engine.none_to_env_ids(env_ids)
        obj = self._objs[obj_id]
        body_vel = obj.data.body_link_vel_w.torch[env_ids, :, :3]

        body_order_sim2common = self._body_order_sim2common[obj_id]
        body_vel = body_vel[..., body_order_sim2common, :]
        return body_vel
    
    def get_body_ang_vel(self, obj_id, env_ids=None):
        env_ids = engine.none_to_env_ids(env_ids)
        obj = self._objs[obj_id]
        body_ang_vel = obj.data.body_link_vel_w.torch[env_ids, :, 3:]

        body_order_sim2common = self._body_order_sim2common[obj_id]
        body_ang_vel = body_ang_vel[..., body_order_sim2common, :]
        return body_ang_vel
    
    def get_contact_forces(self, obj_id):
        sensor = self._ground_contact_sensors[obj_id]
        forces = sensor.data.net_normal_forces_w.torch

        body_order_sim2common = self._sensor_body_order_sim2common[obj_id]
        forces = forces[:, body_order_sim2common, :]
        return forces
    
    def get_ground_contact_forces(self, obj_id):
        sensor = self._ground_contact_sensors[obj_id]
        forces = sensor.data.normal_force_matrix_w.torch
        forces = forces.sum(dim=-2)

        body_order_sim2common = self._sensor_body_order_sim2common[obj_id]
        forces = forces[:, body_order_sim2common, :]
        return forces
    
    def set_root_pos(self, env_id, obj_id, root_pos):
        env_ids = engine.none_to_env_ids(env_id)
        root_link_pose = self._objs[obj_id].data.root_link_pose_w.torch
        root_link_pose[env_ids, :3] = root_pos
        root_link_pose[env_ids, :2] += self._env_offsets[env_ids]
        
        self._flag_obj_needs_reset(env_id, obj_id)
        return
    
    def set_root_rot(self, env_id, obj_id, root_rot):
        env_ids = engine.none_to_env_ids(env_id)
        root_link_pose = self._objs[obj_id].data.root_link_pose_w.torch
        root_link_pose[env_ids, 3:] = root_rot
        
        self._flag_obj_needs_reset(env_id, obj_id)
        return
    
    def set_root_vel(self, env_id, obj_id, root_vel):
        env_ids = engine.none_to_env_ids(env_id)
        root_link_vel = self._objs[obj_id].data.root_link_vel_w.torch
        root_link_vel[env_ids, :3] = root_vel
        
        self._flag_obj_needs_reset(env_id, obj_id)
        return
    
    def set_root_ang_vel(self, env_id, obj_id, root_ang_vel):
        env_ids = engine.none_to_env_ids(env_id)
        root_link_vel = self._objs[obj_id].data.root_link_vel_w.torch
        root_link_vel[env_ids, 3:] = root_ang_vel
        
        self._flag_obj_needs_reset(env_id, obj_id)
        return
    
    def set_dof_pos(self, env_id, obj_id, dof_pos):
        env_ids = engine.none_to_env_ids(env_id)
        joint_pos = self._objs[obj_id].data.joint_pos.torch

        if (not is_scalar(dof_pos)):
            dof_order_common2sim = self._dof_order_common2sim[obj_id]
            dof_pos = dof_pos[..., dof_order_common2sim]

        joint_pos[env_ids] = dof_pos
        
        self._flag_obj_needs_reset(env_id, obj_id)
        return
    
    def set_dof_vel(self, env_id, obj_id, dof_vel):
        env_ids = engine.none_to_env_ids(env_id)
        joint_vel = self._objs[obj_id].data.joint_vel.torch

        if (not is_scalar(dof_vel)):
            dof_order_common2sim = self._dof_order_common2sim[obj_id]
            dof_vel = dof_vel[..., dof_order_common2sim]

        joint_vel[env_ids] = dof_vel
        
        self._flag_obj_needs_reset(env_id, obj_id)
        return
    
    def set_body_pos(self, env_id, obj_id, body_pos):
        env_ids = engine.none_to_env_ids(env_id)
        body_link_pose = self._objs[obj_id].data.body_link_pose_w.torch

        if (not is_scalar(body_pos)):
            body_order_common2sim = self._body_order_common2sim[obj_id]
            body_pos = body_pos[..., body_order_common2sim, :]
        
        body_link_pose[env_ids, :, :3] = body_pos
        body_link_pose[env_ids, :, :2] += self._env_offsets[env_ids].unsqueeze(-2)

        # don't need to flag reset after setting body states since those 
        # do not directly affect the simulator
        return
    
    def set_body_rot(self, env_id, obj_id, body_rot):
        env_ids = engine.none_to_env_ids(env_id)
        body_link_pose = self._objs[obj_id].data.body_link_pose_w.torch

        if (not is_scalar(body_rot)):
            body_order_common2sim = self._body_order_common2sim[obj_id]
            body_rot = body_rot[..., body_order_common2sim, :]
        
        body_link_pose[env_ids, :, 3:] = body_rot
        return
    
    def set_body_vel(self, env_id, obj_id, body_vel):
        env_ids = engine.none_to_env_ids(env_id)
        body_link_vel = self._objs[obj_id].data.body_link_vel_w.torch

        if (not is_scalar(body_vel)):
            body_order_common2sim = self._body_order_common2sim[obj_id]
            body_vel = body_vel[..., body_order_common2sim, :]
        
        body_link_vel[env_ids, :, :3] = body_vel
        return
    
    def set_body_ang_vel(self, env_id, obj_id, body_ang_vel):
        env_ids = engine.none_to_env_ids(env_id)
        body_link_vel = self._objs[obj_id].data.body_link_vel_w.torch

        if (not is_scalar(body_ang_vel)):
            body_order_common2sim = self._body_order_common2sim[obj_id]
            body_ang_vel = body_ang_vel[..., body_order_common2sim, :]
        
        body_link_vel[env_ids, :, 3:] = body_ang_vel
        return
    
    def set_body_forces(self, env_id, obj_id, body_id, forces):
        if (env_id is None or len(env_id) > 0):
            assert(len(forces.shape) == 2)
            
            env_ids = engine.none_to_env_ids(env_id)
            sim_body_id = self._body_order_sim2common[obj_id][body_id]
            body_forces = self._body_forces[obj_id]
            body_forces[env_ids, sim_body_id] = forces

            obj = self._objs[obj_id]
            obj.permanent_wrench_composer.set_forces_and_torques_index(forces=body_forces, is_global=True)
        return
    
    def find_obj_body_id(self, obj_id, body_name):
        obj = self._objs[obj_id]
        if (body_name in obj.body_names):
            sim_body_id = obj.body_names.index(body_name)
            body_id = int(self._body_order_common2sim[obj_id][sim_body_id])
        else:
            body_id = -1
        return body_id

    def draw_lines(self, env_id, start_verts, end_verts, cols, line_width):
        env_offset = self._env_offsets[env_id].cpu().numpy()
        start_pts = start_verts.copy()
        end_pts = end_verts.copy()

        start_pts[:, :2] += env_offset
        end_pts[:, :2] += env_offset
        line_widths = [line_width] * start_pts.shape[0]

        self._draw_interface.draw_lines(start_pts.tolist(), end_pts.tolist(), cols.tolist(), line_widths)
        return

    def register_keyboard_callback(self, key_str, callback_func):
        key_code = str_to_key_code(key_str)
        assert(key_code not in self._keyboard_callbacks)
        self._keyboard_callbacks[key_code] = callback_func
        return
    
    def enabled_record_video(self):
        return hasattr(self, "_video_recorder")

    def get_video_recording(self):
        return self._video_recorder.get_video()

    def start_video_recording(self):
        self._video_recorder.clear()
        self._recording = True
        return
    
    def stop_video_recording(self):
        self._recording = False
        return
    
    def get_sim(self):
        return self._sim
    
    def _build_ground(self):
        import isaaclab.sim as sim_utils
        from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg
        from pxr import UsdPhysics

        ground_path = GROUND_PATH
        env_offset_max = torch.max(torch.abs(self._env_offsets)).item()
        texture_width = 2.0 * env_offset_max + 1000.0

        physics_material = PhysxRigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0,
                                                     restitution=0.0)
        plane_cfg = sim_utils.GroundPlaneCfg(physics_material=physics_material, color=(1.0, 1.0, 1.0),
                                             size=(texture_width, texture_width))
        self._ground = sim_utils.spawn_ground_plane(prim_path=ground_path, cfg=plane_cfg)

        # add rigid body schema to terrain to enable contact sensors
        UsdPhysics.RigidBodyAPI.Apply(self._stage.GetPrimAtPath(ground_path))
        UsdPhysics.RigidBodyAPI.Get(self._stage, ground_path).GetKinematicEnabledAttr().Set(True)
        return
    
    def _compute_env_offsets(self, num_envs):
        env_ids = torch.arange(num_envs, device=self._device)

        num_rows = int(np.ceil(np.sqrt(num_envs)))
        num_cols = int(np.ceil(num_envs / num_rows))
        col = torch.floor(env_ids / num_rows)
        row = torch.remainder(env_ids, num_rows)

        env_spacing = self._get_env_spacing()
        row_offset = 0.5 * env_spacing * (num_rows - 1)
        col_offset = 0.5 * env_spacing * (num_cols - 1)

        offsets = torch.zeros([num_envs, 2], device=self._device, dtype=torch.float)
        offsets[:, 0] = row * env_spacing - row_offset
        offsets[:, 1] = col * env_spacing - col_offset
        return offsets
    
    def _build_lights(self):
        import isaaclab.sim as sim_utils

        light_quat = torch_util.euler_xyz_to_quat(torch.tensor(0.423),
                                                  torch.tensor(0.0), 
                                                  torch.tensor(0.588))
        distant_light_path = LIGHT_PATH + "/distant_light_xform"
        sim_utils.create_prim(distant_light_path, "Xform", orientation=light_quat.tolist())

        distant_light_cfg = sim_utils.DistantLightCfg(intensity=3800.0, color=(1.0, 0.97, 0.93), angle=8.0)
        self._distant_light = distant_light_cfg.func(distant_light_path + "/distant_light", distant_light_cfg)

        dome_light_cfg = sim_utils.DomeLightCfg(intensity=420.0, color=(0.82, 0.86, 0.92), visible_in_primary_ray=False)
        self._dome_light = dome_light_cfg.func(LIGHT_PATH + "/dome_light", dome_light_cfg)

        carb_settings = carb.settings.get_settings()
        carb_settings.set_int("/rtx/background/source/type", 2)
        carb_settings.set_float_array("/rtx/background/source/color", BACKGROUND_COLOR)
        return
    
    def _build_camera(self):
        from isaaclab.sim.utils import enable_extension
        from omni.kit.viewport.utility.camera_state import ViewportCameraState

        enable_extension("isaacsim.core.rendering_manager")
        self._camera_state = ViewportCameraState(CAMERA_PATH)
        return

    def _build_ground_mesh(self):
        from pxr import Sdf

        texture_file = os.path.abspath(GROUND_TEXTURE_FILE)
        texture_scale = DEFAULT_GROUND_TEXTURE_SIZE / GROUND_TILE_SIZE

        shader_path = GROUND_PATH + "/Looks/theGrid/Shader"
        shader_prim = self._stage.GetPrimAtPath(shader_path)
        shader_prim.GetAttribute("inputs:diffuse_texture").Set(Sdf.AssetPath(texture_file))
        shader_prim.GetAttribute("inputs:texture_scale").Set((texture_scale, texture_scale))
        
        shader_prim.CreateAttribute("inputs:albedo_desaturation", Sdf.ValueTypeNames.Float).Set(0.2)
        shader_prim.CreateAttribute("inputs:albedo_brightness", Sdf.ValueTypeNames.Float).Set(0.45)
        shader_prim.CreateAttribute("inputs:albedo_add", Sdf.ValueTypeNames.Float).Set(0.065)
        shader_prim.GetAttribute("inputs:reflection_roughness_texture_influence").Set(0.0)
        shader_prim.GetAttribute("inputs:reflection_roughness_constant").Set(0.15)
        shader_prim.GetAttribute("inputs:specular_level").Set(0.6)
        return
    
    def _build_draw_interface(self):
        from isaaclab.sim.utils import enable_extension

        enable_extension("isaacsim.util.debug_draw")
        from isaacsim.util.debug_draw import _debug_draw
        self._draw_interface = _debug_draw.acquire_debug_draw_interface()
        return
    
    def _setup_keyboard(self):
        import omni

        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)

        self._keyboard_callbacks = dict()
        return
    
    def _create_simulator(self, sim_timestep, visualize, record_video):
        launcher_args = {
            "headless": not visualize,
            "device": self._device,
            "enable_cameras": record_video or visualize,
        }
        if (visualize):
            launcher_args["visualizer"] = ["kit"]
        
        if (record_video):
            launcher_args["video"] = True
            
        self._app_launcher = AppLauncher(launcher_args)
        
        import isaaclab.sim as sim_utils
        from isaaclab_physx.physics import PhysxCfg
        from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg
        
        physx_cfg = PhysxCfg(bounce_threshold_velocity=0.2,
                             max_position_iteration_count=4,
                             max_velocity_iteration_count=0,
                             gpu_max_rigid_contact_count=8 * 1024 * 1024,
                             # disabled to match the PhysX default used with earlier versions of Isaac Lab
                             enable_external_forces_every_iteration=False)
        physics_material = PhysxRigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)

        visualizer_cfgs = []
        if (visualize):
            from isaaclab_visualizers.kit import KitVisualizerCfg
            visualizer_cfgs.append(KitVisualizerCfg(background_color=None))

        sim_cfg = sim_utils.SimulationCfg(device=self._device, dt=sim_timestep,
                                          render_interval=self._sim_steps,
                                          physics=physx_cfg,
                                          physics_material=physics_material,
                                          visualizer_cfgs=visualizer_cfgs,
                                          use_newton_actuators=False)
        
        self._sim = sim_utils.SimulationContext(sim_cfg)
        self._stage = sim_utils.get_current_stage()
        
        # disable delays during rendering
        carb_settings = carb.settings.get_settings()
        carb_settings.set_bool("/app/runLoops/main/rateLimitEnabled", False)
        return
    
    def _get_env_spacing(self):
        return self._env_spacing
    
    def _parse_usd_path(self, asset_file):
        asset_root, asset_ext = os.path.splitext(asset_file)
        if (asset_ext != ".usd" and asset_ext != ".usda"):
            asset_file = asset_root + ".usd"
        return asset_file
    
    def _build_actuator_cfg(self, control_mode):
        from isaaclab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg

        if (control_mode == engine.ControlMode.none):
            actuator_cfg = IdealPDActuatorCfg(joint_names_expr=[".*"], stiffness=0, damping=0, effort_limit=0)
        elif (control_mode == engine.ControlMode.pos):
            actuator_cfg = ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=None, damping=None, effort_limit=None)
        elif (control_mode == engine.ControlMode.vel):
            actuator_cfg = ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0, damping=None, effort_limit=None)
        elif (control_mode == engine.ControlMode.torque):
            actuator_cfg = IdealPDActuatorCfg(joint_names_expr=[".*"], stiffness=0, damping=0, effort_limit=None)
        elif (control_mode == engine.ControlMode.pd_explicit):
            actuator_cfg = IdealPDActuatorCfg(joint_names_expr=[".*"], stiffness=None, damping=None, effort_limit=None)
        else:
            assert(False), "Unsupported control mode: {}".format(control_mode)

        return actuator_cfg

    def _pre_sim_step(self):
        num_objs = self.get_objs_per_env()
        for obj_id in range(num_objs):
            obj = self._objs[obj_id]

            if (self.get_obj_type(obj_id) == engine.ObjType.articulated):
                self._refresh_joint_pos(obj)

            obj.write_data_to_sim()
        return

    def _refresh_joint_pos(self, obj):
        # Isaac Lab only reads the joint positions from the simulator when they are accessed.
        # Accessing them here keeps the torques of the implicit actuators, which are computed
        # in write_data_to_sim() and returned by get_dof_forces(), consistent with the current sim step.
        obj.data.joint_pos
        return

    def _sim_step(self):
        self._sim.step(render=False)
        return

    def _post_sim_step(self):
        sim_timestep = self.get_sim_timestep()
        
        for obj in self._objs:
            obj.update(sim_timestep)
            
        for sensor in self._ground_contact_sensors:
            if (sensor is not None):
                sensor.update(sim_timestep)
        return

    def _clear_forces(self):
        for obj_id, obj in enumerate(self._objs):
            if (obj.permanent_wrench_composer.active):
                obj.permanent_wrench_composer.reset()
                self._body_forces[obj_id][:] = 0.0
        return
    
    def _validate_envs(self):
        num_envs = self.get_num_envs()
        objs_per_env = self.get_objs_per_env()

        for i in range(0, num_envs):
            curr_objs = self._obj_cfgs[i]
            num_objs = len(curr_objs)
            assert(num_objs == objs_per_env), "All envs must have the same number of objects."
        return
    
    def _build_envs(self):
        num_envs = self.get_num_envs()
        for i in range(num_envs):
            env_path = ENV_PATH_TEMPLATE.format(i)
            self._stage.DefinePrim(env_path, "Xform")
        return
    
    def _build_objs(self):
        self._objs = []
        objs_per_env = self.get_objs_per_env()

        for obj_id in range(objs_per_env):
            obj = self._build_obj(obj_id)
            self._objs.append(obj)
        return
    
    def _build_obj(self, obj_id):
        num_envs = self.get_num_envs()
        created = [False for i in range(num_envs)]
        num_created = 0

        for env_id in range(num_envs):
            if (not created[env_id]):
                obj_cfg = self._obj_cfgs[env_id][obj_id]
                asset_file = obj_cfg.asset_file
                clone_envs = []

                for j in range(env_id + 1, num_envs):
                    if (not created[j]):
                        other_obj_cfg = self._obj_cfgs[j][obj_id]
                        other_asset_file = other_obj_cfg.asset_file

                        if (other_asset_file == asset_file):
                            assert(obj_cfg.is_valid_clone(other_obj_cfg)), "Cloning requires all objects that share the same asset file to have the same configurations."
                            clone_envs.append(j)
                
                self._build_obj_prim(env_id=env_id, obj_id=obj_id, obj_cfg=obj_cfg)
                created[env_id] = True
                
                num_clones = len(clone_envs)
                if (num_clones > 0):
                    self._clone_obj_prim(source_env_id=env_id, obj_id=obj_id, clone_env_ids=clone_envs)

                    for clone_env_id in clone_envs:
                        created[clone_env_id] = True
                
                num_created += num_clones + 1
                Logger.print("Building Obj:{:d} in {:d}/{:d} envs".format(obj_id, num_created, num_envs), end='\r')

        Logger.print("")

        # build a object to wrap all of the objects in the parallel envs
        obj_cfg = self._obj_cfgs[0][obj_id]
        multi_obj_prim = self._build_multi_obj_prim(obj_id, obj_cfg)

        return multi_obj_prim
    
    def _build_obj_prim(self, env_id, obj_id, obj_cfg):
        obj_type = obj_cfg.obj_type
        if (obj_type == engine.ObjType.rigid):
            self._build_rigid_prim(env_id, obj_id, obj_cfg)
        elif (obj_type == engine.ObjType.articulated):
            self._build_articulated_prim(env_id, obj_id, obj_cfg)
        else:
            assert(False), "Unsupported obj type: {}".format(obj_type.name)
        return

    def _build_visual_material(self, obj_cfg):
        import isaaclab.sim as sim_utils
        
        color = obj_cfg.color
        if (color is not None):
            visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(color[0], color[1], color[2]),
                                                          metallic=0.0, roughness=0.3)
        else:
            visual_material = None
        return visual_material
    
    def _build_rigid_prim(self, env_id, obj_id, obj_cfg):
        import isaaclab.sim as sim_utils
        from isaaclab_physx.sim.schemas import PhysxRigidBodyPropertiesCfg

        visual_material = self._build_visual_material(obj_cfg)
        rigid_props = PhysxRigidBodyPropertiesCfg(kinematic_enabled=obj_cfg.fix_root,
                                                  max_depenetration_velocity=10.0,
                                                  angular_damping=0.01,
                                                  max_linear_velocity=1000.0,
                                                  max_angular_velocity=1000.0)
        usd_asset_file = self._parse_usd_path(obj_cfg.asset_file)
        usd_cfg = sim_utils.UsdFileCfg(usd_path=usd_asset_file, 
                                       visual_material=visual_material, 
                                       rigid_props=rigid_props,
                                       activate_contact_sensors=True)
        
        prim_path = OBJ_PATH_TEMPLATE.format(env_id, obj_id)
        usd_cfg.func(prim_path, usd_cfg, translation=obj_cfg.start_pos.tolist(), orientation=obj_cfg.start_rot.tolist())

        if (obj_cfg.is_visual):
            self._disable_prim_collisions(prim_path)
        return
    
    def _build_articulated_prim(self, env_id, obj_id, obj_cfg):
        import isaaclab.sim as sim_utils
        from isaaclab_physx.sim.schemas import PhysxArticulationRootPropertiesCfg, PhysxRigidBodyPropertiesCfg

        visual_material = self._build_visual_material(obj_cfg)
        articulation_props = PhysxArticulationRootPropertiesCfg(
                    enabled_self_collisions=obj_cfg.enable_self_collisions, 
                    fix_root_link=obj_cfg.fix_root)
        
        rigid_props = sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=10.0,
                                                       angular_damping=0.01,
                                                       max_linear_velocity=1000.0,
                                                       max_angular_velocity=1000.0)
        usd_asset_file = self._parse_usd_path(obj_cfg.asset_file)
        usd_cfg = sim_utils.UsdFileCfg(usd_path=usd_asset_file, 
                                       visual_material=visual_material,
                                       articulation_props=articulation_props,
                                       rigid_props=rigid_props,
                                       activate_contact_sensors=True)

        prim_path = OBJ_PATH_TEMPLATE.format(env_id, obj_id)
        usd_cfg.func(prim_path, usd_cfg, translation=obj_cfg.start_pos.tolist(), orientation=obj_cfg.start_rot.tolist())

        if (obj_cfg.is_visual):
            self._disable_prim_collisions(prim_path)
        return
    
    def _build_multi_obj_prim(self, obj_id, obj_cfg):
        obj_type = obj_cfg.obj_type
        if (obj_type == engine.ObjType.rigid):
            multi_obj_prim = self._build_multi_rigid_prim(obj_id)
        elif (obj_type == engine.ObjType.articulated):
            multi_obj_prim = self._build_multi_articulated_prim(obj_id, obj_cfg)
        else:
            assert(False), "Unsupported obj type: {}".format(obj_type.name)
        return multi_obj_prim

    def _build_multi_rigid_prim(self, obj_id):
        from isaaclab.assets import RigidObject, RigidObjectCfg

        regex = OBJ_PATH_TEMPLATE.format(".*", obj_id)
        multi_obj_cfg = RigidObjectCfg(prim_path=regex, spawn=None)
        multi_obj_prim = RigidObject(multi_obj_cfg)

        return multi_obj_prim

    def _build_multi_articulated_prim(self, obj_id, obj_cfg):
        from isaaclab.assets import ArticulationCfg
        from isaaclab_physx.assets import Articulation
        
        if (obj_cfg.disable_motors):
            control_mode = engine.ControlMode.none
        else:
            control_mode = self.get_control_mode()

        actuator_cfg = self._build_actuator_cfg(control_mode)

        regex = OBJ_PATH_TEMPLATE.format(".*", obj_id)
        multi_obj_cfg = ArticulationCfg(prim_path=regex, spawn=None,
                                        actuator_value_resolution_debug_print=False,
                                        actuators={ACTUATOR_GROUP: actuator_cfg})

        class UncheckedArticulation(Articulation):
            # Override Isaac Lab error when default joint positions are outside of joint limits
            # this is the case for some characters, but should not cause problems with the simulation
            def _validate_cfg(self):
                return
        
        multi_obj_prim = UncheckedArticulation(multi_obj_cfg)

        return multi_obj_prim
    
    def _clone_obj_prim(self, source_env_id, obj_id, clone_env_ids):
        import isaaclab.sim as sim_utils
        from isaaclab.cloner import usd_replicate

        source_path = OBJ_PATH_TEMPLATE.format(source_env_id, obj_id)
        clone_path_template = OBJ_PATH_TEMPLATE.format("{}", obj_id)
        usd_replicate(self._stage, sources=[source_path], destinations=[clone_path_template],
                      env_ids=torch.tensor(clone_env_ids))

        for clone_env_id in clone_env_ids:
            clone_cfg = self._obj_cfgs[clone_env_id][obj_id]
            clone_prim = self._stage.GetPrimAtPath(OBJ_PATH_TEMPLATE.format(clone_env_id, obj_id))
            sim_utils.standardize_xform_ops(clone_prim, translation=clone_cfg.start_pos.tolist(),
                                            orientation=clone_cfg.start_rot.tolist())
        return
    
    def _disable_prim_collisions(self, prim_path):
        import isaaclab.sim as sim_utils

        child_prims = sim_utils.get_all_matching_child_prims(prim_path, traverse_instance_prims=True)
        for col_prim in child_prims:
            if (col_prim.IsInstanceable()):
                col_prim.SetInstanceable(False)

        collision_props = sim_utils.schemas.CollisionBaseCfg(collision_enabled=False)
        sim_utils.schemas.modify_collision_properties(prim_path, collision_props)
        return

    def _get_sim_pd_gains(self, obj_id):
        obj = self._objs[obj_id]
        if (obj.actuators.has_implicit_actuators):
            kp = obj.data.joint_stiffness.torch
            kd = obj.data.joint_damping.torch
        else:
            actuator = obj.actuators[ACTUATOR_GROUP]
            kp = actuator.stiffness
            kd = actuator.damping
        return kp, kd
    
    def _build_body_order_tensors(self):
        self._body_order_sim2common = []
        self._body_order_common2sim = []
        self._dof_order_sim2common = []
        self._dof_order_common2sim = []

        objs_per_env = self.get_objs_per_env()
        for obj_id in range(objs_per_env):
            obj_type = self.get_obj_type(obj_id)
            if (obj_type == engine.ObjType.articulated):
                obj = self._objs[obj_id]
                body_order_sim2common, body_order_common2sim, dof_order_sim2common, dof_order_common2sim = self._build_body_order(obj)

                body_order_sim2common = torch.tensor(body_order_sim2common, device=self._device, dtype=torch.long)
                body_order_common2sim = torch.tensor(body_order_common2sim, device=self._device, dtype=torch.long)
                dof_order_sim2common = torch.tensor(dof_order_sim2common, device=self._device, dtype=torch.long)
                dof_order_common2sim = torch.tensor(dof_order_common2sim, device=self._device, dtype=torch.long)
            else:
                body_order_sim2common = torch.zeros(1, device=self._device, dtype=torch.long)
                body_order_common2sim = torch.zeros(1, device=self._device, dtype=torch.long)
                dof_order_sim2common = torch.zeros(1, device=self._device, dtype=torch.long)
                dof_order_common2sim = torch.zeros(1, device=self._device, dtype=torch.long)
            
            self._body_order_sim2common.append(body_order_sim2common)
            self._body_order_common2sim.append(body_order_common2sim)
            self._dof_order_sim2common.append(dof_order_sim2common)
            self._dof_order_common2sim.append(dof_order_common2sim)

        return

    def _build_sensor_order_tensors(self):
        self._sensor_body_order_sim2common = []

        objs_per_env = self.get_objs_per_env()
        for obj_id in range(objs_per_env):
            sensor = self._ground_contact_sensors[obj_id]
            if (sensor is not None):
                obj_type = self.get_obj_type(obj_id)
                if (obj_type == engine.ObjType.articulated):
                    body_names = sensor.body_names
                    body_common2sim = [self.find_obj_body_id(obj_id, name) for name in body_names]
                    body_sim2common = [body_common2sim.index(i) for i in range(len(body_common2sim))]

                    body_sim2common = torch.tensor(body_sim2common, device=self._device, dtype=torch.long)
                    self._sensor_body_order_sim2common.append(body_sim2common)
                else:
                    body_sim2common = torch.tensor([0], device=self._device, dtype=torch.long)
                    self._sensor_body_order_sim2common.append(body_sim2common)
            else:
                self._sensor_body_order_sim2common.append(None)
        return
    
    def _build_ground_contact_sensors(self):
        from isaaclab.sensors import ContactSensorCfg, ContactSensor

        ground_prim_paths = [GROUND_PATH + ".*"]
        
        self._ground_contact_sensors = []
        objs_per_env = self.get_objs_per_env()
        timestep = self.get_timestep()

        for obj_id in range(objs_per_env):
            # find child primitive that contains ContactReportAPI
            obj_path = OBJ_PATH_TEMPLATE.format(0, obj_id)
            contact_prim_path = self._find_contact_prim_path(obj_path)

            if (contact_prim_path is not None):
                contact_prim_name = os.path.basename(contact_prim_path)
                sensor_regex = OBJ_PATH_TEMPLATE.format(".*", obj_id) + "/{:s}/.*".format(contact_prim_name)

                sensor_cfg = ContactSensorCfg(prim_path=sensor_regex, 
                                              update_period=timestep,
                                              filter_prim_paths_expr=ground_prim_paths)
                sensor = ContactSensor(sensor_cfg)
            else:
                sensor = None
        
            self._ground_contact_sensors.append(sensor)
        return
    
    def _build_body_order(self, obj):
        meta_data = obj.root_view.shared_metatype
        link_names = meta_data.link_names
        link_parent_indices = meta_data.link_parent_indices
        joint_dof_counts = meta_data.joint_dof_counts
        joint_dof_offsets = meta_data.joint_dof_offsets

        num_links = len(link_names)
        link_children = [[] for i in range(num_links)]
        for link_name in link_names:
            if (link_name in link_parent_indices):
                parent_id = link_parent_indices[link_name]
                link_id = link_names.index(link_name)
                link_children[parent_id].append(link_id)
        
        body_order_sim2common = []
        def _dfs_children_links(link_id):
            body_order_sim2common.append(link_id)
            child_ids = link_children[link_id]
            for child_id in child_ids:
                _dfs_children_links(child_id)
        _dfs_children_links(0)

        dof_order_sim2common = []
        for link_id in body_order_sim2common[1:]:
            dof_offset = joint_dof_offsets[link_id - 1]
            dof_count = joint_dof_counts[link_id - 1]
            dof_indices = list(range(dof_offset, dof_offset + dof_count))
            dof_order_sim2common += dof_indices

        body_order_common2sim = [body_order_sim2common.index(i) for i in range(len(body_order_sim2common))]
        dof_order_common2sim = [dof_order_sim2common.index(i) for i in range(len(dof_order_sim2common))]

        return body_order_sim2common, body_order_common2sim, dof_order_sim2common, dof_order_common2sim

    def _build_sim_tensors(self):
        num_envs = self.get_num_envs()
        num_objs = self.get_objs_per_env()
        self._objs_need_reset = torch.zeros([num_envs, num_objs], device=self._device, dtype=torch.bool)

        self._body_forces = []
        for obj_id in range(num_objs):
            num_bodies = self.get_obj_num_bodies(obj_id)
            body_forces = torch.zeros([num_envs, num_bodies, 3], device=self._device, dtype=torch.float)
            self._body_forces.append(body_forces)
        return
    
    def _filter_env_collisions(self):
        from isaaclab.cloner import filter_collisions

        num_envs = self.get_num_envs()
        physics_scene_path = self._sim.cfg.physics_prim_path
        env_prim_paths = [ENV_PATH_TEMPLATE.format(i) for i in range(num_envs)]

        filter_collisions(self._stage, physics_scene_path, COLLISION_ROOT_PATH,
                          env_prim_paths, global_paths=[])
        return
    
    def _flag_obj_needs_reset(self, env_id, obj_id):
        if (env_id is None or len(env_id) > 0):
            env_ids = engine.none_to_env_ids(env_id)
            self._objs_need_reset[env_ids, obj_id] = True
        return
    
    def _update_reset_objs(self):
        num_objs = self.get_objs_per_env()

        for obj_id in range(num_objs):
            reset_flags = self._objs_need_reset[:, obj_id]
            env_ids = reset_flags.nonzero(as_tuple=False)
            env_ids = env_ids.flatten()

            if (len(env_ids) > 0):
                self._reset_obj(obj_id, env_ids)
                self._objs_need_reset[:, obj_id] = False
        return
    
    def _reset_obj(self, obj_id, env_ids):
        obj = self._objs[obj_id]
        obj_type = self.get_obj_type(obj_id)

        root_pose_data = obj.data.root_link_pose_w.torch[env_ids]
        root_vel_data = obj.data.root_link_vel_w.torch[env_ids]

        if (obj_type == engine.ObjType.articulated):
            dof_pos_data = obj.data.joint_pos.torch[env_ids]
            dof_vel_data = obj.data.joint_vel.torch[env_ids]

        obj.write_root_link_pose_to_sim_index(root_pose=root_pose_data, env_ids=env_ids)
        self._write_root_link_vel_to_sim(obj, root_vel_data, env_ids)

        if (obj_type == engine.ObjType.articulated):
            obj.write_joint_state_to_sim_index(position=dof_pos_data, velocity=dof_vel_data, env_ids=env_ids)

        obj.reset(env_ids)

        ground_contact_sensor = self._ground_contact_sensors[obj_id]
        if (ground_contact_sensor is not None):
            ground_contact_sensor.reset(env_ids)
        return

    def _write_root_link_vel_to_sim(self, obj, root_vel, env_ids):
        # jp hack: this should be removed once Isaac fixes this bug in a new version
        # Isaac Lab 3.0.0-EA writes the root link velocity of articulations to the simulator
        # without converting it to the velocity of the root's center of mass, which is the velocity
        # used by PhysX, so the conversion is done here before writing the center of mass velocity
        root_pos = obj.data.root_link_pose_w.torch[env_ids, :3]
        root_com_pos = obj.data.root_com_pose_w.torch[env_ids, :3]
        lin_vel = root_vel[:, :3]
        ang_vel = root_vel[:, 3:]

        com_lin_vel = lin_vel + torch.cross(ang_vel, root_com_pos - root_pos, dim=-1)
        root_com_vel = torch.cat([com_lin_vel, ang_vel], dim=-1)
        obj.write_root_com_velocity_to_sim_index(root_velocity=root_com_vel, env_ids=env_ids)
        return
    
    def _find_contact_prim_path(self, obj_path):
        from pxr import PhysxSchema

        obj_prim = self._stage.GetPrimAtPath(obj_path)
        prim_children = obj_prim.GetAllChildren()
                
        contact_prim_path = None
        for prim_child in prim_children:
            prim_grandchildren = prim_child.GetAllChildren()
                
            if (len(prim_grandchildren) > 0):
                prim_grandchild = prim_grandchildren[0]
                has_contact_api = prim_grandchild.HasAPI(PhysxSchema.PhysxContactReportAPI)
                    
                if (has_contact_api):
                    contact_prim_path = prim_child.GetPrimPath().pathString
                    break
        
        return contact_prim_path
    
    def _on_keyboard_event(self, event):
        if (event.type == carb.input.KeyboardEventType.KEY_PRESS):
            if (event.input in self._keyboard_callbacks):
                callback = self._keyboard_callbacks[event.input]
                callback()
        return

    def _build_video_recorder(self):
        self._video_recorder = isaac_lab_recorder.IsaacLabVideoRecorder(self)
        Logger.print("Video recording enabled")
        return

    def _hide_unrecorded_envs(self):
        from pxr import UsdGeom

        visible_env_radius = 20.0
        env_dist = torch.linalg.norm(self._env_offsets - self._env_offsets[0], dim=-1)
        hidden_env_ids = torch.nonzero(env_dist > visible_env_radius).squeeze(-1).tolist()

        for env_id in hidden_env_ids:
            env_prim = self._stage.GetPrimAtPath(ENV_PATH_TEMPLATE.format(env_id))
            UsdGeom.Imageable(env_prim).MakeInvisible()

        num_visible = self.get_num_envs() - len(hidden_env_ids)
        Logger.print("Rendering {:d}/{:d} envs for video recording".format(num_visible, self.get_num_envs()))
        return