import abc
import enum
import time
import numpy as np
import torch

class ControlMode(enum.Enum):
    none = 0
    pos = 1
    vel = 2
    torque = 3
    pd_explicit = 4

class ObjType(enum.Enum):
    rigid = 0
    articulated = 1

def none_to_env_ids(env_ids):
    if (env_ids is None):
        env_ids = slice(None)
    return env_ids

class Engine(abc.ABC):
    def __init__(self, visualize):
        if (visualize):
            self._prev_frame_time = 0.0
        return
    
    @abc.abstractmethod
    def get_name(self):
        return
    
    @abc.abstractmethod
    def create_env(self):
        return
    
    @abc.abstractmethod
    def create_obj(self, env_id, obj_type, asset_file, name, is_visual=False, enable_self_collisions=True, 
                     fix_root=False, start_pos=None, start_rot=None, color=None, disable_motors=False):
        return

    def initialize_sim(self):
        self._build_dof_cmd()
        return
    
    def set_cmd(self, obj_id, cmd):
        self._obj_dof_cmd[obj_id][:] = cmd
        return

    @abc.abstractmethod
    def step(self):
        return
    
    @abc.abstractmethod
    def set_camera_pose(self, pos, look_at):
        return
    
    @abc.abstractmethod
    def get_camera_pos(self):
        return
    
    @abc.abstractmethod
    def get_camera_dir(self):
        return
    
    def render(self):
        now = time.time()
        delta = now - self._prev_frame_time
        time_step = self.get_timestep()

        if (delta < time_step):
            time.sleep(time_step - delta)

        self._prev_frame_time = time.time()
        return
    
    @abc.abstractmethod
    def get_timestep(self):
        return 0.0
    
    @abc.abstractmethod
    def get_num_envs(self):
        return 0
    
    @abc.abstractmethod
    def get_gravity(self):
        return np.array([0.0, 0.0, -9.81])
    
    @abc.abstractmethod
    def get_control_mode(self):
        return
    
    @abc.abstractmethod
    def get_obj_type(self, obj_id):
        return
    
    @abc.abstractmethod
    def get_obj_fix_root(self, obj_id):
        return

    @abc.abstractmethod
    def get_obj_num_dofs(self, obj_id):
        return
    
    @abc.abstractmethod
    def get_obj_num_bodies(self, obj_id):
        return
    
    @abc.abstractmethod
    def get_obj_body_names(self, obj_id):
        return

    @abc.abstractmethod
    def find_obj_body_id(self, obj_id, body_name):
        return 
    
    @abc.abstractmethod
    def get_obj_torque_limits(self, env_id, obj_id):
        return
    
    @abc.abstractmethod
    def get_obj_dof_limits(self, env_id, obj_id):
        return
    
    @abc.abstractmethod
    def get_obj_pd_gains(self, env_id, obj_id):
        return
    
    @abc.abstractmethod
    def calc_obj_mass(self, env_id, obj_id):
        return
    
    @abc.abstractmethod
    def get_root_pos(self, obj_id, env_ids=None):
        return
    
    @abc.abstractmethod
    def get_root_rot(self, obj_id, env_ids=None):
        return
    
    @abc.abstractmethod
    def get_root_vel(self, obj_id, env_ids=None):
        return
    
    @abc.abstractmethod
    def get_root_ang_vel(self, obj_id, env_ids=None):
        return
    
    @abc.abstractmethod
    def get_dof_pos(self, obj_id, env_ids=None):
        return
    
    @abc.abstractmethod
    def get_dof_vel(self, obj_id, env_ids=None):
        return
    
    @abc.abstractmethod
    def get_dof_forces(self, obj_id):
        return
    
    @abc.abstractmethod
    def get_body_pos(self, obj_id, env_ids=None):
        return
    
    @abc.abstractmethod
    def get_body_rot(self, obj_id, env_ids=None):
        return
    
    @abc.abstractmethod
    def get_body_vel(self, obj_id, env_ids=None):
        return
    
    @abc.abstractmethod
    def get_body_ang_vel(self, obj_id, env_ids=None):
        return
    
    @abc.abstractmethod
    def get_contact_forces(self, obj_id):
        return
    
    @abc.abstractmethod
    def get_ground_contact_forces(self, obj_id):
        return
    
    @abc.abstractmethod
    def set_root_pos(self, env_id, obj_id, root_pos):
        return
    
    @abc.abstractmethod
    def set_root_rot(self, env_id, obj_id, root_rot):
        return
    
    @abc.abstractmethod
    def set_root_vel(self, env_id, obj_id, root_vel):
        return
    
    @abc.abstractmethod
    def set_root_ang_vel(self, env_id, obj_id, root_ang_vel):
        return
    
    @abc.abstractmethod
    def set_dof_pos(self, env_id, obj_id, dof_pos):
        return
    
    @abc.abstractmethod
    def set_dof_vel(self, env_id, obj_id, dof_vel):
        return
    
    @abc.abstractmethod
    def set_body_vel(self, env_id, obj_id, body_vel):
        return
    
    @abc.abstractmethod
    def set_body_ang_vel(self, env_id, obj_id, body_ang_vel):
        return
    
    @abc.abstractmethod
    def set_body_forces(self, env_id, obj_id, body_id, forces):
        return

    def get_root_pos_obs(self, obj_id, env_ids=None):
        return self.get_root_pos(obj_id, env_ids)
    
    def get_root_rot_obs(self, obj_id, env_ids=None):
        return self.get_root_rot(obj_id, env_ids)
    
    def get_root_vel_obs(self, obj_id, env_ids=None):
        return self.get_root_vel(obj_id, env_ids)
    
    def get_root_ang_vel_obs(self, obj_id, env_ids=None):
        return self.get_root_ang_vel(obj_id, env_ids)
    
    def get_dof_pos_obs(self, obj_id, env_ids=None):
        return self.get_dof_pos(obj_id, env_ids)
    
    def get_dof_vel_obs(self, obj_id, env_ids=None):
        return self.get_dof_vel(obj_id, env_ids)
    
    def get_body_pos_obs(self, obj_id, env_ids=None):
        return self.get_body_pos(obj_id, env_ids)
    
    def get_body_rot_obs(self, obj_id, env_ids=None):
        return self.get_body_rot(obj_id, env_ids)
    
    def get_body_vel_obs(self, obj_id, env_ids=None):
        return self.get_body_vel(obj_id, env_ids)
    
    def get_body_ang_vel_obs(self, obj_id, env_ids=None):
        return self.get_body_ang_vel(obj_id, env_ids)
    
    def draw_lines(self, env_id, start_verts, end_verts, cols, line_width):
        return
    
    def register_keyboard_callback(self, key_str, callback_func):
        return

    def enabled_record_video(self):
        return False
  
    def get_video_recording(self):
        return None
 
    def start_video_recording(self):
        return

    def stop_video_recording(self):
        return

    def _has_dof(self, obj_id):
        return self.get_obj_num_dofs(obj_id) > 0

    def _build_dof_cmd(self):
        num_envs = self.get_num_envs()
        objs_per_env = self.get_objs_per_env()
        obj_num_dofs = [self.get_obj_num_dofs(obj_id) for obj_id in range(objs_per_env)]
        total_dofs = sum(obj_num_dofs)

        self._flat_dof_cmd = torch.zeros([num_envs, total_dofs], device=self._device, dtype=torch.float32)
        self._obj_dof_cmd = []
        dof_offset = 0

        for num_dofs in obj_num_dofs:
            obj_dof_cmd = self._flat_dof_cmd[:, dof_offset:dof_offset + num_dofs]
            self._obj_dof_cmd.append(obj_dof_cmd)
            dof_offset += num_dofs
        return

    def _get_obj_dof_cmd(self, obj_id):
        return self._obj_dof_cmd[obj_id]

    @abc.abstractmethod
    def _apply_cmd(self):
        return
