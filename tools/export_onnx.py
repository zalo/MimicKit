"""Export MimicKit actor models to ONNX for the web demo.

Reconstructs the actor network directly from checkpoint weights — no env/engine needed.
Supports all agent types: ASE (with latent input), AMP, ADD, DeepMimic/PPO, AWR, LCP.
Bakes obs normalization, actor network, and action unnormalization into a single module.
Also bakes MJCF XML + config metadata into the ONNX file for single-file deployment.

Usage:
    # Export a single model:
    python tools/export_onnx.py \
        --arg_file args/ase_humanoid_sword_shield_args.txt \
        --model_file data/models/ase_humanoid_sword_shield_model.pt \
        --output web/ase_humanoid_sword_shield_actor.onnx

    # Export all pre-trained models in data/models/:
    python tools/export_onnx.py --batch
"""
import sys, os
import argparse
import json
import re
import numpy as np
import torch
import torch.nn as nn
import yaml

# ── Model-to-config mapping ─────────────────────────────────────────────────
MODEL_CONFIGS = {
    'ase_humanoid_sword_shield':             'args/ase_humanoid_sword_shield_args.txt',
    'amp_humanoid_spinkick':                 'args/amp_humanoid_args.txt',
    'amp_location_humanoid':                 'args/amp_location_humanoid_args.txt',
    'amp_steering_humanoid':                 'args/amp_steering_humanoid_args.txt',
    'amp_steering_humanoid_sword_shield':    'args/amp_steering_humanoid_sword_shield_args.txt',
    'add_humanoid_spinkick':                 'args/add_humanoid_args.txt',
    'add_g1_run':                            'args/add_g1_args.txt',
    'deepmimic_humanoid_spinkick':           'args/deepmimic_humanoid_ppo_args.txt',
    'deepmimic_humanoid_awr_spinkick':       'args/deepmimic_humanoid_awr_args.txt',
    'deepmimic_go2_pace':                    'args/deepmimic_go2_ppo_args.txt',
    'deepmimic_g1_double_kong':              'args/deepmimic_g1_ppo_args.txt',
    'deepmimic_g1_spinkick':                 'args/deepmimic_g1_ppo_args.txt',
    'deepmimic_pi_plus_walk':                'args/deepmimic_pi_plus_ppo_args.txt',
    'deepmimic_smpl_spinkick':              'args/deepmimic_smpl_ppo_args.txt',
    'lcp_g1_walk':                           'args/lcp_g1_ppo_args.txt',
}

MODEL_DISPLAY_NAMES = {
    'ase_humanoid_sword_shield':             'ASE Humanoid - Sword & Shield',
    'amp_humanoid_spinkick':                 'AMP Humanoid - Spinkick',
    'amp_location_humanoid':                 'AMP Humanoid - Location Task',
    'amp_steering_humanoid':                 'AMP Humanoid - Steering Task',
    'amp_steering_humanoid_sword_shield':    'AMP Humanoid S&S - Steering',
    'add_humanoid_spinkick':                 'ADD Humanoid - Spinkick',
    'add_g1_run':                            'ADD Unitree G1 - Run',
    'deepmimic_humanoid_spinkick':           'DeepMimic Humanoid - Spinkick',
    'deepmimic_humanoid_awr_spinkick':       'DeepMimic Humanoid AWR - Spinkick',
    'deepmimic_go2_pace':                    'DeepMimic Unitree Go2 - Pace',
    'deepmimic_g1_double_kong':              'DeepMimic Unitree G1 - Double Kong',
    'deepmimic_g1_spinkick':                 'DeepMimic Unitree G1 - Spinkick',
    'deepmimic_pi_plus_walk':                'DeepMimic Pi Plus - Walk',
    'deepmimic_smpl_spinkick':               'DeepMimic SMPL - Spinkick',
    'lcp_g1_walk':                           'LCP Unitree G1 - Walk',
}

parser = argparse.ArgumentParser()
parser.add_argument('--arg_file', default=None)
parser.add_argument('--model_file', default=None)
parser.add_argument('--output', default=None)
parser.add_argument('--batch', action='store_true',
                    help='Export all pre-trained models in data/models/')
parser.add_argument('--models_dir', default='data/models')
parser.add_argument('--out_dir', default='web')
cli_args = parser.parse_args()


# ── Helper: parse arg files to extract env/agent config paths ────────────────

def parse_arg_file(arg_file):
    """Parse a MimicKit args.txt file into a dict."""
    result = {}
    with open(arg_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith('--'):
                parts = line.split(None, 1)
                key = parts[0].lstrip('-')
                val = parts[1] if len(parts) > 1 else 'true'
                result[key] = val
    return result

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ── Reconstruct actor from checkpoint ────────────────────────────────────────

def reconstruct_actor(ckpt):
    """Reconstruct the actor MLP from checkpoint weight shapes.

    Returns (actor_layers, mean_net, obs_dim, act_dim, latent_dim, is_ase).
    """
    # Find actor layer keys and sort by index
    actor_keys = sorted([k for k in ckpt if k.startswith('_model._actor_layers.')])
    mean_w_key = '_model._action_dist._mean_net.weight'
    mean_b_key = '_model._action_dist._mean_net.bias'

    # Detect ASE by presence of encoder
    is_ase = any(k.startswith('_model._enc_') for k in ckpt)

    obs_dim = int(ckpt['_obs_norm._mean'].shape[0])
    act_dim = int(ckpt['_a_norm._mean'].shape[0])
    latent_dim = int(ckpt['_model._enc_out.weight'].shape[0]) if is_ase else 0

    # Build actor Sequential from weight shapes
    # Keys like: _model._actor_layers.0.weight, _model._actor_layers.0.bias,
    #            _model._actor_layers.2.weight, etc.
    # Extract layer indices that have weights (= Linear layers)
    prefix = '_model._actor_layers.'
    weight_layers = []
    for key in actor_keys:
        if key.endswith('.weight'):
            # e.g. "_model._actor_layers.0.weight" -> idx=0
            local = key[len(prefix):]  # "0.weight"
            idx = int(local.split('.')[0])
            w = ckpt[key]
            weight_layers.append((idx, w.shape[1], w.shape[0]))

    weight_layers.sort(key=lambda x: x[0])

    # Reconstruct Sequential: Linear + ReLU after EVERY linear layer (including last).
    # MimicKit's net_builder always appends activation after each Linear.
    sequential_layers = []
    for i, (idx, in_f, out_f) in enumerate(weight_layers):
        sequential_layers.append(nn.Linear(in_f, out_f))
        sequential_layers.append(nn.ReLU())

    actor_layers = nn.Sequential(*sequential_layers)
    mean_net = nn.Linear(ckpt[mean_w_key].shape[1], act_dim)

    # Load weights — map checkpoint keys to sequential indices
    # The sequential has Linear at positions 0, 2, 4, ... and ReLU at 1, 3, 5, ...
    # Checkpoint indices map directly (0, 2, 4, ... are Linear in the original Sequential)
    actor_state = {}
    for key in actor_keys:
        local_key = key[len(prefix):]  # e.g. "0.weight", "2.bias"
        # Remap the original indices to our sequential indices
        parts = local_key.split('.')
        orig_idx = int(parts[0])
        # Find which position this layer is in our sorted weight_layers
        seq_idx = None
        for pos, (widx, _, _) in enumerate(weight_layers):
            if widx == orig_idx:
                seq_idx = pos * 2  # account for ReLU layers between Linear layers
                break
        if seq_idx is not None:
            new_key = f'{seq_idx}.{parts[1]}'
            actor_state[new_key] = ckpt[key]
    actor_layers.load_state_dict(actor_state)

    mean_net.weight.data = ckpt[mean_w_key]
    mean_net.bias.data = ckpt[mean_b_key]

    return actor_layers, mean_net, obs_dim, act_dim, latent_dim, is_ase


class ActorWrapper(nn.Module):
    """Unified wrapper for all agent types.

    ASE: forward(obs, latent) → action
    Others: forward(obs) → action
    """
    def __init__(self, ckpt, is_ase):
        super().__init__()
        self.is_ase = is_ase

        self.register_buffer('obs_mean', ckpt['_obs_norm._mean'].clone())
        self.register_buffer('obs_std', ckpt['_obs_norm._std'].clone().clamp(min=1e-4))
        # obs_clip is always 10.0 in MimicKit
        self.obs_clip = 10.0

        self.register_buffer('a_mean', ckpt['_a_norm._mean'].clone())
        self.register_buffer('a_std', ckpt['_a_norm._std'].clone().clamp(min=1e-4))

        actor_layers, mean_net, obs_dim, act_dim, latent_dim, _ = reconstruct_actor(ckpt)
        self.actor_layers = actor_layers
        self.mean_net = mean_net
        self._obs_dim = obs_dim
        self._act_dim = act_dim
        self._latent_dim = latent_dim

    def forward(self, raw_obs, latent=None):
        norm_obs = torch.clamp((raw_obs - self.obs_mean) / self.obs_std,
                               -self.obs_clip, self.obs_clip)
        if self.is_ase and latent is not None:
            x = torch.cat([norm_obs, latent], dim=-1)
        else:
            x = norm_obs
        h = self.actor_layers(x)
        norm_action = self.mean_net(h)
        return norm_action * self.a_std + self.a_mean


def _exp_map_to_quat(exp_map):
    """Convert exponential map [ex, ey, ez] to quaternion [x, y, z, w]."""
    import math
    ex, ey, ez = float(exp_map[0]), float(exp_map[1]), float(exp_map[2])
    angle = math.sqrt(ex*ex + ey*ey + ez*ez)
    if angle < 1e-8:
        return [0.0, 0.0, 0.0, 1.0]
    axis = [ex/angle, ey/angle, ez/angle]
    half = angle / 2.0
    s = math.sin(half)
    return [axis[0]*s, axis[1]*s, axis[2]*s, math.cos(half)]


def detect_agent_type_from_config(agent_config_path):
    """Detect agent type from the agent config YAML."""
    cfg = load_yaml(agent_config_path)
    name = cfg.get('agent_name', '').upper()
    if name == 'ASE': return 'ase'
    if name == 'ADD': return 'add'
    if name == 'AMP': return 'amp'
    if name == 'AWR': return 'awr'
    if name == 'LCP': return 'lcp'
    if name == 'PPO': return 'ppo'
    return 'ppo'


def export_single(arg_file, model_file, output_path, model_basename=None):
    """Export a single model to ONNX with baked metadata."""
    print(f"\n{'='*70}")
    print(f"Exporting: {model_file}")
    print(f"  arg_file: {arg_file}")
    print(f"  output:   {output_path}")
    print(f"{'='*70}")

    # Parse config files
    args_dict = parse_arg_file(arg_file)
    env_config_path = args_dict.get('env_config')
    agent_config_path = args_dict.get('agent_config')
    env_config = load_yaml(env_config_path) if env_config_path else {}

    agent_type = detect_agent_type_from_config(agent_config_path) if agent_config_path else 'ppo'

    # Load checkpoint
    ckpt = torch.load(model_file, map_location='cpu', weights_only=False)

    # Check if ASE from checkpoint structure (more reliable than config)
    is_ase = any(k.startswith('_model._enc_') for k in ckpt)
    if is_ase:
        agent_type = 'ase'

    # Reconstruct and wrap
    wrapper = ActorWrapper(ckpt, is_ase)
    wrapper.eval()

    obs_dim = wrapper._obs_dim
    act_dim = wrapper._act_dim
    latent_dim = wrapper._latent_dim

    print(f"  agent_type={agent_type}, obs_dim={obs_dim}, act_dim={act_dim}, latent_dim={latent_dim}")

    # Export to ONNX
    dummy_obs = torch.randn(1, obs_dim)

    if is_ase:
        dummy_z = torch.randn(1, latent_dim)
        with torch.no_grad():
            pt_action = wrapper(dummy_obs, dummy_z)

        torch.onnx.export(
            wrapper, (dummy_obs, dummy_z), output_path,
            input_names=["obs", "latent"], output_names=["action"],
            dynamic_axes={"obs": {0: "batch"}, "latent": {0: "batch"}, "action": {0: "batch"}},
            opset_version=17,
        )

        import onnxruntime as ort
        sess = ort.InferenceSession(output_path)
        ort_action = sess.run(["action"], {"obs": dummy_obs.numpy(), "latent": dummy_z.numpy()})[0]
    else:
        with torch.no_grad():
            pt_action = wrapper(dummy_obs)

        # For non-ASE, forward signature is just (obs,)
        class SimpleForward(nn.Module):
            def __init__(self, w):
                super().__init__()
                self.w = w
            def forward(self, obs):
                return self.w(obs)

        simple = SimpleForward(wrapper)
        simple.eval()

        torch.onnx.export(
            simple, (dummy_obs,), output_path,
            input_names=["obs"], output_names=["action"],
            dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
            opset_version=17,
        )

        import onnxruntime as ort
        sess = ort.InferenceSession(output_path)
        ort_action = sess.run(["action"], {"obs": dummy_obs.numpy()})[0]

    max_diff = np.max(np.abs(pt_action.numpy() - ort_action))
    print(f"  PyTorch vs ONNX max_diff: {max_diff:.2e}")
    assert max_diff < 1e-3, f"ONNX diverges: {max_diff}"

    # ── Bake metadata ────────────────────────────────────────────────────────
    import onnx
    model = onnx.load(output_path)

    meta = {}
    meta['agent_type'] = agent_type
    meta['obs_dim'] = obs_dim
    meta['act_dim'] = act_dim
    meta['latent_dim'] = latent_dim
    if model_basename:
        meta['model_name'] = model_basename
        meta['display_name'] = MODEL_DISPLAY_NAMES.get(model_basename, model_basename)
    meta['obs_mean'] = wrapper.obs_mean.numpy().tolist()
    meta['obs_std'] = wrapper.obs_std.numpy().tolist()
    meta['a_mean'] = wrapper.a_mean.numpy().tolist()
    meta['a_std'] = wrapper.a_std.numpy().tolist()

    # Extract env config metadata
    char_file = env_config.get('char_file')
    init_pose = env_config.get('init_pose')
    global_obs = env_config.get('global_obs', False)
    key_bodies = env_config.get('key_bodies', [])

    if init_pose and isinstance(init_pose, list) and len(init_pose) > 6:
        # MimicKit init_pose format: [x, y, z, exp_x, exp_y, exp_z, dof0, dof1, ...]
        # [0:3] = root position, [3:6] = root rotation as exp map, [6:] = joint DOFs
        meta['init_root_pos'] = init_pose[:3]
        # Convert exp map to quaternion
        exp_map = init_pose[3:6]
        meta['init_root_rot_quat'] = _exp_map_to_quat(exp_map)
        meta['init_dof_pos'] = init_pose[6:]

    meta['global_obs'] = bool(global_obs)

    # Bake MJCF XML
    if char_file and os.path.isfile(char_file):
        with open(char_file) as f:
            meta['mjcf_xml'] = f.read()
        print(f"  Baked MJCF from {char_file} ({len(meta['mjcf_xml'])} chars)")

    # Compute pelvis_z and tpose_pelvis_z from init_pose
    if init_pose and len(init_pose) >= 3:
        meta['pelvis_z'] = float(init_pose[2])
        # T-pose pelvis is typically ~0.2m higher than init pose for humanoid characters
        # This is used for positioning during articulation construction
        meta['tpose_pelvis_z'] = float(init_pose[2]) + 0.2

    # For key_body_ids, we need to map body names to indices from the MJCF
    # This requires parsing the MJCF to get body order
    if char_file and os.path.isfile(char_file) and key_bodies:
        body_names = _extract_body_names(char_file)
        key_ids = []
        for name in key_bodies:
            if name in body_names:
                key_ids.append(body_names.index(name))
            else:
                print(f"  WARNING: key body '{name}' not found in MJCF")
        if key_ids:
            meta['key_body_ids'] = key_ids

    # Write metadata
    sentinel_key = 'mimickit_config'
    config_json = json.dumps(meta, separators=(',', ':'))
    model.metadata_props.append(onnx.StringStringEntryProto(key=sentinel_key, value=config_json))
    for key, value in meta.items():
        if key == 'mjcf_xml':
            continue
        model.metadata_props.append(onnx.StringStringEntryProto(key=key, value=json.dumps(value)))

    onnx.save(model, output_path, save_as_external_data=False)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    if size_mb < 0.1:
        print("  WARNING: File very small — re-saving with internal weights...")
        model = onnx.load(output_path)
        onnx.save(model, output_path, save_as_external_data=False)
        size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print(f"  Output: {output_path} ({size_mb:.1f} MB)")
    print(f"  Metadata: {len(meta)} keys ({len(config_json)} bytes)")
    return meta


def _extract_body_names(mjcf_path):
    """Extract body names in DFS order from MJCF XML."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    worldbody = root.find('worldbody')

    names = []
    def visit(elem):
        name = elem.get('name')
        if name:
            names.append(name)
        for child in elem.findall('body'):
            visit(child)

    for body in worldbody.findall('body'):
        visit(body)
    return names


def main():
    if cli_args.batch:
        models_dir = cli_args.models_dir
        out_dir = cli_args.out_dir
        os.makedirs(out_dir, exist_ok=True)

        pt_files = sorted(f for f in os.listdir(models_dir) if f.endswith('_model.pt'))
        if not pt_files:
            print(f"No *_model.pt files found in {models_dir}")
            return

        print(f"Found {len(pt_files)} models to export:")
        for f in pt_files:
            print(f"  {f}")

        manifest = {}
        failed = []

        for pt_file in pt_files:
            basename = pt_file.replace('_model.pt', '')
            arg_file = MODEL_CONFIGS.get(basename)
            if not arg_file:
                print(f"\n  SKIP: No arg_file mapping for '{basename}'")
                failed.append((basename, 'no arg_file mapping'))
                continue
            if not os.path.isfile(arg_file):
                print(f"\n  SKIP: arg_file not found: {arg_file}")
                failed.append((basename, f'arg_file missing: {arg_file}'))
                continue

            model_file = os.path.join(models_dir, pt_file)
            output_path = os.path.join(out_dir, f'{basename}_actor.onnx')

            try:
                meta = export_single(arg_file, model_file, output_path,
                                     model_basename=basename)
                manifest[basename] = {
                    'file': f'{basename}_actor.onnx',
                    'display_name': MODEL_DISPLAY_NAMES.get(basename, basename),
                    'agent_type': meta['agent_type'],
                    'obs_dim': meta['obs_dim'],
                    'act_dim': meta['act_dim'],
                    'latent_dim': meta['latent_dim'],
                }
            except Exception as e:
                print(f"\n  FAILED: {basename}: {e}")
                import traceback
                traceback.print_exc()
                failed.append((basename, str(e)))

        manifest_path = os.path.join(out_dir, 'models.json')
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"\n{'='*70}")
        print(f"Manifest written to {manifest_path}")
        print(f"  Exported: {len(manifest)}/{len(pt_files)}")
        if failed:
            print(f"  Failed ({len(failed)}):")
            for name, reason in failed:
                print(f"    {name}: {reason}")
        print(f"{'='*70}")

    else:
        arg_file = cli_args.arg_file or 'args/ase_humanoid_sword_shield_args.txt'
        model_file = cli_args.model_file or 'data/models/ase_humanoid_sword_shield_model.pt'
        output = cli_args.output or 'web/ase_humanoid_sword_shield_actor.onnx'
        export_single(arg_file, model_file, output)
        print("\nPASS: All verifications passed.")


if __name__ == "__main__":
    main()
