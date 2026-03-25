"""Export the original ASE LLC (from the ASE repo) to ONNX.

The ASE LLC takes (obs_253, latent_64) → action_31.
It uses a different obs format than MimicKit (body-centric vs joint-centric).

Usage:
    python tools/export_ase_llc_onnx.py
"""
import os, json, torch, torch.nn as nn, numpy as np
import onnx, onnxruntime as ort

ASE_MODEL = '/home/selstad/Desktop/ASE/ase/data/models/ase_llc_reallusion_sword_shield.pth'
OUTPUT = 'web/ase_llc.onnx'

print(f"Loading ASE LLC from {ASE_MODEL}")
ckpt = torch.load(ASE_MODEL, map_location='cpu', weights_only=False)

class ASELLCWrapper(nn.Module):
    """Wraps the original ASE LLC: (raw_obs, latent) → action."""
    def __init__(self, ckpt):
        super().__init__()
        rms = ckpt['running_mean_std']
        self.register_buffer('obs_mean', rms['running_mean'].float())
        self.register_buffer('obs_var', rms['running_var'].float())
        self.obs_clip = 5.0

        model = ckpt['model']
        # The ASE actor has a style MLP that processes the latent,
        # then concatenates with the main MLP input
        # Architecture: actor_mlp._style_mlp processes latent
        #               actor_mlp._dense_layers processes [obs, style_features]
        #               mu outputs action

        # Style MLP: latent(64) → 512 → ReLU → 256 → ReLU
        self.style_mlp = nn.Sequential(
            nn.Linear(model['a2c_network.actor_mlp._style_mlp.0.weight'].shape[1],
                      model['a2c_network.actor_mlp._style_mlp.0.weight'].shape[0]),
            nn.ReLU(),
            nn.Linear(model['a2c_network.actor_mlp._style_mlp.2.weight'].shape[1],
                      model['a2c_network.actor_mlp._style_mlp.2.weight'].shape[0]),
            nn.ReLU(),
        )
        self.style_mlp[0].weight.data = model['a2c_network.actor_mlp._style_mlp.0.weight']
        self.style_mlp[0].bias.data = model['a2c_network.actor_mlp._style_mlp.0.bias']
        self.style_mlp[2].weight.data = model['a2c_network.actor_mlp._style_mlp.2.weight']
        self.style_mlp[2].bias.data = model['a2c_network.actor_mlp._style_mlp.2.bias']

        # Style dense: 256 → 64 (compresses style features)
        self.style_dense = nn.Linear(
            model['a2c_network.actor_mlp._style_dense.weight'].shape[1],
            model['a2c_network.actor_mlp._style_dense.weight'].shape[0])
        self.style_dense.weight.data = model['a2c_network.actor_mlp._style_dense.weight']
        self.style_dense.bias.data = model['a2c_network.actor_mlp._style_dense.bias']

        # Dense layers: [obs(253) + style(64) = 317] → 1024 → 1024 → 512
        # Note: these use indices 0,1,2 (no ReLU interleaved in state_dict)
        self.dense = nn.Sequential(
            nn.Linear(model['a2c_network.actor_mlp._dense_layers.0.weight'].shape[1],
                      model['a2c_network.actor_mlp._dense_layers.0.weight'].shape[0]),
            nn.ReLU(),
            nn.Linear(model['a2c_network.actor_mlp._dense_layers.1.weight'].shape[1],
                      model['a2c_network.actor_mlp._dense_layers.1.weight'].shape[0]),
            nn.ReLU(),
            nn.Linear(model['a2c_network.actor_mlp._dense_layers.2.weight'].shape[1],
                      model['a2c_network.actor_mlp._dense_layers.2.weight'].shape[0]),
            nn.ReLU(),
        )
        self.dense[0].weight.data = model['a2c_network.actor_mlp._dense_layers.0.weight']
        self.dense[0].bias.data = model['a2c_network.actor_mlp._dense_layers.0.bias']
        self.dense[2].weight.data = model['a2c_network.actor_mlp._dense_layers.1.weight']
        self.dense[2].bias.data = model['a2c_network.actor_mlp._dense_layers.1.bias']
        self.dense[4].weight.data = model['a2c_network.actor_mlp._dense_layers.2.weight']
        self.dense[4].bias.data = model['a2c_network.actor_mlp._dense_layers.2.bias']

        # Output: mu(512 → 31)
        self.mu = nn.Linear(model['a2c_network.mu.weight'].shape[1],
                            model['a2c_network.mu.weight'].shape[0])
        self.mu.weight.data = model['a2c_network.mu.weight']
        self.mu.bias.data = model['a2c_network.mu.bias']

    def forward(self, obs: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        # Normalize obs
        norm_obs = (obs - self.obs_mean) / torch.sqrt(self.obs_var + 1e-5)
        norm_obs = torch.clamp(norm_obs, -self.obs_clip, self.obs_clip)
        # Style features from latent
        style = self.style_mlp(latent)
        style = self.style_dense(style)
        # Concatenate obs + style
        x = torch.cat([norm_obs, style], dim=-1)
        h = self.dense(x)
        action = self.mu(h)
        return action

wrapper = ASELLCWrapper(ckpt)
wrapper.eval()

obs_dim = wrapper.obs_mean.shape[0]
latent_dim = 64
act_dim = wrapper.mu.weight.shape[0]
print(f"obs_dim={obs_dim}, latent_dim={latent_dim}, act_dim={act_dim}")

# Export
dummy_obs = torch.randn(1, obs_dim)
dummy_z = torch.randn(1, latent_dim)

with torch.no_grad():
    pt_out = wrapper(dummy_obs, dummy_z)

torch.onnx.export(wrapper, (dummy_obs, dummy_z), OUTPUT,
    input_names=['obs', 'latent'], output_names=['action'],
    dynamic_axes={'obs': {0: 'batch'}, 'latent': {0: 'batch'}, 'action': {0: 'batch'}},
    opset_version=17)

# Verify
sess = ort.InferenceSession(OUTPUT)
ort_out = sess.run(['action'], {'obs': dummy_obs.numpy(), 'latent': dummy_z.numpy()})[0]
diff = np.max(np.abs(pt_out.numpy() - ort_out))
print(f"Max diff: {diff:.2e}")
assert diff < 1e-3

# Bake metadata
model = onnx.load(OUTPUT)
meta = {
    'model_type': 'ase_llc',
    'obs_format': 'ase_body_centric',
    'obs_dim': obs_dim,
    'latent_dim': latent_dim,
    'act_dim': act_dim,
    'obs_layout': {
        'root_h': [0, 1],
        'local_body_pos': [1, 49],
        'local_body_rot': [49, 151],
        'local_body_vel': [151, 202],
        'local_body_ang_vel': [202, 253],
    },
    'obs_mean': wrapper.obs_mean.numpy().tolist(),
    'obs_var': wrapper.obs_var.numpy().tolist(),
}
entry = onnx.StringStringEntryProto(key='mimickit_config', value=json.dumps(meta, separators=(',', ':')))
model.metadata_props.append(entry)
onnx.save(model, OUTPUT, save_as_external_data=False)

size_kb = os.path.getsize(OUTPUT) / 1024
print(f"Exported: {OUTPUT} ({size_kb:.0f} KB)")
print("PASS")
