# MimicKit Unity Port

Native Unity port of the MimicKit web demo. Runs an ASE (Adversarial Skill Embeddings) humanoid
policy using Unity's built-in PhysX ArticulationBody system and the Inference Engine (Sentis) for
ONNX neural network inference.

## Requirements

- Unity 6000.0+ (Unity 6)
- Packages (auto-resolved from `Packages/manifest.json`):
  - `com.unity.ai.inference` 2.5+ (ONNX inference)
  - `com.unity.nuget.newtonsoft-json` 3.2+ (JSON parsing)
  - `com.unity.modules.physics` (PhysX)

## Setup

1. Open Unity Hub and create a new project, or open an existing one.
2. Copy the contents of this `unity/` folder into your project root (merging `Assets/`, `Packages/`, `ProjectSettings/`).
3. Let Unity resolve packages (it will download Inference Engine and Newtonsoft.Json).
4. Open or create a scene.
5. Create an empty GameObject, name it `MimicKit`.
6. Add the `MimicKitController` component.
7. Drag `Assets/MimicKit/Models/ase_humanoid_sword_shield_actor.onnx` into the **Model Asset** field.
8. (Optional) Add the `MimicKitSceneSetup` component for auto-creating ground, camera, and lighting.
9. Press Play.

## How It Works

The ONNX model contains baked metadata (`mimickit_config`) with:
- **MJCF XML**: Full MuJoCo character description (bodies, joints, geoms, actuators)
- **Config**: Observation/action dimensions, normalization stats, initial pose, action bounds, key body IDs

At runtime, `MimicKitController`:
1. Loads the ONNX model via Unity Inference Engine
2. Reads `mimickit_config` from `ONNXModelMetadata.MetadataProps`
3. Parses the MJCF XML to build an `ArticulationBody` hierarchy with correct joint types, limits, drives, mass, and inertia
4. Each `FixedUpdate` (at 120Hz), runs 1 physics substep; every 4th substep, runs the neural network policy
5. The policy outputs joint position targets, which are applied as ArticulationDrive targets

## PhysX Settings

The `ProjectSettings/DynamicsManager.asset` configures PhysX to match Isaac Lab training:
- **Solver**: TGS (solverType: 1)
- **Gravity**: -9.81 Y
- **Fixed Timestep**: 1/120s (TimeManager.asset)
- **Solver Iterations**: 16 position, 4 velocity
- **Bounce Threshold**: 0.2
- **Sleep Threshold**: 5e-5
- **Friction**: Patch friction

The `TagManager.asset` defines layer 8 as "Humanoid" with self-collision disabled.

## Skill Presets

Set `presetIndex` in the inspector to select a latent skill:
- 0: Walk Forward
- 1: Stand Still
- 2: Frenzy Attack
- 3: Overhead Strike
- 4: Reliable Kick
- 5: Jump Back
- -1: Random latent

## Architecture

```
MimicKitController.cs
├── LoadModel()          - Load ONNX via ModelLoader
├── ParseMetadata()      - Extract mimickit_config JSON from ONNX metadata
├── BuildArticulation()  - Create ArticulationBody hierarchy from MJCF
├── RunPolicyStep()      - Build observation → ONNX inference → apply actions
├── BuildObservation()   - Compute 158-dim observation vector (matches Python)
└── ApplyActions()       - Set ArticulationDrive targets from policy output

MJCFParser.cs (static)
└── Parse()              - Convert MJCF XML → bodies, joints, geoms, FK data
```
