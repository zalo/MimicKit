using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Unity.InferenceEngine;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

/// <summary>
/// Loads a MimicKit ONNX model, parses its baked MJCF + config metadata,
/// builds a Unity ArticulationBody hierarchy, and runs the neural network
/// policy at 30Hz to drive joint targets.
///
/// Drop this on an empty GameObject. Assign the ONNX ModelAsset in the inspector.
/// Requires packages: com.unity.ai.inference (2.5+), com.unity.nuget.newtonsoft-json.
/// </summary>
public class MimicKitController : MonoBehaviour
{
    [Header("Model")]
    [Tooltip("ONNX model asset with baked MimicKit metadata")]
    public ModelAsset modelAsset;

    [Header("Simulation")]
    [Tooltip("Physics substeps per policy step (4 = 120Hz physics / 30Hz policy)")]
    public int substeps = 4;

    [Header("Skill Preset")]
    [Tooltip("Index into latent presets (0 = Walk Forward, -1 = random)")]
    public int presetIndex = 0;

    // --- Parsed metadata ---
    MimicKitConfig config;
    MJCFData mjcf;

    // --- Unity objects ---
    ArticulationBody rootBody;
    List<BodyEntry> bodyEntries = new List<BodyEntry>();
    Dictionary<string, ArticulationBody> bodyMap = new Dictionary<string, ArticulationBody>();

    // --- Inference ---
    Worker worker;
    Tensor<float> obsTensor;
    Tensor<float> latentTensor;
    float[] latentVec;
    float[] currentAction;
    int physStep;

    // --- Latent presets (from latent_codes.txt, embedded) ---
    static readonly Dictionary<string, float[]> Presets = new Dictionary<string, float[]>();
    static readonly List<string> PresetNames = new List<string>();

    struct BodyEntry
    {
        public string name;
        public ArticulationBody body;
        public Transform transform;
    }

    // ===== Lifecycle =====

    void Start()
    {
        LoadModel();
        ParseMetadata();
        BuildArticulation();
        InitInference();
        SetLatentPreset(presetIndex);
    }

    void FixedUpdate()
    {
        physStep++;
        if (physStep >= substeps)
        {
            physStep = 0;
            RunPolicyStep();
        }
    }

    void OnDestroy()
    {
        obsTensor?.Dispose();
        latentTensor?.Dispose();
        worker?.Dispose();
    }

    // ===== Model Loading =====

    void LoadModel()
    {
        if (modelAsset == null)
            throw new Exception("MimicKitController: Assign the ONNX ModelAsset in the inspector.");

        var model = ModelLoader.Load(modelAsset);
        worker = new Worker(model, BackendType.CPU);
    }

    void ParseMetadata()
    {
        // Read metadata from ONNX via Unity's ONNXModelMetadata
        var meta = ONNXModelMetadata.CreateFrom(modelAsset);
        if (meta.MetadataProps == null || !meta.MetadataProps.ContainsKey("mimickit_config"))
            throw new Exception("ONNX model missing 'mimickit_config' metadata key.");

        string json = meta.MetadataProps["mimickit_config"];
        var root = JObject.Parse(json);

        config = new MimicKitConfig
        {
            obs_dim = root["obs_dim"]?.Value<int>() ?? 158,
            act_dim = root["act_dim"]?.Value<int>() ?? 31,
            latent_dim = root["latent_dim"]?.Value<int>() ?? 64,
            obs_mean = ParseFloatArray(root["obs_mean"]),
            obs_std = ParseFloatArray(root["obs_std"]),
            init_dof_pos = ParseFloatArray(root["init_dof_pos"]),
            init_root_pos = ParseFloatArray(root["init_root_pos"]),
            init_root_rot_quat = ParseFloatArray(root["init_root_rot_quat"]),
            action_low = ParseFloatArray(root["action_low"]),
            action_high = ParseFloatArray(root["action_high"]),
            key_body_ids = ParseIntArray(root["key_body_ids"]),
            global_obs = root["global_obs"]?.Value<bool>() ?? false,
            pelvis_z = root["pelvis_z"]?.Value<float>() ?? 0.93f,
            tpose_pelvis_z = root["tpose_pelvis_z"]?.Value<float>() ?? 0.93f,
        };

        // Parse MJCF XML baked inside metadata
        string mjcfXml = root["mjcf_xml"]?.Value<string>();
        if (string.IsNullOrEmpty(mjcfXml))
            throw new Exception("ONNX metadata missing mjcf_xml.");

        mjcf = MJCFParser.Parse(mjcfXml);
        Debug.Log($"MimicKit: Parsed {mjcf.bodies.Count} bodies, {mjcf.joints.Count} joints, {mjcf.actuatorOrder.Count} actuators");

        // Parse latent presets from metadata if available, else use hardcoded
        LoadLatentPresets(root);
    }

    // ===== Articulation Construction =====

    void BuildArticulation()
    {
        // The MJCF data is Z-up. Unity is Y-up.
        // We build everything in Unity's Y-up frame by swapping Z<->Y during construction.

        foreach (var body in mjcf.bodies)
        {
            GameObject go = new GameObject(body.name);

            ArticulationBody ab;
            if (body.parent == null)
            {
                // Root body - attach to this GameObject
                go.transform.SetParent(transform, false);
                ab = go.AddComponent<ArticulationBody>();
                rootBody = ab;

                // Root is free (default ArticulationJointType for root)
                ab.useGravity = true;
                ab.immovable = false;
            }
            else
            {
                go.transform.SetParent(bodyMap[body.parent].transform, false);
                ab = go.AddComponent<ArticulationBody>();
            }

            // Position in parent-local space (Z-up -> Y-up: swap Y and Z)
            go.transform.localPosition = ZupToYup(body.localPos);

            // Mass and inertia
            ab.mass = Mathf.Max(body.mass, 0.001f);
            if (body.inertia != null)
            {
                // Swap Y/Z for inertia tensor diagonal
                ab.inertiaTensor = new Vector3(body.inertia[0], body.inertia[2], body.inertia[1]);
                ab.inertiaTensorRotation = Quaternion.identity;
            }
            if (body.com != null)
                ab.centerOfMass = new Vector3(body.com[0], body.com[2], body.com[1]);

            // Match Isaac Lab rigid body properties
            ab.linearDamping = 0.0f;
            ab.angularDamping = 0.01f;
            ab.maxDepenetrationVelocity = 10f;
            ab.maxLinearVelocity = 1000f;
            ab.maxAngularVelocity = 1000f;
            ab.solverIterations = 16;
            ab.solverVelocityIterations = 4;
            ab.sleepThreshold = 5e-5f;

            // Add colliders for each geom
            foreach (var geom in body.geoms)
                AddCollider(go, geom);

            bodyMap[body.name] = ab;
            bodyEntries.Add(new BodyEntry { name = body.name, body = ab, transform = go.transform });
        }

        // Configure joints
        foreach (var jdata in mjcf.joints)
        {
            if (!bodyMap.ContainsKey(jdata.child_body)) continue;
            var ab = bodyMap[jdata.child_body];

            // Parent anchor = localPos in parent frame (Z-up -> Y-up)
            ab.anchorPosition = Vector3.zero;
            ab.parentAnchorPosition = ZupToYup(jdata.localPos0);

            // Joint frame rotation (Z-up quat [w,x,y,z] -> Y-up)
            var lr = jdata.localRot; // [w,x,y,z]
            Quaternion jointFrame = ZupQuatToYup(new Quaternion(lr[1], lr[2], lr[3], lr[0]));
            ab.anchorRotation = jointFrame;
            ab.parentAnchorRotation = jointFrame;

            ab.jointFriction = 0f;
            ab.maxJointVelocity = 1000000f;

            if (jdata.jointType == "spherical")
            {
                ab.jointType = ArticulationJointType.SphericalJoint;

                // Configure drives for each axis
                for (int i = 0; i < jdata.axes.Count; i++)
                {
                    var ax = jdata.axes[i];
                    int physAxis = jdata.axisMap[i]; // 0=twist, 1=swing1, 2=swing2

                    var drive = new ArticulationDrive
                    {
                        stiffness = ax.stiffness,
                        damping = ax.damping,
                        forceLimit = ax.maxForce,
                        lowerLimit = ax.range[0] * Mathf.Rad2Deg,
                        upperLimit = ax.range[1] * Mathf.Rad2Deg,
                    };

                    switch (physAxis)
                    {
                        case 0: // twist (X)
                            ab.twistLock = ArticulationDofLock.LimitedMotion;
                            ab.xDrive = drive;
                            break;
                        case 1: // swing1 (Y)
                            ab.swingYLock = ArticulationDofLock.LimitedMotion;
                            ab.yDrive = drive;
                            break;
                        case 2: // swing2 (Z)
                            ab.swingZLock = ArticulationDofLock.LimitedMotion;
                            ab.zDrive = drive;
                            break;
                    }
                }

                // Lock unused axes
                for (int i = jdata.axes.Count; i < 3; i++)
                {
                    int physAxis = i; // Force consecutive [0,1,2]
                    switch (physAxis)
                    {
                        case 0: ab.twistLock = ArticulationDofLock.LockedMotion; break;
                        case 1: ab.swingYLock = ArticulationDofLock.LockedMotion; break;
                        case 2: ab.swingZLock = ArticulationDofLock.LockedMotion; break;
                    }
                }
            }
            else // revolute
            {
                ab.jointType = ArticulationJointType.RevoluteJoint;
                var ax = jdata.axes[0];
                ab.twistLock = ArticulationDofLock.LimitedMotion;
                ab.xDrive = new ArticulationDrive
                {
                    stiffness = ax.stiffness,
                    damping = ax.damping,
                    forceLimit = ax.maxForce,
                    lowerLimit = ax.range[0] * Mathf.Rad2Deg,
                    upperLimit = ax.range[1] * Mathf.Rad2Deg,
                };
                ab.swingYLock = ArticulationDofLock.LockedMotion;
                ab.swingZLock = ArticulationDofLock.LockedMotion;
            }
        }

        // Fixed joints
        foreach (var fj in mjcf.fixedJoints)
        {
            if (!bodyMap.ContainsKey(fj.child_body)) continue;
            var ab = bodyMap[fj.child_body];
            ab.jointType = ArticulationJointType.FixedJoint;
            ab.parentAnchorPosition = ZupToYup(fj.localPos0);
        }

        // Disable self-collision on the articulation (matching training)
        if (rootBody != null)
        {
            // Unity doesn't have a single flag; we rely on collision layer filtering.
            // Place the humanoid on a dedicated layer and disable self-collision in Physics settings.
            int layer = LayerMask.NameToLayer("Humanoid");
            if (layer < 0) layer = 8; // fallback to layer 8
            SetLayerRecursive(rootBody.gameObject, layer);
            Physics.IgnoreLayerCollision(layer, layer, true);
        }

        // Apply initial pose
        ApplyInitPose();

        Debug.Log($"MimicKit: Built articulation with {bodyEntries.Count} links");
    }

    void ApplyInitPose()
    {
        if (config.init_root_pos == null || config.init_dof_pos == null) return;

        var rp = config.init_root_pos;
        var rq = config.init_root_rot_quat; // [x,y,z,w]

        // Teleport root (Z-up -> Y-up)
        Vector3 rootPos = ZupToYup(rp);
        Quaternion rootRot = ZupQuatToYup(new Quaternion(rq[0], rq[1], rq[2], rq[3]));
        rootBody.TeleportRoot(rootPos, rootRot);

        // Set DOF positions
        // PhysX articulation DOF order: iterate joints in hierarchy order
        SetDofPositions(config.init_dof_pos);

        // Also set drive targets to init pose
        SetDriveTargets(config.init_dof_pos);
    }

    void SetDofPositions(float[] dofPos)
    {
        var dofStarts = new List<int>();
        rootBody.GetDofStartIndices(dofStarts);

        for (int i = 0; i < mjcf.dofInfo.Count; i++)
        {
            var dof = mjcf.dofInfo[i];
            if (!bodyMap.ContainsKey(dof.child_body)) continue;
            var ab = bodyMap[dof.child_body];
            int bodyIdx = bodyEntries.FindIndex(e => e.name == dof.child_body);
            if (bodyIdx < 0 || bodyIdx >= dofStarts.Count) continue;

            int startIdx = dofStarts[bodyIdx];
            int localAxis = dof.physx_axis;

            // jointPosition is a reducedCoordinate buffer indexed by dofStart + localAxis
            var jp = new ArticulationReducedSpace(0f, 0f, 0f);
            jp[localAxis] = dofPos[i];
            ab.jointPosition = jp;
            ab.jointVelocity = new ArticulationReducedSpace(0f, 0f, 0f);
        }
    }

    void SetDriveTargets(float[] targets)
    {
        for (int i = 0; i < mjcf.dofInfo.Count; i++)
        {
            var dof = mjcf.dofInfo[i];
            if (!bodyMap.ContainsKey(dof.child_body)) continue;
            var ab = bodyMap[dof.child_body];

            // Drive targets in Unity are in degrees for revolute, radians for spherical expmap
            float val = targets[i];
            int axis = dof.physx_axis;

            // SetDriveTarget uses the reduced coordinate index
            // For spherical: twist=0, swing1=1, swing2=2
            // The target is in radians (PhysX internal)
            var drive = GetDrive(ab, axis);
            drive.target = val * Mathf.Rad2Deg;
            SetDrive(ab, axis, drive);
        }
    }

    // ===== Inference =====

    void InitInference()
    {
        latentVec = new float[config.latent_dim];
        currentAction = new float[config.act_dim];
    }

    void SetLatentPreset(int index)
    {
        if (PresetNames.Count == 0)
        {
            // Fallback: random latent
            RandomLatent();
            return;
        }

        index = Mathf.Clamp(index, 0, PresetNames.Count - 1);
        var preset = Presets[PresetNames[index]];
        Array.Copy(preset, latentVec, config.latent_dim);
        Debug.Log($"MimicKit: Loaded preset '{PresetNames[index]}'");
    }

    void RandomLatent()
    {
        float norm = 0;
        for (int i = 0; i < latentVec.Length; i++)
        {
            latentVec[i] = GaussianRandom();
            norm += latentVec[i] * latentVec[i];
        }
        norm = Mathf.Sqrt(norm);
        if (norm > 0)
            for (int i = 0; i < latentVec.Length; i++)
                latentVec[i] /= norm;
    }

    void RunPolicyStep()
    {
        float[] obs = BuildObservation();

        // Create tensors
        obsTensor?.Dispose();
        latentTensor?.Dispose();
        obsTensor = new Tensor<float>(new TensorShape(1, config.obs_dim), obs);
        latentTensor = new Tensor<float>(new TensorShape(1, config.latent_dim), latentVec);

        worker.SetInput("obs", obsTensor);
        worker.SetInput("latent", latentTensor);
        worker.Schedule();

        var output = worker.PeekOutput("action") as Tensor<float>;
        if (output != null)
        {
            output.ReadbackAndClone().CopyTo(currentAction, 0);
        }

        ApplyActions(currentAction);
    }

    void ApplyActions(float[] action)
    {
        for (int i = 0; i < mjcf.dofInfo.Count; i++)
        {
            var dof = mjcf.dofInfo[i];
            if (!bodyMap.ContainsKey(dof.child_body)) continue;
            var ab = bodyMap[dof.child_body];

            float a = action[i];
            // Clip to action bounds
            if (config.action_low != null && config.action_high != null)
                a = Mathf.Clamp(a, config.action_low[i], config.action_high[i]);

            int axis = dof.physx_axis;
            var drive = GetDrive(ab, axis);
            drive.target = a * Mathf.Rad2Deg;
            SetDrive(ab, axis, drive);
        }
    }

    // ===== Observation Building =====

    float[] BuildObservation()
    {
        float[] obs = new float[config.obs_dim];
        int idx = 0;

        // Root state (convert Y-up back to Z-up for the policy)
        Vector3 rootPosYup = rootBody.transform.position;
        Quaternion rootRotYup = rootBody.transform.rotation;
        float[] rootPos = YupToZup(rootPosYup);
        float[] rootRot = YupQuatToZup(rootRotYup); // [x,y,z,w]

        Vector3 rvYup = rootBody.velocity;
        float[] rootVel = YupToZup(rvYup);
        Vector3 rawYup = rootBody.angularVelocity;
        float[] rootAngVel = YupToZup(rawYup);

        float heading = CalcHeading(rootRot);
        float[] headingInv = AxisAngleToQuat(new float[] { 0, 0, 1 }, -heading);

        // [0] Root height (Z in Z-up)
        obs[idx++] = rootPos[2];

        // [1..6] Root rotation tan_norm
        float[] localRootRot;
        if (config.global_obs)
            localRootRot = rootRot;
        else
            localRootRot = QuatMul(headingInv, rootRot);
        float[] tn = QuatToTanNorm(localRootRot);
        for (int i = 0; i < 6; i++) obs[idx++] = tn[i];

        // [7..9] Root velocity
        float[] lv = config.global_obs ? rootVel : QuatRotateVec(headingInv, rootVel);
        obs[idx++] = lv[0]; obs[idx++] = lv[1]; obs[idx++] = lv[2];

        // [10..12] Root angular velocity
        float[] la = config.global_obs ? rootAngVel : QuatRotateVec(headingInv, rootAngVel);
        obs[idx++] = la[0]; obs[idx++] = la[1]; obs[idx++] = la[2];

        // [13..108] Joint rotations as tan_norm
        float[] dofPositions = ReadDofPositions();
        List<float[]> jointRotQuats = new List<float[]>();

        foreach (var kj in mjcf.kinematicJoints)
        {
            float[] quat;
            if (kj.type == "SPHERICAL")
                quat = ExpMapToQuat(dofPositions[kj.dof_idx], dofPositions[kj.dof_idx + 1], dofPositions[kj.dof_idx + 2]);
            else if (kj.type == "HINGE")
                quat = AxisAngleToQuat(kj.axis, dofPositions[kj.dof_idx]);
            else
                quat = new float[] { 0, 0, 0, 1 };

            jointRotQuats.Add(quat);
            float[] jtn = QuatToTanNorm(quat);
            for (int k = 0; k < 6; k++) obs[idx++] = jtn[k];
        }

        // DOF velocities
        float[] dofVel = ReadDofVelocities();
        for (int i = 0; i < dofVel.Length; i++) obs[idx++] = dofVel[i];

        // Key body positions via FK
        if (mjcf.fk_parent_indices != null)
        {
            int numBodies = mjcf.fk_parent_indices.Length;
            float[][] fkPos = new float[numBodies][];
            float[][] fkRot = new float[numBodies][];
            fkPos[0] = rootPos;
            fkRot[0] = rootRot;

            for (int j = 1; j < numBodies; j++)
            {
                float[] jRot = jointRotQuats[j - 1];
                float[] localTrans = mjcf.fk_local_translations[j];
                float[] localRotFK = mjcf.fk_local_rotations[j];
                int parentIdx = mjcf.fk_parent_indices[j];

                float[] worldTrans = QuatRotateVec(fkRot[parentIdx], localTrans);
                fkPos[j] = new float[]
                {
                    fkPos[parentIdx][0] + worldTrans[0],
                    fkPos[parentIdx][1] + worldTrans[1],
                    fkPos[parentIdx][2] + worldTrans[2]
                };
                float[] localCombined = QuatMul(localRotFK, jRot);
                fkRot[j] = QuatMul(fkRot[parentIdx], localCombined);
            }

            int[] keyIds = config.key_body_ids;
            foreach (int bid in keyIds)
            {
                float[] bp = fkPos[bid];
                float[] rel = { bp[0] - rootPos[0], bp[1] - rootPos[1], bp[2] - rootPos[2] };
                float[] lr2 = config.global_obs ? rel : QuatRotateVec(headingInv, rel);
                obs[idx++] = lr2[0]; obs[idx++] = lr2[1]; obs[idx++] = lr2[2];
            }
        }

        return obs;
    }

    float[] ReadDofPositions()
    {
        float[] dofPos = new float[mjcf.dofInfo.Count];
        for (int i = 0; i < mjcf.dofInfo.Count; i++)
        {
            var dof = mjcf.dofInfo[i];
            if (!bodyMap.ContainsKey(dof.child_body)) continue;
            var ab = bodyMap[dof.child_body];
            dofPos[i] = ab.jointPosition[dof.physx_axis];
        }
        return dofPos;
    }

    float[] ReadDofVelocities()
    {
        float[] dofVel = new float[mjcf.dofInfo.Count];
        for (int i = 0; i < mjcf.dofInfo.Count; i++)
        {
            var dof = mjcf.dofInfo[i];
            if (!bodyMap.ContainsKey(dof.child_body)) continue;
            var ab = bodyMap[dof.child_body];
            dofVel[i] = ab.jointVelocity[dof.physx_axis];
        }
        return dofVel;
    }

    // ===== Coordinate Conversion (Z-up <-> Y-up) =====

    static Vector3 ZupToYup(float[] v) => new Vector3(v[0], v[2], v[1]);
    static float[] YupToZup(Vector3 v) => new float[] { v.x, v.z, v.y };

    /// <summary>Z-up quaternion [x,y,z,w] (Unity Quaternion) -> Y-up Unity Quaternion</summary>
    static Quaternion ZupQuatToYup(Quaternion q)
    {
        // Swap Y and Z axes: apply a -90 deg rotation around X
        // For Z-up to Y-up: (x, y, z, w) -> (x, z, -y, w) then adjust
        // Proper conversion: swap y<->z, negate new z
        return new Quaternion(q.x, q.z, -q.y, q.w);
    }

    /// <summary>Y-up Unity Quaternion -> Z-up [x,y,z,w] array</summary>
    static float[] YupQuatToZup(Quaternion q)
    {
        // Inverse of ZupQuatToYup: (x, y, z, w) -> (x, -z, y, w)
        return new float[] { q.x, -q.z, q.y, q.w };
    }

    // ===== Quaternion / Vector Math (Z-up frame, matching web impl) =====

    static float[] QuatRotateVec(float[] q, float[] v)
    {
        float qx = q[0], qy = q[1], qz = q[2], qw = q[3];
        float vx = v[0], vy = v[1], vz = v[2];
        float tx = 2 * (qy * vz - qz * vy);
        float ty = 2 * (qz * vx - qx * vz);
        float tz = 2 * (qx * vy - qy * vx);
        return new float[]
        {
            vx + qw * tx + (qy * tz - qz * ty),
            vy + qw * ty + (qz * tx - qx * tz),
            vz + qw * tz + (qx * ty - qy * tx)
        };
    }

    static float[] QuatMul(float[] a, float[] b)
    {
        float ax = a[0], ay = a[1], az = a[2], aw = a[3];
        float bx = b[0], by = b[1], bz = b[2], bw = b[3];
        return new float[]
        {
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz
        };
    }

    static float[] QuatNorm(float[] q)
    {
        float len = Mathf.Sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        if (len < 1e-12f) return new float[] { 0, 0, 0, 1 };
        return new float[] { q[0] / len, q[1] / len, q[2] / len, q[3] / len };
    }

    static float[] AxisAngleToQuat(float[] axis, float angle)
    {
        float ha = angle * 0.5f;
        float s = Mathf.Sin(ha), c = Mathf.Cos(ha);
        float len = Mathf.Sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
        if (len < 1e-12f) return new float[] { 0, 0, 0, 1 };
        return QuatNorm(new float[] { axis[0] / len * s, axis[1] / len * s, axis[2] / len * s, c });
    }

    static float[] ExpMapToQuat(float ex, float ey, float ez)
    {
        float angle = Mathf.Sqrt(ex * ex + ey * ey + ez * ez);
        if (angle < 1e-5f) return new float[] { 0, 0, 0, 1 };
        return AxisAngleToQuat(new float[] { ex / angle, ey / angle, ez / angle }, angle);
    }

    static float CalcHeading(float[] q)
    {
        float[] rotDir = QuatRotateVec(q, new float[] { 1, 0, 0 });
        return Mathf.Atan2(rotDir[1], rotDir[0]);
    }

    static float[] QuatToTanNorm(float[] q)
    {
        float[] tan = QuatRotateVec(q, new float[] { 1, 0, 0 });
        float[] norm = QuatRotateVec(q, new float[] { 0, 0, 1 });
        return new float[] { tan[0], tan[1], tan[2], norm[0], norm[1], norm[2] };
    }

    // ===== Helpers =====

    static ArticulationDrive GetDrive(ArticulationBody ab, int axis)
    {
        switch (axis)
        {
            case 0: return ab.xDrive;
            case 1: return ab.yDrive;
            case 2: return ab.zDrive;
            default: return ab.xDrive;
        }
    }

    static void SetDrive(ArticulationBody ab, int axis, ArticulationDrive drive)
    {
        switch (axis)
        {
            case 0: ab.xDrive = drive; break;
            case 1: ab.yDrive = drive; break;
            case 2: ab.zDrive = drive; break;
        }
    }

    void AddCollider(GameObject go, MJCFGeom geom)
    {
        if (geom.type == "sphere")
        {
            var col = go.AddComponent<SphereCollider>();
            col.radius = geom.radius;
            col.center = ZupToYup(geom.pos);
        }
        else if (geom.type == "capsule" && geom.fromto != null)
        {
            var col = go.AddComponent<CapsuleCollider>();
            var ft = geom.fromto;
            Vector3 p0 = ZupToYup(new float[] { ft[0], ft[1], ft[2] });
            Vector3 p1 = ZupToYup(new float[] { ft[3], ft[4], ft[5] });
            Vector3 dir = p1 - p0;
            float len = dir.magnitude;
            col.radius = geom.radius;
            col.height = len + 2 * geom.radius;
            col.center = (p0 + p1) * 0.5f;
            // Determine capsule direction (0=X, 1=Y, 2=Z)
            float ax = Mathf.Abs(dir.x), ay = Mathf.Abs(dir.y), az = Mathf.Abs(dir.z);
            if (ay >= ax && ay >= az) col.direction = 1;
            else if (ax >= az) col.direction = 0;
            else col.direction = 2;
        }
        else if (geom.type == "box" && geom.halfExtents != null)
        {
            var col = go.AddComponent<BoxCollider>();
            col.size = new Vector3(geom.halfExtents[0] * 2, geom.halfExtents[2] * 2, geom.halfExtents[1] * 2);
            col.center = ZupToYup(geom.pos);
        }
        else if (geom.type == "cylinder")
        {
            // Approximate cylinder with capsule
            var col = go.AddComponent<CapsuleCollider>();
            col.radius = geom.radius;
            if (geom.fromto != null)
            {
                var ft = geom.fromto;
                Vector3 p0 = ZupToYup(new float[] { ft[0], ft[1], ft[2] });
                Vector3 p1 = ZupToYup(new float[] { ft[3], ft[4], ft[5] });
                Vector3 dv = p1 - p0;
                col.height = dv.magnitude + 2 * geom.radius;
                col.center = (p0 + p1) * 0.5f;
                float aax = Mathf.Abs(dv.x), aay = Mathf.Abs(dv.y), aaz = Mathf.Abs(dv.z);
                if (aay >= aax && aay >= aaz) col.direction = 1;
                else if (aax >= aaz) col.direction = 0;
                else col.direction = 2;
            }
            else
            {
                col.height = (geom.halfHeight * 2) + 2 * geom.radius;
                col.center = ZupToYup(geom.pos);
                col.direction = 1; // Y-up default
            }
        }
    }

    static void SetLayerRecursive(GameObject go, int layer)
    {
        go.layer = layer;
        foreach (Transform child in go.transform)
            SetLayerRecursive(child.gameObject, layer);
    }

    static float GaussianRandom()
    {
        float u1 = UnityEngine.Random.value;
        float u2 = UnityEngine.Random.value;
        while (u1 == 0) u1 = UnityEngine.Random.value;
        return Mathf.Sqrt(-2f * Mathf.Log(u1)) * Mathf.Cos(2f * Mathf.PI * u2);
    }

    static float[] ParseFloatArray(JToken token)
    {
        if (token == null) return null;
        return token.ToObject<float[]>();
    }

    static int[] ParseIntArray(JToken token)
    {
        if (token == null) return null;
        return token.ToObject<int[]>();
    }

    void LoadLatentPresets(JObject root)
    {
        Presets.Clear();
        PresetNames.Clear();

        // Hardcoded presets matching the web demo
        var presetData = new Dictionary<string, float[]>
        {
            ["Walk Forward"] = new float[] {-0.1452f,-0.3008f,0.2979f,-0.0948f,-0.0689f,-0.0289f,-0.1291f,-0.1262f,0.0073f,0.1827f,0.0792f,-0.1077f,0.1575f,0.0871f,-0.0939f,-0.0555f,-0.0178f,-0.0836f,-0.0614f,0.1349f,0.0806f,0.1352f,0.2347f,0.0255f,-0.0159f,0.0209f,0.0127f,-0.0154f,0.0275f,-0.1192f,-0.0503f,0.0199f,-0.0181f,0.0478f,-0.1603f,-0.1162f,-0.0469f,0.1446f,-0.0181f,-0.1132f,-0.0137f,0.0032f,0.2209f,0.006f,0.0243f,-0.1142f,0.0293f,0.0628f,-0.2274f,0.3274f,0.1841f,-0.0464f,-0.1146f,0.0573f,0.218f,0.024f,-0.2133f,0.1156f,0.0734f,-0.2137f,0.0967f,0.0419f,-0.0053f,-0.0451f},
            ["Stand Still"] = new float[] {0.0823f,0.0665f,0.0626f,0.0715f,-0.102f,-0.1834f,-0.2367f,-0.0636f,0.0142f,-0.0189f,0.2447f,0.1604f,-0.1471f,-0.1849f,0.0624f,-0.2257f,0.0198f,0.1578f,-0.0465f,0.1572f,-0.0124f,-0.2348f,0.0626f,0.0321f,0.1301f,-0.1334f,0.065f,-0.0826f,-0.1048f,-0.1132f,-0.098f,0.0812f,-0.1861f,-0.194f,0.0958f,-0.0116f,-0.1397f,-0.1597f,0.0273f,-0.0089f,-0.1377f,0.0304f,-0.0064f,0.0053f,0.1048f,0.0377f,-0.0781f,-0.149f,0.1906f,-0.263f,0.1719f,0.0553f,-0.12f,0.0875f,-0.039f,0.1607f,0.1923f,-0.1103f,0.023f,0.0306f,0.1093f,-0.2063f,-0.0024f,-0.0024f},
            ["Frenzy Attack"] = new float[] {0.0595f,0.1815f,-0.0247f,-0.1175f,0.078f,-0.2466f,-0.1239f,0.0443f,-0.077f,0.0418f,-0.1932f,-0.1295f,0.0082f,0.1895f,-0.0754f,0.3816f,0.0177f,0.1045f,0.1036f,0.1625f,-0.1621f,0.0862f,-0.1163f,-0.0197f,0.0552f,0.0826f,-0.0476f,0.1087f,-0.0694f,-0.2286f,-0.1396f,-0.1861f,0.0406f,0.0124f,-0.0301f,-0.1529f,0.1192f,0.1319f,-0.0435f,0.2294f,-0.0147f,-0.1469f,0.0053f,-0.1236f,0.02f,-0.1372f,-0.0791f,-0.0429f,-0.1183f,-0.1449f,0.1248f,-0.1061f,-0.0326f,0.0421f,0.0182f,-0.1089f,-0.1872f,-0.0533f,0.0121f,-0.1538f,-0.1053f,-0.1679f,0.1177f,0.1281f},
            ["Overhead Strike"] = new float[] {-0.0435f,0.0218f,-0.0879f,0.0008f,-0.0988f,-0.2055f,-0.2171f,0.0351f,0.1084f,-0.1148f,-0.1765f,-0.2086f,-0.0867f,-0.1654f,-0.2069f,0.1668f,-0.0254f,-0.0173f,0.1342f,0.1428f,0.1529f,0.0857f,0.0885f,0.1159f,0.0952f,-0.1534f,-0.2632f,-0.2138f,-0.0789f,-0.2705f,-0.0208f,-0.0089f,-0.1241f,-0.0723f,0.0792f,-0.0995f,0.0075f,0.2476f,-0.0161f,0.1016f,0.1169f,-0.0589f,-0.0114f,-0.1281f,-0.0169f,-0.0347f,0.1385f,-0.0887f,-0.0419f,-0.078f,-0.2256f,-0.0677f,-0.0981f,-0.0445f,-0.0427f,0.0467f,-0.2272f,0.0132f,-0.0887f,0.0037f,-0.1922f,-0.0025f,-0.0177f,0.1369f},
            ["Reliable Kick"] = new float[] {0.1687f,-0.0675f,0.0733f,0.1104f,0.1888f,-0.2833f,0.2764f,-0.0861f,-0.0017f,-0.0958f,-0.0473f,0.0215f,0.1835f,0.0089f,-0.0972f,0.0592f,0.1358f,0.1714f,-0.0865f,-0.1585f,0.0647f,0.298f,0.084f,-0.1818f,0.1705f,0.1095f,0.0928f,0.0421f,-0.0365f,0.1083f,0.1311f,0.0091f,-0.0703f,-0.0693f,-0.0918f,-0.0285f,0.0624f,0.1283f,-0.134f,0.0859f,-0.0116f,0.0606f,0.2157f,0.268f,-0.0739f,-0.0086f,0.1455f,0.0935f,0.0415f,0.029f,-0.0822f,0.0896f,0.0423f,-0.054f,-0.1751f,-0.0879f,0.0447f,0.0172f,0.1776f,0.1071f,0.2338f,-0.1036f,0.0489f,0.0024f},
            ["Jump Back"] = new float[] {-0.0278f,-0.1597f,0.0535f,-0.0853f,-0.063f,-0.0871f,-0.0606f,0.2214f,-0.1453f,0.0232f,0.2877f,0.1577f,0.2037f,0.1257f,0.0539f,-0.1776f,-0.0447f,0.0179f,-0.0674f,-0.0797f,0.0667f,-0.0078f,0.0265f,-0.177f,0.1485f,0.0266f,-0.1997f,-0.2082f,0.0736f,-0.1618f,-0.1899f,-0.2072f,0.0501f,0.1432f,0.0351f,0.1971f,-0.0809f,0.0912f,0.0234f,-0.0278f,-0.1118f,0.0764f,0.1357f,0.0962f,0.116f,0.2222f,-0.0974f,-0.0412f,0.1753f,0.1406f,-0.0213f,-0.0485f,0.055f,-0.0168f,0.166f,-0.2522f,0.0545f,-0.0637f,0.0657f,-0.0786f,0.1337f,-0.1115f,-0.0833f,0.1007f},
        };

        foreach (var kv in presetData)
        {
            Presets[kv.Key] = kv.Value;
            PresetNames.Add(kv.Key);
        }
    }
}

// ===== Data Structures =====

[Serializable]
public class MimicKitConfig
{
    public int obs_dim;
    public int act_dim;
    public int latent_dim;
    public float[] obs_mean;
    public float[] obs_std;
    public float[] init_dof_pos;
    public float[] init_root_pos;
    public float[] init_root_rot_quat;
    public float[] action_low;
    public float[] action_high;
    public int[] key_body_ids;
    public bool global_obs;
    public float pelvis_z;
    public float tpose_pelvis_z;
}

// ===== MJCF Parser =====

[Serializable]
public class MJCFBody
{
    public string name;
    public string parent;
    public float[] pos;      // world pos (Z-up)
    public float[] localPos; // parent-local pos (Z-up)
    public List<MJCFGeom> geoms = new List<MJCFGeom>();
    public float mass;
    public float[] inertia;  // [Ixx, Iyy, Izz]
    public float[] com;      // center of mass local (Z-up)
}

[Serializable]
public class MJCFGeom
{
    public string name;
    public string type;
    public float[] pos;
    public float radius;
    public float[] fromto;
    public float[] halfExtents;
    public float halfHeight;
    public float density = 1000f;
}

[Serializable]
public class MJCFJointAxis
{
    public string name;
    public float[] mjcf_axis;
    public float stiffness;
    public float damping;
    public float maxForce;
    public float[] range;
    public float armature;
}

[Serializable]
public class MJCFJoint
{
    public string name;
    public string parent_body;
    public string child_body;
    public List<MJCFJointAxis> axes = new List<MJCFJointAxis>();
    public int[] axisMap;
    public float[] localPos0;
    public float[] localRot; // [w,x,y,z]
    public string jointType; // "spherical" or "revolute"
}

[Serializable]
public class MJCFFixedJoint
{
    public string name;
    public string parent_body;
    public string child_body;
    public float[] localPos0;
}

[Serializable]
public class DofInfo
{
    public string joint_name;
    public string axis_name;
    public int physx_axis;
    public string child_body;
}

[Serializable]
public class KinematicJoint
{
    public string name;
    public string child_body;
    public string type; // "SPHERICAL", "HINGE", "FIXED"
    public int dof_idx;
    public int dof_dim;
    public float[] axis;
}

[Serializable]
public class MJCFData
{
    public List<MJCFBody> bodies = new List<MJCFBody>();
    public List<MJCFJoint> joints = new List<MJCFJoint>();
    public List<MJCFFixedJoint> fixedJoints = new List<MJCFFixedJoint>();
    public List<string> actuatorOrder = new List<string>();
    public List<DofInfo> dofInfo = new List<DofInfo>();
    public List<KinematicJoint> kinematicJoints = new List<KinematicJoint>();
    public int[] fk_parent_indices;
    public float[][] fk_local_translations;
    public float[][] fk_local_rotations;
    public int act_dim;
}

/// <summary>
/// Parses MuJoCo MJCF XML into the data structures needed for articulation building.
/// Direct port of the web demo's parseMJCF function.
/// </summary>
public static class MJCFParser
{
    public static MJCFData Parse(string xmlText)
    {
        var doc = new System.Xml.XmlDocument();
        doc.LoadXml(xmlText);
        var root = doc.DocumentElement;

        var data = new MJCFData();

        // Detect fixed bodies (bodies with no joints and no freejoint)
        var fixedBodies = new HashSet<string>();
        var worldbody = root.SelectSingleNode("worldbody");
        foreach (System.Xml.XmlNode topBody in worldbody.SelectNodes("body"))
            ScanFixedBodies(topBody, false, fixedBodies);

        // Parse actuators
        var actuatorMap = new Dictionary<string, (float gear, float maxForce)>();
        var actuatorNode = root.SelectSingleNode("actuator");
        if (actuatorNode != null)
        {
            foreach (System.Xml.XmlNode mot in actuatorNode.SelectNodes("motor"))
            {
                string jname = mot.Attributes?["joint"]?.Value;
                if (string.IsNullOrEmpty(jname)) continue;
                float gear = ParseFloat(mot, "gear", 1f);
                float[] frc = ParseVec(mot, "actuatorfrcrange", 2);
                float maxForce = frc != null ? Mathf.Max(Mathf.Abs(frc[0]), Mathf.Abs(frc[1])) : gear;
                actuatorMap[jname] = (gear, maxForce);
                data.actuatorOrder.Add(jname);
            }
        }

        // Process bodies
        var pelvis = worldbody.SelectSingleNode("body");
        float[] pelvisPos = ParseVec(pelvis, "pos", 3) ?? new float[] { 0, 0, 0 };
        ProcessBody(pelvis, null, pelvisPos, fixedBodies, actuatorMap, data);

        // Build dofInfo from actuator order
        foreach (string actName in data.actuatorOrder)
        {
            foreach (var jdata in data.joints)
            {
                for (int ai = 0; ai < jdata.axes.Count; ai++)
                {
                    if (jdata.axes[ai].name == actName)
                    {
                        data.dofInfo.Add(new DofInfo
                        {
                            joint_name = jdata.name,
                            axis_name = actName,
                            physx_axis = jdata.axisMap[ai],
                            child_body = jdata.child_body
                        });
                        break;
                    }
                }
            }
        }

        // Build FK data
        var bodyNames = data.bodies.Select(b => b.name).ToList();
        data.fk_parent_indices = data.bodies.Select(b =>
            b.parent == null ? -1 : bodyNames.IndexOf(b.parent)).ToArray();
        data.fk_local_translations = data.bodies.Select(b => b.localPos).ToArray();
        data.fk_local_rotations = data.bodies.Select(b => new float[] { 0, 0, 0, 1 }).ToArray();

        // Build kinematicJoints
        int dofIdx = 0;
        for (int bi = 1; bi < data.bodies.Count; bi++)
        {
            string bname = data.bodies[bi].name;
            var jdata2 = data.joints.Find(j => j.child_body == bname);
            var fjdata = data.fixedJoints.Find(j => j.child_body == bname);

            if (jdata2 != null)
            {
                int nDofs = jdata2.axes.Count;
                data.kinematicJoints.Add(new KinematicJoint
                {
                    name = jdata2.name,
                    child_body = bname,
                    type = nDofs > 1 ? "SPHERICAL" : "HINGE",
                    dof_idx = dofIdx,
                    dof_dim = nDofs,
                    axis = nDofs == 1 ? jdata2.axes[0].mjcf_axis : null
                });
                dofIdx += nDofs;
            }
            else
            {
                string n = fjdata != null ? fjdata.name : bname;
                data.kinematicJoints.Add(new KinematicJoint
                {
                    name = n, child_body = bname, type = "FIXED",
                    dof_idx = dofIdx, dof_dim = 0
                });
            }
        }

        // Force consecutive PhysX axis assignment for spherical joints
        foreach (var kj in data.kinematicJoints)
        {
            if (kj.type != "SPHERICAL") continue;
            for (int d = 0; d < 3; d++)
                data.dofInfo[kj.dof_idx + d].physx_axis = d;
        }
        foreach (var jdata3 in data.joints)
        {
            if (jdata3.jointType == "spherical" && jdata3.axisMap != null)
                jdata3.axisMap = new int[] { 0, 1, 2 };
        }

        data.act_dim = data.actuatorOrder.Count;
        return data;
    }

    static void ScanFixedBodies(System.Xml.XmlNode el, bool isRoot, HashSet<string> fixedBodies)
    {
        if (!isRoot)
        {
            var joints = el.SelectNodes("joint");
            bool hasHinge = false;
            bool hasFree = false;
            foreach (System.Xml.XmlNode j in joints)
            {
                string jtype = j.Attributes?["type"]?.Value ?? "hinge";
                if (jtype == "free") hasFree = true;
                else if (j.Name == "freejoint") hasFree = true;
                else hasHinge = true;
            }
            if (el.SelectSingleNode("freejoint") != null) hasFree = true;
            if (!hasHinge && !hasFree)
            {
                string name = el.Attributes?["name"]?.Value;
                if (name != null) fixedBodies.Add(name);
            }
        }
        foreach (System.Xml.XmlNode child in el.SelectNodes("body"))
            ScanFixedBodies(child, false, fixedBodies);
    }

    static void ProcessBody(System.Xml.XmlNode el, string parentName, float[] parentWorldPos,
        HashSet<string> fixedBodies, Dictionary<string, (float gear, float maxForce)> actuatorMap, MJCFData data)
    {
        string name = el.Attributes?["name"]?.Value;
        float[] localPos = ParseVec(el, "pos", 3) ?? new float[] { 0, 0, 0 };
        float[] worldPos = { parentWorldPos[0] + localPos[0], parentWorldPos[1] + localPos[1], parentWorldPos[2] + localPos[2] };

        // Parse geoms
        var geoms = new List<MJCFGeom>();
        foreach (System.Xml.XmlNode ge in el.SelectNodes("geom"))
        {
            var g = new MJCFGeom
            {
                name = ge.Attributes?["name"]?.Value ?? name,
                type = ge.Attributes?["type"]?.Value ?? "sphere",
                pos = ParseVec(ge, "pos", 3) ?? new float[] { 0, 0, 0 },
                density = ParseFloat(ge, "density", 1000f)
            };
            float[] size = ParseVec(ge, "size", 0);
            if (g.type == "sphere") g.radius = size != null ? size[0] : 0.05f;
            else if (g.type == "capsule")
            {
                g.radius = size != null ? size[0] : 0.05f;
                g.fromto = ParseVec(ge, "fromto", 6);
            }
            else if (g.type == "box")
            {
                g.halfExtents = size != null ? new float[] { size[0], size[1], size[2] } : new float[] { 0.05f, 0.05f, 0.05f };
            }
            else if (g.type == "cylinder")
            {
                g.radius = size != null ? size[0] : 0.05f;
                g.halfHeight = size != null && size.Length > 1 ? size[1] : 0.1f;
                g.fromto = ParseVec(ge, "fromto", 6);
                if (g.fromto != null)
                {
                    var ft = g.fromto;
                    g.halfHeight = Mathf.Sqrt(Sq(ft[3] - ft[0]) + Sq(ft[4] - ft[1]) + Sq(ft[5] - ft[2])) / 2f;
                }
            }
            geoms.Add(g);
        }

        // Compute mass, COM, inertia from geoms
        float totalMass = 0;
        float[] com = { 0, 0, 0 };
        foreach (var g in geoms)
        {
            float mass; float[] center;
            ComputeGeomMass(g, out mass, out center);
            com[0] += mass * center[0]; com[1] += mass * center[1]; com[2] += mass * center[2];
            totalMass += mass;
        }
        if (totalMass > 0) { com[0] /= totalMass; com[1] /= totalMass; com[2] /= totalMass; }

        // Simplified inertia (diagonal approximation)
        float[] inertia = { 0, 0, 0 };
        foreach (var g in geoms)
        {
            float mass; float[] center;
            ComputeGeomMass(g, out mass, out center);
            float[] Ii = ComputeGeomInertia(g, mass);
            float dx = center[0] - com[0], dy = center[1] - com[1], dz = center[2] - com[2];
            inertia[0] += Ii[0] + mass * (dy * dy + dz * dz);
            inertia[1] += Ii[1] + mass * (dx * dx + dz * dz);
            inertia[2] += Ii[2] + mass * (dx * dx + dy * dy);
        }

        data.bodies.Add(new MJCFBody
        {
            name = name, parent = parentName, pos = worldPos, localPos = localPos,
            geoms = geoms, mass = totalMass, inertia = inertia, com = com
        });

        // Parse joints
        var jointEls = new List<System.Xml.XmlNode>();
        foreach (System.Xml.XmlNode je in el.SelectNodes("joint"))
        {
            string jtype = je.Attributes?["type"]?.Value ?? "hinge";
            if (jtype != "free" && je.Name != "freejoint")
                jointEls.Add(je);
        }

        if (fixedBodies.Contains(name))
        {
            data.fixedJoints.Add(new MJCFFixedJoint
            {
                name = name + "_fixed", parent_body = parentName,
                child_body = name, localPos0 = localPos
            });
        }
        else if (jointEls.Count > 0 && parentName != null)
        {
            var axesData = new List<MJCFJointAxis>();
            var jointAxes = new List<float[]>();

            foreach (var je in jointEls)
            {
                float[] axis = ParseVec(je, "axis", 3) ?? new float[] { 1, 0, 0 };
                jointAxes.Add(axis);
                float[] rng = ParseVec(je, "range", 2) ?? new float[] { -3.14159f, 3.14159f };
                string jname = je.Attributes?["name"]?.Value;
                float gear = 100, maxForce = 100;
                if (jname != null && actuatorMap.ContainsKey(jname))
                {
                    gear = actuatorMap[jname].gear;
                    maxForce = actuatorMap[jname].maxForce;
                }
                axesData.Add(new MJCFJointAxis
                {
                    name = jname,
                    mjcf_axis = axis,
                    stiffness = ParseFloat(je, "stiffness", 0),
                    damping = ParseFloat(je, "damping", 0),
                    maxForce = maxForce,
                    range = rng,
                    armature = ParseFloat(je, "armature", 0)
                });
            }

            int[] axisMap;
            float[] q;
            ComputeJointFrame(jointAxes, out q, out axisMap);
            float[] localRot = { q[3], q[0], q[1], q[2] }; // [w,x,y,z]

            string jointName = jointEls.Count > 1
                ? System.Text.RegularExpressions.Regex.Replace(jointEls[0].Attributes["name"].Value, "_[^_]+$", "")
                : jointEls[0].Attributes["name"].Value;

            data.joints.Add(new MJCFJoint
            {
                name = jointName, parent_body = parentName, child_body = name,
                axes = axesData,
                axisMap = axisMap.Take(jointEls.Count).ToArray(),
                localPos0 = localPos,
                localRot = localRot,
                jointType = jointEls.Count > 1 ? "spherical" : "revolute"
            });
        }

        foreach (System.Xml.XmlNode child in el.SelectNodes("body"))
            ProcessBody(child, name, worldPos, fixedBodies, actuatorMap, data);
    }

    // ===== Joint Frame Computation (port from web) =====

    static void ComputeJointFrame(List<float[]> jointAxes, out float[] q, out int[] axisMap)
    {
        axisMap = new int[] { 0, 1, 2 };
        int n = jointAxes.Count;

        if (n == 0) { q = new float[] { 0, 0, 0, 1 }; return; }
        if (n == 1) { q = GetRotationQuat(new float[] { 1, 0, 0 }, jointAxes[0]); return; }

        float[] Q = GetRotationQuat(jointAxes[0], new float[] { 1, 0, 0 });
        float[] b = Normalize3(QuatRotate3(Q, jointAxes[1]));

        if (n == 2)
        {
            if (Mathf.Abs(Dot3(b, new float[] { 0, 1, 0 })) > Mathf.Abs(Dot3(b, new float[] { 0, 0, 1 })))
            {
                axisMap[1] = 1;
                float[] c = Normalize3(Cross3(jointAxes[0], jointAxes[1]));
                q = Mat33ToQuat(Normalize3(jointAxes[0]), Normalize3(jointAxes[1]), c);
            }
            else
            {
                axisMap[1] = 2; axisMap[2] = 1;
                float[] c = Normalize3(Cross3(jointAxes[1], jointAxes[0]));
                q = Mat33ToQuat(Normalize3(jointAxes[0]), c, Normalize3(jointAxes[1]));
            }
            return;
        }

        // n == 3
        if (Mathf.Abs(Dot3(b, new float[] { 0, 1, 0 })) > Mathf.Abs(Dot3(b, new float[] { 0, 0, 1 })))
        {
            axisMap[1] = 1; axisMap[2] = 2;
            q = Mat33ToQuat(Normalize3(jointAxes[0]), Normalize3(jointAxes[1]), Normalize3(jointAxes[2]));
        }
        else
        {
            axisMap[1] = 2; axisMap[2] = 1;
            q = Mat33ToQuat(Normalize3(jointAxes[0]), Normalize3(jointAxes[2]), Normalize3(jointAxes[1]));
        }
    }

    static float[] GetRotationQuat(float[] from, float[] to)
    {
        float[] u = Normalize3(from), v = Normalize3(to);
        float d = Dot3(u, v);
        if (d > 1 - 1e-6f) return new float[] { 0, 0, 0, 1 };
        if (d < 1e-6f - 1)
        {
            float[] axis = Cross3(new float[] { 1, 0, 0 }, u);
            if (Dot3(axis, axis) < 1e-6f) axis = Cross3(new float[] { 0, 1, 0 }, u);
            axis = Normalize3(axis);
            return new float[] { axis[0], axis[1], axis[2], 0 };
        }
        float[] c = Cross3(u, v);
        float s = Mathf.Sqrt((1 + d) * 2);
        float inv = 1f / s;
        float[] qq = { c[0] * inv, c[1] * inv, c[2] * inv, 0.5f * s };
        float qn = Mathf.Sqrt(qq[0] * qq[0] + qq[1] * qq[1] + qq[2] * qq[2] + qq[3] * qq[3]);
        return new float[] { qq[0] / qn, qq[1] / qn, qq[2] / qn, qq[3] / qn };
    }

    static float[] Mat33ToQuat(float[] col0, float[] col1, float[] col2)
    {
        float m00 = col0[0], m10 = col0[1], m20 = col0[2];
        float m01 = col1[0], m11 = col1[1], m21 = col1[2];
        float m02 = col2[0], m12 = col2[1], m22 = col2[2];
        float tr = m00 + m11 + m22;
        float x, y, z, w;
        if (tr >= 0)
        {
            float h = Mathf.Sqrt(tr + 1); w = 0.5f * h; float f = 0.5f / h;
            x = (m21 - m12) * f; y = (m02 - m20) * f; z = (m10 - m01) * f;
        }
        else
        {
            int i = 0;
            if (m11 > m00) i = 1;
            float[] diag = { m00, m11, m22 };
            if (m22 > diag[i]) i = 2;
            if (i == 0) { float h = Mathf.Sqrt(m00 - m11 - m22 + 1); x = 0.5f * h; float f = 0.5f / h; y = (m01 + m10) * f; z = (m20 + m02) * f; w = (m21 - m12) * f; }
            else if (i == 1) { float h = Mathf.Sqrt(m11 - m22 - m00 + 1); y = 0.5f * h; float f = 0.5f / h; z = (m12 + m21) * f; x = (m01 + m10) * f; w = (m02 - m20) * f; }
            else { float h = Mathf.Sqrt(m22 - m00 - m11 + 1); z = 0.5f * h; float f = 0.5f / h; x = (m20 + m02) * f; y = (m12 + m21) * f; w = (m10 - m01) * f; }
        }
        float n = Mathf.Sqrt(x * x + y * y + z * z + w * w);
        return new float[] { x / n, y / n, z / n, w / n };
    }

    // ===== Vector helpers =====
    static float Dot3(float[] a, float[] b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    static float[] Cross3(float[] a, float[] b) => new float[] { a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0] };
    static float[] Normalize3(float[] v)
    {
        float n = Mathf.Sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        return n < 1e-12f ? v : new float[] { v[0] / n, v[1] / n, v[2] / n };
    }
    static float[] QuatRotate3(float[] q, float[] v)
    {
        float qx = q[0], qy = q[1], qz = q[2], qw = q[3];
        float tx = 2 * (qy * v[2] - qz * v[1]);
        float ty = 2 * (qz * v[0] - qx * v[2]);
        float tz = 2 * (qx * v[1] - qy * v[0]);
        return new float[] { v[0] + qw * tx + (qy * tz - qz * ty), v[1] + qw * ty + (qz * tx - qx * tz), v[2] + qw * tz + (qx * ty - qy * tx) };
    }
    static float Sq(float x) => x * x;

    static float[] ParseVec(System.Xml.XmlNode node, string attr, int n)
    {
        string val = node.Attributes?[attr]?.Value;
        if (string.IsNullOrEmpty(val)) return null;
        var parts = val.Trim().Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
        float[] result = new float[parts.Length];
        for (int i = 0; i < parts.Length; i++) result[i] = float.Parse(parts[i], System.Globalization.CultureInfo.InvariantCulture);
        return n > 0 && result.Length >= n ? result.Take(n).ToArray() : result;
    }

    static float ParseFloat(System.Xml.XmlNode node, string attr, float def)
    {
        string val = node.Attributes?[attr]?.Value;
        if (string.IsNullOrEmpty(val)) return def;
        return float.Parse(val, System.Globalization.CultureInfo.InvariantCulture);
    }

    static void ComputeGeomMass(MJCFGeom g, out float mass, out float[] center)
    {
        center = g.pos ?? new float[] { 0, 0, 0 };
        float vol = 0;
        if (g.type == "sphere") vol = (4f / 3f) * Mathf.PI * g.radius * g.radius * g.radius;
        else if (g.type == "capsule" && g.fromto != null)
        {
            var ft = g.fromto;
            float halfH = Mathf.Sqrt(Sq(ft[3] - ft[0]) + Sq(ft[4] - ft[1]) + Sq(ft[5] - ft[2])) / 2f;
            vol = Mathf.PI * g.radius * g.radius * (2 * halfH) + (4f / 3f) * Mathf.PI * g.radius * g.radius * g.radius;
            center = new float[] { (ft[0] + ft[3]) / 2, (ft[1] + ft[4]) / 2, (ft[2] + ft[5]) / 2 };
        }
        else if (g.type == "box" && g.halfExtents != null)
        {
            vol = 8 * g.halfExtents[0] * g.halfExtents[1] * g.halfExtents[2];
        }
        else if (g.type == "cylinder")
        {
            float hh = g.halfHeight > 0 ? g.halfHeight : 0.1f;
            vol = Mathf.PI * g.radius * g.radius * (2 * hh);
            if (g.fromto != null)
            {
                var ft = g.fromto;
                center = new float[] { (ft[0] + ft[3]) / 2, (ft[1] + ft[4]) / 2, (ft[2] + ft[5]) / 2 };
            }
        }
        mass = vol * g.density;
    }

    static float[] ComputeGeomInertia(MJCFGeom g, float mass)
    {
        if (g.type == "sphere")
        {
            float I = (2f / 5f) * mass * g.radius * g.radius;
            return new float[] { I, I, I };
        }
        if (g.type == "box" && g.halfExtents != null)
        {
            float a = 2 * g.halfExtents[0], b = 2 * g.halfExtents[1], c = 2 * g.halfExtents[2];
            return new float[] { mass * (b * b + c * c) / 12, mass * (a * a + c * c) / 12, mass * (a * a + b * b) / 12 };
        }
        // Default for capsule/cylinder: approximate
        float r = g.radius;
        float I2 = mass * r * r * 0.4f;
        return new float[] { I2, I2, I2 };
    }
}
