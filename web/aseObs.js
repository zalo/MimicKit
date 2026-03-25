/**
 * Compute the 253-dim ASE observation vector from PhysX body state.
 *
 * Layout (17 bodies):
 *   [0:1]     root_height
 *   [1:49]    local_body_pos (16 non-root bodies × 3, heading-local, relative to root)
 *   [49:151]  local_body_rot (17 bodies × 6 tan_norm, heading-local)
 *   [151:202] local_body_vel (17 bodies × 3, heading-local)
 *   [202:253] local_body_ang_vel (17 bodies × 3, heading-local)
 *
 * All vectors rotated into heading-local frame using calc_heading_quat_inv(root_rot).
 */

function buildASEObs(links) {
    const obs = new Float32Array(253);
    const numBodies = links.length; // 17

    // Read all body state from PhysX
    const bodyPos = [], bodyRot = [], bodyVel = [], bodyAngVel = [];
    for (let i = 0; i < numBodies; i++) {
        const link = links[i].link;
        const pose = link.getGlobalPose();
        const p = pose.get_p();
        const q = pose.get_q();
        const v = link.getLinearVelocity();
        const av = link.getAngularVelocity();
        bodyPos.push([p.get_x(), p.get_y(), p.get_z()]);
        bodyRot.push([q.get_x(), q.get_y(), q.get_z(), q.get_w()]); // xyzw
        bodyVel.push([v.get_x(), v.get_y(), v.get_z()]);
        bodyAngVel.push([av.get_x(), av.get_y(), av.get_z()]);
    }

    const rootPos = bodyPos[0];
    const rootRot = bodyRot[0];

    // Heading inverse quaternion
    const headingInv = calcHeadingQuatInv(rootRot);

    let idx = 0;

    // [0] Root height
    obs[idx++] = rootPos[2];

    // [1:49] Local body positions (non-root, heading-local, relative to root)
    for (let i = 1; i < numBodies; i++) {
        const rel = [bodyPos[i][0] - rootPos[0], bodyPos[i][1] - rootPos[1], bodyPos[i][2] - rootPos[2]];
        const local = quatRotateVec(headingInv, rel);
        obs[idx++] = local[0];
        obs[idx++] = local[1];
        obs[idx++] = local[2];
    }

    // [49:151] Local body rotations (all bodies, heading-local, tan_norm)
    // For root: just tan_norm of root_rot (NOT heading-local, per local_root_obs=True)
    const rootTN = quatToTanNorm(rootRot);
    for (let k = 0; k < 6; k++) obs[idx++] = rootTN[k];

    // Non-root bodies: heading_rot * body_rot → tan_norm
    for (let i = 1; i < numBodies; i++) {
        const localRot = quatMul(headingInv, bodyRot[i]);
        const tn = quatToTanNorm(localRot);
        for (let k = 0; k < 6; k++) obs[idx++] = tn[k];
    }

    // [151:202] Local body velocities (all bodies, heading-local)
    for (let i = 0; i < numBodies; i++) {
        const lv = quatRotateVec(headingInv, bodyVel[i]);
        obs[idx++] = lv[0];
        obs[idx++] = lv[1];
        obs[idx++] = lv[2];
    }

    // [202:253] Local body angular velocities (all bodies, heading-local)
    for (let i = 0; i < numBodies; i++) {
        const lav = quatRotateVec(headingInv, bodyAngVel[i]);
        obs[idx++] = lav[0];
        obs[idx++] = lav[1];
        obs[idx++] = lav[2];
    }

    return obs;
}

// Note: quatMul, quatRotateVec, quatToTanNorm, calcHeadingQuatInv
// are defined in index.html's math utilities section.
