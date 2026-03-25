/**
 * ASE task observation functions for HLC controllers.
 * Each function appends task-specific obs to the base 253-dim ASE obs.
 *
 * Heading (5 dims): local_tar_dir(2) + tar_speed(1) + local_face_dir(2)
 * Location (2 dims): local_tar_pos(2) — heading-local XY offset to target
 * Reach (3 dims): local_tar_pos(3) — heading-local XYZ offset to target
 * Strike (15 dims): local_tar_pos(3) + local_tar_rot(6) + local_tar_vel(3) + local_tar_angvel(3)
 */

function computeHeadingTaskObs(rootRot, tarDirXY, tarSpeed, faceDirXY) {
    // tarDirXY: [x, y] world-frame target direction (unit vector)
    // faceDirXY: [x, y] world-frame target facing direction (unit vector)
    const headingInv = calcHeadingQuatInv(rootRot);

    const tarDir3d = [tarDirXY[0], tarDirXY[1], 0];
    const localTarDir = quatRotateVec(headingInv, tarDir3d);

    const faceDir3d = [faceDirXY[0], faceDirXY[1], 0];
    const localFaceDir = quatRotateVec(headingInv, faceDir3d);

    return new Float32Array([localTarDir[0], localTarDir[1], tarSpeed, localFaceDir[0], localFaceDir[1]]);
}

function computeLocationTaskObs(rootPos, rootRot, tarPosXY) {
    // tarPosXY: [x, y] world-frame target position
    const headingInv = calcHeadingQuatInv(rootRot);
    const tarPos3d = [tarPosXY[0] - rootPos[0], tarPosXY[1] - rootPos[1], 0];
    const local = quatRotateVec(headingInv, tarPos3d);
    return new Float32Array([local[0], local[1]]);
}

function computeReachTaskObs(rootPos, rootRot, tarPosXYZ) {
    const headingInv = calcHeadingQuatInv(rootRot);
    const rel = [tarPosXYZ[0] - rootPos[0], tarPosXYZ[1] - rootPos[1], tarPosXYZ[2] - rootPos[2]];
    const local = quatRotateVec(headingInv, rel);
    return new Float32Array([local[0], local[1], local[2]]);
}

function computeStrikeTaskObs(rootPos, rootRot, tarPos, tarRot, tarVel, tarAngVel) {
    const headingInv = calcHeadingQuatInv(rootRot);

    // Local target position (relative to root, heading-local, but keep world Z for target)
    const relPos = [tarPos[0] - rootPos[0], tarPos[1] - rootPos[1], tarPos[2]];
    const localPos = quatRotateVec(headingInv, relPos);

    // Local target rotation (heading-local tan_norm)
    const localRot = quatMul(headingInv, tarRot);
    const rotObs = quatToTanNorm(localRot);

    // Local target velocities
    const localVel = quatRotateVec(headingInv, tarVel);
    const localAngVel = quatRotateVec(headingInv, tarAngVel);

    return new Float32Array([
        localPos[0], localPos[1], localPos[2],
        rotObs[0], rotObs[1], rotObs[2], rotObs[3], rotObs[4], rotObs[5],
        localVel[0], localVel[1], localVel[2],
        localAngVel[0], localAngVel[1], localAngVel[2],
    ]);
}
