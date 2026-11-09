"""Joint rig-motion estimation from two RGB-D camera tracks, without loop fitting."""
from __future__ import annotations

import cv2
import numpy as np

from sparse_rgbd_odometry import (DEFAULT_SETTINGS, TrackerSettings, angle_deg,
                                  make_reference, register)


def skew(p):
    p = np.asarray(p)
    result = np.zeros(p.shape[:-1] + (3, 3))
    result[..., 0, 1], result[..., 0, 2] = -p[..., 2], p[..., 1]
    result[..., 1, 0], result[..., 1, 2] = p[..., 2], -p[..., 0]
    result[..., 2, 0], result[..., 2, 1] = -p[..., 1], p[..., 0]
    return result


def se3_exp(delta):
    w, v = np.asarray(delta[:3]), np.asarray(delta[3:])
    theta = np.linalg.norm(w)
    cross = skew(w)
    if theta < 1e-7:
        jacobian = np.eye(3) + .5 * cross + cross @ cross / 6
    else:
        jacobian = (np.eye(3) + (1-np.cos(theta))/theta**2 * cross
                    + (theta-np.sin(theta))/theta**3 * cross @ cross)
    matrix = np.eye(4)
    matrix[:3, :3] = cv2.Rodrigues(w)[0]
    matrix[:3, 3] = jacobian @ v
    return matrix


def observation_residual(transform, observation, with_jacobian=True):
    e, points, pixels, intr = (observation[k] for k in ("extrinsic", "points", "pixels", "intrinsics"))
    body = points @ e[:3, :3].T + e[:3, 3]
    moved = body @ transform[:3, :3].T + transform[:3, 3]
    camera = (moved - e[:3, 3]) @ e[:3, :3]
    x, y, z = camera.T
    if np.any(z <= .05):
        raise ValueError("Joint prediction is behind a camera")
    predicted = np.column_stack((intr.fx*x/z+intr.ppx, intr.fy*y/z+intr.ppy))
    residual = predicted - pixels
    if not with_jacobian:
        return residual
    projection = np.zeros((len(points), 2, 3))
    projection[:, 0, 0], projection[:, 0, 2] = intr.fx/z, -intr.fx*x/(z*z)
    projection[:, 1, 1], projection[:, 1, 2] = intr.fy/z, -intr.fy*y/(z*z)
    geometric = np.concatenate((-skew(moved), np.broadcast_to(np.eye(3), (len(points), 3, 3))), axis=2)
    jacobian = np.einsum("nij,jk,nkl->nil", projection, e[:3, :3].T, geometric)
    return residual, jacobian


def joint_cost(transform, observations, huber=1.):
    total = 0.
    for obs in observations.values():
        lengths = np.linalg.norm(observation_residual(transform, obs, False), axis=1)
        total += float(np.mean(np.where(lengths <= huber, .5*lengths**2, huber*(lengths-.5*huber))))
    return total


def fit_joint_motion(observations, candidates, iterations=10):
    if not observations or not candidates:
        raise ValueError("No rigid-scene observations")
    transform = min(candidates, key=lambda m: joint_cost(m, observations)).copy()
    damping = 1e-4
    for _ in range(iterations):
        rows, values, weights = [], [], []
        for obs in observations.values():
            residual, jacobian = observation_residual(transform, obs)
            lengths = np.linalg.norm(residual, axis=1)
            weight = np.minimum(1., 1. / np.maximum(lengths, 1e-9)) / len(lengths)
            rows.append(jacobian.reshape(-1, 6))
            values.append(residual.ravel())
            weights.append(np.repeat(weight, 2))
        j, residual, weight = np.vstack(rows), np.concatenate(values), np.concatenate(weights)
        hessian = j.T @ (j * weight[:, None])
        if np.linalg.cond(hessian) > 1e13:
            raise ValueError("Joint motion is poorly constrained")
        gradient = j.T @ (weight * residual)
        old_cost = joint_cost(transform, observations)
        updated = False
        for _ in range(6):
            delta = np.linalg.solve(hessian + damping * np.diag(np.maximum(np.diag(hessian), 1e-6)), -gradient)
            proposal = se3_exp(delta) @ transform
            if joint_cost(proposal, observations) <= old_cost + 1e-12:
                transform, updated = proposal, True
                damping = max(1e-7, damping*.3)
                break
            damping *= 10
        if not updated or np.linalg.norm(delta) < 1e-6:
            break
    quality = {}
    for camera, observation in observations.items():
        error = np.linalg.norm(observation_residual(transform, observation, False), axis=1)
        quality[camera] = {"joint_reprojection_rmse_px": float(np.sqrt(np.mean(error**2))),
                           "joint_inlier_fraction": float(np.mean(error < 2.)), "points": len(error)}
        if quality[camera]["joint_reprojection_rmse_px"] > 1.5 or quality[camera]["joint_inlier_fraction"] < .65:
            raise ValueError(f"Camera {camera} is inconsistent with the shared rig motion")
    return transform, quality


class DualOdometry:
    def __init__(self, extrinsics, cameras=("a", "b"), keyframe_distance=.05,
                 keyframe_angle=5., settings=DEFAULT_SETTINGS, camera_strategy='all'):
        if camera_strategy not in ('all', 'alternate'):
            raise ValueError("camera_strategy must be 'all' or 'alternate'")
        self.extrinsics = {c: np.asarray(extrinsics[c], float) for c in cameras}
        self.cameras = cameras
        self.settings = settings
        self.camera_strategy = camera_strategy
        self.keyframe_distance, self.keyframe_angle = keyframe_distance, keyframe_angle
        self.references = {}
        self.reference_pose = self.last_pose = np.eye(4)
        self.last_frames = None
        self.keyframes = 0
        self.next_camera_index = 0

    def references_for(self, frames, pose):
        references = {}
        for camera in self.cameras:
            try:
                references[camera] = make_reference(frames[camera], pose, self.settings)
            except ValueError:
                pass
        return references

    def estimate(self, references, reference_pose, frames):
        prediction = np.linalg.inv(self.last_pose) @ reference_pose
        observations, candidates, qualities, errors = {}, [], {}, {}
        available = [camera for camera in self.cameras if camera in references]
        if self.camera_strategy == 'alternate' and available:
            primary = self.cameras[self.next_camera_index % len(self.cameras)]
            self.next_camera_index += 1
            order = ([primary] if primary in references else []) + [
                camera for camera in available if camera != primary]
        else:
            order = available
        for camera in order:
            reference = references[camera]
            e = self.extrinsics[camera]
            try:
                transform, quality, tracks = register(reference, frames[camera],
                    np.linalg.inv(e) @ prediction @ e, return_tracks=True,
                    settings=self.settings)
                observations[camera] = dict(tracks, extrinsic=e, intrinsics=frames[camera].intrinsics)
                candidates.append(e @ transform @ np.linalg.inv(e))
                qualities[camera] = quality
                if self.camera_strategy == 'alternate':
                    break
            except ValueError as error:
                errors[camera] = str(error)
        if not candidates:
            raise ValueError("Both cameras failed: " + str(errors))
        if len(candidates) == 2:
            disagreement = np.linalg.inv(candidates[0]) @ candidates[1]
            if np.linalg.norm(disagreement[:3, 3]) > .08 or angle_deg(disagreement[:3, :3]) > 4.:
                raise ValueError("Camera motions disagree: check time alignment, geometry or moving objects")
        if len(candidates) == 1:
            # The per-camera PnP estimate is already refined and checked against
            # current depth. Avoid duplicating that nonlinear solve in fast mode.
            transform = candidates[0]
        else:
            transform, joint = fit_joint_motion(observations, candidates)
            for camera in qualities:
                qualities[camera].update(joint[camera])
        return transform, {"cameras_used": list(observations), "cameras": qualities, "camera_errors": errors}

    def track(self, frames):
        if not self.references:
            references = self.references_for(frames, np.eye(4))
            if not references:
                return None, {"status": "rejected", "reason": "No textured depth reference"}
            self.references = references
            self.last_frames = frames
            self.keyframes += 1
            return np.eye(4), {"status": "origin", "cameras_used": list(references), "keyframes": self.keyframes}
        references, reference_pose = self.references, self.reference_pose
        fallback = False
        try:
            transform, quality = self.estimate(references, reference_pose, frames)
        except ValueError as primary:
            try:
                references = self.references_for(self.last_frames, self.last_pose)
                reference_pose = self.last_pose
                transform, quality = self.estimate(references, reference_pose, frames)
                fallback = True
            except ValueError as error:
                return None, {"status": "rejected", "reason": str(error), "primary_reason": str(primary)}
        pose = reference_pose @ np.linalg.inv(transform)
        step = np.linalg.inv(self.last_pose) @ pose
        translation, rotation = np.linalg.norm(step[:3, 3]), angle_deg(step[:3, :3])
        if translation > .25 or rotation > 25:
            return None, {"status": "rejected", "reason": "Implausibly large motion from last accepted frame"}
        if (fallback or np.linalg.norm(transform[:3, 3]) > self.keyframe_distance
                or angle_deg(transform[:3, :3]) > self.keyframe_angle):
            updated = self.references_for(frames, pose)
            if updated:
                self.references, self.reference_pose = updated, pose.copy()
                self.keyframes += 1
        quality.update(status="accepted", fallback_to_last_frame=fallback,
                       camera_strategy=self.camera_strategy, keyframes=self.keyframes,
                       step_translation_m=float(translation), step_rotation_deg=rotation)
        self.last_pose, self.last_frames = pose, frames
        return pose, quality
