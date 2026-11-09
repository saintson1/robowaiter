"""Metric keyframe RGB-D odometry: LK tracking, forward/backward check, PnP RANSAC.

No known-distance scaling or loop correction is performed. A pose after a failed
frame is valid only if it is registered directly to an already positioned frame.
"""
from dataclasses import dataclass

import cv2
import numpy as np


def angle_deg(rotation):
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))))


@dataclass
class Reference:
    frame: object
    gray: np.ndarray
    pixels: np.ndarray
    points: np.ndarray
    pose: np.ndarray


@dataclass(frozen=True)
class TrackerSettings:
    max_corners: int = 1600
    min_tracks: int = 40
    lk_window: int = 21
    lk_max_level: int = 3
    lk_iterations: int = 30
    pnp_iterations: int = 200


DEFAULT_SETTINGS = TrackerSettings()


def grayscale(image):
    """Return a uint8 tracking image from RGB or rectified monochrome IR."""
    image = np.asarray(image)
    if image.ndim == 2 and image.dtype == np.uint8:
        return image
    if image.ndim == 3 and image.shape[2] == 3 and image.dtype == np.uint8:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    raise ValueError(f"Expected uint8 RGB or mono tracking image, got {image.shape} {image.dtype}")


def make_reference(frame, pose, settings=DEFAULT_SETTINGS):
    gray = grayscale(frame.rgb)
    z = frame.depth.astype(np.float32) * frame.depth_scale_m
    z = cv2.medianBlur(z, 3)
    mask = np.uint8((z > .25) & (z < 5.)) * 255
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
    pixels = cv2.goodFeaturesToTrack(gray, maxCorners=settings.max_corners, qualityLevel=.01,
                                      minDistance=4, mask=mask, blockSize=5)
    if pixels is None or len(pixels) < settings.min_tracks:
        raise ValueError("Too few textured points with valid depth")
    uv = pixels.reshape(-1, 2)
    ij = np.rint(uv).astype(int)
    depths = z[ij[:, 1], ij[:, 0]]
    intr = frame.intrinsics
    points = np.column_stack(((uv[:, 0] - intr.ppx) * depths / intr.fx,
                              (uv[:, 1] - intr.ppy) * depths / intr.fy, depths))
    return Reference(frame, gray, pixels, points.astype(np.float64), pose.copy())


def register(reference, current, initial_transform=None, *, return_tracks=False,
             settings=DEFAULT_SETTINGS):
    gray = grayscale(current.rgb)
    intr = current.intrinsics
    k = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1.]])
    options = dict(winSize=(settings.lk_window, settings.lk_window),
                   maxLevel=settings.lk_max_level,
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                             settings.lk_iterations, .01))
    predicted = None
    flags = 0
    if initial_transform is not None:
        r = cv2.Rodrigues(initial_transform[:3, :3])[0]
        predicted = cv2.projectPoints(reference.points, r, initial_transform[:3, 3], k, None)[0].astype(np.float32)
        flags = cv2.OPTFLOW_USE_INITIAL_FLOW
    target, good_forward, _ = cv2.calcOpticalFlowPyrLK(reference.gray, gray, reference.pixels,
                                                    predicted, flags=flags, **options)
    if target is None:
        raise ValueError("Optical flow failed")
    backward, good_backward, _ = cv2.calcOpticalFlowPyrLK(gray, reference.gray, target, None, **options)
    if backward is None:
        raise ValueError("Backward optical flow failed")
    uv = target.reshape(-1, 2)
    valid = ((good_forward.ravel() != 0) & (good_backward.ravel() != 0)
             & (np.linalg.norm(backward.reshape(-1, 2) - reference.pixels.reshape(-1, 2), axis=1) < .75)
             & (uv[:, 0] >= 2) & (uv[:, 0] < gray.shape[1] - 2)
             & (uv[:, 1] >= 2) & (uv[:, 1] < gray.shape[0] - 2))
    objects, images = reference.points[valid], uv[valid].astype(np.float64)
    if len(objects) < settings.min_tracks:
        raise ValueError(f"Too few consistent optical tracks: {len(objects)}")
    success, r, t, inliers = cv2.solvePnPRansac(objects, images, k, None,
                  iterationsCount=settings.pnp_iterations,
                  reprojectionError=1.5, confidence=.999, flags=cv2.SOLVEPNP_EPNP)
    if (not success or inliers is None or len(inliers) < settings.min_tracks
            or len(inliers) / len(objects) < .5):
        raise ValueError("Too few rigid-scene PnP inliers")
    indices = inliers.ravel()
    r, t = cv2.solvePnPRefineLM(objects[indices], images[indices], k, None, r, t)
    # RANSAC's pixel threshold admits borderline tracks. Trim those before the
    # final least-squares fit so depth edges and LK interpolation do not bias it.
    for _ in range(2):
        error = np.linalg.norm(cv2.projectPoints(objects[indices], r, t, k, None)[0].reshape(-1, 2)
                               - images[indices], axis=1)
        median = float(np.median(error))
        cutoff = max(.35, median + 2.5 * float(np.median(np.abs(error - median))))
        keep = error <= cutoff
        if keep.sum() < settings.min_tracks or keep.all():
            break
        indices = indices[keep]
        r, t = cv2.solvePnPRefineLM(objects[indices], images[indices], k, None, r, t)
    rotation = cv2.Rodrigues(r)[0]
    moved = objects[indices] @ rotation.T + t.ravel()
    projected = cv2.projectPoints(objects[indices], r, t, k, None)[0].reshape(-1, 2)
    reprojection = np.linalg.norm(projected - images[indices], axis=1)
    if float(np.sqrt(np.mean(reprojection ** 2))) > 1.2 or (moved[:, 2] <= 0).any():
        raise ValueError("Invalid PnP reprojection or cheirality")
    pixel = np.rint(images[indices]).astype(int)
    observed_z = current.depth[pixel[:, 1], pixel[:, 0]] * current.depth_scale_m
    depth_valid = (observed_z > .25) & (observed_z < 5.)
    depth_error = np.abs(moved[depth_valid, 2] - observed_z[depth_valid])
    if len(depth_error) < 30 or np.mean(depth_error < .08) < .65:
        raise ValueError("PnP motion is inconsistent with current depth")
    spread = np.ptp(images[indices], axis=0) / [gray.shape[1], gray.shape[0]]
    if np.min(spread) < .2:
        raise ValueError("Tracks are concentrated in too small an image region")
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, t.ravel()
    quality = {"reference_index": reference.frame.index, "reference_points": len(reference.points),
               "consistent_tracks": len(objects), "inliers": len(indices),
               "inlier_fraction": len(indices) / len(objects),
               "reprojection_rmse_px": float(np.sqrt(np.mean(reprojection ** 2))),
               "depth_median_abs_error_m": float(np.median(depth_error)),
               "image_span_fraction": spread.tolist()}
    if return_tracks:
        return transform, quality, {"points": objects[indices], "pixels": images[indices]}
    return transform, quality


class KeyframeOdometry:
    def __init__(self, settings=DEFAULT_SETTINGS):
        self.settings = settings
        self.reference = self.last = None
        self.last_pose = np.eye(4)
        self.keyframes = 0

    def track(self, frame):
        if self.reference is None:
            self.reference = make_reference(frame, np.eye(4), self.settings)
            self.last = frame
            self.keyframes += 1
            return np.eye(4), {"status": "origin", "keyframes": self.keyframes}
        reference = self.reference
        used_fallback = False
        try:
            prediction = np.linalg.inv(self.last_pose) @ reference.pose
            transform, quality = register(reference, frame, prediction, settings=self.settings)
        except ValueError as primary_error:
            if reference.frame.index == self.last.index:
                return None, {"status": "rejected", "reason": str(primary_error)}
            try:
                reference = make_reference(self.last, self.last_pose, self.settings)
                transform, quality = register(reference, frame, np.eye(4), settings=self.settings)
                used_fallback = True
            except ValueError as fallback_error:
                return None, {"status": "rejected", "reason": str(fallback_error)}
        pose = reference.pose @ np.linalg.inv(transform)
        step = np.linalg.inv(self.last_pose) @ pose
        if np.linalg.norm(step[:3, 3]) > .2 or angle_deg(step[:3, :3]) > 20:
            return None, {"status": "rejected", "reason": "Implausibly large motion from last tracked frame"}
        promote = (used_fallback or np.linalg.norm(transform[:3, 3]) > .10
                   or angle_deg(transform[:3, :3]) > 10
                   or quality["inliers"] < .5 * len(reference.points))
        if promote:
            try:
                self.reference = make_reference(frame, pose, self.settings)
                self.keyframes += 1
            except ValueError:
                self.reference = reference
        quality.update(status="accepted", fallback_to_last_frame=used_fallback,
                       keyframes=self.keyframes, step_translation_m=float(np.linalg.norm(step[:3, 3])),
                       step_rotation_deg=angle_deg(step[:3, :3]))
        self.last, self.last_pose = frame, pose
        return pose, quality
