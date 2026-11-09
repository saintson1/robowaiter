"""ROS-independent occupancy mapping and manual endpoint loop closure.

The core deliberately depends only on NumPy and the Python standard library so
that the geometry, ray tracing, persistence and loop correction can be tested
on a machine without ROS installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import re
from typing import Iterable, Sequence

import numpy as np


def wrap_angle(value: float) -> float:
    """Return an angle in [-pi, pi)."""
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class SE2:
    x: float
    y: float
    yaw: float

    def matrix(self) -> np.ndarray:
        cosine, sine = math.cos(self.yaw), math.sin(self.yaw)
        return np.array(((cosine, -sine, self.x),
                         (sine, cosine, self.y),
                         (0.0, 0.0, 1.0)), dtype=np.float64)

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> 'SE2':
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError('SE2 matrix must be finite and 3x3')
        return cls(float(matrix[0, 2]), float(matrix[1, 2]),
                   wrap_angle(math.atan2(matrix[1, 0], matrix[0, 0])))

    def inverse(self) -> 'SE2':
        cosine, sine = math.cos(self.yaw), math.sin(self.yaw)
        return SE2(-cosine * self.x - sine * self.y,
                   sine * self.x - cosine * self.y,
                   wrap_angle(-self.yaw))

    def compose(self, other: 'SE2') -> 'SE2':
        return SE2.from_matrix(self.matrix() @ other.matrix())

    def transform_points(self, points_xy: np.ndarray) -> np.ndarray:
        points = np.asarray(points_xy, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError('points must be Nx2')
        cosine, sine = math.cos(self.yaw), math.sin(self.yaw)
        result = np.empty_like(points)
        result[:, 0] = cosine * points[:, 0] - sine * points[:, 1] + self.x
        result[:, 1] = sine * points[:, 0] + cosine * points[:, 1] + self.y
        return result


def se2_log(transform: SE2) -> np.ndarray:
    """Logarithm coordinates [vx, vy, omega] for an SE(2) transform."""
    theta = wrap_angle(transform.yaw)
    translation = np.array((transform.x, transform.y), dtype=np.float64)
    if abs(theta) < 1e-9:
        velocity = translation
    else:
        a = math.sin(theta) / theta
        b = (1.0 - math.cos(theta)) / theta
        determinant = a * a + b * b
        velocity = np.array(((a * translation[0] + b * translation[1]) / determinant,
                             (-b * translation[0] + a * translation[1]) / determinant))
    return np.array((velocity[0], velocity[1], theta), dtype=np.float64)


def se2_exp(tangent: Sequence[float]) -> SE2:
    """Exponential map for [vx, vy, omega]."""
    vx, vy, theta = map(float, tangent)
    if abs(theta) < 1e-9:
        x, y = vx, vy
    else:
        a = math.sin(theta) / theta
        b = (1.0 - math.cos(theta)) / theta
        x, y = a * vx - b * vy, b * vx + a * vy
    return SE2(x, y, wrap_angle(theta))


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        values = (self.fx, self.fy, self.cx, self.cy)
        if self.width < 1 or self.height < 1 or self.fx <= 0 or self.fy <= 0:
            raise ValueError('invalid camera intrinsics')
        if not all(math.isfinite(value) for value in values):
            raise ValueError('camera intrinsics must be finite')


@dataclass(frozen=True)
class RayBatch:
    """Downsampled depth rays stored in the robot base frame."""
    origin_xy: np.ndarray
    endpoints_xy: np.ndarray
    obstacle: np.ndarray

    def __post_init__(self) -> None:
        origin = np.asarray(self.origin_xy, dtype=np.float32)
        endpoints = np.asarray(self.endpoints_xy, dtype=np.float32)
        obstacle = np.asarray(self.obstacle, dtype=np.bool_)
        if origin.shape != (2,) or endpoints.ndim != 2 or endpoints.shape[1] != 2:
            raise ValueError('invalid ray batch geometry')
        if obstacle.shape != (len(endpoints),):
            raise ValueError('obstacle mask does not match endpoints')
        if not np.isfinite(origin).all() or not np.isfinite(endpoints).all():
            raise ValueError('ray geometry must be finite')
        object.__setattr__(self, 'origin_xy', origin.copy())
        object.__setattr__(self, 'endpoints_xy', endpoints.copy())
        object.__setattr__(self, 'obstacle', obstacle.copy())


def make_camera_extrinsic(x: float, y: float, z: float, yaw_deg: float,
                          pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Return T_base_from_optical using the same convention as mila_live_recording.

    Camera optical axes are x right, y down, z forward. Base axes are x forward,
    y left, z up. Positive pitch in this convention points the optical z axis
    downwards, matching the existing odometry calibration.
    """
    values = np.asarray((x, y, z, yaw_deg, pitch_deg, roll_deg), dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('camera extrinsic must be finite')
    yaw, pitch, roll = np.radians((yaw_deg, pitch_deg, roll_deg))
    rx = np.array(((1.0, 0.0, 0.0),
                   (0.0, math.cos(roll), -math.sin(roll)),
                   (0.0, math.sin(roll), math.cos(roll))))
    ry = np.array(((math.cos(pitch), 0.0, math.sin(pitch)),
                   (0.0, 1.0, 0.0),
                   (-math.sin(pitch), 0.0, math.cos(pitch))))
    rz = np.array(((math.cos(yaw), -math.sin(yaw), 0.0),
                   (math.sin(yaw), math.cos(yaw), 0.0),
                   (0.0, 0.0, 1.0)))
    optical_to_base_axes = np.array(((0.0, 0.0, 1.0),
                                     (-1.0, 0.0, 0.0),
                                     (0.0, -1.0, 0.0)))
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rz @ ry @ rx @ optical_to_base_axes
    result[:3, 3] = values[:3]
    return result


def project_depth(depth_m: np.ndarray, intrinsics: CameraIntrinsics,
                  t_base_from_optical: np.ndarray, *, pixel_stride: int,
                  min_depth_m: float, max_depth_m: float,
                  min_obstacle_height_m: float,
                  max_obstacle_height_m: float) -> RayBatch:
    """Project a depth image to downsampled 2D rays in the base frame.

    Every valid return contributes a clearing endpoint. Only returns inside the
    configured height interval are obstacle endpoints.
    """
    depth = np.asarray(depth_m)
    transform = np.asarray(t_base_from_optical, dtype=np.float64)
    if depth.shape != (intrinsics.height, intrinsics.width):
        raise ValueError('depth shape does not match CameraInfo')
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError('camera transform must be finite and 4x4')
    if pixel_stride < 1:
        raise ValueError('pixel_stride must be positive')
    limits = (min_depth_m, max_depth_m, min_obstacle_height_m, max_obstacle_height_m)
    if not all(math.isfinite(value) for value in limits):
        raise ValueError('projection limits must be finite')
    if min_depth_m <= 0 or max_depth_m <= min_depth_m:
        raise ValueError('invalid depth interval')
    if max_obstacle_height_m <= min_obstacle_height_m:
        raise ValueError('invalid obstacle height interval')

    offset = pixel_stride // 2
    rows = np.arange(offset, intrinsics.height, pixel_stride, dtype=np.int32)
    columns = np.arange(offset, intrinsics.width, pixel_stride, dtype=np.int32)
    uu, vv = np.meshgrid(columns, rows)
    zz = depth[vv, uu].astype(np.float64, copy=False)
    valid = np.isfinite(zz) & (zz >= min_depth_m) & (zz <= max_depth_m)
    if not np.any(valid):
        return RayBatch(transform[:2, 3], np.empty((0, 2), np.float32),
                        np.empty(0, np.bool_))
    z = zz[valid]
    u, v = uu[valid], vv[valid]
    optical = np.column_stack(((u - intrinsics.cx) * z / intrinsics.fx,
                               (v - intrinsics.cy) * z / intrinsics.fy,
                               z))
    base = optical @ transform[:3, :3].T + transform[:3, 3]
    obstacle = ((base[:, 2] >= min_obstacle_height_m)
                & (base[:, 2] <= max_obstacle_height_m))
    return RayBatch(transform[:2, 3], base[:, :2], obstacle)


@dataclass(frozen=True)
class MapperConfig:
    resolution: float = 0.05
    initial_size_x_m: float = 7.0
    initial_size_y_m: float = 8.0
    max_size_x_m: float = 20.0
    max_size_y_m: float = 20.0
    expansion_cells: int = 64
    occupied_probability: float = 0.70
    free_probability: float = 0.40
    log_odds_min: float = -4.0
    log_odds_max: float = 4.0
    pgm_occupied_threshold: float = 0.65
    # 205 gray encodes occupancy 50/255 ~= 0.196078. Keep the free
    # threshold just below that so nav2_map_server reloads it as unknown.
    pgm_free_threshold: float = 0.196

    def __post_init__(self) -> None:
        if not math.isfinite(self.resolution) or self.resolution <= 0:
            raise ValueError('resolution must be positive')
        sizes = (self.initial_size_x_m, self.initial_size_y_m,
                 self.max_size_x_m, self.max_size_y_m)
        if not all(math.isfinite(value) and value > 0 for value in sizes):
            raise ValueError('initial map size must be positive')
        if (self.initial_size_x_m > self.max_size_x_m
                or self.initial_size_y_m > self.max_size_y_m):
            raise ValueError('initial map size exceeds configured maximum')
        if self.expansion_cells < 1:
            raise ValueError('expansion_cells must be positive')
        for name in ('occupied_probability', 'free_probability',
                     'pgm_occupied_threshold', 'pgm_free_threshold'):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise ValueError(name + ' must be between zero and one')
        if self.occupied_probability <= 0.5 or self.free_probability >= 0.5:
            raise ValueError('occupied/free probabilities must straddle 0.5')
        if self.log_odds_min >= self.log_odds_max:
            raise ValueError('invalid log-odds bounds')
        if self.pgm_free_threshold >= self.pgm_occupied_threshold:
            raise ValueError('PGM thresholds overlap')


class DynamicLogOddsGrid:
    """Dynamically expanding 2D log-odds occupancy grid."""
    def __init__(self, config: MapperConfig):
        self.config = config
        width = max(1, int(math.ceil(config.initial_size_x_m / config.resolution)))
        height = max(1, int(math.ceil(config.initial_size_y_m / config.resolution)))
        self.origin_x = -0.5 * width * config.resolution
        self.origin_y = -0.5 * height * config.resolution
        self.log_odds = np.zeros((height, width), dtype=np.float32)
        self.observed = np.zeros((height, width), dtype=np.bool_)
        self.occupied_delta = math.log(config.occupied_probability / (1.0 - config.occupied_probability))
        self.free_delta = math.log(config.free_probability / (1.0 - config.free_probability))

    @property
    def width(self) -> int:
        return int(self.log_odds.shape[1])

    @property
    def height(self) -> int:
        return int(self.log_odds.shape[0])

    def _raw_indices(self, points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points_xy, dtype=np.float64)
        columns = np.floor((points[:, 0] - self.origin_x) / self.config.resolution).astype(np.int64)
        rows = np.floor((points[:, 1] - self.origin_y) / self.config.resolution).astype(np.int64)
        return rows, columns

    def ensure_contains(self, points_xy: np.ndarray) -> None:
        points = np.asarray(points_xy, dtype=np.float64)
        if points.size == 0:
            return
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError('map points must be finite Nx2')
        rows, columns = self._raw_indices(points)
        chunk = self.config.expansion_cells
        left = max(0, int(math.ceil(max(0, -int(columns.min())) / chunk)) * chunk)
        bottom = max(0, int(math.ceil(max(0, -int(rows.min())) / chunk)) * chunk)
        right_needed = max(0, int(columns.max()) - self.width + 1)
        top_needed = max(0, int(rows.max()) - self.height + 1)
        right = int(math.ceil(right_needed / chunk)) * chunk if right_needed else 0
        top = int(math.ceil(top_needed / chunk)) * chunk if top_needed else 0
        if not any((left, right, bottom, top)):
            return
        new_shape = (self.height + bottom + top, self.width + left + right)
        max_width = int(math.ceil(self.config.max_size_x_m / self.config.resolution))
        max_height = int(math.ceil(self.config.max_size_y_m / self.config.resolution))
        if new_shape[1] > max_width or new_shape[0] > max_height:
            raise MemoryError(
                f'map expansion to {new_shape[1]}x{new_shape[0]} cells exceeds '
                f'configured maximum {max_width}x{max_height}')
        new_odds = np.zeros(new_shape, dtype=np.float32)
        new_observed = np.zeros(new_shape, dtype=np.bool_)
        rows_slice = slice(bottom, bottom + self.height)
        columns_slice = slice(left, left + self.width)
        new_odds[rows_slice, columns_slice] = self.log_odds
        new_observed[rows_slice, columns_slice] = self.observed
        self.log_odds, self.observed = new_odds, new_observed
        self.origin_x -= left * self.config.resolution
        self.origin_y -= bottom * self.config.resolution

    def world_to_grid(self, points_xy: np.ndarray) -> np.ndarray:
        rows, columns = self._raw_indices(points_xy)
        return np.column_stack((rows, columns)).astype(np.int64, copy=False)

    @staticmethod
    def ray_cells(row0: int, column0: int, row1: int, column1: int) -> Iterable[tuple[int, int]]:
        """Inclusive integer Bresenham line as (row, column)."""
        x0, y0, x1, y1 = column0, row0, column1, row1
        dx, sx = abs(x1 - x0), 1 if x0 < x1 else -1
        dy, sy = -abs(y1 - y0), 1 if y0 < y1 else -1
        error = dx + dy
        while True:
            yield y0, x0
            if x0 == x1 and y0 == y1:
                break
            twice = 2 * error
            if twice >= dy:
                error += dy
                x0 += sx
            if twice <= dx:
                error += dx
                y0 += sy

    def integrate(self, pose: SE2, ray_batches: Sequence[RayBatch]) -> dict[str, int]:
        transformed: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        all_points: list[np.ndarray] = []
        for batch in ray_batches:
            if len(batch.endpoints_xy) == 0:
                continue
            origin = pose.transform_points(batch.origin_xy.reshape(1, 2))[0]
            endpoints = pose.transform_points(batch.endpoints_xy)
            transformed.append((origin, endpoints, batch.obstacle))
            all_points.extend((origin.reshape(1, 2), endpoints))
        if not transformed:
            return {'rays': 0, 'free_cells': 0, 'occupied_cells': 0}
        self.ensure_contains(np.vstack(all_points))

        free_cells: set[tuple[int, int]] = set()
        occupied_cells: set[tuple[int, int]] = set()
        unique_rays: dict[tuple[int, int, int, int], bool] = {}
        for origin, endpoints, obstacle in transformed:
            origin_cell = self.world_to_grid(origin.reshape(1, 2))[0]
            endpoint_cells = self.world_to_grid(endpoints)
            for cell, is_obstacle in zip(endpoint_cells, obstacle):
                key = (int(origin_cell[0]), int(origin_cell[1]), int(cell[0]), int(cell[1]))
                unique_rays[key] = unique_rays.get(key, False) or bool(is_obstacle)

        for (row0, column0, row1, column1), is_obstacle in unique_rays.items():
            cells = list(self.ray_cells(row0, column0, row1, column1))
            if is_obstacle:
                free_cells.update(cells[:-1])
                occupied_cells.add(cells[-1])
            else:
                free_cells.update(cells)
        free_cells.difference_update(occupied_cells)

        if free_cells:
            rows, columns = zip(*free_cells)
            self.log_odds[rows, columns] += self.free_delta
            self.observed[rows, columns] = True
        if occupied_cells:
            rows, columns = zip(*occupied_cells)
            self.log_odds[rows, columns] += self.occupied_delta
            self.observed[rows, columns] = True
        np.clip(self.log_odds, self.config.log_odds_min, self.config.log_odds_max,
                out=self.log_odds)
        return {'rays': len(unique_rays), 'free_cells': len(free_cells),
                'occupied_cells': len(occupied_cells)}

    def occupancy_values(self) -> np.ndarray:
        values = np.full(self.log_odds.shape, -1, dtype=np.int8)
        if np.any(self.observed):
            probabilities = 1.0 / (1.0 + np.exp(-self.log_odds[self.observed]))
            values[self.observed] = np.rint(probabilities * 100.0).astype(np.int8)
        return values

    def pgm_pixels(self) -> np.ndarray:
        probabilities = 1.0 / (1.0 + np.exp(-self.log_odds))
        pixels = np.full(self.log_odds.shape, 205, dtype=np.uint8)
        pixels[self.observed & (probabilities <= self.config.pgm_free_threshold)] = 254
        pixels[self.observed & (probabilities >= self.config.pgm_occupied_threshold)] = 0
        return np.flipud(pixels)


@dataclass
class MappingFrame:
    stamp_ns: int
    raw_pose: SE2
    path_distance_m: float
    rays: tuple[RayBatch, ...]
    corrected_pose: SE2 | None = None


@dataclass
class OccupancyMappingCore:
    config: MapperConfig = field(default_factory=MapperConfig)

    def __post_init__(self) -> None:
        self.grid = DynamicLogOddsGrid(self.config)
        self.frames: list[MappingFrame] = []
        self.finalized = False
        self.integration_totals = {'rays': 0, 'free_cells': 0, 'occupied_cells': 0}

    def reset(self) -> None:
        self.grid = DynamicLogOddsGrid(self.config)
        self.frames.clear()
        self.finalized = False
        self.integration_totals = {'rays': 0, 'free_cells': 0, 'occupied_cells': 0}

    def add_frame(self, stamp_ns: int, pose: SE2, ray_batches: Sequence[RayBatch]) -> dict[str, int]:
        if self.finalized:
            raise RuntimeError('map is finalized; reset before adding observations')
        if stamp_ns < 0 or (self.frames and stamp_ns <= self.frames[-1].stamp_ns):
            raise ValueError('frame timestamps must be nonnegative and increasing')
        if not all(math.isfinite(value) for value in (pose.x, pose.y, pose.yaw)):
            raise ValueError('pose must be finite')
        if self.frames:
            previous = self.frames[-1].raw_pose
            distance = self.frames[-1].path_distance_m + math.hypot(
                pose.x - previous.x, pose.y - previous.y)
        else:
            distance = 0.0
        copied_rays = tuple(RayBatch(batch.origin_xy, batch.endpoints_xy, batch.obstacle)
                            for batch in ray_batches)
        frame = MappingFrame(int(stamp_ns), pose, distance, copied_rays)
        self.frames.append(frame)
        stats = self.grid.integrate(pose, copied_rays)
        for name, value in stats.items():
            self.integration_totals[name] += value
        return stats

    def finish_loop(self, *, min_path_length_m: float = 1.0,
                    max_residual_translation_m: float = 1.5,
                    max_residual_yaw_rad: float = math.radians(60.0)) -> dict[str, float]:
        """Close a manually confirmed endpoint loop and rebuild the map.

        The operator guarantees that the final physical pose equals the first
        physical pose. The SE(2) residual is distributed along accumulated path
        length using an interpolated left correction.
        """
        if self.finalized:
            raise RuntimeError('map is already finalized')
        if len(self.frames) < 2:
            raise ValueError('at least two mapping frames are required')
        total_length = self.frames[-1].path_distance_m
        if total_length < min_path_length_m:
            raise ValueError('path is too short for endpoint loop closure')
        first, last = self.frames[0].raw_pose, self.frames[-1].raw_pose
        residual = first.inverse().compose(last)
        residual_translation = math.hypot(residual.x, residual.y)
        residual_yaw = abs(wrap_angle(residual.yaw))
        if residual_translation > max_residual_translation_m:
            raise ValueError('endpoint translation residual exceeds safety limit')
        if residual_yaw > max_residual_yaw_rad:
            raise ValueError('endpoint yaw residual exceeds safety limit')

        final_correction = first.compose(last.inverse())
        correction_tangent = se2_log(final_correction)
        corrected: list[SE2] = []
        for frame in self.frames:
            fraction = frame.path_distance_m / total_length
            correction = se2_exp(correction_tangent * fraction)
            frame.corrected_pose = correction.compose(frame.raw_pose)
            corrected.append(frame.corrected_pose)
        # Avoid a few ulps at the endpoint: the operator asserted exact closure.
        self.frames[-1].corrected_pose = first

        self.grid = DynamicLogOddsGrid(self.config)
        self.integration_totals = {'rays': 0, 'free_cells': 0, 'occupied_cells': 0}
        for frame in self.frames:
            stats = self.grid.integrate(frame.corrected_pose, frame.rays)
            for name, value in stats.items():
                self.integration_totals[name] += value
        self.finalized = True
        return {'path_length_m': total_length,
                'residual_translation_m': residual_translation,
                'residual_yaw_rad': residual_yaw,
                'frames': float(len(self.frames))}

    @staticmethod
    def _valid_map_name(map_name: str) -> str:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', map_name):
            raise ValueError('map_name must be a simple filename stem')
        if map_name.lower().endswith(('.pgm', '.yaml', '.yml')):
            raise ValueError('map_name must not include an extension')
        return map_name

    def save(self, directory: str | os.PathLike[str], map_name: str) -> tuple[Path, Path]:
        if not np.any(self.grid.observed):
            raise ValueError('cannot save an empty map')
        name = self._valid_map_name(map_name)
        output = Path(directory).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        pgm_path, yaml_path = output / f'{name}.pgm', output / f'{name}.yaml'
        pgm_temporary = output / f'.{name}.pgm.tmp'
        yaml_temporary = output / f'.{name}.yaml.tmp'

        pixels = self.grid.pgm_pixels()
        header = (f'P5\n# CREATOR: dual_rgbd_global_mapper\n'
                  f'{self.grid.width} {self.grid.height}\n255\n').encode('ascii')
        pgm_temporary.write_bytes(header + pixels.tobytes(order='C'))
        yaml_text = (
            f'image: {pgm_path.name}\n'
            'mode: trinary\n'
            f'resolution: {self.config.resolution:.9g}\n'
            f'origin: [{self.grid.origin_x:.9g}, {self.grid.origin_y:.9g}, 0.0]\n'
            'negate: 0\n'
            f'occupied_thresh: {self.config.pgm_occupied_threshold:.9g}\n'
            f'free_thresh: {self.config.pgm_free_threshold:.9g}\n')
        yaml_temporary.write_text(yaml_text, encoding='utf-8')
        os.replace(pgm_temporary, pgm_path)
        os.replace(yaml_temporary, yaml_path)
        return pgm_path, yaml_path
