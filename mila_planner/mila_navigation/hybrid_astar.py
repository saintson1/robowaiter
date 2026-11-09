"""ROS-independent forward-only Hybrid A* planner.

The planner works in metric world coordinates, preserves a continuous pose for
each search node, and discretizes only the closed-list key.  Motion primitives
are constant-curvature forward arcs, so every returned segment respects the
configured minimum turning radius and never contains reverse motion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import itertools
import math
from typing import Iterable

import numpy as np


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class GridMap:
    """A ROS OccupancyGrid-like map stored with row 0 at map-local y=0."""

    data: np.ndarray
    resolution: float
    origin_x: float = 0.0
    origin_y: float = 0.0
    origin_yaw: float = 0.0

    def __post_init__(self) -> None:
        array = np.asarray(self.data)
        if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
            raise ValueError('map data must be a non-empty 2D array')
        if not math.isfinite(self.resolution) or self.resolution <= 0.0:
            raise ValueError('resolution must be positive and finite')
        if not all(math.isfinite(v) for v in
                   (self.origin_x, self.origin_y, self.origin_yaw)):
            raise ValueError('map origin must be finite')
        object.__setattr__(self, 'data', array.astype(np.int16, copy=False))

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    def world_to_local(self, x: float, y: float) -> tuple[float, float]:
        dx, dy = x - self.origin_x, y - self.origin_y
        c, s = math.cos(self.origin_yaw), math.sin(self.origin_yaw)
        return c * dx + s * dy, -s * dx + c * dy

    def local_to_world(self, x: float, y: float) -> tuple[float, float]:
        c, s = math.cos(self.origin_yaw), math.sin(self.origin_yaw)
        return (self.origin_x + c * x - s * y,
                self.origin_y + s * x + c * y)

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        local_x, local_y = self.world_to_local(x, y)
        return math.floor(local_x / self.resolution), math.floor(local_y / self.resolution)

    def cell_center_world(self, ix: int, iy: int) -> tuple[float, float]:
        return self.local_to_world((ix + 0.5) * self.resolution,
                                   (iy + 0.5) * self.resolution)

    def contains_cell(self, ix: int, iy: int) -> bool:
        return 0 <= ix < self.width and 0 <= iy < self.height


@dataclass(frozen=True)
class PlannerConfig:
    footprint_length_m: float = 0.60
    footprint_width_m: float = 0.45
    safety_margin_m: float = 0.30
    min_turning_radius_m: float = 0.70
    goal_tolerance_m: float = 0.15
    primitive_length_m: float = 0.15
    collision_sample_m: float = 0.05
    yaw_bins: int = 36
    steering_fractions: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
    occupied_threshold: int = 65
    unknown_penalty: float = 3.0
    turn_penalty: float = 0.08
    steering_change_penalty: float = 0.04
    heuristic_weight: float = 1.5
    max_expansions: int = 200_000

    def __post_init__(self) -> None:
        positive = (self.footprint_length_m, self.footprint_width_m,
                    self.min_turning_radius_m, self.goal_tolerance_m,
                    self.primitive_length_m, self.collision_sample_m)
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError('planner dimensions and distances must be positive and finite')
        if not math.isfinite(self.safety_margin_m) or self.safety_margin_m < 0.0:
            raise ValueError('safety margin must be nonnegative and finite')
        if self.yaw_bins < 8 or self.max_expansions < 1:
            raise ValueError('yaw_bins must be >=8 and max_expansions positive')
        if not 0 <= self.occupied_threshold <= 100:
            raise ValueError('occupied_threshold must be in [0, 100]')
        if self.unknown_penalty < 0.0 or self.turn_penalty < 0.0:
            raise ValueError('cost penalties must be nonnegative')
        if not math.isfinite(self.heuristic_weight) or self.heuristic_weight < 1.0:
            raise ValueError('heuristic_weight must be finite and >=1')
        if not self.steering_fractions or any(
                not math.isfinite(value) or abs(value) > 1.0
                for value in self.steering_fractions):
            raise ValueError('steering fractions must be finite in [-1, 1]')

    @property
    def effective_length_m(self) -> float:
        return self.footprint_length_m + 2.0 * self.safety_margin_m

    @property
    def effective_width_m(self) -> float:
        return self.footprint_width_m + 2.0 * self.safety_margin_m

    @property
    def curvatures(self) -> tuple[float, ...]:
        return tuple(value / self.min_turning_radius_m
                     for value in self.steering_fractions)


@dataclass(frozen=True)
class FootprintResult:
    collision: bool
    unknown_fraction: float


class FootprintChecker:
    def __init__(self, grid: GridMap, config: PlannerConfig):
        self.grid = grid
        self.config = config
        self.half_length = config.effective_length_m / 2.0
        self.half_width = config.effective_width_m / 2.0
        occupied_y, occupied_x = np.nonzero(grid.data >= config.occupied_threshold)
        self.occupied_x = (occupied_x.astype(float) + 0.5) * grid.resolution
        self.occupied_y = (occupied_y.astype(float) + 0.5) * grid.resolution
        self.has_unknown = bool(np.any(grid.data < 0))
        # Occupied cells are squares, not points. Padding by their half diagonal
        # makes the center-based vectorized test conservative.
        self.cell_padding = grid.resolution / math.sqrt(2.0)

    def check(self, pose: Pose2D) -> FootprintResult:
        """Check the inflated oriented rectangle at ``pose``.

        Cells outside the global map are collision. Unknown cells are traversable
        and reported as a fraction so the search can penalize, not forbid, them.
        """
        # Work in map-local coordinates, where the occupancy array is axis aligned.
        cx, cy = self.grid.world_to_local(pose.x, pose.y)
        yaw = wrap_angle(pose.yaw - self.grid.origin_yaw)
        c, s = math.cos(yaw), math.sin(yaw)
        corners = np.array([
            [self.half_length, self.half_width],
            [self.half_length, -self.half_width],
            [-self.half_length, self.half_width],
            [-self.half_length, -self.half_width],
        ])
        rotation = np.array([[c, -s], [s, c]])
        corners = corners @ rotation.T + [cx, cy]
        map_width_m = self.grid.width * self.grid.resolution
        map_height_m = self.grid.height * self.grid.resolution
        if ((corners[:, 0] < 0.0).any() or (corners[:, 0] >= map_width_m).any()
                or (corners[:, 1] < 0.0).any() or (corners[:, 1] >= map_height_m).any()):
            return FootprintResult(True, 0.0)

        min_x, max_x = float(corners[:, 0].min()), float(corners[:, 0].max())
        min_y, max_y = float(corners[:, 1].min()), float(corners[:, 1].max())
        nearby = ((self.occupied_x >= min_x - self.cell_padding)
                  & (self.occupied_x <= max_x + self.cell_padding)
                  & (self.occupied_y >= min_y - self.cell_padding)
                  & (self.occupied_y <= max_y + self.cell_padding))
        if np.any(nearby):
            dx = self.occupied_x[nearby] - cx
            dy = self.occupied_y[nearby] - cy
            longitudinal = c * dx + s * dy
            lateral = -s * dx + c * dy
            if np.any((np.abs(longitudinal) <= self.half_length + self.cell_padding)
                      & (np.abs(lateral) <= self.half_width + self.cell_padding)):
                return FootprintResult(True, 0.0)

        if not self.has_unknown:
            return FootprintResult(False, 0.0)

        min_ix = max(0, math.floor(min_x / self.grid.resolution))
        max_ix = min(self.grid.width - 1,
                     math.floor(max_x / self.grid.resolution))
        min_iy = max(0, math.floor(min_y / self.grid.resolution))
        max_iy = min(self.grid.height - 1,
                     math.floor(max_y / self.grid.resolution))
        xs = (np.arange(min_ix, max_ix + 1) + 0.5) * self.grid.resolution
        ys = (np.arange(min_iy, max_iy + 1) + 0.5) * self.grid.resolution
        xx, yy = np.meshgrid(xs, ys)
        dx, dy = xx - cx, yy - cy
        # Inverse body rotation: map-local point -> robot-local point.
        longitudinal = c * dx + s * dy
        lateral = -s * dx + c * dy
        inside = ((np.abs(longitudinal) <= self.half_length + 1e-12)
                  & (np.abs(lateral) <= self.half_width + 1e-12))
        values = self.grid.data[min_iy:max_iy + 1, min_ix:max_ix + 1][inside]
        if values.size == 0:
            return FootprintResult(True, 0.0)
        return FootprintResult(False, float(np.mean(values < 0)))


@dataclass
class _SearchNode:
    pose: Pose2D
    cost: float
    curvature: float | None
    parent: '_SearchNode | None' = field(repr=False)


class PlanningError(RuntimeError):
    pass


class HybridAStarPlanner:
    def __init__(self, grid: GridMap, config: PlannerConfig = PlannerConfig()):
        self.grid = grid
        self.config = config
        self.checker = FootprintChecker(grid, config)

    @staticmethod
    def advance(pose: Pose2D, distance: float, curvature: float) -> Pose2D:
        if distance < 0.0:
            raise ValueError('reverse motion is forbidden')
        if abs(curvature) < 1e-12:
            return Pose2D(pose.x + distance * math.cos(pose.yaw),
                          pose.y + distance * math.sin(pose.yaw), pose.yaw)
        new_yaw = pose.yaw + curvature * distance
        x = pose.x + (math.sin(new_yaw) - math.sin(pose.yaw)) / curvature
        y = pose.y - (math.cos(new_yaw) - math.cos(pose.yaw)) / curvature
        return Pose2D(x, y, wrap_angle(new_yaw))

    def sample_primitive(self, pose: Pose2D, curvature: float) -> list[Pose2D]:
        if abs(curvature) > 1.0 / self.config.min_turning_radius_m + 1e-12:
            raise ValueError('curvature violates minimum turning radius')
        count = max(1, math.ceil(self.config.primitive_length_m /
                                 self.config.collision_sample_m))
        return [self.advance(pose, self.config.primitive_length_m * i / count, curvature)
                for i in range(1, count + 1)]

    def _state_key(self, pose: Pose2D) -> tuple[int, int, int]:
        ix, iy = self.grid.world_to_cell(pose.x, pose.y)
        yaw = wrap_angle(pose.yaw - self.grid.origin_yaw)
        yaw_index = int(round((yaw % (2.0 * math.pi)) * self.config.yaw_bins /
                              (2.0 * math.pi))) % self.config.yaw_bins
        return ix, iy, yaw_index

    @staticmethod
    def _distance(a: Pose2D, b: Pose2D) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    def _transition(self, node: _SearchNode, curvature: float) -> tuple[Pose2D, float] | None:
        samples = self.sample_primitive(node.pose, curvature)
        unknown = 0.0
        for sample in samples:
            result = self.checker.check(sample)
            if result.collision:
                return None
            unknown += result.unknown_fraction
        unknown /= len(samples)
        normalized_turn = abs(curvature) * self.config.min_turning_radius_m
        cost = self.config.primitive_length_m * (
            1.0 + self.config.unknown_penalty * unknown
            + self.config.turn_penalty * normalized_turn)
        if node.curvature is not None:
            cost += self.config.steering_change_penalty * abs(
                curvature - node.curvature) * self.config.min_turning_radius_m
        return samples[-1], cost

    def plan(self, start: Pose2D, goal: Pose2D) -> list[Pose2D]:
        """Return a dense forward-only path from start to goal position.

        Goal yaw is intentionally ignored.  This matches a car-like robot that
        cannot rotate in place and only needs to reach the requested coordinate.
        """
        if self.checker.check(start).collision:
            raise PlanningError('start footprint is in collision or outside map')
        # Goal orientation is not constrained. Reject the coordinate only when
        # the footprint collides for every discretized arrival orientation.
        feasible_goal_orientation = any(
            not self.checker.check(Pose2D(
                goal.x, goal.y,
                self.grid.origin_yaw + 2.0 * math.pi * index / self.config.yaw_bins)).collision
            for index in range(self.config.yaw_bins))
        if not feasible_goal_orientation:
            raise PlanningError('goal footprint is in collision or outside map')
        if self._distance(start, goal) <= self.config.goal_tolerance_m:
            return [start]

        start_node = _SearchNode(start, 0.0, None, None)
        counter = itertools.count()
        queue: list[tuple[float, int, _SearchNode]] = [
            (self.config.heuristic_weight * self._distance(start, goal),
             next(counter), start_node)]
        best_cost = {self._state_key(start): 0.0}
        expansions = 0

        while queue and expansions < self.config.max_expansions:
            _priority, _tie, node = heapq.heappop(queue)
            key = self._state_key(node.pose)
            if node.cost > best_cost.get(key, math.inf) + 1e-12:
                continue
            if self._distance(node.pose, goal) <= self.config.goal_tolerance_m:
                return self._reconstruct(node)
            expansions += 1
            for curvature in self.config.curvatures:
                transition = self._transition(node, curvature)
                if transition is None:
                    continue
                pose, added_cost = transition
                new_cost = node.cost + added_cost
                new_key = self._state_key(pose)
                if new_cost + 1e-12 >= best_cost.get(new_key, math.inf):
                    continue
                best_cost[new_key] = new_cost
                child = _SearchNode(pose, new_cost, curvature, node)
                priority = new_cost + self.config.heuristic_weight * self._distance(pose, goal)
                heapq.heappush(queue, (priority, next(counter), child))

        if expansions >= self.config.max_expansions:
            raise PlanningError(f'planning exceeded {self.config.max_expansions} expansions')
        raise PlanningError('no forward-only collision-free path found')

    def _reconstruct(self, node: _SearchNode) -> list[Pose2D]:
        chain: list[_SearchNode] = []
        current: _SearchNode | None = node
        while current is not None:
            chain.append(current)
            current = current.parent
        chain.reverse()
        result = [chain[0].pose]
        for child in chain[1:]:
            assert child.parent is not None and child.curvature is not None
            result.extend(self.sample_primitive(child.parent.pose, child.curvature))
        return result


def path_length(path: Iterable[Pose2D]) -> float:
    points = list(path)
    return sum(math.hypot(b.x - a.x, b.y - a.y)
               for a, b in zip(points, points[1:]))
