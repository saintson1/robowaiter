"""ROS-independent forward-only path tracking for the MILA robot."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float = 0.0


@dataclass(frozen=True)
class VelocityCommand:
    linear: float = 0.0
    angular: float = 0.0


@dataclass(frozen=True)
class ControllerConfig:
    min_turn_radius: float = 0.70
    min_speed: float = 0.04
    max_speed: float = 0.15
    max_acceleration: float = 0.10
    max_deceleration: float = 0.20
    lookahead_distance: float = 0.45
    goal_tolerance: float = 0.15
    curvature_slowdown: float = 1.0

    def __post_init__(self) -> None:
        values = (
            self.min_turn_radius,
            self.min_speed,
            self.max_speed,
            self.max_acceleration,
            self.max_deceleration,
            self.lookahead_distance,
            self.goal_tolerance,
            self.curvature_slowdown,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("controller parameters must be positive and finite")
        if self.min_speed > self.max_speed:
            raise ValueError("min_speed must not exceed max_speed")


class PurePursuitController:
    """Track a kinematically feasible path without reverse or in-place turns."""

    def __init__(self, config: ControllerConfig | None = None) -> None:
        self.config = config or ControllerConfig()

    @staticmethod
    def _distance(a: Pose2D, b: Pose2D) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    def _target(self, path: Sequence[Pose2D], pose: Pose2D) -> Pose2D:
        nearest = min(range(len(path)), key=lambda index: self._distance(path[index], pose))
        travelled = self._distance(path[nearest], pose)
        for index in range(nearest + 1, len(path)):
            travelled += self._distance(path[index - 1], path[index])
            if travelled >= self.config.lookahead_distance:
                return path[index]
        return path[-1]

    def compute(
        self,
        path: Sequence[Pose2D],
        pose: Pose2D,
        current_speed: float,
        dt: float,
    ) -> VelocityCommand:
        values = (pose.x, pose.y, pose.yaw, current_speed, dt)
        if not all(math.isfinite(value) for value in values) or dt <= 0.0:
            raise ValueError("pose, speed and positive dt must be finite")
        if not path:
            return VelocityCommand()
        if any(not all(math.isfinite(value) for value in (point.x, point.y, point.yaw))
               for point in path):
            raise ValueError("path contains a non-finite pose")

        goal_distance = self._distance(path[-1], pose)
        if goal_distance <= self.config.goal_tolerance:
            return VelocityCommand()

        target = self._target(path, pose)
        dx, dy = target.x - pose.x, target.y - pose.y
        cosine, sine = math.cos(pose.yaw), math.sin(pose.yaw)
        local_x = cosine * dx + sine * dy
        local_y = -sine * dx + cosine * dy
        squared_distance = local_x * local_x + local_y * local_y
        if squared_distance < 1e-8:
            return VelocityCommand()

        curvature = 2.0 * local_y / squared_distance
        curvature_limit = 1.0 / self.config.min_turn_radius
        curvature = max(-curvature_limit, min(curvature_limit, curvature))

        # Slow down on tight curves and near the goal. The minimum working speed
        # is retained until the positional tolerance is reached.
        curve_factor = 1.0 + self.config.curvature_slowdown * abs(curvature) * self.config.min_turn_radius
        curve_speed = self.config.max_speed / curve_factor
        remaining = max(0.0, goal_distance - self.config.goal_tolerance)
        braking_speed = math.sqrt(2.0 * self.config.max_deceleration * remaining)
        target_speed = min(curve_speed, braking_speed, self.config.max_speed)
        target_speed = max(self.config.min_speed, target_speed)

        speed = max(0.0, min(self.config.max_speed, current_speed))
        if target_speed >= speed:
            speed = min(target_speed, speed + self.config.max_acceleration * dt)
        else:
            speed = max(target_speed, speed - self.config.max_deceleration * dt)

        angular = speed * curvature
        if speed <= 0.0:
            return VelocityCommand()
        return VelocityCommand(linear=speed, angular=angular)

