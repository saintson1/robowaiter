"""ROS-independent final velocity gate and watchdog."""
from __future__ import annotations

from dataclasses import dataclass
import math

from .controller import VelocityCommand


@dataclass(frozen=True)
class SafetyConfig:
    min_turn_radius: float = 0.70
    max_speed: float = 0.15
    max_acceleration: float = 0.10
    max_deceleration: float = 0.20
    command_timeout: float = 0.25
    odometry_timeout: float = 0.25

    def __post_init__(self) -> None:
        values = (
            self.min_turn_radius,
            self.max_speed,
            self.max_acceleration,
            self.max_deceleration,
            self.command_timeout,
            self.odometry_timeout,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("safety parameters must be positive and finite")


@dataclass(frozen=True)
class SafetyDecision:
    command: VelocityCommand
    allowed: bool
    reason: str


class SafetySupervisor:
    """Only fresh, finite and kinematically valid commands can pass."""

    def __init__(self, config: SafetyConfig | None = None) -> None:
        self.config = config or SafetyConfig()
        self._command = VelocityCommand()
        self._command_stamp: float | None = None
        self._command_valid = False
        self._odom_stamp: float | None = None
        self._odom_valid = False
        self._last_output = VelocityCommand()
        self._last_output_stamp: float | None = None

    def submit_command(self, command: VelocityCommand, stamp: float) -> None:
        if not math.isfinite(stamp):
            raise ValueError("command timestamp must be finite")
        self._command = command
        self._command_stamp = stamp
        self._command_valid = all(math.isfinite(value)
                                  for value in (command.linear, command.angular))

    def update_odometry(self, stamp: float, valid: bool = True) -> None:
        if not math.isfinite(stamp):
            raise ValueError("odometry timestamp must be finite")
        self._odom_stamp = stamp
        self._odom_valid = bool(valid)

    def _stop(self, now: float, reason: str) -> SafetyDecision:
        self._last_output = VelocityCommand()
        self._last_output_stamp = now
        return SafetyDecision(self._last_output, False, reason)

    @staticmethod
    def _fresh(now: float, stamp: float | None, timeout: float) -> bool:
        return stamp is not None and 0.0 <= now - stamp <= timeout

    def evaluate(self, now: float) -> SafetyDecision:
        if not math.isfinite(now):
            raise ValueError("evaluation timestamp must be finite")
        if not self._command_valid:
            return self._stop(now, "invalid_nav_command")
        if not self._fresh(now, self._command_stamp, self.config.command_timeout):
            return self._stop(now, "stale_nav_command")
        if not self._odom_valid:
            return self._stop(now, "invalid_odometry")
        if not self._fresh(now, self._odom_stamp, self.config.odometry_timeout):
            return self._stop(now, "stale_odometry")

        requested = self._command
        linear = max(0.0, min(self.config.max_speed, requested.linear))
        curvature_limit = 1.0 / self.config.min_turn_radius
        requested_curvature = (requested.angular / requested.linear
                               if requested.linear > 0.0 else 0.0)
        curvature = max(-curvature_limit, min(curvature_limit, requested_curvature))

        if self._last_output_stamp is None:
            # Establish a zero-speed baseline; the next 20 Hz cycle performs
            # the first bounded acceleration step.
            linear = 0.0
        else:
            dt = now - self._last_output_stamp
            if dt < 0.0:
                return self._stop(now, "non_monotonic_time")
            previous = self._last_output.linear
            if linear >= previous:
                linear = min(linear, previous + self.config.max_acceleration * dt)
            else:
                linear = max(linear, previous - self.config.max_deceleration * dt)
        angular = linear * curvature if linear > 0.0 else 0.0

        self._last_output = VelocityCommand(linear, angular)
        self._last_output_stamp = now
        return SafetyDecision(self._last_output, True, "ok")

