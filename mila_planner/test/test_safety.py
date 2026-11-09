import math
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mila_navigation.controller import VelocityCommand
from mila_navigation.safety import SafetySupervisor


class SafetyTests(unittest.TestCase):
    def fresh(self, command=VelocityCommand(0.1, 0.0), stamp=1.0):
        supervisor = SafetySupervisor()
        supervisor.submit_command(command, stamp)
        supervisor.update_odometry(stamp)
        return supervisor

    def test_fresh_command_passes(self):
        supervisor = self.fresh()
        first = supervisor.evaluate(1.0)
        supervisor.submit_command(VelocityCommand(0.1, 0.0), 1.1)
        supervisor.update_odometry(1.1)
        decision = supervisor.evaluate(1.1)
        self.assertTrue(first.allowed)
        self.assertEqual(first.command, VelocityCommand())
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.reason, 'ok')
        self.assertAlmostEqual(decision.command.linear, 0.01)

    def test_clamps_reverse_speed_and_rotation_in_place(self):
        decision = self.fresh(VelocityCommand(-1.0, 4.0)).evaluate(1.0)
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.command.linear, 0.0)
        self.assertEqual(decision.command.angular, 0.0)

    def test_clamps_speed_and_curvature(self):
        supervisor = self.fresh(VelocityCommand(2.0, 4.0))
        supervisor.evaluate(1.0)
        supervisor.submit_command(VelocityCommand(2.0, 4.0), 1.1)
        supervisor.update_odometry(1.1)
        decision = supervisor.evaluate(1.1)
        self.assertAlmostEqual(decision.command.linear, 0.01)
        self.assertLessEqual(abs(decision.command.angular), 0.01 / 0.70 + 1e-12)

    def test_stale_command_immediately_stops(self):
        decision = self.fresh().evaluate(1.251)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, 'stale_nav_command')
        self.assertEqual(decision.command, VelocityCommand())

    def test_stale_odometry_immediately_stops(self):
        supervisor = SafetySupervisor()
        supervisor.submit_command(VelocityCommand(0.1, 0), 1.20)
        supervisor.update_odometry(1.0)
        decision = supervisor.evaluate(1.26)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, 'stale_odometry')

    def test_nan_command_is_rejected(self):
        decision = self.fresh(VelocityCommand(math.nan, 0)).evaluate(1.0)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, 'invalid_nav_command')

    def test_acceleration_is_limited_between_outputs(self):
        supervisor = self.fresh(VelocityCommand(0.0, 0.0), 1.0)
        supervisor.evaluate(1.0)
        supervisor.submit_command(VelocityCommand(0.15, 0), 1.1)
        supervisor.update_odometry(1.1)
        decision = supervisor.evaluate(1.1)
        self.assertAlmostEqual(decision.command.linear, 0.01)


if __name__ == '__main__':
    unittest.main()

