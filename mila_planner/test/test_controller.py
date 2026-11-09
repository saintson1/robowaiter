import math
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mila_navigation.controller import ControllerConfig, Pose2D, PurePursuitController


class PurePursuitTests(unittest.TestCase):
    def setUp(self):
        self.controller = PurePursuitController()

    def test_straight_path_accelerates_forward(self):
        path = [Pose2D(0, 0), Pose2D(1, 0), Pose2D(2, 0)]
        command = self.controller.compute(path, Pose2D(0, 0), 0.0, 0.1)
        self.assertAlmostEqual(command.linear, 0.01)
        self.assertAlmostEqual(command.angular, 0.0)

    def test_curvature_obeys_minimum_radius(self):
        path = [Pose2D(0, 0), Pose2D(0.1, 1.0), Pose2D(0.2, 2.0)]
        command = self.controller.compute(path, Pose2D(0, 0), 0.15, 0.1)
        self.assertGreater(command.linear, 0.0)
        self.assertLessEqual(abs(command.angular), command.linear / 0.70 + 1e-12)

    def test_goal_position_stops_without_orientation_requirement(self):
        path = [Pose2D(0, 0), Pose2D(1, 0, math.pi)]
        command = self.controller.compute(path, Pose2D(0.90, 0, 0), 0.1, 0.1)
        self.assertEqual(command.linear, 0.0)
        self.assertEqual(command.angular, 0.0)

    def test_empty_path_stops(self):
        self.assertEqual(self.controller.compute([], Pose2D(0, 0), 0.1, 0.1).linear, 0.0)

    def test_deceleration_is_limited(self):
        config = ControllerConfig(curvature_slowdown=10.0)
        controller = PurePursuitController(config)
        path = [Pose2D(0, 0), Pose2D(0.2, 1), Pose2D(0.3, 2)]
        command = controller.compute(path, Pose2D(0, 0), 0.15, 0.1)
        self.assertGreaterEqual(command.linear, 0.13 - 1e-12)

    def test_rejects_nonfinite_input(self):
        with self.assertRaises(ValueError):
            self.controller.compute([Pose2D(1, 0)], Pose2D(math.nan, 0), 0.0, 0.1)


if __name__ == '__main__':
    unittest.main()

