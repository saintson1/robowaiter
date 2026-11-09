import math
import unittest

import numpy as np

from mila_navigation.hybrid_astar import (FootprintChecker, GridMap,
                                           HybridAStarPlanner, PlannerConfig,
                                           PlanningError, Pose2D, path_length)


class GridMapTests(unittest.TestCase):
    def test_rotated_origin_round_trip(self):
        grid = GridMap(np.zeros((20, 30), np.int8), .1, 2., -1., math.pi / 2)
        x, y = grid.cell_center_world(4, 7)
        self.assertEqual(grid.world_to_cell(x, y), (4, 7))

    def test_invalid_map_is_rejected(self):
        with self.assertRaises(ValueError):
            GridMap(np.zeros(3), .05)
        with self.assertRaises(ValueError):
            GridMap(np.zeros((2, 2)), 0.)


class FootprintTests(unittest.TestCase):
    def setUp(self):
        self.data = np.zeros((100, 100), np.int8)
        self.grid = GridMap(self.data, .1)
        self.config = PlannerConfig()

    def test_inflated_footprint_respects_thirty_centimeter_margin(self):
        # Effective half width is .45/2 + .30 = .525 m.
        self.grid.data[54, 50] = 100  # center y=5.45, within inflated footprint
        checker = FootprintChecker(self.grid, self.config)
        self.assertTrue(checker.check(Pose2D(5., 5., 0.)).collision)
        self.grid.data[54, 50] = 0
        self.grid.data[56, 50] = 100  # center y=5.65, outside padded footprint
        checker = FootprintChecker(self.grid, self.config)
        self.assertFalse(checker.check(Pose2D(5., 5., 0.)).collision)

    def test_unknown_is_traversable_and_reported(self):
        self.grid.data[:] = -1
        result = FootprintChecker(self.grid, self.config).check(Pose2D(5., 5., 0.))
        self.assertFalse(result.collision)
        self.assertEqual(result.unknown_fraction, 1.)

    def test_outside_map_is_collision(self):
        result = FootprintChecker(self.grid, self.config).check(Pose2D(.2, .2, 0.))
        self.assertTrue(result.collision)


class HybridAStarTests(unittest.TestCase):
    def test_open_map_straight_path(self):
        grid = GridMap(np.zeros((100, 100), np.int8), .1)
        planner = HybridAStarPlanner(grid)
        start, goal = Pose2D(2., 2., 0.), Pose2D(5., 2., 0.)
        path = planner.plan(start, goal)
        self.assertLessEqual(math.hypot(path[-1].x - goal.x, path[-1].y - goal.y), .15)
        self.assertAlmostEqual(path_length(path), 3., delta=.25)
        self.assertTrue(all(abs(p.y - 2.) < 1e-9 for p in path))

    def test_motion_primitives_are_forward_and_respect_radius(self):
        planner = HybridAStarPlanner(GridMap(np.zeros((100, 100), np.int8), .1))
        start = Pose2D(5., 5., .4)
        curvature = 1. / planner.config.min_turning_radius_m
        path = [start] + planner.sample_primitive(start, curvature)
        for previous, current in zip(path, path[1:]):
            dx, dy = current.x - previous.x, current.y - previous.y
            self.assertGreater(dx * math.cos(previous.yaw) + dy * math.sin(previous.yaw), 0.)
        total_yaw = abs((path[-1].yaw - path[0].yaw + math.pi) % (2 * math.pi) - math.pi)
        self.assertAlmostEqual(total_yaw / planner.config.primitive_length_m,
                               curvature, places=6)

    def test_unknown_map_can_be_planned_with_penalty(self):
        grid = GridMap(np.full((80, 80), -1, np.int8), .1)
        planner = HybridAStarPlanner(grid, PlannerConfig(unknown_penalty=5.))
        path = planner.plan(Pose2D(2., 2., 0.), Pose2D(4., 2., 0.))
        self.assertGreater(len(path), 2)

    def test_occupied_wall_is_avoided(self):
        data = np.zeros((100, 100), np.int8)
        data[:56, 49:51] = 100
        grid = GridMap(data, .1)
        config = PlannerConfig(primitive_length_m=.20, max_expansions=100_000)
        planner = HybridAStarPlanner(grid, config)
        path = planner.plan(Pose2D(2., 3., 0.), Pose2D(8., 3., 0.))
        self.assertTrue(any(point.y > 6.1 for point in path))
        self.assertTrue(all(not planner.checker.check(point).collision for point in path))

    def test_colliding_start_or_goal_is_rejected(self):
        data = np.zeros((60, 60), np.int8)
        data[20, 20] = 100
        planner = HybridAStarPlanner(GridMap(data, .1))
        with self.assertRaisesRegex(PlanningError, 'start'):
            planner.plan(Pose2D(2.05, 2.05, 0.), Pose2D(4., 4., 0.))
        with self.assertRaisesRegex(PlanningError, 'goal'):
            planner.plan(Pose2D(1., 1., 0.), Pose2D(2.05, 2.05, 0.))


if __name__ == '__main__':
    unittest.main()
