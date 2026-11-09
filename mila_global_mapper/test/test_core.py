import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dual_rgbd_global_mapper.core import (
    CameraIntrinsics,
    DynamicLogOddsGrid,
    MapperConfig,
    OccupancyMappingCore,
    RayBatch,
    SE2,
    project_depth,
    se2_exp,
    se2_log,
)


class GeometryTests(unittest.TestCase):
    def test_se2_log_exp_round_trip(self):
        original = SE2(1.2, -0.7, 0.8)
        restored = se2_exp(se2_log(original))
        self.assertAlmostEqual(restored.x, original.x, places=9)
        self.assertAlmostEqual(restored.y, original.y, places=9)
        self.assertAlmostEqual(restored.yaw, original.yaw, places=9)

    def test_projection_marks_height_interval(self):
        depth = np.ones((2, 2), dtype=np.float32)
        intrinsics = CameraIntrinsics(2, 2, 1.0, 1.0, 0.5, 0.5)
        transform = np.eye(4)
        batch = project_depth(
            depth, intrinsics, transform, pixel_stride=1,
            min_depth_m=0.2, max_depth_m=2.0,
            min_obstacle_height_m=0.5, max_obstacle_height_m=1.5)
        self.assertEqual(len(batch.endpoints_xy), 4)
        self.assertTrue(np.all(batch.obstacle))


class GridTests(unittest.TestCase):
    def test_ray_marks_free_and_occupied(self):
        config = MapperConfig(resolution=1.0, initial_size_x_m=2.0,
                              initial_size_y_m=2.0, expansion_cells=2)
        grid = DynamicLogOddsGrid(config)
        batch = RayBatch(np.array([0.0, 0.0]), np.array([[2.0, 0.0]]), np.array([True]))
        stats = grid.integrate(SE2(0.0, 0.0, 0.0), [batch])
        self.assertEqual(stats['occupied_cells'], 1)
        endpoint = grid.world_to_grid(np.array([[2.0, 0.0]]))[0]
        self.assertGreater(grid.log_odds[tuple(endpoint)], 0.0)
        self.assertGreaterEqual(grid.width, 4)

    def test_manual_loop_closure_ends_at_start(self):
        core = OccupancyMappingCore(MapperConfig(resolution=0.2))
        rays = (RayBatch(np.array([0.0, 0.0]), np.array([[1.0, 0.0]]), np.array([True])),)
        core.add_frame(1, SE2(0.0, 0.0, 0.0), rays)
        core.add_frame(2, SE2(1.0, 0.0, 0.0), rays)
        core.add_frame(3, SE2(0.2, 0.1, math.radians(5.0)), rays)
        result = core.finish_loop(min_path_length_m=0.5)
        self.assertGreater(result['residual_translation_m'], 0.0)
        end = core.frames[-1].corrected_pose
        self.assertIsNotNone(end)
        self.assertAlmostEqual(end.x, 0.0)
        self.assertAlmostEqual(end.y, 0.0)
        self.assertAlmostEqual(end.yaw, 0.0)
        self.assertTrue(core.finalized)

    def test_save_writes_ros_pgm_and_yaml(self):
        core = OccupancyMappingCore(MapperConfig(resolution=0.1))
        rays = (RayBatch(np.array([0.0, 0.0]), np.array([[1.0, 0.0]]), np.array([True])),)
        core.add_frame(1, SE2(0.0, 0.0, 0.0), rays)
        with tempfile.TemporaryDirectory() as directory:
            pgm, yaml = core.save(directory, 'room')
            self.assertTrue(pgm.is_file())
            self.assertTrue(yaml.is_file())
            pgm_bytes = pgm.read_bytes()
            self.assertTrue(pgm_bytes.startswith(b'P5\n'))
            pixels = pgm_bytes.split(b'\n', 4)[4]
            self.assertIn(205, pixels)
            text = yaml.read_text(encoding='utf-8')
            self.assertIn('image: room.pgm', text)
            self.assertIn('mode: trinary', text)
            self.assertIn('free_thresh: 0.196', text)
            # map_server computes occupancy as (255-gray)/255. Gray 205
            # must remain between the YAML free and occupied thresholds.
            self.assertGreater((255.0 - 205.0) / 255.0, 0.196)
            self.assertLess((255.0 - 205.0) / 255.0, 0.65)

    def test_map_growth_is_bounded(self):
        config = MapperConfig(resolution=0.1, initial_size_x_m=1.0,
                              initial_size_y_m=1.0, max_size_x_m=2.0,
                              max_size_y_m=2.0, expansion_cells=2)
        grid = DynamicLogOddsGrid(config)
        with self.assertRaises(MemoryError):
            grid.ensure_contains(np.array([[10.0, 0.0]]))


if __name__ == '__main__':
    unittest.main()
