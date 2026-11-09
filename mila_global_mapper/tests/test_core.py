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
    make_camera_extrinsic,
    project_depth,
    se2_exp,
    se2_log,
)


def assert_pose_close(testcase: unittest.TestCase, actual: SE2, expected: SE2,
                      places: int = 9) -> None:
    testcase.assertAlmostEqual(actual.x, expected.x, places=places)
    testcase.assertAlmostEqual(actual.y, expected.y, places=places)
    angular_error = math.atan2(
        math.sin(actual.yaw - expected.yaw),
        math.cos(actual.yaw - expected.yaw),
    )
    testcase.assertAlmostEqual(angular_error, 0.0, places=places)


class SE2Tests(unittest.TestCase):
    def test_compose_inverse_and_point_transform(self) -> None:
        pose = SE2(1.2, -0.3, math.radians(35.0))
        assert_pose_close(self, pose.compose(pose.inverse()), SE2(0.0, 0.0, 0.0))
        assert_pose_close(self, pose.inverse().compose(pose), SE2(0.0, 0.0, 0.0))

        point = np.array([[0.4, -0.1]], dtype=np.float64)
        recovered = pose.inverse().transform_points(pose.transform_points(point))
        np.testing.assert_allclose(recovered, point, atol=1e-12)

    def test_exp_log_round_trip(self) -> None:
        for pose in (
            SE2(0.3, -0.7, 0.0),
            SE2(1.1, 0.4, 0.8),
            SE2(-0.2, 2.0, -1.3),
        ):
            assert_pose_close(self, se2_exp(se2_log(pose)), pose)


class ProjectionTests(unittest.TestCase):
    def test_optical_axes_match_ros_convention(self) -> None:
        transform = make_camera_extrinsic(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        rotation = transform[:3, :3]
        np.testing.assert_allclose(rotation @ [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(rotation @ [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(rotation @ [0.0, 1.0, 0.0], [0.0, 0.0, -1.0], atol=1e-12)

    def test_nominal_camera_look_directions_are_mirrored(self) -> None:
        right = make_camera_extrinsic(.2801, -.1178, .1087, -30.0, 10.0, 180.0)
        left = make_camera_extrinsic(.2801, .1178, .1087, 30.0, 10.0, 0.0)
        right_forward = right[:3, :3] @ np.array([0.0, 0.0, 1.0])
        left_forward = left[:3, :3] @ np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(right_forward, [0.8528685, -0.4924039, -0.1736482], atol=1e-6)
        np.testing.assert_allclose(left_forward, [0.8528685, 0.4924039, -0.1736482], atol=1e-6)

    def test_depth_projection_and_height_mask(self) -> None:
        intrinsics = CameraIntrinsics(3, 3, 1.0, 1.0, 1.0, 1.0)
        transform = make_camera_extrinsic(0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        depth = np.full((3, 3), np.nan, dtype=np.float32)
        depth[1, 1] = 2.0
        batch = project_depth(
            depth,
            intrinsics,
            transform,
            pixel_stride=1,
            min_depth_m=0.3,
            max_depth_m=5.0,
            min_obstacle_height_m=0.5,
            max_obstacle_height_m=1.5,
        )
        np.testing.assert_allclose(batch.origin_xy, [0.0, 0.0])
        np.testing.assert_allclose(batch.endpoints_xy, [[2.0, 0.0]], atol=1e-7)
        np.testing.assert_array_equal(batch.obstacle, [True])

        clearing_only = project_depth(
            depth,
            intrinsics,
            transform,
            pixel_stride=1,
            min_depth_m=0.3,
            max_depth_m=5.0,
            min_obstacle_height_m=1.1,
            max_obstacle_height_m=1.5,
        )
        np.testing.assert_array_equal(clearing_only.obstacle, [False])

    def test_projection_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            CameraIntrinsics(0, 3, 1.0, 1.0, 1.0, 1.0)
        intrinsics = CameraIntrinsics(3, 3, 1.0, 1.0, 1.0, 1.0)
        with self.assertRaises(ValueError):
            project_depth(
                np.ones((2, 3)), intrinsics, np.eye(4), pixel_stride=1,
                min_depth_m=.3, max_depth_m=5.0,
                min_obstacle_height_m=.1, max_obstacle_height_m=1.0,
            )
        with self.assertRaises(ValueError):
            RayBatch(np.zeros(2), np.zeros((2, 2)), np.zeros(1, dtype=bool))


class GridTests(unittest.TestCase):
    def test_bresenham_is_inclusive(self) -> None:
        self.assertEqual(
            list(DynamicLogOddsGrid.ray_cells(1, 2, 1, 5)),
            [(1, 2), (1, 3), (1, 4), (1, 5)],
        )
        self.assertEqual(
            list(DynamicLogOddsGrid.ray_cells(0, 0, 3, 3)),
            [(0, 0), (1, 1), (2, 2), (3, 3)],
        )

    def test_expansion_preserves_existing_cells(self) -> None:
        grid = DynamicLogOddsGrid(MapperConfig(
            resolution=1.0,
            initial_size_x_m=2.0,
            initial_size_y_m=2.0,
            expansion_cells=2,
        ))
        grid.log_odds[0, 0] = 1.25
        grid.observed[0, 0] = True
        old_world = np.array([[grid.origin_x + 0.5, grid.origin_y + 0.5]])
        grid.ensure_contains(np.array([[-3.0, -3.0], [4.0, 4.0]]))
        row, column = grid.world_to_grid(old_world)[0]
        self.assertTrue(grid.observed[row, column])
        self.assertAlmostEqual(float(grid.log_odds[row, column]), 1.25)
        self.assertGreater(grid.width, 2)
        self.assertGreater(grid.height, 2)

    def test_occupied_endpoint_wins_over_clearing_ray(self) -> None:
        config = MapperConfig(
            resolution=1.0,
            initial_size_x_m=8.0,
            initial_size_y_m=4.0,
            expansion_cells=2,
        )
        grid = DynamicLogOddsGrid(config)
        rays = RayBatch(
            np.array([0.0, 0.0]),
            np.array([[2.0, 0.0], [3.0, 0.0]]),
            np.array([True, False]),
        )
        stats = grid.integrate(SE2(0.0, 0.0, 0.0), [rays])
        endpoint = grid.world_to_grid(np.array([[2.0, 0.0]]))[0]
        self.assertAlmostEqual(
            float(grid.log_odds[endpoint[0], endpoint[1]]),
            grid.occupied_delta,
            places=6,
        )
        self.assertEqual(stats['occupied_cells'], 1)

    def test_invalid_mapper_config_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MapperConfig(resolution=0.0)
        with self.assertRaises(ValueError):
            MapperConfig(occupied_probability=0.4)
        with self.assertRaises(ValueError):
            MapperConfig(pgm_free_threshold=0.8, pgm_occupied_threshold=0.7)


class PersistenceAndClosureTests(unittest.TestCase):
    @staticmethod
    def make_ray() -> RayBatch:
        return RayBatch(
            np.array([0.0, 0.0]),
            np.array([[1.0, 0.0], [1.0, 0.5]], dtype=np.float32),
            np.array([True, True]),
        )

    def test_pgm_orientation_values_and_yaml(self) -> None:
        config = MapperConfig(
            resolution=1.0,
            initial_size_x_m=2.0,
            initial_size_y_m=2.0,
            expansion_cells=2,
        )
        core = OccupancyMappingCore(config)
        core.grid.observed[0, 0] = True
        core.grid.log_odds[0, 0] = config.log_odds_min
        core.grid.observed[1, 1] = True
        core.grid.log_odds[1, 1] = config.log_odds_max

        with tempfile.TemporaryDirectory() as directory:
            pgm_path, yaml_path = core.save(directory, 'room')
            data = pgm_path.read_bytes()
            marker = b'255\n'
            pixels = data[data.index(marker) + len(marker):]
            self.assertEqual(pixels, bytes((205, 0, 254, 205)))
            yaml_text = yaml_path.read_text(encoding='utf-8')
            self.assertIn('image: room.pgm', yaml_text)
            self.assertIn('resolution: 1', yaml_text)
            self.assertIn('origin: [-1, -1, 0.0]', yaml_text)

        with self.assertRaises(ValueError):
            core.save(Path(tempfile.gettempdir()), '../escape')

    def test_manual_endpoint_closure_rebuilds_and_finalizes(self) -> None:
        config = MapperConfig(
            resolution=.25,
            initial_size_x_m=5.0,
            initial_size_y_m=5.0,
            expansion_cells=4,
        )
        core = OccupancyMappingCore(config)
        ray = self.make_ray()
        poses = (
            SE2(0.0, 0.0, 0.0),
            SE2(1.0, 0.0, 0.1),
            SE2(1.0, 1.0, 0.1),
            SE2(0.2, 0.1, 0.08),
        )
        for index, pose in enumerate(poses, start=1):
            core.add_frame(index, pose, [ray])

        result = core.finish_loop()
        self.assertTrue(core.finalized)
        self.assertEqual(result['frames'], 4.0)
        assert_pose_close(self, core.frames[0].corrected_pose, poses[0])
        assert_pose_close(self, core.frames[-1].corrected_pose, poses[0])
        self.assertTrue(np.any(core.grid.observed))
        with self.assertRaises(RuntimeError):
            core.add_frame(5, SE2(0.0, 0.0, 0.0), [ray])

    def test_empty_save_and_bad_timestamps_are_rejected(self) -> None:
        core = OccupancyMappingCore()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                core.save(directory, 'empty')
        ray = self.make_ray()
        core.add_frame(10, SE2(0.0, 0.0, 0.0), [ray])
        with self.assertRaises(ValueError):
            core.add_frame(10, SE2(0.1, 0.0, 0.0), [ray])
        with self.assertRaises(ValueError):
            core.add_frame(9, SE2(0.1, 0.0, 0.0), [ray])


if __name__ == '__main__':
    unittest.main()
