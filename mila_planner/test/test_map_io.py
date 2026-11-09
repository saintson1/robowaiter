from pathlib import Path
import tempfile
import unittest

import numpy as np

from mila_navigation.map_io import load_ros_map, read_pgm


class MapIoTests(unittest.TestCase):
    def test_binary_pgm_preserves_leading_whitespace_pixel(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.pgm'
            path.write_bytes(b'P5\n2 1\n255\n' + bytes((10, 254)))
            np.testing.assert_allclose(read_pgm(path), [[10 / 255, 254 / 255]])

    def test_real_saved_map_loads_as_trinary_grid(self):
        path = Path(__file__).resolve().parents[2] / 'maps' / 'global_map.yaml'
        if not path.exists():
            self.skipTest('saved map is not included in the source presentation')
        grid, config = load_ros_map(path)
        self.assertEqual(grid.shape, (224, 268))
        self.assertEqual(set(np.unique(grid)).difference({-1, 0, 100}), set())
        self.assertEqual(float(config['resolution']), 0.05)


if __name__ == '__main__':
    unittest.main()
