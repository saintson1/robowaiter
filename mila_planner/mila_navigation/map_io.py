"""ROS-independent reader for standard trinary PGM/YAML occupancy maps."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


def _tokens(data: bytes):
    index = 0
    while index < len(data):
        while index < len(data) and chr(data[index]).isspace():
            index += 1
        if index < len(data) and data[index] == ord('#'):
            while index < len(data) and data[index] not in (10, 13):
                index += 1
            continue
        if index >= len(data):
            return
        start = index
        while (index < len(data) and not chr(data[index]).isspace()
               and data[index] != ord('#')):
            index += 1
        yield data[start:index], index


def read_pgm(path: Path) -> np.ndarray:
    data = path.read_bytes()
    iterator = iter(_tokens(data))
    try:
        magic, _ = next(iterator)
        width_raw, _ = next(iterator)
        height_raw, _ = next(iterator)
        maximum_raw, header_end = next(iterator)
    except StopIteration as error:
        raise ValueError(f'incomplete PGM header: {path}') from error
    if magic not in (b'P2', b'P5'):
        raise ValueError(f'unsupported PGM type {magic!r}')
    width, height, maximum = int(width_raw), int(height_raw), int(maximum_raw)
    if width <= 0 or height <= 0 or not 0 < maximum <= 65535:
        raise ValueError('invalid PGM dimensions/range')
    if magic == b'P2':
        values = [int(token) for token, _ in iterator]
        if len(values) != width * height:
            raise ValueError('incorrect P2 pixel count')
        image = np.asarray(values, dtype=np.float64).reshape(height, width)
    else:
        offset = header_end
        if data[offset:offset + 2] == b'\r\n':
            offset += 2
        elif offset < len(data) and chr(data[offset]).isspace():
            offset += 1
        else:
            raise ValueError('P5 header has no pixel-data separator')
        dtype = np.uint8 if maximum < 256 else np.dtype('>u2')
        count = width * height
        image = np.frombuffer(data, dtype=dtype, count=count, offset=offset)
        if image.size != count:
            raise ValueError('incorrect P5 pixel count')
        image = image.reshape(height, width).astype(np.float64)
    return image / maximum


def load_ros_map(yaml_path: Path):
    config = yaml.safe_load(yaml_path.read_text(encoding='utf-8'))
    required = ('image', 'resolution', 'origin', 'negate',
                'occupied_thresh', 'free_thresh')
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f'map YAML lacks {missing}')
    image_path = Path(config['image'])
    if not image_path.is_absolute():
        image_path = yaml_path.parent / image_path
    pixels = read_pgm(image_path.resolve())
    occupancy_probability = pixels if int(config['negate']) else 1.0 - pixels
    occupied = occupancy_probability >= float(config['occupied_thresh'])
    free = occupancy_probability <= float(config['free_thresh'])
    grid = np.full(pixels.shape, -1, dtype=np.int8)
    grid[free] = 0
    grid[occupied] = 100
    # Image row zero is the top; OccupancyGrid row zero is the map's lower edge.
    return np.flipud(grid), config

