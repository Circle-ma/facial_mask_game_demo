# coding: utf-8

import numpy as np
from . import Sim3DR_Cython


def get_normal(vertices, triangles):
    normal = np.zeros_like(vertices, dtype=np.float32)
    Sim3DR_Cython.get_normal(normal, vertices, triangles, vertices.shape[0], triangles.shape[0])
    return normal


def rasterize(vertices, triangles, colors, height=None, width=None):
    assert height is not None and width is not None
    bg = np.zeros((height, width, 3), dtype=np.uint8)

    buffer = np.zeros((height, width), dtype=np.float32) - 1e8

    if colors.dtype != np.float32:
        colors = colors.astype(np.float32)
    Sim3DR_Cython.rasterize(bg, vertices, triangles, colors, buffer, triangles.shape[0], height, width)

    depth = buffer.copy()
    buffer[buffer != -1e8] = 1
    buffer[buffer == -1e8] = 0
        
    buffer = buffer[::-1, :]
    return bg, buffer, depth
