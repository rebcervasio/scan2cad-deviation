"""Synthetic scan generator: samples a mesh surface and adds realistic noise."""
from __future__ import annotations
import numpy as np
import trimesh


def simulate_scan(
    mesh: trimesh.Trimesh,
    n_points: int = 8000,
    noise_std: float = 0.5,
    rigid_translation: np.ndarray = None,
    rigid_rotation_deg: float = 0.0,
) -> np.ndarray:
    """Sample the mesh surface and add Gaussian noise plus an optional rigid offset."""
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    points = points + np.random.normal(0, noise_std, points.shape)

    if rigid_translation is not None:
        points = points + rigid_translation

    if rigid_rotation_deg != 0.0:
        angle = np.deg2rad(rigid_rotation_deg)
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        centroid = points.mean(axis=0)
        points = (R @ (points - centroid).T).T + centroid

    return points
