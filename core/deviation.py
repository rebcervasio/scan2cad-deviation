"""Point-to-mesh deviation computation and mesh colorization."""
from __future__ import annotations
import numpy as np
import trimesh
import tempfile


def compute_deviation(points: np.ndarray, mesh: trimesh.Trimesh) -> np.ndarray:
    """Unsigned distance from each scan point to the nearest mesh surface, in mesh units."""
    _, distances, _ = trimesh.proximity.closest_point(mesh, points)
    return distances


def _distance_to_rgb(distances: np.ndarray, tolerance: float) -> np.ndarray:
    # green at 0, red at >= tolerance
    ratio = np.clip(distances / tolerance, 0, 1)
    r = np.clip(2 * ratio, 0, 1)
    g = np.clip(2 * (1 - ratio), 0, 1)
    b = np.zeros_like(ratio)
    return np.stack([r, g, b], axis=1)


def colorize_mesh_by_deviation(
    points: np.ndarray,
    mesh: trimesh.Trimesh,
    distances: np.ndarray,
    tolerance: float,
) -> str:
    """Interpolate point deviations onto mesh vertices, export a colored GLB, return its path."""
    from scipy.spatial import cKDTree

    point_colors = _distance_to_rgb(distances, tolerance)
    vertices = np.array(mesh.vertices)

    # Interpolate point colors onto mesh vertices using inverse-distance weighting
    tree = cKDTree(points)
    k = min(5, len(points))
    dists, idxs = tree.query(vertices, k=k)

    if k == 1:
        vertex_colors = point_colors[idxs]
    else:
        weights = 1.0 / (dists + 1e-9)
        weights /= weights.sum(axis=1, keepdims=True)
        vertex_colors = (weights[:, :, None] * point_colors[idxs]).sum(axis=1)

    colored_mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.array(mesh.faces),
        vertex_colors=(vertex_colors * 255).astype(np.uint8),
    )

    tmp = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
    tmp.close()
    colored_mesh.export(tmp.name)
    return tmp.name


def compute_statistics(distances: np.ndarray, tolerance: float) -> dict:
    return {
        "max_deviation_mm": round(float(distances.max()), 4),
        "mean_deviation_mm": round(float(distances.mean()), 4),
        "rms_deviation_mm": round(float(np.sqrt(np.mean(distances**2))), 4),
        "median_deviation_mm": round(float(np.median(distances)), 4),
        "pct_within_tolerance": round(float(np.mean(distances <= tolerance) * 100), 1),
        "tolerance_mm": tolerance,
        "n_scan_points": len(distances),
    }
