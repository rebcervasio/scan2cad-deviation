"""ICP-based point cloud to mesh registration using scipy (no open3d dependency for registration)."""
from __future__ import annotations
import numpy as np
import trimesh
from scipy.spatial import cKDTree


def _rigid_transform_svd(src: np.ndarray, tgt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # least-squares rigid alignment (Kabsch/Umeyama), reflection-corrected
    src_c = src.mean(axis=0)
    tgt_c = tgt.mean(axis=0)
    A = (src - src_c).T @ (tgt - tgt_c)
    U, _, Vt = np.linalg.svd(A)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = tgt_c - R @ src_c
    return R, t


def align_point_cloud_to_mesh(
    pcd_points: np.ndarray,
    mesh: trimesh.Trimesh,
    max_iter: int = 50,
    n_ref: int = 10000,
    tol: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Point-to-point ICP. Returns (aligned_points, 4x4 transform, fitness)."""
    ref_pts, _ = trimesh.sample.sample_surface(mesh, n_ref)
    tree = cKDTree(ref_pts)

    src = pcd_points.copy()
    T = np.eye(4)

    prev_mean = np.inf
    for _ in range(max_iter):
        distances, indices = tree.query(src, k=1)
        matched = ref_pts[indices]

        # Filter outlier correspondences (> 3× median distance)
        med = np.median(distances)
        mask = distances < 3 * med + 1e-9
        if mask.sum() < 6:
            break

        R, t = _rigid_transform_svd(src[mask], matched[mask])
        if not np.isfinite(R).all():
            break

        src = (R @ src.T).T + t

        step = np.eye(4)
        step[:3, :3] = R
        step[:3, 3] = t
        T = step @ T

        mean_dist = distances[mask].mean()
        if abs(prev_mean - mean_dist) < tol:
            break
        prev_mean = mean_dist

    fitness = float(np.mean(distances < distances.mean() * 2 + 1e-9))
    return src, T, fitness
