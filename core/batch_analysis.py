"""Production-run simulation + Hotelling's T2 multivariate SPC.

Simulates a batch of parts measured at shared canonical surface points, with
a tool-wear defect that kicks in partway through the run. Deviations get
bucketed into zones (k-means) rather than tracked per-point, then monitored
with a T2 control chart: Mahalanobis distance of each part's zone vector from
the baseline mean, using the baseline covariance. Fit T2 once from reference
(in-control) parts, then score everything against it (a single PC1 refit per
window looked tempting, but its sign flips run to run, so it kept flagging
healthy parts).
"""
from __future__ import annotations
import numpy as np
import trimesh
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree
from scipy.stats import f as f_dist

N_ZONES_DEFAULT = 10


def simulate_production_run(
    mesh: trimesh.Trimesh,
    n_parts: int = 40,
    n_points: int = 1200,
    base_noise_std: float = 0.05,
    drift_start_part: int = 15,
    drift_rate: float = 0.03,
    drift_radius: float = 15.0,
    seed: int = 0,
) -> dict:
    """
    Every part is measured at the same canonical surface points, so deviation
    vectors line up across parts. Parts after drift_start_part grow a radial
    defect within drift_radius of a feature point (picked automatically as
    the surface point farthest from centroid) on top of the baseline noise.
    """
    rng = np.random.default_rng(seed)
    canonical_points, face_idx = trimesh.sample.sample_surface(mesh, n_points)
    canonical_points = np.array(canonical_points)
    normals = np.array(mesh.face_normals[face_idx])

    centroid = mesh.centroid
    drift_center = canonical_points[np.argmax(np.linalg.norm(canonical_points - centroid, axis=1))]

    dist_to_center = np.linalg.norm(canonical_points - drift_center, axis=1)
    falloff = np.clip(1 - dist_to_center / drift_radius, 0, 1) ** 2

    deviations = np.zeros((n_parts, n_points))
    for k in range(n_parts):
        noise = rng.normal(0, base_noise_std, size=(n_points, 3))
        wear_progress = max(0, k - drift_start_part) * drift_rate
        radial_offset = (falloff * wear_progress)[:, None] * normals
        part_points = canonical_points + noise + radial_offset
        _, dist, _ = trimesh.proximity.closest_point(mesh, part_points)
        deviations[k] = dist

    # aggregate into zones for SPC (k-means on canonical points, not deviations)
    n_zones = min(N_ZONES_DEFAULT, max(n_parts // 4, 4))
    zone_centers, zone_ids = kmeans2(canonical_points, n_zones, seed=seed, minit="++")

    return {
        "canonical_points": canonical_points,
        "normals": normals,
        "deviations": deviations,
        "drift_center": drift_center,
        "falloff": falloff,
        "zone_ids": zone_ids,
        "n_zones": n_zones,
    }


def pca_drift_analysis(
    deviations: np.ndarray,
    zone_ids: np.ndarray,
    n_zones: int,
    n_baseline: int = 15,
    alpha: float = 0.0027,
) -> dict:
    """
    Hotelling's T2 control chart over zone-aggregated deviation features.

    Point-level deviations get averaged into per-zone features first, since
    thousands of raw points against a few dozen parts is a small-n-large-p
    setup where any "dominant direction" just fits noise, not signal.

    First n_baseline parts define the in-control mean/covariance (needs
    n_baseline > n_zones so the covariance is invertible). Every part gets
    scored by Mahalanobis distance from that baseline, checked against the
    F-distribution control limit (Tracy/Young/Mason).
    """
    if n_baseline <= n_zones:
        raise ValueError(f"n_baseline ({n_baseline}) must exceed n_zones ({n_zones}) for a stable covariance estimate")

    n_parts = deviations.shape[0]
    zone_features = np.zeros((n_parts, n_zones))
    for z in range(n_zones):
        mask = zone_ids == z
        zone_features[:, z] = deviations[:, mask].mean(axis=1) if mask.any() else 0.0

    # phase 1: baseline mean/covariance from first n_baseline parts
    baseline = zone_features[:n_baseline]
    mean = baseline.mean(axis=0)
    cov = np.cov(baseline, rowvar=False)
    cov += np.eye(n_zones) * 1e-9 
    # phase 2: score all parts against that baseline
    inv_cov = np.linalg.inv(cov)

    diffs = zone_features - mean
    t2 = np.einsum("ij,jk,ik->i", diffs, inv_cov, diffs)

    p, n = n_zones, n_baseline
    f_crit = f_dist.ppf(1 - alpha, p, n - p)
    ucl = (p * (n + 1) * (n - 1)) / (n * (n - p)) * f_crit

    violations = np.where(t2 > ucl)[0]
    violations = violations[violations >= n_baseline]
    first_violation = int(violations[0]) if len(violations) else None

    eigvals = np.linalg.eigvalsh(cov)[::-1]
    explained_variance_ratio = (eigvals / eigvals.sum())[: min(5, n_zones)].tolist()

    # zone-level sigmas for the flagged (or worst) part, for the mesh overlay
    contrib_part = first_violation if first_violation is not None else int(np.argmax(t2))
    zone_std = np.sqrt(np.diag(cov))
    contributions = diffs[contrib_part] / (zone_std + 1e-9)
    point_loading = contributions[zone_ids]

    return {
        "baseline_mean": mean,
        "t2": t2,
        "ucl": float(ucl),
        "explained_variance_ratio": explained_variance_ratio,
        "first_violation_part": first_violation,
        "contrib_part": contrib_part,
        "contributions": contributions,
        "point_loading": point_loading,
    }


def project_tolerance_crossing(
    part_indices: np.ndarray, max_deviation_per_part: np.ndarray, tolerance: float, fit_from_part: int
) -> int | None:
    """Linear-fit the post-drift trend and project when it crosses `tolerance`."""
    mask = part_indices >= fit_from_part
    if mask.sum() < 3:
        return None
    slope, intercept = np.polyfit(part_indices[mask], max_deviation_per_part[mask], 1)
    if slope <= 0:
        return None
    crossing = (tolerance - intercept) / slope
    return int(np.ceil(crossing)) if crossing > part_indices[mask][0] else None


def colorize_mesh_by_mode(mesh: trimesh.Trimesh, canonical_points: np.ndarray, loading: np.ndarray) -> str:
    """Diverging-colormap GLB of the signed loading: blue = pulls deviation down, red = up."""
    import tempfile

    vertices = np.array(mesh.vertices)
    tree = cKDTree(canonical_points)
    k = min(5, len(canonical_points))
    dists, idxs = tree.query(vertices, k=k)

    if k == 1:
        vertex_loading = loading[idxs]
    else:
        weights = 1.0 / (dists + 1e-9)
        weights /= weights.sum(axis=1, keepdims=True)
        vertex_loading = (weights * loading[idxs]).sum(axis=1)

    scale = np.max(np.abs(vertex_loading)) + 1e-9
    ratio = np.clip(vertex_loading / scale, -1, 1)  # [-1, 1]
    r = np.where(ratio > 0, ratio, 0)
    b = np.where(ratio < 0, -ratio, 0)
    g = 1 - np.abs(ratio)
    vertex_colors = np.stack([r + g, g, b + g], axis=1)
    vertex_colors = np.clip(vertex_colors, 0, 1)

    colored_mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.array(mesh.faces),
        vertex_colors=(vertex_colors * 255).astype(np.uint8),
    )

    tmp = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
    tmp.close()
    colored_mesh.export(tmp.name)
    return tmp.name
