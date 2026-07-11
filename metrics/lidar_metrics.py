"""LiDAR fidelity metrics for SimVal."""
from __future__ import annotations

from typing import Any

import numpy as np


def _chamfer_distance(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diff = p[:, None, :] - q[None, :, :]
    dist2 = (diff ** 2).sum(axis=-1)
    return dist2.min(axis=1), dist2.min(axis=0)


def _subsample(points: np.ndarray, n_subsample: int, seed: int = 42) -> np.ndarray:
    if len(points) <= n_subsample:
        return points.astype(np.float32)
    rng = np.random.default_rng(seed=seed)
    idx = rng.choice(len(points), n_subsample, replace=False)
    return points[idx].astype(np.float32)


def compute_dcd(
    reference: np.ndarray,
    query: np.ndarray,
    alpha: float = 1000.0,
    n_subsample: int = 2048,
) -> float:
    p = _subsample(reference, n_subsample)
    q = _subsample(query, n_subsample)

    if len(p) == 0 or len(q) == 0:
        return 1.0

    eps = 1e-8
    all_points = np.concatenate([p, q], axis=0)
    bbox_diag = float(np.linalg.norm(all_points.max(axis=0) - all_points.min(axis=0)))
    if bbox_diag < eps:
        return 0.0

    d_pq, d_qp = _chamfer_distance(p, q)
    chamfer = 0.5 * (float(np.mean(d_pq)) + float(np.mean(d_qp)))
    normalized_geometry = chamfer / (bbox_diag ** 2 + eps)
    density_penalty = abs(len(p) - len(q)) / max(len(p), len(q), 1)
    local_weight = 1.0 - float(np.exp(-alpha * normalized_geometry))
    dcd = normalized_geometry * (1.0 + density_penalty + local_weight)
    return float(np.clip(dcd, 0.0, 1.0))


def compute_dcd_sequence(
    reference_scans: list,
    query_scans: list,
    alpha: float = 1000.0,
    n_subsample: int = 2048,
) -> dict[str, Any]:
    assert len(reference_scans) == len(query_scans), (
        f"Scan count mismatch: {len(reference_scans)} vs {len(query_scans)}"
    )

    scores = []
    for ref, qry in zip(reference_scans, query_scans):
        if ref is None or qry is None or len(ref) == 0 or len(qry) == 0:
            continue
        scores.append(compute_dcd(ref, qry, alpha=alpha, n_subsample=n_subsample))

    if not scores:
        return {
            "dcd_mean": None,
            "dcd_std": None,
            "dcd_max": None,
            "dcd_per_scan": [],
            "interpretation": "no valid scan pairs",
        }

    mean = float(np.mean(scores))
    if mean < 0.1:
        interpretation = "high fidelity - LiDAR geometry closely matches"
    elif mean < 0.3:
        interpretation = "moderate gap - geometric or density difference"
    else:
        interpretation = "significant gap - LiDAR behavior differs substantially"

    return {
        "dcd_mean": mean,
        "dcd_std": float(np.std(scores)),
        "dcd_max": float(np.max(scores)),
        "dcd_per_scan": scores,
        "interpretation": interpretation,
    }


def compute_point_density_gap(reference_scans: list, query_scans: list) -> dict[str, Any]:
    ref_counts = [len(s) for s in reference_scans if s is not None and len(s) > 0]
    query_counts = [len(s) for s in query_scans if s is not None and len(s) > 0]

    if not ref_counts or not query_counts:
        return {
            "gap_fraction": None,
            "gazebo_mean_count": None,
            "isaac_mean_count": None,
            "interpretation": "no valid scans",
        }

    gazebo_mean = float(np.mean(ref_counts))
    isaac_mean = float(np.mean(query_counts))
    gap = float((gazebo_mean - isaac_mean) / max(isaac_mean, 1.0))

    direction = "more" if gap > 0 else "fewer"
    return {
        "gap_fraction": gap,
        "gazebo_mean_count": gazebo_mean,
        "isaac_mean_count": isaac_mean,
        "interpretation": (
            f"Gazebo returns {abs(gap):.1%} {direction} points per scan than Isaac Sim RTX"
        ),
    }


def compute_all_lidar_metrics(reference_scans: list, query_scans: list) -> dict[str, Any]:
    result = compute_dcd_sequence(reference_scans, query_scans)
    density = compute_point_density_gap(reference_scans, query_scans)
    result.update(
        {
            "density_gap_fraction": density["gap_fraction"],
            "gazebo_mean_count": density["gazebo_mean_count"],
            "isaac_mean_count": density["isaac_mean_count"],
            "density_interpretation": density["interpretation"],
        }
    )
    return result
