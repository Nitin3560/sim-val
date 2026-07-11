from __future__ import annotations

import numpy as np
import pytest

from metrics.lidar_metrics import (
    compute_all_lidar_metrics,
    compute_dcd,
    compute_dcd_sequence,
    compute_point_density_gap,
)


def make_scan(
    n_points: int = 5000,
    seed: int = 0,
    scale: float = 1.0,
    offset: float = 0.0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points = rng.standard_normal((n_points, 3)).astype(np.float32)
    return points * scale + offset


def make_scans(n_scans: int = 10, n_points: int = 5000, seed: int = 0) -> list:
    return [make_scan(n_points=n_points, seed=seed + i) for i in range(n_scans)]


class TestDCD:
    def test_identical_clouds_give_zero_dcd(self):
        scan = make_scan(seed=0)
        result = compute_dcd(scan, scan.copy())
        assert result < 0.01

    def test_different_clouds_give_nonzero_dcd(self):
        result = compute_dcd(make_scan(seed=0), make_scan(seed=1))
        assert result > 0.0

    def test_dcd_stays_in_valid_range(self):
        result = compute_dcd(make_scan(seed=0), make_scan(seed=1))
        assert 0.0 <= result <= 1.0

    def test_larger_offset_gives_higher_dcd(self):
        ref = make_scan(seed=0)
        small_offset = compute_dcd(ref, make_scan(seed=0, offset=0.1))
        large_offset = compute_dcd(ref, make_scan(seed=0, offset=2.0))
        assert large_offset > small_offset

    def test_dcd_detects_density_difference(self):
        ref = make_scan(n_points=5000, seed=0)
        full = make_scan(n_points=5000, seed=0)
        sparse = make_scan(n_points=1000, seed=0)
        dcd_full = compute_dcd(ref, full)
        dcd_sparse = compute_dcd(ref, sparse)
        assert dcd_sparse > dcd_full


class TestDCDSequence:
    def test_identical_sequences_give_near_zero(self):
        scans = make_scans(seed=0)
        result = compute_dcd_sequence(scans, [s.copy() for s in scans])
        assert result["dcd_mean"] < 0.01

    def test_different_sequences_give_nonzero(self):
        result = compute_dcd_sequence(make_scans(seed=0), make_scans(seed=99))
        assert result["dcd_mean"] > 0.0

    def test_output_keys_complete(self):
        result = compute_dcd_sequence(make_scans(seed=0), make_scans(seed=1))
        for key in ["dcd_mean", "dcd_std", "dcd_max", "dcd_per_scan", "interpretation"]:
            assert key in result

    def test_scan_count_mismatch_raises(self):
        with pytest.raises(AssertionError):
            compute_dcd_sequence(make_scans(n_scans=5), make_scans(n_scans=8))

    def test_interpretation_present_and_nonempty(self):
        result = compute_dcd_sequence(make_scans(seed=0), make_scans(seed=1))
        assert isinstance(result["interpretation"], str)
        assert len(result["interpretation"]) > 0


class TestDensityGap:
    def test_same_density_gives_near_zero_gap(self):
        result = compute_point_density_gap(
            make_scans(n_points=5000),
            make_scans(n_points=5000),
        )
        assert abs(result["gap_fraction"]) < 0.01

    def test_gazebo_denser_gives_positive_gap(self):
        gazebo = make_scans(n_points=15000)
        isaac = make_scans(n_points=11000)
        result = compute_point_density_gap(gazebo, isaac)
        assert result["gap_fraction"] > 0.0
        assert "more" in result["interpretation"]

    def test_output_keys_complete(self):
        result = compute_point_density_gap(make_scans(), make_scans())
        for key in ["gap_fraction", "gazebo_mean_count", "isaac_mean_count", "interpretation"]:
            assert key in result


class TestAllLidarMetrics:
    def test_wrapper_returns_all_required_keys(self):
        result = compute_all_lidar_metrics(make_scans(seed=0), make_scans(seed=1))
        for key in ["dcd_mean", "density_gap_fraction", "gazebo_mean_count", "isaac_mean_count"]:
            assert key in result

    def test_wrapper_identical_inputs_makes_sense(self):
        scans = make_scans(seed=0)
        result = compute_all_lidar_metrics(scans, [s.copy() for s in scans])
        assert result["dcd_mean"] < 0.01
        assert abs(result["density_gap_fraction"]) < 0.01
