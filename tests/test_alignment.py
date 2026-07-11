from __future__ import annotations

import numpy as np
import pytest

from pipeline.align_trajectories import (
    _detect_offset,
    align_trajectories,
    compute_alignment_quality,
)
from schema.sensor_record import SensorRecord


N, H, W = 60, 48, 64


def make_trajectory_record(
    offset_s: float = 0.0,
    source: str = "gazebo",
    n_frames: int = N,
    seed: int = 0,
) -> SensorRecord:
    rng = np.random.default_rng(seed)
    local_t = np.linspace(0.0, 20.0, n_frames)
    trajectory_t = local_t + offset_s
    x = 2.0 * np.sin(2.0 * np.pi * trajectory_t / 20.0)
    y = 1.0 * np.sin(4.0 * np.pi * trajectory_t / 20.0 + 0.3)
    positions = np.column_stack([x, y, -3.0 * np.ones(n_frames)]).astype(np.float32)
    velocities = np.gradient(positions, local_t, axis=0).astype(np.float32)

    return SensorRecord(
        scenario_id="test",
        source=source,
        rgb_frames=rng.integers(0, 255, (n_frames, H, W, 3), dtype=np.uint8),
        depth_frames=rng.uniform(0.5, 5.0, (n_frames, H, W)).astype(np.float32),
        lidar_scans=[],
        positions=positions,
        velocities=velocities,
        timestamps=local_t.astype(np.float64),
        trust_state=(
            rng.uniform(0.0, 1.0, (n_frames, 3)).astype(np.float32)
            if source == "gazebo"
            else None
        ),
    )


class TestDetectOffset:
    def test_zero_offset_detected_correctly(self):
        t = np.linspace(0.0, 20.0, 300)
        sig = np.sin(2.0 * np.pi * t / 20.0)
        offset = _detect_offset(t, sig, t, sig)
        assert abs(offset) < 0.5

    def test_known_offset_recovered(self):
        t = np.linspace(0.0, 20.0, 300)
        gazebo_signal = np.sin(2.0 * np.pi * t / 20.0)
        isaac_signal = np.sin(2.0 * np.pi * (t + 2.0) / 20.0)
        offset = _detect_offset(t, gazebo_signal, t, isaac_signal)
        assert abs(offset - 2.0) < 0.5


class TestAlignTrajectories:
    def test_no_offset_alignment_trivial(self):
        g = make_trajectory_record(offset_s=0.0, source="gazebo")
        i = make_trajectory_record(offset_s=0.0, source="isaac_sim")
        g_aligned, i_aligned = align_trajectories(g, i)
        quality = compute_alignment_quality(g_aligned, i_aligned)
        assert quality["position_rmse_m"] < 0.2
        assert quality["alignment_quality"] in ("GOOD", "ACCEPTABLE")

    def test_offset_alignment_recovers(self):
        g = make_trajectory_record(offset_s=0.0, source="gazebo")
        i = make_trajectory_record(offset_s=2.0, source="isaac_sim")
        g_aligned, i_aligned = align_trajectories(g, i)
        quality = compute_alignment_quality(g_aligned, i_aligned)
        assert quality["position_rmse_m"] < 0.5
        assert quality["alignment_quality"] in ("GOOD", "ACCEPTABLE")

    def test_aligned_records_same_length(self):
        g = make_trajectory_record(offset_s=0.0, source="gazebo")
        i = make_trajectory_record(offset_s=1.0, source="isaac_sim")
        g_aligned, i_aligned = align_trajectories(g, i)
        assert len(g_aligned.timestamps) == len(i_aligned.timestamps)
        assert g_aligned.rgb_frames.shape[0] == i_aligned.rgb_frames.shape[0]

    def test_aligned_records_validate(self):
        g = make_trajectory_record(offset_s=0.0, source="gazebo")
        i = make_trajectory_record(offset_s=1.0, source="isaac_sim")
        g_aligned, i_aligned = align_trajectories(g, i)
        g_aligned.validate()
        i_aligned.validate()

    def test_timestamps_start_at_zero(self):
        g = make_trajectory_record(offset_s=0.0, source="gazebo")
        i = make_trajectory_record(offset_s=1.0, source="isaac_sim")
        g_aligned, i_aligned = align_trajectories(g, i)
        assert g_aligned.timestamps[0] < 1.0
        assert i_aligned.timestamps[0] < 1.0

    def test_trust_state_preserved_for_gazebo(self):
        g = make_trajectory_record(offset_s=0.0, source="gazebo")
        i = make_trajectory_record(offset_s=0.0, source="isaac_sim")
        g_aligned, i_aligned = align_trajectories(g, i)
        assert g_aligned.trust_state is not None
        assert i_aligned.trust_state is None

    def test_short_overlap_raises(self):
        g = make_trajectory_record(offset_s=0.0, source="gazebo", n_frames=30)
        i = make_trajectory_record(offset_s=0.0, source="isaac_sim", n_frames=30)
        i.timestamps = np.linspace(0.0, 1.0, 30)
        with pytest.raises(AssertionError, match="Overlap region too short"):
            align_trajectories(g, i)


class TestAlignmentQuality:
    def test_quality_keys_present(self):
        g = make_trajectory_record(source="gazebo")
        i = make_trajectory_record(source="isaac_sim")
        result = compute_alignment_quality(*align_trajectories(g, i))
        for key in [
            "position_rmse_m",
            "overlap_duration_s",
            "n_aligned_frames",
            "alignment_quality",
        ]:
            assert key in result

    def test_quality_label_valid(self):
        g = make_trajectory_record(source="gazebo")
        i = make_trajectory_record(source="isaac_sim")
        result = compute_alignment_quality(*align_trajectories(g, i))
        assert result["alignment_quality"] in ("GOOD", "ACCEPTABLE", "POOR")

    def test_overlap_duration_positive(self):
        g = make_trajectory_record(source="gazebo")
        i = make_trajectory_record(source="isaac_sim")
        result = compute_alignment_quality(*align_trajectories(g, i))
        assert result["overlap_duration_s"] > 2.0

    def test_identical_records_give_good_quality(self):
        g = make_trajectory_record(offset_s=0.0, source="gazebo")
        i = make_trajectory_record(offset_s=0.0, source="isaac_sim")
        result = compute_alignment_quality(*align_trajectories(g, i))
        assert result["alignment_quality"] == "GOOD"
        assert result["position_rmse_m"] < 0.2
