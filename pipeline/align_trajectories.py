"""Trajectory alignment for paired simulator recordings."""
from __future__ import annotations

import numpy as np
from scipy.signal import correlate

from schema.sensor_record import SensorRecord


def align_trajectories(
    gazebo: SensorRecord,
    isaac: SensorRecord,
) -> tuple[SensorRecord, SensorRecord]:
    g_times = gazebo.timestamps - gazebo.timestamps[0]
    i_times = isaac.timestamps - isaac.timestamps[0]

    offset_s = _detect_offset(
        g_times,
        np.linalg.norm(gazebo.positions, axis=1),
        i_times,
        np.linalg.norm(isaac.positions, axis=1),
    )
    i_times_aligned = i_times + offset_s

    t_start = max(float(g_times[0]), float(i_times_aligned[0]))
    t_end = min(float(g_times[-1]), float(i_times_aligned[-1]))
    overlap_s = t_end - t_start
    assert overlap_s >= 2.0, (
        f"Overlap region too short: {overlap_s:.2f}s. "
        f"Detected offset: {offset_s:.3f}s."
    )

    g_mask = (g_times >= t_start) & (g_times <= t_end)
    target_times = g_times[g_mask]

    gazebo_aligned = _resample_record(gazebo, g_times, target_times)
    isaac_aligned = _resample_record(isaac, i_times_aligned, target_times)

    assert len(gazebo_aligned.timestamps) == len(isaac_aligned.timestamps), (
        "Aligned records must have the same number of frames"
    )
    return gazebo_aligned, isaac_aligned


def compute_alignment_quality(
    gazebo_aligned: SensorRecord,
    isaac_aligned: SensorRecord,
) -> dict:
    diff = gazebo_aligned.positions - isaac_aligned.positions
    position_rmse_m = float(np.sqrt(np.mean(diff**2)))
    duration_s = float(gazebo_aligned.timestamps[-1] - gazebo_aligned.timestamps[0])

    if position_rmse_m < 0.2:
        quality = "GOOD"
    elif position_rmse_m < 0.5:
        quality = "ACCEPTABLE"
    else:
        quality = "POOR"

    return {
        "position_rmse_m": position_rmse_m,
        "overlap_duration_s": duration_s,
        "n_aligned_frames": len(gazebo_aligned.timestamps),
        "alignment_quality": quality,
    }


def _detect_offset(
    g_times: np.ndarray,
    g_signal: np.ndarray,
    i_times: np.ndarray,
    i_signal: np.ndarray,
    dt: float = 1.0 / 15.0,
) -> float:
    t_min = max(float(g_times[0]), float(i_times[0]))
    t_max = min(float(g_times[-1]), float(i_times[-1]))
    common_times = np.arange(t_min, t_max, dt)
    if len(common_times) < 2:
        return 0.0

    g_resampled = np.interp(common_times, g_times, g_signal)
    i_resampled = np.interp(common_times, i_times, i_signal)
    g_resampled = g_resampled - g_resampled.mean()
    i_resampled = i_resampled - i_resampled.mean()

    correlation = correlate(g_resampled, i_resampled, mode="full")
    lags = np.arange(-(len(i_resampled) - 1), len(g_resampled))
    best_lag = int(lags[np.argmax(correlation)])
    return float(best_lag * dt)


def _resample_record(
    record: SensorRecord,
    source_times: np.ndarray,
    target_times: np.ndarray,
) -> SensorRecord:
    nn_indices = _nearest_indices(source_times, target_times)
    lidar_scans = []
    if record.lidar_scans and len(record.lidar_scans) == len(source_times):
        lidar_scans = [record.lidar_scans[int(i)] for i in nn_indices]

    trust_state = None
    if record.trust_state is not None:
        trust_state = _interp_continuous(record.trust_state, source_times, target_times)

    resampled = SensorRecord(
        scenario_id=record.scenario_id,
        source=record.source,
        rgb_frames=record.rgb_frames[nn_indices],
        depth_frames=record.depth_frames[nn_indices],
        lidar_scans=lidar_scans,
        positions=_interp_continuous(record.positions, source_times, target_times),
        velocities=_interp_continuous(record.velocities, source_times, target_times),
        timestamps=(target_times - target_times[0]).astype(np.float64),
        trust_state=trust_state,
    )
    resampled.validate()
    return resampled


def _nearest_indices(source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    right = np.searchsorted(source_times, target_times, side="left")
    right = np.clip(right, 0, len(source_times) - 1)
    left = np.clip(right - 1, 0, len(source_times) - 1)
    choose_left = np.abs(target_times - source_times[left]) <= np.abs(
        source_times[right] - target_times
    )
    return np.where(choose_left, left, right).astype(int)


def _interp_continuous(
    values: np.ndarray,
    source_times: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    out = np.zeros((len(target_times), values.shape[1]), dtype=np.float32)
    for col in range(values.shape[1]):
        out[:, col] = np.interp(target_times, source_times, values[:, col]).astype(
            np.float32
        )
    return out
