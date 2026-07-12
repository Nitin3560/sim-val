"""Load Isaac Sim runner outputs into SensorRecord objects."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from schema.sensor_record import SensorRecord


def extract_isaac(output_dir: str) -> SensorRecord:
    directory = Path(output_dir)
    with (directory / "meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)

    timestamps = np.load(directory / "timestamps.npy").astype(np.float64)
    timestamps = timestamps - timestamps[0]

    lidar_path = directory / "lidar_scans.npy"
    lidar_scans = (
        list(np.load(lidar_path, allow_pickle=True)) if lidar_path.exists() else []
    )

    record = SensorRecord(
        scenario_id=meta["scenario_id"],
        source="isaac_sim",
        rgb_frames=np.load(directory / "rgb_frames.npy").astype(np.uint8),
        depth_frames=np.load(directory / "depth_frames.npy").astype(np.float32),
        lidar_scans=lidar_scans,
        positions=np.load(directory / "positions.npy").astype(np.float32),
        velocities=np.load(directory / "velocities.npy").astype(np.float32),
        timestamps=timestamps,
        trust_state=None,
    )
    record.validate()
    return record
