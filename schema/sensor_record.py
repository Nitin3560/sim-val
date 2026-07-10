"""Shared sensor data schema for Gazebo and Isaac Sim records."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class SensorRecord:
    """Time-aligned sensor data from one scenario recording."""
    scenario_id: str
    source: str

    rgb_frames: np.ndarray
    depth_frames: np.ndarray
    lidar_scans: list

    positions: np.ndarray
    velocities: np.ndarray
    timestamps: np.ndarray

    trust_state: Optional[np.ndarray] = None

    def validate(self) -> None:
        """Validate array shapes and source-specific fields."""
        N = len(self.timestamps)

        if N == 0:
            raise ValueError("SensorRecord has zero frames")

        if self.source not in ("gazebo", "isaac_sim"):
            raise ValueError(
                f"source must be 'gazebo' or 'isaac_sim', got '{self.source}'"
            )

        if self.rgb_frames.shape[0] != N:
            raise AssertionError(
                f"rgb_frames first dim {self.rgb_frames.shape[0]} != N={N}"
            )

        if len(self.rgb_frames.shape) != 4 or self.rgb_frames.shape[3] != 3:
            raise AssertionError(
                f"rgb_frames must be (N,H,W,3), got {self.rgb_frames.shape}"
            )

        if self.depth_frames.shape[0] != N:
            raise AssertionError(
                f"depth_frames first dim {self.depth_frames.shape[0]} != N={N}"
            )

        if len(self.depth_frames.shape) != 3:
            raise AssertionError(
                f"depth_frames must be (N,H,W), got {self.depth_frames.shape}"
            )

        if self.positions.shape != (N, 3):
            raise AssertionError(
                f"positions must be (N,3), got {self.positions.shape}"
            )

        if self.velocities.shape != (N, 3):
            raise AssertionError(
                f"velocities must be (N,3), got {self.velocities.shape}"
            )

        if self.trust_state is not None:
            if self.trust_state.shape != (N, 3):
                raise AssertionError(
                    f"trust_state must be (N,3), got {self.trust_state.shape}"
                )
            if self.source == "isaac_sim":
                raise AssertionError(
                    "trust_state should be None for isaac_sim records - "
                    "TwinGuard does not run inside Isaac Sim"
                )

        # Empty lidar_scans is allowed for camera-only records.
        if self.lidar_scans and len(self.lidar_scans) != N:
            raise AssertionError(
                f"lidar_scans length {len(self.lidar_scans)} != N={N}"
            )

    def save(self, output_dir: str) -> None:
        """Save arrays and metadata to disk."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        np.save(out / "rgb_frames.npy", self.rgb_frames)
        np.save(out / "depth_frames.npy", self.depth_frames)
        np.save(
            out / "lidar_scans.npy",
            np.array(self.lidar_scans, dtype=object),
            allow_pickle=True,
        )
        np.save(out / "positions.npy", self.positions)
        np.save(out / "velocities.npy", self.velocities)
        np.save(out / "timestamps.npy", self.timestamps)

        if self.trust_state is not None:
            np.save(out / "trust_state.npy", self.trust_state)

        meta = {
            "scenario_id": self.scenario_id,
            "source": self.source,
            "n_frames": int(len(self.timestamps)),
            "rgb_shape": list(self.rgb_frames.shape),
            "depth_shape": list(self.depth_frames.shape),
            "duration_s": float(
                self.timestamps[-1] - self.timestamps[0]
                if len(self.timestamps) > 1 else 0.0
            ),
            "has_lidar": len(self.lidar_scans) > 0,
            "has_trust_state": self.trust_state is not None,
        }
        with (out / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, input_dir: str) -> SensorRecord:
        """Load a SensorRecord from disk."""
        d = Path(input_dir)
        with (d / "meta.json").open("r", encoding="utf-8") as f:
            meta = json.load(f)

        trust_path = d / "trust_state.npy"
        trust_state = np.load(trust_path) if trust_path.exists() else None

        lidar_path = d / "lidar_scans.npy"
        lidar_scans = (
            list(np.load(lidar_path, allow_pickle=True))
            if lidar_path.exists()
            else []
        )

        return cls(
            scenario_id=meta["scenario_id"],
            source=meta["source"],
            rgb_frames=np.load(d / "rgb_frames.npy"),
            depth_frames=np.load(d / "depth_frames.npy"),
            lidar_scans=lidar_scans,
            positions=np.load(d / "positions.npy"),
            velocities=np.load(d / "velocities.npy"),
            timestamps=np.load(d / "timestamps.npy"),
            trust_state=trust_state,
        )
