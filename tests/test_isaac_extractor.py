from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from isaac.extract_isaac import extract_isaac


def write_fake_isaac_output(tmp_dir: str, n_frames: int = 20) -> None:
    directory = Path(tmp_dir)
    rng = np.random.default_rng(42)

    np.save(
        directory / "rgb_frames.npy",
        rng.integers(0, 255, (n_frames, 48, 64, 3), dtype=np.uint8),
    )
    np.save(
        directory / "depth_frames.npy",
        rng.uniform(0.5, 20.0, (n_frames, 48, 64)).astype(np.float32),
    )
    np.save(
        directory / "lidar_scans.npy",
        np.array(
            [rng.standard_normal((5000, 3)).astype(np.float32) for _ in range(n_frames)],
            dtype=object,
        ),
        allow_pickle=True,
    )
    np.save(directory / "positions.npy", rng.standard_normal((n_frames, 3)).astype(np.float32))
    np.save(directory / "velocities.npy", np.zeros((n_frames, 3), dtype=np.float32))
    np.save(directory / "timestamps.npy", np.linspace(2.0, 22.0, n_frames))

    with (directory / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"scenario_id": "hover_3m_nominal", "source": "isaac_sim", "n_frames": n_frames},
            f,
        )


class TestExtractIsaac:
    def test_produces_valid_record(self, tmp_path):
        write_fake_isaac_output(str(tmp_path))
        record = extract_isaac(str(tmp_path))
        record.validate()
        assert record.source == "isaac_sim"

    def test_trust_state_always_none(self, tmp_path):
        write_fake_isaac_output(str(tmp_path))
        record = extract_isaac(str(tmp_path))
        assert record.trust_state is None

    def test_timestamps_start_at_zero(self, tmp_path):
        write_fake_isaac_output(str(tmp_path))
        record = extract_isaac(str(tmp_path))
        assert abs(record.timestamps[0]) < 1e-6

    def test_rgb_dtype_uint8(self, tmp_path):
        write_fake_isaac_output(str(tmp_path))
        record = extract_isaac(str(tmp_path))
        assert record.rgb_frames.dtype == np.uint8

    def test_lidar_scans_loaded(self, tmp_path):
        write_fake_isaac_output(str(tmp_path), n_frames=20)
        record = extract_isaac(str(tmp_path))
        assert len(record.lidar_scans) == 20

    def test_missing_lidar_file_gives_empty_list(self, tmp_path):
        write_fake_isaac_output(str(tmp_path))
        (tmp_path / "lidar_scans.npy").unlink()
        record = extract_isaac(str(tmp_path))
        assert record.lidar_scans == []
        record.validate()

    def test_scenario_id_matches_meta(self, tmp_path):
        write_fake_isaac_output(str(tmp_path))
        record = extract_isaac(str(tmp_path))
        assert record.scenario_id == "hover_3m_nominal"
