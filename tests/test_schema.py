import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from schema.sensor_record import SensorRecord


def make_gazebo_record(N: int = 30) -> SensorRecord:
    rng = np.random.default_rng(seed=0)
    return SensorRecord(
        scenario_id="hover_3m_nominal",
        source="gazebo",
        rgb_frames=rng.integers(0, 255, (N, 480, 640, 3), dtype=np.uint8),
        depth_frames=rng.uniform(0.5, 20.0, (N, 480, 640)).astype(np.float32),
        lidar_scans=[
            rng.standard_normal((
                rng.integers(8000, 15000), 3
            )).astype(np.float32)
            for _ in range(N)
        ],
        positions=rng.standard_normal((N, 3)).astype(np.float32),
        velocities=rng.standard_normal((N, 3)).astype(np.float32),
        timestamps=np.linspace(0.0, 20.0, N),
        trust_state=rng.uniform(0.0, 1.0, (N, 3)).astype(np.float32),
    )


def make_isaac_record(N: int = 30) -> SensorRecord:
    rng = np.random.default_rng(seed=1)
    return SensorRecord(
        scenario_id="hover_3m_nominal",
        source="isaac_sim",
        rgb_frames=rng.integers(0, 255, (N, 480, 640, 3), dtype=np.uint8),
        depth_frames=rng.uniform(0.5, 20.0, (N, 480, 640)).astype(np.float32),
        lidar_scans=[
            rng.standard_normal((
                rng.integers(6000, 12000), 3
            )).astype(np.float32)
            for _ in range(N)
        ],
        positions=rng.standard_normal((N, 3)).astype(np.float32),
        velocities=rng.standard_normal((N, 3)).astype(np.float32),
        timestamps=np.linspace(0.0, 20.0, N),
        trust_state=None,
    )


def test_gazebo_record_validates():
    make_gazebo_record().validate()


def test_isaac_record_validates():
    make_isaac_record().validate()


def test_save_load_roundtrip_gazebo():
    original = make_gazebo_record(N=20)
    with tempfile.TemporaryDirectory() as tmp:
        original.save(tmp)
        loaded = SensorRecord.load(tmp)

    assert loaded.source == "gazebo"
    assert loaded.scenario_id == "hover_3m_nominal"
    assert np.array_equal(original.rgb_frames, loaded.rgb_frames)
    assert np.allclose(original.depth_frames, loaded.depth_frames)
    assert np.allclose(original.positions, loaded.positions)
    assert np.allclose(original.timestamps, loaded.timestamps)
    assert loaded.trust_state is not None
    assert np.allclose(original.trust_state, loaded.trust_state)
    loaded.validate()


def test_save_load_roundtrip_isaac():
    original = make_isaac_record(N=20)
    with tempfile.TemporaryDirectory() as tmp:
        original.save(tmp)
        loaded = SensorRecord.load(tmp)

    assert loaded.source == "isaac_sim"
    assert loaded.trust_state is None
    assert np.array_equal(original.rgb_frames, loaded.rgb_frames)
    loaded.validate()


def test_validate_catches_rgb_shape_mismatch():
    record = make_gazebo_record(N=30)
    record.rgb_frames = np.random.randint(0, 255, (25, 480, 640, 3), dtype=np.uint8)
    with pytest.raises(AssertionError, match="rgb_frames"):
        record.validate()


def test_validate_catches_wrong_source():
    record = make_gazebo_record()
    record.source = "blender"
    with pytest.raises(ValueError, match="source"):
        record.validate()


def test_validate_catches_trust_state_on_isaac():
    record = make_isaac_record()
    record.trust_state = np.zeros((30, 3), dtype=np.float32)
    with pytest.raises(AssertionError, match="trust_state"):
        record.validate()


def test_validate_catches_positions_shape():
    record = make_gazebo_record(N=30)
    record.positions = np.zeros((30, 4), dtype=np.float32)
    with pytest.raises(AssertionError, match="positions"):
        record.validate()


def test_meta_json_written_correctly():
    record = make_gazebo_record(N=15)
    with tempfile.TemporaryDirectory() as tmp:
        record.save(tmp)
        with (Path(tmp) / "meta.json").open("r", encoding="utf-8") as f:
            meta = json.load(f)

    assert meta["scenario_id"] == "hover_3m_nominal"
    assert meta["source"] == "gazebo"
    assert meta["n_frames"] == 15
    assert meta["has_trust_state"] is True
    assert meta["has_lidar"] is True
