from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gazebo.extract_gazebo import (
    TOPIC_DEPTH,
    TOPIC_ODOM,
    TOPIC_ODOM_TYPE,
    TOPIC_RGB,
    TOPIC_TRUST,
    extract_gazebo,
    make_typestore,
    _interp_to_camera,
)


class TestInterpToCamera:
    def test_same_length_passthrough(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        times = np.linspace(0.0, 1.0, 3)
        result = _interp_to_camera(data, times)
        assert result.shape == (3, 2)
        assert np.allclose(result, data)

    def test_upsampling_produces_correct_shape(self):
        data = np.random.default_rng(0).standard_normal((3, 3))
        result = _interp_to_camera(data, np.linspace(0.0, 1.0, 10))
        assert result.shape == (10, 3)

    def test_empty_data_returns_zeros(self):
        result = _interp_to_camera(np.array([]).reshape(0, 3), np.linspace(0.0, 1.0, 5))
        assert result.shape == (5, 3)
        assert np.allclose(result, 0.0)

    def test_monotonic_interpolation(self):
        data = np.array([[0.0], [1.0], [2.0]])
        result = _interp_to_camera(data, np.linspace(0.0, 1.0, 5))
        assert np.all(np.diff(result[:, 0]) >= 0.0)


class TestExtractGazebo:
    def test_extraction_produces_valid_record(self, tmp_path):
        bag_path = _write_minimal_bag(tmp_path, n_frames=20)
        record = extract_gazebo(str(bag_path), "hover_3m_nominal")
        record.validate()
        assert record.source == "gazebo"
        assert record.scenario_id == "hover_3m_nominal"

    def test_rgb_shape_correct(self, tmp_path):
        bag_path = _write_minimal_bag(tmp_path, n_frames=15)
        record = extract_gazebo(str(bag_path), "test")
        assert record.rgb_frames.shape == (15, 48, 64, 3)
        assert record.rgb_frames.dtype == np.uint8

    def test_trust_state_present(self, tmp_path):
        bag_path = _write_minimal_bag(tmp_path, n_frames=15)
        record = extract_gazebo(str(bag_path), "test")
        assert record.trust_state is not None
        assert record.trust_state.shape == (15, 3)
        assert record.trust_state.dtype == np.float32

    def test_timestamps_start_at_zero(self, tmp_path):
        bag_path = _write_minimal_bag(tmp_path, n_frames=15)
        record = extract_gazebo(str(bag_path), "test")
        assert abs(record.timestamps[0]) < 1e-6

    def test_lidar_scans_empty(self, tmp_path):
        bag_path = _write_minimal_bag(tmp_path, n_frames=15)
        record = extract_gazebo(str(bag_path), "test")
        assert record.lidar_scans == []

    def test_empty_bag_raises(self, tmp_path):
        from rosbags.rosbag2 import Writer

        empty_bag = tmp_path / "empty_bag"
        with Writer(empty_bag, version=9):
            pass
        with pytest.raises(AssertionError, match="No RGB frames"):
            extract_gazebo(str(empty_bag), "test")


def _write_minimal_bag(tmp_path: Path, n_frames: int = 20) -> Path:
    from rosbags.rosbag2 import Writer
    from rosbags.typesys.stores.ros2_humble import (
        builtin_interfaces__msg__Time as Time,
        geometry_msgs__msg__Point as Point,
        geometry_msgs__msg__PointStamped as PointStamped,
        sensor_msgs__msg__Image as Image,
        std_msgs__msg__Header as Header,
    )

    typestore = make_typestore()
    odom_cls = typestore.types[TOPIC_ODOM_TYPE]
    bag_path = tmp_path / "test_bag"
    h, w = 48, 64
    rng = np.random.default_rng(42)

    with Writer(bag_path, version=9) as writer:
        conn_rgb = writer.add_connection(TOPIC_RGB, "sensor_msgs/msg/Image", typestore=typestore)
        conn_depth = writer.add_connection(TOPIC_DEPTH, "sensor_msgs/msg/Image", typestore=typestore)
        conn_odom = writer.add_connection(TOPIC_ODOM, TOPIC_ODOM_TYPE, typestore=typestore)
        conn_trust = writer.add_connection(
            TOPIC_TRUST, "geometry_msgs/msg/PointStamped", typestore=typestore
        )

        for i in range(n_frames):
            t_ns = int(i * 1e9 / 15.0)
            stamp = Time(sec=i, nanosec=0)
            rgb_msg = Image(
                header=Header(stamp=stamp, frame_id="camera"),
                height=h,
                width=w,
                encoding="rgb8",
                is_bigendian=False,
                step=w * 3,
                data=rng.integers(0, 255, h * w * 3, dtype=np.uint8),
            )
            writer.write(conn_rgb, t_ns, typestore.serialize_cdr(rgb_msg, "sensor_msgs/msg/Image"))

            depth_data = rng.uniform(0.5, 5.0, h * w).astype(np.float32).view(np.uint8)
            depth_msg = Image(
                header=Header(stamp=stamp, frame_id="depth"),
                height=h,
                width=w,
                encoding="32FC1",
                is_bigendian=False,
                step=w * 4,
                data=depth_data,
            )
            writer.write(
                conn_depth, t_ns, typestore.serialize_cdr(depth_msg, "sensor_msgs/msg/Image")
            )

            odom_msg = odom_cls(
                timestamp=t_ns // 1000,
                timestamp_sample=t_ns // 1000,
                pose_frame=1,
                position=np.array([0.1 * i, 0.2 * i, -3.0], dtype=np.float32),
                q=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                velocity_frame=1,
                velocity=np.array([0.1, 0.2, 0.0], dtype=np.float32),
                angular_velocity=np.zeros(3, dtype=np.float32),
                position_variance=np.zeros(3, dtype=np.float32),
                orientation_variance=np.zeros(3, dtype=np.float32),
                velocity_variance=np.zeros(3, dtype=np.float32),
                reset_counter=0,
                quality=0,
            )
            writer.write(conn_odom, t_ns, typestore.serialize_cdr(odom_msg, TOPIC_ODOM_TYPE))

            trust_msg = PointStamped(
                header=Header(stamp=stamp, frame_id="map"),
                point=Point(
                    x=float(rng.uniform(0.7, 1.0)),
                    y=float(rng.uniform(0.0, 0.3)),
                    z=float(rng.uniform(0.8, 1.0)),
                ),
            )
            writer.write(
                conn_trust,
                t_ns,
                typestore.serialize_cdr(trust_msg, "geometry_msgs/msg/PointStamped"),
            )

    return bag_path
