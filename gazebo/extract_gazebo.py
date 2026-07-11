"""Extract Gazebo ROS 2 bags into SensorRecord objects."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

from schema.sensor_record import SensorRecord


TOPIC_RGB = "/drone_0/twinguard/camera/image_raw"
TOPIC_DEPTH = "/drone_0/twinguard/camera/depth"
TOPIC_ODOM = "/drone_0/fmu/out/vehicle_odometry"
TOPIC_TRUST = "/drone_0/twinguard/trust_state"
TOPICS = {TOPIC_RGB, TOPIC_DEPTH, TOPIC_ODOM, TOPIC_TRUST}

PX4_VEHICLE_ODOMETRY_MSG = """
uint64 timestamp
uint64 timestamp_sample
uint8 pose_frame
float32[3] position
float32[4] q
uint8 velocity_frame
float32[3] velocity
float32[3] angular_velocity
float32[3] position_variance
float32[3] orientation_variance
float32[3] velocity_variance
uint8 reset_counter
int8 quality
"""


def make_typestore() -> Any:
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    if TOPIC_ODOM_TYPE not in typestore.types:
        typestore.register(get_types_from_msg(PX4_VEHICLE_ODOMETRY_MSG, TOPIC_ODOM_TYPE))
    return typestore


TOPIC_ODOM_TYPE = "px4_msgs/msg/VehicleOdometry"


def extract_gazebo(bag_path: str, scenario_id: str) -> SensorRecord:
    typestore = make_typestore()
    rgb_frames: list[np.ndarray] = []
    depth_frames: list[np.ndarray] = []
    rgb_times: list[float] = []
    odom_times: list[float] = []
    trust_times: list[float] = []
    positions: list[list[float]] = []
    velocities: list[list[float]] = []
    trust_states: list[list[float]] = []

    with Reader(Path(bag_path)) as reader:
        connections = [c for c in reader.connections if c.topic in TOPICS]
        for connection, timestamp_ns, rawdata in reader.messages(connections=connections):
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            timestamp_s = timestamp_ns * 1e-9

            if connection.topic == TOPIC_RGB:
                rgb_frames.append(_decode_rgb_image(msg))
                rgb_times.append(timestamp_s)
            elif connection.topic == TOPIC_DEPTH:
                depth_frames.append(_decode_depth_image(msg))
            elif connection.topic == TOPIC_ODOM:
                odom_times.append(timestamp_s)
                positions.append([float(v) for v in msg.position])
                velocities.append([float(v) for v in msg.velocity])
            elif connection.topic == TOPIC_TRUST:
                trust_times.append(timestamp_s)
                trust_states.append([float(msg.point.x), float(msg.point.y), float(msg.point.z)])

    n_frames = len(rgb_frames)
    assert n_frames > 0, (
        f"No RGB frames found in bag at {bag_path}. Check that {TOPIC_RGB} was recorded."
    )

    timestamps = np.array(rgb_times, dtype=np.float64)
    positions_arr = _interp_to_camera(
        _rows_or_empty(positions), timestamps, _times_or_none(odom_times)
    )
    velocities_arr = _interp_to_camera(
        _rows_or_empty(velocities), timestamps, _times_or_none(odom_times)
    )
    trust_arr = None
    if trust_states:
        trust_arr = _interp_to_camera(
            _rows_or_empty(trust_states), timestamps, _times_or_none(trust_times)
        )

    record = SensorRecord(
        scenario_id=scenario_id,
        source="gazebo",
        rgb_frames=np.array(rgb_frames, dtype=np.uint8),
        depth_frames=_stack_depth(depth_frames, n_frames),
        lidar_scans=[],
        positions=positions_arr.astype(np.float32),
        velocities=velocities_arr.astype(np.float32),
        timestamps=(timestamps - timestamps[0]).astype(np.float64),
        trust_state=trust_arr.astype(np.float32) if trust_arr is not None else None,
    )
    record.validate()
    return record


def _decode_rgb_image(msg: Any) -> np.ndarray:
    channels = max(int(msg.step // msg.width), 1)
    frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, channels)
    if msg.encoding.lower() in ("bgr8", "bgra8"):
        frame = frame[..., :3][..., ::-1]
    else:
        frame = frame[..., :3]
    return frame.copy()


def _decode_depth_image(msg: Any) -> np.ndarray:
    if msg.encoding.upper() == "32FC1":
        dtype = np.float32
    elif msg.encoding.upper() in ("16UC1", "MONO16"):
        dtype = np.uint16
    else:
        dtype = np.float32
    return np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width).astype(np.float32)


def _stack_depth(depth_frames: list[np.ndarray], n_frames: int) -> np.ndarray:
    if len(depth_frames) >= n_frames:
        return np.array(depth_frames[:n_frames], dtype=np.float32)
    if not depth_frames:
        h, w = 1, 1
        return np.zeros((n_frames, h, w), dtype=np.float32)
    pad = [depth_frames[-1].copy() for _ in range(n_frames - len(depth_frames))]
    return np.array(depth_frames + pad, dtype=np.float32)


def _rows_or_empty(rows: list[list[float]], width: int = 3) -> np.ndarray:
    if not rows:
        return np.zeros((0, width), dtype=np.float64)
    return np.array(rows, dtype=np.float64)


def _times_or_none(times: list[float]) -> np.ndarray | None:
    if not times:
        return None
    return np.array(times, dtype=np.float64)


def _interp_to_camera(
    data: np.ndarray,
    target_times: np.ndarray,
    source_times: np.ndarray | None = None,
) -> np.ndarray:
    if len(data) == 0:
        width = data.shape[1] if data.ndim == 2 else 3
        return np.zeros((len(target_times), width), dtype=np.float64)

    if source_times is None:
        source_times = np.linspace(target_times[0], target_times[-1], len(data))

    if len(data) == 1:
        return np.repeat(data.astype(np.float64), len(target_times), axis=0)

    result = np.zeros((len(target_times), data.shape[1]), dtype=np.float64)
    for col in range(data.shape[1]):
        result[:, col] = np.interp(target_times, source_times, data[:, col])
    return result
