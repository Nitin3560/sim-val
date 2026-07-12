"""Headless Isaac Sim runner for SimVal scenarios."""
from __future__ import annotations

import json
import math
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from isaac.build_scene import build_scene


def run_scenario(scenario_path: str, output_dir: str) -> None:
    modules = _load_isaac_modules()
    SimulationApp = modules["SimulationApp"]
    simulation_app = SimulationApp({"headless": True})

    try:
        commands = modules["commands"]()
        rep = modules["rep"]()
        stage_utils = modules["stage_utils"]()
        AnnotatorRegistry = modules["AnnotatorRegistry"]()
        Gf = modules["Gf"]()

        with Path(scenario_path).open("r", encoding="utf-8") as f:
            scenario = yaml.safe_load(f)

        scenario_id = scenario["scenario_id"]
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        scene_usd = Path("outputs") / "scenes" / scenario_id / "scene.usd"
        if not scene_usd.exists():
            build_scene(scenario_path, str(scene_usd.parent))
        stage_utils.open_stage(str(scene_usd))

        cam_cfg = scenario["sensors"]["camera"]
        lidar_cfg = scenario["sensors"]["lidar"]
        width = int(cam_cfg["width"])
        height = int(cam_cfg["height"])
        camera_fps = int(cam_cfg["fps"])
        lidar_hz = int(lidar_cfg["hz"])
        physics_hz = 60

        rgb_rp = rep.create.render_product("/World/UAV/body/camera_rgb", [width, height])
        rgb_anno = AnnotatorRegistry.get_annotator("rgb")
        rgb_anno.attach(rgb_rp)
        depth_anno = AnnotatorRegistry.get_annotator("distance_to_image_plane")
        depth_anno.attach(rgb_rp)

        _, lidar_sensor = commands.execute(
            "IsaacSensorCreateRtxLidar",
            path="/World/UAV/body/lidar",
            parent="/World/UAV/body/lidar_mount",
            config="Example_Rotary",
            translation=(0.0, 0.0, 0.0),
            orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),
        )
        lidar_rp = rep.create.render_product(lidar_sensor.GetPath(), [1, 1])
        lidar_anno = AnnotatorRegistry.get_annotator(
            "RtxSensorCpuIsaacCreateRTXLidarScanBuffer"
        )
        lidar_anno.attach(lidar_rp)

        stage = stage_utils.get_stage()
        uav_xf = stage.GetPrimAtPath("/World/UAV").GetAttribute("xformOp:translate")
        positions_ned = _expand_trajectory(scenario["trajectory"], physics_hz)

        rgb_frames: list[np.ndarray] = []
        depth_frames: list[np.ndarray] = []
        lidar_scans: list[np.ndarray] = []
        positions: list[list[float]] = []
        timestamps: list[float] = []

        camera_every = max(physics_hz // camera_fps, 1)
        lidar_every = max(physics_hz // lidar_hz, 1)

        for step, (ned_x, ned_y, ned_z) in enumerate(positions_ned):
            uav_xf.Set(Gf.Vec3d(ned_x, -ned_y, -ned_z))
            rep.orchestrator.step(rt_subframes=1)
            timestamp_s = step / physics_hz

            if step % camera_every == 0:
                rgb_data = rgb_anno.get_data()
                if rgb_data is not None and getattr(rgb_data, "size", 0) > 0:
                    rgb_frames.append(np.asarray(rgb_data)[..., :3].copy())
                    depth_data = depth_anno.get_data()
                    depth_frames.append(
                        np.asarray(depth_data).copy()
                        if depth_data is not None
                        else np.full((height, width), np.nan, dtype=np.float32)
                    )
                    positions.append([ned_x, ned_y, ned_z])
                    timestamps.append(timestamp_s)

            if step % lidar_every == 0:
                lidar_data = lidar_anno.get_data()
                if lidar_data is not None and "data" in lidar_data:
                    points = lidar_data["data"]
                    if points is not None and len(points) > 0:
                        lidar_scans.append(np.asarray(points).copy())

        _save_outputs(
            out,
            scenario_id,
            rgb_frames,
            depth_frames,
            lidar_scans,
            positions,
            timestamps,
        )
    finally:
        simulation_app.close()


def _load_isaac_modules() -> dict[str, Any]:
    from isaacsim import SimulationApp

    return {
        "SimulationApp": SimulationApp,
        "commands": lambda: import_module("omni.kit.commands"),
        "rep": lambda: import_module("omni.replicator.core"),
        "stage_utils": lambda: import_module("omni.isaac.core.utils.stage"),
        "AnnotatorRegistry": lambda: import_module(
            "omni.replicator.core"
        ).AnnotatorRegistry,
        "Gf": lambda: import_module("pxr.Gf"),
    }


def _save_outputs(
    out: Path,
    scenario_id: str,
    rgb_frames: list[np.ndarray],
    depth_frames: list[np.ndarray],
    lidar_scans: list[np.ndarray],
    positions: list[list[float]],
    timestamps: list[float],
) -> None:
    n_frames = len(rgb_frames)
    np.save(out / "rgb_frames.npy", np.array(rgb_frames, dtype=np.uint8))
    np.save(out / "depth_frames.npy", np.array(depth_frames, dtype=np.float32))
    np.save(
        out / "lidar_scans.npy",
        np.array(lidar_scans, dtype=object),
        allow_pickle=True,
    )
    np.save(out / "positions.npy", np.array(positions, dtype=np.float32))
    np.save(out / "velocities.npy", np.zeros((n_frames, 3), dtype=np.float32))
    np.save(out / "timestamps.npy", np.array(timestamps, dtype=np.float64))

    meta = {"scenario_id": scenario_id, "source": "isaac_sim", "n_frames": n_frames}
    with (out / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _expand_trajectory(
    traj: dict,
    physics_hz: int,
) -> list[tuple[float, float, float]]:
    positions: list[tuple[float, float, float]] = []

    if "waypoints" in traj:
        for wp in traj["waypoints"]:
            hold_steps = int(float(wp["hold_s"]) * physics_hz)
            point = (float(wp["x"]), float(wp["y"]), float(wp["z"]))
            positions.extend([point] * hold_steps)
        return positions

    if traj.get("mode") == "circle" or "center" in traj:
        center = traj.get("center", {"x": 0.0, "y": 0.0, "z": -5.0})
        cx = float(center.get("x", 0.0))
        cy = float(center.get("y", 0.0))
        cz = float(center.get("z", -5.0))
        radius = float(traj.get("radius_m", 3.0))
        period = float(traj.get("period_s", 18.0))
        total = float(traj.get("total_duration_s", 36.0))
        for i in range(int(total * physics_hz)):
            angle = 2.0 * math.pi * (i / physics_hz) / period
            positions.append(
                (cx + radius * math.cos(angle), cy + radius * math.sin(angle), cz)
            )
        return positions

    return positions


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: /isaac-sim/python.sh isaac/run_scenario.py <scenario.yaml> <output_dir>")
        sys.exit(1)
    run_scenario(sys.argv[1], sys.argv[2])
