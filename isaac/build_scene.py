"""Build Isaac Sim USD scenes from SimVal scenario YAML files."""
from __future__ import annotations

import math
from pathlib import Path

import yaml
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics


def build_scene(scenario_path: str, output_dir: str) -> str:
    with Path(scenario_path).open("r", encoding="utf-8") as f:
        scenario = yaml.safe_load(f)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    scene_path = str(out / "scene.usd")

    stage = Usd.Stage.CreateNew(scene_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    _add_ground(stage)
    _add_light(stage, scenario.get("environment", {}).get("lighting", "midday_sun"))
    _add_obstacles(stage, scenario.get("environment", {}).get("obstacles", []))
    _add_uav(stage, scenario)

    stage.GetRootLayer().Save()
    return scene_path


def build_all_scenes(
    config_dir: str = "config",
    output_base: str = "outputs/scenes",
) -> dict[str, str]:
    results: dict[str, str] = {}
    for yaml_path in sorted(Path(config_dir).glob("scenario_*.yaml")):
        with yaml_path.open("r", encoding="utf-8") as f:
            scenario = yaml.safe_load(f)
        sid = scenario["scenario_id"]
        results[sid] = build_scene(str(yaml_path), str(Path(output_base) / sid))
    return results


def _add_ground(stage: Usd.Stage) -> None:
    ground = UsdGeom.Mesh.Define(stage, "/World/Ground")
    ground.CreatePointsAttr(
        [
            Gf.Vec3f(-50.0, -50.0, 0.0),
            Gf.Vec3f(50.0, -50.0, 0.0),
            Gf.Vec3f(50.0, 50.0, 0.0),
            Gf.Vec3f(-50.0, 50.0, 0.0),
        ]
    )
    ground.CreateFaceVertexCountsAttr([4])
    ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    ground.CreateNormalsAttr([Gf.Vec3f(0.0, 0.0, 1.0)])
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())


def _add_light(stage: Usd.Stage, lighting: str) -> None:
    light = UsdLux.DistantLight.Define(stage, "/World/SunLight")
    if lighting == "midday_sun":
        light.CreateIntensityAttr(3000.0)
        xf = UsdGeom.Xformable(light.GetPrim())
        xf.AddRotateXOp().Set(45.0)
        xf.AddRotateZOp().Set(-30.0)
    elif lighting == "overcast":
        light.CreateIntensityAttr(800.0)
    else:
        light.CreateIntensityAttr(2000.0)


def _add_obstacles(stage: Usd.Stage, obstacles: list[dict]) -> None:
    for idx, obs in enumerate(obstacles):
        obs_type = obs.get("type", "box")
        if obs_type not in ("box", "occlusion_plane"):
            continue

        cube = UsdGeom.Cube.Define(stage, f"/World/Obstacle_{idx}")
        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.AddTranslateOp().Set(
            _ned_to_isaac(obs.get("x", 0.0), obs.get("y", 0.0), obs.get("z", 0.0))
        )
        if obs_type == "box":
            xf.AddScaleOp().Set(
                Gf.Vec3d(
                    float(obs.get("w", 0.5)),
                    float(obs.get("d", 0.5)),
                    float(obs.get("h", 3.0)),
                )
            )
        else:
            xf.AddScaleOp().Set(Gf.Vec3d(2.0, 0.05, 2.0))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


def _add_uav(stage: Usd.Stage, scenario: dict) -> None:
    uav = UsdGeom.Xform.Define(stage, "/World/UAV")
    UsdGeom.Xform.Define(stage, "/World/UAV/body")
    UsdGeom.Xformable(uav.GetPrim()).AddTranslateOp().Set(_initial_position(scenario))

    cam_cfg = scenario.get("sensors", {}).get("camera", {})
    camera = UsdGeom.Camera.Define(stage, "/World/UAV/body/camera_rgb")
    camera.CreateProjectionAttr("perspective")
    aperture = 36.0
    fov_deg = float(cam_cfg.get("fov_deg", 90.0))
    focal_len = aperture / (2.0 * math.tan(math.radians(fov_deg / 2.0)))
    camera.CreateFocalLengthAttr(focal_len)
    camera.CreateHorizontalApertureAttr(aperture)

    lidar_mount = UsdGeom.Xform.Define(stage, "/World/UAV/body/lidar_mount")
    UsdGeom.Xformable(lidar_mount.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.1))


def _initial_position(scenario: dict) -> Gf.Vec3d:
    trajectory = scenario.get("trajectory", {})
    if "waypoints" in trajectory:
        wp0 = trajectory["waypoints"][0]
        return _ned_to_isaac(wp0.get("x", 0.0), wp0.get("y", 0.0), wp0.get("z", -3.0))

    center = trajectory.get("center", {"x": 0.0, "y": 0.0, "z": -3.0})
    radius = float(trajectory.get("radius_m", 0.0))
    return _ned_to_isaac(
        float(center.get("x", 0.0)) + radius,
        center.get("y", 0.0),
        center.get("z", -3.0),
    )


def _ned_to_isaac(x: float, y: float, z: float) -> Gf.Vec3d:
    return Gf.Vec3d(float(x), -float(y), -float(z))
