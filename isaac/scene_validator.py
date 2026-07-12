"""Validate required USD scene structure."""
from __future__ import annotations

from pxr import Usd, UsdGeom


REQUIRED_PRIMS = [
    "/World",
    "/World/Ground",
    "/World/SunLight",
    "/World/UAV",
    "/World/UAV/body",
    "/World/UAV/body/camera_rgb",
    "/World/UAV/body/lidar_mount",
]


def validate_scene(scene_path: str) -> dict:
    stage = Usd.Stage.Open(scene_path)
    issues: list[str] = []
    passed: list[str] = []

    if stage is None:
        return {
            "scene_path": scene_path,
            "valid": False,
            "issues": ["Could not open USD stage"],
            "checks_passed": [],
        }

    for path in REQUIRED_PRIMS:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            passed.append(f"Found: {path} ({prim.GetTypeName()})")
        else:
            issues.append(f"Missing required prim: {path}")

    camera = stage.GetPrimAtPath("/World/UAV/body/camera_rgb")
    if camera.IsValid():
        focal_length = camera.GetAttribute("focalLength")
        if focal_length and focal_length.HasValue():
            passed.append(f"camera_rgb focalLength: {focal_length.Get():.2f} mm")
        else:
            issues.append("camera_rgb missing focalLength attribute")

    up_axis = UsdGeom.GetStageUpAxis(stage)
    if up_axis == UsdGeom.Tokens.z:
        passed.append("upAxis: Z")
    else:
        issues.append(f"upAxis is '{up_axis}', expected 'Z'")

    return {
        "scene_path": scene_path,
        "valid": len(issues) == 0,
        "issues": issues,
        "checks_passed": passed,
    }
