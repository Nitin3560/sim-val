from __future__ import annotations

from pathlib import Path

import yaml
from pxr import Usd

from isaac.build_scene import build_all_scenes, build_scene
from isaac.scene_validator import validate_scene


def write_minimal_scenario(
    tmp_dir: str,
    scenario_id: str = "test_scene",
    obstacles: list | None = None,
) -> str:
    scenario = {
        "scenario_id": scenario_id,
        "trajectory": {
            "waypoints": [
                {"x": 0.0, "y": 0.0, "z": -3.0, "hold_s": 3.0},
                {"x": 2.0, "y": 0.0, "z": -3.0, "hold_s": 3.0},
            ],
            "total_duration_s": 10.0,
        },
        "sensors": {
            "camera": {"width": 640, "height": 480, "fov_deg": 90.0, "fps": 15},
            "lidar": {"pattern": "rotary_16beam", "range_m": 20.0, "hz": 10},
            "depth": {"width": 640, "height": 480, "fps": 15},
        },
        "environment": {
            "lighting": "midday_sun",
            "ground_texture": "concrete",
            "obstacles": obstacles or [],
        },
        "degradation": "none",
    }
    path = Path(tmp_dir) / f"scenario_{scenario_id}.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(scenario, f)
    return str(path)


class TestBuildScene:
    def test_creates_usd_file(self, tmp_path):
        scenario_path = write_minimal_scenario(str(tmp_path))
        scene_path = build_scene(scenario_path, str(tmp_path / "scene"))
        assert Path(scene_path).exists()
        assert Path(scene_path).stat().st_size > 0

    def test_returns_path_string(self, tmp_path):
        scenario_path = write_minimal_scenario(str(tmp_path))
        result = build_scene(scenario_path, str(tmp_path / "scene"))
        assert isinstance(result, str)
        assert result.endswith(".usd")

    def test_usd_opens_cleanly(self, tmp_path):
        scenario_path = write_minimal_scenario(str(tmp_path))
        scene_path = build_scene(scenario_path, str(tmp_path / "scene"))
        assert Usd.Stage.Open(scene_path) is not None

    def test_z_up_axis(self, tmp_path):
        scenario_path = write_minimal_scenario(str(tmp_path))
        scene_path = build_scene(scenario_path, str(tmp_path / "scene"))
        stage = Usd.Stage.Open(scene_path)
        assert stage.GetMetadata("upAxis") == "Z"

    def test_default_prim_is_world(self, tmp_path):
        scenario_path = write_minimal_scenario(str(tmp_path))
        scene_path = build_scene(scenario_path, str(tmp_path / "scene"))
        stage = Usd.Stage.Open(scene_path)
        default_prim = stage.GetDefaultPrim()
        assert default_prim.IsValid()
        assert default_prim.GetPath().pathString == "/World"


class TestSceneValidator:
    def test_valid_scene_passes(self, tmp_path):
        scenario_path = write_minimal_scenario(str(tmp_path))
        scene_path = build_scene(scenario_path, str(tmp_path / "scene"))
        report = validate_scene(scene_path)
        assert report["valid"], "\n".join(report["issues"])
        assert len(report["issues"]) == 0

    def test_required_prims_all_present(self, tmp_path):
        scenario_path = write_minimal_scenario(str(tmp_path))
        scene_path = build_scene(scenario_path, str(tmp_path / "scene"))
        passed_text = " ".join(validate_scene(scene_path)["checks_passed"])
        for path in [
            "/World/UAV/body/camera_rgb",
            "/World/UAV/body/lidar_mount",
            "/World/Ground",
            "/World/SunLight",
        ]:
            assert path in passed_text

    def test_camera_has_focal_length(self, tmp_path):
        scenario_path = write_minimal_scenario(str(tmp_path))
        scene_path = build_scene(scenario_path, str(tmp_path / "scene"))
        report = validate_scene(scene_path)
        assert any("focalLength" in check for check in report["checks_passed"])

    def test_scene_with_box_obstacle(self, tmp_path):
        obstacles = [{"type": "box", "x": 2.0, "y": 0.0, "z": 0.0, "w": 0.5, "d": 0.5, "h": 3.0}]
        scenario_path = write_minimal_scenario(str(tmp_path), obstacles=obstacles)
        report = validate_scene(build_scene(scenario_path, str(tmp_path / "scene")))
        assert report["valid"], "\n".join(report["issues"])

    def test_scene_with_occlusion_plane(self, tmp_path):
        obstacles = [
            {
                "type": "occlusion_plane",
                "trigger_s": 8.0,
                "duration_s": 2.0,
                "x": 0.5,
                "y": 0.0,
                "z": -3.0,
            }
        ]
        scenario_path = write_minimal_scenario(str(tmp_path), obstacles=obstacles)
        report = validate_scene(build_scene(scenario_path, str(tmp_path / "scene")))
        assert report["valid"], "\n".join(report["issues"])


class TestBuildAllScenes:
    def test_builds_all_three_scenarios(self, tmp_path):
        results = build_all_scenes(
            config_dir="config",
            output_base=str(tmp_path / "scenes"),
        )
        assert len(results) == 3
        for path in results.values():
            assert Path(path).exists()

    def test_all_built_scenes_validate(self, tmp_path):
        results = build_all_scenes(
            config_dir="config",
            output_base=str(tmp_path / "scenes"),
        )
        for sid, path in results.items():
            report = validate_scene(path)
            assert report["valid"], f"{sid}:\n" + "\n".join(report["issues"])
