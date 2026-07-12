from __future__ import annotations

import json

import numpy as np
import pytest

from pipeline.ekf_propagation import compute_trust_state_delta
from pipeline.run_pipeline import _build_summary
from schema.sensor_record import SensorRecord


N, H, W = 30, 48, 64


def make_paired_records(
    trust_delta: float = 0.0,
    seed: int = 0,
) -> tuple[SensorRecord, SensorRecord]:
    rng = np.random.default_rng(seed)
    timestamps = np.linspace(0.0, 20.0, N)
    positions = np.column_stack(
        [
            2.0 * np.sin(2.0 * np.pi * timestamps / 20.0),
            np.zeros(N),
            -3.0 * np.ones(N),
        ]
    ).astype(np.float32)

    trust_g = rng.uniform(0.8, 1.0, (N, 3)).astype(np.float32)
    trust_i = trust_g.copy()
    trust_i[:, 2] -= trust_delta

    gazebo = SensorRecord(
        scenario_id="hover_3m_nominal",
        source="gazebo",
        rgb_frames=rng.integers(0, 255, (N, H, W, 3), dtype=np.uint8),
        depth_frames=rng.uniform(0.5, 5.0, (N, H, W)).astype(np.float32),
        lidar_scans=[],
        positions=positions,
        velocities=np.zeros((N, 3), dtype=np.float32),
        timestamps=timestamps.astype(np.float64),
        trust_state=trust_g,
    )
    isaac = SensorRecord(
        scenario_id="hover_3m_nominal",
        source="isaac_sim",
        rgb_frames=rng.integers(0, 200, (N, H, W, 3), dtype=np.uint8),
        depth_frames=rng.uniform(0.5, 5.0, (N, H, W)).astype(np.float32),
        lidar_scans=[
            rng.standard_normal((5000, 3)).astype(np.float32) for _ in range(N)
        ],
        positions=positions,
        velocities=np.zeros((N, 3), dtype=np.float32),
        timestamps=timestamps.astype(np.float64),
        trust_state=None,
    )
    return gazebo, isaac


class TestEKFPropagation:
    def test_zero_delta_gives_low_risk(self):
        trust = np.random.default_rng(0).uniform(0.8, 1.0, (N, 3)).astype(np.float32)
        result = compute_trust_state_delta(trust, trust.copy())
        assert result["calibration_risk"] == "LOW"
        assert result["authority_scale_delta_mean"] < 0.01

    def test_large_delta_gives_high_risk(self):
        g_trust = np.ones((N, 3), dtype=np.float32) * 0.9
        i_trust = g_trust.copy()
        i_trust[:, 2] -= 0.2
        result = compute_trust_state_delta(g_trust, i_trust)
        assert result["calibration_risk"] == "HIGH"
        assert result["authority_scale_delta_mean"] > 0.15

    def test_medium_delta_gives_medium_risk(self):
        g_trust = np.ones((N, 3), dtype=np.float32) * 0.9
        i_trust = g_trust.copy()
        i_trust[:, 2] -= 0.1
        result = compute_trust_state_delta(g_trust, i_trust)
        assert result["calibration_risk"] == "MEDIUM"

    def test_output_keys_complete(self):
        trust = np.random.default_rng(0).uniform(0.8, 1.0, (N, 3)).astype(np.float32)
        result = compute_trust_state_delta(trust, trust.copy())
        for key in [
            "trust_delta_mean",
            "residual_delta_mean",
            "authority_scale_delta_mean",
            "gazebo_authority_mean",
            "isaac_authority_mean",
            "calibration_risk",
            "assessment",
            "recommendation",
        ]:
            assert key in result

    def test_shape_mismatch_raises(self):
        with pytest.raises(AssertionError, match="Shape mismatch"):
            compute_trust_state_delta(
                np.ones((N, 3), dtype=np.float32),
                np.ones((N + 1, 3), dtype=np.float32),
            )

    def test_assessment_is_nonempty_string(self):
        trust = np.random.default_rng(0).uniform(0.8, 1.0, (N, 3)).astype(np.float32)
        result = compute_trust_state_delta(trust, trust.copy())
        assert isinstance(result["assessment"], str)
        assert len(result["assessment"]) > 10

    def test_recommendation_mentions_threshold(self):
        g_trust = np.ones((N, 3), dtype=np.float32) * 0.9
        i_trust = g_trust.copy()
        i_trust[:, 2] -= 0.2
        result = compute_trust_state_delta(g_trust, i_trust)
        assert "threshold" in result["recommendation"].lower()


class TestBuildSummary:
    def _make_camera_metrics(
        self,
        psnr: float = 28.0,
        ssim: float = 0.65,
        lpips: float = 0.25,
    ) -> dict:
        return {
            "psnr_mean_db": psnr,
            "ssim_mean": ssim,
            "lpips_mean": lpips,
            "psnr_std_db": 1.0,
            "ssim_std": 0.05,
            "lpips_std": 0.03,
            "psnr_min_db": 20.0,
            "ssim_min": 0.5,
            "lpips_max": 0.4,
            "psnr_per_frame": [],
            "ssim_per_frame": [],
            "lpips_per_frame": [],
            "note": "lower = more similar",
        }

    def test_summary_has_required_keys(self):
        result = _build_summary(self._make_camera_metrics(), None, None)
        for key in [
            "camera_fidelity",
            "camera_assessment",
            "lidar_fidelity",
            "autonomy_impact",
            "recommendation",
            "calibration_risk",
            "key_finding",
        ]:
            assert key in result

    def test_high_ssim_gives_positive_assessment(self):
        result = _build_summary(self._make_camera_metrics(ssim=0.85, lpips=0.10), None, None)
        assert "high" in result["camera_assessment"].lower()

    def test_low_ssim_gives_gap_assessment(self):
        result = _build_summary(self._make_camera_metrics(ssim=0.30, lpips=0.55), None, None)
        assert "significant" in result["camera_assessment"].lower()

    def test_no_ekf_gives_unknown_risk(self):
        result = _build_summary(self._make_camera_metrics(), None, None)
        assert result["calibration_risk"] == "UNKNOWN"

    def test_with_ekf_gives_specific_risk(self):
        ekf = {
            "assessment": "Gazebo overestimates by 12%",
            "recommendation": "Tighten thresholds by 12%",
            "calibration_risk": "MEDIUM",
            "gazebo_authority_mean": 0.92,
            "isaac_authority_mean": 0.80,
            "authority_scale_delta_mean": 0.12,
        }
        result = _build_summary(self._make_camera_metrics(), None, ekf)
        assert result["calibration_risk"] == "MEDIUM"
        assert "12%" in result["key_finding"]

    def test_key_finding_is_nonempty(self):
        result = _build_summary(self._make_camera_metrics(), None, None)
        assert isinstance(result["key_finding"], str)
        assert len(result["key_finding"]) > 10


class TestRunPipelineIntegration:
    def test_pipeline_produces_report_json(self, tmp_path, monkeypatch):
        gazebo_rec, isaac_rec = make_paired_records(trust_delta=0.1)
        _patch_pipeline(monkeypatch, gazebo_rec, isaac_rec)

        from pipeline.run_pipeline import run_pipeline

        report = run_pipeline(
            scenario_id="hover_3m_nominal",
            gazebo_bag="fake_bag",
            isaac_dir="fake_dir",
            output_dir=str(tmp_path / "reports"),
        )

        report_path = tmp_path / "reports" / "gap_report_hover_3m_nominal.json"
        loaded = json.loads(report_path.read_text(encoding="utf-8"))
        assert loaded["scenario_id"] == "hover_3m_nominal"
        assert "camera" in loaded
        assert "summary" in loaded
        assert "key_finding" in loaded["summary"]
        assert report["scenario_id"] == "hover_3m_nominal"

    def test_pipeline_report_has_all_sections(self, tmp_path, monkeypatch):
        gazebo_rec, isaac_rec = make_paired_records()
        _patch_pipeline(monkeypatch, gazebo_rec, isaac_rec)

        from pipeline.run_pipeline import run_pipeline

        report = run_pipeline(
            scenario_id="hover_3m_nominal",
            gazebo_bag="fake_bag",
            isaac_dir="fake_dir",
            output_dir=str(tmp_path / "reports"),
        )

        for section in ["scenario_id", "generated_at", "alignment", "camera", "summary"]:
            assert section in report

    def test_lidar_skipped_when_gazebo_has_none(self, tmp_path, monkeypatch):
        gazebo_rec, isaac_rec = make_paired_records()
        _patch_pipeline(monkeypatch, gazebo_rec, isaac_rec)

        from pipeline.run_pipeline import run_pipeline

        report = run_pipeline(
            scenario_id="hover_3m_nominal",
            gazebo_bag="fake_bag",
            isaac_dir="fake_dir",
            output_dir=str(tmp_path / "reports"),
        )

        assert report["lidar"] is None


def _patch_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    gazebo_rec: SensorRecord,
    isaac_rec: SensorRecord,
) -> None:
    import pipeline.run_pipeline as rp

    monkeypatch.setattr(rp, "extract_gazebo", lambda bag, sid: gazebo_rec)
    monkeypatch.setattr(rp, "extract_isaac", lambda directory: isaac_rec)
    monkeypatch.setattr(
        rp,
        "compute_all_camera_metrics",
        lambda reference, query: {
            "psnr_mean_db": 28.0,
            "psnr_std_db": 1.0,
            "psnr_min_db": 24.0,
            "psnr_per_frame": [],
            "ssim_mean": 0.65,
            "ssim_std": 0.05,
            "ssim_min": 0.50,
            "ssim_per_frame": [],
            "lpips_mean": 0.25,
            "lpips_std": 0.03,
            "lpips_max": 0.40,
            "lpips_per_frame": [],
            "note": "lower = more similar",
        },
    )
