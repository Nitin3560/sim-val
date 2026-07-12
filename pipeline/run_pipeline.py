"""SimVal comparison pipeline and gap-report writer."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from gazebo.extract_gazebo import extract_gazebo
from isaac.extract_isaac import extract_isaac
from metrics.camera_metrics import compute_all_camera_metrics
from metrics.lidar_metrics import compute_all_lidar_metrics
from pipeline.align_trajectories import align_trajectories, compute_alignment_quality
from pipeline.ekf_propagation import compute_trust_state_delta


def run_pipeline(
    scenario_id: str,
    gazebo_bag: str,
    isaac_dir: str,
    isaac_replay_bag: str | None = None,
    output_dir: str = "outputs/reports",
) -> dict[str, Any]:
    gazebo_rec = extract_gazebo(gazebo_bag, scenario_id)
    isaac_rec = extract_isaac(isaac_dir)
    gazebo_aligned, isaac_aligned = align_trajectories(gazebo_rec, isaac_rec)

    alignment = compute_alignment_quality(gazebo_aligned, isaac_aligned)
    camera = compute_all_camera_metrics(
        gazebo_aligned.rgb_frames,
        isaac_aligned.rgb_frames,
    )

    lidar = None
    if gazebo_aligned.lidar_scans and isaac_aligned.lidar_scans:
        lidar = compute_all_lidar_metrics(
            gazebo_aligned.lidar_scans,
            isaac_aligned.lidar_scans,
        )

    ekf = _compute_ekf_if_available(
        scenario_id,
        gazebo_aligned,
        isaac_replay_bag,
    )

    report: dict[str, Any] = {
        "scenario_id": scenario_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "alignment": alignment,
        "camera": camera,
        "lidar": lidar,
        "ekf_propagation": ekf,
        "summary": _build_summary(camera, lidar, ekf),
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / f"gap_report_{scenario_id}.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(report), f, indent=2)

    return report


def _compute_ekf_if_available(
    scenario_id: str,
    gazebo_aligned: Any,
    isaac_replay_bag: str | None,
) -> dict[str, Any] | None:
    if isaac_replay_bag is None or gazebo_aligned.trust_state is None:
        return None

    try:
        replay_rec = extract_gazebo(isaac_replay_bag, f"{scenario_id}_isaac_replay")
        _, replay_aligned = align_trajectories(gazebo_aligned, replay_rec)
    except Exception:
        return None

    if replay_aligned.trust_state is None:
        return None
    return compute_trust_state_delta(
        gazebo_aligned.trust_state,
        replay_aligned.trust_state,
    )


def _build_summary(
    camera: dict[str, Any],
    lidar: dict[str, Any] | None,
    ekf: dict[str, Any] | None,
) -> dict[str, Any]:
    psnr = float(camera["psnr_mean_db"])
    ssim = float(camera["ssim_mean"])
    lpips = float(camera["lpips_mean"])

    if ssim > 0.7 and lpips < 0.2:
        camera_assessment = "high camera fidelity - minimal rendering gap"
    elif ssim > 0.5 or lpips < 0.35:
        camera_assessment = "moderate camera gap - may affect VO quality scores"
    else:
        camera_assessment = "significant camera gap - VO calibration risk present"

    lidar_fidelity = "LiDAR not available (gz_x500_depth has no ray-cast LiDAR)"
    if lidar is not None:
        lidar_fidelity = (
            f"DCD={lidar['dcd_mean']:.4f} "
            f"density_gap={lidar['density_gap_fraction']:+.1%}"
        )

    if ekf is None:
        autonomy_impact = (
            "EKF propagation pending - replay Isaac Sim sensor data through "
            "TwinGuard to obtain trust_state comparison."
        )
        recommendation = (
            "Replay Isaac Sim outputs through TwinGuard to complete calibration "
            "risk assessment."
        )
        risk = "UNKNOWN"
        key_finding = f"Camera: {camera_assessment}. EKF propagation pending Isaac Sim replay."
    else:
        autonomy_impact = ekf["assessment"]
        recommendation = ekf["recommendation"]
        risk = ekf["calibration_risk"]
        direction = (
            "overestimates"
            if ekf["gazebo_authority_mean"] > ekf["isaac_authority_mean"]
            else "underestimates"
        )
        key_finding = (
            f"Gazebo {direction} TwinGuard authority_scale by "
            f"{_format_percent(ekf['authority_scale_delta_mean'])} "
            "vs Isaac Sim RTX baseline "
            f"(calibration risk: {ekf['calibration_risk']})."
        )

    return {
        "camera_fidelity": f"PSNR={psnr:.1f}dB SSIM={ssim:.3f} LPIPS={lpips:.3f}",
        "camera_assessment": camera_assessment,
        "lidar_fidelity": lidar_fidelity,
        "autonomy_impact": autonomy_impact,
        "recommendation": recommendation,
        "calibration_risk": risk,
        "key_finding": key_finding,
    }


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _format_percent(value: float) -> str:
    text = f"{value:.1%}"
    return text.replace(".0%", "%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SimVal comparison pipeline.")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--gazebo-bag", required=True)
    parser.add_argument("--isaac-dir", required=True)
    parser.add_argument("--isaac-replay-bag", default=None)
    parser.add_argument("--output-dir", default="outputs/reports")
    args = parser.parse_args()

    run_pipeline(
        scenario_id=args.scenario,
        gazebo_bag=args.gazebo_bag,
        isaac_dir=args.isaac_dir,
        isaac_replay_bag=args.isaac_replay_bag,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
