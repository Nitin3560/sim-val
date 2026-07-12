"""TwinGuard trust-state delta metrics for SimVal."""
from __future__ import annotations

from typing import Any

import numpy as np


def compute_trust_state_delta(
    gazebo_trust: np.ndarray,
    isaac_trust: np.ndarray,
) -> dict[str, Any]:
    assert gazebo_trust.shape == isaac_trust.shape, (
        f"Shape mismatch: {gazebo_trust.shape} vs {isaac_trust.shape}"
    )
    assert gazebo_trust.ndim == 2 and gazebo_trust.shape[1] == 3, (
        f"Expected (N, 3) trust_state array, got {gazebo_trust.shape}"
    )

    g_trust = gazebo_trust[:, 0]
    i_trust = isaac_trust[:, 0]
    g_residual = gazebo_trust[:, 1]
    i_residual = isaac_trust[:, 1]
    g_authority = gazebo_trust[:, 2]
    i_authority = isaac_trust[:, 2]

    trust_delta = float(np.mean(np.abs(g_trust - i_trust)))
    residual_delta = float(np.mean(np.abs(g_residual - i_residual)))
    authority_delta = float(np.mean(np.abs(g_authority - i_authority)))
    gazebo_authority_mean = float(np.mean(g_authority))
    isaac_authority_mean = float(np.mean(i_authority))

    if authority_delta > 0.15:
        risk = "HIGH"
    elif authority_delta > 0.05:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    overestimates = gazebo_authority_mean > isaac_authority_mean
    direction = "overestimates" if overestimates else "underestimates"
    consequence = "over-permissive" if overestimates else "over-conservative"
    assessment = (
        f"Gazebo {direction} authority_scale by {authority_delta:.1%} vs "
        f"Isaac Sim RTX baseline. TwinGuard thresholds calibrated in Gazebo "
        f"are {consequence} by ~{authority_delta:.1%}."
    )

    if risk == "LOW":
        recommendation = (
            "Gazebo sensor fidelity is sufficient for TwinGuard threshold "
            "calibration in this scenario. No adjustment needed."
        )
    else:
        verb = "Tighten" if overestimates else "Relax"
        recommendation = (
            f"{verb} TwinGuard authority_scale thresholds by {authority_delta:.1%} "
            "to remain calibrated against Isaac Sim RTX ground truth."
        )

    return {
        "trust_delta_mean": trust_delta,
        "residual_delta_mean": residual_delta,
        "authority_scale_delta_mean": authority_delta,
        "gazebo_authority_mean": gazebo_authority_mean,
        "isaac_authority_mean": isaac_authority_mean,
        "calibration_risk": risk,
        "assessment": assessment,
        "recommendation": recommendation,
    }
