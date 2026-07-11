from __future__ import annotations

import numpy as np
import pytest

from metrics.camera_metrics import (
    compute_all_camera_metrics,
    compute_lpips,
    compute_psnr,
    compute_ssim,
)

N, H, W = 8, 64, 64


def make_frames(seed: int = 0, noise_std: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(30, 220, (N, H, W, 3), dtype=np.uint8)
    if noise_std > 0:
        noise = rng.normal(0, noise_std, base.shape).astype(np.int16)
        base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return base


class TestPSNR:
    def test_identical_inputs_give_very_high_psnr(self):
        frames = make_frames(seed=0)
        result = compute_psnr(frames, frames.copy())
        assert result["psnr_mean_db"] > 60.0

    def test_different_inputs_give_lower_psnr(self):
        result = compute_psnr(make_frames(seed=0), make_frames(seed=1))
        assert result["psnr_mean_db"] < 30.0

    def test_more_noise_gives_lower_psnr(self):
        ref = make_frames(seed=0)
        low = compute_psnr(ref, make_frames(seed=0, noise_std=5.0))
        high = compute_psnr(ref, make_frames(seed=0, noise_std=30.0))
        assert low["psnr_mean_db"] > high["psnr_mean_db"]

    def test_output_keys_complete(self):
        result = compute_psnr(make_frames(0), make_frames(1))
        for key in ["psnr_mean_db", "psnr_std_db", "psnr_min_db", "psnr_per_frame"]:
            assert key in result
        assert len(result["psnr_per_frame"]) == N

    def test_identical_inputs_stay_finite(self):
        result = compute_psnr(make_frames(0), make_frames(0))
        assert np.isfinite(result["psnr_mean_db"])

    def test_shape_mismatch_raises(self):
        ref = make_frames(0)
        bad = np.random.randint(0, 255, (N + 1, H, W, 3), dtype=np.uint8)
        with pytest.raises(AssertionError):
            compute_psnr(ref, bad)


class TestSSIM:
    def test_identical_inputs_give_ssim_one(self):
        frames = make_frames(seed=0)
        result = compute_ssim(frames, frames.copy())
        assert abs(result["ssim_mean"] - 1.0) < 1e-5

    def test_different_inputs_give_lower_ssim(self):
        result = compute_ssim(make_frames(0), make_frames(1))
        assert result["ssim_mean"] < 0.95

    def test_ssim_stays_in_valid_range(self):
        result = compute_ssim(make_frames(0), make_frames(1))
        assert 0.0 <= result["ssim_mean"] <= 1.0

    def test_output_keys_complete(self):
        result = compute_ssim(make_frames(0), make_frames(1))
        for key in ["ssim_mean", "ssim_std", "ssim_min", "ssim_per_frame"]:
            assert key in result


class TestLPIPS:
    def test_identical_inputs_give_near_zero_lpips(self):
        frames = make_frames(seed=0)
        result = compute_lpips(frames, frames.copy())
        assert result["lpips_mean"] < 0.05

    def test_different_inputs_give_higher_lpips(self):
        ref = make_frames(seed=0)
        same = compute_lpips(ref, ref.copy())
        diff = compute_lpips(ref, make_frames(seed=99))
        assert diff["lpips_mean"] > same["lpips_mean"] + 0.01

    def test_lpips_stays_in_valid_range(self):
        result = compute_lpips(make_frames(0), make_frames(1))
        assert 0.0 <= result["lpips_mean"] <= 1.0

    def test_note_says_lower_is_better(self):
        result = compute_lpips(make_frames(0), make_frames(1))
        assert "lower" in result["note"]

    def test_output_keys_complete(self):
        result = compute_lpips(make_frames(0), make_frames(1))
        for key in ["lpips_mean", "lpips_std", "lpips_max", "lpips_per_frame", "note"]:
            assert key in result


class TestAllCameraMetrics:
    def test_wrapper_returns_all_required_keys(self):
        result = compute_all_camera_metrics(make_frames(0), make_frames(1))
        for key in ["psnr_mean_db", "ssim_mean", "lpips_mean"]:
            assert key in result

    def test_wrapper_with_identical_inputs_is_self_consistent(self):
        frames = make_frames(0)
        result = compute_all_camera_metrics(frames, frames.copy())
        assert result["psnr_mean_db"] > 60.0
        assert result["ssim_mean"] > 0.99
        assert result["lpips_mean"] < 0.05
