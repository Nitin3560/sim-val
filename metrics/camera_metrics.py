"""Camera fidelity metrics for SimVal."""
from __future__ import annotations

from typing import Any

import lpips
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity as _ssim

_LPIPS_FN: lpips.LPIPS | None = None


def _lpips_fn() -> lpips.LPIPS:
    global _LPIPS_FN
    if _LPIPS_FN is None:
        _LPIPS_FN = lpips.LPIPS(net="alex")
        _LPIPS_FN.eval()
    return _LPIPS_FN


def _assert_rgb_sequence(reference: np.ndarray, query: np.ndarray) -> None:
    assert reference.shape == query.shape, (
        f"Shape mismatch: {reference.shape} vs {query.shape}"
    )
    assert reference.ndim == 4 and reference.shape[-1] == 3, (
        f"Expected (N,H,W,3), got {reference.shape}"
    )


def _to_lpips_tensor(frame: np.ndarray) -> torch.Tensor:
    frame = np.ascontiguousarray(frame)
    tensor = torch.from_numpy(frame).float() / 127.5 - 1.0
    return tensor.permute(2, 0, 1).unsqueeze(0)


def compute_psnr(reference: np.ndarray, query: np.ndarray) -> dict[str, Any]:
    _assert_rgb_sequence(reference, query)
    scores = []
    for r, q in zip(reference, query):
        score = 100.0 if np.array_equal(r, q) else float(_psnr(r, q, data_range=255))
        scores.append(score)
    return {
        "psnr_mean_db": float(np.mean(scores)),
        "psnr_std_db": float(np.std(scores)),
        "psnr_min_db": float(np.min(scores)),
        "psnr_per_frame": scores,
    }


def compute_ssim(reference: np.ndarray, query: np.ndarray) -> dict[str, Any]:
    _assert_rgb_sequence(reference, query)
    scores = [
        float(_ssim(r, q, channel_axis=2, data_range=255))
        for r, q in zip(reference, query)
    ]
    return {
        "ssim_mean": float(np.mean(scores)),
        "ssim_std": float(np.std(scores)),
        "ssim_min": float(np.min(scores)),
        "ssim_per_frame": scores,
    }


def compute_lpips(reference: np.ndarray, query: np.ndarray) -> dict[str, Any]:
    _assert_rgb_sequence(reference, query)
    scores: list[float] = []
    metric = _lpips_fn()
    with torch.no_grad():
        for r, q in zip(reference, query):
            score = metric(_to_lpips_tensor(r), _to_lpips_tensor(q)).item()
            scores.append(float(score))
    return {
        "lpips_mean": float(np.mean(scores)),
        "lpips_std": float(np.std(scores)),
        "lpips_max": float(np.max(scores)),
        "lpips_per_frame": scores,
        "note": "lower = more perceptually similar",
    }


def compute_all_camera_metrics(
    reference_rgb: np.ndarray,
    query_rgb: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    result.update(compute_psnr(reference_rgb, query_rgb))
    result.update(compute_ssim(reference_rgb, query_rgb))
    result.update(compute_lpips(reference_rgb, query_rgb))
    return result
