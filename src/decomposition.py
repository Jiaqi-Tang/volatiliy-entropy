"""Block-average multi-scale decomposition for return series."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_K = 11
BASE_INTERVAL_MINUTES = 5
RECONSTRUCTION_TOLERANCE = 1e-12


@dataclass(frozen=True)
class DecompositionInput:
    name: str
    input_csv: Path
    output_csv: Path


@dataclass(frozen=True)
class DecompositionPaths:
    final_csv: Path = Path("data/final/eurusd_5m_log_returns_final.csv")
    shuffle_csv: Path = Path("data/baselines/eurusd_5m_log_returns_shuffle.csv")
    gaussian_csv: Path = Path("data/baselines/eurusd_5m_log_returns_gaussian.csv")
    output_dir: Path = Path("data/decomposition")

    @property
    def report_json(self) -> Path:
        return self.output_dir / "decomposition_report.json"

    def inputs(self) -> list[DecompositionInput]:
        return [
            DecompositionInput(
                name="final",
                input_csv=self.final_csv,
                output_csv=self.output_dir / "final_decomposition.csv",
            ),
            DecompositionInput(
                name="shuffle",
                input_csv=self.shuffle_csv,
                output_csv=self.output_dir / "shuffle_decomposition.csv",
            ),
            DecompositionInput(
                name="gaussian",
                input_csv=self.gaussian_csv,
                output_csv=self.output_dir / "gaussian_decomposition.csv",
            ),
        ]


def run_decomposition(
    paths: DecompositionPaths | None = None,
    k: int = DEFAULT_K,
) -> dict[str, Any]:
    paths = paths or DecompositionPaths()
    if k < 1:
        raise ValueError("k must be at least 1")

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "K": k,
        "base_interval_minutes": BASE_INTERVAL_MINUTES,
        "block_size_max": 2**k,
        "max_scale_minutes": BASE_INTERVAL_MINUTES * (2**k),
        "max_scale_days": BASE_INTERVAL_MINUTES * (2**k) / (60 * 24),
        "reconstruction_tolerance": RECONSTRUCTION_TOLERANCE,
        "series": {},
    }

    for item in paths.inputs():
        report["series"][item.name] = decompose_csv(item, k=k)

    paths.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def decompose_csv(item: DecompositionInput, k: int) -> dict[str, Any]:
    frame = pd.read_csv(item.input_csv, usecols=["timestamp_utc", "log_return"])
    if frame.empty:
        raise ValueError(f"Input dataset is empty: {item.input_csv}")
    if frame["log_return"].isna().any():
        raise ValueError(f"Input contains NaN log_return values: {item.input_csv}")

    values = frame["log_return"].astype(float).to_numpy()
    if not np.isfinite(values).all():
        raise ValueError(f"Input contains non-finite log_return values: {item.input_csv}")

    n = len(values)
    block_size_max = 2**k
    if n % block_size_max != 0:
        raise ValueError(
            f"Input length {n} is not divisible by 2**{k} ({block_size_max}): "
            f"{item.input_csv}"
        )

    details, final_approximation = decompose_values(values, k=k)
    reconstruction = final_approximation.copy()
    for detail in details:
        reconstruction += detail
    error = values - reconstruction
    max_abs_error = float(np.max(np.abs(error)))
    mean_abs_error = float(np.mean(np.abs(error)))
    if max_abs_error > RECONSTRUCTION_TOLERANCE:
        raise ValueError(
            f"Reconstruction error {max_abs_error} exceeds tolerance "
            f"{RECONSTRUCTION_TOLERANCE} for {item.name}"
        )

    output = pd.DataFrame(
        {
            "index": np.arange(n, dtype=np.int64),
            "timestamp_utc": frame["timestamp_utc"],
            "original": values,
        }
    )
    for scale, detail in enumerate(details, start=1):
        output[f"D_{scale:02d}"] = detail
    output[f"A_{k:02d}"] = final_approximation

    item.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(item.output_csv, index=False)

    return {
        "input_csv": str(item.input_csv),
        "output_csv": str(item.output_csv),
        "N": int(n),
        "K": int(k),
        "block_size_max": int(block_size_max),
        "max_scale_minutes": int(BASE_INTERVAL_MINUTES * block_size_max),
        "max_scale_days": BASE_INTERVAL_MINUTES * block_size_max / (60 * 24),
        "timestamp_start_utc": str(frame["timestamp_utc"].iloc[0]),
        "timestamp_end_utc": str(frame["timestamp_utc"].iloc[-1]),
        "max_abs_reconstruction_error": max_abs_error,
        "mean_abs_reconstruction_error": mean_abs_error,
    }


def decompose_values(values: np.ndarray, k: int) -> tuple[list[np.ndarray], np.ndarray]:
    approximations = [values.astype(float, copy=True)]
    for scale in range(1, k + 1):
        block_size = 2**scale
        approximations.append(_expanded_block_mean(values, block_size))

    details = [
        approximations[scale - 1] - approximations[scale]
        for scale in range(1, k + 1)
    ]
    return details, approximations[k]


def _expanded_block_mean(values: np.ndarray, block_size: int) -> np.ndarray:
    if len(values) % block_size != 0:
        raise ValueError(
            f"Series length {len(values)} is not divisible by block size {block_size}"
        )
    block_means = values.reshape(-1, block_size).mean(axis=1)
    return np.repeat(block_means, block_size)
