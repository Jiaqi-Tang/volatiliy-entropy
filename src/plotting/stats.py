"""Statistical helpers used by plotting functions."""

from __future__ import annotations

import statistics

import numpy as np

from src.scale_utils import (
    compress_component,
    component_repeat_length,
    original_lags_from_compressed_lags,
)


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def normal_quantiles_for_values(
    values: np.ndarray,
    quantile_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    std = float(np.sqrt(np.var(values, ddof=0)))
    probabilities = np.linspace(
        1.0 / (quantile_points + 1),
        quantile_points / (quantile_points + 1),
        quantile_points,
    )
    empirical_quantiles = np.quantile(values, probabilities)
    gaussian_quantiles = np.array(
        [statistics.NormalDist(mu=0.0, sigma=std).inv_cdf(p) for p in probabilities]
    )
    return gaussian_quantiles, empirical_quantiles


def autocorrelation(values: np.ndarray, max_lag: int) -> np.ndarray:
    if max_lag >= len(values):
        raise ValueError("max_lag must be smaller than the series length")

    centered = values.astype(float) - float(np.mean(values))
    denom = float(np.dot(centered, centered))
    if denom == 0.0:
        raise ValueError("Cannot compute autocorrelation for a constant series")

    acf = np.empty(max_lag, dtype=float)
    for lag in range(1, max_lag + 1):
        acf[lag - 1] = float(np.dot(centered[lag:], centered[:-lag]) / denom)
    return acf


def compressed_layer_autocorrelation(
    values: np.ndarray,
    layer: str,
    max_original_lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    repeat_length = component_repeat_length(layer)
    compressed = compress_component(values, layer)
    max_compressed_lag = max_original_lag // repeat_length
    if max_compressed_lag < 1:
        raise ValueError(
            f"max_original_lag={max_original_lag} is too small for {layer} "
            f"with repeat length {repeat_length}"
        )
    max_compressed_lag = min(max_compressed_lag, len(compressed) - 1)
    compressed_lags = np.arange(1, max_compressed_lag + 1)
    original_lags = original_lags_from_compressed_lags(compressed_lags, layer)
    return original_lags, autocorrelation(compressed, max_compressed_lag)

