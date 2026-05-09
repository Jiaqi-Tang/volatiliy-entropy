"""EDA plots for the final EUR/USD return dataset and baselines."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.scale_utils import (
    compress_component,
    component_repeat_length,
    decomposition_components,
    original_lags_from_compressed_lags,
)


@dataclass(frozen=True)
class PlotPaths:
    final_csv: Path = Path("data/final/eurusd_5m_log_returns_final.csv")
    shuffle_csv: Path = Path(
        "data/baselines/eurusd_5m_log_returns_shuffle.csv"
    )
    gaussian_csv: Path = Path(
        "data/baselines/eurusd_5m_log_returns_gaussian.csv"
    )
    output_dir: Path = Path("plots/eda/returns")


@dataclass(frozen=True)
class DecompositionPlotPaths:
    final_csv: Path = Path("data/decomposition/final_decomposition.csv")
    shuffle_csv: Path = Path("data/decomposition/shuffle_decomposition.csv")
    gaussian_csv: Path = Path("data/decomposition/gaussian_decomposition.csv")
    output_dir: Path = Path("plots/eda/decomposition")


def create_eda_plots(paths: PlotPaths | None = None, max_acf_lag: int = 288) -> list[Path]:
    paths = paths or PlotPaths()
    if max_acf_lag < 1:
        raise ValueError("max_acf_lag must be at least 1")
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    final_returns = _read_returns(paths.final_csv)
    shuffle_returns = _read_returns(paths.shuffle_csv)
    gaussian_returns = _read_returns(paths.gaussian_csv)

    outputs = [
        plot_return_line(
            final_returns,
            paths.output_dir / "final_returns_line.png",
            title="EUR/USD Final 5m Log Returns",
        ),
        plot_return_line(
            shuffle_returns,
            paths.output_dir / "shuffle_returns_line.png",
            title="Shuffled Baseline 5m Log Returns",
        ),
        plot_return_line(
            gaussian_returns,
            paths.output_dir / "gaussian_returns_line.png",
            title="Gaussian Baseline 5m Log Returns",
        ),
        plot_histogram_comparison(
            final_returns,
            gaussian_returns,
            paths.output_dir / "final_vs_gaussian_histogram.png",
        ),
        plot_ecdf_comparison(
            final_returns,
            gaussian_returns,
            paths.output_dir / "final_vs_gaussian_ecdf.png",
        ),
        plot_qq_against_zero_mean_gaussian(
            final_returns,
            paths.output_dir / "final_qq_gaussian.png",
        ),
        plot_acf_comparison(
            final_returns,
            shuffle_returns,
            gaussian_returns,
            paths.output_dir / "final_vs_baselines_returns_acf.png",
            max_lag=max_acf_lag,
            title="Autocorrelation of 5m Log Returns",
            transform_label="returns",
        ),
        plot_acf_comparison(
            np.abs(final_returns),
            np.abs(shuffle_returns),
            np.abs(gaussian_returns),
            paths.output_dir / "final_vs_baselines_abs_returns_acf.png",
            max_lag=max_acf_lag,
            title="Autocorrelation of Absolute 5m Log Returns",
            transform_label="absolute returns",
        ),
    ]
    return outputs


def create_decomposition_plots(
    paths: DecompositionPlotPaths | None = None,
    k: int = 11,
    short_max_acf_lag: int = 1440,
    long_max_acf_lag: int = 6336,
) -> list[Path]:
    paths = paths or DecompositionPlotPaths()
    if k < 1:
        raise ValueError("k must be at least 1")
    if short_max_acf_lag < 1:
        raise ValueError("short_max_acf_lag must be at least 1")
    if long_max_acf_lag < 1:
        raise ValueError("long_max_acf_lag must be at least 1")
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    final_frame = _read_decomposition(paths.final_csv, k)
    shuffle_frame = _read_decomposition(paths.shuffle_csv, k)
    gaussian_frame = _read_decomposition(paths.gaussian_csv, k)

    return [
        plot_decomposition_layers(
            final_frame,
            paths.output_dir / "final_layers.png",
            title="EUR/USD Final Decomposition Layers",
            k=k,
        ),
        plot_decomposition_layers(
            shuffle_frame,
            paths.output_dir / "shuffle_layers.png",
            title="Shuffled Baseline Decomposition Layers",
            k=k,
        ),
        plot_decomposition_layers(
            gaussian_frame,
            paths.output_dir / "gaussian_layers.png",
            title="Gaussian Baseline Decomposition Layers",
            k=k,
        ),
        plot_layer_histogram_grid(
            final_frame,
            gaussian_frame,
            paths.output_dir / "layer_histograms_grid.png",
            k=k,
        ),
        plot_layer_qq_grid(
            final_frame,
            paths.output_dir / "layer_qq_gaussian_grid.png",
            k=k,
        ),
        plot_layer_acf_grid(
            final_frame,
            shuffle_frame,
            gaussian_frame,
            paths.output_dir / "layer_acf_returns_short_scales.png",
            layers=[f"D_{scale:02d}" for scale in range(1, min(k, 6) + 1)],
            max_lag=short_max_acf_lag,
            absolute=False,
            title="Short-Scale Layer Autocorrelation",
        ),
        plot_layer_acf_grid(
            final_frame,
            shuffle_frame,
            gaussian_frame,
            paths.output_dir / "layer_acf_abs_returns_short_scales.png",
            layers=[f"D_{scale:02d}" for scale in range(1, min(k, 6) + 1)],
            max_lag=short_max_acf_lag,
            absolute=True,
            title="Short-Scale Absolute Layer Autocorrelation",
        ),
        plot_layer_acf_grid(
            final_frame,
            shuffle_frame,
            gaussian_frame,
            paths.output_dir / "layer_acf_returns_long_scales.png",
            layers=[f"D_{scale:02d}" for scale in range(7, k + 1)] + [f"A_{k:02d}"],
            max_lag=long_max_acf_lag,
            absolute=False,
            title="Long-Scale Layer Autocorrelation",
        ),
        plot_layer_acf_grid(
            final_frame,
            shuffle_frame,
            gaussian_frame,
            paths.output_dir / "layer_acf_abs_returns_long_scales.png",
            layers=[f"D_{scale:02d}" for scale in range(7, k + 1)] + [f"A_{k:02d}"],
            max_lag=long_max_acf_lag,
            absolute=True,
            title="Long-Scale Absolute Layer Autocorrelation",
        ),
    ]


def plot_return_line(returns: np.ndarray, output_path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(np.arange(len(returns)), returns, linewidth=0.25, alpha=0.75)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Observation index")
    ax.set_ylabel("Log return")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_histogram_comparison(
    final_returns: np.ndarray,
    gaussian_returns: np.ndarray,
    output_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = 300
    ax.hist(
        final_returns,
        bins=bins,
        density=True,
        alpha=0.55,
        label="EUR/USD final",
        color="#2f6f9f",
    )
    ax.hist(
        gaussian_returns,
        bins=bins,
        density=True,
        alpha=0.45,
        label="Gaussian baseline",
        color="#c76d3b",
    )

    _add_mean_median_lines(ax, final_returns, "EUR/USD", "#17486b")
    _add_mean_median_lines(ax, gaussian_returns, "Gaussian", "#8d3f19")

    ax.set_title("Distribution of 5m Log Returns")
    ax.set_xlabel("Log return")
    ax.set_ylabel("Density")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_ecdf_comparison(
    final_returns: np.ndarray,
    gaussian_returns: np.ndarray,
    output_path: Path,
) -> Path:
    final_x, final_y = _ecdf(final_returns)
    gaussian_x, gaussian_y = _ecdf(gaussian_returns)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(final_x, final_y, linewidth=1.2, label="EUR/USD final", color="#2f6f9f")
    ax.plot(
        gaussian_x,
        gaussian_y,
        linewidth=1.2,
        label="Gaussian baseline",
        color="#c76d3b",
    )
    ax.set_title("Empirical CDF of 5m Log Returns")
    ax.set_xlabel("Log return")
    ax.set_ylabel("Cumulative probability")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_qq_against_zero_mean_gaussian(
    returns: np.ndarray,
    output_path: Path,
    quantile_points: int = 2_000,
) -> Path:
    variance = float(np.var(returns, ddof=0))
    std = float(np.sqrt(variance))
    probabilities = np.linspace(
        1.0 / (quantile_points + 1),
        quantile_points / (quantile_points + 1),
        quantile_points,
    )
    empirical_quantiles = np.quantile(returns, probabilities)
    gaussian_quantiles = np.array(
        [statistics.NormalDist(mu=0.0, sigma=std).inv_cdf(p) for p in probabilities]
    )

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        gaussian_quantiles,
        empirical_quantiles,
        s=4,
        alpha=0.35,
        color="#2f6f9f",
        edgecolors="none",
    )
    low = min(float(gaussian_quantiles.min()), float(empirical_quantiles.min()))
    high = max(float(gaussian_quantiles.max()), float(empirical_quantiles.max()))
    ax.plot([low, high], [low, high], color="black", linewidth=1.0, alpha=0.8)
    ax.set_title("QQ Plot: EUR/USD Final Returns vs N(0, empirical variance)")
    ax.set_xlabel("Gaussian theoretical quantile")
    ax.set_ylabel("EUR/USD empirical quantile")
    ax.ticklabel_format(axis="both", style="sci", scilimits=(-3, 3))
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_acf_comparison(
    final_values: np.ndarray,
    shuffle_values: np.ndarray,
    gaussian_values: np.ndarray,
    output_path: Path,
    max_lag: int,
    title: str,
    transform_label: str,
) -> Path:
    final_acf = _autocorrelation(final_values, max_lag)
    shuffle_acf = _autocorrelation(shuffle_values, max_lag)
    gaussian_acf = _autocorrelation(gaussian_values, max_lag)
    lags = np.arange(1, max_lag + 1)
    band = 1.96 / np.sqrt(len(final_values))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(lags, final_acf, linewidth=1.2, label="EUR/USD final", color="#2f6f9f")
    ax.plot(
        lags,
        shuffle_acf,
        linewidth=1.0,
        label="Shuffled baseline",
        color="#5c9f52",
        alpha=0.9,
    )
    ax.plot(
        lags,
        gaussian_acf,
        linewidth=1.0,
        label="Gaussian baseline",
        color="#c76d3b",
        alpha=0.9,
    )
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.8)
    ax.axhline(band, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(-band, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Lag")
    ax.set_ylabel(f"ACF of {transform_label}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_decomposition_layers(
    frame: pd.DataFrame,
    output_path: Path,
    title: str,
    k: int,
) -> Path:
    layers = decomposition_components(k, include_original=True)
    x = frame["index"].to_numpy()

    fig, axes = plt.subplots(
        len(layers),
        1,
        figsize=(16, 24),
        sharex=True,
        constrained_layout=True,
    )
    fig.suptitle(title, fontsize=16)

    for axis, layer in zip(axes, layers):
        axis.plot(x, frame[layer].to_numpy(), linewidth=0.2, alpha=0.8)
        axis.axhline(0.0, color="black", linewidth=0.6, alpha=0.55)
        axis.set_ylabel(layer)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

    axes[-1].set_xlabel("Observation index")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_layer_histogram_grid(
    final_frame: pd.DataFrame,
    gaussian_frame: pd.DataFrame,
    output_path: Path,
    k: int,
) -> Path:
    layers = decomposition_components(k, include_original=False)
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes_flat = axes.ravel()

    for axis, layer in zip(axes_flat, layers):
        final_values = final_frame[layer].to_numpy()
        gaussian_values = gaussian_frame[layer].to_numpy()
        axis.hist(
            final_values,
            bins=180,
            density=True,
            alpha=0.55,
            label="EUR/USD",
            color="#2f6f9f",
        )
        axis.hist(
            gaussian_values,
            bins=180,
            density=True,
            alpha=0.45,
            label="Gaussian",
            color="#c76d3b",
        )
        axis.axvline(np.mean(final_values), color="#17486b", linewidth=0.8)
        axis.axvline(np.median(final_values), color="#17486b", linestyle="--", linewidth=0.8)
        axis.set_title(layer)
        axis.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))

    for axis in axes_flat[len(layers):]:
        axis.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Layer Distributions: EUR/USD vs Gaussian Baseline", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_layer_qq_grid(
    final_frame: pd.DataFrame,
    output_path: Path,
    k: int,
    quantile_points: int = 2_000,
) -> Path:
    layers = decomposition_components(k, include_original=False)
    probabilities = np.linspace(
        1.0 / (quantile_points + 1),
        quantile_points / (quantile_points + 1),
        quantile_points,
    )

    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes_flat = axes.ravel()

    for axis, layer in zip(axes_flat, layers):
        values = final_frame[layer].to_numpy()
        std = float(np.sqrt(np.var(values, ddof=0)))
        empirical_quantiles = np.quantile(values, probabilities)
        gaussian_quantiles = np.array(
            [statistics.NormalDist(mu=0.0, sigma=std).inv_cdf(p) for p in probabilities]
        )
        axis.scatter(
            gaussian_quantiles,
            empirical_quantiles,
            s=3,
            alpha=0.3,
            color="#2f6f9f",
            edgecolors="none",
        )
        low = min(float(gaussian_quantiles.min()), float(empirical_quantiles.min()))
        high = max(float(gaussian_quantiles.max()), float(empirical_quantiles.max()))
        axis.plot([low, high], [low, high], color="black", linewidth=0.8, alpha=0.75)
        axis.set_title(layer)
        axis.ticklabel_format(axis="both", style="sci", scilimits=(-3, 3))

    for axis in axes_flat[len(layers):]:
        axis.axis("off")

    fig.suptitle("Layer QQ Plots vs N(0, Layer Variance)", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_layer_acf_grid(
    final_frame: pd.DataFrame,
    shuffle_frame: pd.DataFrame,
    gaussian_frame: pd.DataFrame,
    output_path: Path,
    layers: list[str],
    max_lag: int,
    absolute: bool,
    title: str,
) -> Path:
    fig, axes = plt.subplots(
        len(layers),
        1,
        figsize=(16, 22),
        sharex=True,
        constrained_layout=True,
    )
    fig.suptitle(f"{title} (compressed repeated block values for plotting)", fontsize=16)

    for axis, layer in zip(axes, layers):
        final_values = final_frame[layer].to_numpy()
        shuffle_values = shuffle_frame[layer].to_numpy()
        gaussian_values = gaussian_frame[layer].to_numpy()
        compressed_n = len(compress_component(final_values, layer))
        band = 1.96 / np.sqrt(compressed_n)
        if absolute:
            final_values = np.abs(final_values)
            shuffle_values = np.abs(shuffle_values)
            gaussian_values = np.abs(gaussian_values)

        final_lags, final_acf = _compressed_layer_autocorrelation(
            final_values, layer, max_lag
        )
        shuffle_lags, shuffle_acf = _compressed_layer_autocorrelation(
            shuffle_values, layer, max_lag
        )
        gaussian_lags, gaussian_acf = _compressed_layer_autocorrelation(
            gaussian_values, layer, max_lag
        )

        axis.plot(final_lags, final_acf, linewidth=1.0, label="EUR/USD")
        axis.plot(
            shuffle_lags,
            shuffle_acf,
            linewidth=0.9,
            label="Shuffled",
            alpha=0.9,
        )
        axis.plot(
            gaussian_lags,
            gaussian_acf,
            linewidth=0.9,
            label="Gaussian",
            alpha=0.9,
        )
        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.75)
        axis.axhline(band, color="black", linestyle="--", linewidth=0.7, alpha=0.45)
        axis.axhline(-band, color="black", linestyle="--", linewidth=0.7, alpha=0.45)
        axis.set_ylabel(layer)

    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Lag")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _read_returns(path: Path) -> np.ndarray:
    frame = pd.read_csv(path, usecols=["log_return"])
    if frame.empty:
        raise ValueError(f"Return file is empty: {path}")
    return frame["log_return"].astype(float).to_numpy()


def _add_mean_median_lines(
    ax: plt.Axes,
    returns: np.ndarray,
    label_prefix: str,
    color: str,
) -> None:
    mean = float(np.mean(returns))
    median = float(np.median(returns))
    ax.axvline(mean, color=color, linestyle="-", linewidth=1.0, alpha=0.85)
    ax.axvline(median, color=color, linestyle="--", linewidth=1.0, alpha=0.85)
    ax.plot([], [], color=color, linestyle="-", label=f"{label_prefix} mean")
    ax.plot([], [], color=color, linestyle="--", label=f"{label_prefix} median")


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def _read_decomposition(path: Path, k: int) -> pd.DataFrame:
    columns = ["index", "original"] + decomposition_components(k, include_original=False)
    frame = pd.read_csv(path, usecols=columns)
    missing_columns = [column for column in columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing decomposition columns in {path}: {missing_columns}")
    return frame


def _compressed_layer_autocorrelation(
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
    return original_lags, _autocorrelation(compressed, max_compressed_lag)


def _autocorrelation(values: np.ndarray, max_lag: int) -> np.ndarray:
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
