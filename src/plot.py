"""EDA plots for the final EUR/USD return dataset and baselines."""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PlotPaths:
    final_csv: Path = Path("data/final_analysis/eurusd_5m_log_returns_final.csv")
    shuffle_csv: Path = Path(
        "data/final_analysis/baselines/eurusd_5m_log_returns_shuffle.csv"
    )
    gaussian_csv: Path = Path(
        "data/final_analysis/baselines/eurusd_5m_log_returns_gaussian.csv"
    )
    output_dir: Path = Path("plots/eda")


def create_eda_plots(paths: PlotPaths | None = None) -> list[Path]:
    paths = paths or PlotPaths()
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
    ]
    return outputs


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create EDA plots for EUR/USD final returns and baselines."
    )
    parser.add_argument("--final-csv", type=Path, default=PlotPaths.final_csv)
    parser.add_argument("--shuffle-csv", type=Path, default=PlotPaths.shuffle_csv)
    parser.add_argument("--gaussian-csv", type=Path, default=PlotPaths.gaussian_csv)
    parser.add_argument("--output-dir", type=Path, default=PlotPaths.output_dir)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = create_eda_plots(
        PlotPaths(
            final_csv=args.final_csv,
            shuffle_csv=args.shuffle_csv,
            gaussian_csv=args.gaussian_csv,
            output_dir=args.output_dir,
        )
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
