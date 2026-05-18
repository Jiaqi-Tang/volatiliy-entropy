"""High-level entropy plot orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.globals.constants import DEFAULT_K
from src.globals.paths import ENTROPY_GAPS_CSV, ENTROPY_PLOTS_DIR, LAYER_ENTROPY_CSV
from src.plotting.primitives import plot_entropy_gaps, plot_entropy_metric
from src.plotting.readers import read_entropy_gaps, read_layer_entropy
from src.utils.validation import require_positive_k


@dataclass(frozen=True)
class EntropyPlotPaths:
    layer_entropy_csv: Path = LAYER_ENTROPY_CSV
    entropy_gaps_csv: Path = ENTROPY_GAPS_CSV
    output_dir: Path = ENTROPY_PLOTS_DIR


def create_entropy_plots(
    paths: EntropyPlotPaths | None = None,
    k: int = DEFAULT_K,
) -> list[Path]:
    paths = paths or EntropyPlotPaths()
    require_positive_k(k)
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    layer_entropy = read_layer_entropy(paths.layer_entropy_csv, k)
    entropy_gaps = read_entropy_gaps(paths.entropy_gaps_csv, k)

    return [
        plot_entropy_metric(
            layer_entropy,
            paths.output_dir / "permutation_entropy.png",
            metric="permutation_entropy",
            title="Permutation Entropy by Decomposition Component",
            ylabel="Permutation entropy",
            k=k,
        ),
        plot_entropy_metric(
            layer_entropy,
            paths.output_dir / "normalized_entropy.png",
            metric="normalized_entropy",
            title="Normalized Permutation Entropy by Decomposition Component",
            ylabel="Normalized entropy",
            k=k,
        ),
        plot_entropy_gaps(
            entropy_gaps,
            paths.output_dir / "entropy_gaps.png",
            k=k,
        ),
    ]
