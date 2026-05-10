"""Permutation entropy metrics for decomposition components."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.scale_utils import (
    component_repeat_length,
    component_scale,
    component_scale_minutes,
    component_type,
    compress_component,
    decomposition_components,
)


DEFAULT_K = 11
BASE_INTERVAL_MINUTES = 5
EMBEDDING_DIMENSION = 3
DELAY = 1
JITTER_SEED = 314
JITTER_MAGNITUDE = 1e-10


@dataclass(frozen=True)
class EntropyInput:
    name: str
    decomposition_csv: Path


@dataclass(frozen=True)
class EntropyPaths:
    final_decomposition_csv: Path = Path("data/decomposition/final_decomposition.csv")
    shuffle_decomposition_csv: Path = Path("data/decomposition/shuffle_decomposition.csv")
    gaussian_decomposition_csv: Path = Path("data/decomposition/gaussian_decomposition.csv")
    output_dir: Path = Path("results/entropy")

    @property
    def layer_entropy_csv(self) -> Path:
        return self.output_dir / "layer_entropy.csv"

    @property
    def entropy_gaps_csv(self) -> Path:
        return self.output_dir / "entropy_gaps.csv"

    @property
    def report_json(self) -> Path:
        return self.output_dir / "entropy_report.json"

    def inputs(self) -> list[EntropyInput]:
        return [
            EntropyInput("final", self.final_decomposition_csv),
            EntropyInput("shuffle", self.shuffle_decomposition_csv),
            EntropyInput("gaussian", self.gaussian_decomposition_csv),
        ]


def compute_entropy_metrics(
    paths: EntropyPaths | None = None,
    k: int = DEFAULT_K,
    embedding_dimension: int = EMBEDDING_DIMENSION,
    delay: int = DELAY,
    jitter_seed: int = JITTER_SEED,
    jitter_magnitude: float = JITTER_MAGNITUDE,
) -> dict[str, Any]:
    paths = paths or EntropyPaths()
    if k < 1:
        raise ValueError("k must be at least 1")
    if embedding_dimension < 2:
        raise ValueError("embedding_dimension must be at least 2")
    if delay < 1:
        raise ValueError("delay must be at least 1")
    if jitter_magnitude < 0:
        raise ValueError("jitter_magnitude must be non-negative")

    paths.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    pattern_counts: dict[str, dict[str, dict[str, int]]] = {}
    series_report: dict[str, Any] = {}
    for item in paths.inputs():
        item_rows, item_counts, item_report = _compute_series_entropy(
            item,
            k=k,
            embedding_dimension=embedding_dimension,
            delay=delay,
            jitter_seed=jitter_seed,
            jitter_magnitude=jitter_magnitude,
        )
        rows.extend(item_rows)
        pattern_counts[item.name] = item_counts
        series_report[item.name] = item_report

    layer_entropy = pd.DataFrame(rows)
    layer_entropy.to_csv(paths.layer_entropy_csv, index=False)

    entropy_gaps = _compute_entropy_gaps(layer_entropy)
    entropy_gaps.to_csv(paths.entropy_gaps_csv, index=False)

    report = {
        "K": k,
        "base_interval_minutes": BASE_INTERVAL_MINUTES,
        "embedding_dimension": embedding_dimension,
        "delay": delay,
        "log_base": "natural",
        "normalization_denominator": math.log(math.factorial(embedding_dimension)),
        "jitter_seed": jitter_seed,
        "jitter_magnitude": jitter_magnitude,
        "compression_rule": (
            "Deterministic repeated block values are removed before entropy "
            "calculation using component-specific repeat lengths."
        ),
        "layer_entropy_csv": str(paths.layer_entropy_csv),
        "entropy_gaps_csv": str(paths.entropy_gaps_csv),
        "series": series_report,
        "pattern_counts": pattern_counts,
    }
    paths.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _compute_series_entropy(
    item: EntropyInput,
    k: int,
    embedding_dimension: int,
    delay: int,
    jitter_seed: int,
    jitter_magnitude: float,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]], dict[str, Any]]:
    components = decomposition_components(k, include_original=False)
    frame = pd.read_csv(item.decomposition_csv, usecols=components)
    if frame.empty:
        raise ValueError(f"Decomposition file is empty: {item.decomposition_csv}")

    rows: list[dict[str, Any]] = []
    pattern_counts: dict[str, dict[str, int]] = {}
    component_report: dict[str, Any] = {}

    for component in components:
        values = frame[component].astype(float).to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(
                f"Component {component} has non-finite values: {item.decomposition_csv}"
            )

        compressed = compress_component(values, component)
        if len(compressed) < embedding_dimension:
            raise ValueError(
                f"Compressed component too short for entropy: {item.name} {component}"
            )

        component_seed = _component_jitter_seed(jitter_seed, item.name, component)
        jittered = _add_jitter(compressed, component_seed, jitter_magnitude)
        entropy_result = _permutation_entropy(jittered, embedding_dimension, delay)

        scale = component_scale(component)
        scale_minutes = component_scale_minutes(component, BASE_INTERVAL_MINUTES)
        repeat_length = component_repeat_length(component)
        kind = component_type(component)
        rows.append(
            {
                "series": item.name,
                "component": component,
                "k": scale,
                "component_type": kind,
                "scale_minutes": scale_minutes,
                "scale_days": scale_minutes / (60 * 24),
                "repeat_length": repeat_length,
                "effective_n": len(compressed),
                "ordinal_windows": entropy_result["ordinal_windows"],
                "permutation_entropy": entropy_result["permutation_entropy"],
                "normalized_entropy": entropy_result["normalized_entropy"],
            }
        )
        pattern_counts[component] = entropy_result["pattern_counts"]
        component_report[component] = {
            "repeat_length": repeat_length,
            "expanded_n": int(len(values)),
            "effective_n": int(len(compressed)),
            "ordinal_windows": entropy_result["ordinal_windows"],
            "component_jitter_seed": component_seed,
            "compressed_unique_values": int(pd.Series(compressed).nunique()),
        }

    report = {
        "input_csv": str(item.decomposition_csv),
        "expanded_N": int(len(frame)),
        "components": component_report,
    }
    return rows, pattern_counts, report


def _permutation_entropy(
    values: np.ndarray,
    embedding_dimension: int,
    delay: int,
) -> dict[str, Any]:
    ordinal_windows = len(values) - (embedding_dimension - 1) * delay
    if ordinal_windows <= 0:
        raise ValueError("Series is too short for permutation entropy")

    permutations = list(itertools.permutations(range(embedding_dimension)))
    counts = {"".join(str(index) for index in pattern): 0 for pattern in permutations}

    for start in range(ordinal_windows):
        window = values[start : start + embedding_dimension * delay : delay]
        pattern = tuple(np.argsort(window, kind="mergesort"))
        counts["".join(str(index) for index in pattern)] += 1

    probabilities = np.array(
        [count / ordinal_windows for count in counts.values() if count > 0],
        dtype=float,
    )
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    normalized_entropy = float(entropy / math.log(math.factorial(embedding_dimension)))
    return {
        "ordinal_windows": int(ordinal_windows),
        "pattern_counts": counts,
        "permutation_entropy": entropy,
        "normalized_entropy": normalized_entropy,
    }


def _compute_entropy_gaps(layer_entropy: pd.DataFrame) -> pd.DataFrame:
    index_columns = [
        "component",
        "k",
        "component_type",
        "scale_minutes",
        "scale_days",
        "repeat_length",
    ]
    wide = layer_entropy.pivot_table(
        index=index_columns,
        columns="series",
        values="normalized_entropy",
        aggfunc="first",
    ).reset_index()
    required_series = ["final", "shuffle", "gaussian"]
    missing_series = [series for series in required_series if series not in wide.columns]
    if missing_series:
        raise ValueError(f"Missing entropy series for gap calculation: {missing_series}")

    wide = wide.rename(
        columns={
            "final": "final_normalized_entropy",
            "shuffle": "shuffle_normalized_entropy",
            "gaussian": "gaussian_normalized_entropy",
        }
    )
    wide["entropy_gap_shuffle"] = (
        wide["shuffle_normalized_entropy"] - wide["final_normalized_entropy"]
    )
    wide["entropy_gap_gaussian"] = (
        wide["gaussian_normalized_entropy"] - wide["final_normalized_entropy"]
    )
    return wide[
        [
            *index_columns,
            "final_normalized_entropy",
            "shuffle_normalized_entropy",
            "gaussian_normalized_entropy",
            "entropy_gap_shuffle",
            "entropy_gap_gaussian",
        ]
    ].sort_values("k")


def _add_jitter(
    values: np.ndarray,
    seed: int,
    magnitude: float,
) -> np.ndarray:
    if magnitude == 0:
        return values.copy()
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-magnitude, magnitude, size=len(values))
    return values + jitter


def _component_jitter_seed(base_seed: int, series: str, component: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{series}:{component}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (2**32)
