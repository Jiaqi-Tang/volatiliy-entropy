"""CSV readers and schema checks for plotting inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.globals.columns import COMPONENT, COMPONENT_TYPE, INDEX, LOG_RETURN, ORIGINAL, SERIES
from src.scale_utils import decomposition_components
from src.utils.validation import require_columns


def read_returns(path: Path) -> np.ndarray:
    frame = pd.read_csv(path, usecols=[LOG_RETURN])
    if frame.empty:
        raise ValueError(f"Return file is empty: {path}")
    return frame[LOG_RETURN].astype(float).to_numpy()


def read_decomposition(path: Path, k: int) -> pd.DataFrame:
    columns = [INDEX, ORIGINAL] + decomposition_components(k, include_original=False)
    frame = pd.read_csv(path, usecols=columns)
    require_columns(frame, columns, path)
    return frame


def read_volatility(path: Path, k: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required_columns = {
        SERIES,
        COMPONENT,
        COMPONENT_TYPE,
        "detail_energy_share",
        "total_component_energy_share",
        "rms_volatility",
        "annualized_rms_volatility",
    }
    require_columns(frame, required_columns, path)
    validate_components(frame, path, k)
    return frame


def read_layer_entropy(path: Path, k: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required_columns = {
        SERIES,
        COMPONENT,
        "permutation_entropy",
        "normalized_entropy",
    }
    require_columns(frame, required_columns, path)
    validate_components(frame, path, k)
    return frame


def read_entropy_gaps(path: Path, k: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required_columns = {
        COMPONENT,
        "entropy_gap_shuffle",
        "entropy_gap_gaussian",
    }
    require_columns(frame, required_columns, path)
    validate_components(frame, path, k)
    return frame


def validate_components(frame: pd.DataFrame, path: Path, k: int) -> None:
    expected_components = set(decomposition_components(k, include_original=False))
    unexpected_components = sorted(set(frame[COMPONENT]).difference(expected_components))
    if unexpected_components:
        raise ValueError(f"Unexpected components in {path}: {unexpected_components}")
