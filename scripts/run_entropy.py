from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.entropy import (
    DEFAULT_K,
    DELAY,
    EMBEDDING_DIMENSION,
    JITTER_MAGNITUDE,
    JITTER_SEED,
    EntropyPaths,
    compute_entropy_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute permutation entropy metrics for decomposition components."
    )
    parser.add_argument(
        "--final-decomposition-csv",
        type=Path,
        default=Path("data/decomposition/final_decomposition.csv"),
    )
    parser.add_argument(
        "--shuffle-decomposition-csv",
        type=Path,
        default=Path("data/decomposition/shuffle_decomposition.csv"),
    )
    parser.add_argument(
        "--gaussian-decomposition-csv",
        type=Path,
        default=Path("data/decomposition/gaussian_decomposition.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/entropy"))
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--embedding-dimension", type=int, default=EMBEDDING_DIMENSION)
    parser.add_argument("--delay", type=int, default=DELAY)
    parser.add_argument("--jitter-seed", type=int, default=JITTER_SEED)
    parser.add_argument("--jitter-magnitude", type=float, default=JITTER_MAGNITUDE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compute_entropy_metrics(
        EntropyPaths(
            final_decomposition_csv=args.final_decomposition_csv,
            shuffle_decomposition_csv=args.shuffle_decomposition_csv,
            gaussian_decomposition_csv=args.gaussian_decomposition_csv,
            output_dir=args.output_dir,
        ),
        k=args.k,
        embedding_dimension=args.embedding_dimension,
        delay=args.delay,
        jitter_seed=args.jitter_seed,
        jitter_magnitude=args.jitter_magnitude,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
