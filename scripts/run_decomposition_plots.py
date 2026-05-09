from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.plots import DecompositionPlotPaths, create_decomposition_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create decomposition-layer EDA plots."
    )
    parser.add_argument(
        "--final-csv",
        type=Path,
        default=Path("data/decomposition/final_decomposition.csv"),
    )
    parser.add_argument(
        "--shuffle-csv",
        type=Path,
        default=Path("data/decomposition/shuffle_decomposition.csv"),
    )
    parser.add_argument(
        "--gaussian-csv",
        type=Path,
        default=Path("data/decomposition/gaussian_decomposition.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("plots/eda/decomposition"))
    parser.add_argument("--k", type=int, default=11)
    parser.add_argument("--short-max-acf-lag", type=int, default=1440)
    parser.add_argument("--long-max-acf-lag", type=int, default=6336)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = create_decomposition_plots(
        DecompositionPlotPaths(
            final_csv=args.final_csv,
            shuffle_csv=args.shuffle_csv,
            gaussian_csv=args.gaussian_csv,
            output_dir=args.output_dir,
        ),
        k=args.k,
        short_max_acf_lag=args.short_max_acf_lag,
        long_max_acf_lag=args.long_max_acf_lag,
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
