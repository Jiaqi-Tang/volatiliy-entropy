from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.decomposition import DEFAULT_K, DecompositionPaths, run_decomposition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create block-average multi-scale decompositions."
    )
    parser.add_argument(
        "--final-csv",
        type=Path,
        default=Path("data/final/eurusd_5m_log_returns_final.csv"),
    )
    parser.add_argument(
        "--shuffle-csv",
        type=Path,
        default=Path("data/baselines/eurusd_5m_log_returns_shuffle.csv"),
    )
    parser.add_argument(
        "--gaussian-csv",
        type=Path,
        default=Path("data/baselines/eurusd_5m_log_returns_gaussian.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/decomposition"))
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_decomposition(
        DecompositionPaths(
            final_csv=args.final_csv,
            shuffle_csv=args.shuffle_csv,
            gaussian_csv=args.gaussian_csv,
            output_dir=args.output_dir,
        ),
        k=args.k,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
