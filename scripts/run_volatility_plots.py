from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.plots import VolatilityPlotPaths, create_volatility_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create volatility and energy plots from layer volatility results."
    )
    parser.add_argument(
        "--volatility-csv",
        type=Path,
        default=Path("results/volatility/layer_volatility.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/results/volatility"),
    )
    parser.add_argument("--k", type=int, default=11)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = create_volatility_plots(
        VolatilityPlotPaths(
            volatility_csv=args.volatility_csv,
            output_dir=args.output_dir,
        ),
        k=args.k,
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
