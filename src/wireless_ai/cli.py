from __future__ import annotations

import argparse
import json
from pathlib import Path

from .baseline import train_random_forest
from .dataset import generate_link_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Wireless AI experiment lab")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--samples", type=int, default=10000)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--output", type=Path, default=Path("results/baseline.json"))
    args = parser.parse_args()

    dataset = generate_link_dataset(args.samples, args.seed)
    _, metrics = train_random_forest(dataset, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
