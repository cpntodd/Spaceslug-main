"""Small CLI for the first dataset and tiny-model workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import verify_bundle
from .tiny_model import TinyBigramModel


def main() -> int:
    parser = argparse.ArgumentParser(prog="spaceslug")
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("dataset-verify")
    verify.add_argument("bundle", type=Path)
    train = subparsers.add_parser("tiny-train")
    train.add_argument("checkpoint", type=Path)
    train.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    if args.command == "dataset-verify":
        bundle = verify_bundle(args.bundle)
        print(f"dataset={bundle.manifest['dataset_id']} revision={bundle.manifest['revision']}")
        print(f"records={bundle.manifest['record_count']} splits={bundle.stats()}")
        return 0
    model = TinyBigramModel.create(2)
    sequences = [[0, 1, 0, 1], [0, 1, 0, 1]]
    before, _ = model.loss_and_gradients(sequences)
    for _ in range(args.steps):
        model.train_step(sequences, learning_rate=0.5)
    after, _ = model.loss_and_gradients(sequences)
    model.save(args.checkpoint)
    print(f"initial_loss={before:.9f} final_loss={after:.9f} checkpoint={args.checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
