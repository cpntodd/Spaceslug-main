"""Small CLI for the first dataset and tiny-model workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import verify_bundle
from .tiny_model import TinyBigramModel
from .tiny_artifact import write_tiny_artifact
from .tiny_training import TinyTrainingConfig, save_training_checkpoint, train_tiny
from .tiny_dense_training import DenseTinyTrainingConfig, load_dense_checkpoint, save_dense_checkpoint, train_dense_tiny, write_experiment_record
from .tiny_dense_artifact import write_dense_tiny_artifact
from .projected_attention_training import ProjectedAttentionConfig, run_projected_training
from .regression import compare_reports
from .regression_series import load_regression_series, write_regression_series
from .tui import SpaceslugTui
from .cpu_verification import verify_cpu_training
from .inference_session import InferenceSession
from .projected_attention_reference import ProjectedTinyAttentionModel
from .tokenizer import default_tokenizer


def main() -> int:
    parser = argparse.ArgumentParser(prog="spaceslug")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("tui")
    cpu_gate = subparsers.add_parser("tiny-cpu-verify")
    cpu_gate.add_argument("bundle", type=Path)
    cpu_gate.add_argument("--steps", type=int, default=2)
    cpu_gate.add_argument("--learning-rate", type=float, default=0.1)
    infer = subparsers.add_parser("tiny-infer")
    infer.add_argument("tokens", nargs="+", type=int)
    gpu_chain = subparsers.add_parser("tiny-gpu-chain")
    gpu_chain.add_argument("tokens", nargs="+", type=int)
    verify = subparsers.add_parser("dataset-verify")
    verify.add_argument("bundle", type=Path)
    train = subparsers.add_parser("tiny-train")
    train.add_argument("checkpoint", type=Path)
    train.add_argument("--steps", type=int, default=20)
    dataset_train = subparsers.add_parser("tiny-train-dataset")
    dataset_train.add_argument("bundle", type=Path)
    dataset_train.add_argument("checkpoint", type=Path)
    dataset_train.add_argument("artifact", type=Path)
    dataset_train.add_argument("--steps", type=int, default=20)
    dataset_train.add_argument("--learning-rate", type=float, default=0.2)
    dense_train = subparsers.add_parser("tiny-dense-train")
    dense_train.add_argument("bundle", type=Path)
    dense_train.add_argument("checkpoint", type=Path)
    dense_train.add_argument("artifact", type=Path)
    dense_train.add_argument("experiment", type=Path)
    dense_train.add_argument("--resume", type=Path)
    dense_train.add_argument("--steps", type=int, default=30)
    dense_train.add_argument("--learning-rate", type=float, default=0.5)
    dense_train.add_argument("--hidden-size", type=int, default=16)
    dense_train.add_argument("--gradient-clip", type=float)
    dense_train.add_argument("--memory-budget-bytes", type=int)
    projected_train = subparsers.add_parser("tiny-attention-train")
    projected_train.add_argument("bundle", type=Path)
    projected_train.add_argument("checkpoint", type=Path)
    projected_train.add_argument("artifact", type=Path)
    projected_train.add_argument("experiment", type=Path)
    projected_train.add_argument("--steps", type=int, default=10)
    projected_train.add_argument("--learning-rate", type=float, default=0.2)
    projected_train.add_argument("--batch-size", type=int, default=1)
    projected_train.add_argument("--resume", type=Path)
    projected_train.add_argument("--max-seconds", type=float)
    projected_train.add_argument("--early-stop-patience", type=int)
    regression = subparsers.add_parser("tiny-regression")
    regression.add_argument("reports", nargs="+", type=Path)
    regression.add_argument("--max-loss-increase", type=float, default=0.0)
    regression.add_argument("--max-accuracy-drop", type=float, default=0.0)
    series = subparsers.add_parser("tiny-regression-series")
    series.add_argument("output", type=Path)
    series.add_argument("reports", nargs="+", type=Path)
    series.add_argument("--max-loss-increase", type=float, default=0.0)
    series.add_argument("--max-accuracy-drop", type=float, default=0.0)
    args = parser.parse_args()
    if args.command == "tiny-infer":
        import json
        session = InferenceSession(__import__("spaceslug.backend", fromlist=["BackendSession"]).BackendSession("/mnt/Data/Projects/Cpntodd_Cactus/vulkan-runtime", "runtime"), ProjectedTinyAttentionModel(259))
        print(json.dumps(session.run(args.tokens), sort_keys=True))
        return 0
    if args.command == "tiny-gpu-chain":
        import json
        session = InferenceSession(__import__("spaceslug.backend", fromlist=["BackendSession"]).BackendSession("/mnt/Data/Projects/Cpntodd_Cactus/vulkan-runtime", "runtime"), ProjectedTinyAttentionModel(259))
        print(json.dumps(session.run_gpu_chain_plan(args.tokens), sort_keys=True))
        return 0
    if args.command == "tiny-cpu-verify":
        import json
        result = verify_cpu_training(args.bundle, steps=args.steps, learning_rate=args.learning_rate)
        print(json.dumps(result.__dict__, sort_keys=True))
        return 0 if result.passed else 1
    if args.command == "tui":
        SpaceslugTui().run()
        return 0
    if args.command == "dataset-verify":
        bundle = verify_bundle(args.bundle)
        print(f"dataset={bundle.manifest['dataset_id']} revision={bundle.manifest['revision']}")
        print(f"records={bundle.manifest['record_count']} splits={bundle.stats()}")
        return 0
    if args.command == "tiny-regression-series":
        reports = [__import__("json").loads(path.read_text(encoding="utf-8")) for path in args.reports]
        record = write_regression_series(args.output, args.output.name, reports, max_loss_increase=args.max_loss_increase, max_accuracy_drop=args.max_accuracy_drop)
        print(__import__("json").dumps(load_regression_series(record)["comparison"], sort_keys=True))
        return 0 if load_regression_series(record)["comparison"]["pass"] else 1
    if args.command == "tiny-regression":
        reports = [__import__("json").loads(path.read_text(encoding="utf-8")) for path in args.reports]
        comparison = compare_reports(reports, max_loss_increase=args.max_loss_increase, max_accuracy_drop=args.max_accuracy_drop)
        print(__import__("json").dumps(comparison, sort_keys=True))
        return 0 if comparison["pass"] else 1
    if args.command == "tiny-attention-train":
        bundle = verify_bundle(args.bundle)
        result = run_projected_training(
            bundle, ProjectedAttentionConfig(args.steps, args.learning_rate, args.batch_size, max_seconds=args.max_seconds, early_stop_patience=args.early_stop_patience),
            tokenizer=default_tokenizer(), checkpoint=args.checkpoint, artifact=args.artifact,
            experiment=args.experiment, resume=args.resume,
        )
        print(f"initial_loss={result['metrics']['initial_train_loss']:.9f} final_loss={result['metrics']['final_train_loss']:.9f}")
        print(f"artifact_revision={result['artifact_revision']} experiment={result['experiment']}")
        return 0
    if args.command == "tiny-dense-train":
        bundle = verify_bundle(args.bundle)
        tokenizer = default_tokenizer()
        model = None
        prior_steps = 0
        if args.resume:
            model, previous_metrics = load_dense_checkpoint(args.resume)
            prior_steps = int(previous_metrics["optimizer_step"])
        model, metrics = train_dense_tiny(
            bundle, DenseTinyTrainingConfig(args.steps, args.learning_rate, args.hidden_size, args.gradient_clip, args.memory_budget_bytes), tokenizer=tokenizer,
            model=model, prior_steps=prior_steps
        )
        save_dense_checkpoint(args.checkpoint, model, metrics)
        artifact = write_dense_tiny_artifact(args.artifact, model, tokenizer)
        record = write_experiment_record(args.experiment, args.experiment.name, metrics)
        print(f"initial_loss={metrics['initial_train_loss']:.9f} final_loss={metrics['final_train_loss']:.9f}")
        print(f"checkpoint={args.checkpoint} artifact={args.artifact} revision={artifact['revision']} experiment={record}")
        return 0
    if args.command == "tiny-train-dataset":
        bundle = verify_bundle(args.bundle)
        tokenizer = default_tokenizer()
        model, optimizer, metrics = train_tiny(
            bundle, TinyTrainingConfig(args.steps, args.learning_rate), tokenizer=tokenizer
        )
        save_training_checkpoint(args.checkpoint, model, optimizer, metrics)
        manifest = write_tiny_artifact(args.artifact, model, tokenizer)
        print(f"initial_loss={metrics['initial_train_loss']:.9f} final_loss={metrics['final_train_loss']:.9f}")
        print(f"checkpoint={args.checkpoint} artifact={args.artifact} revision={manifest['revision']}")
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
