"""Command line interface for the notebook-to-Python runner."""

from __future__ import annotations

import argparse
from pathlib import Path


PAPER_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 1.0]


def add_common_args(parser: argparse.ArgumentParser, max_len_default: int = 0) -> None:
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument("--cache-dir", default=".cache", type=Path)
    parser.add_argument("--output-dir", default="results", type=Path)
    parser.add_argument("--seed", default=66, type=int)
    parser.add_argument("--per-class", default=500, type=int)
    parser.add_argument("--max-len", default=max_len_default, type=int)
    parser.add_argument("--threshold", default=0.55, type=float)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--roberta-batch-size", default=256, type=int)
    parser.add_argument(
        "--no-logic-cache",
        action="store_true",
        help="ignore cached sentence-level AMR logic and regenerate logic for this run",
    )
    parser.add_argument(
        "--no-forgetting",
        action="store_true",
        help="use the original contradiction formula without forgetting unused premise variables",
    )
    parser.add_argument(
        "--no-explanation",
        action="store_true",
        help="for e-SNLI, run with premise and claim only",
    )


def build_config(args, dataset: str) -> ExperimentConfig:
    from .experiments import ExperimentConfig

    return ExperimentConfig(
        dataset=dataset,
        per_class=args.per_class,
        seed=args.seed,
        max_len=args.max_len,
        threshold=args.threshold,
        batch_size=args.batch_size,
        roberta_batch_size=args.roberta_batch_size,
        use_forgetting=not args.no_forgetting,
        include_explanation=not args.no_explanation,
    )


def cmd_run(args) -> None:
    from .experiments import run_experiment

    run_experiment(
        build_config(args, args.dataset),
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        reuse_logic_cache=not args.no_logic_cache,
    )


def cmd_paper(args) -> None:
    from .experiments import run_experiment

    for dataset in args.datasets:
        print(f"\n[paper] dataset={dataset}", flush=True)
        run_experiment(
            build_config(args, dataset),
            data_dir=args.data_dir,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            reuse_logic_cache=not args.no_logic_cache,
        )


def cmd_parameter_analysis(args) -> None:
    from .experiments import run_parameter_analysis

    config = build_config(args, args.dataset)
    run_parameter_analysis(
        config,
        thresholds=args.thresholds,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        reuse_logic_cache=not args.no_logic_cache,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="identification-repro",
        description="Run the AAAI 2025 identification notebook experiments.",
    )
    subparsers = parser.add_subparsers(required=True)

    run_parser = subparsers.add_parser("run", help="Run one dataset experiment")
    add_common_args(run_parser)
    run_parser.add_argument("--dataset", choices=["esnli", "sick", "mnli"], required=True)
    run_parser.set_defaults(func=cmd_run)

    paper_parser = subparsers.add_parser("paper", help="Run the three paper datasets")
    add_common_args(paper_parser)
    paper_parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["esnli", "sick", "mnli"],
        default=["esnli", "sick", "mnli"],
    )
    paper_parser.set_defaults(func=cmd_paper)

    analysis_parser = subparsers.add_parser(
        "parameter-analysis",
        help="Run the paper threshold sweep",
    )
    add_common_args(analysis_parser, max_len_default=20)
    analysis_parser.add_argument("--dataset", choices=["esnli", "sick", "mnli"], default="esnli")
    analysis_parser.add_argument("--thresholds", nargs="+", type=float, default=PAPER_THRESHOLDS)
    analysis_parser.set_defaults(func=cmd_parameter_analysis)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
