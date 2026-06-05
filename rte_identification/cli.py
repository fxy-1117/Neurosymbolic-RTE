"""Command line interface for the identification experiments."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path


PAPER_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 1.0]
TASKS = [
    ("main", "Run e-SNLI, SICK, and MNLI main results"),
    ("esnli", "Run e-SNLI main result"),
    ("sick", "Run SICK main result"),
    ("mnli", "Run MNLI main result"),
    ("param", "Run e-SNLI parameter analysis"),
    ("noforget", "Run e-SNLI without forgetting"),
    ("noexpl", "Run e-SNLI without explanation"),
]


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


def cmd_main(args) -> None:
    from .experiments import run_experiment

    for dataset in args.datasets:
        print(f"\n[task] dataset={dataset}", flush=True)
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


def run_named_task(name: str, args) -> None:
    from .experiments import run_experiment, run_parameter_analysis

    if name == "main":
        for dataset in ["esnli", "sick", "mnli"]:
            print(f"\n[task] {name}: {dataset}", flush=True)
            run_experiment(
                build_config(args, dataset),
                data_dir=args.data_dir,
                cache_dir=args.cache_dir,
                output_dir=args.output_dir,
                reuse_logic_cache=not args.no_logic_cache,
            )
        return

    if name in {"esnli", "sick", "mnli"}:
        print(f"\n[task] {name}", flush=True)
        run_experiment(
            build_config(args, name),
            data_dir=args.data_dir,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            reuse_logic_cache=not args.no_logic_cache,
        )
        return

    if name == "param":
        print("\n[task] param", flush=True)
        config = replace(build_config(args, "esnli"), max_len=20)
        run_parameter_analysis(
            config,
            thresholds=PAPER_THRESHOLDS,
            data_dir=args.data_dir,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            reuse_logic_cache=not args.no_logic_cache,
        )
        return

    if name == "noforget":
        print("\n[task] noforget", flush=True)
        config = replace(build_config(args, "esnli"), use_forgetting=False)
        run_experiment(
            config,
            data_dir=args.data_dir,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            reuse_logic_cache=not args.no_logic_cache,
        )
        return

    if name == "noexpl":
        print("\n[task] noexpl", flush=True)
        config = replace(build_config(args, "esnli"), include_explanation=False)
        run_experiment(
            config,
            data_dir=args.data_dir,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            reuse_logic_cache=not args.no_logic_cache,
        )
        return

    raise ValueError(f"Unknown task: {name}")


def selected_tasks(selection: str) -> list[str]:
    task_names = [name for name, _description in TASKS]
    selection = selection.strip().lower()
    if selection in {"q", "quit", "exit"}:
        return []
    if selection in {"all", "*"}:
        return task_names

    selected = []
    for item in selection.replace(",", " ").split():
        if item.isdigit():
            index = int(item) - 1
            if index < 0 or index >= len(task_names):
                raise ValueError(f"Task number out of range: {item}")
            selected.append(task_names[index])
        elif item in task_names:
            selected.append(item)
        else:
            raise ValueError(f"Unknown task: {item}")
    return list(dict.fromkeys(selected))


def print_task_list() -> None:
    print("Task list:")
    for index, (name, description) in enumerate(TASKS, start=1):
        print(f"  {index}. {name:<8} {description}")


def cmd_tasks(args) -> None:
    print_task_list()
    selection = args.select
    if not selection:
        selection = input("\nChoose tasks by number or name, separated by commas: ")

    tasks = selected_tasks(selection)
    if not tasks:
        print("No task selected.")
        return

    for task in tasks:
        run_named_task(task, args)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="rte-identification",
        description="Run the RTE identification experiments.",
    )
    subparsers = parser.add_subparsers()

    tasks_parser = subparsers.add_parser("tasks", help="Choose experiments from a task list")
    add_common_args(tasks_parser)
    tasks_parser.add_argument(
        "--select",
        help="tasks to run, e.g. 'main', '1,5', or 'esnli,param'",
    )
    tasks_parser.set_defaults(func=cmd_tasks)

    run_parser = subparsers.add_parser("run", help="Run one dataset experiment")
    add_common_args(run_parser)
    run_parser.add_argument("--dataset", choices=["esnli", "sick", "mnli"], required=True)
    run_parser.set_defaults(func=cmd_run)

    main_parser = subparsers.add_parser("main", help="Run the three main datasets")
    add_common_args(main_parser)
    main_parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["esnli", "sick", "mnli"],
        default=["esnli", "sick", "mnli"],
    )
    main_parser.set_defaults(func=cmd_main)

    analysis_parser = subparsers.add_parser(
        "parameter-analysis",
        help="Run the threshold sweep",
    )
    add_common_args(analysis_parser, max_len_default=20)
    analysis_parser.add_argument("--dataset", choices=["esnli", "sick", "mnli"], default="esnli")
    analysis_parser.add_argument("--thresholds", nargs="+", type=float, default=PAPER_THRESHOLDS)
    analysis_parser.set_defaults(func=cmd_parameter_analysis)

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        tasks_parser.parse_args([], namespace=args)
        args.func = cmd_tasks
    return args


def main() -> None:
    args = parse_args(sys.argv[1:])
    args.func(args)


if __name__ == "__main__":
    main()
