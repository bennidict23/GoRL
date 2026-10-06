from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .artifacts import ArtifactError, RunArtifacts
from .config import (
    METHODS,
    PROFILES,
    TASKS,
    ConfigError,
    default_config_root,
    load_run_config,
)
from .runtime import apply_process_runtime, runtime_values_for_main_process


PipelineRunner = Callable[[Any, RunArtifacts], Mapping[str, Any] | None]
PairRunner = Callable[..., int]


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gorl")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="train one task, method, and seed")
    train.add_argument("--task", choices=TASKS, required=True)
    train.add_argument("--method", choices=METHODS, required=True)
    train.add_argument("--seed", type=_non_negative_int, required=True)
    train.add_argument("--teacher-seed", type=_non_negative_int)
    train.add_argument("--decoder-seed", type=_non_negative_int)
    train.add_argument("--encoder-seed", type=_non_negative_int)
    train.add_argument("--eval-seed", type=_non_negative_int)
    train.add_argument("--profile", choices=PROFILES, default="auto")
    train.add_argument("--config", type=Path, help="TOML hyperparameter overlay")
    train.add_argument("--config-root", type=Path)
    train.add_argument("--output-root", type=Path, default=Path("runs"))
    train.add_argument("--run-id")
    train.add_argument(
        "--wandb-mode",
        choices=("disabled", "offline", "online"),
        default="disabled",
    )
    train.add_argument("--wandb-project", default="gorl-benchmark")
    train.add_argument("--wandb-entity")
    train.add_argument(
        "--smoke",
        action="store_true",
        help="run the real pipeline with its minimal smoke budget",
    )
    train.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve config and artifacts without starting training",
    )

    pair = subparsers.add_parser(
        "train-pair",
        help="train Humanoid FM and diffusion from one fresh teacher",
    )
    pair.add_argument(
        "--task",
        choices=("HumanoidStand", "HumanoidRun"),
        required=True,
    )
    pair.add_argument("--teacher-seed", type=_non_negative_int, required=True)
    pair.add_argument("--fm-seed", type=_non_negative_int, required=True)
    pair.add_argument("--diffusion-seed", type=_non_negative_int, required=True)
    pair.add_argument(
        "--gpus",
        type=_non_negative_int,
        nargs="+",
        required=True,
        metavar="GPU",
        help=(
            "one or two logical GPU ordinals within the parent process's "
            "CUDA-visible device list"
        ),
    )
    pair.add_argument("--profile", choices=("auto", "humanoid"), default="auto")
    pair.add_argument("--config", type=Path, help="TOML hyperparameter overlay")
    pair.add_argument("--config-root", type=Path)
    pair.add_argument("--output-root", type=Path, default=Path("runs"))
    pair.add_argument("--run-id")
    pair.add_argument(
        "--wandb-mode",
        choices=("disabled", "offline", "online"),
        default="disabled",
    )
    pair.add_argument("--wandb-project", default="gorl-benchmark")
    pair.add_argument("--wandb-entity")
    pair.add_argument(
        "--smoke",
        action="store_true",
        help="run both real branches with their minimal smoke budgets",
    )
    pair.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve and record both branches without starting subprocesses",
    )

    return parser


def _load_pipeline_runner() -> PipelineRunner:
    try:
        from .pipeline import run
    except ModuleNotFoundError as error:
        if error.name != "gorl.pipeline":
            raise
        raise RuntimeError(
            "the training pipeline is not installed; configuration and "
            "dry-run commands remain available"
        ) from error
    return run


def _load_pair_runner() -> PairRunner:
    from .humanoid.pair import run_from_namespace

    return run_from_namespace


def _apply_runtime_environment(config: Any) -> None:
    precision, triton_gemm, autotune_level = runtime_values_for_main_process(config)
    apply_process_runtime(
        os.environ,
        matmul_precision=precision,
        triton_gemm=triton_gemm,
        autotune_level=autotune_level,
    )


def _train(
    args: argparse.Namespace,
    *,
    argv: Sequence[str],
    pipeline_runner: PipelineRunner | None = None,
    command_argv: Sequence[str] | None = None,
) -> int:
    if args.smoke and args.dry_run:
        raise ConfigError("--smoke and --dry-run are mutually exclusive")
    config = load_run_config(
        task=args.task,
        method=args.method,
        seed=args.seed,
        teacher_seed=args.teacher_seed,
        decoder_seed=args.decoder_seed,
        encoder_seed=args.encoder_seed,
        eval_seed=args.eval_seed,
        profile=args.profile,
        config_root=args.config_root or default_config_root(),
        overlay=args.config,
        output_root=args.output_root,
        smoke=args.smoke,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )
    command = (
        list(command_argv)
        if command_argv is not None
        else [sys.executable, "-m", "gorl", *argv]
    )
    _apply_runtime_environment(config)
    artifacts = RunArtifacts.create(
        config,
        argv=command,
        cwd=Path.cwd(),
        run_id=args.run_id,
    )
    print(f"run directory: {artifacts.run_dir}")
    if args.dry_run:
        artifacts.mark_dry_run()
        print(f"resolved config: {artifacts.resolved_config_path}")
        return 0

    runner = pipeline_runner or _load_pipeline_runner()
    artifacts.mark_running()
    try:
        result = runner(config, artifacts)
        if result is not None:
            artifacts.complete(result)
        elif artifacts.read_manifest().get("status") != "complete":
            raise ArtifactError(
                "pipeline returned no result and did not complete the manifest"
            )
        artifacts.finish_tracking(exit_code=0)
    except BaseException as error:
        try:
            artifacts.mark_failed(error)
        finally:
            artifacts.finish_tracking(exit_code=1)
        raise
    return 0


def _train_pair(
    args: argparse.Namespace,
    *,
    argv: Sequence[str],
    pair_runner: PairRunner | None = None,
) -> int:
    if args.smoke and args.dry_run:
        raise ConfigError("--smoke and --dry-run are mutually exclusive")
    runner = pair_runner or _load_pair_runner()
    return runner(
        args,
        command=[sys.executable, "-m", "gorl", *argv],
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(arguments)
    try:
        if args.command == "train":
            return _train(args, argv=arguments)
        if args.command == "train-pair":
            return _train_pair(args, argv=arguments)
    except (ArtifactError, ConfigError, RuntimeError) as error:
        parser.error(str(error))
    raise AssertionError(f"unhandled command: {args.command}")
