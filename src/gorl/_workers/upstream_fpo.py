"""Private subprocess worker for reproducible PPO, FPO, and DPPO baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


UPSTREAM_FPO_COMMIT = "418c2554f7cd22d52e14c07d951280929d73bf2f"
DPPO_FIX_COMMIT = "964dd78c6fb64de8c52eeb7ce80561c43455128f"
DPPO_FIX_PARENT = UPSTREAM_FPO_COMMIT
DPPO_FIX_AUTHOR_NAME = "GoRL Authors"
DPPO_FIX_AUTHOR_EMAIL = "gorl-authors@users.noreply.github.com"

_SOURCE_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = (
    _SOURCE_ROOT
    if (_SOURCE_ROOT / "pyproject.toml").is_file()
    else Path(sys.prefix) / "share" / "gorl"
)
DPPO_PATCH_PATH = (
    PROJECT_ROOT
    / ("scripts" if (PROJECT_ROOT / "pyproject.toml").is_file() else "tools")
    / "patches"
    / "dppo_scalar_sigma.patch"
)

PAPER_LR = 3e-4
PAPER_NUM_ENVS = 2048
PAPER_NUM_EVAL_ENVS = 128
PAPER_EPISODE_LENGTH = 1000
PAPER_BATCH_SIZE = 1024
PAPER_NUM_MINIBATCHES = 32
PAPER_UNROLL_LENGTH = 30
PAPER_NUM_UPDATES_PER_BATCH = 16
PAPER_NUM_EVALS = 10
PAPER_PPO_CLIP = 0.1
PAPER_PPO_ENTROPY_COST = 0.01
PAPER_FPO_CLIP = 0.05
PAPER_DPPO_CLIP = 0.2
PAPER_DPPO_SDE_SIGMA = 0.05
PAPER_FLOW_STEPS = 10
PAPER_FPO_OUTPUT_MODE = "u_but_supervise_as_eps"
PAPER_FPO_N_SAMPLES_PER_ACTION = 8
PAPER_FPO_TIMESTEP_EMBED_DIM = 8
PAPER_FPO_POLICY_MLP_OUTPUT_SCALE = 0.25

TRAINER_BACKENDS = {
    "ppo": "upstream_fpo_ppo",
    "fpo": "upstream_fpo",
    "dppo": "upstream_fpo_denoising_mdp",
}
ACTION_SEMANTICS = "legacy_pre_tanh"


class BaselineError(RuntimeError):
    pass


class NumericalMetricError(BaselineError):
    def __init__(
        self,
        *,
        source: str,
        metric: str,
        value: float,
        context: Mapping[str, Any],
    ) -> None:
        message = (
            f"non-finite {source} metric {metric}={value!r} "
            f"at env_steps={context.get('env_steps')}"
        )
        self.failure = {
            "type": "non_finite_metric",
            "source": source,
            "metric": metric,
            "value": repr(value),
            "message": message,
            **dict(context),
        }
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class Runtime:
    jax: Any
    np: Any
    registry: Any
    dm_control_suite_params: Any
    fpo: Any
    ppo: Any
    rollouts: Any


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one upstream MuJoCo Playground baseline and write results "
            "directly to an explicit output directory."
        )
    )
    parser.add_argument(
        "--method",
        choices=("ppo", "fpo", "dppo"),
        required=True,
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-timesteps", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--fpo-root",
        type=Path,
        default=None,
        help=(
            "Explicit FPO checkout. When omitted, PPO/FPO use GORL_FPO_ROOT, "
            "DPPO uses GORL_DPPO_ROOT, and otherwise the pinned "
            "scripts/fetch_dependencies.py locations are used."
        ),
    )
    parser.add_argument("--num-envs", type=int, default=PAPER_NUM_ENVS)
    parser.add_argument("--num-eval-envs", type=int, default=PAPER_NUM_EVAL_ENVS)
    parser.add_argument("--episode-length", type=int, default=PAPER_EPISODE_LENGTH)
    parser.add_argument("--batch-size", type=int, default=PAPER_BATCH_SIZE)
    parser.add_argument("--num-minibatches", type=int, default=PAPER_NUM_MINIBATCHES)
    parser.add_argument("--unroll-length", type=int, default=PAPER_UNROLL_LENGTH)
    parser.add_argument(
        "--num-updates-per-batch",
        type=int,
        default=PAPER_NUM_UPDATES_PER_BATCH,
    )
    parser.add_argument("--num-evals", type=int, default=PAPER_NUM_EVALS)
    parser.add_argument("--learning-rate", type=float, default=PAPER_LR)
    parser.add_argument("--clip-epsilon", type=float, default=None)
    parser.add_argument("--entropy-cost", type=float, default=None)
    parser.add_argument("--discounting", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--reward-scaling", type=float, default=10.0)
    parser.add_argument(
        "--normalize-observations",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--normalize-advantage",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--value-loss-coeff", type=float, default=0.25)
    parser.add_argument("--flow-steps", type=int, default=None)
    parser.add_argument(
        "--output-mode",
        choices=("u", "u_but_supervise_as_eps"),
        default=None,
    )
    parser.add_argument("--samples-per-action", type=int, default=None)
    parser.add_argument("--timestep-embedding-dim", type=int, default=None)
    parser.add_argument("--policy-output-scale", type=float, default=None)
    parser.add_argument("--sde-sigma", type=float, default=None)
    return parser.parse_args(argv)


def expected_dependency_commit(method: str) -> str:
    return DPPO_FIX_COMMIT if method == "dppo" else UPSTREAM_FPO_COMMIT


def default_dependency_path(method: str) -> Path:
    directory = "fpo_dppo_fix" if method == "dppo" else "fpo"
    root = PROJECT_ROOT if (PROJECT_ROOT / "pyproject.toml").is_file() else Path.cwd()
    return root / "external" / directory


def dependency_environment_variable(method: str) -> str:
    return "GORL_DPPO_ROOT" if method == "dppo" else "GORL_FPO_ROOT"


def resolve_dependency_path(
    method: str,
    explicit: Path | None,
    *,
    environ: dict[str, str] | None = None,
) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    environment = os.environ if environ is None else environ
    configured = environment.get(dependency_environment_variable(method))
    if configured:
        return Path(configured).expanduser().resolve()
    return default_dependency_path(method).resolve()


def git_output(path: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), *arguments],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise BaselineError(f"invalid FPO checkout {path}: {error}") from error


def validate_dependency(path: Path, method: str) -> dict[str, Any]:
    source = path / "playground" / "src" / "flow_policy"
    if not source.is_dir():
        raise BaselineError(f"missing FPO playground sources: {source}")
    root = Path(git_output(path, "rev-parse", "--show-toplevel")).resolve()
    if root != path.resolve():
        raise BaselineError(
            f"FPO dependency must be an independent checkout: {path} belongs to {root}"
        )
    commit = git_output(path, "rev-parse", "HEAD")
    expected = expected_dependency_commit(method)
    if commit != expected:
        raise BaselineError(
            f"{method} requires FPO commit {expected}, got {commit} at {path}"
        )
    status = git_output(path, "status", "--porcelain")
    if status:
        raise BaselineError(f"refusing to use dirty FPO checkout: {path}")
    value: dict[str, Any] = {
        "path": str(path),
        "commit": commit,
        "clean": True,
        "source": "akanazawa/fpo",
    }
    if method == "dppo":
        parent = git_output(path, "rev-parse", "HEAD^")
        if parent != DPPO_FIX_PARENT:
            raise BaselineError(
                "DPPO fix provenance mismatch: "
                f"expected parent {DPPO_FIX_PARENT}, got {parent}"
            )
        value["patch"] = {
            "commit": DPPO_FIX_COMMIT,
            "parent": parent,
            "author": {
                "name": DPPO_FIX_AUTHOR_NAME,
                "email": DPPO_FIX_AUTHOR_EMAIL,
            },
            "file": str(DPPO_PATCH_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(DPPO_PATCH_PATH),
            "description": (
                "Broadcast scalar sde_sigma directly in denoising-MDP "
                "likelihood computation."
            ),
        }
    return value


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise BaselineError(f"missing provenance file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_runtime(fpo_root: Path) -> Runtime:
    os.environ["WANDB_MODE"] = "disabled"
    playground_src = (fpo_root / "playground" / "src").resolve()
    sys.path.insert(0, str(playground_src))

    import jax
    import numpy as np
    from mujoco_playground import registry
    from mujoco_playground.config import dm_control_suite_params

    from flow_policy import fpo, ppo, rollouts

    loaded_from = Path(fpo.__file__).resolve()
    if not loaded_from.is_relative_to(playground_src):
        raise BaselineError(
            f"flow_policy resolved outside requested checkout: {loaded_from}"
        )
    return Runtime(
        jax=jax,
        np=np,
        registry=registry,
        dm_control_suite_params=dm_control_suite_params,
        fpo=fpo,
        ppo=ppo,
        rollouts=rollouts,
    )


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "num_timesteps": args.num_timesteps,
        "num_envs": args.num_envs,
        "num_eval_envs": args.num_eval_envs,
        "episode_length": args.episode_length,
        "batch_size": args.batch_size,
        "num_minibatches": args.num_minibatches,
        "unroll_length": args.unroll_length,
        "num_updates_per_batch": args.num_updates_per_batch,
        "num_evals": args.num_evals,
    }
    invalid = {name: value for name, value in positive.items() if value <= 0}
    if invalid:
        raise BaselineError(f"positive integer arguments required: {invalid}")
    if args.seed < 0:
        raise BaselineError(f"seed must be non-negative, got {args.seed}")
    if args.num_evals < 2:
        raise BaselineError("num_evals must be at least 2 (pre-update and final)")
    if args.learning_rate <= 0:
        raise BaselineError("learning_rate must be positive")
    if not 0 < args.discounting <= 1:
        raise BaselineError("discounting must be in (0, 1]")
    if not 0 <= args.gae_lambda <= 1:
        raise BaselineError("gae_lambda must be in [0, 1]")
    if args.reward_scaling <= 0:
        raise BaselineError("reward_scaling must be positive")
    if args.value_loss_coeff < 0:
        raise BaselineError("value_loss_coeff must be non-negative")
    if resolved_clip_epsilon(args) <= 0:
        raise BaselineError("clip_epsilon must be positive")
    if resolved_entropy_cost(args) < 0:
        raise BaselineError("entropy_cost must be non-negative")
    if args.method != "ppo" and resolved_entropy_cost(args) != 0:
        raise BaselineError("FPO and DPPO do not define an entropy loss")
    if args.method == "ppo" and any(
        value is not None
        for value in (
            args.flow_steps,
            args.output_mode,
            args.samples_per_action,
            args.timestep_embedding_dim,
            args.policy_output_scale,
            args.sde_sigma,
        )
    ):
        raise BaselineError("decoder options do not apply to PPO")
    if args.method == "dppo" and resolved_sde_sigma(args) <= 0:
        raise BaselineError("DPPO requires sde_sigma > 0")
    if args.method != "ppo":
        if resolved_flow_steps(args) <= 0:
            raise BaselineError("flow_steps must be positive")
        if resolved_samples_per_action(args) <= 0:
            raise BaselineError("samples_per_action must be positive")
        embedding_dim = resolved_timestep_embedding_dim(args)
        if embedding_dim <= 0 or embedding_dim % 2:
            raise BaselineError(
                "timestep_embedding_dim must be a positive even integer"
            )
        if resolved_policy_output_scale(args) <= 0:
            raise BaselineError("policy_output_scale must be positive")
        if resolved_sde_sigma(args) < 0:
            raise BaselineError("sde_sigma must be non-negative")
    if args.method == "ppo" and args.task in {"HumanoidStand", "HumanoidRun"}:
        raise BaselineError(
            "Humanoid PPO uses the Brax official profile, not this upstream worker"
        )


def resolved_clip_epsilon(args: argparse.Namespace) -> float:
    if args.clip_epsilon is not None:
        return args.clip_epsilon
    if args.method == "ppo":
        return PAPER_PPO_CLIP
    if args.method == "dppo":
        return PAPER_DPPO_CLIP
    return PAPER_FPO_CLIP


def resolved_entropy_cost(args: argparse.Namespace) -> float:
    if args.entropy_cost is not None:
        return args.entropy_cost
    return PAPER_PPO_ENTROPY_COST if args.method == "ppo" else 0.0


def resolved_flow_steps(args: argparse.Namespace) -> int:
    return PAPER_FLOW_STEPS if args.flow_steps is None else args.flow_steps


def resolved_sde_sigma(args: argparse.Namespace) -> float:
    if args.sde_sigma is not None:
        return args.sde_sigma
    return PAPER_DPPO_SDE_SIGMA if args.method == "dppo" else 0.0


def resolved_samples_per_action(args: argparse.Namespace) -> int:
    if args.samples_per_action is not None:
        return args.samples_per_action
    return PAPER_FPO_N_SAMPLES_PER_ACTION


def resolved_timestep_embedding_dim(args: argparse.Namespace) -> int:
    if args.timestep_embedding_dim is not None:
        return args.timestep_embedding_dim
    return PAPER_FPO_TIMESTEP_EMBED_DIM


def resolved_policy_output_scale(args: argparse.Namespace) -> float:
    if args.policy_output_scale is not None:
        return args.policy_output_scale
    return PAPER_FPO_POLICY_MLP_OUTPUT_SCALE


def build_config(args: argparse.Namespace, runtime: Runtime) -> Any:
    if args.method == "ppo":
        params = runtime.dm_control_suite_params.brax_ppo_config(args.task)
        params.num_timesteps = args.num_timesteps
        params.num_envs = args.num_envs
        params.num_evals = args.num_evals
        params.episode_length = args.episode_length
        params.batch_size = args.batch_size
        params.num_minibatches = args.num_minibatches
        params.unroll_length = args.unroll_length
        params.num_updates_per_batch = args.num_updates_per_batch
        params.learning_rate = args.learning_rate
        params.entropy_cost = resolved_entropy_cost(args)
        params.clipping_epsilon = resolved_clip_epsilon(args)
        params.discounting = args.discounting
        params.gae_lambda = args.gae_lambda
        params.reward_scaling = args.reward_scaling
        params.normalize_observations = args.normalize_observations
        params.normalize_advantage = args.normalize_advantage
        params.value_loss_coeff = args.value_loss_coeff
        return runtime.ppo.PpoConfig(**params)

    config = {
        "loss_mode": "denoising_mdp" if args.method == "dppo" else "fpo",
        "flow_steps": resolved_flow_steps(args),
        "sde_sigma": resolved_sde_sigma(args),
        "num_timesteps": args.num_timesteps,
        "num_envs": args.num_envs,
        "num_evals": args.num_evals,
        "episode_length": args.episode_length,
        "batch_size": args.batch_size,
        "num_minibatches": args.num_minibatches,
        "unroll_length": args.unroll_length,
        "num_updates_per_batch": args.num_updates_per_batch,
        "learning_rate": args.learning_rate,
        "clipping_epsilon": resolved_clip_epsilon(args),
        "discounting": args.discounting,
        "gae_lambda": args.gae_lambda,
        "reward_scaling": args.reward_scaling,
        "normalize_observations": args.normalize_observations,
        "normalize_advantage": args.normalize_advantage,
        "value_loss_coeff": args.value_loss_coeff,
        "output_mode": args.output_mode or PAPER_FPO_OUTPUT_MODE,
        "n_samples_per_action": resolved_samples_per_action(args),
        "timestep_embed_dim": resolved_timestep_embedding_dim(args),
        "policy_mlp_output_scale": resolved_policy_output_scale(args),
    }
    return runtime.fpo.FpoConfig(**config)


def initialize_agent(
    args: argparse.Namespace,
    runtime: Runtime,
    env: Any,
    config: Any,
) -> Any:
    key = runtime.jax.random.key(args.seed)
    if args.method == "ppo":
        return runtime.ppo.PpoState.init(prng=key, env=env, config=config)
    return runtime.fpo.FpoState.init(prng=key, env=env, config=config)


def evaluate(
    agent_state: Any,
    *,
    args: argparse.Namespace,
    runtime: Runtime,
    config: Any,
    eval_index: int,
    outer_iter: int,
    env_steps: int,
    is_final: bool,
) -> dict[str, Any]:
    outputs = runtime.rollouts.eval_policy(
        agent_state,
        prng=runtime.jax.random.fold_in(
            runtime.jax.random.key(args.seed),
            outer_iter,
        ),
        num_envs=args.num_eval_envs,
        max_episode_length=config.episode_length,
    )
    metrics = {
        key: runtime.np.asarray(value).item()
        for key, value in outputs.scalar_metrics.items()
    }
    record = {
        "eval_index": eval_index,
        "outer_iter": outer_iter,
        "env_steps": env_steps,
        "phase": "post_update" if outer_iter else "pre_update",
        "is_final": is_final,
        **metrics,
    }
    require_finite_metrics("evaluation", record)
    return record


def intermediate_eval_iterations(
    outer_iters: int,
    num_evals: int,
) -> frozenset[int]:
    count = max(num_evals - 2, 0)
    if count == 0 or outer_iters <= 1:
        return frozenset()
    points = {
        max(1, min(outer_iters - 1, round(index * outer_iters / (count + 1))))
        for index in range(1, count + 1)
    }
    return frozenset(points)


def prepare_output_dir(path: Path) -> Path:
    output = path.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    owned = (
        "eval_metrics.jsonl",
        "train_metrics.jsonl",
        "summary.json",
        "worker_manifest.json",
    )
    collisions = [output / name for name in owned if (output / name).exists()]
    if collisions:
        rendered = ", ".join(str(path) for path in collisions)
        raise BaselineError(f"refusing to overwrite baseline outputs: {rendered}")
    return output


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def require_finite_metrics(source: str, record: Mapping[str, Any]) -> None:
    context = {
        key: record[key]
        for key in ("eval_index", "outer_iter", "env_steps", "phase", "is_final")
        if key in record
    }
    for key, value in record.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise NumericalMetricError(
                source=source,
                metric=str(key),
                value=value,
                context=context,
            )


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            to_jsonable(value),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                to_jsonable(value),
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )


def code_provenance() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        clean = (
            subprocess.check_output(
                ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            == ""
        )
    except (OSError, subprocess.CalledProcessError):
        commit = None
        clean = None
    return {"commit": commit, "clean": clean}


def config_manifest(config: Any, args: argparse.Namespace) -> dict[str, Any]:
    names = (
        "num_envs",
        "episode_length",
        "batch_size",
        "num_minibatches",
        "unroll_length",
        "num_updates_per_batch",
        "num_evals",
        "learning_rate",
        "clipping_epsilon",
        "discounting",
        "gae_lambda",
        "normalize_advantage",
        "normalize_observations",
        "reward_scaling",
        "value_loss_coeff",
        "entropy_cost",
        "flow_steps",
        "loss_mode",
        "sde_sigma",
        "output_mode",
        "timestep_embed_dim",
        "n_samples_per_action",
        "policy_mlp_output_scale",
    )
    value = {name: getattr(config, name) for name in names if hasattr(config, name)}
    value["num_eval_envs"] = args.num_eval_envs
    return to_jsonable(value)


def make_summary(
    args: argparse.Namespace,
    eval_records: Sequence[dict[str, Any]],
    *,
    resolved_env_steps: int,
    dependency: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    if not eval_records or not eval_records[-1].get("is_final"):
        raise BaselineError("last evaluation must be the explicit final evaluation")
    if eval_records[-1].get("phase") != "post_update":
        raise BaselineError("final evaluation must happen after the final update")
    returns = [float(record["reward_mean"]) for record in eval_records]
    return {
        "status": "complete",
        "method": args.method,
        "task": args.task,
        "seed": args.seed,
        "requested_env_steps": args.num_timesteps,
        "resolved_env_steps": resolved_env_steps,
        "final_return": returns[-1],
        "stage_best_return": max(returns),
        "overall_best_return": max(returns),
        "eval_returns": returns,
        "final_eval": eval_records[-1],
        "dependency": dependency,
        "output_dir": str(output_dir),
    }


def partial_results(
    eval_records: Sequence[dict[str, Any]],
    *,
    train_records: int,
    failure: Mapping[str, Any],
) -> dict[str, Any]:
    returns = [float(record["reward_mean"]) for record in eval_records]
    completed_env_steps = max(
        (
            int(failure.get("env_steps", 0)),
            *(int(record["env_steps"]) for record in eval_records),
        )
    )
    return {
        "finite_evaluation_count": len(eval_records),
        "finite_train_record_count": train_records,
        "completed_env_steps": completed_env_steps,
        "last_finite_evaluation": eval_records[-1] if eval_records else None,
        "best_finite_return": max(returns) if returns else None,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    fpo_root = resolve_dependency_path(args.method, args.fpo_root)
    dependency = validate_dependency(fpo_root, args.method)
    runtime = load_runtime(fpo_root)
    config = build_config(args, runtime)

    steps_per_outer_iter = config.iterations_per_env * config.num_envs
    outer_iters = config.num_timesteps // steps_per_outer_iter
    if outer_iters < 1:
        raise BaselineError(
            "num_timesteps is too small for one update: "
            f"{config.num_timesteps} < {steps_per_outer_iter}"
        )
    resolved_env_steps = outer_iters * steps_per_outer_iter
    output_dir = prepare_output_dir(args.output_dir)
    eval_path = output_dir / "eval_metrics.jsonl"
    train_path = output_dir / "train_metrics.jsonl"

    env_config = runtime.registry.get_default_config(args.task)
    env = runtime.registry.load(args.task, config=env_config)
    agent_state = initialize_agent(args, runtime, env, config)
    rollout_state = runtime.rollouts.BatchedRolloutState.init(
        env,
        prng=runtime.jax.random.key(args.seed),
        num_envs=config.num_envs,
    )

    try:
        resolved_env_config = env_config.to_dict()
    except AttributeError:
        resolved_env_config = repr(env_config)
    manifest = {
        "schema_version": 1,
        "status": "running",
        "method": args.method,
        "task": args.task,
        "seed": args.seed,
        "trainer_backend": TRAINER_BACKENDS[args.method],
        "action_semantics": ACTION_SEMANTICS,
        "environment_action_transform": "tanh",
        "requested_env_steps": args.num_timesteps,
        "resolved_env_steps": resolved_env_steps,
        "outer_iters": outer_iters,
        "steps_per_outer_iter": steps_per_outer_iter,
        "config": config_manifest(config, args),
        "environment_config": resolved_env_config,
        "dependency": dependency,
        "code": code_provenance(),
        "command": sys.argv,
        "wandb": {"mode": "disabled", "writer": "parent_pipeline"},
        "failure": None,
        "result": {
            "final_return": None,
            "stage_best_return": None,
            "overall_best_return": None,
        },
    }
    write_json(output_dir / "worker_manifest.json", manifest)

    eval_records: list[dict[str, Any]] = []
    train_records = 0
    last_env_steps = 0
    try:
        pre_update = evaluate(
            agent_state,
            args=args,
            runtime=runtime,
            config=config,
            eval_index=0,
            outer_iter=0,
            env_steps=0,
            is_final=False,
        )
        eval_records.append(pre_update)
        append_jsonl(eval_path, pre_update)

        intermediate = intermediate_eval_iterations(outer_iters, config.num_evals)
        eval_index = 1
        for outer_iter in range(1, outer_iters + 1):
            rollout_state, transitions = rollout_state.rollout(
                agent_state,
                episode_length=config.episode_length,
                iterations_per_env=config.iterations_per_env,
            )
            env_steps = outer_iter * steps_per_outer_iter
            last_env_steps = env_steps
            rollout_record = {
                "outer_iter": outer_iter,
                "env_steps": env_steps,
                "return_mean": float(
                    runtime.np.mean(runtime.np.asarray(transitions.reward))
                ),
            }
            require_finite_metrics("train", rollout_record)
            agent_state, metrics = agent_state.training_step(transitions)
            train_record = {
                **rollout_record,
                **{
                    str(key): float(runtime.np.mean(runtime.np.asarray(value)))
                    for key, value in metrics.items()
                },
            }
            require_finite_metrics("train", train_record)
            append_jsonl(train_path, train_record)
            train_records += 1

            if outer_iter in intermediate:
                record = evaluate(
                    agent_state,
                    args=args,
                    runtime=runtime,
                    config=config,
                    eval_index=eval_index,
                    outer_iter=outer_iter,
                    env_steps=env_steps,
                    is_final=False,
                )
                eval_index += 1
                eval_records.append(record)
                append_jsonl(eval_path, record)

        final_eval = evaluate(
            agent_state,
            args=args,
            runtime=runtime,
            config=config,
            eval_index=eval_index,
            outer_iter=outer_iters,
            env_steps=resolved_env_steps,
            is_final=True,
        )
        eval_records.append(final_eval)
        append_jsonl(eval_path, final_eval)
    except Exception as error:
        failure = (
            error.failure
            if isinstance(error, NumericalMetricError)
            else {
                "type": "worker_exception",
                "exception_type": type(error).__name__,
                "message": str(error),
                "env_steps": last_env_steps,
            }
        )
        manifest.update(
            {
                "status": "failed",
                "failure": failure,
                "partial_results": partial_results(
                    eval_records,
                    train_records=train_records,
                    failure=failure,
                ),
            }
        )
        write_json(output_dir / "worker_manifest.json", manifest)
        raise

    summary = make_summary(
        args,
        eval_records,
        resolved_env_steps=resolved_env_steps,
        dependency=dependency,
        output_dir=output_dir,
    )
    write_json(output_dir / "summary.json", summary)
    manifest["status"] = "complete"
    manifest["result"] = {
        key: summary[key]
        for key in (
            "final_return",
            "stage_best_return",
            "overall_best_return",
        )
    }
    write_json(output_dir / "worker_manifest.json", manifest)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ["WANDB_MODE"] = "disabled"
    try:
        summary = run(args)
    except BaselineError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
