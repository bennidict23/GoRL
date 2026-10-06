"""From-scratch GoRL orchestration for the Humanoid profile."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from gorl.artifacts import (
    RunArtifacts,
    atomic_write_json,
    benchmark_schedule_provenance,
)
from gorl.checkpoints import save_pickle_checkpoint, sha256_file
from gorl.config import (
    JAX_MATMUL_PRECISIONS,
    ConfigError,
    RunConfig,
    resolve_stage_settings,
)
from gorl.runtime import humanoid_gorl_runtime_plan
from gorl.seeding import SeedResolution, resolve_seed
from gorl.tracking import EvalEvent, FailureEvent, StageTimeline, Tracker, WandbConfig

from ._decoder_worker import DecoderWorkerError, launch_decoder_worker
from ._ppo_worker import PPOWorkerError, launch_ppo_worker
from .backend import (
    _load_environment,
    _load_runtime,
    _network_factory,
    train_latent_encoder,
    train_teacher,
)
from .checkpoint import load_state, save_state
from .shared_prefix import (
    SharedPrefixArtifact,
    SharedPrefixBundle,
    require_bundle_contract,
    verify_shared_prefix,
)
from .types import (
    DecoderKind,
    EventSource,
    HumanoidEvaluationConfig,
    HumanoidLatentStageConfig,
    HumanoidMetricEvent,
    HumanoidNonFiniteMetricError,
    HumanoidPPOConfig,
    HumanoidPPOExecutionProfile,
    HumanoidPPOState,
    HumanoidTrainingResult,
    HumanoidWarmStart,
    Initialization,
)

if TYPE_CHECKING:
    import numpy as np


class HumanoidPipelineError(RuntimeError):
    pass


class HumanoidTeacherDatasetQualityError(RuntimeError):
    def __init__(self, gate: Mapping[str, Any]) -> None:
        self.gate = dict(gate)
        super().__init__(
            "Humanoid teacher dataset return mean is below the configured "
            f"minimum: {gate['observed']} < {gate['threshold']}"
        )


_METHOD_DECODER = {
    "gorl_fm": DecoderKind.FM,
    "gorl_diffusion": DecoderKind.DIFFUSION,
}

_PPO_FIELDS = (
    "action_repeat",
    "batch_size",
    "bootstrap_on_timeout",
    "clipping_epsilon_value",
    "discounting",
    "entropy_cost",
    "episode_length",
    "gae_lambda",
    "learning_rate",
    "max_grad_norm",
    "normalize_advantage",
    "normalize_observations",
    "normalize_observations_mode",
    "normalize_observations_std_eps",
    "num_envs",
    "num_minibatches",
    "num_resets_per_eval",
    "num_updates_per_batch",
    "reward_scaling",
    "unroll_length",
    "use_pmap_on_reset",
    "vf_loss_coefficient",
)


def _matmul_precision_plan(config: RunConfig) -> dict[str, str]:
    runtime = config.section("runtime")
    training = config.section("training")
    teacher = str(runtime.get("jax_default_matmul_precision", "default"))
    downstream = str(training.get("downstream_jax_default_matmul_precision", teacher))
    for phase, value in (("teacher", teacher), ("downstream", downstream)):
        if value not in JAX_MATMUL_PRECISIONS:
            choices = ", ".join(sorted(JAX_MATMUL_PRECISIONS))
            raise HumanoidPipelineError(
                f"{phase} matmul precision must be one of {choices}; got {value!r}"
            )
    return {"teacher": teacher, "downstream": downstream}


def _triton_phase(
    values: Mapping[str, Any],
    *,
    process: str,
    applied: bool,
) -> dict[str, Any]:
    effective = values.get("effective")
    return {
        "configured": values.get("configured"),
        "effective": effective,
        "effective_label": (
            ("unset" if effective is None else ("enabled" if effective else "disabled"))
            if applied
            else None
        ),
        "source": values.get("source"),
        "process": process,
        "applied": applied,
    }


def _autotune_phase(
    values: Mapping[str, Any],
    *,
    process: str,
    applied: bool,
) -> dict[str, Any]:
    effective = values.get("effective")
    return {
        "configured": values.get("configured"),
        "effective": effective,
        "effective_label": ("unset" if effective is None else str(effective))
        if applied
        else None,
        "source": values.get("source"),
        "process": process,
        "applied": applied,
    }


def _runtime_schedule(
    config: RunConfig,
    precision_schedule: Mapping[str, Any],
    *,
    teacher_applied: bool,
    downstream_applied: bool,
    worker: Mapping[str, Any] | None = None,
    latent_workers: Sequence[Mapping[str, Any]] = (),
    decoder_workers: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    plan = humanoid_gorl_runtime_plan(config)
    separate = bool(plan["teacher_process_required"])
    separate_latent = bool(plan["latent_process_required"])
    separate_decoder = bool(plan["decoder_fit_process_required"])
    teacher_process = "teacher_worker" if separate else "main"
    return {
        "schema_version": 1,
        "execution_profile": plan["execution_profile"],
        "execution_contract": plan["execution_contract"],
        "process_requirements": plan["process_requirements"],
        "process_model": (
            "per_ppo_and_decoder_fit_subprocess"
            if separate_latent and separate_decoder
            else (
                "per_ppo_phase_subprocess"
                if separate_latent
                else ("teacher_subprocess" if separate else "single_process")
            )
        ),
        "teacher_process_required": separate,
        "latent_process_required": separate_latent,
        "decoder_fit_process_required": separate_decoder,
        "teacher": {
            "process": teacher_process,
            "jax_default_matmul_precision": dict(precision_schedule["teacher"]),
            "xla_gpu_triton_gemm_any": _triton_phase(
                plan["teacher"]["xla_gpu_triton_gemm_any"],
                process=teacher_process,
                applied=teacher_applied,
            ),
            "xla_gpu_autotune_level": _autotune_phase(
                plan["teacher"]["xla_gpu_autotune_level"],
                process=teacher_process,
                applied=teacher_applied,
            ),
        },
        "downstream": {
            "process": "main",
            "ppo_process": "latent_worker" if separate_latent else "main",
            "decoder_fit_process": ("decoder_worker" if separate_decoder else "main"),
            "jax_default_matmul_precision": dict(precision_schedule["downstream"]),
            "xla_gpu_triton_gemm_any": _triton_phase(
                plan["downstream"]["xla_gpu_triton_gemm_any"],
                process="main",
                applied=downstream_applied,
            ),
            "xla_gpu_autotune_level": _autotune_phase(
                plan["downstream"]["xla_gpu_autotune_level"],
                process="main",
                applied=downstream_applied,
            ),
        },
        "teacher_worker": None if worker is None else dict(worker),
        "latent_workers": [dict(value) for value in latent_workers],
        "decoder_workers": [dict(value) for value in decoder_workers],
    }


def _worker_effective_precision(worker: Mapping[str, Any]) -> str | None:
    runtime = worker.get("runtime")
    if isinstance(runtime, Mapping) and bool(
        runtime.get("jax_default_matmul_precision_applied")
    ):
        return runtime.get("jax_default_matmul_precision_effective")
    bootstrap = worker.get("bootstrap")
    if isinstance(bootstrap, Mapping):
        post_import = bootstrap.get("post_import_environment")
        if isinstance(post_import, Mapping):
            value = post_import.get("jax_default_matmul_precision")
            return None if value in {None, "default"} else str(value)
    return None


def _effective_matmul_precision(value: str) -> str | None:
    return None if value == "default" else value


def _apply_matmul_precision(value: str) -> str | None:
    effective = _effective_matmul_precision(value)
    if effective is None:
        os.environ.pop("JAX_DEFAULT_MATMUL_PRECISION", None)
    else:
        os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = effective
    import jax

    jax.config.update("jax_default_matmul_precision", effective)
    return jax.config.jax_default_matmul_precision


def _precision_phase(
    configured: str,
    *,
    effective: str | None = None,
    applied: bool,
) -> dict[str, Any]:
    return {
        "configured": configured,
        "effective": effective,
        "effective_label": (
            ("platform_default" if effective is None else effective)
            if applied
            else None
        ),
        "applied": applied,
    }


def _validate_effective_precision(
    *,
    phase: str,
    configured: str,
    effective: str | None,
) -> None:
    expected = _effective_matmul_precision(configured)
    if effective != expected:
        raise HumanoidPipelineError(
            f"JAX did not apply the configured {phase} matmul precision: "
            f"{effective!r} != {expected!r}"
        )


def _training_event_record(event: HumanoidMetricEvent) -> dict[str, Any]:
    if event.return_mean is None:
        raise HumanoidPipelineError(
            "Humanoid training evaluation is missing "
            f"eval/episode_reward at env_steps={event.actual_environment_steps}"
        )
    if event.return_std is None:
        raise HumanoidPipelineError(
            "Humanoid training evaluation is missing "
            f"eval/episode_reward_std at env_steps={event.actual_environment_steps}"
        )
    return {
        "actual_env_steps": event.actual_environment_steps,
        "return_mean": event.return_mean,
        "return_std": event.return_std,
        "metrics": dict(event.metrics),
    }


def _validate_training_result(
    result: HumanoidTrainingResult,
    observed: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> None:
    if not observed:
        raise HumanoidPipelineError(f"{label} produced no return evaluations")
    if int(observed[-1]["actual_env_steps"]) != result.actual_environment_steps:
        raise HumanoidPipelineError(
            f"last {label} callback step does not match the final state: "
            f"{observed[-1]['actual_env_steps']} != "
            f"{result.actual_environment_steps}"
        )
    returned = [
        event for event in result.events if event.source is EventSource.TRAINING_EVAL
    ]
    if len(returned) != len(result.events) or len(returned) != len(observed):
        raise HumanoidPipelineError(
            f"{label} callback and returned training event counts differ"
        )
    for event, record in zip(returned, observed, strict=True):
        if (
            event.actual_environment_steps != record["actual_env_steps"]
            or event.return_mean != record["return_mean"]
            or event.return_std != record["return_std"]
            or dict(event.metrics) != record["metrics"]
        ):
            raise HumanoidPipelineError(
                f"{label} callback and returned training events differ"
            )
    returns = [float(record["return_mean"]) for record in observed]
    if result.final_return != returns[-1]:
        raise HumanoidPipelineError(
            f"{label} final_return does not match its last callback"
        )
    if result.best_return != max(returns):
        raise HumanoidPipelineError(
            f"{label} best_return does not match its callback curve"
        )


def _checkpointable_best_metric(
    result: HumanoidTrainingResult,
    *,
    label: str,
) -> tuple[float | None, int | None]:
    if result.best_state is None:
        return None, None
    checkpoint_step = result.best_state.actual_environment_steps
    matches = [
        event
        for event in result.events
        if event.source is EventSource.TRAINING_EVAL
        and event.actual_environment_steps == checkpoint_step
        and event.return_mean is not None
    ]
    if not matches:
        raise HumanoidPipelineError(
            f"{label} checkpointable best has no matching training evaluation at "
            f"env_steps={checkpoint_step}"
        )
    return float(matches[-1].return_mean), int(checkpoint_step)


def _failure_values(
    error: BaseException,
    *,
    phase: str,
    env_steps: int,
) -> dict[str, Any]:
    if isinstance(error, HumanoidTeacherDatasetQualityError):
        return {
            "type": "teacher_dataset_quality_gate_failed",
            "source": phase,
            "error_type": type(error).__name__,
            "env_steps": env_steps,
            "message": str(error),
            "metric": error.gate["metric"],
            "threshold": error.gate["threshold"],
            "observed": error.gate["observed"],
            "dataset_quality_gate": dict(error.gate),
        }
    if isinstance(error, HumanoidNonFiniteMetricError):
        failure = error.to_dict()
        failure["phase"] = phase
        return failure
    if isinstance(error, (PPOWorkerError, DecoderWorkerError)):
        failure = dict(error.failure)
        failure["phase"] = phase
        failure.setdefault("env_steps", env_steps)
        failure.setdefault("message", str(error))
        return failure
    return {
        "type": "humanoid_pipeline_exception",
        "source": phase,
        "error_type": type(error).__name__,
        "env_steps": env_steps,
        "message": str(error) or type(error).__name__,
    }


def _record_failed_tracker(
    tracker: Tracker,
    event: FailureEvent,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    tracking_failures: list[dict[str, str]] = []
    try:
        tracker.log_failure(event)
    except BaseException as error:
        tracking_failures.append(
            {
                "operation": "log_failure",
                "error_type": type(error).__name__,
                "message": str(error) or type(error).__name__,
            }
        )
    try:
        warnings = [warning.to_dict() for warning in tracker.warnings.read_all()]
    except BaseException as error:
        tracking_failures.append(
            {
                "operation": "read_warnings",
                "error_type": type(error).__name__,
                "message": str(error) or type(error).__name__,
            }
        )
        warnings = []
    return tracking_failures, warnings


def _finite_partial(
    records: Sequence[Mapping[str, Any]],
    *,
    completed_env_steps: int,
) -> dict[str, Any]:
    values = [dict(record) for record in records]
    best = (
        max(values, key=lambda record: float(record["return_mean"])) if values else None
    )
    return {
        "finite_evaluation_count": len(values),
        "completed_env_steps": completed_env_steps,
        "last_finite_evaluation": values[-1] if values else None,
        "best_finite_evaluation": best,
        "best_finite_return": (
            float(best["return_mean"]) if best is not None else None
        ),
    }


def run(
    config: RunConfig,
    artifacts: RunArtifacts,
    *,
    shared_prefix: SharedPrefixBundle | None = None,
) -> Mapping[str, Any]:
    """Run one Humanoid GoRL branch, optionally from a sealed fresh prefix."""

    if shared_prefix is None:
        return _run(config, artifacts, shared_prefix=None)
    require_bundle_contract(shared_prefix, config)
    verify_shared_prefix(shared_prefix)
    try:
        return _run(config, artifacts, shared_prefix=shared_prefix)
    finally:
        verify_shared_prefix(shared_prefix)


def _run(
    config: RunConfig,
    artifacts: RunArtifacts,
    *,
    shared_prefix: SharedPrefixBundle | None,
) -> Mapping[str, Any]:
    """Internal implementation shared by fresh and sealed-prefix branches."""

    _validate_capabilities(config)
    precision_plan = _matmul_precision_plan(config)
    precision_schedule = {
        "teacher": _precision_phase(
            precision_plan["teacher"],
            applied=False,
        ),
        "downstream": _precision_phase(
            precision_plan["downstream"],
            applied=False,
        ),
    }
    phase_plan = humanoid_gorl_runtime_plan(config)
    separate_teacher_process = bool(phase_plan["teacher_process_required"])
    separate_latent_process = bool(phase_plan["latent_process_required"])
    latent_worker_provenance: list[Mapping[str, Any]] = []
    decoder_worker_provenance: list[Mapping[str, Any]] = []
    runtime_schedule = _runtime_schedule(
        config,
        precision_schedule,
        teacher_applied=False,
        downstream_applied=False,
    )
    artifacts.update_manifest(
        {
            "precision_schedule": precision_schedule,
            "runtime_schedule": runtime_schedule,
        }
    )
    shared_prefix_manifest = (
        None if shared_prefix is None else _shared_prefix_manifest(shared_prefix)
    )
    if shared_prefix_manifest is not None:
        artifacts.update_manifest({"shared_prefix": shared_prefix_manifest})
    if shared_prefix is not None:
        effective_downstream_precision = _apply_matmul_precision(
            precision_plan["downstream"]
        )
        precision_schedule["downstream"] = _precision_phase(
            precision_plan["downstream"],
            effective=effective_downstream_precision,
            applied=True,
        )
        _validate_effective_precision(
            phase="downstream",
            configured=precision_plan["downstream"],
            effective=effective_downstream_precision,
        )
        runtime_schedule = _runtime_schedule(
            config,
            precision_schedule,
            teacher_applied=False,
            downstream_applied=True,
        )
    elif separate_teacher_process:
        effective_downstream_precision = _apply_matmul_precision(
            precision_plan["downstream"]
        )
        precision_schedule["downstream"] = _precision_phase(
            precision_plan["downstream"],
            effective=effective_downstream_precision,
            applied=True,
        )
        _validate_effective_precision(
            phase="downstream",
            configured=precision_plan["downstream"],
            effective=effective_downstream_precision,
        )
        runtime_schedule = _runtime_schedule(
            config,
            precision_schedule,
            teacher_applied=False,
            downstream_applied=True,
        )
    else:
        effective_teacher_precision = _apply_matmul_precision(precision_plan["teacher"])
        precision_schedule["teacher"] = _precision_phase(
            precision_plan["teacher"],
            effective=effective_teacher_precision,
            applied=True,
        )
        _validate_effective_precision(
            phase="teacher",
            configured=precision_plan["teacher"],
            effective=effective_teacher_precision,
        )
        runtime_schedule = _runtime_schedule(
            config,
            precision_schedule,
            teacher_applied=True,
            downstream_applied=False,
        )
    artifacts.update_manifest(
        {
            "precision_schedule": precision_schedule,
            "runtime_schedule": runtime_schedule,
        }
    )
    handoff_selection = _encoder_handoff(config)
    stage0_initialization = _stage0_encoder_initialization(config)
    ppo_values = [
        _stage_ppo_values(config, stage) for stage in range(len(config.stage_steps))
    ]
    latent_stages = tuple(range(len(config.stage_steps)))
    decoder_kinds = tuple(_decoder_kind(config, stage) for stage in latent_stages)
    teacher_requested_steps = _teacher_requested_steps(config)
    teacher_values = _stage_ppo_values(config, 0)
    teacher_seed = _teacher_seed_resolution(config)
    teacher_config = _make_ppo_config(
        config,
        values=teacher_values,
        requested_steps=teacher_requested_steps,
        seed=teacher_seed.seed,
        stage_index=0,
    )
    predicted_teacher_steps = _resolved_actual_steps(
        teacher_requested_steps,
        teacher_values,
    )
    compute_stage_steps = tuple(
        _resolved_actual_steps(
            config.stage_steps[stage],
            ppo_values[stage],
        )
        for stage in range(len(config.stage_steps))
    )
    training_values = config.section("training")
    benchmark_start = int(training_values.get("benchmark_start_env_steps", 0))
    benchmark_schedule = benchmark_schedule_provenance(config)
    timeline = StageTimeline(
        config.stage_steps,
        compute_stage_steps=compute_stage_steps,
        benchmark_start_env_steps=benchmark_start,
        compute_start_env_steps=predicted_teacher_steps,
    )
    tracker = artifacts.start_tracking(
        wandb=_wandb_config(config),
    )
    collection_values = config.section("collection")
    episode_length = int(teacher_values["episode_length"])
    teacher_episodes = (
        int(collection_values["episodes"])
        if config.smoke
        else int(
            collection_values.get(
                "teacher_episodes",
                collection_values["episodes"],
            )
        )
    )
    teacher_dir = artifacts.run_dir / "teacher"
    teacher_dir.mkdir(parents=True, exist_ok=True)
    teacher_events_path = teacher_dir / "training_events.jsonl"
    teacher_records: list[dict[str, Any]] = []
    teacher_completed_steps = (
        predicted_teacher_steps if shared_prefix is not None else 0
    )
    teacher_phase = (
        "shared_prefix_checkpoint" if shared_prefix is not None else "teacher_training"
    )
    teacher_worker_provenance: Mapping[str, Any] | None = None
    initial_decoder_worker_provenance: Mapping[str, Any] | None = None
    teacher_collection_manifest: dict[str, Any] | None = None
    teacher_dataset_quality_gate: dict[str, Any] | None = None
    teacher_final_checkpoint: dict[str, Any] | None = None
    teacher_best_checkpoint: dict[str, Any] | None = None
    teacher_source_checkpoint: dict[str, Any] | None = None

    def on_teacher_metric(event: HumanoidMetricEvent) -> None:
        nonlocal teacher_completed_steps
        _append_metric_event(teacher_events_path, event)
        record_teacher_metric(event)

    def record_teacher_metric(event: HumanoidMetricEvent) -> None:
        nonlocal teacher_completed_steps
        if event.source is not EventSource.TRAINING_EVAL:
            return
        teacher_completed_steps = max(
            teacher_completed_steps,
            event.actual_environment_steps,
        )
        teacher_records.append(_training_event_record(event))

    try:
        teacher_collection_seed = _teacher_collection_seed_resolution(config)
        if shared_prefix is None:
            teacher_report_config = (
                _report_config(config, teacher_values)
                if bool(training_values.get("teacher_deterministic_report", False))
                else None
            )
            if separate_teacher_process:
                teacher_phase = "teacher_worker"
                outcome = launch_ppo_worker(
                    phase="teacher",
                    ppo_config=teacher_config,
                    report_config=teacher_report_config,
                    phase_runtime=phase_plan["teacher"],
                    worker_dir=teacher_dir / "worker",
                    events_path=teacher_events_path,
                    callback=record_teacher_metric,
                    input_artifacts={},
                )
                teacher_result = outcome.result
                teacher_worker_provenance = outcome.provenance
                effective_teacher_precision = _worker_effective_precision(
                    outcome.provenance
                )
                precision_schedule["teacher"] = _precision_phase(
                    precision_plan["teacher"],
                    effective=effective_teacher_precision,
                    applied=True,
                )
                _validate_effective_precision(
                    phase="teacher",
                    configured=precision_plan["teacher"],
                    effective=effective_teacher_precision,
                )
                runtime_schedule = _runtime_schedule(
                    config,
                    precision_schedule,
                    teacher_applied=True,
                    downstream_applied=True,
                    worker=teacher_worker_provenance,
                )
                artifacts.update_manifest(
                    {
                        "precision_schedule": precision_schedule,
                        "runtime_schedule": runtime_schedule,
                    }
                )
            else:
                teacher_result = train_teacher(
                    teacher_config,
                    callback=on_teacher_metric,
                    report_config=teacher_report_config,
                )
            if not teacher_result.from_scratch:
                raise HumanoidPipelineError(
                    "teacher backend did not report from-scratch initialization"
                )
            if teacher_result.actual_environment_steps != predicted_teacher_steps:
                raise HumanoidPipelineError(
                    "predicted and observed teacher step counts differ: "
                    f"{predicted_teacher_steps} != "
                    f"{teacher_result.actual_environment_steps}"
                )
            _validate_training_result(
                teacher_result,
                teacher_records,
                label="Humanoid teacher",
            )
            _validate_declared_teacher_steps(
                config,
                teacher_result.actual_environment_steps,
            )
            teacher_phase = "teacher_checkpoint"
            teacher_final_checkpoint = _seeded_artifact(
                _save_policy_checkpoint(
                    teacher_result.state,
                    teacher_dir / "final_state.pkl",
                ),
                teacher_seed,
            )
            teacher_best_checkpoint = None
            if teacher_result.best_state is not None:
                teacher_best_checkpoint = _seeded_artifact(
                    _save_policy_checkpoint(
                        teacher_result.best_state,
                        teacher_dir / "best_state.pkl",
                    ),
                    teacher_seed,
                )
            teacher_source_state = teacher_result.state
            teacher_source_checkpoint = teacher_final_checkpoint
            teacher_actual_steps = teacher_result.actual_environment_steps
            teacher_initialization = "from_scratch"
            teacher_final_return = teacher_result.final_return
            teacher_best_return = teacher_result.best_return
            teacher_events = [
                _metric_event_dict(event) for event in teacher_result.events
            ]
            teacher_report = _report_manifest(teacher_result)
            decoder_data_selection = "final"

            if not separate_teacher_process:
                teacher_phase = "downstream_precision"
                effective_downstream_precision = _apply_matmul_precision(
                    precision_plan["downstream"]
                )
                precision_schedule["downstream"] = _precision_phase(
                    precision_plan["downstream"],
                    effective=effective_downstream_precision,
                    applied=True,
                )
                _validate_effective_precision(
                    phase="downstream",
                    configured=precision_plan["downstream"],
                    effective=effective_downstream_precision,
                )
                runtime_schedule = _runtime_schedule(
                    config,
                    precision_schedule,
                    teacher_applied=True,
                    downstream_applied=True,
                )
                artifacts.update_manifest(
                    {
                        "precision_schedule": precision_schedule,
                        "runtime_schedule": runtime_schedule,
                    }
                )

            teacher_phase = "teacher_collection"
            teacher_dataset = _collect_policy_dataset(
                state=teacher_source_state,
                decoder=None,
                num_episodes=teacher_episodes,
                num_envs=_collection_num_envs(collection_values),
                episode_length=episode_length,
                seed=teacher_collection_seed.seed,
                stochastic=bool(collection_values.get("stochastic", True)),
                transition_order=str(
                    collection_values.get(
                        "teacher_transition_order",
                        collection_values.get("transition_order", "episode_major"),
                    )
                ),
            )
            teacher_dataset_path = teacher_dataset.save(
                teacher_dir / "decoder_dataset.npz"
            )
            teacher_collection_manifest = _dataset_manifest(
                teacher_dataset,
                teacher_dataset_path,
                seed_resolution=teacher_collection_seed,
            )
            teacher_backbone_point = (
                None
                if benchmark_start <= 0
                else {
                    "source": "teacher_dataset_mean",
                    "return_mean": float(teacher_dataset.stats.return_mean),
                    "return_std": float(teacher_dataset.stats.return_std),
                    "actual_compute_env_steps": teacher_actual_steps,
                }
            )
        else:
            teacher_phase = "shared_prefix_checkpoint"
            teacher_source_state = load_state(shared_prefix.checkpoint.path)
            _validate_shared_teacher_state(
                config,
                teacher_source_state,
                requested_steps=teacher_requested_steps,
                actual_steps=predicted_teacher_steps,
                seed=teacher_seed.seed,
            )
            _validate_declared_teacher_steps(
                config,
                teacher_source_state.actual_environment_steps,
            )
            teacher_actual_steps = teacher_source_state.actual_environment_steps
            teacher_final_checkpoint = _seeded_artifact(
                {
                    **_shared_prefix_artifact(shared_prefix.checkpoint),
                    "format": "gorl_humanoid_ppo_state",
                    "shared_prefix_id": shared_prefix.shared_prefix_id,
                },
                teacher_seed,
            )
            teacher_best_checkpoint = None
            teacher_source_checkpoint = teacher_final_checkpoint
            teacher_initialization = "shared_prefix"
            teacher_final_return = None
            teacher_best_return = None
            teacher_events = []
            teacher_report = None
            decoder_data_selection = "shared_prefix"
            teacher_phase = "shared_prefix_dataset"
            teacher_dataset = _load_shared_decoder_dataset(
                shared_prefix.dataset.path,
                include_latent=(
                    float(
                        _decoder_stage_values(config, 0).get(
                            "pairwise_loss_weight", 0.0
                        )
                    )
                    > 0.0
                ),
            )
            teacher_dataset_path = shared_prefix.dataset.path
            teacher_collection_manifest = _shared_prefix_dataset_manifest(
                shared_prefix,
                teacher_dataset,
            )
            benchmark_anchor = shared_prefix.benchmark_anchor
            if benchmark_start > 0 and benchmark_anchor is None:
                raise HumanoidPipelineError(
                    "shared-prefix benchmark anchor is required for this task"
                )
            teacher_backbone_point = (
                None
                if benchmark_anchor is None
                else {
                    "source": benchmark_anchor.source,
                    "return_mean": benchmark_anchor.return_mean,
                    "return_std": benchmark_anchor.return_std,
                    "actual_compute_env_steps": (
                        benchmark_anchor.actual_compute_env_steps
                    ),
                }
            )

        teacher_phase = "teacher_dataset_quality_gate"
        teacher_dataset_quality_gate = _teacher_dataset_quality_gate_record(
            config,
            teacher_collection_manifest,
        )
        if (
            teacher_dataset_quality_gate["enabled"]
            and not teacher_dataset_quality_gate["passed"]
        ):
            raise HumanoidTeacherDatasetQualityError(teacher_dataset_quality_gate)

        teacher_phase = "teacher_decoder_training"
        initial_decoder_seed = _decoder_seed_resolution(config, 0)
        (
            initial_decoder_fit,
            initial_decoder_config,
            initial_decoder_worker_provenance,
        ) = _fit_configured_decoder(
            config,
            dataset=teacher_dataset,
            dataset_path=teacher_dataset_path,
            target_stage=0,
            metrics_path=teacher_dir / "decoder_metrics.jsonl",
            worker_dir=teacher_dir / "decoder_worker",
            phase_runtime=phase_plan["downstream"],
            previous_fit=None,
        )
        if initial_decoder_worker_provenance is not None:
            decoder_worker_provenance.append(initial_decoder_worker_provenance)
            runtime_schedule = _runtime_schedule(
                config,
                precision_schedule,
                teacher_applied=precision_schedule["teacher"]["applied"],
                downstream_applied=precision_schedule["downstream"]["applied"],
                worker=teacher_worker_provenance,
                latent_workers=latent_worker_provenance,
                decoder_workers=decoder_worker_provenance,
            )
            artifacts.update_manifest({"runtime_schedule": runtime_schedule})
        decoder_fit = initial_decoder_fit
        decoder = initial_decoder_fit.decoder
        teacher_phase = "teacher_decoder_checkpoint"
        initial_decoder_checkpoint = _save_decoder_checkpoint(
            teacher_dir / "decoder_best.pkl",
            config=config,
            target_stage=0,
            source=(
                "fresh_teacher-final"
                if shared_prefix is None
                else f"shared_prefix-{shared_prefix.shared_prefix_id}"
            ),
            dataset_path=teacher_dataset_path,
            decoder_config=initial_decoder_config,
            fitted=initial_decoder_fit,
            warm_start_checkpoint=None,
            seed_resolution=initial_decoder_seed,
        )

        teacher_manifest = {
            "status": "complete",
            "ppo_execution_profile": phase_plan["execution_profile"],
            "ppo_execution_contract": phase_plan["execution_contract"]["teacher"],
            "precision_schedule": precision_schedule,
            "runtime_schedule": runtime_schedule,
            "teacher_worker": teacher_worker_provenance,
            "initial_decoder_worker": initial_decoder_worker_provenance,
            "initialization": teacher_initialization,
            "requested_env_steps": teacher_requested_steps,
            "actual_env_steps": teacher_actual_steps,
            "seed": teacher_seed.seed,
            "seed_resolution": teacher_seed.to_dict(),
            "training_eval_deterministic": (
                teacher_source_state.training_eval_deterministic
            ),
            "primary_metric": _primary_metric_manifest(config),
            "final_return": teacher_final_return,
            "best_return": teacher_best_return,
            "events": teacher_events,
            "deterministic_report": teacher_report,
            "checkpoints": {
                "final": teacher_final_checkpoint,
                "best": teacher_best_checkpoint,
                "selected_for_decoder_data": teacher_source_checkpoint,
                "decoder_data_selection": decoder_data_selection,
            },
            "collection": teacher_collection_manifest,
            "dataset_quality_gate": teacher_dataset_quality_gate,
            "benchmark_backbone_point": teacher_backbone_point,
            "initial_decoder": _decoder_manifest(
                initial_decoder_fit,
                initial_decoder_config,
                initial_decoder_checkpoint,
                seed_resolution=initial_decoder_seed,
            ),
            **(
                {}
                if shared_prefix_manifest is None
                else {"shared_prefix": shared_prefix_manifest}
            ),
        }
        teacher_phase = "teacher_manifest"
        atomic_write_json(teacher_dir / "manifest.json", teacher_manifest)
    except BaseException as error:
        if isinstance(error, (PPOWorkerError, DecoderWorkerError)):
            worker = error.failure.get("worker")
            if isinstance(worker, Mapping):
                if isinstance(error, DecoderWorkerError):
                    initial_decoder_worker_provenance = dict(worker)
                    decoder_worker_provenance.append(initial_decoder_worker_provenance)
                else:
                    teacher_worker_provenance = dict(worker)
                    effective_worker_precision = _worker_effective_precision(worker)
                    if effective_worker_precision is not None:
                        precision_schedule["teacher"] = _precision_phase(
                            precision_plan["teacher"],
                            effective=effective_worker_precision,
                            applied=True,
                        )
                runtime_schedule = _runtime_schedule(
                    config,
                    precision_schedule,
                    teacher_applied=precision_schedule["teacher"]["applied"],
                    downstream_applied=precision_schedule["downstream"]["applied"],
                    worker=teacher_worker_provenance,
                    latent_workers=latent_worker_provenance,
                    decoder_workers=decoder_worker_provenance,
                )
                artifacts.update_manifest(
                    {
                        "precision_schedule": precision_schedule,
                        "runtime_schedule": runtime_schedule,
                    }
                )
        raw_failure_steps = (
            error.env_steps
            if isinstance(error, HumanoidNonFiniteMetricError)
            else teacher_completed_steps
        )
        failure_steps = min(max(int(raw_failure_steps), 0), predicted_teacher_steps)
        failure = _failure_values(
            error,
            phase=teacher_phase,
            env_steps=failure_steps,
        )
        failure_event = FailureEvent(
            task=config.task,
            method=config.method,
            seed=config.seed,
            stage=0,
            local_env_steps=0,
            env_steps=0,
            benchmark_env_steps=0.0,
            compute_actual_env_steps=failure_steps,
            plot_env_steps=0.0,
            error_type=str(failure["type"]),
            message=str(failure["message"]),
        )
        tracking_failures, tracking_warnings = _record_failed_tracker(
            tracker,
            failure_event,
        )
        if tracking_failures:
            failure["tracking_failures"] = tracking_failures
        partial_results = _finite_partial(
            teacher_records,
            completed_env_steps=failure_steps,
        )
        partial_results["phase"] = teacher_phase
        partial_results["events_path"] = str(teacher_events_path)
        if teacher_collection_manifest is not None:
            partial_results["collection"] = teacher_collection_manifest
        failed_teacher = {
            "status": "failed",
            "phase": teacher_phase,
            "ppo_execution_profile": phase_plan["execution_profile"],
            "ppo_execution_contract": phase_plan["execution_contract"]["teacher"],
            "precision_schedule": precision_schedule,
            "runtime_schedule": runtime_schedule,
            "teacher_worker": teacher_worker_provenance,
            "initial_decoder_worker": initial_decoder_worker_provenance,
            "initialization": (
                "from_scratch" if shared_prefix is None else "shared_prefix"
            ),
            "requested_env_steps": teacher_requested_steps,
            "planned_actual_env_steps": predicted_teacher_steps,
            "actual_env_steps": failure_steps,
            "seed": teacher_seed.seed,
            "seed_resolution": teacher_seed.to_dict(),
            "primary_metric": _primary_metric_manifest(config),
            "final_return": None,
            "best_return": None,
            "stage_best_return": None,
            "overall_best_return": None,
            "deterministic_final_return": None,
            "checkpoints": {
                "final": teacher_final_checkpoint,
                "best": teacher_best_checkpoint,
                "selected_for_decoder_data": teacher_source_checkpoint,
            },
            "collection": teacher_collection_manifest,
            "dataset_quality_gate": teacher_dataset_quality_gate,
            "failure": failure,
            "partial_results": partial_results,
            "tracking_warnings": tracking_warnings,
            **(
                {}
                if shared_prefix_manifest is None
                else {"shared_prefix": shared_prefix_manifest}
            ),
        }
        atomic_write_json(teacher_dir / "manifest.json", failed_teacher)
        failed_stage = {
            "stage": 0,
            "status": "failed",
            "phase": "teacher",
            "precision_schedule": precision_schedule,
            "runtime_schedule": runtime_schedule,
            "requested_env_steps": teacher_requested_steps,
            "planned_actual_env_steps": predicted_teacher_steps,
            "actual_env_steps": failure_steps,
            "evaluations": [],
            "final_return": None,
            "best_return": None,
            "stage_best_return": None,
            "overall_best_return": None,
            "deterministic_final_return": None,
            "failure": failure,
            "partial_results": partial_results,
            "tracking_warnings": tracking_warnings,
            **(
                {}
                if shared_prefix_manifest is None
                else {"shared_prefix": shared_prefix_manifest}
            ),
        }
        artifacts.write_stage_manifest(0, failed_stage)
        failed_summary = {
            "status": "failed",
            "task": config.task,
            "method": config.method,
            "seed": config.seed,
            "profile": config.profile,
            "fresh_from_scratch": shared_prefix is None,
            "execution_mode": (
                "ordinary_fresh_train"
                if shared_prefix is None
                else "shared_prefix_branch"
            ),
            "precision_schedule": precision_schedule,
            "runtime_schedule": runtime_schedule,
            "teacher": failed_teacher,
            "stage_steps": list(config.stage_steps),
            "compute_stage_steps": list(compute_stage_steps),
            "benchmark_start_env_steps": benchmark_start,
            "compute_start_env_steps": predicted_teacher_steps,
            "stages": [],
            "failed_stage": failed_stage,
            "final_return": None,
            "stage_best_return": None,
            "overall_best_return": None,
            "deterministic_final_return": None,
            "target_return": config.target_return,
            "performance_target_met": None,
            "failure": failure,
            "partial_results": partial_results,
            "tracking_warnings": tracking_warnings,
        }
        atomic_write_json(artifacts.run_dir / "summary.json", failed_summary)
        artifacts.update_manifest(
            {
                "final_return": None,
                "stage_best_return": None,
                "overall_best_return": None,
                "failure": failure,
                "partial_results": partial_results,
                "tracking_warnings": tracking_warnings,
            }
        )
        if isinstance(error, Exception):
            raise HumanoidPipelineError(str(failure["message"])) from error
        raise

    stage_summaries: list[dict[str, Any]] = []
    all_returns: list[float] = []
    previous_final: float | None = None
    current_stage = 0
    current_phase = "benchmark_backbone"
    current_completed_steps = 0
    current_records: list[dict[str, Any]] = []
    current_eval_events: list[EvalEvent] = []
    input_decoder_manifest = _decoder_manifest(
        initial_decoder_fit,
        initial_decoder_config,
        initial_decoder_checkpoint,
        seed_resolution=initial_decoder_seed,
    )
    warm_state = None if stage0_initialization == "fresh" else teacher_source_state
    warm_checkpoint_manifest = (
        None if stage0_initialization == "fresh" else teacher_source_checkpoint
    )
    current_worker_provenance: Mapping[str, Any] | None = None
    current_decoder_worker_provenance: Mapping[str, Any] | None = None
    try:
        if benchmark_start > 0:
            if teacher_backbone_point is None:
                raise HumanoidPipelineError("missing Humanoid backbone anchor")
            teacher_point = _teacher_backbone_event(
                config=config,
                benchmark_start=benchmark_start,
                actual_steps=teacher_actual_steps,
                return_mean=float(teacher_backbone_point["return_mean"]),
                return_std=float(teacher_backbone_point["return_std"]),
            )
            tracker.log_eval(teacher_point)
            previous_final = teacher_point.return_mean
            all_returns.append(teacher_point.return_mean)
        for stage in latent_stages:
            current_stage = stage
            current_phase = "latent_training"
            current_completed_steps = 0
            current_records = []
            current_eval_events = []
            current_worker_provenance = None
            current_decoder_worker_provenance = None
            requested_steps = config.stage_steps[stage]
            stage_dir = artifacts.stages_dir / f"stage-{stage}"
            stage_dir.mkdir(parents=True, exist_ok=True)
            raw_events_path = stage_dir / "backend_events.jsonl"

            def record_metric(
                event: HumanoidMetricEvent,
                *,
                stage_index: int = stage,
            ) -> None:
                nonlocal current_completed_steps
                if event.source is not EventSource.TRAINING_EVAL:
                    return
                current_completed_steps = max(
                    current_completed_steps,
                    event.actual_environment_steps,
                )
                record = _training_event_record(event)
                current_records.append(record)
                index_offset = 1 if benchmark_start > 0 and stage_index == 0 else 0
                tracked = timeline.event(
                    task=config.task,
                    method=config.method,
                    seed=config.seed,
                    stage=stage_index,
                    index=len(current_eval_events) + index_offset,
                    local_env_steps=event.actual_environment_steps,
                    return_mean=event.return_mean,
                    return_std=event.return_std,
                )
                if benchmark_start > 0:
                    tracked = replace(
                        tracked,
                        plot_env_steps=tracked.plot_env_steps + 0.5,
                    )
                current_eval_events.append(tracked)
                tracker.log_eval(tracked)

            def on_metric(
                event: HumanoidMetricEvent,
                *,
                stage_index: int = stage,
            ) -> None:
                _append_metric_event(raw_events_path, event)
                record_metric(event, stage_index=stage_index)

            stage_ppo_config = _make_ppo_config(
                config,
                values=ppo_values[stage],
                requested_steps=requested_steps,
                seed=_encoder_seed_resolution(config, stage).seed,
                stage_index=stage,
            )
            evaluation_config = _report_config(
                config,
                ppo_values[stage],
            )
            stage_warm_start = (
                None if warm_state is None else HumanoidWarmStart(warm_state)
            )
            latent_stage_config = HumanoidLatentStageConfig(
                ppo=stage_ppo_config,
                decoder_kind=decoder_kinds[stage],
                latent_size=int(config.section("ppo").get("latent_dim", 21)),
            )
            if separate_latent_process:
                outcome = launch_ppo_worker(
                    phase="latent",
                    ppo_config=stage_ppo_config,
                    stage_config=latent_stage_config,
                    decoder=decoder,
                    warm_start=stage_warm_start,
                    report_config=evaluation_config,
                    phase_runtime=phase_plan["downstream"],
                    worker_dir=stage_dir / "worker",
                    events_path=raw_events_path,
                    callback=record_metric,
                    input_artifacts={
                        "decoder": input_decoder_manifest,
                        "warm_start": warm_checkpoint_manifest,
                    },
                )
                result = outcome.result
                current_worker_provenance = outcome.provenance
                latent_worker_provenance.append(outcome.provenance)
                runtime_schedule = _runtime_schedule(
                    config,
                    precision_schedule,
                    teacher_applied=precision_schedule["teacher"]["applied"],
                    downstream_applied=precision_schedule["downstream"]["applied"],
                    worker=teacher_worker_provenance,
                    latent_workers=latent_worker_provenance,
                    decoder_workers=decoder_worker_provenance,
                )
                artifacts.update_manifest({"runtime_schedule": runtime_schedule})
            else:
                result = train_latent_encoder(
                    latent_stage_config,
                    decoder,
                    warm_start=stage_warm_start,
                    callback=on_metric,
                    report_config=evaluation_config,
                )
            expected_initialization = (
                Initialization.FROM_SCRATCH
                if stage_warm_start is None
                else Initialization.WARM_START
            )
            if result.initialization is not expected_initialization:
                raise HumanoidPipelineError(
                    "Humanoid encoder initialization mismatch at stage "
                    f"{stage}: expected {expected_initialization.value}, got "
                    f"{result.initialization.value}"
                )
            if result.actual_environment_steps != compute_stage_steps[stage]:
                raise HumanoidPipelineError(
                    "predicted and observed stage step counts differ at "
                    f"stage {stage}: {compute_stage_steps[stage]} != "
                    f"{result.actual_environment_steps}"
                )
            _validate_training_result(
                result,
                current_records,
                label=f"Humanoid stage {stage}",
            )

            current_phase = "stage_checkpoint"
            encoder_seed = _encoder_seed_resolution(config, stage)
            final_checkpoint = _seeded_artifact(
                _save_policy_checkpoint(
                    result.state,
                    stage_dir / "encoder_final.pkl",
                ),
                encoder_seed,
            )
            best_state = result.best_state or result.state
            best_checkpoint = _seeded_artifact(
                _save_policy_checkpoint(
                    best_state,
                    stage_dir / "encoder_best.pkl",
                ),
                encoder_seed,
            )
            best_selection = (
                "backend_best"
                if result.best_state is not None
                else "final_fallback_no_checkpointable_best"
            )
            checkpointable_best_return, checkpointable_best_step = (
                _checkpointable_best_metric(
                    result,
                    label=f"Humanoid stage {stage}",
                )
            )
            handoff_state, handoff_effective = _select_handoff_state(
                result,
                handoff_selection,
            )
            handoff_checkpoint = (
                best_checkpoint if handoff_effective == "best" else final_checkpoint
            )

            stage_return_values = [event.return_mean for event in current_eval_events]
            stability = _stage_stability(
                stage_return_values,
                previous_final=previous_final,
            )
            outgoing_collection: dict[str, Any] | None = None
            outgoing_decoder: dict[str, Any] | None = None
            if stage + 1 < len(config.stage_steps):
                collection_seed = _collection_seed_resolution(config, stage)
                current_phase = "stage_collection"
                dataset = _collect_policy_dataset(
                    state=handoff_state,
                    decoder=decoder,
                    num_episodes=int(collection_values["episodes"]),
                    num_envs=_collection_num_envs(collection_values),
                    episode_length=int(ppo_values[stage]["episode_length"]),
                    seed=collection_seed.seed,
                    stochastic=bool(collection_values.get("stochastic", True)),
                    transition_order=str(
                        collection_values.get("transition_order", "episode_major")
                    ),
                )
                dataset_path = dataset.save(stage_dir / "decoder_dataset.npz")
                current_phase = "stage_decoder_training"
                decoder_seed = _decoder_seed_resolution(config, stage + 1)
                (
                    fitted,
                    decoder_config,
                    current_decoder_worker_provenance,
                ) = _fit_configured_decoder(
                    config,
                    dataset=dataset,
                    dataset_path=dataset_path,
                    target_stage=stage + 1,
                    metrics_path=stage_dir / "decoder_metrics.jsonl",
                    worker_dir=stage_dir / "decoder_worker",
                    phase_runtime=phase_plan["downstream"],
                    previous_fit=decoder_fit,
                )
                if current_decoder_worker_provenance is not None:
                    decoder_worker_provenance.append(current_decoder_worker_provenance)
                    runtime_schedule = _runtime_schedule(
                        config,
                        precision_schedule,
                        teacher_applied=precision_schedule["teacher"]["applied"],
                        downstream_applied=precision_schedule["downstream"]["applied"],
                        worker=teacher_worker_provenance,
                        latent_workers=latent_worker_provenance,
                        decoder_workers=decoder_worker_provenance,
                    )
                    artifacts.update_manifest({"runtime_schedule": runtime_schedule})
                current_phase = "stage_decoder_checkpoint"
                decoder_checkpoint = _save_decoder_checkpoint(
                    stage_dir / "decoder_next_best.pkl",
                    config=config,
                    target_stage=stage + 1,
                    source=f"stage-{stage}-{handoff_effective}",
                    dataset_path=dataset_path,
                    decoder_config=decoder_config,
                    fitted=fitted,
                    warm_start_checkpoint=(
                        input_decoder_manifest["checkpoint"]
                        if str(
                            _decoder_stage_values(
                                config,
                                stage + 1,
                            ).get("initialization", "fresh")
                        )
                        == "warm"
                        else None
                    ),
                    seed_resolution=decoder_seed,
                )
                outgoing_collection = _dataset_manifest(
                    dataset,
                    dataset_path,
                    seed_resolution=collection_seed,
                )
                outgoing_decoder = _decoder_manifest(
                    fitted,
                    decoder_config,
                    decoder_checkpoint,
                    seed_resolution=decoder_seed,
                )
                decoder = fitted.decoder
                decoder_fit = fitted

            stage_summary = {
                "stage": stage,
                "nominal_stage_index": stage + (1 if benchmark_start > 0 else 0),
                "nominal_start_env_steps": benchmark_start
                + sum(config.stage_steps[:stage]),
                "nominal_end_env_steps": benchmark_start
                + sum(config.stage_steps[: stage + 1]),
                "status": "complete",
                "precision": precision_schedule["downstream"],
                "requested_env_steps": result.requested_environment_steps,
                "actual_env_steps": result.actual_environment_steps,
                "policy_seed": encoder_seed.seed,
                "policy_seed_resolution": encoder_seed.to_dict(),
                "evaluation_seed": config.eval_seed,
                "evaluation_seed_resolution": _evaluation_seed_resolution(
                    config, stage
                ).to_dict(),
                "encoder_initialization": (
                    "from_scratch_latent_encoder"
                    if stage_warm_start is None
                    else "warm_start_policy_value_normalizer"
                ),
                "stage0_encoder_initialization_configured": (
                    stage0_initialization if stage == 0 else None
                ),
                "ppo_execution_profile": phase_plan["execution_profile"],
                "ppo_execution_contract": phase_plan["execution_contract"]["latent"],
                "ppo_worker": current_worker_provenance,
                "input_decoder": input_decoder_manifest,
                "evaluations": [event.to_dict() for event in current_eval_events],
                "stability": stability,
                "primary_metric": _primary_metric_manifest(config),
                "deterministic_report": _report_manifest(result),
                "checkpoints": {
                    "final": final_checkpoint,
                    "best": best_checkpoint,
                    "best_selection": best_selection,
                    "checkpointable_best_return": checkpointable_best_return,
                    "checkpointable_best_actual_env_steps": (checkpointable_best_step),
                    "handoff": handoff_checkpoint,
                    "handoff_configured": handoff_selection,
                    "handoff_effective": handoff_effective,
                },
                "outgoing_collection": outgoing_collection,
                "outgoing_decoder": outgoing_decoder,
                "outgoing_decoder_worker": current_decoder_worker_provenance,
            }
            current_phase = "stage_manifest"
            artifacts.write_stage_manifest(stage, stage_summary)
            stage_summaries.append(stage_summary)
            all_returns.extend(stage_return_values)
            previous_final = float(result.final_return)
            warm_state = handoff_state
            warm_checkpoint_manifest = handoff_checkpoint
            if outgoing_decoder is not None:
                input_decoder_manifest = outgoing_decoder
        if not stage_summaries or not all_returns:
            raise HumanoidPipelineError("Humanoid pipeline produced no stages")
        current_phase = "tracking_snapshot"
        tracking_warnings = [
            warning.to_dict() for warning in tracker.warnings.read_all()
        ]
        final_stability = stage_summaries[-1]["stability"]
        final_return = float(final_stability["final_return"])
        final_stage_best = float(final_stability["best_return"])
        summary = {
            "status": "complete",
            "task": config.task,
            "method": config.method,
            "seed": config.seed,
            "profile": config.profile,
            "fresh_from_scratch": shared_prefix is None,
            "execution_mode": (
                "ordinary_fresh_train"
                if shared_prefix is None
                else "shared_prefix_branch"
            ),
            "precision_schedule": precision_schedule,
            "runtime_schedule": runtime_schedule,
            "teacher": teacher_manifest,
            "stage_steps": list(config.stage_steps),
            "compute_stage_steps": list(compute_stage_steps),
            "benchmark_start_env_steps": benchmark_start,
            "compute_start_env_steps": teacher_actual_steps,
            "benchmark_schedule": benchmark_schedule,
            "encoder_handoff": handoff_selection,
            "stage0_encoder_initialization": stage0_initialization,
            "decoder_stage_types": [kind.value for kind in decoder_kinds],
            "metric_policy": _primary_metric_manifest(config),
            "backbone_point_source": (
                "teacher_dataset_mean" if benchmark_start > 0 else None
            ),
            "stages": stage_summaries,
            "final_return": final_return,
            "stage_best_return": final_stage_best,
            "overall_best_return": max(all_returns),
            "deterministic_final_return": _deterministic_final_return(
                stage_summaries[-1]
            ),
            "target_return": config.target_return,
            "performance_target_met": final_return >= config.target_return,
            "tracking_warnings": tracking_warnings,
            **(
                {}
                if shared_prefix_manifest is None
                else {"shared_prefix": shared_prefix_manifest}
            ),
        }
        current_phase = "summary_write"
        atomic_write_json(artifacts.run_dir / "summary.json", summary)
        return summary
    except BaseException as error:
        if isinstance(error, (PPOWorkerError, DecoderWorkerError)):
            worker = error.failure.get("worker")
            if isinstance(worker, Mapping):
                if isinstance(error, DecoderWorkerError):
                    current_decoder_worker_provenance = dict(worker)
                    decoder_worker_provenance.append(current_decoder_worker_provenance)
                else:
                    current_worker_provenance = dict(worker)
                    latent_worker_provenance.append(current_worker_provenance)
                runtime_schedule = _runtime_schedule(
                    config,
                    precision_schedule,
                    teacher_applied=precision_schedule["teacher"]["applied"],
                    downstream_applied=precision_schedule["downstream"]["applied"],
                    worker=teacher_worker_provenance,
                    latent_workers=latent_worker_provenance,
                    decoder_workers=decoder_worker_provenance,
                )
                artifacts.update_manifest({"runtime_schedule": runtime_schedule})
        stage_limit = compute_stage_steps[current_stage]
        raw_failure_steps = (
            error.env_steps
            if isinstance(error, HumanoidNonFiniteMetricError)
            else current_completed_steps
        )
        failure_steps = min(max(int(raw_failure_steps), 0), stage_limit)
        failure = _failure_values(
            error,
            phase=current_phase,
            env_steps=failure_steps,
        )
        coordinates = timeline.coordinates(current_stage, failure_steps)
        if benchmark_start > 0:
            coordinates["plot_env_steps"] = float(coordinates["plot_env_steps"]) + 0.5
        failure_event = FailureEvent(
            task=config.task,
            method=config.method,
            seed=config.seed,
            stage=current_stage,
            error_type=str(failure["type"]),
            message=str(failure["message"]),
            **coordinates,
        )
        tracking_failures, tracking_warnings = _record_failed_tracker(
            tracker,
            failure_event,
        )
        if tracking_failures:
            failure["tracking_failures"] = tracking_failures
        partial_results = _finite_partial(
            current_records,
            completed_env_steps=failure_steps,
        )
        completed_stages = [
            stage for stage in stage_summaries if int(stage["stage"]) < current_stage
        ]
        partial_results.update(
            {
                "phase": current_phase,
                "completed_stage_count": len(completed_stages),
                "completed_stage_indices": [
                    int(stage["stage"]) for stage in completed_stages
                ],
            }
        )
        failed_stage = {
            "stage": current_stage,
            "nominal_stage_index": current_stage + (1 if benchmark_start > 0 else 0),
            "nominal_start_env_steps": benchmark_start
            + sum(config.stage_steps[:current_stage]),
            "nominal_end_env_steps": benchmark_start
            + sum(config.stage_steps[: current_stage + 1]),
            "status": "failed",
            "phase": current_phase,
            "precision": precision_schedule["downstream"],
            "requested_env_steps": config.stage_steps[current_stage],
            "planned_actual_env_steps": stage_limit,
            "actual_env_steps": failure_steps,
            "input_decoder": input_decoder_manifest,
            "ppo_execution_profile": phase_plan["execution_profile"],
            "ppo_execution_contract": phase_plan["execution_contract"]["latent"],
            "ppo_worker": current_worker_provenance,
            "decoder_worker": current_decoder_worker_provenance,
            "evaluations": [event.to_dict() for event in current_eval_events],
            "primary_metric": _primary_metric_manifest(config),
            "deterministic_report": None,
            "checkpoints": {},
            "outgoing_collection": None,
            "outgoing_decoder": None,
            "final_return": None,
            "best_return": None,
            "stage_best_return": None,
            "overall_best_return": None,
            "deterministic_final_return": None,
            "failure": failure,
            "partial_results": partial_results,
            "tracking_warnings": tracking_warnings,
        }
        artifacts.write_stage_manifest(current_stage, failed_stage)
        failed_summary = {
            "status": "failed",
            "task": config.task,
            "method": config.method,
            "seed": config.seed,
            "profile": config.profile,
            "fresh_from_scratch": shared_prefix is None,
            "execution_mode": (
                "ordinary_fresh_train"
                if shared_prefix is None
                else "shared_prefix_branch"
            ),
            "precision_schedule": precision_schedule,
            "runtime_schedule": runtime_schedule,
            "teacher": teacher_manifest,
            "stage_steps": list(config.stage_steps),
            "compute_stage_steps": list(compute_stage_steps),
            "benchmark_start_env_steps": benchmark_start,
            "compute_start_env_steps": teacher_actual_steps,
            "benchmark_schedule": benchmark_schedule,
            "encoder_handoff": handoff_selection,
            "stage0_encoder_initialization": stage0_initialization,
            "decoder_stage_types": [kind.value for kind in decoder_kinds],
            "metric_policy": _primary_metric_manifest(config),
            "backbone_point_source": (
                "teacher_dataset_mean" if benchmark_start > 0 else None
            ),
            "stages": completed_stages,
            "failed_stage": failed_stage,
            "final_return": None,
            "stage_best_return": None,
            "overall_best_return": None,
            "deterministic_final_return": None,
            "target_return": config.target_return,
            "performance_target_met": None,
            "failure": failure,
            "partial_results": partial_results,
            "tracking_warnings": tracking_warnings,
            **(
                {}
                if shared_prefix_manifest is None
                else {"shared_prefix": shared_prefix_manifest}
            ),
        }
        atomic_write_json(artifacts.run_dir / "summary.json", failed_summary)
        artifacts.update_manifest(
            {
                "final_return": None,
                "stage_best_return": None,
                "overall_best_return": None,
                "failure": failure,
                "partial_results": partial_results,
                "tracking_warnings": tracking_warnings,
            }
        )
        if isinstance(error, Exception):
            raise HumanoidPipelineError(str(failure["message"])) from error
        raise


def _validate_capabilities(config: RunConfig) -> None:
    if config.profile != "humanoid":
        raise HumanoidPipelineError(
            f"Humanoid pipeline requires profile=humanoid, got {config.profile}"
        )
    try:
        final_kind = _METHOD_DECODER[config.method]
    except KeyError as error:
        raise HumanoidPipelineError(
            f"unsupported Humanoid GoRL method: {config.method}"
        ) from error

    decoder = config.section("decoder")
    declared_type = str(decoder.get("type", ""))
    if declared_type != final_kind.value:
        raise HumanoidPipelineError(
            f"{config.method} requires decoder.type={final_kind.value}, "
            f"got {declared_type!r}"
        )
    collection = config.section("collection")
    if collection.get("action_semantics") != "bounded":
        raise HumanoidPipelineError(
            "Humanoid collection requires action_semantics='bounded'"
        )
    ppo = config.section("ppo")
    if not bool(ppo.get("bounded_latent", False)):
        raise HumanoidPipelineError("Humanoid PPO requires bounded_latent=true")

    previous_kind: DecoderKind | None = None
    for stage in range(len(config.stage_steps)):
        values = _decoder_stage_values(config, stage)
        kind = _decoder_kind(config, stage)
        target = str(values.get("target", ""))
        if target not in {"action_bounded", "bounded"}:
            raise HumanoidPipelineError(
                "Humanoid shared decoder training requires bounded action "
                f"targets at stage {stage}"
            )
        initialization = str(values.get("initialization", "fresh"))
        if initialization not in {"fresh", "cold", "warm"}:
            raise HumanoidPipelineError(
                f"unsupported decoder initialization at stage {stage}: "
                f"{initialization!r}"
            )
        if initialization == "warm":
            if previous_kind is None:
                raise HumanoidPipelineError(
                    "stage0 decoder cannot warm-start without a previous decoder"
                )
            if previous_kind is not kind:
                raise HumanoidPipelineError(
                    "warm decoder type switch is invalid at stage "
                    f"{stage}: {previous_kind.value} -> {kind.value}"
                )
        pairwise = float(values.get("pairwise_loss_weight", 0.0))
        if not math.isfinite(pairwise) or pairwise < 0.0:
            raise HumanoidPipelineError(
                f"invalid decoder pairwise_loss_weight at stage {stage}: {pairwise}"
            )
        if kind is DecoderKind.FM and values.get("convention", "reverse") != "reverse":
            raise HumanoidPipelineError(
                f"Humanoid FM stage {stage} requires reverse convention"
            )
        previous_kind = kind

    if previous_kind is not final_kind:
        raise HumanoidPipelineError(
            f"{config.method} final stage must use {final_kind.value}, got "
            f"{previous_kind.value if previous_kind is not None else 'none'}"
        )


def _validate_shared_teacher_state(
    config: RunConfig,
    state: HumanoidPPOState,
    *,
    requested_steps: int,
    actual_steps: int,
    seed: int,
) -> None:
    expected = {
        "task": config.task,
        "seed": seed,
        "stage_index": 0,
        "decoder_kind": DecoderKind.IDENTITY,
        "requested_environment_steps": requested_steps,
        "actual_environment_steps": actual_steps,
    }
    mismatches = [
        f"{name}={getattr(state, name)!r} (expected {value!r})"
        for name, value in expected.items()
        if getattr(state, name) != value
    ]
    if state.action_size != state.latent_size:
        mismatches.append(
            f"latent_size={state.latent_size!r} "
            f"(expected action_size={state.action_size!r})"
        )
    if mismatches:
        raise HumanoidPipelineError(
            "shared-prefix teacher checkpoint does not match its contract: "
            + "; ".join(mismatches)
        )


def _decoder_stage_values(config: RunConfig, stage: int) -> dict[str, Any]:
    try:
        values = resolve_stage_settings(config.values, "decoder", stage)
    except ConfigError as error:
        raise HumanoidPipelineError(str(error)) from error
    if config.smoke:
        smoke = config.section("smoke")
        if "decoder_train_steps" in smoke:
            values["train_steps"] = int(smoke["decoder_train_steps"])
            values.pop("epochs", None)
        elif "decoder_epochs" in smoke:
            values["epochs"] = int(smoke["decoder_epochs"])
            values.pop("train_steps", None)
        if "decoder_batch_size" in smoke:
            values["batch_size"] = int(smoke["decoder_batch_size"])
    return values


def _decoder_kind(config: RunConfig, stage: int) -> DecoderKind:
    try:
        return DecoderKind(str(_decoder_stage_values(config, stage)["type"]))
    except (KeyError, ValueError) as error:
        raise HumanoidPipelineError(
            f"decoder stage {stage} must select type='fm' or 'diffusion'"
        ) from error


def _teacher_seed_resolution(config: RunConfig) -> SeedResolution:
    return resolve_seed(
        base_seed=config.teacher_seed,
        role="teacher",
        schedule="fixed",
        stage=None,
    )


def _teacher_collection_seed_resolution(config: RunConfig) -> SeedResolution:
    role = str(config.section("collection").get("teacher_seed_role", "teacher"))
    base_seed = config.seed if role == "primary" else config.teacher_seed
    return resolve_seed(
        base_seed=base_seed,
        role=role,
        schedule="fixed",
        stage=None,
    )


def _decoder_seed_resolution(
    config: RunConfig,
    target_stage: int,
) -> SeedResolution:
    values = _decoder_stage_values(config, target_stage)
    role = str(values.get("seed_role", "decoder"))
    role_seed = config.teacher_seed if role == "teacher" else config.decoder_seed
    return resolve_seed(
        base_seed=role_seed,
        role=role,
        schedule=str(values.get("seed_schedule", "stage_offset")),
        stage=target_stage,
    )


def _collection_seed_resolution(
    config: RunConfig,
    source_stage: int,
) -> SeedResolution:
    return resolve_seed(
        base_seed=config.seed,
        role="primary",
        schedule=str(config.section("collection").get("seed_schedule", "stage_offset")),
        stage=source_stage,
    )


def _encoder_seed_resolution(config: RunConfig, stage: int) -> SeedResolution:
    return resolve_seed(
        base_seed=config.encoder_seed,
        role="encoder",
        schedule="fixed",
        stage=stage,
    )


def _evaluation_seed_resolution(config: RunConfig, stage: int) -> SeedResolution:
    return resolve_seed(
        base_seed=config.eval_seed,
        role="evaluation",
        schedule="fixed",
        stage=stage,
    )


def _seeded_artifact(
    artifact: Mapping[str, Any],
    resolution: SeedResolution,
) -> dict[str, Any]:
    return {**artifact, "seed_resolution": resolution.to_dict()}


def _encoder_handoff(config: RunConfig) -> str:
    selection = str(config.section("training").get("encoder_handoff", ""))
    if selection not in {"best", "final"}:
        raise HumanoidPipelineError(
            "Humanoid training.encoder_handoff must be explicitly set to "
            "'best' or 'final'"
        )
    return selection


def _stage0_encoder_initialization(config: RunConfig) -> str:
    initialization = str(
        config.section("training").get("stage0_encoder_initialization", "")
    )
    if initialization not in {"fresh", "teacher_warm_start"}:
        raise HumanoidPipelineError(
            "Humanoid training.stage0_encoder_initialization must be explicitly "
            "set to 'fresh' or 'teacher_warm_start'"
        )
    return initialization


def _select_handoff_state(
    result: HumanoidTrainingResult,
    selection: str,
) -> tuple[HumanoidPPOState, str]:
    if selection == "final":
        return result.state, "final"
    if selection != "best":
        raise ValueError(f"unsupported encoder handoff selection: {selection}")
    if result.best_state is None:
        return result.state, "final_fallback_no_checkpointable_best"
    return result.best_state, "best"


def _stage_ppo_values(config: RunConfig, stage: int) -> dict[str, Any]:
    try:
        return resolve_stage_settings(config.values, "ppo", stage)
    except ConfigError as error:
        raise HumanoidPipelineError(str(error)) from error


def _make_ppo_config(
    config: RunConfig,
    *,
    values: Mapping[str, Any],
    requested_steps: int,
    seed: int,
    stage_index: int,
) -> HumanoidPPOConfig:
    overrides = {
        key: values[key]
        for key in _PPO_FIELDS
        if (
            key in values
            and values[key] is not None
            and not (
                key == "max_grad_norm"
                and isinstance(values[key], str)
                and values[key].lower() == "none"
            )
        )
    }
    if "clip_epsilon" in values:
        overrides["clipping_epsilon"] = values["clip_epsilon"]
    evaluation = config.section("evaluation")
    return HumanoidPPOConfig(
        task=config.task,
        seed=seed,
        requested_environment_steps=requested_steps,
        stage_index=stage_index,
        num_evals=int(values["num_evals"]),
        num_eval_envs=int(values["num_eval_envs"]),
        training_eval_deterministic=bool(evaluation.get("deterministic", False)),
        ppo_overrides=overrides,
        execution_profile=HumanoidPPOExecutionProfile(
            config.section("runtime").get(
                "humanoid_ppo_execution_profile",
                HumanoidPPOExecutionProfile.NATIVE.value,
            )
        ),
    )


def _resolved_actual_steps(
    requested_steps: int,
    values: Mapping[str, Any],
) -> int:
    action_repeat = int(values.get("action_repeat", 1))
    steps_per_training_step = (
        int(values["batch_size"])
        * int(values["unroll_length"])
        * int(values["num_minibatches"])
        * action_repeat
    )
    eval_intervals = max(int(values["num_evals"]) - 1, 1)
    resets = max(int(values.get("num_resets_per_eval", 0)), 1)
    updates_per_interval = math.ceil(
        requested_steps / (eval_intervals * steps_per_training_step * resets)
    )
    return eval_intervals * updates_per_interval * steps_per_training_step * resets


def _teacher_requested_steps(config: RunConfig) -> int:
    if config.smoke:
        return config.stage_steps[0]
    training = config.section("training")
    return int(
        training.get(
            "teacher_requested_env_steps",
            sum(config.stage_steps),
        )
    )


def _validate_declared_teacher_steps(
    config: RunConfig,
    observed_steps: int,
) -> None:
    if config.smoke:
        return
    declared = config.section("training").get("teacher_actual_env_steps")
    if declared is not None and int(declared) != observed_steps:
        raise HumanoidPipelineError(
            "fresh teacher actual steps do not match the configured evidence "
            f"value: {observed_steps} != {declared}"
        )


def _report_config(
    config: RunConfig,
    ppo_values: Mapping[str, Any],
) -> HumanoidEvaluationConfig:
    evaluation = config.section("evaluation")
    episodes = int(evaluation.get("episodes", ppo_values["num_eval_envs"]))
    if config.smoke:
        episodes = min(episodes, int(ppo_values["num_eval_envs"]))
    return HumanoidEvaluationConfig(
        seed=config.eval_seed,
        num_envs=episodes,
        episode_length=int(ppo_values["episode_length"]),
    )


def _collection_num_envs(values: Mapping[str, Any]) -> int:
    return int(values.get("num_envs", min(64, int(values["episodes"]))))


def _fit_configured_decoder(
    config: RunConfig,
    *,
    dataset: Any,
    dataset_path: Path,
    target_stage: int,
    metrics_path: Path,
    worker_dir: Path,
    phase_runtime: Mapping[str, Any],
    previous_fit: Any | None,
) -> tuple[Any, Any, Mapping[str, Any] | None]:
    from gorl.algorithms.decoder_training import fit_decoder

    values = _decoder_stage_values(config, target_stage)
    initialization = str(values.get("initialization", "fresh"))
    warm_start = previous_fit if initialization == "warm" else None
    execution_profile = str(
        config.section("runtime").get(
            "humanoid_ppo_execution_profile",
            HumanoidPPOExecutionProfile.NATIVE.value,
        )
    )
    if execution_profile == HumanoidPPOExecutionProfile.OFFICIAL_DIRECT_COMPAT.value:
        outcome = launch_decoder_worker(
            run_config=config,
            target_stage=target_stage,
            dataset_path=dataset_path,
            warm_start=warm_start,
            phase_runtime=phase_runtime,
            worker_dir=worker_dir,
            metrics_path=metrics_path,
        )
        return outcome.result, outcome.decoder_config, outcome.provenance

    decoder_config = _decoder_training_config(
        config,
        dataset=dataset,
        target_stage=target_stage,
    )
    fitted = fit_decoder(
        dataset,
        decoder_config,
        warm_start=warm_start,
        metric_callback=lambda record: _append_json(metrics_path, record),
    )
    return fitted, decoder_config, None


def _decoder_training_config(
    config: RunConfig,
    *,
    dataset: Any,
    target_stage: int,
) -> Any:
    from gorl.algorithms.decoder_training import (
        DecoderTrainingConfig,
        prepare_decoder_data,
        train_steps_from_epochs,
    )

    values = _decoder_stage_values(config, target_stage)
    method = str(values["type"])
    raw_clip = values.get("target_clip")
    decoder_config = DecoderTrainingConfig(
        method=method,
        seed=_decoder_seed_resolution(config, target_stage).seed,
        train_steps=int(values.get("train_steps", 1)),
        batch_size=int(values["batch_size"]),
        learning_rate=float(values["learning_rate"]),
        validation_fraction=float(values.get("validation_fraction", 0.1)),
        eval_interval=int(values.get("eval_interval", 100)),
        eval_batches=int(values.get("eval_batches", 8)),
        endpoint_eval_size=int(values.get("endpoint_eval_size", 4096)),
        checkpoint_metric=str(
            values.get(
                "checkpoint_metric",
                "val_endpoint_mse" if method == "fm" else "val_loss",
            )
        ),
        action_source="bounded",
        action_clip=(
            float(raw_clip) if raw_clip is not None and float(raw_clip) > 0.0 else None
        ),
        max_transitions=(
            int(values["max_transitions"])
            if values.get("max_transitions") is not None
            else None
        ),
        normalize_observations=bool(values.get("normalize_observations", True)),
        observation_epsilon=float(values.get("observation_epsilon", 1e-8)),
        hidden_dims=tuple(int(width) for width in values["hidden_sizes"]),
        decoder_steps=int(
            values.get(
                "flow_steps",
                values.get("diffusion_steps", 10),
            )
        ),
        timestep_embed_dim=int(values.get("timestep_embedding_dim", 8)),
        output_scale=(
            float(values["output_scale"])
            if values.get("output_scale") is not None
            else None
        ),
        n_samples_per_action=int(values.get("samples_per_action", 8)),
        pairwise_loss_weight=float(values.get("pairwise_loss_weight", 0.0)),
        zero_final_layer=bool(values.get("zero_final_layer", True)),
        diffusion_alpha_min=float(values.get("diffusion_alpha_min", 1e-3)),
        diffusion_alpha_max=float(values.get("diffusion_alpha_max", 1.0)),
        diffusion_parameterization=str(
            values.get("diffusion_parameterization", "epsilon_delta")
        ),
        diffusion_schedule=str(values.get("diffusion_schedule", "linear_alpha_bar")),
        split_unit=str(values.get("split_unit", "episode")),
        batch_sampling=str(values.get("batch_sampling", "with_replacement")),
        rng_schedule=str(values.get("rng_schedule", "independent_streams")),
        validation_schedule=str(values.get("validation_schedule", "interval_sampled")),
        early_stopping_patience=(
            int(values["early_stopping_patience"])
            if values.get("early_stopping_patience") is not None
            else None
        ),
        warm_observation_stats_mode=str(
            values.get("warm_observation_stats_mode", "cumulative")
        ),
    )
    if "epochs" in values:
        prepared = prepare_decoder_data(dataset, decoder_config)
        decoder_config = replace(
            decoder_config,
            train_steps=train_steps_from_epochs(
                int(values["epochs"]),
                prepared,
                batch_size=decoder_config.batch_size,
            ),
        )
    return decoder_config


def _flatten_rollout_batch(
    outputs: tuple[np.ndarray, ...],
    *,
    completed: int,
    transition_order: str,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Flatten one vectorized rollout without changing its transition order."""

    import numpy as np

    if transition_order not in {"episode_major", "batched_time_major"}:
        raise ValueError(
            "transition_order must be 'episode_major' or 'batched_time_major'"
        )
    (
        observations,
        next_observations,
        latent_actions,
        environment_actions,
        rewards,
        valid,
        steps,
    ) = outputs
    returns = (rewards * valid).sum(axis=0)
    lengths = valid.sum(axis=0)
    batch_size = valid.shape[1]

    if transition_order == "episode_major":
        mask = valid.swapaxes(0, 1)

        def flatten(value: np.ndarray) -> np.ndarray:
            return value.swapaxes(0, 1)[mask]

        ids = np.broadcast_to(
            np.arange(completed, completed + batch_size)[:, None],
            mask.shape,
        )
    else:
        mask = valid

        def flatten(value: np.ndarray) -> np.ndarray:
            return value[mask]

        ids = np.broadcast_to(
            np.arange(completed, completed + batch_size)[None, :],
            mask.shape,
        )

    flat_ids = ids[mask]
    arrays = {
        "obs": flatten(observations),
        "next_obs": flatten(next_observations),
        "latent_action": flatten(latent_actions),
        "env_action": flatten(environment_actions),
        "reward": flatten(rewards),
        "episode_id": flat_ids,
        "step": flatten(steps),
        "episode_return": returns[flat_ids - completed],
    }
    return arrays, returns, lengths


def _collect_policy_dataset(
    *,
    state: HumanoidPPOState,
    decoder: Any | None,
    num_episodes: int,
    num_envs: int,
    episode_length: int,
    seed: int,
    stochastic: bool,
    transition_order: str = "episode_major",
) -> Any:
    """Collect complete bounded-action trajectories in the requested order."""

    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from gorl.algorithms.collection import (
        CollectedDataset,
        dataset_stats,
    )

    if num_episodes < 2 or num_envs < 1 or episode_length < 1:
        raise ValueError("collection requires at least two episodes and positive sizes")
    if transition_order not in {"episode_major", "batched_time_major"}:
        raise ValueError(
            "transition_order must be 'episode_major' or 'batched_time_major'"
        )
    runtime = _load_runtime()
    environment = _load_environment(runtime, state.task)
    if decoder is not None:
        if int(decoder.action_size) != int(environment.action_size):
            raise ValueError("decoder and environment action sizes differ")
    normalize = (
        running_statistics.normalize
        if bool(state.ppo_parameters.get("normalize_observations", False))
        else _identity_preprocessor
    )
    factory = _network_factory(runtime, state.ppo_parameters, bounded=True)
    network = factory(
        environment.observation_size,
        state.latent_size,
        preprocess_observations_fn=normalize,
    )
    make_policy = ppo_networks.make_inference_fn(network, compute_value=False)
    policy = make_policy(state.brax_params, deterministic=not stochastic)

    rollout_cache: dict[int, Any] = {}

    def rollout_for(batch_size: int) -> Any:
        cached = rollout_cache.get(batch_size)
        if cached is not None:
            return cached

        @runtime.jax.jit
        def rollout(prng: Any) -> tuple[Any, ...]:
            reset_keys = runtime.jax.random.split(prng, batch_size)
            environment_state = runtime.jax.vmap(environment.reset)(reset_keys)

            def step_fn(carry: tuple[Any, Any, Any], step: Any):
                current_state, step_prng, done = carry
                step_prng, action_prng = runtime.jax.random.split(step_prng)
                latent_action, _ = policy(current_state.obs, action_prng)
                decoded = (
                    latent_action
                    if decoder is None
                    else decoder.decode(current_state.obs, latent_action)
                )
                environment_action = runtime.jnp.clip(decoded, -1.0, 1.0)
                next_state = runtime.jax.vmap(environment.step)(
                    current_state,
                    environment_action,
                )
                valid = ~done
                new_done = done | (next_state.done > 0.5)
                return (next_state, step_prng, new_done), (
                    current_state.obs,
                    next_state.obs,
                    latent_action,
                    environment_action,
                    next_state.reward,
                    valid,
                    runtime.jnp.broadcast_to(step, (batch_size,)),
                )

            initial_done = runtime.jnp.zeros((batch_size,), dtype=bool)
            _, outputs = runtime.jax.lax.scan(
                step_fn,
                (environment_state, prng, initial_done),
                runtime.jnp.arange(episode_length),
            )
            return outputs

        rollout_cache[batch_size] = rollout
        return rollout

    pieces: dict[str, list[Any]] = {
        "obs": [],
        "next_obs": [],
        "latent_action": [],
        "env_action": [],
        "reward": [],
        "episode_id": [],
        "step": [],
        "episode_return": [],
    }
    episode_returns: list[Any] = []
    episode_lengths: list[Any] = []
    completed = 0
    rollout_prng = runtime.jax.random.PRNGKey(seed)
    while completed < num_episodes:
        batch_size = min(num_envs, num_episodes - completed)
        rollout_prng, batch_prng = runtime.jax.random.split(rollout_prng)
        outputs = rollout_for(batch_size)(batch_prng)
        host_outputs = tuple(
            runtime.np.asarray(runtime.jax.device_get(value)) for value in outputs
        )
        batch_arrays, returns, lengths = _flatten_rollout_batch(
            host_outputs,
            completed=completed,
            transition_order=transition_order,
        )
        for name, value in batch_arrays.items():
            pieces[name].append(value)
        episode_returns.append(returns)
        episode_lengths.append(lengths)
        completed += batch_size

    arrays = {
        name: runtime.np.concatenate(values).astype(
            runtime.np.int32 if name in {"episode_id", "step"} else runtime.np.float32,
            copy=False,
        )
        for name, values in pieces.items()
    }
    arrays["episode_returns"] = runtime.np.concatenate(episode_returns).astype(
        runtime.np.float32,
        copy=False,
    )
    arrays["episode_lengths"] = runtime.np.concatenate(episode_lengths).astype(
        runtime.np.int32,
        copy=False,
    )
    for name, value in arrays.items():
        if not runtime.np.isfinite(value).all():
            raise FloatingPointError(f"non-finite collected array: {name}")
    return CollectedDataset(
        arrays=arrays,
        stats=dataset_stats(arrays),
        deterministic_policy=not stochastic,
        collection_seed=seed,
        transition_order=transition_order,
    )


def _identity_preprocessor(observation: Any, _: Any) -> Any:
    return observation


def _save_policy_checkpoint(
    state: HumanoidPPOState,
    path: Path,
) -> dict[str, Any]:
    saved = save_state(state, path)
    return {
        "path": str(saved),
        "format": "gorl_humanoid_ppo_state",
        "sha256": sha256_file(saved),
        "size_bytes": saved.stat().st_size,
    }


def _save_decoder_checkpoint(
    path: Path,
    *,
    config: RunConfig,
    target_stage: int,
    source: str,
    dataset_path: Path,
    decoder_config: Any,
    fitted: Any,
    warm_start_checkpoint: Mapping[str, Any] | None,
    seed_resolution: SeedResolution,
) -> dict[str, Any]:
    saved = save_pickle_checkpoint(
        path,
        {
            "schema_version": 1,
            "task": config.task,
            "method": config.method,
            "profile": config.profile,
            "source": source,
            "target_stage": target_stage,
            "seed_resolution": seed_resolution.to_dict(),
            "dataset_path": str(dataset_path.resolve()),
            "config": asdict(decoder_config),
            "warm_start_checkpoint": (
                None if warm_start_checkpoint is None else dict(warm_start_checkpoint)
            ),
            "decoder": fitted.decoder,
            "selection": {
                "metric": fitted.best_metric,
                "step": fitted.best_step,
                "value": fitted.best_value,
            },
            "observation_stats": asdict(fitted.observation_stats),
        },
    )
    return _seeded_artifact(saved, seed_resolution)


def _dataset_manifest(
    dataset: Any,
    path: Path,
    *,
    seed_resolution: SeedResolution,
) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "seed": dataset.collection_seed,
        "seed_resolution": seed_resolution.to_dict(),
        "deterministic_policy": dataset.deterministic_policy,
        "transition_order": getattr(dataset, "transition_order", "unspecified"),
        "rng_schedule": getattr(dataset, "rng_schedule", "unspecified"),
        "stats": asdict(dataset.stats),
    }


def _teacher_dataset_quality_gate_record(
    config: RunConfig,
    collection: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if collection is None:
        raise HumanoidPipelineError("teacher collection manifest is missing")
    stats = collection.get("stats")
    if not isinstance(stats, Mapping):
        raise HumanoidPipelineError(
            "teacher collection manifest is missing dataset statistics"
        )
    raw_observed = stats.get("return_mean")
    if (
        isinstance(raw_observed, bool)
        or not isinstance(raw_observed, (int, float))
        or not math.isfinite(float(raw_observed))
    ):
        raise HumanoidPipelineError(
            "teacher dataset return_mean must be a finite number"
        )

    configured = config.section("training").get("teacher_dataset_min_return_mean")
    threshold = None if configured is None else float(configured)
    enabled = threshold is not None and not config.smoke
    observed = float(raw_observed)
    passed = observed >= threshold if enabled and threshold is not None else None
    if config.smoke and threshold is not None:
        disabled_reason = "smoke"
    elif threshold is None:
        disabled_reason = "not_configured"
    else:
        disabled_reason = None
    return {
        "metric": "collection.stats.return_mean",
        "enabled": enabled,
        "threshold": threshold,
        "observed": observed,
        "passed": passed,
        "disabled_reason": disabled_reason,
    }


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    return value


def _shared_prefix_artifact(artifact: SharedPrefixArtifact) -> dict[str, Any]:
    return {
        "path": str(artifact.path),
        "sha256": artifact.sha256,
        "size_bytes": artifact.size_bytes,
    }


def _shared_prefix_manifest(bundle: SharedPrefixBundle) -> dict[str, Any]:
    return {
        "shared_prefix_id": bundle.shared_prefix_id,
        "manifest_path": str(bundle.manifest_path),
        "contract_sha256": bundle.contract_sha256,
        "contract": _plain_json(bundle.contract),
        "checkpoint": _shared_prefix_artifact(bundle.checkpoint),
        "dataset": _shared_prefix_artifact(bundle.dataset),
        "dataset_stats": asdict(bundle.dataset_stats),
        "benchmark_anchor": (
            None if bundle.benchmark_anchor is None else asdict(bundle.benchmark_anchor)
        ),
    }


def _load_shared_decoder_dataset(
    path: Path,
    *,
    include_latent: bool,
) -> Any:
    from gorl.algorithms.decoder_training import load_decoder_dataset

    return load_decoder_dataset(
        path,
        action_source="bounded",
        include_latent=include_latent,
    )


def _shared_prefix_dataset_manifest(
    bundle: SharedPrefixBundle,
    dataset: Any,
) -> dict[str, Any]:
    collection = _plain_json(bundle.contract["collection"])
    seed_resolution = dict(collection["seed_resolution"])
    if dataset.num_episodes < 1 or dataset.num_transitions < 1:
        raise HumanoidPipelineError("shared-prefix decoder dataset is empty")
    return {
        **_shared_prefix_artifact(bundle.dataset),
        "seed": int(seed_resolution["seed"]),
        "seed_resolution": seed_resolution,
        "deterministic_policy": not bool(collection["stochastic"]),
        "transition_order": str(collection["transition_order"]),
        "rng_schedule": "sealed_shared_prefix",
        "stats": asdict(bundle.dataset_stats),
        "shared_prefix_id": bundle.shared_prefix_id,
    }


def _decoder_manifest(
    fitted: Any,
    decoder_config: Any,
    checkpoint: Mapping[str, Any],
    *,
    seed_resolution: SeedResolution,
) -> dict[str, Any]:
    return {
        "config": asdict(decoder_config),
        "seed_resolution": seed_resolution.to_dict(),
        "checkpoint": dict(checkpoint),
        "best_step": fitted.best_step,
        "best_metric": fitted.best_metric,
        "best_value": fitted.best_value,
        "best_metrics": dict(fitted.best_metrics),
        "final_metrics": dict(fitted.final_metrics),
        "action_key": fitted.action_key,
        "action_is_bounded": fitted.action_is_bounded,
        "action_clip_fraction": fitted.action_clip_fraction,
        "train_episode_ids": list(fitted.train_episode_ids),
        "validation_episode_ids": list(fitted.validation_episode_ids),
    }


def _stage_stability(
    returns: Sequence[float],
    *,
    previous_final: float | None,
) -> dict[str, float | None]:
    initial = float(returns[0])
    final = float(returns[-1])
    best = max(float(value) for value in returns)
    return {
        "initial_return": initial,
        "final_return": final,
        "best_return": best,
        "stage_internal_drift": best - final,
        "transition_drop_from_previous_final": (
            None if previous_final is None else previous_final - initial
        ),
    }


def _teacher_backbone_event(
    *,
    config: RunConfig,
    benchmark_start: int,
    actual_steps: int,
    return_mean: float,
    return_std: float,
) -> EvalEvent:
    return EvalEvent(
        task=config.task,
        method=config.method,
        seed=config.seed,
        stage=0,
        index=0,
        local_env_steps=0,
        env_steps=actual_steps,
        benchmark_env_steps=float(benchmark_start),
        compute_actual_env_steps=actual_steps,
        plot_env_steps=float(benchmark_start),
        return_mean=return_mean,
        return_std=return_std,
    )


def _report_manifest(result: HumanoidTrainingResult) -> dict[str, Any] | None:
    report = result.report
    if report is None:
        return None
    returns = report.episode_returns
    return {
        "deterministic": True,
        "seed": report.seed,
        "episodes": len(returns),
        "actual_env_steps": report.actual_environment_steps,
        "return_mean": report.return_mean,
        "return_std": report.return_std,
        "return_min": min(returns),
        "return_max": max(returns),
        "event": _metric_event_dict(report.event),
    }


def _deterministic_final_return(stage: Mapping[str, Any]) -> float | None:
    report = stage.get("deterministic_report")
    if not isinstance(report, Mapping):
        return None
    value = report.get("return_mean")
    return None if value is None else float(value)


def _primary_metric_manifest(config: RunConfig) -> dict[str, Any]:
    deterministic = bool(config.section("evaluation").get("deterministic", False))
    return {
        "field": "final_return",
        "source": "last_training_eval",
        "deterministic": deterministic,
        "separate_deterministic_field": "deterministic_final_return",
    }


def _wandb_config(config: RunConfig) -> WandbConfig:
    tracking = config.values.get("tracking", {})
    wandb = tracking.get("wandb", {}) if isinstance(tracking, Mapping) else {}
    if not isinstance(wandb, Mapping):
        raise HumanoidPipelineError("tracking.wandb must be a table")
    return WandbConfig(
        mode=config.wandb_mode,
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=str(wandb.get("name") or f"{config.task}--{config.method}"),
        group=str(wandb.get("group") or f"{config.task}--{config.method}"),
        job_type=str(wandb.get("job_type") or "benchmark"),
        tags=(config.task, config.method, config.profile),
        config=config.to_dict(),
    )


def _metric_event_dict(event: HumanoidMetricEvent) -> dict[str, Any]:
    value = asdict(event)
    value["source"] = event.source.value
    return value


def _append_metric_event(path: Path, event: HumanoidMetricEvent) -> None:
    _append_json(path, _metric_event_dict(event))


def _append_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(value), sort_keys=True, allow_nan=False))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
