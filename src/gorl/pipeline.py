"""Unified execution pipeline for all public GoRL benchmark methods."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import RunArtifacts, atomic_write_json
from .config import ConfigError, RunConfig, resolve_stage_settings
from .seeding import SeedResolution, resolve_seed
from .tracking import FailureEvent, StageTimeline, Tracker, WandbConfig


class PipelineError(RuntimeError):
    pass


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(value), sort_keys=True, allow_nan=False))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _wandb_config(config: RunConfig) -> WandbConfig:
    tracking = config.values.get("tracking", {})
    wandb = tracking.get("wandb", {}) if isinstance(tracking, Mapping) else {}
    if not isinstance(wandb, Mapping):
        raise PipelineError("tracking.wandb must be a table")
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


def _record_failed_tracking(
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


def _stage_ppo_values(config: RunConfig, stage: int) -> dict[str, Any]:
    try:
        return resolve_stage_settings(config.values, "ppo", stage)
    except ConfigError as error:
        raise PipelineError(str(error)) from error


def _encoder_initialization(
    config: RunConfig, stage: int, values: Mapping[str, Any]
) -> str:
    if stage == 0:
        return "fresh"
    configured = values.get("encoder_initialization")
    if configured is not None:
        return str(configured)
    return (
        "fresh"
        if bool(config.section("ppo").get("fresh_encoder_each_stage", True))
        else "warm_start"
    )


def _dm_stage_configs(config: RunConfig) -> tuple[Any, ...]:
    from .algorithms.dm_control_training import (
        EncoderInitialization,
        PPOStageConfig,
    )

    collection = config.section("collection")
    rollout_seed_offset = int(collection.get("seed_offset", 1))
    stages = []
    for stage, requested_steps in enumerate(config.stage_steps):
        values = _stage_ppo_values(config, stage)
        stage_seed = config.teacher_seed if stage == 0 else config.encoder_seed
        stages.append(
            PPOStageConfig(
                task=config.task,
                seed=stage_seed,
                eval_seed=config.eval_seed,
                num_timesteps=requested_steps,
                num_envs=int(values["num_envs"]),
                num_eval_envs=int(values["num_eval_envs"]),
                episode_length=int(values["episode_length"]),
                batch_size=int(values["batch_size"]),
                num_minibatches=int(values["num_minibatches"]),
                unroll_length=int(values["unroll_length"]),
                num_updates_per_batch=int(values["num_updates_per_batch"]),
                num_evals=int(values["num_evals"]),
                action_repeat=int(values.get("action_repeat", 1)),
                learning_rate=float(values["learning_rate"]),
                clipping_epsilon=float(values["clip_epsilon"]),
                entropy_cost=float(values["entropy_cost"]),
                discounting=float(values["discounting"]),
                gae_lambda=float(values.get("gae_lambda", 0.95)),
                reward_scaling=float(values["reward_scaling"]),
                value_loss_coeff=float(values.get("value_loss_coeff", 0.25)),
                normalize_observations=bool(values.get("normalize_observations", True)),
                normalize_advantage=bool(values.get("normalize_advantage", True)),
                z_regularization=float(values["z_regularization"]),
                max_grad_norm=float(values["max_grad_norm"]),
                max_policy_scale=float(values["policy_scale_cap"]),
                tanh_entropy_correction=bool(
                    values.get("tanh_entropy_correction", False)
                ),
                observation_stats_accumulation=str(
                    values.get("observation_stats_accumulation", "cumulative")
                ),
                observation_stats_timing=str(
                    values.get("observation_stats_timing", "rollout_consistent")
                ),
                rollout_seed_offset=rollout_seed_offset,
                encoder_initialization=EncoderInitialization(
                    _encoder_initialization(config, stage, values)
                ),
            )
        )
    return tuple(stages)


def _resolved_stage_steps(stage_configs: Sequence[Any]) -> tuple[int, ...]:
    from .algorithms.dm_control_training import build_ppo_config

    actual: list[int] = []
    for stage in stage_configs:
        ppo = build_ppo_config(stage)
        steps_per_update = ppo.iterations_per_env * ppo.num_envs
        updates = stage.num_timesteps // steps_per_update
        if updates < 1:
            raise PipelineError(
                f"stage budget {stage.num_timesteps} is smaller than one "
                f"PPO update ({steps_per_update} environment steps)"
            )
        actual.append(updates * steps_per_update)
    return tuple(actual)


def _decoder_stage_values(config: RunConfig, target_stage: int) -> dict[str, Any]:
    try:
        return resolve_stage_settings(config.values, "decoder", target_stage)
    except ConfigError as error:
        raise PipelineError(str(error)) from error


def _decoder_seed_settings(config: RunConfig, target_stage: int) -> dict[str, str]:
    values = _decoder_stage_values(config, target_stage)
    return {
        "seed_role": str(values.get("seed_role", "decoder")),
        "seed_schedule": str(values.get("seed_schedule", "stage_offset")),
    }


def _decoder_seed_resolution(
    config: RunConfig,
    source_stage: int,
) -> SeedResolution:
    settings = _decoder_seed_settings(config, source_stage + 1)
    role = settings["seed_role"]
    role_seed = config.teacher_seed if role == "teacher" else config.decoder_seed
    return resolve_seed(
        base_seed=role_seed,
        role=role,
        schedule=settings["seed_schedule"],
        stage=source_stage,
        namespace_offset=12,
        stage_stride=10,
    )


def _decoder_seed(config: RunConfig, source_stage: int) -> int:
    return _decoder_seed_resolution(config, source_stage).seed


def _collection_seed_resolution(
    config: RunConfig,
    source_stage: int,
) -> SeedResolution:
    schedule = str(config.section("collection").get("seed_schedule", "stage_offset"))
    return resolve_seed(
        base_seed=config.seed,
        role="primary",
        schedule=schedule,
        stage=source_stage,
        namespace_offset=992,
        stage_stride=1001,
    )


def _collection_seed(config: RunConfig, source_stage: int) -> int:
    return _collection_seed_resolution(config, source_stage).seed


def _policy_seed_resolution(config: RunConfig, stage: int) -> SeedResolution:
    role = "teacher" if stage == 0 else "encoder"
    base_seed = config.teacher_seed if stage == 0 else config.encoder_seed
    return resolve_seed(
        base_seed=base_seed,
        role=role,
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


def _decoder_training_config(
    config: RunConfig,
    *,
    source_stage: int,
    dataset_path: Path,
) -> Any:
    from .algorithms.decoder_training import (
        DecoderTrainingConfig,
        prepare_decoder_data,
        train_steps_from_epochs,
    )

    values = _decoder_stage_values(config, source_stage + 1)
    method = "fm" if config.method == "gorl_fm" else "diffusion"
    initialization = str(values.get("initialization", "fresh"))
    if initialization not in {"fresh", "cold"}:
        raise PipelineError(
            "dm_control pipeline does not carry a previous fitted decoder for "
            f"decoder.initialization={initialization!r}; refusing to ignore "
            "the requested warm start"
        )
    target = str(values.get("target", "action_pre_tanh"))
    action_source = "bounded" if target in {"action_bounded", "bounded"} else "pre_tanh"
    raw_clip = values.get("target_clip")
    action_clip = (
        float(raw_clip) if raw_clip is not None and float(raw_clip) > 0.0 else None
    )
    train_steps = int(values.get("train_steps", 1))
    decoder_config = DecoderTrainingConfig(
        method=method,
        seed=_decoder_seed_resolution(config, source_stage).seed,
        train_steps=train_steps,
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
        action_source=action_source,
        action_clip=action_clip,
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
        prepared = prepare_decoder_data(dataset_path, decoder_config)
        decoder_config = replace(
            decoder_config,
            train_steps=train_steps_from_epochs(
                int(values["epochs"]),
                prepared,
                batch_size=decoder_config.batch_size,
            ),
        )
    return decoder_config


def _checkpoint_payload(
    *,
    config: RunConfig,
    stage: int,
    selection: str,
    agent: Any,
    evaluation: Any,
    requested_env_steps: int,
    actual_env_steps: int,
) -> dict[str, Any]:
    encoder = agent.encoder
    seed_resolution = _policy_seed_resolution(config, stage)
    return {
        "schema_version": 1,
        "task": config.task,
        "method": config.method,
        "profile": config.profile,
        "stage": stage,
        "selection": selection,
        "requested_env_steps": requested_env_steps,
        "actual_env_steps": actual_env_steps,
        "seed_resolution": seed_resolution.to_dict(),
        "evaluation": asdict(evaluation),
        "encoder": {
            "config": encoder.config,
            "params": encoder.params,
            "observation_stats": encoder.obs_stats,
            "optimizer_state": encoder.opt_state,
            "prng": encoder.prng,
            "steps": encoder.steps,
        },
        "decoder": agent.decoder,
    }


def _stage_stability(
    evaluations: Sequence[Any],
    previous_final: float | None,
) -> dict[str, float | None]:
    final = float(evaluations[-1].return_mean)
    best = max(float(record.return_mean) for record in evaluations)
    initial = float(evaluations[0].return_mean)
    return {
        "initial_return": initial,
        "final_return": final,
        "best_return": best,
        "stage_internal_drift": best - final,
        "transition_drop_from_previous_final": (
            None if previous_final is None else previous_final - initial
        ),
    }


def _run_dm_control_gorl(config: RunConfig, artifacts: RunArtifacts) -> dict[str, Any]:
    from mujoco_playground import registry

    from .algorithms.collection import collect_dataset
    from .algorithms.decoder_training import fit_decoder
    from .algorithms.decoders import IdentityDecoder
    from .algorithms.dm_control_training import (
        EncoderInitialization,
        train_stage,
    )
    from .checkpoints import save_pickle_checkpoint, sha256_file

    if config.method not in {"gorl_fm", "gorl_diffusion"}:
        raise PipelineError(f"not a GoRL method: {config.method}")
    stage_configs = _dm_stage_configs(config)
    compute_steps = _resolved_stage_steps(stage_configs)
    training = config.section("training")
    timeline = StageTimeline(
        config.stage_steps,
        compute_stage_steps=compute_steps,
        benchmark_start_env_steps=int(training.get("benchmark_start_env_steps", 0)),
        compute_start_env_steps=int(training.get("teacher_actual_env_steps", 0)),
    )
    decoder: Any
    previous_output: Any | None = None
    stage_summaries: list[dict[str, Any]] = []
    all_returns: list[float] = []
    previous_final: float | None = None
    tracker = artifacts.start_tracking(
        wandb=_wandb_config(config),
    )
    current_stage = 0
    current_phase = "environment_setup"
    current_completed_steps = 0
    current_events: list[Any] = []
    current_tracked_events: list[Any] = []
    collection_manifest: dict[str, Any] | None = None
    decoder_manifest: dict[str, Any] | None = None
    try:
        env = registry.load(
            config.task,
            config=registry.get_default_config(config.task),
        )
        decoder = IdentityDecoder(action_size=int(env.action_size))
        for stage, stage_config in enumerate(stage_configs):
            current_stage = stage
            current_phase = "stage_setup"
            current_completed_steps = 0
            current_events = []
            current_tracked_events = []
            stage_dir = artifacts.stages_dir / f"stage-{stage}"
            stage_dir.mkdir(parents=True, exist_ok=True)
            collection_manifest = None
            decoder_manifest = None

            if stage > 0:
                assert previous_output is not None
                collection_values = config.section("collection")
                source_stage = stage - 1
                collection_seed = _collection_seed_resolution(config, source_stage)
                current_phase = "stage_collection"
                dataset = collect_dataset(
                    agent=previous_output.best_agent,
                    num_episodes=int(collection_values["episodes"]),
                    num_envs=int(
                        collection_values.get(
                            "num_envs",
                            min(64, int(collection_values["episodes"])),
                        )
                    ),
                    episode_length=stage_config.episode_length,
                    seed=collection_seed.seed,
                    deterministic_policy=not bool(
                        collection_values.get("stochastic", True)
                    ),
                )
                dataset_path = (
                    artifacts.stages_dir
                    / f"stage-{source_stage}"
                    / "decoder_dataset.npz"
                )
                dataset.save(dataset_path)
                collection_manifest = {
                    "path": str(dataset_path.resolve()),
                    "sha256": sha256_file(dataset_path),
                    "size_bytes": dataset_path.stat().st_size,
                    "seed": dataset.collection_seed,
                    "seed_resolution": collection_seed.to_dict(),
                    "num_envs": int(
                        collection_values.get(
                            "num_envs",
                            min(64, int(collection_values["episodes"])),
                        )
                    ),
                    "deterministic_policy": dataset.deterministic_policy,
                    "stats": asdict(dataset.stats),
                }

                current_phase = "stage_decoder_training"
                decoder_seed = _decoder_seed_resolution(config, source_stage)
                decoder_config = _decoder_training_config(
                    config,
                    source_stage=source_stage,
                    dataset_path=dataset_path,
                )
                decoder_metrics_path = stage_dir / "decoder_metrics.jsonl"
                fitted = fit_decoder(
                    dataset_path,
                    decoder_config,
                    metric_callback=lambda record, path=decoder_metrics_path: (
                        _append_jsonl(path, record)
                    ),
                )
                decoder = fitted.decoder
                current_phase = "stage_decoder_checkpoint"
                decoder_checkpoint = save_pickle_checkpoint(
                    stage_dir / "decoder_best.pkl",
                    {
                        "schema_version": 1,
                        "task": config.task,
                        "method": config.method,
                        "source_stage": source_stage,
                        "target_stage": stage,
                        "seed_resolution": decoder_seed.to_dict(),
                        "config": asdict(decoder_config),
                        "decoder": fitted.decoder,
                        "selection": {
                            "metric": fitted.best_metric,
                            "step": fitted.best_step,
                            "value": fitted.best_value,
                        },
                        "observation_stats": asdict(fitted.observation_stats),
                    },
                )
                decoder_manifest = {
                    "config": asdict(decoder_config),
                    "seed_resolution": decoder_seed.to_dict(),
                    "checkpoint": decoder_checkpoint,
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

            def on_eval(record: Any, stage_index: int = stage) -> None:
                nonlocal current_completed_steps
                current_completed_steps = max(
                    current_completed_steps,
                    int(record.env_steps),
                )
                event = timeline.event(
                    task=config.task,
                    method=config.method,
                    seed=config.seed,
                    stage=stage_index,
                    index=record.index,
                    local_env_steps=record.env_steps,
                    return_mean=record.return_mean,
                    return_std=record.return_std,
                )
                current_events.append(record)
                current_tracked_events.append(event)
                tracker.log_eval(event)

            initialization = stage_config.encoder_initialization
            needs_previous = initialization in {
                EncoderInitialization.WARM_START,
                EncoderInitialization.GAUSSIAN_WARM_VALUE,
            }
            current_phase = "latent_training"
            output = train_stage(
                decoder=decoder,
                config=stage_config,
                previous_agent=(
                    previous_output.best_agent
                    if needs_previous and previous_output is not None
                    else None
                ),
                eval_callback=on_eval,
            )
            if output.actual_env_steps != compute_steps[stage]:
                raise PipelineError(
                    "precomputed and observed PPO step counts differ: "
                    f"{compute_steps[stage]} != {output.actual_env_steps}"
                )
            if not current_events:
                raise PipelineError(f"GoRL stage {stage} produced no evaluations")
            if current_events[-1].env_steps != output.actual_env_steps:
                raise PipelineError(
                    "last GoRL callback step does not match the final output: "
                    f"{current_events[-1].env_steps} != {output.actual_env_steps}"
                )
            if len(current_events) != len(output.evaluations):
                raise PipelineError(
                    f"GoRL stage {stage} callback and output counts differ"
                )
            for observed, returned in zip(
                current_events,
                output.evaluations,
                strict=True,
            ):
                if asdict(observed) != asdict(returned):
                    raise PipelineError(
                        f"GoRL stage {stage} callback and output evaluations differ"
                    )

            best_eval = max(output.evaluations, key=lambda record: record.return_mean)
            final_eval = output.evaluations[-1]
            current_phase = "stage_checkpoint"
            final_checkpoint = save_pickle_checkpoint(
                stage_dir / "encoder_final.pkl",
                _checkpoint_payload(
                    config=config,
                    stage=stage,
                    selection="final",
                    agent=output.agent,
                    evaluation=final_eval,
                    requested_env_steps=output.requested_env_steps,
                    actual_env_steps=output.actual_env_steps,
                ),
            )
            best_checkpoint = save_pickle_checkpoint(
                stage_dir / "encoder_best.pkl",
                _checkpoint_payload(
                    config=config,
                    stage=stage,
                    selection="best",
                    agent=output.best_agent,
                    evaluation=best_eval,
                    requested_env_steps=output.requested_env_steps,
                    actual_env_steps=output.actual_env_steps,
                ),
            )
            stability = _stage_stability(
                output.evaluations,
                previous_final,
            )
            stage_summary = {
                "stage": stage,
                "status": "complete",
                "requested_env_steps": output.requested_env_steps,
                "actual_env_steps": output.actual_env_steps,
                "steps_per_update": output.steps_per_update,
                "policy_seed": stage_config.seed,
                "policy_seed_resolution": _policy_seed_resolution(
                    config, stage
                ).to_dict(),
                "rollout_seed": (stage_config.seed + stage_config.rollout_seed_offset),
                "evaluation_seed": stage_config.eval_seed,
                "evaluation_seed_resolution": _evaluation_seed_resolution(
                    config, stage
                ).to_dict(),
                "encoder_initialization": initialization.value,
                "evaluations": [asdict(record) for record in output.evaluations],
                "stability": stability,
                "last_train_metrics": output.last_train_metrics,
                "collection": collection_manifest,
                "decoder": decoder_manifest,
                "checkpoints": {
                    "final": final_checkpoint,
                    "best": best_checkpoint,
                },
            }
            current_phase = "stage_manifest"
            artifacts.write_stage_manifest(stage, stage_summary)
            stage_summaries.append(stage_summary)
            all_returns.extend(
                float(record.return_mean) for record in output.evaluations
            )
            previous_final = output.final_return
            previous_output = output
        if not stage_summaries or not all_returns:
            raise PipelineError("GoRL pipeline produced no evaluations")
        current_phase = "tracking_snapshot"
        tracking_warnings = [
            warning.to_dict() for warning in tracker.warnings.read_all()
        ]
        final_return = float(stage_summaries[-1]["stability"]["final_return"])
        final_stage_best = float(stage_summaries[-1]["stability"]["best_return"])
        summary = {
            "status": "complete",
            "task": config.task,
            "method": config.method,
            "seed": config.seed,
            "profile": config.profile,
            "stage_steps": list(config.stage_steps),
            "compute_stage_steps": list(compute_steps),
            "stages": stage_summaries,
            "final_return": final_return,
            "stage_best_return": final_stage_best,
            "overall_best_return": max(all_returns),
            "target_return": config.target_return,
            "performance_target_met": final_return >= config.target_return,
            "tracking_warnings": tracking_warnings,
        }
        current_phase = "summary_write"
        atomic_write_json(artifacts.run_dir / "summary.json", summary)
        return summary
    except BaseException as error:
        stage_limit = compute_steps[current_stage]
        failure_steps = min(
            max(int(current_completed_steps), 0),
            stage_limit,
        )
        failure = {
            "type": "dm_control_pipeline_exception",
            "source": current_phase,
            "error_type": type(error).__name__,
            "env_steps": failure_steps,
            "message": str(error) or type(error).__name__,
        }
        tracking_failures, tracking_warnings = _record_failed_tracking(
            tracker,
            FailureEvent(
                task=config.task,
                method=config.method,
                seed=config.seed,
                stage=current_stage,
                error_type=str(failure["type"]),
                message=str(failure["message"]),
                **timeline.coordinates(current_stage, failure_steps),
            ),
        )
        if tracking_failures:
            failure["tracking_failures"] = tracking_failures
        finite_evaluations = [asdict(record) for record in current_events]
        best_finite = (
            max(finite_evaluations, key=lambda record: float(record["return_mean"]))
            if finite_evaluations
            else None
        )
        completed_stages = [
            stage for stage in stage_summaries if int(stage["stage"]) < current_stage
        ]
        partial_results = {
            "phase": current_phase,
            "finite_evaluation_count": len(finite_evaluations),
            "completed_env_steps": failure_steps,
            "last_finite_evaluation": (
                finite_evaluations[-1] if finite_evaluations else None
            ),
            "best_finite_evaluation": best_finite,
            "best_finite_return": (
                float(best_finite["return_mean"]) if best_finite is not None else None
            ),
            "completed_stage_count": len(completed_stages),
            "completed_stage_indices": [
                int(stage["stage"]) for stage in completed_stages
            ],
        }
        failed_stage = {
            "stage": current_stage,
            "status": "failed",
            "phase": current_phase,
            "requested_env_steps": config.stage_steps[current_stage],
            "planned_actual_env_steps": stage_limit,
            "actual_env_steps": failure_steps,
            "evaluations": [event.to_dict() for event in current_tracked_events],
            "collection": collection_manifest,
            "decoder": decoder_manifest,
            "checkpoints": {},
            "final_return": None,
            "best_return": None,
            "stage_best_return": None,
            "overall_best_return": None,
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
            "stage_steps": list(config.stage_steps),
            "compute_stage_steps": list(compute_steps),
            "stages": completed_stages,
            "failed_stage": failed_stage,
            "final_return": None,
            "stage_best_return": None,
            "overall_best_return": None,
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
            raise PipelineError(str(failure["message"])) from error
        raise


def _baseline_arguments(config: RunConfig) -> tuple[str, ...]:
    from .trainers.adapters import baseline_worker_arguments
    from .trainers.types import Method

    method = Method(config.method)
    decoder = None if method is Method.PPO else config.section("decoder")
    return baseline_worker_arguments(
        method,
        config.section("ppo"),
        decoder,
    )


def _read_worker_evaluations(
    path: Path,
    *,
    allow_empty: bool = False,
    allow_invalid_tail: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    def parse_line(line: str, line_number: int) -> dict[str, Any]:
        try:
            value = json.loads(line, parse_constant=reject_constant)
        except (json.JSONDecodeError, ValueError) as error:
            raise PipelineError(
                f"invalid baseline event at {path}:{line_number}"
            ) from error
        if not isinstance(value, dict):
            raise PipelineError(
                f"baseline event at {path}:{line_number} is not an object"
            )
        try:
            env_steps = value["env_steps"]
            return_mean = value["reward_mean"]
            return_std = value.get("reward_std", 0.0)
        except KeyError as error:
            raise PipelineError(
                f"baseline event at {path}:{line_number} is missing {error.args[0]}"
            ) from error
        if (
            isinstance(env_steps, bool)
            or not isinstance(env_steps, int)
            or env_steps < 0
        ):
            raise PipelineError(
                f"baseline event at {path}:{line_number} has invalid env_steps"
            )
        for name, metric in (
            ("reward_mean", return_mean),
            ("reward_std", return_std),
        ):
            if (
                isinstance(metric, bool)
                or not isinstance(metric, (int, float))
                or not math.isfinite(metric)
            ):
                raise PipelineError(
                    f"baseline event at {path}:{line_number} has "
                    f"non-finite or non-numeric {name}"
                )
        if return_std < 0:
            raise PipelineError(
                f"baseline event at {path}:{line_number} has negative reward_std"
            )
        return value

    records: list[dict[str, Any]] = []
    artifact_failure: dict[str, Any] | None = None
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                value = parse_line(line, line_number)
            except PipelineError as error:
                if not allow_invalid_tail:
                    raise
                artifact_failure = {
                    "type": "legacy_invalid_evaluation_artifact",
                    "message": (
                        "stopped at an invalid legacy baseline event after "
                        f"{len(records)} finite record(s)"
                    ),
                    "path": str(path),
                    "line_number": line_number,
                    "finite_prefix_count": len(records),
                    "cause": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                }
                break
            records.append(value)
    if not records and not allow_empty:
        raise PipelineError(f"baseline worker wrote no evaluations: {path}")
    return records, artifact_failure


def _run_baseline(config: RunConfig, artifacts: RunArtifacts) -> dict[str, Any]:
    from .actions import ActionSemantics, ActionSpec
    from .trainers.adapters import create_trainer
    from .trainers.types import (
        Method,
        PPOProfile,
        RunStatus,
        TrainingRequest,
    )

    method = Method(config.method)
    profile = PPOProfile(config.ppo_backend) if method is Method.PPO else None
    if method is Method.PPO and profile is PPOProfile.BRAX:
        return _run_humanoid_ppo_baseline(config, artifacts)
    training = config.section("training")
    requested_steps = int(training["total_env_steps"])
    output_dir = artifacts.run_dir / "baseline"
    request = TrainingRequest(
        method=method,
        task=config.task,
        seed=config.seed,
        output_dir=output_dir,
        action_spec=ActionSpec(semantics=ActionSemantics.LEGACY_PRE_TANH),
        ppo_profile=profile,
        environment_steps=requested_steps,
        arguments=_baseline_arguments(config),
    )
    tracker = artifacts.start_tracking(
        wandb=_wandb_config(config),
    )
    trainer = create_trainer(method, profile)
    result = trainer.train(request, check=False, capture_output=False)
    eval_path = output_dir / "eval_metrics.jsonl"
    if result.status is RunStatus.SUCCEEDED and not eval_path.is_file():
        raise PipelineError(
            f"baseline backend did not write canonical evaluations: {eval_path}"
        )
    records, evaluation_artifact_failure = (
        _read_worker_evaluations(
            eval_path,
            allow_empty=result.status is RunStatus.FAILED,
            allow_invalid_tail=result.status is RunStatus.FAILED,
        )
        if eval_path.is_file()
        else ([], None)
    )
    resolved_steps = int(result.metadata.get("resolved_env_steps", requested_steps))
    failure_values: dict[str, Any] | None = None
    worker_partial_values: dict[str, Any] = {}
    completed_steps = resolved_steps
    if result.status is RunStatus.FAILED:
        worker_failure = result.metadata.get("worker_failure")
        if isinstance(worker_failure, Mapping):
            failure_values = dict(worker_failure)
            if evaluation_artifact_failure is not None:
                failure_values["evaluation_artifact_failure"] = (
                    evaluation_artifact_failure
                )
        elif evaluation_artifact_failure is not None:
            failure_values = {
                **evaluation_artifact_failure,
                "worker_returncode": result.returncode,
            }
        else:
            failure_values = {
                "type": "worker_exit",
                "message": f"baseline worker exited with code {result.returncode}",
                "returncode": result.returncode,
            }
        worker_partial = result.metadata.get("partial_results")
        worker_partial_values = (
            dict(worker_partial) if isinstance(worker_partial, Mapping) else {}
        )
        raw_completed_steps = worker_partial_values.get(
            "completed_env_steps",
            failure_values.get(
                "env_steps",
                records[-1]["env_steps"] if records else 0,
            ),
        )
        completed_steps = (
            int(raw_completed_steps)
            if isinstance(raw_completed_steps, (int, float))
            and not isinstance(raw_completed_steps, bool)
            and math.isfinite(raw_completed_steps)
            else 0
        )
        completed_steps = min(max(completed_steps, 0), resolved_steps)
    timeline = StageTimeline(
        (requested_steps,),
        compute_stage_steps=(resolved_steps,),
    )
    for index, record in enumerate(records):
        tracker.log_eval(
            timeline.event(
                task=config.task,
                method=config.method,
                seed=config.seed,
                stage=0,
                index=index,
                local_env_steps=int(record["env_steps"]),
                return_mean=float(record["reward_mean"]),
                return_std=float(record.get("reward_std", 0.0)),
            )
        )
    if result.status is RunStatus.FAILED:
        assert failure_values is not None
        coordinates = timeline.coordinates(0, completed_steps)
        error_type = str(failure_values.get("type") or "worker_exit")
        message = str(
            failure_values.get("message")
            or f"baseline worker exited with code {result.returncode}"
        )
        if (
            evaluation_artifact_failure is not None
            and failure_values.get("type") != evaluation_artifact_failure["type"]
        ):
            tracker.log_failure(
                FailureEvent(
                    task=config.task,
                    method=config.method,
                    seed=config.seed,
                    stage=0,
                    error_type=str(evaluation_artifact_failure["type"]),
                    message=str(evaluation_artifact_failure["message"]),
                    **coordinates,
                )
            )
        tracker.log_failure(
            FailureEvent(
                task=config.task,
                method=config.method,
                seed=config.seed,
                stage=0,
                error_type=error_type,
                message=message,
                **coordinates,
            )
        )
    tracking_warnings = [warning.to_dict() for warning in tracker.warnings.read_all()]
    if result.status is RunStatus.FAILED:
        assert failure_values is not None
        returns = [float(record["reward_mean"]) for record in records]
        partial_results = {
            "finite_evaluation_count": len(records),
            "completed_env_steps": completed_steps,
            "last_finite_evaluation": records[-1] if records else None,
            "best_finite_return": max(returns) if returns else None,
            "worker": worker_partial_values,
            "evaluation_artifact_failure": evaluation_artifact_failure,
        }
        failed_stage = {
            "stage": 0,
            "status": "failed",
            "requested_env_steps": requested_steps,
            "planned_actual_env_steps": resolved_steps,
            "actual_env_steps": completed_steps,
            "evaluations": records,
            "final_return": None,
            "best_return": None,
            "failure": failure_values,
            "partial_results": partial_results,
            "tracking_warnings": tracking_warnings,
        }
        artifacts.write_stage_manifest(0, failed_stage)
        failed_summary = {
            "status": "failed",
            "task": config.task,
            "method": config.method,
            "seed": config.seed,
            "profile": config.profile,
            "requested_env_steps": requested_steps,
            "actual_env_steps": completed_steps,
            "final_return": None,
            "stage_best_return": None,
            "overall_best_return": None,
            "target_return": config.target_return,
            "performance_target_met": None,
            "backend": dict(result.metadata),
            "failure": failure_values,
            "partial_results": partial_results,
            "tracking_warnings": tracking_warnings,
        }
        atomic_write_json(artifacts.run_dir / "summary.json", failed_summary)
        artifacts.update_manifest(
            {
                "failure": failure_values,
                "partial_results": partial_results,
                "tracking_warnings": tracking_warnings,
            }
        )
        raise PipelineError(str(failure_values.get("message") or "baseline failed"))

    returns = [float(record["reward_mean"]) for record in records]
    summary = {
        "status": "complete",
        "task": config.task,
        "method": config.method,
        "seed": config.seed,
        "profile": config.profile,
        "requested_env_steps": requested_steps,
        "actual_env_steps": resolved_steps,
        "final_return": returns[-1],
        "stage_best_return": max(returns),
        "overall_best_return": max(returns),
        "target_return": config.target_return,
        "performance_target_met": returns[-1] >= config.target_return,
        "backend": dict(result.metadata),
        "tracking_warnings": tracking_warnings,
    }
    artifacts.write_stage_manifest(
        0,
        {
            "stage": 0,
            "status": "complete",
            "requested_env_steps": requested_steps,
            "actual_env_steps": resolved_steps,
            "evaluations": records,
            "final_return": returns[-1],
            "best_return": max(returns),
            "tracking_warnings": tracking_warnings,
        },
    )
    atomic_write_json(artifacts.run_dir / "summary.json", summary)
    return summary


def _resolved_humanoid_ppo_steps(
    requested_steps: int,
    ppo: Mapping[str, Any],
) -> int:
    action_repeat = int(ppo.get("action_repeat", 1))
    steps_per_training_step = (
        int(ppo["batch_size"])
        * int(ppo["unroll_length"])
        * int(ppo["num_minibatches"])
        * action_repeat
    )
    eval_intervals = max(int(ppo["num_evals"]) - 1, 1)
    resets = max(int(ppo.get("num_resets_per_eval", 0)), 1)
    updates_per_interval = math.ceil(
        requested_steps / (eval_intervals * steps_per_training_step * resets)
    )
    return eval_intervals * updates_per_interval * steps_per_training_step * resets


def _run_humanoid_ppo_baseline(
    config: RunConfig, artifacts: RunArtifacts
) -> dict[str, Any]:
    from .checkpoints import sha256_file
    from .humanoid._ppo_worker import PPOWorkerError, launch_ppo_worker
    from .humanoid.backend import train_teacher
    from .humanoid.checkpoint import save_state
    from .humanoid.types import (
        EventSource,
        HumanoidEvaluationConfig,
        HumanoidNonFiniteMetricError,
        HumanoidPPOConfig,
        HumanoidPPOExecutionProfile,
    )
    from .runtime import humanoid_gorl_runtime_plan

    ppo = config.section("ppo")
    training = config.section("training")
    requested_steps = int(training["total_env_steps"])
    overrides: dict[str, object] = {}
    for key in (
        "action_repeat",
        "batch_size",
        "bootstrap_on_timeout",
        "clipping_epsilon_value",
        "discounting",
        "entropy_cost",
        "episode_length",
        "gae_lambda",
        "learning_rate",
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
    ):
        if key in ppo and ppo[key] is not None:
            overrides[key] = ppo[key]
    if "clip_epsilon" in ppo:
        overrides["clipping_epsilon"] = ppo["clip_epsilon"]
    max_grad_norm = ppo.get("max_grad_norm")
    if (
        not (isinstance(max_grad_norm, str) and max_grad_norm.lower() == "none")
        and max_grad_norm is not None
    ):
        overrides["max_grad_norm"] = max_grad_norm
    report_envs = int(
        ppo["num_eval_envs"]
        if config.smoke
        else config.section("evaluation").get("episodes", 128)
    )
    ppo_config = HumanoidPPOConfig(
        task=config.task,
        seed=config.seed,
        requested_environment_steps=requested_steps,
        stage_index=0,
        num_evals=int(ppo["num_evals"]),
        num_eval_envs=int(ppo["num_eval_envs"]),
        training_eval_deterministic=False,
        ppo_overrides=overrides,
        execution_profile=HumanoidPPOExecutionProfile(
            config.section("runtime").get(
                "humanoid_ppo_execution_profile",
                HumanoidPPOExecutionProfile.NATIVE.value,
            )
        ),
    )
    execution_plan = humanoid_gorl_runtime_plan(config)
    report_config = HumanoidEvaluationConfig(
        seed=config.eval_seed,
        num_envs=report_envs,
        episode_length=int(ppo["episode_length"]),
    )
    planned_actual_steps = _resolved_humanoid_ppo_steps(requested_steps, ppo)
    timeline = StageTimeline(
        (requested_steps,),
        compute_stage_steps=(planned_actual_steps,),
    )
    tracker = artifacts.start_tracking(
        wandb=_wandb_config(config),
    )
    training_events: list[dict[str, Any]] = []
    training_returns: list[float] = []
    completed_steps = 0
    result = None
    worker_provenance: Mapping[str, Any] | None = None

    def on_metric(event: Any) -> None:
        nonlocal completed_steps
        if event.source is not EventSource.TRAINING_EVAL:
            return
        raw_steps = int(event.actual_environment_steps)
        completed_steps = max(completed_steps, raw_steps)
        if event.return_mean is None:
            raise PipelineError(
                "Humanoid PPO training evaluation is missing "
                f"eval/episode_reward at env_steps={raw_steps}"
            )
        if event.return_std is None:
            raise PipelineError(
                "Humanoid PPO training evaluation is missing "
                f"eval/episode_reward_std at env_steps={raw_steps}"
            )
        record = {
            "actual_env_steps": raw_steps,
            "return_mean": event.return_mean,
            "return_std": event.return_std,
            "metrics": dict(event.metrics),
        }
        training_events.append(record)
        training_returns.append(event.return_mean)
        tracker.log_eval(
            timeline.event(
                task=config.task,
                method=config.method,
                seed=config.seed,
                stage=0,
                index=len(training_returns) - 1,
                local_env_steps=min(raw_steps, planned_actual_steps),
                return_mean=event.return_mean,
                return_std=event.return_std or 0.0,
            )
        )

    metric_policy = {
        "field": "final_return",
        "source": "last_training_eval",
        "deterministic": False,
        "separate_deterministic_field": "deterministic_final_return",
    }
    backend_manifest = {
        "trainer_backend": "brax_official_ppo",
        "action_semantics": "bounded",
        "execution_profile": execution_plan["execution_profile"],
        "execution_contract": execution_plan["execution_contract"],
        "worker": None,
    }
    artifacts.update_manifest({"humanoid_ppo_execution": backend_manifest})
    try:
        if execution_plan["teacher_process_required"]:
            outcome = launch_ppo_worker(
                phase="teacher",
                ppo_config=ppo_config,
                report_config=report_config,
                phase_runtime=execution_plan["teacher"],
                worker_dir=artifacts.run_dir / "worker",
                events_path=artifacts.run_dir / "backend_events.jsonl",
                callback=on_metric,
                input_artifacts={},
            )
            result = outcome.result
            worker_provenance = outcome.provenance
            backend_manifest["worker"] = worker_provenance
            artifacts.update_manifest({"humanoid_ppo_execution": backend_manifest})
        else:
            result = train_teacher(
                ppo_config,
                callback=on_metric,
                report_config=report_config,
            )
        actual_steps = result.state.actual_environment_steps
        if actual_steps != planned_actual_steps:
            raise PipelineError(
                "predicted and observed Humanoid PPO step counts differ: "
                f"{planned_actual_steps} != {actual_steps}"
            )
        if result.report is None:
            raise PipelineError("Humanoid PPO did not produce a deterministic report")
        if not training_returns:
            raise PipelineError(
                "Humanoid PPO produced no stochastic training evaluations"
            )
        if training_events[-1]["actual_env_steps"] != actual_steps:
            raise PipelineError(
                "last Humanoid PPO callback step does not match the final state: "
                f"{training_events[-1]['actual_env_steps']} != {actual_steps}"
            )
        returned_events = [
            event
            for event in result.events
            if event.source is EventSource.TRAINING_EVAL
        ]
        if len(returned_events) != len(result.events) or len(returned_events) != len(
            training_events
        ):
            raise PipelineError(
                "Humanoid PPO callback and returned training event counts differ"
            )
        for returned, observed in zip(returned_events, training_events, strict=True):
            if (
                returned.actual_environment_steps != observed["actual_env_steps"]
                or returned.return_mean != observed["return_mean"]
                or returned.return_std != observed["return_std"]
                or dict(returned.metrics) != observed["metrics"]
            ):
                raise PipelineError(
                    "Humanoid PPO callback and returned training events differ"
                )
        if result.final_return != training_returns[-1]:
            raise PipelineError(
                "Humanoid PPO result final_return does not match its last callback"
            )
        if result.best_return != max(training_returns):
            raise PipelineError(
                "Humanoid PPO result best_return does not match its callback curve"
            )

        stage_dir = artifacts.stages_dir / "stage-0"
        final_path = save_state(result.state, stage_dir / "ppo_final.pkl")
        checkpoints: dict[str, Any] = {
            "final": {
                "path": str(final_path),
                "format": "pickle",
                "sha256": sha256_file(final_path),
                "size_bytes": final_path.stat().st_size,
            }
        }
        if result.best_state is not None:
            best_path = save_state(result.best_state, stage_dir / "ppo_best.pkl")
            checkpoints["best"] = {
                "path": str(best_path),
                "format": "pickle",
                "sha256": sha256_file(best_path),
                "size_bytes": best_path.stat().st_size,
            }
        final_return = training_returns[-1]
        best_return = max(training_returns)
        deterministic_final_return = result.report.return_mean
        stage_manifest = {
            "stage": 0,
            "status": "complete",
            "initialization": result.initialization.value,
            "requested_env_steps": requested_steps,
            "actual_env_steps": actual_steps,
            "training_eval_deterministic": False,
            "canonical_report_deterministic": True,
            "ppo_execution_profile": execution_plan["execution_profile"],
            "ppo_execution_contract": execution_plan["execution_contract"]["teacher"],
            "ppo_worker": worker_provenance,
            "primary_metric": metric_policy,
            "training_events": training_events,
            "deterministic_report": {
                "return_mean": result.report.return_mean,
                "return_std": result.report.return_std,
                "num_episodes": len(result.report.episode_returns),
            },
            "checkpoints": checkpoints,
            "final_return": final_return,
            "best_return": best_return,
            "stage_best_return": best_return,
            "overall_best_return": best_return,
            "deterministic_final_return": deterministic_final_return,
        }
        summary = {
            "status": "complete",
            "task": config.task,
            "method": config.method,
            "seed": config.seed,
            "profile": config.profile,
            "requested_env_steps": requested_steps,
            "actual_env_steps": actual_steps,
            "final_return": final_return,
            "stage_best_return": best_return,
            "overall_best_return": best_return,
            "deterministic_final_return": deterministic_final_return,
            "target_return": config.target_return,
            "performance_target_met": final_return >= config.target_return,
            "backend": backend_manifest,
            "metric_policy": metric_policy,
            "evaluation_contract": {
                "canonical_curve": "stochastic_training_eval_only",
                "primary_final": "last_stochastic_training_eval",
                "deterministic_report": "separate_auxiliary_field",
            },
        }
        tracking_warnings = [
            warning.to_dict() for warning in tracker.warnings.read_all()
        ]
        stage_manifest["tracking_warnings"] = tracking_warnings
        summary["tracking_warnings"] = tracking_warnings
        artifacts.write_stage_manifest(0, stage_manifest)
        atomic_write_json(artifacts.run_dir / "summary.json", summary)
    except BaseException as error:
        if isinstance(error, PPOWorkerError):
            worker = error.failure.get("worker")
            if isinstance(worker, Mapping):
                worker_provenance = dict(worker)
                backend_manifest["worker"] = worker_provenance
                artifacts.update_manifest({"humanoid_ppo_execution": backend_manifest})
            failure = dict(error.failure)
            failure.setdefault("env_steps", completed_steps)
        elif isinstance(error, HumanoidNonFiniteMetricError):
            failure = error.to_dict()
        else:
            message = str(error) or type(error).__name__
            failure = {
                "type": "humanoid_ppo_exception",
                "source": "humanoid_ppo_baseline",
                "error_type": type(error).__name__,
                "env_steps": min(completed_steps, planned_actual_steps),
                "message": message,
            }
        raw_failure_steps = failure.get("env_steps", completed_steps)
        if (
            isinstance(raw_failure_steps, (int, float))
            and not isinstance(raw_failure_steps, bool)
            and math.isfinite(raw_failure_steps)
        ):
            failure_steps = int(raw_failure_steps)
        else:
            failure_steps = completed_steps
        failure_steps = min(max(failure_steps, 0), planned_actual_steps)
        completed_steps = min(
            max(completed_steps, failure_steps),
            planned_actual_steps,
        )
        tracking_failures, tracking_warnings = _record_failed_tracking(
            tracker,
            FailureEvent(
                task=config.task,
                method=config.method,
                seed=config.seed,
                stage=0,
                error_type=str(failure["type"]),
                message=str(failure["message"]),
                **timeline.coordinates(0, failure_steps),
            ),
        )
        if tracking_failures:
            failure["tracking_failures"] = tracking_failures
        last_finite = training_events[-1] if training_events else None
        best_finite = (
            max(training_events, key=lambda event: float(event["return_mean"]))
            if training_events
            else None
        )
        partial_results = {
            "finite_evaluation_count": len(training_events),
            "completed_env_steps": completed_steps,
            "last_finite_evaluation": last_finite,
            "best_finite_evaluation": best_finite,
            "best_finite_return": (
                float(best_finite["return_mean"]) if best_finite is not None else None
            ),
        }
        failed_stage = {
            "stage": 0,
            "status": "failed",
            "initialization": (
                result.initialization.value if result is not None else "from_scratch"
            ),
            "requested_env_steps": requested_steps,
            "planned_actual_env_steps": planned_actual_steps,
            "actual_env_steps": completed_steps,
            "training_eval_deterministic": False,
            "canonical_report_deterministic": True,
            "ppo_execution_profile": execution_plan["execution_profile"],
            "ppo_execution_contract": execution_plan["execution_contract"]["teacher"],
            "ppo_worker": worker_provenance,
            "primary_metric": metric_policy,
            "training_events": training_events,
            "deterministic_report": None,
            "checkpoints": {},
            "final_return": None,
            "best_return": None,
            "stage_best_return": None,
            "overall_best_return": None,
            "deterministic_final_return": None,
            "failure": failure,
            "partial_results": partial_results,
            "tracking_warnings": tracking_warnings,
        }
        artifacts.write_stage_manifest(0, failed_stage)
        failed_summary = {
            "status": "failed",
            "task": config.task,
            "method": config.method,
            "seed": config.seed,
            "profile": config.profile,
            "requested_env_steps": requested_steps,
            "planned_actual_env_steps": planned_actual_steps,
            "actual_env_steps": completed_steps,
            "final_return": None,
            "stage_best_return": None,
            "overall_best_return": None,
            "deterministic_final_return": None,
            "target_return": config.target_return,
            "performance_target_met": None,
            "backend": backend_manifest,
            "metric_policy": metric_policy,
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
            raise PipelineError(str(failure["message"])) from error
        raise
    return summary


def _run_humanoid_gorl(config: RunConfig, artifacts: RunArtifacts) -> dict[str, Any]:
    try:
        from .humanoid.pipeline import run as run_humanoid
    except ModuleNotFoundError as error:
        if error.name in {"gorl.humanoid", "gorl.humanoid.pipeline"}:
            raise PipelineError("the humanoid Brax backend is not installed") from error
        raise
    return dict(run_humanoid(config, artifacts))


def run(config: RunConfig, artifacts: RunArtifacts) -> dict[str, Any]:
    """Run one resolved task × method × seed cell."""

    if config.method in {"ppo", "fpo", "dppo"}:
        return _run_baseline(config, artifacts)
    if config.profile == "dm_control":
        return _run_dm_control_gorl(config, artifacts)
    if config.profile == "humanoid":
        return _run_humanoid_gorl(config, artifacts)
    raise PipelineError(
        f"unsupported method/profile pair: {config.method}/{config.profile}"
    )
