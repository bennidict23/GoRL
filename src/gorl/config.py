from __future__ import annotations

import copy
import math
import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from gorl.humanoid.types import HumanoidPPOExecutionProfile


TASKS = (
    "CheetahRun",
    "FingerSpin",
    "FingerTurnHard",
    "FishSwim",
    "HopperStand",
    "WalkerWalk",
    "HumanoidStand",
    "HumanoidRun",
)

METHODS = ("ppo", "fpo", "dppo", "gorl_fm", "gorl_diffusion")
PROFILES = ("auto", "dm_control", "humanoid")
PPO_BACKENDS = {
    "dm_control": "legacy_latent",
    "humanoid": "brax",
}
JAX_MATMUL_PRECISIONS = frozenset({"default", "high", "highest"})

_TOP_LEVEL_KEYS = {
    "schema_version",
    "task",
    "method",
    "family",
    "profile",
    "target_return",
    "stage_steps",
    "environment",
    "runtime",
    "ppo",
    "collection",
    "decoder",
    "evaluation",
    "training",
    "smoke",
    "upstream_dependencies",
    "profile_overrides",
    "method_overrides",
}

_OVERLAY_RESERVED_KEYS = {
    "environment",
    "schema_version",
    "task",
    "method",
    "profile",
    "profile_overrides",
    "method_overrides",
    "upstream_dependencies",
}

_ENVIRONMENT_KEYS = {
    "action_semantics",
    "backend",
    "episode_length",
    "suite",
}

_BASELINE_PPO_KEYS = {
    "batch_size",
    "clip_epsilon",
    "discounting",
    "entropy_cost",
    "episode_length",
    "gae_lambda",
    "learning_rate",
    "normalize_advantage",
    "normalize_observations",
    "num_envs",
    "num_eval_envs",
    "num_evals",
    "num_minibatches",
    "num_updates_per_batch",
    "reward_scaling",
    "unroll_length",
    "value_loss_coeff",
}
_RUNTIME_KEYS = {
    "humanoid_ppo_execution_profile",
    "jax_default_matmul_precision",
    "xla_gpu_autotune_level",
    "xla_gpu_triton_gemm_any",
}
_HUMANOID_OFFICIAL_PPO_KEYS = {
    "action_repeat",
    "batch_size",
    "bootstrap_on_timeout",
    "clip_epsilon",
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
    "num_eval_envs",
    "num_evals",
    "num_minibatches",
    "num_resets_per_eval",
    "num_updates_per_batch",
    "reward_scaling",
    "unroll_length",
    "use_pmap_on_reset",
    "vf_loss_coefficient",
}
_GORL_PPO_COMMON_KEYS = {
    "action_repeat",
    "batch_size",
    "discounting",
    "entropy_cost",
    "episode_length",
    "gae_lambda",
    "learning_rate",
    "normalize_advantage",
    "normalize_observations",
    "num_envs",
    "num_eval_envs",
    "num_evals",
    "num_minibatches",
    "num_updates_per_batch",
    "reward_scaling",
    "unroll_length",
}
_DM_CONTROL_PPO_KEYS = _GORL_PPO_COMMON_KEYS | {
    "clip_epsilon",
    "encoder_initialization",
    "fresh_encoder_each_stage",
    "max_grad_norm",
    "observation_stats_accumulation",
    "observation_stats_timing",
    "policy_scale_cap",
    "tanh_entropy_correction",
    "value_loss_coeff",
    "z_regularization",
}
_HUMANOID_GORL_PPO_KEYS = _HUMANOID_OFFICIAL_PPO_KEYS | {
    "bounded_latent",
    "latent_dim",
}
_GORL_DECODER_KEYS = {
    "batch_sampling",
    "batch_size",
    "checkpoint_metric",
    "convention",
    "diffusion_alpha_max",
    "diffusion_alpha_min",
    "diffusion_parameterization",
    "diffusion_schedule",
    "diffusion_steps",
    "early_stopping_patience",
    "endpoint_eval_size",
    "epochs",
    "eval_batches",
    "eval_interval",
    "flow_steps",
    "hidden_sizes",
    "initialization",
    "learning_rate",
    "max_transitions",
    "normalize_observations",
    "observation_epsilon",
    "output_scale",
    "pairwise_loss_weight",
    "rng_schedule",
    "samples_per_action",
    "seed_role",
    "seed_schedule",
    "split_unit",
    "target",
    "target_clip",
    "timestep_embedding_dim",
    "train_steps",
    "type",
    "validation_fraction",
    "validation_schedule",
    "warm_observation_stats_mode",
    "zero_final_layer",
}
_BASELINE_DECODER_KEYS = {
    "diffusion_steps",
    "flow_steps",
    "output_mode",
    "policy_output_scale",
    "samples_per_action",
    "sde_sigma",
    "timestep_embedding_dim",
    "type",
}
_SMOKE_PPO_KEYS = {
    "batch_size",
    "episode_length",
    "num_envs",
    "num_eval_envs",
    "num_evals",
    "num_minibatches",
    "num_updates_per_batch",
    "unroll_length",
}
_BASELINE_SMOKE_KEYS = _SMOKE_PPO_KEYS | {"env_steps"}
_GORL_SMOKE_KEYS = _SMOKE_PPO_KEYS | {
    "collection_episodes",
    "decoder_batch_size",
    "decoder_epochs",
    "decoder_train_steps",
    "stage_steps",
}


class ConfigError(ValueError):
    pass


def _copy(value: Any) -> Any:
    return copy.deepcopy(value)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = _copy(dict(base))
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = _copy(value)
    return merged


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            value = tomllib.load(stream)
    except FileNotFoundError as error:
        raise ConfigError(f"configuration file does not exist: {path}") from error
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"invalid TOML in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ConfigError(f"configuration root must be a table: {path}")
    return value


def _validate_top_level(
    value: Mapping[str, Any], path: Path, *, overlay: bool = False
) -> None:
    unknown = sorted(set(value).difference(_TOP_LEVEL_KEYS))
    if unknown:
        names = ", ".join(unknown)
        raise ConfigError(f"unknown configuration key(s) in {path}: {names}")
    if overlay:
        reserved = sorted(set(value).intersection(_OVERLAY_RESERVED_KEYS))
        if reserved:
            names = ", ".join(reserved)
            raise ConfigError(
                f"{path} may not override identity/provenance key(s): {names}"
            )


def default_config_root() -> Path:
    configured = os.environ.get("GORL_CONFIG_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()

    source_root = Path(__file__).resolve().parents[2] / "configs"
    if (source_root / "tasks").is_dir():
        return source_root

    installed_root = Path(sys.prefix) / "share" / "gorl" / "configs"
    if (installed_root / "tasks").is_dir():
        return installed_root

    raise ConfigError(
        "default configs were not found; pass --config-root or set GORL_CONFIG_ROOT"
    )


def _positive_steps(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError("stage_steps must be a non-empty TOML array")
    if any(
        isinstance(step, bool) or not isinstance(step, int) or step <= 0
        for step in value
    ):
        raise ConfigError("stage_steps must contain positive integers")
    return tuple(value)


def _benchmark_stage_steps(
    resolved: Mapping[str, Any],
    *,
    method: str,
) -> tuple[int, ...]:
    task_steps = _positive_steps(resolved.get("stage_steps"))
    if method not in {"ppo", "fpo", "dppo"}:
        return task_steps

    training = resolved.get("training")
    if not isinstance(training, Mapping):
        raise ConfigError("training must be a table")
    total = training.get("total_env_steps")
    if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
        raise ConfigError("training.total_env_steps must be a positive integer")
    return (total,)


def _with_derived_environment(resolved: Mapping[str, Any]) -> dict[str, Any]:
    environment = resolved.get("environment")
    if not isinstance(environment, Mapping):
        raise ConfigError("environment must be a table")
    ppo = resolved.get("ppo")
    if not isinstance(ppo, Mapping):
        raise ConfigError("ppo must be a table")
    episode_length = ppo.get("episode_length")
    if (
        isinstance(episode_length, bool)
        or not isinstance(episode_length, int)
        or episode_length <= 0
    ):
        raise ConfigError("ppo.episode_length must be a positive integer")
    return _deep_merge(
        resolved,
        {"environment": {"episode_length": episode_length}},
    )


def _seed(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"{name} must be a non-negative integer")
    return value


def _benchmark_start_env_steps(resolved: Mapping[str, Any]) -> int:
    training = resolved.get("training", {})
    if not isinstance(training, Mapping):
        raise ConfigError("training must be a table")
    value = training.get("benchmark_start_env_steps", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(
            "training.benchmark_start_env_steps must be a non-negative integer"
        )
    return value


def _apply_smoke_overrides(
    resolved: Mapping[str, Any],
) -> dict[str, Any]:
    """Map the compact smoke table onto the normal training schema."""

    value = _copy(dict(resolved))
    smoke = value.get("smoke", {})
    if not isinstance(smoke, Mapping):
        raise ConfigError("smoke must be a table")

    stage_steps = smoke.get("stage_steps")
    selected_smoke_steps: tuple[int, ...] | None = None
    if stage_steps is not None:
        configured_steps = _positive_steps(value.get("stage_steps"))
        smoke_steps = _positive_steps(stage_steps)
        if len(smoke_steps) < len(configured_steps):
            raise ConfigError(
                "smoke.stage_steps must contain at least as many stages as "
                "the resolved schedule"
            )
        selected_smoke_steps = smoke_steps[: len(configured_steps)]
        value["stage_steps"] = list(selected_smoke_steps)

    ppo_keys = {
        "episode_length",
        "num_envs",
        "num_eval_envs",
        "batch_size",
        "num_minibatches",
        "unroll_length",
        "num_updates_per_batch",
        "num_evals",
    }
    ppo_override = {key: smoke[key] for key in ppo_keys if key in smoke}
    if ppo_override:
        ppo = value.get("ppo", {})
        if not isinstance(ppo, Mapping):
            raise ConfigError("ppo must be a table")
        value["ppo"] = _deep_merge(ppo, ppo_override)
        evaluation = value.get("evaluation")
        if "num_eval_envs" in ppo_override and isinstance(evaluation, Mapping):
            value["evaluation"] = _deep_merge(
                evaluation,
                {"episodes": ppo_override["num_eval_envs"]},
            )
    collection = value.get("collection")
    if "collection_episodes" in smoke and isinstance(collection, Mapping):
        value["collection"] = _deep_merge(
            collection, {"episodes": smoke["collection_episodes"]}
        )

    decoder = value.get("decoder")
    if isinstance(decoder, Mapping):
        decoder_override: dict[str, Any] = {}
        if "decoder_train_steps" in smoke:
            decoder_override["train_steps"] = smoke["decoder_train_steps"]
            decoder_override.pop("epochs", None)
        if "decoder_epochs" in smoke:
            decoder_override["epochs"] = smoke["decoder_epochs"]
        if "decoder_batch_size" in smoke:
            decoder_override["batch_size"] = smoke["decoder_batch_size"]
        if decoder_override:
            next_decoder = _deep_merge(decoder, decoder_override)
            if "decoder_train_steps" in smoke:
                next_decoder.pop("epochs", None)
            if "decoder_epochs" in smoke:
                next_decoder.pop("train_steps", None)
            value["decoder"] = next_decoder

    if "env_steps" in smoke:
        training = value.get("training", {})
        if not isinstance(training, Mapping):
            raise ConfigError("training must be a table")
        value["training"] = _deep_merge(
            training, {"total_env_steps": smoke["env_steps"]}
        )

    training = value.get("training")
    if (
        selected_smoke_steps is not None
        and isinstance(training, Mapping)
        and "teacher_requested_env_steps" in training
    ):
        next_training = _deep_merge(
            training,
            {"teacher_requested_env_steps": selected_smoke_steps[0]},
        )
        next_training.pop("teacher_actual_env_steps", None)
        value["training"] = next_training

    return value


def _section_mapping(
    resolved: Mapping[str, Any],
    name: str,
) -> Mapping[str, Any]:
    value = resolved.get(name, {})
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a table")
    return value


def _unknown_keys(
    values: Mapping[str, Any],
    allowed: set[str],
    *,
    path: str,
) -> None:
    unknown = sorted(set(values).difference(allowed))
    if unknown:
        rendered = ", ".join(f"{path}.{key}" for key in unknown)
        raise ConfigError(f"unknown or unsupported configuration key(s): {rendered}")


def _validate_flat_keys(
    values: Mapping[str, Any],
    allowed: set[str],
    *,
    path: str,
) -> None:
    _unknown_keys(values, allowed, path=path)
    nested = sorted(key for key, value in values.items() if isinstance(value, Mapping))
    if nested:
        rendered = ", ".join(f"{path}.{key}" for key in nested)
        raise ConfigError(
            f"configuration value(s) must not be nested tables: {rendered}"
        )


def _validate_staged_section(
    resolved: Mapping[str, Any],
    name: str,
    *,
    allowed_keys: set[str],
    stage_count: int,
    allow_stages: bool,
) -> None:
    values = _section_mapping(resolved, name)
    stage_tables = {
        "stage0",
        "later_stages",
        *(f"stage{stage}" for stage in range(stage_count)),
    }
    for key, value in values.items():
        if isinstance(value, Mapping):
            if not allow_stages or key not in stage_tables:
                raise ConfigError(
                    f"unknown or unsupported configuration table: {name}.{key}"
                )
            _validate_flat_keys(value, allowed_keys, path=f"{name}.{key}")
        elif key not in allowed_keys:
            raise ConfigError(f"unknown or unsupported configuration key: {name}.{key}")


def resolve_stage_settings(
    resolved: Mapping[str, Any],
    section_name: str,
    stage: int,
) -> dict[str, Any]:
    """Resolve base, stage-family, then exact-stage settings without mutation."""

    section = _section_mapping(resolved, section_name)
    values = {
        key: _copy(value)
        for key, value in section.items()
        if not isinstance(value, Mapping)
    }
    selected = section.get("stage0" if stage == 0 else "later_stages", {})
    exact = section.get(f"stage{stage}", {})
    if not isinstance(selected, Mapping) or not isinstance(exact, Mapping):
        raise ConfigError(f"{section_name} stage settings must be tables")
    values.update(_copy(dict(selected)))
    values.update(_copy(dict(exact)))
    return values


def _validate_gorl_decoder(
    resolved: Mapping[str, Any],
    *,
    method: str,
    profile: str,
    stage_count: int,
) -> None:
    decoder = _section_mapping(resolved, "decoder")
    expected_base = "fm" if method == "gorl_fm" else "diffusion"
    if decoder.get("type") != expected_base:
        raise ConfigError(
            f"{method} requires decoder.type={expected_base!r}, "
            f"got {decoder.get('type')!r}"
        )

    previous_kind: str | None = None
    for stage in range(stage_count):
        values = resolve_stage_settings(resolved, "decoder", stage)
        kind = values.get("type")
        allowed_kinds = {expected_base}
        if method == "gorl_diffusion" and profile == "humanoid" and stage == 0:
            allowed_kinds.add("fm")
        if kind not in allowed_kinds:
            expected = ", ".join(sorted(allowed_kinds))
            raise ConfigError(
                f"unsupported decoder type at stage {stage}: {kind!r}; "
                f"expected one of {expected}"
            )

        initialization = str(values.get("initialization", "fresh"))
        if profile == "dm_control" and initialization not in {"fresh", "cold"}:
            raise ConfigError(
                "dm_control decoder stages support only fresh/cold "
                f"initialization, got {initialization!r} at stage {stage}"
            )
        if profile == "humanoid":
            if initialization not in {"fresh", "cold", "warm"}:
                raise ConfigError(
                    f"unsupported Humanoid decoder initialization at stage "
                    f"{stage}: {initialization!r}"
                )
            if initialization == "warm" and (
                previous_kind is None or previous_kind != kind
            ):
                raise ConfigError(
                    "Humanoid decoder warm-start requires a previous decoder "
                    f"of the same type at stage {stage}"
                )

        expected_target = (
            {"action_bounded", "bounded"}
            if profile == "humanoid"
            else {"action_pre_tanh", "pre_tanh"}
        )
        if values.get("target") not in expected_target:
            raise ConfigError(
                f"decoder target at stage {stage} is incompatible with "
                f"profile {profile}: {values.get('target')!r}"
            )
        if kind == "fm" and values.get("convention", "reverse") != "reverse":
            raise ConfigError(f"FM decoder stage {stage} requires convention='reverse'")
        seed_role = values.get("seed_role", "decoder")
        if seed_role not in {"decoder", "teacher"}:
            raise ConfigError(
                f"unsupported decoder seed_role at stage {stage}: {seed_role!r}"
            )
        seed_schedule = values.get("seed_schedule", "stage_offset")
        if seed_schedule not in {"stage_offset", "fixed"}:
            raise ConfigError(
                f"unsupported decoder seed_schedule at stage {stage}: {seed_schedule!r}"
            )
        split_unit = values.get("split_unit", "episode")
        if split_unit not in {"episode", "transition"}:
            raise ConfigError(
                f"unsupported decoder split_unit at stage {stage}: {split_unit!r}"
            )
        batch_sampling = values.get("batch_sampling", "with_replacement")
        if batch_sampling not in {"with_replacement", "epoch_permutation"}:
            raise ConfigError(
                "unsupported decoder batch_sampling at stage "
                f"{stage}: {batch_sampling!r}"
            )
        rng_schedule = values.get("rng_schedule", "independent_streams")
        if rng_schedule not in {
            "independent_streams",
            "legacy_interleaved",
            "state_threaded",
        }:
            raise ConfigError(
                f"unsupported decoder rng_schedule at stage {stage}: {rng_schedule!r}"
            )
        validation_schedule = values.get(
            "validation_schedule",
            "interval_sampled",
        )
        if validation_schedule not in {
            "interval_sampled",
            "epoch_full_batches",
        }:
            raise ConfigError(
                "unsupported decoder validation_schedule at stage "
                f"{stage}: {validation_schedule!r}"
            )
        warm_stats_mode = values.get(
            "warm_observation_stats_mode",
            "cumulative",
        )
        if warm_stats_mode not in {
            "cumulative",
            "frozen",
            "legacy_new_batch",
        }:
            raise ConfigError(
                "unsupported decoder warm_observation_stats_mode at stage "
                f"{stage}: {warm_stats_mode!r}"
            )
        if (
            warm_stats_mode in {"frozen", "legacy_new_batch"}
            and initialization != "warm"
        ):
            raise ConfigError(
                f"decoder warm_observation_stats_mode={warm_stats_mode!r} requires "
                f"warm initialization at stage {stage}"
            )
        patience = values.get("early_stopping_patience")
        if patience is not None and (
            isinstance(patience, bool) or not isinstance(patience, int) or patience < 1
        ):
            raise ConfigError(
                "decoder early_stopping_patience must be a positive integer "
                f"at stage {stage}"
            )
        state_threaded = rng_schedule == "state_threaded"
        epoch_validation = validation_schedule == "epoch_full_batches"
        if state_threaded != epoch_validation:
            raise ConfigError(
                "decoder state_threaded RNG and epoch_full_batches validation "
                f"schedules must be enabled together at stage {stage}"
            )
        if state_threaded:
            if profile != "humanoid":
                raise ConfigError(
                    "state-threaded decoder schedules are supported only by "
                    f"Humanoid stages; got {profile}/{kind} at stage {stage}"
                )
            if split_unit != "transition" or batch_sampling != "epoch_permutation":
                raise ConfigError(
                    "state-threaded decoder schedules require transition split "
                    f"and epoch permutation at stage {stage}"
                )
            if values.get("checkpoint_metric", "val_loss") != "val_loss":
                raise ConfigError(
                    "state-threaded decoder schedules require val_loss "
                    f"checkpoint selection at stage {stage}"
                )
            if patience is None:
                raise ConfigError(
                    "state-threaded decoder schedules require "
                    f"early_stopping_patience at stage {stage}"
                )
        elif rng_schedule == "legacy_interleaved":
            if split_unit != "episode":
                raise ConfigError(
                    "legacy-interleaved decoder RNG requires episode split at "
                    f"stage {stage}"
                )
            if batch_sampling != "with_replacement":
                raise ConfigError(
                    "legacy-interleaved decoder RNG requires with-replacement "
                    f"batches at stage {stage}"
                )
            if validation_schedule != "interval_sampled":
                raise ConfigError(
                    "legacy-interleaved decoder RNG requires interval-sampled "
                    f"validation at stage {stage}"
                )
        elif patience is not None:
            raise ConfigError(
                "decoder early_stopping_patience requires epoch_full_batches "
                f"validation at stage {stage}"
            )
        if kind == "diffusion":
            parameterization = values.get(
                "diffusion_parameterization",
                "epsilon_delta",
            )
            if parameterization not in {"epsilon_delta", "epsilon"}:
                raise ConfigError(
                    "unsupported diffusion_parameterization at stage "
                    f"{stage}: {parameterization!r}"
                )
            schedule = values.get("diffusion_schedule", "linear_alpha_bar")
            if schedule not in {"linear_alpha_bar", "cosine_alpha_bar"}:
                raise ConfigError(
                    f"unsupported diffusion_schedule at stage {stage}: {schedule!r}"
                )
        previous_kind = str(kind)


def _validate_resolved_sections(
    resolved: Mapping[str, Any],
    *,
    method: str,
    profile: str,
    stage_steps: tuple[int, ...],
) -> None:
    is_gorl = method in {"gorl_fm", "gorl_diffusion"}
    is_humanoid_ppo = method == "ppo" and profile == "humanoid"
    if is_gorl:
        ppo_keys = (
            _DM_CONTROL_PPO_KEYS if profile == "dm_control" else _HUMANOID_GORL_PPO_KEYS
        )
    elif is_humanoid_ppo:
        ppo_keys = _HUMANOID_OFFICIAL_PPO_KEYS
    else:
        ppo_keys = _BASELINE_PPO_KEYS
    _validate_staged_section(
        resolved,
        "ppo",
        allowed_keys=ppo_keys,
        stage_count=len(stage_steps),
        allow_stages=is_gorl,
    )
    environment = _section_mapping(resolved, "environment")
    _validate_flat_keys(environment, _ENVIRONMENT_KEYS, path="environment")
    expected_action_semantics = (
        "bounded"
        if profile == "humanoid" and method not in {"fpo", "dppo"}
        else "legacy_pre_tanh"
    )
    expected_environment = {
        "suite": "dm_control",
        "backend": "mujoco_playground",
        "action_semantics": expected_action_semantics,
    }
    for key, expected in expected_environment.items():
        if environment.get(key) != expected:
            raise ConfigError(
                f"{method}/{profile} requires environment.{key}={expected!r}; "
                f"got {environment.get(key)!r}"
            )
    episode_length = environment.get("episode_length")
    if (
        isinstance(episode_length, bool)
        or not isinstance(episode_length, int)
        or episode_length <= 0
    ):
        raise ConfigError("environment.episode_length must be a positive integer")
    for stage in range(len(stage_steps)):
        stage_ppo = resolve_stage_settings(resolved, "ppo", stage)
        if stage_ppo.get("episode_length") != episode_length:
            raise ConfigError(
                "environment.episode_length must equal ppo.episode_length at "
                f"stage {stage}; got {episode_length!r} and "
                f"{stage_ppo.get('episode_length')!r}"
            )
    if is_gorl and profile == "dm_control":
        for stage in range(len(stage_steps)):
            ppo_values = resolve_stage_settings(resolved, "ppo", stage)
            accumulation = ppo_values.get(
                "observation_stats_accumulation",
                "cumulative",
            )
            if accumulation not in {"cumulative", "legacy_batch_only"}:
                raise ConfigError(
                    "unsupported ppo observation_stats_accumulation at stage "
                    f"{stage}: {accumulation!r}"
                )
            timing = ppo_values.get(
                "observation_stats_timing",
                "rollout_consistent",
            )
            if timing not in {"rollout_consistent", "legacy_pre_update"}:
                raise ConfigError(
                    "unsupported ppo observation_stats_timing at stage "
                    f"{stage}: {timing!r}"
                )

    if is_gorl:
        decoder_keys = _GORL_DECODER_KEYS
    elif method in {"fpo", "dppo"}:
        decoder_keys = _BASELINE_DECODER_KEYS
    else:
        decoder_keys = set()
    _validate_staged_section(
        resolved,
        "decoder",
        allowed_keys=decoder_keys,
        stage_count=len(stage_steps),
        allow_stages=is_gorl,
    )

    collection_keys: set[str] = set()
    if is_gorl:
        collection_keys = {
            "action_semantics",
            "episodes",
            "num_envs",
            "seed_schedule",
            "stochastic",
        }
        if profile == "dm_control":
            collection_keys.add("seed_offset")
        else:
            collection_keys.update(
                {
                    "teacher_episodes",
                    "teacher_seed_role",
                    "teacher_transition_order",
                    "transition_order",
                }
            )
    collection = _section_mapping(resolved, "collection")
    _validate_flat_keys(collection, collection_keys, path="collection")

    if not is_gorl:
        training_keys = {"total_env_steps"}
    elif profile == "humanoid":
        training_keys = {
            "benchmark_start_env_steps",
            "downstream_jax_default_matmul_precision",
            "downstream_xla_gpu_autotune_level",
            "downstream_xla_gpu_triton_gemm_any",
            "encoder_handoff",
            "stage0_encoder_initialization",
            "teacher_actual_env_steps",
            "teacher_dataset_min_return_mean",
            "teacher_deterministic_report",
            "teacher_requested_env_steps",
        }
    else:
        training_keys = set()
    training = _section_mapping(resolved, "training")
    _validate_flat_keys(training, training_keys, path="training")
    _benchmark_start_env_steps(resolved)
    downstream_precision = training.get("downstream_jax_default_matmul_precision")
    if (
        downstream_precision is not None
        and downstream_precision not in JAX_MATMUL_PRECISIONS
    ):
        choices = ", ".join(sorted(JAX_MATMUL_PRECISIONS))
        raise ConfigError(
            "training.downstream_jax_default_matmul_precision must be one of "
            f"{choices}; got {downstream_precision!r}"
        )
    downstream_triton_gemm = training.get("downstream_xla_gpu_triton_gemm_any")
    if downstream_triton_gemm is not None and not (
        isinstance(downstream_triton_gemm, bool) or downstream_triton_gemm == "unset"
    ):
        raise ConfigError(
            "training.downstream_xla_gpu_triton_gemm_any must be boolean or 'unset'"
        )
    downstream_autotune = training.get("downstream_xla_gpu_autotune_level")
    if downstream_autotune is not None and not (
        downstream_autotune == "unset"
        or (
            isinstance(downstream_autotune, int)
            and not isinstance(downstream_autotune, bool)
            and downstream_autotune >= 0
        )
    ):
        raise ConfigError(
            "training.downstream_xla_gpu_autotune_level must be a non-negative "
            "integer or 'unset'"
        )
    teacher_deterministic_report = training.get("teacher_deterministic_report")
    if teacher_deterministic_report is not None and not isinstance(
        teacher_deterministic_report, bool
    ):
        raise ConfigError("training.teacher_deterministic_report must be boolean")
    teacher_dataset_min_return_mean = training.get("teacher_dataset_min_return_mean")
    if teacher_dataset_min_return_mean is not None and (
        isinstance(teacher_dataset_min_return_mean, bool)
        or not isinstance(teacher_dataset_min_return_mean, (int, float))
        or not math.isfinite(float(teacher_dataset_min_return_mean))
    ):
        raise ConfigError(
            "training.teacher_dataset_min_return_mean must be a finite number"
        )

    runtime = _section_mapping(resolved, "runtime")
    _validate_flat_keys(runtime, _RUNTIME_KEYS, path="runtime")
    execution_profile = runtime.get("humanoid_ppo_execution_profile")
    consumes_humanoid_brax = profile == "humanoid" and method in {
        "ppo",
        "gorl_fm",
        "gorl_diffusion",
    }
    if execution_profile is not None and not consumes_humanoid_brax:
        raise ConfigError(
            "runtime.humanoid_ppo_execution_profile is supported only by "
            "Humanoid PPO, GoRL-FM, and GoRL-Diffusion Brax paths"
        )
    if execution_profile is not None:
        try:
            HumanoidPPOExecutionProfile(execution_profile)
        except (TypeError, ValueError) as error:
            choices = ", ".join(
                profile.value for profile in HumanoidPPOExecutionProfile
            )
            raise ConfigError(
                "runtime.humanoid_ppo_execution_profile must be one of "
                f"{choices}; got {execution_profile!r}"
            ) from error
    matmul_precision = runtime.get("jax_default_matmul_precision")
    if matmul_precision is not None and matmul_precision not in JAX_MATMUL_PRECISIONS:
        choices = ", ".join(sorted(JAX_MATMUL_PRECISIONS))
        raise ConfigError(
            "runtime.jax_default_matmul_precision must be one of "
            f"{choices}; got {matmul_precision!r}"
        )
    triton_gemm = runtime.get("xla_gpu_triton_gemm_any")
    if triton_gemm is not None and not isinstance(triton_gemm, bool):
        raise ConfigError("runtime.xla_gpu_triton_gemm_any must be boolean")
    autotune_level = runtime.get("xla_gpu_autotune_level")
    if autotune_level is not None and (
        isinstance(autotune_level, bool)
        or not isinstance(autotune_level, int)
        or autotune_level < 0
    ):
        raise ConfigError(
            "runtime.xla_gpu_autotune_level must be a non-negative integer"
        )

    evaluation = _section_mapping(resolved, "evaluation")
    _validate_flat_keys(
        evaluation,
        {"deterministic", "episodes"},
        path="evaluation",
    )
    episodes = evaluation.get("episodes")
    for stage in range(len(stage_steps)):
        stage_ppo = resolve_stage_settings(resolved, "ppo", stage)
        if episodes != stage_ppo.get("num_eval_envs"):
            raise ConfigError(
                "evaluation.episodes must equal ppo.num_eval_envs at "
                f"stage {stage}; got {episodes!r} and "
                f"{stage_ppo.get('num_eval_envs')!r}"
            )
    deterministic = evaluation.get("deterministic")
    if not isinstance(deterministic, bool):
        raise ConfigError("evaluation.deterministic must be boolean")
    if (profile == "dm_control" or method in {"fpo", "dppo"}) and not deterministic:
        raise ConfigError(
            f"{method}/{profile} uses deterministic evaluation; "
            "evaluation.deterministic must be true"
        )
    if is_humanoid_ppo and deterministic:
        raise ConfigError(
            "Humanoid PPO uses stochastic training evaluations plus a separate "
            "deterministic report; evaluation.deterministic must be false"
        )

    smoke = _section_mapping(resolved, "smoke")
    _validate_flat_keys(
        smoke,
        _GORL_SMOKE_KEYS if is_gorl else _BASELINE_SMOKE_KEYS,
        path="smoke",
    )

    if is_gorl:
        expected_semantics = "bounded" if profile == "humanoid" else "legacy_pre_tanh"
        if collection.get("action_semantics") != expected_semantics:
            raise ConfigError(
                f"{profile} collection requires action_semantics={expected_semantics!r}"
            )
        transition_order = collection.get("transition_order")
        if transition_order is not None and transition_order not in {
            "episode_major",
            "batched_time_major",
        }:
            raise ConfigError(
                "collection.transition_order must be 'episode_major' or "
                "'batched_time_major'"
            )
        teacher_transition_order = collection.get("teacher_transition_order")
        if teacher_transition_order is not None and teacher_transition_order not in {
            "episode_major",
            "batched_time_major",
        }:
            raise ConfigError(
                "collection.teacher_transition_order must be 'episode_major' "
                "or 'batched_time_major'"
            )
        collection_seed_schedule = collection.get("seed_schedule", "stage_offset")
        if collection_seed_schedule not in {"stage_offset", "fixed"}:
            raise ConfigError(
                "collection.seed_schedule must be 'stage_offset' or 'fixed'"
            )
        teacher_seed_role = collection.get("teacher_seed_role", "teacher")
        if profile == "humanoid" and teacher_seed_role not in {
            "primary",
            "teacher",
        }:
            raise ConfigError(
                "collection.teacher_seed_role must be 'primary' or 'teacher'"
            )
        _validate_gorl_decoder(
            resolved,
            method=method,
            profile=profile,
            stage_count=len(stage_steps),
        )


@dataclass(frozen=True)
class RunConfig:
    task: str
    method: str
    seed: int
    teacher_seed: int
    decoder_seed: int
    encoder_seed: int
    eval_seed: int
    profile: str
    ppo_backend: str
    stage_steps: tuple[int, ...]
    target_return: float
    smoke: bool
    wandb_mode: str
    wandb_project: str
    wandb_entity: str | None
    output_root: Path
    config_root: Path
    config_sources: tuple[Path, ...]
    values: Mapping[str, Any]

    @property
    def total_benchmark_env_steps(self) -> int:
        return _benchmark_start_env_steps(self.values) + sum(self.stage_steps)

    def section(self, name: str) -> dict[str, Any]:
        value = self.values.get(name, {})
        if not isinstance(value, Mapping):
            raise ConfigError(f"resolved configuration section is not a table: {name}")
        return _copy(dict(value))

    def to_dict(self) -> dict[str, Any]:
        return _copy(dict(self.values))


def load_run_config(
    *,
    task: str,
    method: str,
    seed: int,
    profile: str = "auto",
    teacher_seed: int | None = None,
    decoder_seed: int | None = None,
    encoder_seed: int | None = None,
    eval_seed: int | None = None,
    config_root: str | Path | None = None,
    overlay: str | Path | None = None,
    output_root: str | Path = "runs",
    smoke: bool = False,
    wandb_mode: str = "disabled",
    wandb_project: str = "gorl-benchmark",
    wandb_entity: str | None = None,
) -> RunConfig:
    if task not in TASKS:
        raise ConfigError(f"unsupported task: {task}")
    if method not in METHODS:
        raise ConfigError(f"unsupported method: {method}")
    if profile not in PROFILES:
        raise ConfigError(f"unsupported PPO profile: {profile}")
    if wandb_mode not in {"disabled", "offline", "online"}:
        raise ConfigError(f"unsupported W&B mode: {wandb_mode}")
    if not wandb_project:
        raise ConfigError("wandb_project must be non-empty")

    root = (
        Path(config_root).expanduser().resolve()
        if config_root is not None
        else default_config_root()
    )
    task_path = root / "tasks" / f"{task}.toml"
    method_path = root / "methods" / f"{method}.toml"
    task_values = _read_toml(task_path)
    method_values = _read_toml(method_path)
    _validate_top_level(task_values, task_path)
    _validate_top_level(method_values, method_path)

    if task_values.get("task") != task:
        raise ConfigError(f"{task_path} must declare task = {task!r}")
    if method_values.get("method") != method:
        raise ConfigError(f"{method_path} must declare method = {method!r}")
    if task_values.get("schema_version") != method_values.get("schema_version"):
        raise ConfigError("task and method config schema versions do not match")

    declared_profile = task_values.get("profile")
    if declared_profile not in PPO_BACKENDS:
        raise ConfigError(f"{task_path} has an invalid profile: {declared_profile!r}")
    resolved_profile = declared_profile if profile == "auto" else profile
    if resolved_profile != declared_profile:
        raise ConfigError(
            f"{task} requires profile {declared_profile!r}, not {resolved_profile!r}"
        )

    profile_overrides = method_values.pop("profile_overrides", {})
    if not isinstance(profile_overrides, Mapping):
        raise ConfigError(f"profile_overrides must be a table: {method_path}")
    invalid_profiles = sorted(set(profile_overrides).difference(PPO_BACKENDS))
    if invalid_profiles:
        raise ConfigError(
            f"unknown profile override(s) in {method_path}: "
            f"{', '.join(invalid_profiles)}"
        )

    method_overrides = task_values.pop("method_overrides", {})
    if not isinstance(method_overrides, Mapping):
        raise ConfigError(f"method_overrides must be a table: {task_path}")
    invalid_methods = sorted(set(method_overrides).difference(METHODS))
    if invalid_methods:
        raise ConfigError(
            f"unknown method override(s) in {task_path}: {', '.join(invalid_methods)}"
        )

    resolved = _deep_merge(method_values, task_values)
    selected_profile = profile_overrides.get(resolved_profile, {})
    selected_method = method_overrides.get(method, {})
    if not isinstance(selected_profile, Mapping):
        raise ConfigError(f"profile override {resolved_profile!r} must be a table")
    if not isinstance(selected_method, Mapping):
        raise ConfigError(f"method override {method!r} must be a table")
    resolved = _deep_merge(resolved, selected_profile)
    resolved = _deep_merge(resolved, selected_method)

    sources = [method_path, task_path]
    if overlay is not None:
        overlay_path = Path(overlay).expanduser().resolve()
        overlay_values = _read_toml(overlay_path)
        _validate_top_level(overlay_values, overlay_path, overlay=True)
        if method in {"ppo", "fpo", "dppo"} and "stage_steps" in overlay_values:
            raise ConfigError(
                "standalone baseline stage_steps is derived from "
                "training.total_env_steps; override that field instead"
            )
        resolved = _deep_merge(resolved, overlay_values)
        sources.append(overlay_path)

    if smoke:
        resolved = _apply_smoke_overrides(resolved)
    resolved = _with_derived_environment(resolved)

    if resolved_profile == "humanoid" and method in {
        "ppo",
        "gorl_fm",
        "gorl_diffusion",
    }:
        runtime = dict(_section_mapping(resolved, "runtime"))
        runtime.setdefault(
            "humanoid_ppo_execution_profile",
            HumanoidPPOExecutionProfile.NATIVE.value,
        )
        resolved["runtime"] = runtime

    stage_steps = _benchmark_stage_steps(resolved, method=method)
    target_return = resolved.get("target_return")
    if (
        isinstance(target_return, bool)
        or not isinstance(target_return, (int, float))
        or not math.isfinite(target_return)
    ):
        raise ConfigError("target_return must be a finite number")
    _validate_resolved_sections(
        resolved,
        method=method,
        profile=resolved_profile,
        stage_steps=stage_steps,
    )

    primary_seed = _seed("seed", seed)
    supplied_role_seeds = {
        "teacher_seed": teacher_seed,
        "decoder_seed": decoder_seed,
        "encoder_seed": encoder_seed,
        "eval_seed": eval_seed,
    }
    resolved_seeds = {
        "seed": primary_seed,
        "teacher_seed": _seed(
            "teacher_seed", primary_seed if teacher_seed is None else teacher_seed
        ),
        "decoder_seed": _seed(
            "decoder_seed", primary_seed if decoder_seed is None else decoder_seed
        ),
        "encoder_seed": _seed(
            "encoder_seed", primary_seed if encoder_seed is None else encoder_seed
        ),
        "eval_seed": _seed(
            "eval_seed", primary_seed if eval_seed is None else eval_seed
        ),
    }
    if method in {"ppo", "fpo", "dppo"}:
        mismatched = {
            name: resolved_seeds[name]
            for name, supplied in supplied_role_seeds.items()
            if supplied is not None and resolved_seeds[name] != primary_seed
        }
        if mismatched:
            rendered = ", ".join(
                f"--{name.removesuffix('_seed').replace('_', '-')}-seed={value}"
                for name, value in mismatched.items()
            )
            raise ConfigError(
                f"standalone {method} uses only --seed={primary_seed}; component "
                "role seed overrides apply only to GoRL and must equal --seed "
                f"for standalone baselines (got {rendered})"
            )

    ppo_values = resolved.get("ppo", {})
    if not isinstance(ppo_values, Mapping):
        raise ConfigError("ppo must be a table")
    ppo_values = _deep_merge(ppo_values, {"backend": PPO_BACKENDS[resolved_profile]})
    resolved["ppo"] = ppo_values
    resolved["profile"] = resolved_profile
    resolved["stage_steps"] = list(stage_steps)
    resolved["target_return"] = float(target_return)
    resolved["seeds"] = resolved_seeds
    resolved["run"] = {
        "smoke": bool(smoke),
        "output_root": str(Path(output_root).expanduser().resolve()),
    }
    run_name = f"{task}--{method}--seed{primary_seed}"
    resolved["tracking"] = {
        "wandb": {
            "mode": wandb_mode,
            "project": wandb_project,
            "entity": wandb_entity,
            "name": run_name,
            "group": f"{task}--{method}",
            "job_type": "benchmark",
        }
    }
    resolved["provenance"] = {
        "config_sources": [str(path) for path in sources],
    }

    return RunConfig(
        task=task,
        method=method,
        seed=resolved_seeds["seed"],
        teacher_seed=resolved_seeds["teacher_seed"],
        decoder_seed=resolved_seeds["decoder_seed"],
        encoder_seed=resolved_seeds["encoder_seed"],
        eval_seed=resolved_seeds["eval_seed"],
        profile=resolved_profile,
        ppo_backend=PPO_BACKENDS[resolved_profile],
        stage_steps=stage_steps,
        target_return=float(target_return),
        smoke=bool(smoke),
        wandb_mode=wandb_mode,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        output_root=Path(output_root).expanduser().resolve(),
        config_root=root,
        config_sources=tuple(sources),
        values=resolved,
    )
