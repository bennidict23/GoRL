"""Define and validate immutable Humanoid teacher prefixes.

A prefix binds one teacher checkpoint and its collected dataset to every
teacher-critical configuration field, seed, runtime setting, payload shape, and
file hash. Branch-specific decoder and encoder settings remain outside this
contract so FM and Diffusion can initialize independently.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from gorl.artifacts import atomic_write_json
from gorl.checkpoints import sha256_file
from gorl.config import RunConfig
from gorl.humanoid.execution import (
    EXECUTION_CONTRACT_VERSION,
    resolve_humanoid_ppo_execution_contract,
)
from gorl.humanoid.types import HumanoidPPOExecutionProfile
from gorl.runtime import humanoid_gorl_runtime_plan


SHARED_PREFIX_SCHEMA_VERSION = 2
SHARED_PREFIX_MANIFEST = "shared_prefix.json"
SHARED_PREFIX_PRODUCER_PROTOCOL = "gorl_humanoid_shared_prefix_v2"
SHARED_PREFIX_PRODUCER_PROCESS_MODEL = "dedicated_teacher_subprocess"
SHARED_PREFIX_RUNTIME_SCHEMA_VERSION = 2
_GORL_METHODS = frozenset({"gorl_fm", "gorl_diffusion"})
_HUMANOID_TASKS = frozenset({"HumanoidStand", "HumanoidRun"})
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "shared_prefix_id",
        "contract_sha256",
        "contract",
        "checkpoint",
        "dataset",
        "dataset_stats",
        "benchmark_anchor",
    }
)
_CONTRACT_KEYS = frozenset(
    {
        "schema_version",
        "task",
        "profile",
        "smoke",
        "environment",
        "teacher",
        "collection",
        "runtime",
        "benchmark_start_env_steps",
    }
)
_TEACHER_KEYS = frozenset(
    {
        "config",
        "predicted_actual_environment_steps",
        "declared_actual_environment_steps",
        "seed_resolution",
    }
)
_TEACHER_CONFIG_KEYS = frozenset(
    {
        "task",
        "seed",
        "requested_environment_steps",
        "stage_index",
        "num_evals",
        "num_eval_envs",
        "training_eval_deterministic",
        "ppo_overrides",
        "execution_profile",
    }
)
_COLLECTION_KEYS = frozenset(
    {
        "seed_resolution",
        "episodes",
        "num_envs",
        "episode_length",
        "stochastic",
        "transition_order",
        "action_semantics",
    }
)
_SEED_RESOLUTION_KEYS = frozenset(
    {
        "seed",
        "role",
        "schedule",
        "stage",
        "base_seed",
        "namespace_offset",
        "stage_offset",
    }
)
_ARTIFACT_KEYS = frozenset({"path", "sha256", "size_bytes"})
_RUNTIME_KEYS = frozenset(
    {
        "schema_version",
        "producer_protocol",
        "process_model",
        "execution_profile",
        "execution_contract",
        "teacher",
    }
)
_RUNTIME_TEACHER_KEYS = frozenset(
    {
        "jax_default_matmul_precision",
        "xla_gpu_triton_gemm_any",
        "xla_gpu_autotune_level",
    }
)
_RUNTIME_SETTING_KEYS = frozenset({"configured", "effective", "source"})
_ANCHOR_KEYS = frozenset(
    {
        "source",
        "benchmark_env_steps",
        "actual_compute_env_steps",
        "return_mean",
        "return_std",
    }
)
_DATASET_STATS_KEYS = frozenset(
    {
        "num_episodes",
        "num_transitions",
        "return_mean",
        "return_std",
        "return_min",
        "return_max",
        "length_mean",
        "length_min",
        "length_max",
    }
)
_DATASET_FIELDS = frozenset(
    {
        "obs",
        "next_obs",
        "latent_action",
        "env_action",
        "reward",
        "episode_id",
        "step",
        "episode_return",
        "episode_returns",
        "episode_lengths",
    }
)


class SharedPrefixError(RuntimeError):
    """Raised when a shared Humanoid prefix is incompatible or has changed."""


@dataclass(frozen=True, slots=True)
class SharedPrefixArtifact:
    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class SharedPrefixAnchor:
    source: str
    benchmark_env_steps: int
    actual_compute_env_steps: int
    return_mean: float
    return_std: float


@dataclass(frozen=True, slots=True)
class SharedPrefixDatasetStats:
    num_episodes: int
    num_transitions: int
    return_mean: float
    return_std: float
    return_min: float
    return_max: float
    length_mean: float
    length_min: int
    length_max: int


@dataclass(frozen=True, slots=True)
class SharedPrefixBundle:
    root: Path
    manifest_path: Path
    shared_prefix_id: str
    contract_sha256: str
    contract: Mapping[str, Any]
    checkpoint: SharedPrefixArtifact
    dataset: SharedPrefixArtifact
    dataset_stats: SharedPrefixDatasetStats
    benchmark_anchor: SharedPrefixAnchor | None


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise SharedPrefixError("shared-prefix metadata must be finite JSON") from error
    return encoded.encode("utf-8")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(item) for item in value]
    return value


def _json_copy(value: Any) -> Any:
    return json.loads(_canonical_json_bytes(_thaw_json(value)))


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    label: str,
) -> None:
    actual = set(value)
    if actual == expected:
        return
    missing = sorted(expected.difference(actual))
    unknown = sorted(actual.difference(expected))
    details = []
    if missing:
        details.append(f"missing {', '.join(missing)}")
    if unknown:
        details.append(f"unknown {', '.join(str(item) for item in unknown)}")
    raise SharedPrefixError(f"invalid {label}: {'; '.join(details)}")


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SharedPrefixError(f"{label} must be an object")
    return value


def _require_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise SharedPrefixError(f"{label} must be a boolean")
    return value


def _require_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SharedPrefixError(f"{label} must be an integer >= {minimum}")
    return value


def _require_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SharedPrefixError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise SharedPrefixError(f"{label} must be a finite number")
    return number


def _require_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise SharedPrefixError(f"{label} must be a non-empty string")
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    digest = _require_string(value, label=label)
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise SharedPrefixError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _validate_seed_resolution(
    value: Any,
    *,
    label: str,
    expected_seed: int | None = None,
    expected_role: str | None = None,
) -> None:
    resolution = _require_mapping(value, label=label)
    _require_keys(resolution, _SEED_RESOLUTION_KEYS, label=label)
    seed = _require_int(resolution["seed"], label=f"{label}.seed")
    base_seed = _require_int(resolution["base_seed"], label=f"{label}.base_seed")
    namespace_offset = _require_int(
        resolution["namespace_offset"], label=f"{label}.namespace_offset"
    )
    stage_offset = _require_int(
        resolution["stage_offset"], label=f"{label}.stage_offset"
    )
    role = _require_string(resolution["role"], label=f"{label}.role")
    schedule = _require_string(resolution["schedule"], label=f"{label}.schedule")
    if schedule != "fixed" or resolution["stage"] is not None or stage_offset != 0:
        raise SharedPrefixError(f"{label} must describe a fixed, un-staged seed")
    if seed != base_seed + namespace_offset + stage_offset:
        raise SharedPrefixError(f"{label} does not resolve to its declared seed")
    if expected_seed is not None and seed != expected_seed:
        raise SharedPrefixError(f"{label} seed does not match its consumer")
    if expected_role is not None and role != expected_role:
        raise SharedPrefixError(f"{label} role must be {expected_role!r}")


def _validated_contract(value: Any) -> dict[str, Any]:
    contract = _require_mapping(value, label="shared-prefix contract")
    copied = _json_copy(dict(contract))
    _require_keys(copied, _CONTRACT_KEYS, label="shared-prefix contract")
    schema_version = _require_int(
        copied["schema_version"],
        label="shared-prefix contract schema_version",
        minimum=1,
    )
    if schema_version != SHARED_PREFIX_SCHEMA_VERSION:
        raise SharedPrefixError(
            "unsupported shared-prefix contract schema; regenerate legacy prefixes "
            "with the current execution contract"
        )
    task = _require_string(copied["task"], label="shared-prefix contract task")
    if task not in _HUMANOID_TASKS:
        raise SharedPrefixError(f"unsupported shared-prefix task: {task!r}")
    if copied["profile"] != "humanoid":
        raise SharedPrefixError("shared-prefix contract profile must be 'humanoid'")
    _require_bool(copied["smoke"], label="shared-prefix contract smoke")
    benchmark_steps = _require_int(
        copied["benchmark_start_env_steps"],
        label="shared-prefix contract benchmark_start_env_steps",
    )

    environment = _require_mapping(
        copied["environment"], label="shared-prefix contract environment"
    )
    for key in ("suite", "backend", "action_semantics"):
        _require_string(
            environment.get(key), label=f"shared-prefix contract environment.{key}"
        )
    environment_episode_length = _require_int(
        environment.get("episode_length"),
        label="shared-prefix contract environment.episode_length",
        minimum=1,
    )
    if environment["action_semantics"] != "bounded":
        raise SharedPrefixError("shared-prefix environment actions must be bounded")

    teacher = _require_mapping(
        copied["teacher"], label="shared-prefix contract teacher"
    )
    _require_keys(teacher, _TEACHER_KEYS, label="shared-prefix contract teacher")
    teacher_config = _require_mapping(
        teacher["config"], label="shared-prefix teacher config"
    )
    _require_keys(
        teacher_config,
        _TEACHER_CONFIG_KEYS,
        label="shared-prefix teacher config",
    )
    if teacher_config["task"] != task:
        raise SharedPrefixError(
            "shared-prefix teacher task does not match its contract"
        )
    teacher_seed = _require_int(
        teacher_config["seed"], label="shared-prefix teacher config.seed"
    )
    requested_steps = _require_int(
        teacher_config["requested_environment_steps"],
        label="shared-prefix teacher requested_environment_steps",
        minimum=1,
    )
    stage_index = _require_int(
        teacher_config["stage_index"],
        label="shared-prefix teacher stage_index",
    )
    if stage_index != 0:
        raise SharedPrefixError("shared-prefix teacher must be stage 0")
    _require_int(
        teacher_config["num_evals"],
        label="shared-prefix teacher num_evals",
        minimum=1,
    )
    _require_int(
        teacher_config["num_eval_envs"],
        label="shared-prefix teacher num_eval_envs",
        minimum=1,
    )
    _require_bool(
        teacher_config["training_eval_deterministic"],
        label="shared-prefix teacher training_eval_deterministic",
    )
    teacher_execution_profile = _require_string(
        teacher_config["execution_profile"],
        label="shared-prefix teacher execution_profile",
    )
    try:
        HumanoidPPOExecutionProfile(teacher_execution_profile)
    except ValueError as error:
        raise SharedPrefixError(
            "unsupported shared-prefix teacher execution_profile"
        ) from error
    overrides = _require_mapping(
        teacher_config["ppo_overrides"],
        label="shared-prefix teacher ppo_overrides",
    )
    predicted_steps = _require_int(
        teacher["predicted_actual_environment_steps"],
        label="shared-prefix teacher predicted_actual_environment_steps",
        minimum=1,
    )
    if predicted_steps < requested_steps:
        raise SharedPrefixError(
            "shared-prefix teacher actual steps cannot precede requested steps"
        )
    declared_steps = teacher["declared_actual_environment_steps"]
    if declared_steps is not None:
        _require_int(
            declared_steps,
            label="shared-prefix teacher declared_actual_environment_steps",
            minimum=1,
        )
    _validate_seed_resolution(
        teacher["seed_resolution"],
        label="shared-prefix teacher seed_resolution",
        expected_seed=teacher_seed,
        expected_role="teacher",
    )

    collection = _require_mapping(
        copied["collection"], label="shared-prefix contract collection"
    )
    _require_keys(
        collection,
        _COLLECTION_KEYS,
        label="shared-prefix contract collection",
    )
    _validate_seed_resolution(
        collection["seed_resolution"],
        label="shared-prefix collection seed_resolution",
    )
    _require_int(
        collection["episodes"],
        label="shared-prefix collection episodes",
        minimum=2,
    )
    _require_int(
        collection["num_envs"],
        label="shared-prefix collection num_envs",
        minimum=1,
    )
    episode_length = _require_int(
        collection["episode_length"],
        label="shared-prefix collection episode_length",
        minimum=1,
    )
    _require_bool(collection["stochastic"], label="shared-prefix collection stochastic")
    transition_order = _require_string(
        collection["transition_order"],
        label="shared-prefix collection transition_order",
    )
    if transition_order not in {"episode_major", "batched_time_major"}:
        raise SharedPrefixError("unsupported shared-prefix transition order")
    if collection["action_semantics"] != "bounded":
        raise SharedPrefixError("shared-prefix collection actions must be bounded")
    if environment_episode_length != episode_length:
        raise SharedPrefixError(
            "shared-prefix environment and collection episode lengths differ"
        )
    override_episode_length = _require_int(
        overrides.get("episode_length"),
        label="shared-prefix teacher ppo_overrides.episode_length",
        minimum=1,
    )
    if override_episode_length != episode_length:
        raise SharedPrefixError(
            "shared-prefix teacher and collection episode lengths differ"
        )

    runtime = _require_mapping(
        copied["runtime"], label="shared-prefix contract runtime"
    )
    _require_keys(runtime, _RUNTIME_KEYS, label="shared-prefix contract runtime")
    runtime_schema = _require_int(
        runtime["schema_version"],
        label="shared-prefix runtime schema_version",
        minimum=1,
    )
    if runtime_schema != SHARED_PREFIX_RUNTIME_SCHEMA_VERSION:
        raise SharedPrefixError("unsupported shared-prefix runtime schema")
    if runtime["producer_protocol"] != SHARED_PREFIX_PRODUCER_PROTOCOL:
        raise SharedPrefixError("unsupported shared-prefix producer protocol")
    if runtime["process_model"] != SHARED_PREFIX_PRODUCER_PROCESS_MODEL:
        raise SharedPrefixError("unsupported shared-prefix producer process model")
    runtime_execution_profile = _require_string(
        runtime["execution_profile"],
        label="shared-prefix runtime execution_profile",
    )
    try:
        selected_execution_profile = HumanoidPPOExecutionProfile(
            runtime_execution_profile
        )
    except ValueError as error:
        raise SharedPrefixError(
            "unsupported shared-prefix runtime execution_profile"
        ) from error
    if teacher_execution_profile != selected_execution_profile.value:
        raise SharedPrefixError(
            "shared-prefix teacher and runtime execution profiles differ"
        )
    execution_contract = _require_mapping(
        runtime["execution_contract"],
        label="shared-prefix runtime execution_contract",
    )
    canonical_execution_contract = _json_copy(
        resolve_humanoid_ppo_execution_contract(selected_execution_profile).to_dict()
    )
    if _json_copy(dict(execution_contract)) != canonical_execution_contract:
        raise SharedPrefixError(
            "shared-prefix runtime execution_contract does not match its profile "
            f"or contract version {EXECUTION_CONTRACT_VERSION}"
        )
    runtime_teacher = _require_mapping(
        runtime["teacher"], label="shared-prefix runtime teacher"
    )
    _require_keys(
        runtime_teacher,
        _RUNTIME_TEACHER_KEYS,
        label="shared-prefix runtime teacher",
    )
    _require_string(
        runtime_teacher["jax_default_matmul_precision"],
        label="shared-prefix runtime teacher.jax_default_matmul_precision",
    )
    for key, expected_type in (
        ("xla_gpu_triton_gemm_any", bool),
        ("xla_gpu_autotune_level", int),
    ):
        setting = _require_mapping(
            runtime_teacher[key],
            label=f"shared-prefix runtime teacher.{key}",
        )
        _require_keys(
            setting,
            _RUNTIME_SETTING_KEYS,
            label=f"shared-prefix runtime teacher.{key}",
        )
        configured = setting["configured"]
        effective = setting["effective"]
        if expected_type is bool:
            valid_configured = configured is None or isinstance(configured, bool)
            valid_effective = effective is None or isinstance(effective, bool)
        else:
            valid_configured = configured is None or (
                not isinstance(configured, bool)
                and isinstance(configured, int)
                and configured >= 0
            )
            valid_effective = effective is None or (
                not isinstance(effective, bool)
                and isinstance(effective, int)
                and effective >= 0
            )
        if not valid_configured or not valid_effective:
            raise SharedPrefixError(
                f"invalid shared-prefix runtime teacher.{key} value"
            )
        if configured != effective or setting["source"] != "runtime":
            raise SharedPrefixError(
                f"invalid shared-prefix runtime teacher.{key} resolution"
            )
    del benchmark_steps
    return copied


def _validate_config(config: RunConfig) -> None:
    if config.profile != "humanoid" or config.method not in _GORL_METHODS:
        raise SharedPrefixError(
            "shared prefixes require Humanoid gorl_fm or gorl_diffusion configs"
        )


def teacher_critical_contract(config: RunConfig) -> dict[str, Any]:
    """Return the inputs that must match before two branches share a prefix.

    The teacher PPO object is built through the same private resolvers used by
    the training pipeline. Decoder and downstream component seeds are excluded
    deliberately because those values belong to the branches, not the shared
    prefix.
    """

    _validate_config(config)
    from .pipeline import (
        _collection_num_envs,
        _make_ppo_config,
        _resolved_actual_steps,
        _stage_ppo_values,
        _teacher_collection_seed_resolution,
        _teacher_requested_steps,
        _teacher_seed_resolution,
    )

    ppo_values = _stage_ppo_values(config, 0)
    requested_steps = _teacher_requested_steps(config)
    teacher_seed = _teacher_seed_resolution(config)
    teacher_config = _make_ppo_config(
        config,
        values=ppo_values,
        requested_steps=requested_steps,
        seed=teacher_seed.seed,
        stage_index=0,
    )
    collection = config.section("collection")
    collection_seed = _teacher_collection_seed_resolution(config)
    teacher_episodes = (
        int(collection["episodes"])
        if config.smoke
        else int(collection.get("teacher_episodes", collection["episodes"]))
    )
    transition_order = str(
        collection.get(
            "teacher_transition_order",
            collection.get("transition_order", "episode_major"),
        )
    )
    training = config.section("training")
    runtime_plan = humanoid_gorl_runtime_plan(config)
    contract = {
        "schema_version": SHARED_PREFIX_SCHEMA_VERSION,
        "task": config.task,
        "profile": config.profile,
        "smoke": config.smoke,
        "environment": config.section("environment"),
        "teacher": {
            "config": asdict(teacher_config),
            "predicted_actual_environment_steps": _resolved_actual_steps(
                requested_steps,
                ppo_values,
            ),
            "declared_actual_environment_steps": training.get(
                "teacher_actual_env_steps"
            ),
            "seed_resolution": teacher_seed.to_dict(),
        },
        "collection": {
            "seed_resolution": collection_seed.to_dict(),
            "episodes": teacher_episodes,
            "num_envs": _collection_num_envs(collection),
            "episode_length": int(ppo_values["episode_length"]),
            "stochastic": bool(collection.get("stochastic", True)),
            "transition_order": transition_order,
            "action_semantics": collection.get("action_semantics"),
        },
        "runtime": {
            "schema_version": SHARED_PREFIX_RUNTIME_SCHEMA_VERSION,
            "producer_protocol": SHARED_PREFIX_PRODUCER_PROTOCOL,
            "process_model": SHARED_PREFIX_PRODUCER_PROCESS_MODEL,
            "execution_profile": runtime_plan["execution_profile"],
            "execution_contract": runtime_plan["execution_contract"],
            "teacher": runtime_plan["teacher"],
        },
        "benchmark_start_env_steps": int(training.get("benchmark_start_env_steps", 0)),
    }
    return _validated_contract(contract)


def require_matching_teacher_contracts(
    configs: Sequence[RunConfig],
) -> dict[str, Any]:
    """Return one contract or reject configs that cannot share a prefix."""

    if len(configs) < 2:
        raise SharedPrefixError("at least two configs are required")
    contracts = [teacher_critical_contract(config) for config in configs]
    reference_hash = _sha256_json(contracts[0])
    mismatches = [
        f"{config.method}/seed-{config.seed}"
        for config, contract in zip(configs[1:], contracts[1:], strict=True)
        if _sha256_json(contract) != reference_hash
    ]
    if mismatches:
        names = ", ".join(mismatches)
        raise SharedPrefixError(
            "Humanoid branches do not share the same teacher-critical "
            f"contract: {names}"
        )
    return contracts[0]


def _dataset_stats_from_mapping(value: Any) -> SharedPrefixDatasetStats:
    stats = _require_mapping(value, label="shared-prefix dataset_stats")
    _require_keys(stats, _DATASET_STATS_KEYS, label="shared-prefix dataset_stats")
    num_episodes = _require_int(
        stats["num_episodes"],
        label="shared-prefix dataset_stats.num_episodes",
        minimum=1,
    )
    num_transitions = _require_int(
        stats["num_transitions"],
        label="shared-prefix dataset_stats.num_transitions",
        minimum=1,
    )
    return_mean = _require_number(
        stats["return_mean"], label="shared-prefix dataset_stats.return_mean"
    )
    return_std = _require_number(
        stats["return_std"], label="shared-prefix dataset_stats.return_std"
    )
    return_min = _require_number(
        stats["return_min"], label="shared-prefix dataset_stats.return_min"
    )
    return_max = _require_number(
        stats["return_max"], label="shared-prefix dataset_stats.return_max"
    )
    length_mean = _require_number(
        stats["length_mean"], label="shared-prefix dataset_stats.length_mean"
    )
    length_min = _require_int(
        stats["length_min"],
        label="shared-prefix dataset_stats.length_min",
        minimum=1,
    )
    length_max = _require_int(
        stats["length_max"],
        label="shared-prefix dataset_stats.length_max",
        minimum=1,
    )
    if (
        num_transitions < num_episodes
        or return_std < 0
        or not return_min <= return_mean <= return_max
        or not length_min <= length_mean <= length_max
        or not math.isclose(
            length_mean,
            num_transitions / num_episodes,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise SharedPrefixError("invalid shared-prefix dataset_stats")
    return SharedPrefixDatasetStats(
        num_episodes=num_episodes,
        num_transitions=num_transitions,
        return_mean=return_mean,
        return_std=return_std,
        return_min=return_min,
        return_max=return_max,
        length_mean=length_mean,
        length_min=length_min,
        length_max=length_max,
    )


def _benchmark_anchor_from_mapping(value: Any) -> SharedPrefixAnchor | None:
    if value is None:
        return None
    anchor = _require_mapping(value, label="shared-prefix benchmark_anchor")
    _require_keys(anchor, _ANCHOR_KEYS, label="shared-prefix benchmark_anchor")
    source = _require_string(
        anchor["source"], label="shared-prefix benchmark_anchor.source"
    )
    benchmark_steps = _require_int(
        anchor["benchmark_env_steps"],
        label="shared-prefix benchmark_anchor.benchmark_env_steps",
    )
    compute_steps = _require_int(
        anchor["actual_compute_env_steps"],
        label="shared-prefix benchmark_anchor.actual_compute_env_steps",
    )
    return_mean = _require_number(
        anchor["return_mean"], label="shared-prefix benchmark_anchor.return_mean"
    )
    return_std = _require_number(
        anchor["return_std"], label="shared-prefix benchmark_anchor.return_std"
    )
    if source != "teacher_dataset_mean" or return_std < 0:
        raise SharedPrefixError("invalid shared-prefix benchmark_anchor")
    return SharedPrefixAnchor(
        source=source,
        benchmark_env_steps=benchmark_steps,
        actual_compute_env_steps=compute_steps,
        return_mean=return_mean,
        return_std=return_std,
    )


def _benchmark_anchor_from_dataset(
    contract: Mapping[str, Any],
    stats: SharedPrefixDatasetStats,
) -> SharedPrefixAnchor | None:
    benchmark_steps = contract["benchmark_start_env_steps"]
    if benchmark_steps == 0:
        return None
    return SharedPrefixAnchor(
        source="teacher_dataset_mean",
        benchmark_env_steps=benchmark_steps,
        actual_compute_env_steps=contract["teacher"][
            "predicted_actual_environment_steps"
        ],
        return_mean=stats.return_mean,
        return_std=stats.return_std,
    )


def _stable_file_digest(path: Path, *, label: str) -> tuple[str, int]:
    try:
        before = path.stat()
        digest = sha256_file(path)
        after = path.stat()
    except OSError as error:
        raise SharedPrefixError(f"cannot hash shared-prefix {label}: {path}") from error
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity:
        raise SharedPrefixError(f"shared-prefix {label} changed while it was hashed")
    return digest, after.st_size


def _artifact_record(root: Path, path: Path, *, label: str) -> dict[str, Any]:
    resolved_root = root.resolve()
    try:
        resolved_path = path.expanduser().resolve(strict=True)
        relative = resolved_path.relative_to(resolved_root)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        raise SharedPrefixError(
            f"shared-prefix {label} must be a file inside {resolved_root}: {path}"
        ) from error
    if not resolved_path.is_file():
        raise SharedPrefixError(f"shared-prefix {label} is not a file: {path}")
    digest, size = _stable_file_digest(resolved_path, label=label)
    if size < 1:
        raise SharedPrefixError(f"shared-prefix {label} must not be empty")
    return {
        "path": relative.as_posix(),
        "sha256": digest,
        "size_bytes": size,
    }


def _load_artifact(root: Path, value: Any, *, label: str) -> SharedPrefixArtifact:
    record = _require_mapping(value, label=f"shared-prefix {label} record")
    _require_keys(record, _ARTIFACT_KEYS, label=f"shared-prefix {label} record")
    raw_path = _require_string(record["path"], label=f"shared-prefix {label} path")
    expected_sha256 = _require_sha256(
        record["sha256"], label=f"shared-prefix {label} sha256"
    )
    expected_size = _require_int(
        record["size_bytes"],
        label=f"shared-prefix {label} size_bytes",
        minimum=1,
    )
    relative = Path(raw_path)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != raw_path
    ):
        raise SharedPrefixError(
            f"shared-prefix {label} path must be canonical relative"
        )
    try:
        path = (root / relative).resolve(strict=True)
        path.relative_to(root)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        raise SharedPrefixError(
            f"shared-prefix {label} path escapes its bundle"
        ) from error
    if not path.is_file():
        raise SharedPrefixError(f"invalid shared-prefix {label} artifact")
    actual_sha256, actual_size = _stable_file_digest(path, label=label)
    if actual_size != expected_size or actual_sha256 != expected_sha256:
        raise SharedPrefixError(
            f"shared-prefix {label} changed after the manifest was written"
        )
    return SharedPrefixArtifact(
        path=path,
        sha256=actual_sha256,
        size_bytes=actual_size,
    )


def _validate_checkpoint(
    path: Path,
    contract: Mapping[str, Any],
) -> Any:
    from gorl.decoders.types import DecoderKind

    from .checkpoint import load_state

    try:
        state = load_state(path)
    except (EOFError, OSError, pickle.UnpicklingError, TypeError, ValueError) as error:
        raise SharedPrefixError(
            f"invalid shared-prefix checkpoint payload: {path}"
        ) from error
    teacher = contract["teacher"]
    teacher_config = teacher["config"]
    expected = {
        "task": teacher_config["task"],
        "seed": teacher_config["seed"],
        "stage_index": 0,
        "requested_environment_steps": teacher_config["requested_environment_steps"],
        "actual_environment_steps": teacher["predicted_actual_environment_steps"],
        "training_eval_deterministic": teacher_config["training_eval_deterministic"],
        "action_semantics": "bounded",
    }
    for field, value in expected.items():
        if getattr(state, field) != value:
            raise SharedPrefixError(
                f"shared-prefix checkpoint {field} does not match its contract"
            )
    if state.decoder_kind is not DecoderKind.IDENTITY:
        raise SharedPrefixError(
            "shared-prefix checkpoint must use the identity decoder"
        )
    if state.action_size != state.latent_size:
        raise SharedPrefixError(
            "shared-prefix identity checkpoint action and latent sizes differ"
        )
    ppo_parameters = state.ppo_parameters
    for key, value in teacher_config["ppo_overrides"].items():
        if key not in ppo_parameters or _json_copy(ppo_parameters[key]) != value:
            raise SharedPrefixError(
                f"shared-prefix checkpoint PPO field {key!r} does not match"
            )
    if ppo_parameters.get("num_timesteps") != state.requested_environment_steps:
        raise SharedPrefixError(
            "shared-prefix checkpoint num_timesteps does not match its request"
        )
    if ppo_parameters.get("num_evals") != teacher_config["num_evals"]:
        raise SharedPrefixError(
            "shared-prefix checkpoint num_evals does not match its contract"
        )
    return state


def _finite_array(array: Any, *, label: str, np: Any) -> None:
    flat = array.reshape(-1)
    chunk_size = 1_000_000
    for start in range(0, flat.size, chunk_size):
        if not np.isfinite(flat[start : start + chunk_size]).all():
            raise SharedPrefixError(f"shared-prefix dataset {label} is non-finite")


def _load_dataset_array(
    archive: Any,
    name: str,
    *,
    dtype: Any,
    ndim: int,
    np: Any,
) -> Any:
    array = np.asarray(archive[name])
    if array.dtype != np.dtype(dtype) or array.ndim != ndim:
        raise SharedPrefixError(
            f"shared-prefix dataset {name} must be {np.dtype(dtype)} with {ndim} dims"
        )
    _finite_array(array, label=name, np=np)
    return array


def _validate_transition_order(
    episode_id: Any,
    step: Any,
    episode_lengths: Any,
    *,
    transition_order: str,
    num_envs: int,
    np: Any,
) -> None:
    cursor = 0
    num_episodes = int(episode_lengths.size)
    if transition_order == "episode_major":
        for episode in range(num_episodes):
            length = int(episode_lengths[episode])
            selected_ids = episode_id[cursor : cursor + length]
            selected_steps = step[cursor : cursor + length]
            if not np.all(selected_ids == episode) or not np.array_equal(
                selected_steps,
                np.arange(length, dtype=np.int32),
            ):
                raise SharedPrefixError(
                    "shared-prefix dataset does not use episode_major order"
                )
            cursor += length
    else:
        for start in range(0, num_episodes, num_envs):
            stop = min(start + num_envs, num_episodes)
            batch_ids = np.arange(start, stop, dtype=np.int32)
            batch_lengths = episode_lengths[start:stop]
            for timestep in range(int(batch_lengths.max())):
                selected = batch_ids[batch_lengths > timestep]
                count = int(selected.size)
                if not np.array_equal(
                    episode_id[cursor : cursor + count], selected
                ) or not np.all(step[cursor : cursor + count] == timestep):
                    raise SharedPrefixError(
                        "shared-prefix dataset does not use batched_time_major order"
                    )
                cursor += count
    if cursor != episode_id.size:
        raise SharedPrefixError("shared-prefix dataset transition order is incomplete")


def _validate_dataset(
    path: Path,
    contract: Mapping[str, Any],
    *,
    action_size: int,
) -> SharedPrefixDatasetStats:
    import numpy as np

    collection = contract["collection"]
    expected_episodes = collection["episodes"]
    episode_length = collection["episode_length"]
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = sorted(_DATASET_FIELDS.difference(archive.files))
            if missing:
                raise SharedPrefixError(
                    "shared-prefix dataset is missing fields: " + ", ".join(missing)
                )
            obs = _load_dataset_array(archive, "obs", dtype=np.float32, ndim=2, np=np)
            transition_count = int(obs.shape[0])
            observation_size = int(obs.shape[1])
            if transition_count < 1 or observation_size < 1:
                raise SharedPrefixError("shared-prefix dataset observations are empty")
            del obs
            next_obs = _load_dataset_array(
                archive, "next_obs", dtype=np.float32, ndim=2, np=np
            )
            if next_obs.shape != (transition_count, observation_size):
                raise SharedPrefixError(
                    "shared-prefix dataset next_obs shape does not match obs"
                )
            del next_obs

            latent_action = _load_dataset_array(
                archive, "latent_action", dtype=np.float32, ndim=2, np=np
            )
            if latent_action.shape != (transition_count, action_size):
                raise SharedPrefixError(
                    "shared-prefix dataset latent_action shape does not match checkpoint"
                )
            env_action = _load_dataset_array(
                archive, "env_action", dtype=np.float32, ndim=2, np=np
            )
            if env_action.shape != latent_action.shape:
                raise SharedPrefixError(
                    "shared-prefix dataset env_action shape does not match latent_action"
                )
            chunk_rows = max(1, 1_000_000 // action_size)
            for start in range(0, transition_count, chunk_rows):
                latent_piece = latent_action[start : start + chunk_rows]
                action_piece = env_action[start : start + chunk_rows]
                if (
                    np.any(np.abs(latent_piece) > 1.0 + 1e-6)
                    or np.any(np.abs(action_piece) > 1.0 + 1e-6)
                    or not np.allclose(
                        latent_piece,
                        action_piece,
                        rtol=0.0,
                        atol=1e-6,
                    )
                ):
                    raise SharedPrefixError(
                        "shared-prefix teacher dataset actions must be bounded identity actions"
                    )
            del latent_action, env_action

            reward = _load_dataset_array(
                archive, "reward", dtype=np.float32, ndim=1, np=np
            )
            episode_id = _load_dataset_array(
                archive, "episode_id", dtype=np.int32, ndim=1, np=np
            )
            step = _load_dataset_array(archive, "step", dtype=np.int32, ndim=1, np=np)
            episode_return = _load_dataset_array(
                archive, "episode_return", dtype=np.float32, ndim=1, np=np
            )
            for name, array in (
                ("reward", reward),
                ("episode_id", episode_id),
                ("step", step),
                ("episode_return", episode_return),
            ):
                if array.shape != (transition_count,):
                    raise SharedPrefixError(
                        f"shared-prefix dataset {name} length does not match obs"
                    )
            episode_returns = _load_dataset_array(
                archive, "episode_returns", dtype=np.float32, ndim=1, np=np
            )
            episode_lengths = _load_dataset_array(
                archive, "episode_lengths", dtype=np.int32, ndim=1, np=np
            )
    except SharedPrefixError:
        raise
    except (EOFError, KeyError, OSError, TypeError, ValueError) as error:
        raise SharedPrefixError(
            f"invalid shared-prefix decoder dataset: {path}"
        ) from error

    if episode_returns.shape != (expected_episodes,) or episode_lengths.shape != (
        expected_episodes,
    ):
        raise SharedPrefixError(
            "shared-prefix dataset episode count does not match its contract"
        )
    if (
        np.any(episode_lengths < 1)
        or np.any(episode_lengths > episode_length)
        or int(episode_lengths.astype(np.int64).sum()) != transition_count
    ):
        raise SharedPrefixError("invalid shared-prefix dataset episode lengths")
    if np.any(episode_id < 0) or np.any(episode_id >= expected_episodes):
        raise SharedPrefixError("invalid shared-prefix dataset episode IDs")
    counts = np.bincount(episode_id, minlength=expected_episodes)
    if not np.array_equal(counts, episode_lengths.astype(np.int64)):
        raise SharedPrefixError(
            "shared-prefix dataset episode IDs and lengths disagree"
        )
    if not np.array_equal(episode_return, episode_returns[episode_id]):
        raise SharedPrefixError("shared-prefix dataset per-transition returns disagree")
    summed_rewards = np.bincount(
        episode_id,
        weights=reward.astype(np.float64),
        minlength=expected_episodes,
    )
    if not np.allclose(
        summed_rewards,
        episode_returns.astype(np.float64),
        rtol=1e-5,
        atol=1e-5,
    ):
        raise SharedPrefixError(
            "shared-prefix dataset rewards and episode returns disagree"
        )
    _validate_transition_order(
        episode_id,
        step,
        episode_lengths,
        transition_order=collection["transition_order"],
        num_envs=collection["num_envs"],
        np=np,
    )
    return SharedPrefixDatasetStats(
        num_episodes=expected_episodes,
        num_transitions=transition_count,
        return_mean=float(np.mean(episode_returns)),
        return_std=float(np.std(episode_returns)),
        return_min=float(np.min(episode_returns)),
        return_max=float(np.max(episode_returns)),
        length_mean=float(np.mean(episode_lengths)),
        length_min=int(np.min(episode_lengths)),
        length_max=int(np.max(episode_lengths)),
    )


def _validate_payloads(
    contract: Mapping[str, Any],
    checkpoint_path: Path,
    dataset_path: Path,
) -> SharedPrefixDatasetStats:
    state = _validate_checkpoint(checkpoint_path, contract)
    return _validate_dataset(
        dataset_path,
        contract,
        action_size=state.action_size,
    )


def _prefix_id_payload(
    *,
    contract_sha256: str,
    checkpoint: Mapping[str, Any],
    dataset: Mapping[str, Any],
    dataset_stats: Mapping[str, Any],
    benchmark_anchor: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": SHARED_PREFIX_SCHEMA_VERSION,
        "contract_sha256": contract_sha256,
        "checkpoint_sha256": checkpoint["sha256"],
        "dataset_sha256": dataset["sha256"],
        "dataset_stats": dict(dataset_stats),
        "benchmark_anchor": (
            None if benchmark_anchor is None else dict(benchmark_anchor)
        ),
    }


def write_shared_prefix_manifest(
    root: str | Path,
    *,
    contract: Mapping[str, Any],
    checkpoint_path: str | Path,
    dataset_path: str | Path,
) -> SharedPrefixBundle:
    """Seal a canonical teacher checkpoint and dataset behind a hash manifest."""

    bundle_root = Path(root).expanduser().resolve()
    bundle_root.mkdir(parents=True, exist_ok=True)
    copied_contract = _validated_contract(contract)
    checkpoint = _artifact_record(
        bundle_root, Path(checkpoint_path), label="checkpoint"
    )
    dataset = _artifact_record(bundle_root, Path(dataset_path), label="dataset")
    stats = _validate_payloads(
        copied_contract,
        bundle_root / checkpoint["path"],
        bundle_root / dataset["path"],
    )
    stats_record = asdict(stats)
    benchmark_anchor = _benchmark_anchor_from_dataset(copied_contract, stats)
    benchmark_anchor_record = (
        None if benchmark_anchor is None else asdict(benchmark_anchor)
    )
    contract_sha256 = _sha256_json(copied_contract)
    shared_prefix_id = _sha256_json(
        _prefix_id_payload(
            contract_sha256=contract_sha256,
            checkpoint=checkpoint,
            dataset=dataset,
            dataset_stats=stats_record,
            benchmark_anchor=benchmark_anchor_record,
        )
    )
    manifest = {
        "schema_version": SHARED_PREFIX_SCHEMA_VERSION,
        "shared_prefix_id": shared_prefix_id,
        "contract_sha256": contract_sha256,
        "contract": copied_contract,
        "checkpoint": checkpoint,
        "dataset": dataset,
        "dataset_stats": stats_record,
        "benchmark_anchor": benchmark_anchor_record,
    }
    atomic_write_json(bundle_root / SHARED_PREFIX_MANIFEST, manifest)
    return load_shared_prefix(bundle_root)


def load_shared_prefix(path: str | Path) -> SharedPrefixBundle:
    """Load a trusted bundle and verify its identity and artifact semantics."""

    selected = Path(path).expanduser()
    selected_manifest = (
        selected / SHARED_PREFIX_MANIFEST if selected.is_dir() else selected
    )
    try:
        manifest_path = selected_manifest.resolve(strict=True)
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, RuntimeError, UnicodeError, json.JSONDecodeError) as error:
        raise SharedPrefixError(
            f"cannot read shared-prefix manifest: {selected_manifest}"
        ) from error
    manifest = _require_mapping(raw, label="shared-prefix manifest")
    _require_keys(manifest, _MANIFEST_KEYS, label="shared-prefix manifest")
    schema_version = _require_int(
        manifest["schema_version"],
        label="shared-prefix manifest schema_version",
        minimum=1,
    )
    if schema_version != SHARED_PREFIX_SCHEMA_VERSION:
        raise SharedPrefixError(
            "unsupported shared-prefix manifest schema; regenerate legacy prefixes "
            "with the current execution contract"
        )
    declared_id = _require_sha256(
        manifest["shared_prefix_id"], label="shared-prefix ID"
    )
    declared_contract_sha256 = _require_sha256(
        manifest["contract_sha256"], label="shared-prefix contract_sha256"
    )
    copied_contract = _validated_contract(manifest["contract"])
    declared_stats = _dataset_stats_from_mapping(manifest["dataset_stats"])
    parsed_anchor = _benchmark_anchor_from_mapping(manifest["benchmark_anchor"])
    stats_record = asdict(declared_stats)
    benchmark_anchor_record = None if parsed_anchor is None else asdict(parsed_anchor)
    contract_sha256 = _sha256_json(copied_contract)
    if declared_contract_sha256 != contract_sha256:
        raise SharedPrefixError("shared-prefix contract hash does not match")
    root = manifest_path.parent
    checkpoint = _load_artifact(root, manifest["checkpoint"], label="checkpoint")
    dataset = _load_artifact(root, manifest["dataset"], label="dataset")
    expected_id = _sha256_json(
        _prefix_id_payload(
            contract_sha256=contract_sha256,
            checkpoint=asdict(checkpoint),
            dataset=asdict(dataset),
            dataset_stats=stats_record,
            benchmark_anchor=benchmark_anchor_record,
        )
    )
    if declared_id != expected_id:
        raise SharedPrefixError("shared-prefix ID does not match its contents")
    expected_stats = _validate_payloads(copied_contract, checkpoint.path, dataset.path)
    if declared_stats != expected_stats:
        raise SharedPrefixError("shared-prefix dataset_stats do not match the dataset")
    expected_anchor = _benchmark_anchor_from_dataset(copied_contract, expected_stats)
    if parsed_anchor != expected_anchor:
        raise SharedPrefixError(
            "shared-prefix benchmark_anchor does not match its contract and dataset"
        )
    return SharedPrefixBundle(
        root=root,
        manifest_path=manifest_path,
        shared_prefix_id=expected_id,
        contract_sha256=contract_sha256,
        contract=_freeze_json(copied_contract),
        checkpoint=checkpoint,
        dataset=dataset,
        dataset_stats=declared_stats,
        benchmark_anchor=parsed_anchor,
    )


def verify_shared_prefix(bundle: SharedPrefixBundle) -> None:
    """Revalidate a bundle after a branch finishes using it."""

    current = load_shared_prefix(bundle.manifest_path)
    if current.shared_prefix_id != bundle.shared_prefix_id:
        raise SharedPrefixError("shared-prefix identity changed during branch training")


def require_bundle_contract(
    bundle: SharedPrefixBundle,
    expected: RunConfig | Mapping[str, Any],
) -> None:
    """Reject a valid bundle that belongs to a different branch contract."""

    contract = (
        teacher_critical_contract(expected)
        if isinstance(expected, RunConfig)
        else _validated_contract(expected)
    )
    if _sha256_json(contract) != bundle.contract_sha256:
        raise SharedPrefixError(
            "shared-prefix bundle does not match the branch teacher contract"
        )
