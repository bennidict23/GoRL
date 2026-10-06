"""Private process boundary for Humanoid teacher and latent PPO phases."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import pickle
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


WORKER_SCHEMA_VERSION = 1
WORKER_MODULE = "gorl.humanoid._ppo_worker"
PHASES = ("teacher", "latent")
PROFILES = ("native", "official_direct_compat")
_OPTIONAL_HEADLESS_IMPORTS = frozenset({"mediapy", "tensorboardX"})
_LATENT_PLAYGROUND_EXPORTS = (
    "dm_control_suite",
    "locomotion",
    "manipulation",
    "registry",
    "wrapper",
)

_TEACHER_IMPORT_SURFACE = (
    "absl.app",
    "absl.flags",
    "absl.logging",
    "brax.training.agents.ppo.networks",
    "brax.training.agents.ppo.networks_vision",
    "brax.training.agents.ppo.train",
    "etils.epath",
    "flax.training.orbax_utils",
    "jax",
    "jax.numpy",
    "mediapy",
    "ml_collections.config_dict",
    "mujoco",
    "orbax.checkpoint",
    "tensorboardX",
    "wandb",
    "mujoco_playground",
    "mujoco_playground.config.dm_control_suite_params",
    "mujoco_playground.config.locomotion_params",
    "mujoco_playground.config.manipulation_params",
)

_LATENT_IMPORT_SURFACE = (
    "numpy",
    "tyro",
    "brax.training.agents.ppo.checkpoint",
    "brax.training.agents.ppo.train",
    "mujoco_playground",
)


class PPOWorkerError(RuntimeError):
    """Structured failure returned by a private Humanoid PPO worker."""

    def __init__(self, failure: Mapping[str, Any]) -> None:
        self.failure = dict(failure)
        super().__init__(
            str(self.failure.get("message") or "Humanoid PPO worker failed")
        )


@dataclass(frozen=True, slots=True)
class PPOWorkerOutcome:
    result: Any
    events: tuple[Any, ...]
    provenance: Mapping[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_pickle(path: Path, value: Any) -> None:
    _atomic_bytes(path, pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def _atomic_json(path: Path, value: Any) -> None:
    payload = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_bytes(path, payload)


def _atomic_events(path: Path, events: Sequence[Any]) -> None:
    from dataclasses import asdict

    lines = []
    for event in events:
        value = asdict(event)
        source = value.get("source")
        value["source"] = getattr(source, "value", source)
        lines.append(json.dumps(value, sort_keys=True, allow_nan=False))
    payload = (("\n".join(lines) + "\n") if lines else "").encode("utf-8")
    _atomic_bytes(path, payload)


def _read_pickle(path: Path) -> Any:
    with path.open("rb") as stream:
        return pickle.load(stream)


def _profile_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _triton_flag(raw: str | None, value: bool | None) -> str | None:
    prefix = "--xla_gpu_triton_gemm_any="
    flags = [
        item
        for item in (raw or "").split()
        if item != "--xla_gpu_triton_gemm_any" and not item.startswith(prefix)
    ]
    if value is not None:
        flags.append(f"--xla_gpu_triton_gemm_any={value}")
    return " ".join(flags) or None


def _phase_runtime_value(
    phase_runtime: Mapping[str, Any],
    name: str,
) -> Any:
    value = phase_runtime.get(name)
    return value.get("effective") if isinstance(value, Mapping) else None


def _owned_xla_values(raw: str | None) -> tuple[bool | None, int | None]:
    triton_values: list[bool] = []
    autotune_values: list[int] = []
    for token in (raw or "").split():
        if token.startswith("--xla_gpu_triton_gemm_any="):
            value = token.split("=", 1)[1].lower()
            if value not in {"true", "false"}:
                raise ValueError(f"invalid Triton XLA flag value: {value!r}")
            triton_values.append(value == "true")
        elif token.startswith("--xla_gpu_autotune_level="):
            autotune_values.append(int(token.split("=", 1)[1]))
    if len(triton_values) > 1 or len(autotune_values) > 1:
        raise ValueError("duplicate GoRL-owned XLA flags in PPO worker environment")
    return (
        None if not triton_values else triton_values[0],
        None if not autotune_values else autotune_values[0],
    )


def _runtime_validation(
    *,
    bootstrap: Mapping[str, Any],
    phase_runtime: Mapping[str, Any],
    phase: str,
    profile: str,
) -> dict[str, Any]:
    post_import = bootstrap.get("post_import_environment")
    if not isinstance(post_import, Mapping):
        return {
            "status": "unavailable",
            "reason": "bootstrap_environment_not_provided",
        }

    requested_precision = str(
        phase_runtime.get("jax_default_matmul_precision", "default")
    )
    expected_precision = (
        None if requested_precision == "default" else requested_precision
    )
    observed_precision = post_import.get("jax_default_matmul_precision")
    observed_triton, observed_autotune = _owned_xla_values(post_import.get("XLA_FLAGS"))
    if profile == "official_direct_compat":
        expected_triton = (
            _phase_runtime_value(
                phase_runtime,
                "xla_gpu_triton_gemm_any",
            )
            if phase == "teacher"
            else None
        )
        expected_autotune = None
    else:
        expected_triton = _phase_runtime_value(
            phase_runtime,
            "xla_gpu_triton_gemm_any",
        )
        expected_autotune = _phase_runtime_value(
            phase_runtime,
            "xla_gpu_autotune_level",
        )

    observed_preallocate = post_import.get("XLA_PYTHON_CLIENT_PREALLOCATE")
    observed_gl = post_import.get("MUJOCO_GL")
    checks = {
        "jax_default_matmul_precision": {
            "expected": expected_precision,
            "observed": observed_precision,
            "valid": observed_precision == expected_precision,
        },
        "xla_gpu_triton_gemm_any": {
            "expected": expected_triton,
            "observed": observed_triton,
            "valid": observed_triton == expected_triton,
        },
        "xla_gpu_autotune_level": {
            "expected": expected_autotune,
            "observed": observed_autotune,
            "valid": observed_autotune == expected_autotune,
        },
        "XLA_PYTHON_CLIENT_PREALLOCATE": {
            "expected": "false",
            "observed": observed_preallocate,
            "valid": str(observed_preallocate).lower() == "false",
        },
        "MUJOCO_GL": {
            "expected": "egl",
            "observed": observed_gl,
            "valid": str(observed_gl).lower() == "egl",
        },
    }
    invalid = [name for name, value in checks.items() if not value["valid"]]
    return {
        "status": "valid" if not invalid else "invalid",
        "checks": checks,
        "invalid": invalid,
    }


def _bootstrap(
    *,
    phase: str,
    profile: str,
    post_import_triton: bool | None,
) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError(f"unsupported Humanoid PPO worker phase: {phase!r}")
    if profile not in PROFILES:
        raise ValueError(f"unsupported Humanoid PPO execution profile: {profile!r}")

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("MUJOCO_GL", "egl")
    pre_import = {
        "JAX_DEFAULT_MATMUL_PRECISION": os.environ.get("JAX_DEFAULT_MATMUL_PRECISION"),
        "XLA_FLAGS": os.environ.get("XLA_FLAGS"),
        "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get(
            "XLA_PYTHON_CLIENT_PREALLOCATE"
        ),
        "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
        "jax_loaded": "jax" in sys.modules,
        "brax_loaded": "brax" in sys.modules,
        "mujoco_playground_loaded": "mujoco_playground" in sys.modules,
    }
    imported: list[str] = []
    imported_exports: list[str] = []
    optional_imports_missing: list[str] = []
    if profile == "official_direct_compat":
        surface = (
            _TEACHER_IMPORT_SURFACE if phase == "teacher" else _LATENT_IMPORT_SURFACE
        )
        for module_name in surface:
            try:
                imported_module = importlib.import_module(module_name)
            except ModuleNotFoundError as error:
                if (
                    module_name not in _OPTIONAL_HEADLESS_IMPORTS
                    or error.name != module_name
                ):
                    raise
                optional_imports_missing.append(module_name)
            else:
                imported.append(module_name)
                if phase == "latent" and module_name == "mujoco_playground":
                    for export in _LATENT_PLAYGROUND_EXPORTS:
                        if not hasattr(imported_module, export):
                            raise ImportError(
                                "historical latent import surface is missing "
                                f"mujoco_playground.{export}"
                            )
                        imported_exports.append(f"mujoco_playground.{export}")
        if phase == "teacher":
            flags = _triton_flag(os.environ.get("XLA_FLAGS"), post_import_triton)
            if flags is None:
                os.environ.pop("XLA_FLAGS", None)
            else:
                os.environ["XLA_FLAGS"] = flags

    import jax

    configured_precision = os.environ.get("JAX_DEFAULT_MATMUL_PRECISION")
    expected_precision = (
        None if configured_precision in {None, "default"} else configured_precision
    )
    if profile == "native":
        jax.config.update("jax_default_matmul_precision", expected_precision)
    observed_precision = jax.config.jax_default_matmul_precision
    if observed_precision != expected_precision:
        raise RuntimeError(
            "Humanoid PPO worker precision mismatch: "
            f"{observed_precision!r} != {expected_precision!r}"
        )
    preallocate = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE")
    if str(preallocate).lower() != "false":
        raise RuntimeError(
            "Humanoid PPO worker requires "
            f"XLA_PYTHON_CLIENT_PREALLOCATE=false; got {preallocate!r}"
        )
    mujoco_gl = os.environ.get("MUJOCO_GL")
    if str(mujoco_gl).lower() != "egl":
        raise RuntimeError(
            f"Humanoid PPO worker requires MUJOCO_GL=egl; got {mujoco_gl!r}"
        )
    post_import = {
        "JAX_DEFAULT_MATMUL_PRECISION": os.environ.get("JAX_DEFAULT_MATMUL_PRECISION"),
        "XLA_FLAGS": os.environ.get("XLA_FLAGS"),
        "XLA_PYTHON_CLIENT_PREALLOCATE": preallocate,
        "MUJOCO_GL": mujoco_gl,
        "jax_default_matmul_precision": observed_precision,
        "jax_loaded": "jax" in sys.modules,
        "brax_loaded": "brax" in sys.modules,
        "mujoco_playground_loaded": "mujoco_playground" in sys.modules,
    }
    return {
        "profile": profile,
        "phase": phase,
        "import_surface": (
            "native_lazy"
            if profile == "native"
            else (
                "mujoco_playground_train_jax_ppo_v005"
                if phase == "teacher"
                else "historical_encoder_official_v1"
            )
        ),
        "imported_modules": imported,
        "imported_exports": imported_exports,
        "optional_imports_missing": optional_imports_missing,
        "pre_import_environment": pre_import,
        "post_import_environment": post_import,
        "post_import_triton": post_import_triton,
    }


def _failure(error: BaseException, events: Sequence[Any]) -> dict[str, Any]:
    from .types import HumanoidNonFiniteMetricError

    if isinstance(error, HumanoidNonFiniteMetricError):
        value = error.to_dict()
    else:
        value = {
            "type": "ppo_worker_exception",
            "source": "ppo_worker",
            "error_type": type(error).__name__,
            "message": str(error) or type(error).__name__,
        }
    value["last_finite_env_steps"] = max(
        (int(event.actual_environment_steps) for event in events),
        default=0,
    )
    return value


def execute_request(
    request_path: Path,
    result_path: Path,
    manifest_path: Path,
    events_path: Path,
    *,
    phase: str,
    execution_profile: str,
    bootstrap: Mapping[str, Any] | None = None,
    teacher_trainer: Callable[..., Any] | None = None,
    latent_trainer: Callable[..., Any] | None = None,
) -> int:
    from dataclasses import replace

    from .backend import train_latent_encoder, train_teacher
    from .checkpoint import _host_backed_state
    from .execution import resolve_humanoid_ppo_execution_contract
    from .types import (
        HumanoidEvaluationConfig,
        HumanoidLatentStageConfig,
        HumanoidPPOConfig,
        HumanoidTrainingResult,
        HumanoidWarmStart,
    )

    request = _read_pickle(request_path)
    if not isinstance(request, Mapping) or request.get("schema_version") != 1:
        raise ValueError("unsupported Humanoid PPO worker request")
    if request.get("phase") != phase:
        raise ValueError("Humanoid PPO worker phase differs from its request")
    if _profile_value(request.get("execution_profile")) != execution_profile:
        raise ValueError("Humanoid PPO worker profile differs from its request")

    ppo_config = request.get("ppo_config")
    report_config = request.get("report_config")
    if not isinstance(ppo_config, HumanoidPPOConfig):
        raise TypeError("Humanoid PPO worker request is missing HumanoidPPOConfig")
    if _profile_value(ppo_config.execution_profile) != execution_profile:
        raise ValueError("Humanoid PPO config profile differs from its worker")
    if report_config is not None and not isinstance(
        report_config, HumanoidEvaluationConfig
    ):
        raise TypeError("Humanoid PPO worker request has an invalid report config")
    phase_runtime = request.get("phase_runtime") or {}
    if not isinstance(phase_runtime, Mapping):
        raise TypeError("Humanoid PPO worker request has an invalid phase runtime")

    stage_config = request.get("stage_config")
    decoder = request.get("decoder")
    warm_start = request.get("warm_start")
    if phase == "latent":
        if not isinstance(stage_config, HumanoidLatentStageConfig):
            raise TypeError("latent worker request is missing its stage config")
        if stage_config.ppo != ppo_config:
            raise ValueError("latent worker PPO config differs from its stage config")
        if decoder is None:
            raise TypeError("latent worker request is missing its decoder")
        if warm_start is not None and not isinstance(warm_start, HumanoidWarmStart):
            raise TypeError("latent worker request has an invalid warm start")

    bootstrap_manifest = dict(bootstrap or {})
    try:
        runtime_validation = _runtime_validation(
            bootstrap=bootstrap_manifest,
            phase_runtime=phase_runtime,
            phase=phase,
            profile=execution_profile,
        )
    except (TypeError, ValueError) as error:
        runtime_validation = {
            "status": "invalid",
            "checks": {},
            "invalid": ["XLA_FLAGS"],
            "error": str(error),
        }
    bootstrap_manifest["runtime_validation"] = runtime_validation
    base_manifest: dict[str, Any] = {
        "schema_version": WORKER_SCHEMA_VERSION,
        "status": "running",
        "phase": phase,
        "stage_index": int(ppo_config.stage_index),
        "execution_profile": execution_profile,
        "execution_contract_version": 1,
        "execution_contract": resolve_humanoid_ppo_execution_contract(
            execution_profile
        ).to_dict(),
        "worker": {
            "kind": "private_subprocess",
            "module": WORKER_MODULE,
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "python_executable": sys.executable,
            "cwd": str(Path.cwd()),
        },
        "bootstrap": bootstrap_manifest,
        "request": {
            "path": str(request_path.resolve()),
            "sha256": _sha256(request_path),
            "size_bytes": request_path.stat().st_size,
        },
        "inputs": dict(request.get("input_artifacts") or {}),
        "result": None,
        "events": None,
        "event_count": 0,
        "failure": None,
    }
    _atomic_json(manifest_path, base_manifest)
    events: list[Any] = []

    def on_metric(event: Any) -> None:
        events.append(event)

    try:
        if runtime_validation["status"] == "invalid":
            invalid = ", ".join(runtime_validation["invalid"])
            raise RuntimeError(
                "Humanoid PPO worker runtime validation failed: " + invalid
            )
        checkpoint_dir = manifest_path.parent / "backend_checkpoints"
        if phase == "teacher":
            trainer = teacher_trainer or train_teacher
            result = trainer(
                ppo_config,
                callback=on_metric,
                report_config=report_config,
                checkpoint_dir=checkpoint_dir,
            )
        else:
            trainer = latent_trainer or train_latent_encoder
            result = trainer(
                stage_config,
                decoder,
                warm_start=warm_start,
                callback=on_metric,
                report_config=report_config,
                checkpoint_dir=checkpoint_dir,
            )
        if not isinstance(result, HumanoidTrainingResult):
            raise TypeError("Humanoid PPO worker trainer returned an invalid result")
        host_result = replace(
            result,
            state=_host_backed_state(result.state),
            best_state=(
                None
                if result.best_state is None
                else _host_backed_state(result.best_state)
            ),
        )
        _atomic_events(events_path, events)
        _atomic_pickle(result_path, host_result)
        complete = {
            **base_manifest,
            "status": "complete",
            "result": {
                "path": str(result_path.resolve()),
                "format": "pickle",
                "sha256": _sha256(result_path),
                "size_bytes": result_path.stat().st_size,
                "host_backed": True,
            },
            "events": {
                "path": str(events_path.resolve()),
                "format": "jsonl",
                "sha256": _sha256(events_path),
                "size_bytes": events_path.stat().st_size,
            },
            "event_count": len(events),
        }
        _atomic_json(manifest_path, complete)
        return 0
    except BaseException as error:
        _atomic_events(events_path, events)
        failed = {
            **base_manifest,
            "status": "failed",
            "events": {
                "path": str(events_path.resolve()),
                "format": "jsonl",
                "sha256": _sha256(events_path),
                "size_bytes": events_path.stat().st_size,
            },
            "event_count": len(events),
            "failure": _failure(error, events),
        }
        _atomic_json(manifest_path, failed)
        return 130 if isinstance(error, KeyboardInterrupt) else 1


def _load_events(path: Path) -> tuple[Any, ...]:
    from .types import EventSource, HumanoidMetricEvent

    events = []
    if not path.exists():
        return ()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                events.append(
                    HumanoidMetricEvent(
                        source=EventSource(value["source"]),
                        task=str(value["task"]),
                        seed=int(value["seed"]),
                        stage_index=int(value["stage_index"]),
                        evaluation_index=int(value["evaluation_index"]),
                        requested_environment_steps=int(
                            value["requested_environment_steps"]
                        ),
                        actual_environment_steps=int(value["actual_environment_steps"]),
                        deterministic=bool(value["deterministic"]),
                        metrics=dict(value["metrics"]),
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise PPOWorkerError(
                    {
                        "type": "ppo_worker_protocol",
                        "source": "ppo_worker_events",
                        "message": f"invalid event at {path}:{line_number}: {error}",
                    }
                ) from error
    return tuple(events)


def launch_ppo_worker(
    *,
    phase: str,
    ppo_config: Any,
    report_config: Any,
    phase_runtime: Mapping[str, Any],
    worker_dir: Path,
    events_path: Path,
    callback: Callable[[Any], None],
    stage_config: Any = None,
    decoder: Any = None,
    warm_start: Any = None,
    input_artifacts: Mapping[str, Any] | None = None,
    _worker_module: str = WORKER_MODULE,
) -> PPOWorkerOutcome:
    from gorl.runtime import apply_process_runtime

    profile = _profile_value(getattr(ppo_config, "execution_profile", "native"))
    if phase not in PHASES:
        raise ValueError(f"unsupported Humanoid PPO worker phase: {phase!r}")
    if profile not in PROFILES:
        raise ValueError(f"unsupported Humanoid PPO execution profile: {profile!r}")

    worker_dir.mkdir(parents=True, exist_ok=True)
    request_path = worker_dir / "request.pkl"
    result_path = worker_dir / "result.pkl"
    manifest_path = worker_dir / "manifest.json"
    child_environment = dict(os.environ)
    child_environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    child_environment.setdefault("MUJOCO_GL", "egl")
    triton = phase_runtime.get("xla_gpu_triton_gemm_any")
    triton_effective = triton.get("effective") if isinstance(triton, Mapping) else None
    autotune = phase_runtime.get("xla_gpu_autotune_level")
    autotune_effective = (
        autotune.get("effective") if isinstance(autotune, Mapping) else None
    )
    precision = str(phase_runtime.get("jax_default_matmul_precision", "default"))
    post_import_triton: bool | None = None
    if profile == "official_direct_compat":
        apply_process_runtime(
            child_environment,
            matmul_precision=precision,
            triton_gemm=None,
            autotune_level=None,
        )
        if phase == "teacher":
            post_import_triton = (
                None if triton_effective is None else bool(triton_effective)
            )
    else:
        apply_process_runtime(
            child_environment,
            matmul_precision=precision,
            triton_gemm=triton_effective,
            autotune_level=autotune_effective,
        )

    _atomic_pickle(
        request_path,
        {
            "schema_version": 1,
            "phase": phase,
            "execution_profile": profile,
            "ppo_config": ppo_config,
            "stage_config": stage_config,
            "decoder": decoder,
            "warm_start": warm_start,
            "report_config": report_config,
            "phase_runtime": dict(phase_runtime),
            "input_artifacts": dict(input_artifacts or {}),
        },
    )
    command = [
        sys.executable,
        "-m",
        _worker_module,
        "--request",
        str(request_path),
        "--result",
        str(result_path),
        "--manifest",
        str(manifest_path),
        "--events",
        str(events_path),
        "--phase",
        phase,
        "--execution-profile",
        profile,
        "--post-import-triton",
        (
            "unset"
            if post_import_triton is None
            else ("true" if post_import_triton else "false")
        ),
    ]
    try:
        completed = subprocess.run(command, env=child_environment, check=False)
    except OSError as error:
        raise PPOWorkerError(
            {
                "type": "ppo_worker_launch",
                "source": "ppo_worker_launch",
                "error_type": type(error).__name__,
                "message": str(error) or type(error).__name__,
            }
        ) from error

    if not manifest_path.exists():
        raise PPOWorkerError(
            {
                "type": "ppo_worker_exit",
                "source": "ppo_worker",
                "returncode": completed.returncode,
                "message": "Humanoid PPO worker exited without a manifest",
            }
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PPOWorkerError(
            {
                "type": "ppo_worker_protocol",
                "source": "ppo_worker_manifest",
                "returncode": completed.returncode,
                "message": f"invalid Humanoid PPO worker manifest: {error}",
            }
        ) from error
    provenance = {**manifest, "command": command, "returncode": completed.returncode}
    events_artifact = manifest.get("events")
    if (
        not isinstance(events_artifact, Mapping)
        or not events_path.is_file()
        or _sha256(events_path) != events_artifact.get("sha256")
    ):
        raise PPOWorkerError(
            {
                "type": "ppo_worker_protocol",
                "source": "ppo_worker_events",
                "message": "Humanoid PPO worker event hash mismatch",
            }
        )
    events = _load_events(events_path)
    if int(manifest.get("event_count", -1)) != len(events):
        raise PPOWorkerError(
            {
                "type": "ppo_worker_protocol",
                "source": "ppo_worker_events",
                "message": "Humanoid PPO worker event count mismatch",
            }
        )
    if completed.returncode != 0 or manifest.get("status") != "complete":
        for event in events:
            callback(event)
        failure = manifest.get("failure")
        if not isinstance(failure, Mapping):
            failure = {
                "type": "ppo_worker_exit",
                "source": "ppo_worker",
                "returncode": completed.returncode,
                "message": f"Humanoid PPO worker exited with code {completed.returncode}",
            }
        raise PPOWorkerError({**failure, "worker": provenance})
    result_artifact = manifest.get("result")
    if not isinstance(result_artifact, Mapping):
        raise PPOWorkerError(
            {
                "type": "ppo_worker_protocol",
                "source": "ppo_worker_result",
                "message": "Humanoid PPO worker manifest has no result artifact",
            }
        )
    if not result_path.is_file() or _sha256(result_path) != result_artifact.get(
        "sha256"
    ):
        raise PPOWorkerError(
            {
                "type": "ppo_worker_protocol",
                "source": "ppo_worker_result",
                "message": "Humanoid PPO worker result hash mismatch",
            }
        )
    result = _read_pickle(result_path)
    from .types import EventSource, HumanoidTrainingResult

    if not isinstance(result, HumanoidTrainingResult):
        raise PPOWorkerError(
            {
                "type": "ppo_worker_protocol",
                "source": "ppo_worker_result",
                "message": "Humanoid PPO worker returned an invalid result",
            }
        )
    training_events = tuple(
        event for event in events if event.source is EventSource.TRAINING_EVAL
    )
    if tuple(result.events) != training_events:
        raise PPOWorkerError(
            {
                "type": "ppo_worker_protocol",
                "source": "ppo_worker_events",
                "message": "Humanoid PPO result and event stream differ",
            }
        )
    for event in events:
        callback(event)
    return PPOWorkerOutcome(result=result, events=events, provenance=provenance)


def _parse_optional_bool(value: str) -> bool | None:
    if value == "unset":
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    raise argparse.ArgumentTypeError("expected true, false, or unset")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=WORKER_MODULE)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--execution-profile", choices=PROFILES, required=True)
    parser.add_argument("--post-import-triton", type=_parse_optional_bool, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        bootstrap = _bootstrap(
            phase=args.phase,
            profile=args.execution_profile,
            post_import_triton=args.post_import_triton,
        )
    except BaseException as error:
        from .execution import resolve_humanoid_ppo_execution_contract

        _atomic_events(args.events, ())
        request = None
        if args.request.is_file():
            request = {
                "path": str(args.request.resolve()),
                "sha256": _sha256(args.request),
                "size_bytes": args.request.stat().st_size,
            }
        _atomic_json(
            args.manifest,
            {
                "schema_version": WORKER_SCHEMA_VERSION,
                "status": "failed",
                "phase": args.phase,
                "execution_profile": args.execution_profile,
                "execution_contract_version": 1,
                "execution_contract": resolve_humanoid_ppo_execution_contract(
                    args.execution_profile
                ).to_dict(),
                "worker": {
                    "kind": "private_subprocess",
                    "module": WORKER_MODULE,
                    "pid": os.getpid(),
                    "parent_pid": os.getppid(),
                    "python_executable": sys.executable,
                    "cwd": str(Path.cwd()),
                },
                "bootstrap": None,
                "request": request,
                "result": None,
                "events": {
                    "path": str(args.events.resolve()),
                    "format": "jsonl",
                    "sha256": _sha256(args.events),
                    "size_bytes": args.events.stat().st_size,
                },
                "event_count": 0,
                "failure": {
                    "type": "ppo_worker_bootstrap",
                    "source": "ppo_worker_bootstrap",
                    "error_type": type(error).__name__,
                    "message": str(error) or type(error).__name__,
                    "last_finite_env_steps": 0,
                },
            },
        )
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    return execute_request(
        args.request,
        args.result,
        args.manifest,
        args.events,
        phase=args.phase,
        execution_profile=args.execution_profile,
        bootstrap=bootstrap,
    )


if __name__ == "__main__":
    raise SystemExit(main())
