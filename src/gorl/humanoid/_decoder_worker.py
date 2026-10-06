"""Private process boundary for compatibility-profile decoder fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import signal
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


WORKER_SCHEMA_VERSION = 1
WORKER_MODULE = "gorl.humanoid._decoder_worker"
EXECUTION_PROFILE = "official_direct_compat"
_INTERRUPT_GRACE_SECONDS = 10
_SIGNAL_PROPAGATION_GRACE_SECONDS = 1


class DecoderWorkerError(RuntimeError):
    """Structured failure returned by a private Humanoid decoder worker."""

    def __init__(self, failure: Mapping[str, Any]) -> None:
        self.failure = dict(failure)
        super().__init__(
            str(self.failure.get("message") or "Humanoid decoder worker failed")
        )


class _ForwardedParentSignal(BaseException):
    def __init__(self, signal_number: int) -> None:
        self.signal_number = signal_number
        super().__init__(signal_number)


def _is_strict_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _raise_forwarded_parent_signal(signal_number: int, _frame: Any) -> None:
    raise _ForwardedParentSignal(signal_number)


def _install_termination_handlers() -> dict[int, Any]:
    previous: dict[int, Any] = {}
    for signal_number in (signal.SIGTERM, signal.SIGHUP):
        try:
            previous[signal_number] = signal.getsignal(signal_number)
            signal.signal(signal_number, _raise_forwarded_parent_signal)
        except ValueError:
            for installed, handler in previous.items():
                try:
                    signal.signal(installed, handler)
                except ValueError:
                    pass
            return {}
    return previous


def _restore_signal_handlers(previous: Mapping[int, Any]) -> None:
    for signal_number, handler in previous.items():
        try:
            signal.signal(signal_number, handler)
        except ValueError:
            pass


@contextmanager
def _ignore_signals_while_reaping_or_sealing():
    previous: dict[int, Any] = {}
    for signal_number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            previous[signal_number] = signal.getsignal(signal_number)
            signal.signal(signal_number, signal.SIG_IGN)
        except ValueError:
            break
    try:
        yield
    finally:
        _restore_signal_handlers(previous)


@dataclass(frozen=True, slots=True)
class DecoderWorkerOutcome:
    result: Any
    decoder_config: Any
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
        temporary.unlink(missing_ok=True)


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


def _append_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(value), sort_keys=True, allow_nan=False))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _artifact_from_bytes(
    path: Path,
    payload: bytes,
    *,
    artifact_format: str,
) -> dict[str, Any]:
    source = path.expanduser().resolve()
    return {
        "path": str(source),
        "format": artifact_format,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _read_artifact_bytes(
    value: Any,
    *,
    label: str,
    expected_format: str | None = None,
    expected_path: Path | None = None,
) -> tuple[Path, bytes]:
    if not isinstance(value, Mapping):
        raise TypeError(f"decoder worker manifest is missing its {label} artifact")
    try:
        path = Path(str(value["path"])).expanduser().resolve()
        expected_hash = str(value["sha256"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid {label} artifact description") from error
    expected_size = value.get("size_bytes")
    if not _is_strict_int(expected_size) or expected_size < 0:
        raise ValueError(f"invalid {label} artifact size")
    if expected_format is not None and value.get("format") != expected_format:
        raise ValueError(f"decoder worker {label} artifact format mismatch")
    if expected_path is not None and path != expected_path.expanduser().resolve():
        raise ValueError(f"decoder worker {label} artifact path mismatch")
    if not path.is_file():
        raise FileNotFoundError(f"decoder worker {label} does not exist: {path}")
    payload = path.read_bytes()
    if (
        len(payload) != expected_size
        or hashlib.sha256(payload).hexdigest() != expected_hash
    ):
        raise ValueError(f"decoder worker {label} artifact hash mismatch")
    return path, payload


def _read_artifact_pickle(
    value: Any,
    *,
    label: str,
    expected_path: Path | None = None,
) -> tuple[Path, Any]:
    path, payload = _read_artifact_bytes(
        value,
        label=label,
        expected_format="pickle",
        expected_path=expected_path,
    )
    try:
        return path, pickle.loads(payload)
    except Exception as error:
        raise ValueError(
            f"decoder worker {label} artifact is not valid pickle"
        ) from error


def _artifact(path: Path, *, artifact_format: str) -> dict[str, Any]:
    source = path.expanduser().resolve()
    return {
        "path": str(source),
        "format": artifact_format,
        "sha256": _sha256(source),
        "size_bytes": source.stat().st_size,
    }


def _jsonl_artifact(path: Path) -> tuple[dict[str, Any], int]:
    payload = path.read_bytes()
    text = payload.decode("utf-8")
    return (
        _artifact_from_bytes(path, payload, artifact_format="jsonl"),
        sum(1 for line in text.splitlines() if line),
    )


def _validate_artifact(
    value: Any,
    *,
    label: str,
    expected_format: str | None = None,
) -> Path:
    if not isinstance(value, Mapping):
        raise TypeError(f"decoder worker request is missing its {label} artifact")
    try:
        path = Path(str(value["path"])).expanduser().resolve()
        expected_hash = str(value["sha256"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid {label} artifact description") from error
    expected_size = value.get("size_bytes")
    if not _is_strict_int(expected_size) or expected_size < 0:
        raise ValueError(f"invalid {label} artifact size")
    if expected_format is not None and value.get("format") != expected_format:
        raise ValueError(f"decoder worker {label} artifact format mismatch")
    if not path.is_file():
        raise FileNotFoundError(f"decoder worker {label} does not exist: {path}")
    if path.stat().st_size != expected_size or _sha256(path) != expected_hash:
        raise ValueError(f"decoder worker {label} artifact hash mismatch")
    return path


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
        raise ValueError("duplicate GoRL-owned XLA flags in decoder worker environment")
    return (
        None if not triton_values else triton_values[0],
        None if not autotune_values else autotune_values[0],
    )


def _phase_runtime_value(phase_runtime: Mapping[str, Any], name: str) -> Any:
    value = phase_runtime.get(name)
    return value.get("effective") if isinstance(value, Mapping) else None


def _bootstrap() -> dict[str, Any]:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("MUJOCO_GL", "egl")
    pre_import = {
        "JAX_DEFAULT_MATMUL_PRECISION": os.environ.get("JAX_DEFAULT_MATMUL_PRECISION"),
        "XLA_FLAGS": os.environ.get("XLA_FLAGS"),
        "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get(
            "XLA_PYTHON_CLIENT_PREALLOCATE"
        ),
        "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
        "JAX_PLATFORM_NAME": os.environ.get("JAX_PLATFORM_NAME"),
        "jax_loaded": "jax" in sys.modules,
    }
    if pre_import["jax_loaded"]:
        raise RuntimeError("decoder worker must bootstrap before importing JAX")

    import jax

    post_import = {
        **pre_import,
        "jax_loaded": "jax" in sys.modules,
        "jax_default_matmul_precision": jax.config.jax_default_matmul_precision,
    }
    return {
        "profile": EXECUTION_PROFILE,
        "phase": "decoder_fit",
        "import_surface": "gorl_decoder_training_fresh_process_v1",
        "pre_import_environment": pre_import,
        "post_import_environment": post_import,
    }


def _runtime_validation(
    bootstrap: Mapping[str, Any],
    phase_runtime: Mapping[str, Any],
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
    observed_triton, observed_autotune = _owned_xla_values(post_import.get("XLA_FLAGS"))
    checks = {
        "jax_default_matmul_precision": {
            "expected": expected_precision,
            "observed": post_import.get("jax_default_matmul_precision"),
            "valid": (
                post_import.get("jax_default_matmul_precision") == expected_precision
            ),
        },
        "xla_gpu_triton_gemm_any": {
            "expected": _phase_runtime_value(phase_runtime, "xla_gpu_triton_gemm_any"),
            "observed": observed_triton,
            "valid": observed_triton
            == _phase_runtime_value(phase_runtime, "xla_gpu_triton_gemm_any"),
        },
        "xla_gpu_autotune_level": {
            "expected": _phase_runtime_value(phase_runtime, "xla_gpu_autotune_level"),
            "observed": observed_autotune,
            "valid": observed_autotune
            == _phase_runtime_value(phase_runtime, "xla_gpu_autotune_level"),
        },
        "XLA_PYTHON_CLIENT_PREALLOCATE": {
            "expected": "false",
            "observed": post_import.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "valid": str(post_import.get("XLA_PYTHON_CLIENT_PREALLOCATE")).lower()
            == "false",
        },
        "MUJOCO_GL": {
            "expected": "egl",
            "observed": post_import.get("MUJOCO_GL"),
            "valid": str(post_import.get("MUJOCO_GL")).lower() == "egl",
        },
        "fresh_jax_import": {
            "expected": False,
            "observed": bootstrap.get("pre_import_environment", {}).get("jax_loaded"),
            "valid": bootstrap.get("pre_import_environment", {}).get("jax_loaded")
            is False,
        },
    }
    invalid = [name for name, value in checks.items() if not value["valid"]]
    return {
        "status": "valid" if not invalid else "invalid",
        "checks": checks,
        "invalid": invalid,
    }


def _failure(error: BaseException) -> dict[str, Any]:
    failure = {
        "type": "decoder_worker_exception",
        "source": "decoder_worker",
        "error_type": type(error).__name__,
        "message": str(error) or type(error).__name__,
    }
    if isinstance(error, _ForwardedParentSignal):
        failure["signal"] = error.signal_number
    return failure


def _error_returncode(error: BaseException) -> int:
    if isinstance(error, KeyboardInterrupt):
        return 128 + signal.SIGINT
    if isinstance(error, _ForwardedParentSignal):
        return 128 + error.signal_number
    return 1


def _host_backed_fit(result: Any) -> Any:
    from gorl.algorithms.decoder_training import DecoderFitResult

    if not isinstance(result, DecoderFitResult):
        raise TypeError("decoder worker fitter returned an invalid result")
    import jax
    import numpy as np

    def to_host(value: Any) -> Any:
        if isinstance(value, jax.Array):
            return np.asarray(value)
        return value

    return replace(
        result,
        decoder=jax.tree_util.tree_map(to_host, result.decoder),
        final_decoder=jax.tree_util.tree_map(to_host, result.final_decoder),
    )


def _worker_identity() -> dict[str, Any]:
    return {
        "kind": "private_subprocess",
        "module": WORKER_MODULE,
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "python_executable": sys.executable,
        "cwd": str(Path.cwd()),
    }


def _terminal_manifest_is_valid(
    manifest_path: Path,
    result_path: Path,
    metrics_path: Path,
    *,
    execution_profile: str,
) -> bool:
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            not isinstance(manifest, Mapping)
            or not _is_strict_int(manifest.get("schema_version"))
            or manifest.get("schema_version") != WORKER_SCHEMA_VERSION
            or manifest.get("status") not in {"complete", "failed"}
            or manifest.get("phase") != "decoder_fit"
            or manifest.get("execution_profile") != execution_profile
            or not isinstance(manifest.get("worker"), Mapping)
        ):
            return False
        _, metrics_payload = _read_artifact_bytes(
            manifest.get("metrics"),
            label="metrics",
            expected_format="jsonl",
            expected_path=metrics_path,
        )
        metrics_text = metrics_payload.decode("utf-8")
        metric_count = sum(1 for line in metrics_text.splitlines() if line)
        recorded_metric_count = manifest.get("metric_count")
        if (
            not _is_strict_int(recorded_metric_count)
            or recorded_metric_count < 0
            or recorded_metric_count != metric_count
        ):
            return False
        if manifest.get("status") == "failed":
            return isinstance(manifest.get("failure"), Mapping)
        if manifest.get("failure") is not None:
            return False
        _read_artifact_bytes(
            manifest.get("result"),
            label="result",
            expected_format="pickle",
            expected_path=result_path,
        )
        return True
    except (
        OSError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False


def _seal_outer_failure(
    request_path: Path,
    result_path: Path,
    manifest_path: Path,
    metrics_path: Path,
    *,
    execution_profile: str,
    bootstrap: Mapping[str, Any] | None,
    error: BaseException,
    failure_type: str,
    failure_source: str,
) -> None:
    if _terminal_manifest_is_valid(
        manifest_path,
        result_path,
        metrics_path,
        execution_profile=execution_profile,
    ):
        return
    if not metrics_path.is_file():
        _atomic_bytes(metrics_path, b"")
    metrics_artifact, metric_count = _jsonl_artifact(metrics_path)
    request_artifact = None
    if request_path.is_file():
        try:
            request_payload = request_path.read_bytes()
        except OSError:
            pass
        else:
            request_artifact = _artifact_from_bytes(
                request_path,
                request_payload,
                artifact_format="pickle",
            )
    failure = {
        **_failure(error),
        "type": failure_type,
        "source": failure_source,
    }
    _atomic_json(
        manifest_path,
        {
            "schema_version": WORKER_SCHEMA_VERSION,
            "status": "failed",
            "phase": "decoder_fit",
            "execution_profile": execution_profile,
            "worker": _worker_identity(),
            "bootstrap": None if bootstrap is None else dict(bootstrap),
            "request": request_artifact,
            "inputs": {},
            "result": None,
            "metrics": metrics_artifact,
            "metric_count": metric_count,
            "failure": failure,
        },
    )


def execute_request(
    request_path: Path,
    result_path: Path,
    manifest_path: Path,
    metrics_path: Path,
    *,
    execution_profile: str,
    bootstrap: Mapping[str, Any] | None = None,
    config_builder: Callable[..., Any] | None = None,
    fitter: Callable[..., Any] | None = None,
) -> int:
    try:
        return _execute_request_impl(
            request_path,
            result_path,
            manifest_path,
            metrics_path,
            execution_profile=execution_profile,
            bootstrap=bootstrap,
            config_builder=config_builder,
            fitter=fitter,
        )
    except BaseException as error:
        with _ignore_signals_while_reaping_or_sealing():
            _seal_outer_failure(
                request_path,
                result_path,
                manifest_path,
                metrics_path,
                execution_profile=execution_profile,
                bootstrap=bootstrap,
                error=error,
                failure_type="decoder_worker_execute",
                failure_source="decoder_worker_execute",
            )
        return _error_returncode(error)


def _execute_request_impl(
    request_path: Path,
    result_path: Path,
    manifest_path: Path,
    metrics_path: Path,
    *,
    execution_profile: str,
    bootstrap: Mapping[str, Any] | None = None,
    config_builder: Callable[..., Any] | None = None,
    fitter: Callable[..., Any] | None = None,
) -> int:
    from gorl.algorithms.decoder_training import (
        DecoderFitResult,
        DecoderTrainingConfig,
        DecoderWarmStart,
        fit_decoder,
    )
    from gorl.config import RunConfig

    bootstrap_manifest = dict(bootstrap or {})
    request_payload: bytes | None = None
    request_read_error: OSError | None = None
    if request_path.is_file():
        try:
            request_payload = request_path.read_bytes()
        except OSError as error:
            request_read_error = error
    request_artifact = (
        None
        if request_payload is None
        else _artifact_from_bytes(
            request_path,
            request_payload,
            artifact_format="pickle",
        )
    )
    base_manifest: dict[str, Any] = {
        "schema_version": WORKER_SCHEMA_VERSION,
        "status": "running",
        "phase": "decoder_fit",
        "execution_profile": execution_profile,
        "worker": _worker_identity(),
        "bootstrap": bootstrap_manifest,
        "request": request_artifact,
        "inputs": {},
        "result": None,
        "metrics": None,
        "metric_count": 0,
        "failure": None,
    }
    _atomic_bytes(metrics_path, b"")
    _atomic_json(manifest_path, base_manifest)
    metric_count = 0

    def on_metric(record: Mapping[str, float | int]) -> None:
        nonlocal metric_count
        _append_json(metrics_path, record)
        metric_count += 1

    try:
        if execution_profile != EXECUTION_PROFILE:
            raise ValueError("decoder workers are reserved for official_direct_compat")
        if request_read_error is not None:
            raise request_read_error
        if request_payload is None:
            raise FileNotFoundError(
                f"decoder worker request does not exist: {request_path}"
            )
        try:
            request = pickle.loads(request_payload)
        except Exception as error:
            raise ValueError("decoder worker request is not valid pickle") from error
        if (
            not isinstance(request, Mapping)
            or not _is_strict_int(request.get("schema_version"))
            or request.get("schema_version") != WORKER_SCHEMA_VERSION
        ):
            raise ValueError("unsupported Humanoid decoder worker request")
        if request.get("execution_profile") != execution_profile:
            raise ValueError("decoder worker profile differs from its request")
        run_config = request.get("run_config")
        if not isinstance(run_config, RunConfig):
            raise TypeError("decoder worker request is missing RunConfig")
        target_stage = request.get("target_stage")
        if isinstance(target_stage, bool) or not isinstance(target_stage, int):
            raise TypeError("decoder worker target_stage must be an integer")
        phase_runtime = request.get("phase_runtime")
        if not isinstance(phase_runtime, Mapping):
            raise TypeError("decoder worker request has an invalid phase runtime")
        runtime_validation = _runtime_validation(
            bootstrap_manifest,
            phase_runtime,
        )
        bootstrap_manifest["runtime_validation"] = runtime_validation
        if runtime_validation["status"] == "invalid":
            raise RuntimeError(
                "Humanoid decoder worker runtime validation failed: "
                + ", ".join(runtime_validation["invalid"])
            )

        inputs = request.get("input_artifacts")
        if not isinstance(inputs, Mapping):
            raise TypeError("decoder worker request has invalid input artifacts")
        base_manifest = {
            **base_manifest,
            "bootstrap": bootstrap_manifest,
            "target_stage": target_stage,
            "inputs": dict(inputs),
        }
        _atomic_json(manifest_path, base_manifest)
        dataset_path = _validate_artifact(
            inputs.get("dataset"),
            label="dataset",
            expected_format="npz",
        )
        warm_enabled = bool(request.get("warm_start_enabled"))
        warm_artifact = inputs.get("warm_start")
        if warm_enabled != (warm_artifact is not None):
            raise ValueError("decoder worker warm-start artifact contract mismatch")
        warm_start = None
        if warm_artifact is not None:
            _, warm_start = _read_artifact_pickle(
                warm_artifact,
                label="warm start",
                expected_path=request_path.parent / "warm_start.pkl",
            )
            if not isinstance(warm_start, (DecoderFitResult, DecoderWarmStart)):
                raise TypeError("decoder worker warm-start artifact is invalid")

        if config_builder is None:
            from gorl.humanoid.pipeline import _decoder_training_config

            decoder_config = _decoder_training_config(
                run_config,
                dataset=dataset_path,
                target_stage=target_stage,
            )
        else:
            decoder_config = config_builder(
                run_config,
                dataset=dataset_path,
                target_stage=target_stage,
            )
        if not isinstance(decoder_config, DecoderTrainingConfig):
            raise TypeError("decoder worker config builder returned an invalid config")
        train = fitter or fit_decoder
        fitted = train(
            dataset_path,
            decoder_config,
            warm_start=warm_start,
            metric_callback=on_metric,
        )
        if not isinstance(fitted, DecoderFitResult):
            raise TypeError("decoder worker fitter returned an invalid result")
        host_fitted = _host_backed_fit(fitted)

        # Fail closed if a collector artifact changes while the fit is running.
        _validate_artifact(
            inputs.get("dataset"),
            label="dataset",
            expected_format="npz",
        )
        if warm_artifact is not None:
            _read_artifact_bytes(
                warm_artifact,
                label="warm start",
                expected_format="pickle",
                expected_path=request_path.parent / "warm_start.pkl",
            )
        _read_artifact_bytes(
            request_artifact,
            label="request",
            expected_format="pickle",
            expected_path=request_path,
        )
        bundle = {
            "format": "gorl_humanoid_decoder_worker_result",
            "schema_version": WORKER_SCHEMA_VERSION,
            "fitted": host_fitted,
            "decoder_config": decoder_config,
        }
        _atomic_pickle(result_path, bundle)
        metrics_artifact, durable_metric_count = _jsonl_artifact(metrics_path)
        if durable_metric_count != metric_count:
            raise RuntimeError(
                "decoder worker in-memory and durable metric counts differ"
            )
        complete = {
            **base_manifest,
            "status": "complete",
            "bootstrap": bootstrap_manifest,
            "decoder_config": asdict(decoder_config),
            "fit": {
                "best_step": host_fitted.best_step,
                "best_metric": host_fitted.best_metric,
                "best_value": host_fitted.best_value,
            },
            "result": {
                **_artifact(result_path, artifact_format="pickle"),
                "host_backed": True,
            },
            "metrics": metrics_artifact,
            "metric_count": durable_metric_count,
        }
        _atomic_json(manifest_path, complete)
        return 0
    except BaseException as error:
        with _ignore_signals_while_reaping_or_sealing():
            if _terminal_manifest_is_valid(
                manifest_path,
                result_path,
                metrics_path,
                execution_profile=execution_profile,
            ):
                return _error_returncode(error)
            metrics_artifact, durable_metric_count = _jsonl_artifact(metrics_path)
            failed = {
                **base_manifest,
                "status": "failed",
                "bootstrap": bootstrap_manifest,
                "metrics": metrics_artifact,
                "metric_count": durable_metric_count,
                "failure": _failure(error),
            }
            _atomic_json(manifest_path, failed)
        return _error_returncode(error)


def _wait_for_worker(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
) -> tuple[int, int | None]:
    previous_handlers = _install_termination_handlers()
    process: subprocess.Popen[Any] | None = None

    try:
        try:
            process = subprocess.Popen(command, env=dict(environment))
            return process.wait(), None
        except KeyboardInterrupt:
            if process is None:
                raise
            forwarded_signal = signal.SIGINT
        except _ForwardedParentSignal as error:
            if process is None:
                raise
            forwarded_signal = error.signal_number

        assert process is not None
        with _ignore_signals_while_reaping_or_sealing():
            # A terminal or process-group signal normally reaches the child too.
            # Give it a short chance to seal its failure manifest before forwarding
            # a parent-only signal explicitly.
            try:
                return (
                    process.wait(timeout=_SIGNAL_PROPAGATION_GRACE_SECONDS),
                    forwarded_signal,
                )
            except subprocess.TimeoutExpired:
                pass
            if process.poll() is None:
                process.send_signal(forwarded_signal)
            try:
                return (
                    process.wait(timeout=_INTERRUPT_GRACE_SECONDS),
                    forwarded_signal,
                )
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    return (
                        process.wait(timeout=_INTERRUPT_GRACE_SECONDS),
                        forwarded_signal,
                    )
                except subprocess.TimeoutExpired:
                    process.kill()
                    return process.wait(), forwarded_signal
    finally:
        _restore_signal_handlers(previous_handlers)


def launch_decoder_worker(
    *,
    run_config: Any,
    target_stage: int,
    dataset_path: Path,
    warm_start: Any | None,
    phase_runtime: Mapping[str, Any],
    worker_dir: Path,
    metrics_path: Path,
    _worker_module: str = WORKER_MODULE,
) -> DecoderWorkerOutcome:
    from gorl.runtime import apply_process_runtime

    profile = str(
        run_config.section("runtime").get("humanoid_ppo_execution_profile", "native")
    )
    if profile != EXECUTION_PROFILE:
        raise ValueError("decoder workers are reserved for official_direct_compat")
    source_dataset = dataset_path.expanduser().resolve()
    if not source_dataset.is_file():
        raise FileNotFoundError(
            f"Humanoid decoder dataset must be saved before fitting: {source_dataset}"
        )

    worker_dir.mkdir(parents=True, exist_ok=True)
    request_path = worker_dir / "request.pkl"
    result_path = worker_dir / "result.pkl"
    manifest_path = worker_dir / "manifest.json"
    warm_path = worker_dir / "warm_start.pkl"
    input_artifacts: dict[str, Any] = {
        "dataset": _artifact(source_dataset, artifact_format="npz"),
        "warm_start": None,
    }
    if warm_start is not None:
        _atomic_pickle(warm_path, warm_start)
        input_artifacts["warm_start"] = _artifact(warm_path, artifact_format="pickle")
    else:
        warm_path.unlink(missing_ok=True)

    result_path.unlink(missing_ok=True)
    manifest_path.unlink(missing_ok=True)
    metrics_path.unlink(missing_ok=True)

    _atomic_pickle(
        request_path,
        {
            "schema_version": WORKER_SCHEMA_VERSION,
            "execution_profile": profile,
            "run_config": run_config,
            "target_stage": target_stage,
            "warm_start_enabled": warm_start is not None,
            "phase_runtime": dict(phase_runtime),
            "input_artifacts": input_artifacts,
        },
    )

    child_environment = dict(os.environ)
    child_environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    child_environment.setdefault("MUJOCO_GL", "egl")
    triton = _phase_runtime_value(phase_runtime, "xla_gpu_triton_gemm_any")
    autotune = _phase_runtime_value(phase_runtime, "xla_gpu_autotune_level")
    apply_process_runtime(
        child_environment,
        matmul_precision=str(
            phase_runtime.get("jax_default_matmul_precision", "default")
        ),
        triton_gemm=triton,
        autotune_level=autotune,
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
        "--metrics",
        str(metrics_path),
        "--execution-profile",
        profile,
    ]
    try:
        returncode, parent_signal = _wait_for_worker(
            command,
            environment=child_environment,
        )
    except OSError as error:
        raise DecoderWorkerError(
            {
                "type": "decoder_worker_launch",
                "source": "decoder_worker_launch",
                "error_type": type(error).__name__,
                "message": str(error) or type(error).__name__,
                "worker": {
                    "status": "launch_failed",
                    "command": command,
                    "parent_acceptance": "not_available",
                },
            }
        ) from error

    launch_provenance = {
        "status": "manifest_unavailable",
        "command": command,
        "returncode": returncode,
        "parent_interrupted": parent_signal is not None,
        "parent_signal": parent_signal,
        "manifest_path": str(manifest_path.resolve()),
        "parent_acceptance": "rejected",
    }
    if not manifest_path.is_file():
        raise DecoderWorkerError(
            {
                "type": "decoder_worker_exit",
                "source": "decoder_worker",
                "returncode": returncode,
                "message": "Humanoid decoder worker exited without a manifest",
                "worker": launch_provenance,
            }
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DecoderWorkerError(
            {
                "type": "decoder_worker_protocol",
                "source": "decoder_worker_manifest",
                "returncode": returncode,
                "message": f"invalid Humanoid decoder worker manifest: {error}",
                "worker": launch_provenance,
            }
        ) from error
    if not isinstance(manifest, Mapping):
        raise DecoderWorkerError(
            {
                "type": "decoder_worker_protocol",
                "source": "decoder_worker_manifest",
                "returncode": returncode,
                "message": "Humanoid decoder worker manifest is not a mapping",
                "worker": launch_provenance,
            }
        )
    provenance = {
        **manifest,
        "command": command,
        "returncode": returncode,
        "parent_interrupted": parent_signal is not None,
        "parent_signal": parent_signal,
        "manifest_path": str(manifest_path.resolve()),
        "parent_acceptance": "pending",
    }

    def rejected_protocol(source: str, message: str) -> DecoderWorkerError:
        return DecoderWorkerError(
            {
                "type": "decoder_worker_protocol",
                "source": source,
                "message": message,
                "worker": {
                    **provenance,
                    "parent_acceptance": "rejected",
                },
            }
        )

    if (
        not _is_strict_int(manifest.get("schema_version"))
        or manifest.get("schema_version") != WORKER_SCHEMA_VERSION
        or manifest.get("phase") != "decoder_fit"
        or manifest.get("execution_profile") != profile
    ):
        raise rejected_protocol(
            "decoder_worker_manifest",
            "Humanoid decoder worker manifest contract mismatch",
        )

    if parent_signal is not None:
        sealed_failure = manifest.get("status") == "failed" and isinstance(
            manifest.get("failure"), Mapping
        )
        provenance = {
            **provenance,
            "parent_acceptance": (
                "terminal_failure_observed_unverified"
                if sealed_failure
                else "interrupted_unsealed"
            ),
        }
        failure = manifest.get("failure") if sealed_failure else {}
        raise DecoderWorkerError(
            {
                **failure,
                "type": "decoder_worker_interrupted",
                "source": "decoder_worker_signal",
                "signal": parent_signal,
                "message": (
                    "Humanoid decoder worker was interrupted by parent signal "
                    f"{parent_signal}"
                ),
                "worker": provenance,
            }
        )

    request_artifact = manifest.get("request")
    recorded_inputs = manifest.get("inputs")
    inputs_valid = recorded_inputs == input_artifacts or (
        manifest.get("status") == "failed" and recorded_inputs == {}
    )
    try:
        _read_artifact_bytes(
            request_artifact,
            label="request",
            expected_format="pickle",
            expected_path=request_path,
        )
    except (OSError, TypeError, ValueError) as error:
        raise rejected_protocol("decoder_worker_request", str(error)) from error
    if not inputs_valid:
        raise rejected_protocol(
            "decoder_worker_request",
            "Humanoid decoder worker inputs differ from the launch",
        )
    metrics_artifact = manifest.get("metrics")
    try:
        _, metrics_payload = _read_artifact_bytes(
            metrics_artifact,
            label="metrics",
            expected_format="jsonl",
            expected_path=metrics_path,
        )
        metrics_text = metrics_payload.decode("utf-8")
    except (OSError, TypeError, UnicodeDecodeError, ValueError) as error:
        raise rejected_protocol("decoder_worker_metrics", str(error)) from error
    metric_count = sum(1 for line in metrics_text.splitlines() if line)
    recorded_metric_count = manifest.get("metric_count")
    if not _is_strict_int(recorded_metric_count) or recorded_metric_count < 0:
        raise rejected_protocol(
            "decoder_worker_metrics",
            "Humanoid decoder worker metric count is not an integer",
        )
    if recorded_metric_count != metric_count:
        raise rejected_protocol(
            "decoder_worker_metrics",
            "Humanoid decoder worker metric count mismatch",
        )
    if returncode != 0 or manifest.get("status") != "complete":
        provenance = {
            **provenance,
            "parent_acceptance": "accepted_terminal_failure",
        }
        failure = manifest.get("failure")
        if not isinstance(failure, Mapping):
            failure = {
                "type": "decoder_worker_exit",
                "source": "decoder_worker",
                "returncode": returncode,
                "message": f"Humanoid decoder worker exited with code {returncode}",
            }
        raise DecoderWorkerError({**failure, "worker": provenance})

    result_artifact = manifest.get("result")
    if not isinstance(result_artifact, Mapping):
        raise rejected_protocol(
            "decoder_worker_result",
            "Humanoid decoder worker manifest has no result artifact",
        )
    try:
        _, result_payload = _read_artifact_bytes(
            result_artifact,
            label="result",
            expected_format="pickle",
            expected_path=result_path,
        )
        bundle = pickle.loads(result_payload)
    except Exception as error:
        raise rejected_protocol("decoder_worker_result", str(error)) from error
    if (
        not isinstance(bundle, Mapping)
        or bundle.get("format") != "gorl_humanoid_decoder_worker_result"
        or not _is_strict_int(bundle.get("schema_version"))
        or bundle.get("schema_version") != WORKER_SCHEMA_VERSION
    ):
        raise rejected_protocol(
            "decoder_worker_result",
            "Humanoid decoder worker returned an invalid result",
        )
    from gorl.algorithms.decoder_training import (
        DecoderFitResult,
        DecoderTrainingConfig,
    )

    fitted = bundle.get("fitted")
    decoder_config = bundle.get("decoder_config")
    if not isinstance(fitted, DecoderFitResult) or not isinstance(
        decoder_config, DecoderTrainingConfig
    ):
        raise rejected_protocol(
            "decoder_worker_result",
            "Humanoid decoder worker result types are invalid",
        )
    provenance = {**provenance, "parent_acceptance": "accepted"}
    return DecoderWorkerOutcome(
        result=fitted,
        decoder_config=decoder_config,
        provenance=provenance,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=WORKER_MODULE)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument(
        "--execution-profile",
        choices=(EXECUTION_PROFILE,),
        required=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    previous_handlers = _install_termination_handlers()
    bootstrap: Mapping[str, Any] | None = None
    try:
        try:
            bootstrap = _bootstrap()
            return execute_request(
                args.request,
                args.result,
                args.manifest,
                args.metrics,
                execution_profile=args.execution_profile,
                bootstrap=bootstrap,
            )
        except BaseException as error:
            with _ignore_signals_while_reaping_or_sealing():
                failure_scope = "bootstrap" if bootstrap is None else "execute"
                _seal_outer_failure(
                    args.request,
                    args.result,
                    args.manifest,
                    args.metrics,
                    execution_profile=args.execution_profile,
                    bootstrap=bootstrap,
                    error=error,
                    failure_type=f"decoder_worker_{failure_scope}",
                    failure_source=f"decoder_worker_{failure_scope}",
                )
            return _error_returncode(error)
    finally:
        _restore_signal_handlers(previous_handlers)


if __name__ == "__main__":
    raise SystemExit(main())
