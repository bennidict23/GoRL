"""Run fresh Humanoid FM and Diffusion branches from one teacher prefix.

The ordinary FM command produces the teacher checkpoint and dataset. This
parent process copies and seals those files, starts an independently initialized
Diffusion consumer, and accepts the pair only after both branch outputs and the
shared-prefix lineage verify.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from gorl.artifacts import atomic_write_json
from gorl.config import RunConfig, default_config_root, load_run_config
from gorl.cuda_devices import CudaDeviceSelectionError, resolve_cuda_devices
from gorl.runtime import humanoid_gorl_runtime_plan

from .shared_prefix import (
    SharedPrefixBundle,
    require_matching_teacher_contracts,
    write_shared_prefix_manifest,
)


PAIR_SCHEMA_VERSION = 1
PAIR_MANIFEST = "manifest.json"
_PAIR_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_HUMANOID_TASKS = frozenset({"HumanoidStand", "HumanoidRun"})
_SUCCESS_GROUP_EXIT_GRACE_SECONDS = 0.5


class PairError(RuntimeError):
    """Raised when a fresh Humanoid pair cannot be orchestrated safely."""


class PairDatasetQualityError(PairError):
    """Raised when a producer dataset cannot be shared with another branch."""


class PairSignalInterrupt(KeyboardInterrupt):
    """Turn an orchestrator termination signal into structured cleanup."""

    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(f"train-pair interrupted by signal {signum}")


class BranchProcess(Protocol):
    pid: int

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def terminate(self) -> None: ...


ProcessFactory = Callable[..., BranchProcess]
ProcessGroupSignal = Callable[[int, int], None]
ProcessGroupWaiter = Callable[[int, float], bool]


def _install_cleanup_signal_handlers() -> dict[int, Any]:
    interrupted = False

    def handle(signum: int, _frame: Any) -> None:
        nonlocal interrupted
        if interrupted:
            return
        interrupted = True
        raise PairSignalInterrupt(signum)

    previous: dict[int, Any] = {}
    try:
        for selected in (signal.SIGTERM, signal.SIGHUP):
            previous[selected] = signal.getsignal(selected)
            signal.signal(selected, handle)
    except BaseException:
        _restore_signal_handlers(previous)
        raise
    return previous


def _restore_signal_handlers(previous: Mapping[int, Any]) -> None:
    for selected, handler in previous.items():
        signal.signal(selected, handler)


@dataclass(frozen=True, slots=True)
class PairSpec:
    task: str
    teacher_seed: int
    fm_seed: int
    diffusion_seed: int
    gpus: tuple[int, ...]
    profile: str
    config: Path | None
    config_root: Path
    output_root: Path
    run_id: str | None
    wandb_mode: str
    wandb_project: str
    wandb_entity: str | None
    smoke: bool
    dry_run: bool


@dataclass(frozen=True, slots=True)
class BranchExpectations:
    """Parent-owned facts required to accept a completed branch."""

    pair_id: str
    pair_manifest_path: Path
    request_path: Path
    request_sha256: str
    shared_prefix_manifest_path: Path
    contract_sha256: str
    shared_prefix_id: str | None
    checkpoint_sha256: str | None
    dataset_sha256: str | None
    command: tuple[str, ...]


BranchVerifier = Callable[[str, RunConfig, Path, BranchExpectations], None]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_pair_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _atomic_copy(source: Path, destination: Path) -> Path:
    try:
        source_path = source.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise PairError(f"cannot read prefix artifact: {source}") from error
    if not source_path.is_file():
        raise PairError(f"prefix artifact is not a file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with (
            source_path.open("rb") as input_stream,
            os.fdopen(descriptor, "wb") as output_stream,
        ):
            shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _seal_prefix(
    *,
    prefix_root: Path,
    contract: Mapping[str, Any],
    checkpoint_source: Path,
    dataset_source: Path,
) -> SharedPrefixBundle:
    prefix_root.mkdir(parents=True, exist_ok=False)
    checkpoint = _atomic_copy(checkpoint_source, prefix_root / "final_state.pkl")
    dataset = _atomic_copy(dataset_source, prefix_root / "decoder_dataset.npz")
    return write_shared_prefix_manifest(
        prefix_root,
        contract=contract,
        checkpoint_path=checkpoint,
        dataset_path=dataset,
    )


def _dataset_quality_gate(
    config: RunConfig,
    bundle: SharedPrefixBundle,
) -> dict[str, Any]:
    configured = config.section("training").get("teacher_dataset_min_return_mean")
    threshold = None if configured is None else float(configured)
    enabled = threshold is not None and not config.smoke
    observed = float(bundle.dataset_stats.return_mean)
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


def _validate_spec(spec: PairSpec) -> str:
    if spec.task not in _HUMANOID_TASKS:
        raise PairError("train-pair supports only HumanoidStand and HumanoidRun")
    if len(spec.gpus) not in {1, 2}:
        raise PairError("--gpus requires exactly one or two GPU indices")
    if len(set(spec.gpus)) != len(spec.gpus):
        raise PairError("--gpus must contain distinct GPU indices")
    if spec.smoke and spec.dry_run:
        raise PairError("--smoke and --dry-run are mutually exclusive")
    selected_id = spec.run_id or _default_pair_id()
    if _PAIR_ID.fullmatch(selected_id) is None:
        raise PairError(
            "run_id must start with an alphanumeric character and contain only "
            "letters, numbers, '.', '_' or '-' (maximum 128 characters)"
        )
    return selected_id


def _load_branch_config(
    spec: PairSpec,
    *,
    method: str,
    seed: int,
    output_root: Path,
) -> RunConfig:
    return load_run_config(
        task=spec.task,
        method=method,
        seed=seed,
        teacher_seed=spec.teacher_seed,
        decoder_seed=seed,
        encoder_seed=seed,
        eval_seed=seed,
        profile=spec.profile,
        config_root=spec.config_root,
        overlay=spec.config,
        output_root=output_root,
        smoke=spec.smoke,
        wandb_mode=spec.wandb_mode,
        wandb_project=spec.wandb_project,
        wandb_entity=spec.wandb_entity,
    )


def _require_dedicated_teacher_process(configs: Sequence[RunConfig]) -> None:
    invalid = [
        config.method
        for config in configs
        if not bool(humanoid_gorl_runtime_plan(config)["teacher_process_required"])
    ]
    if invalid:
        raise PairError(
            "train-pair requires the dedicated Humanoid teacher subprocess; "
            "the overlay collapses teacher and downstream runtime for "
            + ", ".join(invalid)
        )


def _public_train_command(
    spec: PairSpec,
    *,
    method: str,
    seed: int,
    output_root: Path,
    run_id: str,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "gorl",
        "train",
        "--task",
        spec.task,
        "--method",
        method,
        "--seed",
        str(seed),
        "--teacher-seed",
        str(spec.teacher_seed),
        "--decoder-seed",
        str(seed),
        "--encoder-seed",
        str(seed),
        "--eval-seed",
        str(seed),
        "--profile",
        spec.profile,
        "--config-root",
        str(spec.config_root),
        "--output-root",
        str(output_root),
        "--run-id",
        run_id,
        "--wandb-mode",
        spec.wandb_mode,
        "--wandb-project",
        spec.wandb_project,
    ]
    if spec.config is not None:
        command.extend(("--config", str(spec.config)))
    if spec.wandb_entity is not None:
        command.extend(("--wandb-entity", spec.wandb_entity))
    if spec.smoke:
        command.append("--smoke")
    return command


def _worker_request(
    spec: PairSpec,
    *,
    pair_id: str,
    diffusion_config: RunConfig,
    contract_sha256: str,
    output_root: Path,
    pair_manifest: Path,
    prefix_manifest: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "pair_id": pair_id,
        "task": spec.task,
        "seed": spec.diffusion_seed,
        "teacher_seed": spec.teacher_seed,
        "decoder_seed": spec.diffusion_seed,
        "encoder_seed": spec.diffusion_seed,
        "eval_seed": spec.diffusion_seed,
        "profile": diffusion_config.profile,
        "config": None if spec.config is None else str(spec.config),
        "config_root": str(spec.config_root),
        "output_root": str(output_root),
        "run_id": "diffusion",
        "wandb_mode": spec.wandb_mode,
        "wandb_project": spec.wandb_project,
        "wandb_entity": spec.wandb_entity,
        "smoke": spec.smoke,
        "pair_manifest": str(pair_manifest),
        "shared_prefix_manifest": str(prefix_manifest),
        "expected_contract_sha256": contract_sha256,
        "expected_shared_prefix_id": None,
    }


def _worker_command(request_path: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "gorl._workers.humanoid_shared_branch",
        "--request",
        str(request_path),
    ]


def _resolve_cuda_devices(
    gpu_ordinals: Sequence[int],
    inherited: str | None,
) -> tuple[str, ...]:
    try:
        return resolve_cuda_devices(gpu_ordinals, inherited).resolved
    except CudaDeviceSelectionError as error:
        raise PairError(str(error)) from error


def _branch_environment(cuda_visible_device: str) -> dict[str, str]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = cuda_visible_device
    return environment


@contextmanager
def _parent_cpu_environment():
    """Prevent checkpoint deserialization from opening a CUDA context."""

    # Brax checkpoints can retain JAX reconstruction helpers after host copy.
    names = (
        "CUDA_VISIBLE_DEVICES",
        "JAX_PLATFORM_NAME",
        "JAX_PLATFORMS",
        "JAX_SKIP_CUDA_CONSTRAINTS_CHECK",
    )
    previous = {name: os.environ.get(name) for name in names}
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ["JAX_PLATFORMS"] = "cpu"
    # The CUDA plugin probes cuInit even when JAX is restricted to CPU.
    os.environ["JAX_SKIP_CUDA_CONSTRAINTS_CHECK"] = "1"
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _branch_record(
    *,
    role: str,
    method: str,
    seed: int,
    gpu_ordinal: int,
    cuda_visible_device: str,
    command: Sequence[str],
    run_dir: Path,
    status: str,
) -> dict[str, Any]:
    return {
        "role": role,
        "method": method,
        "seed": seed,
        "gpu": gpu_ordinal,
        "cuda_visible_device": cuda_visible_device,
        "status": status,
        "exit_code": None,
        "pid": None,
        "command": {
            "argv": list(command),
            "shell": shlex.join(command),
        },
        "run_dir": str(run_dir),
    }


def _process_exit(record: dict[str, Any], exit_code: int) -> None:
    record["exit_code"] = exit_code
    record["status"] = "complete" if exit_code == 0 else "failed"
    record["completed_at"] = _utc_now()


def _process_failure(record: dict[str, Any], error: BaseException) -> None:
    record["status"] = "failed"
    record["exit_code"] = None
    record["completed_at"] = _utc_now()
    record["error"] = {
        "type": type(error).__name__,
        "message": str(error) or type(error).__name__,
    }


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PairError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise PairError(f"{label} root is not an object: {path}")
    return value


def _require_equal(actual: Any, expected: Any, *, label: str) -> None:
    if actual != expected:
        raise PairError(f"branch output {label} does not match the pair plan")


def _output_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PairError(f"branch output {label} is not an object")
    return value


def _verify_artifact(
    value: Any,
    *,
    expected_path: Path,
    expected_sha256: str | None,
    label: str,
) -> None:
    if expected_sha256 is None:
        raise PairError(f"branch output {label} has no sealed hash to verify against")
    artifact = _output_mapping(value, label=label)
    _require_equal(artifact.get("path"), str(expected_path), label=f"{label} path")
    _require_equal(artifact.get("sha256"), expected_sha256, label=f"{label} hash")
    try:
        actual_sha256 = _sha256(expected_path)
    except OSError as error:
        raise PairError(
            f"cannot hash branch output {label}: {expected_path}"
        ) from error
    _require_equal(actual_sha256, expected_sha256, label=f"{label} file hash")


def _verify_shared_prefix_record(
    value: Any,
    *,
    expectations: BranchExpectations,
    label: str,
) -> None:
    if expectations.shared_prefix_id is None:
        raise PairError("diffusion verification requires a sealed shared-prefix ID")
    shared = _output_mapping(value, label=label)
    _require_equal(
        shared.get("shared_prefix_id"),
        expectations.shared_prefix_id,
        label=f"{label} ID",
    )
    _require_equal(
        shared.get("manifest_path"),
        str(expectations.shared_prefix_manifest_path),
        label=f"{label} manifest",
    )
    _require_equal(
        shared.get("contract_sha256"),
        expectations.contract_sha256,
        label=f"{label} contract hash",
    )
    checkpoint_path = (
        expectations.shared_prefix_manifest_path.parent / "final_state.pkl"
    )
    dataset_path = (
        expectations.shared_prefix_manifest_path.parent / "decoder_dataset.npz"
    )
    _verify_artifact(
        shared.get("checkpoint"),
        expected_path=checkpoint_path,
        expected_sha256=expectations.checkpoint_sha256,
        label=f"{label} checkpoint",
    )
    _verify_artifact(
        shared.get("dataset"),
        expected_path=dataset_path,
        expected_sha256=expectations.dataset_sha256,
        label=f"{label} dataset",
    )


def _verify_branch_outputs(
    branch: str,
    config: RunConfig,
    run_dir: Path,
    expectations: BranchExpectations,
) -> None:
    manifest = _read_json_object(run_dir / "manifest.json", label=f"{branch} manifest")
    summary = _read_json_object(run_dir / "summary.json", label=f"{branch} summary")
    resolved_path = run_dir / "resolved_config.json"
    resolved = _read_json_object(resolved_path, label=f"{branch} resolved config")
    _require_equal(manifest.get("status"), "complete", label="manifest status")
    _require_equal(summary.get("status"), "complete", label="summary status")
    for record, label in ((manifest, "manifest"), (summary, "summary")):
        _require_equal(record.get("task"), config.task, label=f"{label} task")
        _require_equal(record.get("seed"), config.seed, label=f"{label} seed")
    _require_equal(
        manifest.get("algorithm", {}).get("name"),
        config.method,
        label="manifest method",
    )
    _require_equal(summary.get("method"), config.method, label="summary method")
    _require_equal(
        manifest.get("seeds"),
        {
            "teacher": config.teacher_seed,
            "decoder": config.decoder_seed,
            "encoder": config.encoder_seed,
            "evaluation": config.eval_seed,
        },
        label="manifest seeds",
    )
    _require_equal(resolved, config.to_dict(), label="resolved config")
    _require_equal(
        manifest.get("config", {}).get("values"),
        resolved,
        label="manifest config",
    )
    _require_equal(
        manifest.get("config", {}).get("sha256"),
        _sha256(resolved_path),
        label="resolved config hash",
    )
    _require_equal(
        manifest.get("command", {}).get("argv"),
        list(expectations.command),
        label=f"{branch} command",
    )
    teacher = _output_mapping(summary.get("teacher"), label=f"{branch} teacher")
    if branch == "fm":
        _require_equal(
            summary.get("fresh_from_scratch"), True, label="FM fresh declaration"
        )
        _require_equal(
            summary.get("execution_mode"),
            "ordinary_fresh_train",
            label="FM execution mode",
        )
        _require_equal(
            teacher.get("initialization"),
            "from_scratch",
            label="FM teacher initialization",
        )
        if (
            manifest.get("shared_prefix") is not None
            or summary.get("shared_prefix") is not None
        ):
            raise PairError("FM producer unexpectedly consumed a shared prefix")
        checkpoints = _output_mapping(
            teacher.get("checkpoints"), label="FM teacher checkpoints"
        )
        _verify_artifact(
            checkpoints.get("final"),
            expected_path=run_dir / "teacher" / "final_state.pkl",
            expected_sha256=expectations.checkpoint_sha256,
            label="FM source checkpoint",
        )
        _verify_artifact(
            teacher.get("collection"),
            expected_path=run_dir / "teacher" / "decoder_dataset.npz",
            expected_sha256=expectations.dataset_sha256,
            label="FM source dataset",
        )
        return
    if branch != "diffusion":
        raise PairError(f"unknown pair branch: {branch}")
    if expectations.shared_prefix_id is None:
        raise PairError("diffusion output verification requires a shared-prefix ID")
    execution = _output_mapping(
        manifest.get("branch_execution"), label="diffusion branch execution"
    )
    expected_execution = {
        "mode": "private_shared_prefix_consumer",
        "ordinary_fresh_train": False,
        "pair_fresh_from_scratch": True,
        "attestation": "verified",
        "pair_id": expectations.pair_id,
        "pair_manifest": str(expectations.pair_manifest_path),
        "request_path": str(expectations.request_path),
        "request_sha256": expectations.request_sha256,
        "shared_prefix_manifest": str(expectations.shared_prefix_manifest_path),
        "shared_prefix_id": expectations.shared_prefix_id,
        "contract_sha256": expectations.contract_sha256,
    }
    for key, expected in expected_execution.items():
        _require_equal(
            execution.get(key), expected, label=f"diffusion attestation {key}"
        )
    try:
        actual_request_sha256 = _sha256(expectations.request_path)
    except OSError as error:
        raise PairError("cannot hash the attested diffusion request") from error
    _require_equal(
        actual_request_sha256,
        expectations.request_sha256,
        label="diffusion request file hash",
    )
    _require_equal(
        summary.get("fresh_from_scratch"),
        False,
        label="diffusion fresh declaration",
    )
    _require_equal(
        summary.get("execution_mode"),
        "shared_prefix_branch",
        label="diffusion summary execution mode",
    )
    _verify_shared_prefix_record(
        manifest.get("shared_prefix"),
        expectations=expectations,
        label="diffusion manifest shared prefix",
    )
    _verify_shared_prefix_record(
        summary.get("shared_prefix"),
        expectations=expectations,
        label="diffusion summary shared prefix",
    )
    _require_equal(
        teacher.get("initialization"),
        "shared_prefix",
        label="diffusion teacher initialization",
    )
    checkpoints = _output_mapping(
        teacher.get("checkpoints"), label="diffusion teacher checkpoints"
    )
    prefix_root = expectations.shared_prefix_manifest_path.parent
    _verify_artifact(
        checkpoints.get("final"),
        expected_path=prefix_root / "final_state.pkl",
        expected_sha256=expectations.checkpoint_sha256,
        label="diffusion teacher checkpoint",
    )
    _verify_artifact(
        teacher.get("collection"),
        expected_path=prefix_root / "decoder_dataset.npz",
        expected_sha256=expectations.dataset_sha256,
        label="diffusion teacher dataset",
    )


def _process_verification_failure(record: dict[str, Any], error: BaseException) -> None:
    record["status"] = "failed_verification"
    record["verification"] = {
        "status": "failed",
        "type": type(error).__name__,
        "message": str(error) or type(error).__name__,
    }


def _process_group_exists(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_group(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while _process_group_exists(pid):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.1, remaining))
    return True


def _cleanup_process_group(
    process: BranchProcess | None,
    record: dict[str, Any],
    *,
    signal_group: ProcessGroupSignal,
    wait_group: ProcessGroupWaiter,
    timeout: float,
    force: bool,
) -> None:
    if process is None:
        return
    cleanup: dict[str, Any] = {
        "status": "checking",
        "residual_detected": False,
        "group_absent": False,
        "term_sent": False,
        "kill_sent": False,
        "observed_parent_exit_code": None,
    }
    record["cleanup"] = cleanup
    try:
        exit_code = process.poll()
        cleanup["observed_parent_exit_code"] = exit_code
        successful_parent = (
            exit_code is not None and not force and record.get("status") == "complete"
        )
        if successful_parent:
            grace = min(timeout, _SUCCESS_GROUP_EXIT_GRACE_SECONDS)
            if wait_group(process.pid, grace):
                cleanup.update({"status": "complete", "group_absent": True})
                return
            cleanup["residual_detected"] = True
        if exit_code is not None and not force:
            if not successful_parent and record["status"] in {
                "running",
                "waiting",
                "waiting_for_prefix",
            }:
                record.update(
                    {
                        "status": "failed_orchestration",
                        "exit_code": exit_code,
                        "completed_at": _utc_now(),
                    }
                )
            if not successful_parent:
                cleanup["status"] = "not_required"
                return
        try:
            signal_group(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            cleanup.update(
                {
                    "status": (
                        "residual_exited_before_term"
                        if successful_parent
                        else "complete"
                    ),
                    "group_absent": True,
                }
            )
            if successful_parent:
                record.update(
                    {
                        "status": "failed_residual_process_group",
                        "completed_at": _utc_now(),
                    }
                )
            elif record["status"] in {"running", "waiting", "waiting_for_prefix"}:
                record.update(
                    {
                        "status": "failed_orchestration",
                        "exit_code": exit_code,
                        "completed_at": _utc_now(),
                    }
                )
            return
        cleanup["term_sent"] = True
        parent_timed_out = False
        if exit_code is None:
            try:
                exit_code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                parent_timed_out = True
            cleanup["observed_parent_exit_code"] = exit_code
        group_exited = (
            wait_group(process.pid, timeout) if not parent_timed_out else False
        )
        if parent_timed_out or not group_exited:
            try:
                signal_group(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                group_exited = True
            else:
                cleanup["kill_sent"] = True
            if exit_code is None:
                exit_code = process.wait(timeout=timeout)
                cleanup["observed_parent_exit_code"] = exit_code
            if not group_exited and not wait_group(process.pid, timeout):
                raise PairError(f"process group {process.pid} survived SIGKILL cleanup")
        cleanup.update(
            {
                "status": ("residual_terminated" if successful_parent else "complete"),
                "group_absent": True,
                "observed_parent_exit_code": exit_code,
            }
        )
        if successful_parent:
            record.update(
                {
                    "status": "failed_residual_process_group",
                    "completed_at": _utc_now(),
                }
            )
        elif record["status"] in {"running", "waiting", "waiting_for_prefix"}:
            record.update(
                {
                    "status": "terminated_orchestration",
                    "exit_code": exit_code,
                    "completed_at": _utc_now(),
                }
            )
    except Exception as error:
        record.update(
            {
                "status": "cleanup_failed",
                "completed_at": _utc_now(),
                "cleanup_error": {
                    "type": type(error).__name__,
                    "message": str(error) or type(error).__name__,
                },
            }
        )
        cleanup.update(
            {
                "status": "failed",
                "error": dict(record["cleanup_error"]),
            }
        )


def _terminalize_pending(manifest: dict[str, Any], error: BaseException) -> None:
    failure = {
        "type": type(error).__name__,
        "message": str(error) or type(error).__name__,
    }
    prefix = manifest["shared_prefix"]
    if prefix["status"] in {"waiting_for_producer", "planned"}:
        prefix.update(
            {
                "status": "failed_orchestration",
                "completed_at": _utc_now(),
                "error": failure,
            }
        )
    branches = manifest["branches"]
    for name, branch in branches.items():
        if branch["status"] in {"waiting", "waiting_for_prefix", "planned"}:
            branch.update(
                {
                    "status": (
                        "not_started_orchestration"
                        if name == "fm"
                        else "blocked_orchestration"
                    ),
                    "completed_at": _utc_now(),
                }
            )
        elif branch["status"] == "running":
            branch.update(
                {"status": "failed_orchestration", "completed_at": _utc_now()}
            )


def _write_resolved_configs(
    pair_root: Path,
    fm_config: RunConfig,
    diffusion_config: RunConfig,
) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    for name, config in (("fm", fm_config), ("diffusion", diffusion_config)):
        path = pair_root / f"resolved_{name}_config.json"
        atomic_write_json(path, config.to_dict())
        records[name] = {
            "path": path.name,
            "sha256": _sha256(path),
        }
    return records


def run_pair(
    spec: PairSpec,
    *,
    command: Sequence[str],
    process_factory: ProcessFactory | None = None,
    branch_verifier: BranchVerifier | None = None,
    signal_process_group: ProcessGroupSignal | None = None,
    wait_process_group: ProcessGroupWaiter | None = None,
    cleanup_timeout: float = 10.0,
    sleep: Callable[[float], None] = time.sleep,
    poll_interval: float = 0.25,
) -> int:
    """Run one immutable fresh pair and return its aggregate exit code."""

    if cleanup_timeout <= 0:
        raise PairError("cleanup_timeout must be positive")
    if poll_interval <= 0:
        raise PairError("poll_interval must be positive")
    selected_id = _validate_spec(spec)
    inherited_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    cuda_devices = _resolve_cuda_devices(spec.gpus, inherited_cuda_devices)
    output_root = spec.output_root.expanduser().resolve()
    pair_root = output_root / spec.task / "gorl_pair" / selected_id
    branch_output_root = pair_root / "branches"
    fm_config = _load_branch_config(
        spec,
        method="gorl_fm",
        seed=spec.fm_seed,
        output_root=branch_output_root,
    )
    diffusion_config = _load_branch_config(
        spec,
        method="gorl_diffusion",
        seed=spec.diffusion_seed,
        output_root=branch_output_root,
    )
    _require_dedicated_teacher_process((fm_config, diffusion_config))
    contract = require_matching_teacher_contracts((fm_config, diffusion_config))
    contract_sha256 = _sha256_json(contract)

    try:
        pair_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise PairError(f"pair directory already exists: {pair_root}") from error

    pair_manifest_path = pair_root / PAIR_MANIFEST
    prefix_root = pair_root / "shared_prefix"
    prefix_manifest = prefix_root / "shared_prefix.json"
    request_path = pair_root / "diffusion_worker_request.json"
    fm_run_dir = (
        branch_output_root / spec.task / "gorl_fm" / f"seed-{spec.fm_seed}" / "fm"
    )
    diffusion_run_dir = (
        branch_output_root
        / spec.task
        / "gorl_diffusion"
        / f"seed-{spec.diffusion_seed}"
        / "diffusion"
    )
    checkpoint_source = fm_run_dir / "teacher" / "final_state.pkl"
    dataset_source = fm_run_dir / "teacher" / "decoder_dataset.npz"
    fm_command = _public_train_command(
        spec,
        method="gorl_fm",
        seed=spec.fm_seed,
        output_root=branch_output_root,
        run_id="fm",
    )
    worker_request = _worker_request(
        spec,
        pair_id=selected_id,
        diffusion_config=diffusion_config,
        contract_sha256=contract_sha256,
        output_root=branch_output_root,
        pair_manifest=pair_manifest_path,
        prefix_manifest=prefix_manifest,
    )
    diffusion_command = _worker_command(request_path)
    config_records = _write_resolved_configs(
        pair_root,
        fm_config,
        diffusion_config,
    )
    atomic_write_json(request_path, worker_request)
    request_sha256 = _sha256(request_path)

    branches = {
        "fm": _branch_record(
            role="fresh_prefix_producer",
            method="gorl_fm",
            seed=spec.fm_seed,
            gpu_ordinal=spec.gpus[0],
            cuda_visible_device=cuda_devices[0],
            command=fm_command,
            run_dir=fm_run_dir,
            status="planned" if spec.dry_run else "waiting",
        ),
        "diffusion": _branch_record(
            role="private_shared_prefix_consumer",
            method="gorl_diffusion",
            seed=spec.diffusion_seed,
            gpu_ordinal=spec.gpus[-1],
            cuda_visible_device=cuda_devices[-1],
            command=diffusion_command,
            run_dir=diffusion_run_dir,
            status="planned" if spec.dry_run else "waiting_for_prefix",
        ),
    }
    manifest: dict[str, Any] = {
        "schema_version": PAIR_SCHEMA_VERSION,
        "kind": "fresh_humanoid_gorl_pair",
        "pair_id": selected_id,
        "status": "dry_run" if spec.dry_run else "initialized",
        "created_at": _utc_now(),
        "command": {
            "argv": list(command),
            "shell": shlex.join(command),
            "working_directory": str(Path.cwd().resolve()),
        },
        "task": spec.task,
        "profile": fm_config.profile,
        "fresh_from_scratch": True,
        "accepts_external_shared_prefix": False,
        "inputs": {
            "requested_profile": spec.profile,
            "config_root": str(spec.config_root.resolve()),
            "overlay": None if spec.config is None else str(spec.config.resolve()),
            "branch_output_root": str(branch_output_root),
            "smoke": spec.smoke,
            "wandb": {
                "mode": spec.wandb_mode,
                "project": spec.wandb_project,
                "entity": spec.wandb_entity,
            },
        },
        "execution": {
            "mode": "parallel" if len(spec.gpus) == 2 else "sequential",
            "gpus": list(spec.gpus),
            "inherited_cuda_visible_devices": inherited_cuda_devices,
            "resolved_cuda_visible_devices": list(cuda_devices),
        },
        "seeds": {
            "teacher": spec.teacher_seed,
            "fm": spec.fm_seed,
            "diffusion": spec.diffusion_seed,
        },
        "resolved_configs": config_records,
        "teacher_contract": {
            "sha256": contract_sha256,
            "values": contract,
        },
        "diffusion_request": {
            "path": str(request_path),
            "sha256": request_sha256,
        },
        "shared_prefix": {
            "status": "planned" if spec.dry_run else "waiting_for_producer",
            "root": str(prefix_root),
            "manifest": str(prefix_manifest),
            "shared_prefix_id": None,
            "expected_shared_prefix_id": None,
        },
        "branches": branches,
    }
    if spec.dry_run:
        manifest["completed_at"] = _utc_now()
        atomic_write_json(pair_manifest_path, manifest)
        print(f"pair directory: {pair_root}")
        print("dry run: no branch subprocesses were started")
        return 0

    atomic_write_json(pair_manifest_path, manifest)
    print(f"pair directory: {pair_root}")
    factory = process_factory or subprocess.Popen
    verify_branch = branch_verifier or _verify_branch_outputs
    signal_group = signal_process_group or os.killpg
    wait_group = wait_process_group or _wait_for_process_group
    fm_process: BranchProcess | None = None
    diffusion_process: BranchProcess | None = None
    bundle: SharedPrefixBundle | None = None
    fm_exit: int | None = None

    def write_manifest() -> None:
        atomic_write_json(pair_manifest_path, manifest)

    def spawn_branch(command: Sequence[str], cuda_visible_device: str) -> BranchProcess:
        return factory(
            list(command),
            cwd=str(Path.cwd()),
            env=_branch_environment(cuda_visible_device),
            start_new_session=True,
        )

    def mark_started(name: str, process: BranchProcess) -> None:
        branches[name].update(
            {"status": "running", "pid": process.pid, "started_at": _utc_now()}
        )
        manifest["status"] = "running"
        write_manifest()

    def finish_branch(
        name: str,
        process: BranchProcess,
        *,
        known_exit: int | None = None,
    ) -> int:
        exit_code = process.wait() if known_exit is None else known_exit
        _process_exit(branches[name], exit_code)
        write_manifest()
        return exit_code

    def branch_expectations(name: str) -> BranchExpectations:
        command_values = branches[name].get("command", {}).get("argv")
        if not isinstance(command_values, list) or not all(
            isinstance(value, str) for value in command_values
        ):
            raise PairError(f"invalid parent-owned command for {name} branch")
        return BranchExpectations(
            pair_id=selected_id,
            pair_manifest_path=pair_manifest_path,
            request_path=request_path,
            request_sha256=str(manifest["diffusion_request"]["sha256"]),
            shared_prefix_manifest_path=prefix_manifest,
            contract_sha256=contract_sha256,
            shared_prefix_id=(None if bundle is None else bundle.shared_prefix_id),
            checkpoint_sha256=(None if bundle is None else bundle.checkpoint.sha256),
            dataset_sha256=(None if bundle is None else bundle.dataset.sha256),
            command=tuple(command_values),
        )

    def verify_completed_branch(name: str, config: RunConfig) -> None:
        if branches[name]["exit_code"] != 0:
            return
        try:
            verify_branch(
                name,
                config,
                Path(branches[name]["run_dir"]),
                branch_expectations(name),
            )
        except Exception as error:
            _process_verification_failure(branches[name], error)
        else:
            branches[name]["verification"] = {"status": "complete"}
        write_manifest()

    def seal_if_ready() -> SharedPrefixBundle | None:
        if not checkpoint_source.is_file() or not dataset_source.is_file():
            return None
        with _parent_cpu_environment():
            sealed = _seal_prefix(
                prefix_root=prefix_root,
                contract=contract,
                checkpoint_source=checkpoint_source,
                dataset_source=dataset_source,
            )
        if sealed.contract_sha256 != contract_sha256:
            raise PairError("sealed shared-prefix contract changed unexpectedly")
        quality_gate = _dataset_quality_gate(fm_config, sealed)
        manifest["shared_prefix"]["dataset_quality_gate"] = quality_gate
        if quality_gate["enabled"] and not quality_gate["passed"]:
            raise PairDatasetQualityError(
                "FM producer dataset return mean is below the configured minimum: "
                f"{quality_gate['observed']} < {quality_gate['threshold']}"
            )
        worker_request["expected_shared_prefix_id"] = sealed.shared_prefix_id
        atomic_write_json(request_path, worker_request)
        manifest["diffusion_request"]["sha256"] = _sha256(request_path)
        manifest["shared_prefix"] = {
            "status": "sealed",
            "sealed_at": _utc_now(),
            "root": str(prefix_root),
            "manifest": str(sealed.manifest_path),
            "shared_prefix_id": sealed.shared_prefix_id,
            "expected_shared_prefix_id": sealed.shared_prefix_id,
            "contract_sha256": sealed.contract_sha256,
            "checkpoint_sha256": sealed.checkpoint.sha256,
            "dataset_sha256": sealed.dataset.sha256,
        }
        write_manifest()
        return sealed

    def try_seal_prefix() -> SharedPrefixBundle | None:
        try:
            return seal_if_ready()
        except Exception as error:
            manifest["shared_prefix"]["status"] = "failed"
            manifest["shared_prefix"]["completed_at"] = _utc_now()
            manifest["shared_prefix"]["error"] = {
                "type": type(error).__name__,
                "message": str(error) or type(error).__name__,
            }
            write_manifest()
            return None

    terminal_error: BaseException | None = None
    previous_signal_handlers: dict[int, Any] = {}
    cleanup_finished = False

    def cleanup_branches() -> None:
        nonlocal cleanup_finished
        if cleanup_finished:
            return
        _cleanup_process_group(
            diffusion_process,
            branches["diffusion"],
            signal_group=signal_group,
            wait_group=wait_group,
            timeout=cleanup_timeout,
            force=(
                terminal_error is not None
                or branches["diffusion"]["status"] != "complete"
            ),
        )
        _cleanup_process_group(
            fm_process,
            branches["fm"],
            signal_group=signal_group,
            wait_group=wait_group,
            timeout=cleanup_timeout,
            force=(
                terminal_error is not None or branches["fm"]["status"] != "complete"
            ),
        )
        cleanup_finished = True

    try:
        previous_signal_handlers = _install_cleanup_signal_handlers()
        try:
            fm_process = spawn_branch(fm_command, cuda_devices[0])
        except Exception as error:
            _process_failure(branches["fm"], error)
            branches["diffusion"].update(
                {
                    "status": "blocked_prefix_unavailable",
                    "completed_at": _utc_now(),
                }
            )
            manifest["shared_prefix"]["status"] = "failed"
            manifest["shared_prefix"]["completed_at"] = _utc_now()
            manifest["shared_prefix"]["error"] = {
                "type": type(error).__name__,
                "message": str(error) or type(error).__name__,
            }
            manifest.update(
                {"status": "failed", "exit_code": 1, "completed_at": _utc_now()}
            )
            write_manifest()
            return 1
        mark_started("fm", fm_process)

        if len(spec.gpus) == 2:
            while bundle is None and manifest["shared_prefix"]["status"] != "failed":
                bundle = try_seal_prefix()
                if bundle is not None:
                    break
                fm_exit = fm_process.poll()
                if fm_exit is not None:
                    break
                sleep(poll_interval)
        else:
            fm_exit = finish_branch("fm", fm_process)
            bundle = try_seal_prefix()

        if bundle is None and manifest["shared_prefix"]["status"] != "failed":
            bundle = try_seal_prefix()
            if bundle is None and manifest["shared_prefix"]["status"] != "failed":
                manifest["shared_prefix"]["status"] = "failed"
                manifest["shared_prefix"]["completed_at"] = _utc_now()
                manifest["shared_prefix"]["error"] = {
                    "type": "prefix_artifacts_missing",
                    "message": "FM producer exited before the teacher prefix was complete",
                }

        if bundle is not None:
            try:
                diffusion_process = spawn_branch(
                    diffusion_command,
                    cuda_devices[-1],
                )
            except Exception as error:
                _process_failure(branches["diffusion"], error)
                write_manifest()
            else:
                mark_started("diffusion", diffusion_process)
        else:
            branches["diffusion"].update(
                {
                    "status": "blocked_prefix_unavailable",
                    "completed_at": _utc_now(),
                }
            )
            write_manifest()

        if fm_exit is None:
            fm_exit = finish_branch("fm", fm_process)
        elif branches["fm"]["status"] == "running":
            finish_branch(
                "fm",
                fm_process,
                known_exit=fm_exit,
            )
        verify_completed_branch("fm", fm_config)

        if diffusion_process is not None:
            finish_branch("diffusion", diffusion_process)
            verify_completed_branch("diffusion", diffusion_config)

        cleanup_branches()
        success = (
            all(branch["status"] == "complete" for branch in branches.values())
            and manifest["shared_prefix"]["status"] == "sealed"
        )
        exit_code = 0 if success else 1
        manifest.update(
            {
                "status": "complete" if success else "failed",
                "exit_code": exit_code,
                "completed_at": _utc_now(),
            }
        )
        write_manifest()
        return exit_code
    except BaseException as error:
        terminal_error = error
        manifest.update(
            {
                "status": "failed",
                "exit_code": 1,
                "completed_at": _utc_now(),
                "error": {
                    "type": type(error).__name__,
                    "message": str(error) or type(error).__name__,
                },
            }
        )
        raise
    finally:
        cleanup_branches()
        if terminal_error is not None:
            _terminalize_pending(manifest, terminal_error)
        finalization_error: BaseException | None = None
        try:
            write_manifest()
        except BaseException as error:
            if terminal_error is None:
                finalization_error = error
        try:
            _restore_signal_handlers(previous_signal_handlers)
        except BaseException as error:
            if terminal_error is None and finalization_error is None:
                finalization_error = error
        if finalization_error is not None:
            raise finalization_error


def run_from_namespace(
    args: argparse.Namespace,
    *,
    command: Sequence[str],
    process_factory: ProcessFactory | None = None,
    branch_verifier: BranchVerifier | None = None,
    signal_process_group: ProcessGroupSignal | None = None,
    wait_process_group: ProcessGroupWaiter | None = None,
    cleanup_timeout: float = 10.0,
    sleep: Callable[[float], None] = time.sleep,
    poll_interval: float = 0.25,
) -> int:
    """Resolve a public CLI namespace and run its fresh pair."""

    config_root = (
        Path(args.config_root).expanduser().resolve()
        if args.config_root is not None
        else default_config_root()
    )
    spec = PairSpec(
        task=args.task,
        teacher_seed=args.teacher_seed,
        fm_seed=args.fm_seed,
        diffusion_seed=args.diffusion_seed,
        gpus=tuple(args.gpus),
        profile=args.profile,
        config=(
            None if args.config is None else Path(args.config).expanduser().resolve()
        ),
        config_root=config_root,
        output_root=Path(args.output_root),
        run_id=args.run_id,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        smoke=args.smoke,
        dry_run=args.dry_run,
    )
    return run_pair(
        spec,
        command=command,
        process_factory=process_factory,
        branch_verifier=branch_verifier,
        signal_process_group=signal_process_group,
        wait_process_group=wait_process_group,
        cleanup_timeout=cleanup_timeout,
        sleep=sleep,
        poll_interval=poll_interval,
    )
