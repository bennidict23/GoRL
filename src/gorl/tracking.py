from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


EVAL_FIELDS = (
    "task",
    "method",
    "seed",
    "stage",
    "index",
    "local_env_steps",
    "env_steps",
    "benchmark_env_steps",
    "compute_actual_env_steps",
    "plot_env_steps",
    "return_mean",
    "return_std",
)

FAILURE_FIELDS = (
    "task",
    "method",
    "seed",
    "stage",
    "local_env_steps",
    "env_steps",
    "benchmark_env_steps",
    "compute_actual_env_steps",
    "plot_env_steps",
    "error_type",
    "message",
)

TRACKING_WARNING_FIELDS = (
    "component",
    "operation",
    "error_type",
    "message",
)

_RECEIPT_COORDINATE_FIELDS = (
    "task",
    "method",
    "seed",
    "stage",
    "index",
    "local_env_steps",
    "env_steps",
    "benchmark_env_steps",
    "compute_actual_env_steps",
    "plot_env_steps",
)


@dataclass(frozen=True)
class EvalEvent:
    task: str
    method: str
    seed: int
    stage: int
    index: int
    local_env_steps: int
    env_steps: int
    benchmark_env_steps: float
    compute_actual_env_steps: int
    plot_env_steps: float
    return_mean: float
    return_std: float

    def __post_init__(self) -> None:
        if not self.task or not self.method:
            raise ValueError("task and method must be non-empty")
        for name in (
            "seed",
            "stage",
            "index",
            "local_env_steps",
            "env_steps",
            "compute_actual_env_steps",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in (
            "benchmark_env_steps",
            "plot_env_steps",
            "return_mean",
            "return_std",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.return_std < 0:
            raise ValueError("return_std must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvalEvent":
        missing = set(EVAL_FIELDS).difference(value)
        extra = set(value).difference(EVAL_FIELDS)
        if missing or extra:
            raise ValueError(
                f"invalid eval fields: missing={sorted(missing)}, extra={sorted(extra)}"
            )
        return cls(**{name: value[name] for name in EVAL_FIELDS})


@dataclass(frozen=True)
class FailureEvent:
    task: str
    method: str
    seed: int
    stage: int
    local_env_steps: int
    env_steps: int
    benchmark_env_steps: float
    compute_actual_env_steps: int
    plot_env_steps: float
    error_type: str
    message: str

    def __post_init__(self) -> None:
        if not self.task or not self.method:
            raise ValueError("task and method must be non-empty")
        if not self.error_type or not self.message:
            raise ValueError("error_type and message must be non-empty")
        for name in (
            "seed",
            "stage",
            "local_env_steps",
            "env_steps",
            "compute_actual_env_steps",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in (
            "benchmark_env_steps",
            "plot_env_steps",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FailureEvent":
        missing = set(FAILURE_FIELDS).difference(value)
        extra = set(value).difference(FAILURE_FIELDS)
        if missing or extra:
            raise ValueError(
                "invalid failure fields: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        return cls(**{name: value[name] for name in FAILURE_FIELDS})


@dataclass(frozen=True)
class TrackingWarning:
    component: str
    operation: str
    error_type: str
    message: str

    def __post_init__(self) -> None:
        for name in TRACKING_WARNING_FIELDS:
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrackingWarning":
        missing = set(TRACKING_WARNING_FIELDS).difference(value)
        extra = set(value).difference(TRACKING_WARNING_FIELDS)
        if missing or extra:
            raise ValueError(
                "invalid tracking warning fields: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        return cls(**{name: str(value[name]) for name in TRACKING_WARNING_FIELDS})


@dataclass(frozen=True)
class StageTimeline:
    stage_steps: Sequence[int]
    compute_stage_steps: Sequence[int] | None = None
    benchmark_start_env_steps: int = 0
    compute_start_env_steps: int = 0
    boundary_gap: float = 0.5

    def __post_init__(self) -> None:
        stage_steps = tuple(int(value) for value in self.stage_steps)
        compute_steps = tuple(
            int(value)
            for value in (
                self.compute_stage_steps
                if self.compute_stage_steps is not None
                else stage_steps
            )
        )
        if not stage_steps or any(value <= 0 for value in stage_steps):
            raise ValueError("stage_steps must contain positive values")
        if len(compute_steps) != len(stage_steps) or any(
            value <= 0 for value in compute_steps
        ):
            raise ValueError("compute_stage_steps must match stage_steps")
        if self.benchmark_start_env_steps < 0 or self.compute_start_env_steps < 0:
            raise ValueError("timeline start offsets must be non-negative")
        if not math.isfinite(self.boundary_gap) or self.boundary_gap < 0:
            raise ValueError("boundary_gap must be finite and non-negative")
        object.__setattr__(self, "stage_steps", stage_steps)
        object.__setattr__(self, "compute_stage_steps", compute_steps)

    def coordinates(self, stage: int, local_env_steps: int) -> dict[str, int | float]:
        if stage < 0 or stage >= len(self.stage_steps):
            raise ValueError(f"stage must be in [0, {len(self.stage_steps)})")
        compute_limit = self.compute_stage_steps[stage]
        if local_env_steps < 0 or local_env_steps > compute_limit:
            raise ValueError(
                f"local_env_steps must be in [0, {compute_limit}] for stage {stage}"
            )

        compute_offset = self.compute_start_env_steps + sum(
            self.compute_stage_steps[:stage]
        )
        benchmark_offset = self.benchmark_start_env_steps + sum(
            self.stage_steps[:stage]
        )
        benchmark_local = local_env_steps * self.stage_steps[stage] / compute_limit
        compute_actual = compute_offset + local_env_steps
        benchmark = benchmark_offset + benchmark_local
        return {
            "local_env_steps": local_env_steps,
            "env_steps": compute_actual,
            "benchmark_env_steps": benchmark,
            "compute_actual_env_steps": compute_actual,
            "plot_env_steps": benchmark + stage * self.boundary_gap,
        }

    def event(
        self,
        *,
        task: str,
        method: str,
        seed: int,
        stage: int,
        index: int,
        local_env_steps: int,
        return_mean: float,
        return_std: float,
    ) -> EvalEvent:
        return EvalEvent(
            task=task,
            method=method,
            seed=seed,
            stage=stage,
            index=index,
            return_mean=return_mean,
            return_std=return_std,
            **self.coordinates(stage, local_env_steps),
        )


class EventStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, event: EvalEvent) -> None:
        line = json.dumps(
            event.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())

    def __iter__(self) -> Iterator[EvalEvent]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    yield EvalEvent.from_dict(value)
                except (json.JSONDecodeError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"invalid event at {self.path}:{line_number}: {error}"
                    ) from error

    def read_all(self) -> list[EvalEvent]:
        return list(self)


class FailureStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, event: FailureEvent) -> None:
        line = json.dumps(
            event.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())

    def __iter__(self) -> Iterator[FailureEvent]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    yield FailureEvent.from_dict(value)
                except (json.JSONDecodeError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"invalid failure event at {self.path}:{line_number}: {error}"
                    ) from error

    def read_all(self) -> list[FailureEvent]:
        return list(self)


class TrackingWarningStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, warning: TrackingWarning) -> None:
        line = json.dumps(
            warning.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())

    def __iter__(self) -> Iterator[TrackingWarning]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    yield TrackingWarning.from_dict(value)
                except (json.JSONDecodeError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"invalid tracking warning at {self.path}:"
                        f"{line_number}: {error}"
                    ) from error

    def read_all(self) -> list[TrackingWarning]:
        return list(self)


@dataclass(frozen=True)
class WandbConfig:
    mode: str = "disabled"
    project: str | None = None
    entity: str | None = None
    name: str | None = None
    group: str | None = None
    job_type: str | None = None
    tags: Sequence[str] = ()
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in {"disabled", "online", "offline"}:
            raise ValueError("wandb mode must be disabled, online, or offline")


def _load_wandb() -> Any:
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "W&B tracking was enabled, but wandb is not installed"
        ) from error
    return wandb


def _wandb_payload(event: EvalEvent) -> dict[str, Any]:
    return {
        "task": event.task,
        "method": event.method,
        "seed": event.seed,
        "stage/index": event.stage,
        "stage/eval_index": event.index,
        "stage/local_env_steps": event.local_env_steps,
        "env_steps": event.env_steps,
        "benchmark/env_steps": event.benchmark_env_steps,
        "compute/actual_env_steps": event.compute_actual_env_steps,
        "plot/env_steps": event.plot_env_steps,
        "eval/return_mean": event.return_mean,
        "eval/return_std": event.return_std,
    }


def _wandb_failure_payload(event: FailureEvent) -> dict[str, Any]:
    return {
        "task": event.task,
        "method": event.method,
        "seed": event.seed,
        "stage/index": event.stage,
        "stage/local_env_steps": event.local_env_steps,
        "env_steps": event.env_steps,
        "benchmark/env_steps": event.benchmark_env_steps,
        "compute/actual_env_steps": event.compute_actual_env_steps,
        "plot/env_steps": event.plot_env_steps,
        "run/status": "failed",
        "benchmark/final_return": None,
        "eval/return_mean": None,
        "failure/type": event.error_type,
        "failure/message": event.message,
    }


class _WandbWriter:
    def __init__(
        self,
        directory: Path,
        config: WandbConfig,
        loader: Callable[[], Any],
        *,
        run_id: str | None = None,
        allow_resume: bool = True,
    ) -> None:
        self.directory = directory
        self.config = config
        self.loader = loader
        self.run_id = run_id
        self.allow_resume = allow_resume
        self.run: Any | None = None
        self.finished = False
        self.offline_event_count = 0
        self.offline_failure_count = 0
        self.resolved_exit_code: int | None = None

    def _persistent_run_id(self) -> str:
        path = self.directory / "wandb_run_id"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            run_id = path.read_text(encoding="utf-8").strip()
            if not run_id:
                raise ValueError(f"empty W&B run id: {path}")
            return run_id
        run_id = uuid.uuid4().hex
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(run_id)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        return run_id

    def _start(self, online: bool) -> Any:
        self.directory.mkdir(parents=True, exist_ok=True)
        wandb = self.loader()
        kwargs: dict[str, Any] = {
            "mode": "online" if online else "offline",
            "dir": str(self.directory),
            "tags": list(self.config.tags),
            "config": dict(self.config.config),
        }
        for name in ("project", "entity", "name", "group", "job_type"):
            value = getattr(self.config, name)
            if value is not None:
                kwargs[name] = value
        if self.run_id is not None:
            kwargs["id"] = self.run_id
            kwargs["resume"] = "allow" if online and self.allow_resume else "never"
        elif online:
            kwargs["id"] = self._persistent_run_id()
            kwargs["resume"] = "allow"
        self.run = wandb.init(**kwargs)
        for axis in (
            "stage/local_env_steps",
            "env_steps",
            "benchmark/env_steps",
            "compute/actual_env_steps",
            "plot/env_steps",
        ):
            self.run.define_metric(axis)
        self.run.define_metric("eval/*", step_metric="plot/env_steps")
        return self.run

    def write(self, event: EvalEvent) -> None:
        if self.config.mode != "online":
            return
        run = self.run if self.run is not None else self._start(online=True)
        run.log(_wandb_payload(event))

    def write_failure(self, event: FailureEvent) -> None:
        if self.config.mode != "online":
            return
        run = self.run if self.run is not None else self._start(online=True)
        run.log(_wandb_failure_payload(event))

    def finish(
        self,
        events: Iterable[EvalEvent],
        failures: Iterable[FailureEvent],
        *,
        exit_code: int | None = None,
    ) -> None:
        if self.finished:
            return
        if self.config.mode == "disabled":
            self.finished = True
            return
        event_values = list(events)
        failure_events = list(failures)
        if self.config.mode == "offline":
            run = self.run if self.run is not None else self._start(online=False)
            for event in event_values[self.offline_event_count :]:
                run.log(_wandb_payload(event))
                self.offline_event_count += 1
            for event in failure_events[self.offline_failure_count :]:
                run.log(_wandb_failure_payload(event))
                self.offline_failure_count += 1
        elif self.run is None:
            self._start(online=True)
        if self.run is not None:
            resolved_exit_code = (
                1 if failure_events else (0 if exit_code is None else exit_code)
            )
            self.run.finish(exit_code=resolved_exit_code)
            self.resolved_exit_code = resolved_exit_code
        self.finished = True

    def finish_failed_without_replay(self) -> None:
        """Close an active run as failed without reading local event stores."""

        if self.finished:
            return
        if self.config.mode == "disabled":
            self.finished = True
            return
        if self.run is None:
            if self.config.mode != "online":
                return
            self._start(online=True)
        self.run.finish(exit_code=1)
        self.finished = True


class Tracker:
    def __init__(
        self,
        directory: str | Path,
        *,
        wandb: WandbConfig | None = None,
        wandb_loader: Callable[[], Any] | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.store = EventStore(self.directory / "events.jsonl")
        self.failures = FailureStore(self.directory / "failures.jsonl")
        self.warnings = TrackingWarningStore(self.directory / "warnings.jsonl")
        self._writer = _WandbWriter(
            self.directory,
            wandb if wandb is not None else WandbConfig(),
            wandb_loader if wandb_loader is not None else _load_wandb,
        )

    def log_eval(self, event: EvalEvent) -> None:
        self.store.append(event)
        self._write_wandb("log_eval", lambda: self._writer.write(event))

    def log_failure(self, event: FailureEvent) -> None:
        self.failures.append(event)
        self._write_wandb(
            "log_failure",
            lambda: self._writer.write_failure(event),
        )

    def finish(self, *, exit_code: int | None = None) -> None:
        if exit_code not in {None, 0, 1}:
            raise ValueError("exit_code must be 0, 1, or None")
        try:
            events = self.store.read_all()
            failures = self.failures.read_all()
        except BaseException:
            if exit_code == 1:
                self._write_wandb(
                    "emergency_finish",
                    self._writer.finish_failed_without_replay,
                )
            raise
        self._write_wandb(
            "finish",
            lambda: self._writer.finish(
                events,
                failures,
                exit_code=exit_code,
            ),
        )

        if self._writer.config.mode == "offline" and self._writer.finished:
            self._write_wandb(
                "write_receipt",
                lambda: self._write_offline_receipt(events, failures),
            )

    def _write_offline_receipt(
        self,
        events: Sequence[EvalEvent],
        failures: Sequence[FailureEvent],
    ) -> None:
        if self._writer.offline_event_count != len(events):
            raise RuntimeError(
                "offline W&B did not log the complete evaluation event stream"
            )
        if self._writer.offline_failure_count != len(failures):
            raise RuntimeError(
                "offline W&B did not log the complete failure event stream"
            )
        run = self._writer.run
        run_id = getattr(run, "id", None)
        if not isinstance(run_id, str) or not run_id.strip():
            raise RuntimeError("W&B SDK did not expose a non-empty offline run ID")
        exit_code = self._writer.resolved_exit_code
        if exit_code not in {0, 1}:
            raise RuntimeError("offline W&B finish did not record a valid exit code")

        raw_wandb_directory = getattr(run, "dir", None)
        if not isinstance(raw_wandb_directory, str) or not raw_wandb_directory:
            raise RuntimeError("W&B SDK did not expose its offline run directory")
        wandb_directory = Path(raw_wandb_directory).resolve()
        wandb_artifact = wandb_directory.parent / f"run-{run_id}.wandb"
        if not wandb_artifact.is_file() or wandb_artifact.stat().st_size <= 0:
            raise RuntimeError("W&B SDK did not write its completed offline artifact")
        try:
            wandb_artifact_path = wandb_artifact.relative_to(self.directory.resolve())
        except ValueError as error:
            raise RuntimeError(
                "W&B offline artifact was written outside the tracking directory"
            ) from error
        wandb_bytes = wandb_artifact.read_bytes()

        event_path = self.store.path
        failure_path = self.failures.path
        event_bytes = event_path.read_bytes() if event_path.exists() else b""
        failure_bytes = failure_path.read_bytes() if failure_path.exists() else b""

        def coordinate(event: EvalEvent) -> dict[str, Any]:
            value = event.to_dict()
            return {name: value[name] for name in _RECEIPT_COORDINATE_FIELDS}

        receipt = {
            "schema_version": 1,
            "mode": "offline",
            "status": "complete" if exit_code == 0 else "failed",
            "exit_code": exit_code,
            "wandb_run_id": run_id,
            "wandb_directory": str(wandb_directory),
            "wandb_artifact": {
                "path": wandb_artifact_path.as_posix(),
                "sha256": hashlib.sha256(wandb_bytes).hexdigest(),
                "size_bytes": len(wandb_bytes),
            },
            "successful_log_count": len(events) + len(failures),
            "successful_eval_log_count": len(events),
            "successful_failure_log_count": len(failures),
            "event_stream": {
                "path": event_path.name,
                "sha256": hashlib.sha256(event_bytes).hexdigest(),
                "size_bytes": len(event_bytes),
                "event_count": len(events),
                "first_coordinate": coordinate(events[0]) if events else None,
                "last_coordinate": coordinate(events[-1]) if events else None,
            },
            "failure_stream": {
                "path": failure_path.name,
                "sha256": hashlib.sha256(failure_bytes).hexdigest(),
                "size_bytes": len(failure_bytes),
                "failure_count": len(failures),
            },
        }
        _atomic_write_json(self.directory / "wandb_receipt.json", receipt)

    def _write_wandb(self, operation: str, callback: Callable[[], None]) -> bool:
        try:
            callback()
            return True
        except Exception as error:
            self.warnings.append(
                TrackingWarning(
                    component="wandb",
                    operation=operation,
                    error_type=type(error).__name__,
                    message=str(error) or type(error).__name__,
                )
            )
            return False

    def __enter__(self) -> "Tracker":
        return self

    def __exit__(self, *_: Any) -> None:
        self.finish()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(
                value,
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()
