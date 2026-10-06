"""Subprocess adapters for the verified local training entrypoints."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from gorl.trainers.entrypoints import (
    BackendAvailability,
    EntrypointCandidate,
    EntrypointResolver,
    ResolvedEntrypoint,
)
from gorl.trainers.types import (
    Method,
    PPOProfile,
    RunStatus,
    TrainingRequest,
    TrainingResult,
)


UPSTREAM_FPO_COMMIT = "418c2554f7cd22d52e14c07d951280929d73bf2f"
DPPO_FIX_COMMIT = "964dd78c6fb64de8c52eeb7ce80561c43455128f"


class RunnerKind(StrEnum):
    UPSTREAM_FPO = "upstream_fpo"


@dataclass(frozen=True, slots=True)
class TrainerEntrypointSpec:
    method: Method
    profile: PPOProfile | None
    backend: str
    runner: RunnerKind
    candidates: tuple[EntrypointCandidate, ...]
    supports_tasks: frozenset[str] | None = None
    dependency_candidates: tuple[Path, ...] = ()
    dependency_environment_variable: str | None = None
    dependency_commit: str | None = None


@dataclass(frozen=True, slots=True)
class _DependencyResolution:
    path: Path | None
    checked_paths: tuple[Path, ...]
    errors: tuple[str, ...]
    explicit_environment_variable: str | None = None


class TrainingExecutionError(RuntimeError):
    def __init__(self, result: TrainingResult) -> None:
        self.result = result
        command = " ".join(result.command)
        super().__init__(
            f"{result.method.value} training failed with exit code "
            f"{result.returncode}: {command}"
        )


class TrainingOutputError(RuntimeError):
    pass


class ScriptTrainerAdapter:
    """Build and run a command without importing either JAX training stack."""

    def __init__(
        self,
        spec: TrainerEntrypointSpec,
        *,
        search_roots: Iterable[str | Path] | None = None,
        python_executable: str | Path | None = None,
    ) -> None:
        self.spec = spec
        self.python_executable = str(python_executable or sys.executable)
        self._resolver = EntrypointResolver(
            spec.backend,
            spec.candidates,
            search_roots=search_roots,
        )

    @property
    def method(self) -> Method:
        return self.spec.method

    @property
    def profile(self) -> PPOProfile | None:
        return self.spec.profile

    @property
    def availability(self) -> BackendAvailability:
        availability = self._resolver.availability()
        if availability.entrypoint is None or not self.spec.dependency_candidates:
            return availability
        resolution = self._find_dependency(
            availability.entrypoint,
        )
        if resolution.path is not None:
            return BackendAvailability(
                backend=availability.backend,
                available=True,
                entrypoint=availability.entrypoint,
                reason=(
                    f"{availability.reason}; dependency found at {resolution.path}"
                ),
                checked_paths=availability.checked_paths + resolution.checked_paths,
            )
        rendered = "\n  - ".join(resolution.errors)
        if resolution.explicit_environment_variable is not None:
            detail = (
                f"explicit {resolution.explicit_environment_variable} checkout is "
                f"incompatible:\n  - {rendered}"
            )
        else:
            detail = f"no compatible dependency checkout was found:\n  - {rendered}"
        return BackendAvailability(
            backend=availability.backend,
            available=False,
            entrypoint=None,
            reason=(
                f"{self.spec.backend} entrypoint exists at "
                f"{availability.entrypoint.script}, but its implementation "
                f"dependency is unavailable; {detail}"
            ),
            checked_paths=availability.checked_paths + resolution.checked_paths,
        )

    def build_command(self, request: TrainingRequest) -> tuple[str, ...]:
        self._validate_request(request)
        entrypoint = self.availability.require()
        command = [self.python_executable, str(entrypoint.script)]
        command.extend(self._upstream_arguments(request, entrypoint))
        command.extend(request.arguments)
        return tuple(command)

    def train(
        self,
        request: TrainingRequest,
        *,
        check: bool = True,
        capture_output: bool = False,
    ) -> TrainingResult:
        command = self.build_command(request)
        entrypoint = self.availability.require()
        request.output_dir.mkdir(parents=True, exist_ok=True)
        environment = _runtime_environment(request.environment)
        completed = subprocess.run(
            command,
            cwd=entrypoint.working_directory,
            env=environment,
            check=False,
            text=True,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.STDOUT if capture_output else None,
        )
        output = completed.stdout if isinstance(completed.stdout, str) else None
        status = RunStatus.SUCCEEDED if completed.returncode == 0 else RunStatus.FAILED
        dependency = self._find_dependency(entrypoint).path
        dependency_metadata = (
            _dependency_metadata(dependency) if dependency is not None else {}
        )
        parsed_metrics = self._read_structured_metrics(request, completed.returncode)
        manifest_metadata = self._read_worker_manifest(
            request,
            completed.returncode,
        )
        worker_metadata = {
            key: parsed_metrics[key]
            for key in ("requested_env_steps", "resolved_env_steps")
            if key in parsed_metrics
        }
        result = TrainingResult(
            method=request.method,
            task=request.task,
            seed=request.seed,
            status=status,
            output_dir=self._artifact_root(request, entrypoint),
            final_return=parsed_metrics.get("final_return"),
            best_return=parsed_metrics.get("overall_best_return"),
            command=command,
            returncode=completed.returncode,
            stdout_tail=output[-20_000:] if output is not None else None,
            metadata={
                "backend": self.spec.backend,
                "trainer_backend": self.spec.backend,
                "action_semantics": request.action_spec.semantics.value,
                "entrypoint": str(entrypoint.script),
                "working_directory": str(entrypoint.working_directory),
                "metrics_parsed": bool(parsed_metrics),
                "worker_manifest_parsed": bool(manifest_metadata),
                "requested_output_dir": str(request.output_dir.resolve()),
                **dependency_metadata,
                **manifest_metadata,
                **worker_metadata,
            },
        )
        if check and completed.returncode != 0:
            raise TrainingExecutionError(result)
        return result

    def _validate_request(self, request: TrainingRequest) -> None:
        if request.method is not self.spec.method:
            raise ValueError(
                f"adapter is for {self.spec.method.value}, got {request.method.value}"
            )
        if request.ppo_profile is not self.spec.profile:
            expected = self.spec.profile.value if self.spec.profile else "none"
            actual = request.ppo_profile.value if request.ppo_profile else "none"
            raise ValueError(
                f"adapter profile mismatch: expected {expected}, got {actual}"
            )
        if (
            self.spec.supports_tasks is not None
            and request.task not in self.spec.supports_tasks
        ):
            supported = ", ".join(sorted(self.spec.supports_tasks))
            raise ValueError(
                f"{self.spec.backend} does not support task {request.task}; "
                f"supported tasks: {supported}"
            )
        _reject_controlled_arguments(
            request.arguments,
            {
                "--fpo-root",
                "--method",
                "--num-timesteps",
                "--output-dir",
                "--seed",
                "--task",
            },
        )

    def _upstream_arguments(
        self,
        request: TrainingRequest,
        entrypoint: ResolvedEntrypoint,
    ) -> list[str]:
        if request.environment_steps is None:
            raise ValueError(
                f"{request.method.value} baseline requires environment_steps"
            )
        arguments = [
            "--method",
            request.method.value,
            "--task",
            request.task,
            "--seed",
            str(request.seed),
            "--num-timesteps",
            str(request.environment_steps),
            "--output-dir",
            str(request.output_dir.resolve()),
        ]
        resolution = self._find_dependency(entrypoint)
        if resolution.path is None:
            rendered = "; ".join(resolution.errors)
            raise ValueError(
                f"{self.spec.backend} implementation dependency is unavailable: "
                f"{rendered}"
            )
        arguments.extend(["--fpo-root", str(resolution.path)])
        return arguments

    def _find_dependency(
        self,
        entrypoint: ResolvedEntrypoint,
    ) -> _DependencyResolution:
        if not self.spec.dependency_candidates:
            return _DependencyResolution(None, (), ())
        expected_commit = self.spec.dependency_commit
        environment_variable = self.spec.dependency_environment_variable
        if expected_commit is None or environment_variable is None:
            raise ValueError(f"{self.spec.backend} dependency provenance is incomplete")
        roots = (entrypoint.source_root, *self._resolver.search_roots)
        checked: list[Path] = []
        errors: list[str] = []
        seen: set[Path] = set()
        configured = os.environ.get(environment_variable)
        if configured:
            path = Path(configured).expanduser().resolve(strict=False)
            checked.append(path)
            error = _dependency_checkout_error(path, expected_commit)
            if error is None:
                return _DependencyResolution(path, tuple(checked), ())
            errors.append(f"{path}: {error}")
            return _DependencyResolution(
                None,
                tuple(checked),
                tuple(errors),
                explicit_environment_variable=environment_variable,
            )
        for root in roots:
            for relative in self.spec.dependency_candidates:
                path = (
                    relative if relative.is_absolute() else root / relative
                ).resolve(strict=False)
                if path in seen:
                    continue
                seen.add(path)
                checked.append(path)
                error = _dependency_checkout_error(path, expected_commit)
                if error is None:
                    return _DependencyResolution(path, tuple(checked), tuple(errors))
                errors.append(f"{path}: {error}")
        return _DependencyResolution(None, tuple(checked), tuple(errors))

    def _read_structured_metrics(
        self,
        request: TrainingRequest,
        returncode: int,
    ) -> dict[str, float | int]:
        if returncode != 0 or self.spec.runner is not RunnerKind.UPSTREAM_FPO:
            return {}
        path = request.output_dir.resolve() / "summary.json"
        if not path.is_file():
            raise TrainingOutputError(
                f"baseline worker exited successfully without summary.json: {path}"
            )
        try:
            value = _read_json_object(path)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            raise TrainingOutputError(f"invalid baseline summary: {path}") from error
        identity = {
            "method": request.method.value,
            "task": request.task,
            "seed": request.seed,
        }
        mismatches = {
            key: (value.get(key), expected)
            for key, expected in identity.items()
            if value.get(key) != expected
        }
        if mismatches:
            raise TrainingOutputError(
                f"baseline summary identity mismatch in {path}: {mismatches}"
            )
        if "status" in value and value.get("status") != "complete":
            raise TrainingOutputError(
                f"baseline summary status must be 'complete' in {path}, "
                f"got {value.get('status')!r}"
            )
        try:
            requested_steps = int(value["requested_env_steps"])
            resolved_steps = int(value["resolved_env_steps"])
            parsed: dict[str, float | int] = {
                "final_return": float(value["final_return"]),
                "overall_best_return": float(value["overall_best_return"]),
                "requested_env_steps": requested_steps,
                "resolved_env_steps": resolved_steps,
            }
        except (KeyError, TypeError, ValueError) as error:
            raise TrainingOutputError(
                f"baseline summary is missing numeric returns or steps: {path}"
            ) from error
        if requested_steps != request.environment_steps:
            raise TrainingOutputError(
                "baseline summary requested_env_steps mismatch in "
                f"{path}: {requested_steps} != {request.environment_steps}"
            )
        if resolved_steps <= 0 or resolved_steps > requested_steps:
            raise TrainingOutputError(
                "baseline summary has invalid resolved_env_steps in "
                f"{path}: {resolved_steps} for request {requested_steps}"
            )
        for name in ("final_return", "overall_best_return"):
            if not math.isfinite(float(parsed[name])):
                raise TrainingOutputError(
                    f"baseline summary has non-finite {name} in {path}"
                )
        return parsed

    def _read_worker_manifest(
        self,
        request: TrainingRequest,
        returncode: int,
    ) -> dict[str, object]:
        path = request.output_dir.resolve() / "worker_manifest.json"
        if not path.is_file():
            return {}
        try:
            value = _read_json_object(path)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            raise TrainingOutputError(
                f"invalid baseline worker manifest: {path}"
            ) from error
        identity = {
            "method": request.method.value,
            "task": request.task,
            "seed": request.seed,
        }
        mismatches = {
            key: (value.get(key), expected)
            for key, expected in identity.items()
            if value.get(key) != expected
        }
        if mismatches:
            raise TrainingOutputError(
                f"baseline worker manifest identity mismatch in {path}: {mismatches}"
            )
        schema_version = value.get("schema_version")
        if schema_version is None:
            if any(name in value for name in ("status", "failure", "partial_results")):
                raise TrainingOutputError(
                    f"unversioned baseline worker manifest uses current fields in {path}"
                )
            status = "legacy_complete" if returncode == 0 else "legacy_failed"
            manifest_schema = "legacy"
        else:
            if (
                isinstance(schema_version, bool)
                or not isinstance(schema_version, int)
                or schema_version != 1
            ):
                raise TrainingOutputError(
                    f"unsupported baseline worker manifest schema_version "
                    f"{schema_version!r} in {path}"
                )
            status = value.get("status")
            allowed_statuses = (
                {"complete"} if returncode == 0 else {"failed", "running"}
            )
            if status not in allowed_statuses:
                raise TrainingOutputError(
                    f"baseline worker exit permits status "
                    f"{sorted(allowed_statuses)!r}, "
                    f"got {status!r} in {path}"
                )
            self._validate_current_worker_result(value, status, path)
            manifest_schema = "current"

        metadata: dict[str, object] = {
            "worker_status": status,
            "worker_manifest_schema": manifest_schema,
        }
        if schema_version is not None:
            metadata["worker_manifest_schema_version"] = schema_version
        for name in ("requested_env_steps", "resolved_env_steps"):
            raw = value.get(name)
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise TrainingOutputError(
                    f"baseline worker manifest has invalid {name} in {path}"
                )
            metadata[name] = raw
        requested_steps = int(metadata["requested_env_steps"])
        resolved_steps = int(metadata["resolved_env_steps"])
        if requested_steps != request.environment_steps:
            raise TrainingOutputError(
                "baseline worker manifest requested_env_steps mismatch in "
                f"{path}: {requested_steps} != {request.environment_steps}"
            )
        if resolved_steps <= 0 or resolved_steps > requested_steps:
            raise TrainingOutputError(
                "baseline worker manifest has invalid resolved_env_steps in "
                f"{path}: {resolved_steps} for request {requested_steps}"
            )
        failure = value.get("failure")
        if failure is not None:
            metadata["worker_failure"] = dict(failure)
        partial = value.get("partial_results")
        if partial is not None:
            metadata["partial_results"] = dict(partial)
        return metadata

    @staticmethod
    def _validate_current_worker_result(
        value: Mapping[str, object],
        status: object,
        path: Path,
    ) -> None:
        result = value.get("result")
        required_returns = (
            "final_return",
            "stage_best_return",
            "overall_best_return",
        )
        if not isinstance(result, dict) or any(
            name not in result for name in required_returns
        ):
            raise TrainingOutputError(
                f"baseline worker manifest has invalid result in {path}"
            )
        if "failure" not in value:
            raise TrainingOutputError(
                f"baseline worker manifest is missing failure field in {path}"
            )
        failure = value.get("failure")
        partial = value.get("partial_results")
        if status == "complete":
            if failure is not None or partial is not None:
                raise TrainingOutputError(
                    f"complete baseline worker has failure evidence in {path}"
                )
            for name in required_returns:
                metric = result[name]
                if (
                    isinstance(metric, bool)
                    or not isinstance(metric, (int, float))
                    or not math.isfinite(metric)
                ):
                    raise TrainingOutputError(
                        f"complete baseline worker has invalid {name} in {path}"
                    )
        elif status == "failed":
            if not isinstance(failure, dict) or not failure:
                raise TrainingOutputError(
                    f"failed baseline worker is missing failure evidence in {path}"
                )
            if not isinstance(partial, dict):
                raise TrainingOutputError(
                    f"failed baseline worker is missing partial_results in {path}"
                )
            if any(result[name] is not None for name in required_returns):
                raise TrainingOutputError(
                    f"failed baseline worker must keep formal returns null in {path}"
                )
        else:
            if failure is not None or partial is not None:
                raise TrainingOutputError(
                    f"running baseline worker has invalid failure evidence in {path}"
                )
            if any(result[name] is not None for name in required_returns):
                raise TrainingOutputError(
                    f"running baseline worker must keep formal returns null in {path}"
                )

    def _artifact_root(
        self,
        request: TrainingRequest,
        entrypoint: ResolvedEntrypoint,
    ) -> Path:
        del entrypoint
        return request.output_dir.resolve()


_UPSTREAM_CANDIDATES = (
    EntrypointCandidate("src/gorl/_workers/upstream_fpo.py"),
    EntrypointCandidate("gorl/_workers/upstream_fpo.py"),
)
_NON_HUMANOID_TASKS = frozenset(
    {
        "CheetahRun",
        "FingerSpin",
        "FingerTurnHard",
        "FishSwim",
        "HopperStand",
        "WalkerWalk",
    }
)


TRAINER_SPECS: Mapping[tuple[Method, PPOProfile | None], TrainerEntrypointSpec] = {
    (Method.PPO, PPOProfile.LEGACY_LATENT): TrainerEntrypointSpec(
        method=Method.PPO,
        profile=PPOProfile.LEGACY_LATENT,
        backend="upstream_fpo_ppo",
        runner=RunnerKind.UPSTREAM_FPO,
        candidates=_UPSTREAM_CANDIDATES,
        supports_tasks=_NON_HUMANOID_TASKS,
        dependency_candidates=(Path("external/fpo"), Path("fpo")),
        dependency_environment_variable="GORL_FPO_ROOT",
        dependency_commit=UPSTREAM_FPO_COMMIT,
    ),
    (Method.FPO, None): TrainerEntrypointSpec(
        method=Method.FPO,
        profile=None,
        backend="upstream_fpo",
        runner=RunnerKind.UPSTREAM_FPO,
        candidates=_UPSTREAM_CANDIDATES,
        dependency_candidates=(Path("external/fpo"), Path("fpo")),
        dependency_environment_variable="GORL_FPO_ROOT",
        dependency_commit=UPSTREAM_FPO_COMMIT,
    ),
    (Method.DPPO, None): TrainerEntrypointSpec(
        method=Method.DPPO,
        profile=None,
        backend="upstream_fpo_denoising_mdp",
        runner=RunnerKind.UPSTREAM_FPO,
        candidates=_UPSTREAM_CANDIDATES,
        dependency_candidates=(Path("external/fpo_dppo_fix"),),
        dependency_environment_variable="GORL_DPPO_ROOT",
        dependency_commit=DPPO_FIX_COMMIT,
    ),
}


def baseline_worker_arguments(
    method: Method,
    ppo: Mapping[str, object],
    decoder: Mapping[str, object] | None = None,
) -> tuple[str, ...]:
    """Translate resolved baseline TOML sections into worker CLI arguments."""

    if method not in {Method.PPO, Method.FPO, Method.DPPO}:
        raise ValueError(f"{method.value} is not a standalone baseline")

    ppo_flags = {
        "learning_rate": "--learning-rate",
        "clip_epsilon": "--clip-epsilon",
        "entropy_cost": "--entropy-cost",
        "discounting": "--discounting",
        "gae_lambda": "--gae-lambda",
        "reward_scaling": "--reward-scaling",
        "value_loss_coeff": "--value-loss-coeff",
        "num_envs": "--num-envs",
        "num_eval_envs": "--num-eval-envs",
        "episode_length": "--episode-length",
        "batch_size": "--batch-size",
        "num_minibatches": "--num-minibatches",
        "unroll_length": "--unroll-length",
        "num_updates_per_batch": "--num-updates-per-batch",
        "num_evals": "--num-evals",
    }
    boolean_ppo_flags = {
        "normalize_observations": "--normalize-observations",
        "normalize_advantage": "--normalize-advantage",
    }
    ignored_ppo = {"backend"}
    unknown_ppo = set(ppo).difference(
        ppo_flags,
        boolean_ppo_flags,
        ignored_ppo,
    )
    if unknown_ppo:
        raise ValueError(
            "unmapped baseline PPO config keys: " + ", ".join(sorted(unknown_ppo))
        )

    arguments: list[str] = []
    for key, option in ppo_flags.items():
        if key in ppo:
            arguments.extend((option, str(ppo[key])))
    for key, option in boolean_ppo_flags.items():
        if key not in ppo:
            continue
        value = ppo[key]
        if not isinstance(value, bool):
            raise TypeError(f"{key} must be boolean, got {value!r}")
        arguments.append(option if value else f"--no-{option.removeprefix('--')}")

    decoder_values = {} if decoder is None else dict(decoder)
    if method is Method.PPO:
        if decoder_values:
            raise ValueError("PPO baseline does not accept a decoder section")
        return tuple(arguments)

    expected_type = "flow_policy" if method is Method.FPO else "denoising_mdp"
    configured_type = decoder_values.pop("type", expected_type)
    if configured_type != expected_type:
        raise ValueError(
            f"{method.value} requires decoder.type={expected_type!r}, "
            f"got {configured_type!r}"
        )
    flow_steps = decoder_values.pop("flow_steps", None)
    diffusion_steps = decoder_values.pop("diffusion_steps", None)
    if flow_steps is not None and diffusion_steps is not None:
        raise ValueError("set only one of flow_steps and diffusion_steps")
    if flow_steps is not None or diffusion_steps is not None:
        selected_steps = flow_steps if flow_steps is not None else diffusion_steps
        arguments.extend(("--flow-steps", str(selected_steps)))

    decoder_flags = {
        "samples_per_action": "--samples-per-action",
        "output_mode": "--output-mode",
        "timestep_embedding_dim": "--timestep-embedding-dim",
        "policy_output_scale": "--policy-output-scale",
        "sde_sigma": "--sde-sigma",
    }
    unknown_decoder = set(decoder_values).difference(decoder_flags)
    if unknown_decoder:
        raise ValueError(
            "unmapped baseline decoder config keys: "
            + ", ".join(sorted(unknown_decoder))
        )
    for key, option in decoder_flags.items():
        if key in decoder_values:
            arguments.extend((option, str(decoder_values[key])))
    return tuple(arguments)


def create_trainer(
    method: Method,
    profile: PPOProfile | None,
    *,
    search_roots: Iterable[str | Path] | None = None,
    python_executable: str | Path | None = None,
) -> ScriptTrainerAdapter:
    try:
        spec = TRAINER_SPECS[(method, profile)]
    except KeyError as exc:
        rendered = profile.value if profile is not None else "none"
        raise ValueError(
            f"no trainer adapter for method={method.value}, profile={rendered}"
        ) from exc
    return ScriptTrainerAdapter(
        spec,
        search_roots=search_roots,
        python_executable=python_executable,
    )


def trainer_availability(
    *,
    search_roots: Iterable[str | Path] | None = None,
) -> dict[tuple[Method, PPOProfile | None], BackendAvailability]:
    return {
        key: ScriptTrainerAdapter(
            spec,
            search_roots=search_roots,
        ).availability
        for key, spec in TRAINER_SPECS.items()
    }


def _runtime_environment(overrides: Mapping[str, str]) -> dict[str, str]:
    environment = os.environ.copy()
    environment.setdefault("PYTHONNOUSERSITE", "1")
    environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    environment.setdefault("MUJOCO_GL", "egl")
    environment.update(overrides)
    return environment


def _read_json_object(path: Path) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def _dependency_metadata(path: Path) -> dict[str, object]:
    commit = _git_output(path, "rev-parse", "HEAD")
    status = _git_output(path, "status", "--porcelain")
    return {
        "implementation_dependency": str(path),
        "implementation_dependency_commit": commit,
        "implementation_dependency_clean": (
            status == "" if status is not None else None
        ),
    }


def _dependency_checkout_error(path: Path, expected_commit: str) -> str | None:
    source = path / "playground" / "src" / "flow_policy"
    if not source.is_dir():
        return f"missing FPO playground sources at {source}"
    root = _git_output(path, "rev-parse", "--show-toplevel")
    if root is None:
        return "not a readable Git checkout"
    if Path(root).resolve() != path.resolve():
        return f"not an independent Git checkout (repository root is {root})"
    commit = _git_output(path, "rev-parse", "HEAD")
    if commit is None:
        return "Git checkout has no resolvable HEAD"
    if commit != expected_commit:
        return f"requires commit {expected_commit}, got {commit}"
    status = _git_output(path, "status", "--porcelain")
    if status is None:
        return "could not inspect Git checkout status"
    if status:
        return "refusing to use a dirty checkout"
    return None


def _git_output(path: Path, *arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), *arguments],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _reject_controlled_arguments(
    arguments: Sequence[str],
    controlled: set[str],
) -> None:
    for argument in arguments:
        option = argument.split("=", 1)[0]
        if option in controlled:
            raise ValueError(
                f"{option} is controlled by TrainingRequest and cannot be "
                "repeated in arguments"
            )
