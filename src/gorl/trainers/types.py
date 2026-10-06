"""Requests and results for the packaged baseline subprocess worker."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Mapping

from gorl.actions import ActionSemantics, ActionSpec


class Method(StrEnum):
    PPO = "ppo"
    FPO = "fpo"
    DPPO = "dppo"


class PPOProfile(StrEnum):
    """PPO implementation family hidden behind the unified public interface."""

    LEGACY_LATENT = "legacy_latent"
    BRAX = "brax"


class RunStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class TrainingRequest:
    method: Method
    task: str
    seed: int
    output_dir: Path
    action_spec: ActionSpec
    ppo_profile: PPOProfile | None = None
    environment_steps: int = 1
    arguments: tuple[str, ...] = ()
    environment: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir).expanduser())
        object.__setattr__(self, "arguments", tuple(self.arguments))
        _validate_training_request(self)


@dataclass(frozen=True, slots=True)
class TrainingResult:
    method: Method
    task: str
    seed: int
    status: RunStatus
    output_dir: Path
    final_return: float | None = None
    best_return: float | None = None
    command: tuple[str, ...] = ()
    returncode: int = 0
    stdout_tail: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "command", tuple(self.command))
        _require_finite_optional("final_return", self.final_return)
        _require_finite_optional("best_return", self.best_return)
        if (
            self.final_return is not None
            and self.best_return is not None
            and self.best_return < self.final_return
        ):
            raise ValueError(
                "best_return cannot be smaller than final_return: "
                f"{self.best_return} < {self.final_return}"
            )
        if self.status is RunStatus.SUCCEEDED and self.returncode != 0:
            raise ValueError("a succeeded result must have returncode 0")
        if self.status is RunStatus.FAILED and self.returncode == 0:
            raise ValueError("a failed result must have a non-zero returncode")


def _validate_training_request(request: TrainingRequest) -> None:
    if not request.task or request.task.strip() != request.task:
        raise ValueError(
            f"task must be a non-empty canonical name, got {request.task!r}"
        )
    if request.seed < 0:
        raise ValueError(f"seed must be non-negative, got {request.seed}")
    if request.environment_steps <= 0:
        raise ValueError(
            f"environment_steps must be positive, got {request.environment_steps}"
        )
    if any("\x00" in argument for argument in request.arguments):
        raise ValueError("arguments cannot contain NUL characters")
    if any(
        "\x00" in key or "\x00" in value for key, value in request.environment.items()
    ):
        raise ValueError("environment entries cannot contain NUL characters")

    if request.method is Method.PPO:
        if request.ppo_profile is not PPOProfile.LEGACY_LATENT:
            actual = (
                "none" if request.ppo_profile is None else request.ppo_profile.value
            )
            raise ValueError(
                "the packaged PPO subprocess worker requires profile "
                f"legacy_latent, got {actual}"
            )
    else:
        if request.ppo_profile is not None:
            raise ValueError(
                f"{request.method.value} does not use a PPO profile; "
                f"got {request.ppo_profile.value}"
            )

    if request.action_spec.semantics is not ActionSemantics.LEGACY_PRE_TANH:
        raise ValueError(
            f"{request.method.value} baseline requires action semantics "
            f"{ActionSemantics.LEGACY_PRE_TANH.value}, "
            f"got {request.action_spec.semantics.value}"
        )


def _require_finite_optional(name: str, value: float | None) -> None:
    if value is not None and not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
