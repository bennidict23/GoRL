"""Dependency-light contracts for the Humanoid Brax PPO backend."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Mapping

from gorl.decoders.types import DecoderKind


STATE_SCHEMA_VERSION = 1
SUPPORTED_TASKS = ("HumanoidStand", "HumanoidRun")

PPO_OVERRIDE_FIELDS = frozenset(
    {
        "action_repeat",
        "batch_size",
        "bootstrap_on_timeout",
        "clipping_epsilon",
        "clipping_epsilon_value",
        "desired_kl",
        "discounting",
        "entropy_cost",
        "episode_length",
        "gae_lambda",
        "learning_rate",
        "learning_rate_schedule",
        "log_training_metrics",
        "max_devices_per_host",
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
        "training_metrics_steps",
        "unroll_length",
        "use_pmap_on_reset",
        "vf_loss_coefficient",
    }
)


class HumanoidTask(StrEnum):
    STAND = "HumanoidStand"
    RUN = "HumanoidRun"


class HumanoidPPOExecutionProfile(StrEnum):
    """Process-level execution semantics for the shared Brax PPO backend."""

    NATIVE = "native"
    OFFICIAL_DIRECT_COMPAT = "official_direct_compat"


class HumanoidPPOPhase(StrEnum):
    TEACHER = "teacher"
    LATENT = "latent"


class Initialization(StrEnum):
    FROM_SCRATCH = "from_scratch"
    WARM_START = "warm_start"


class EventSource(StrEnum):
    TRAINING_EVAL = "training_eval"
    DETERMINISTIC_REPORT = "deterministic_report"


class HumanoidNonFiniteMetricError(RuntimeError):
    """Structured failure raised when official PPO emits a non-finite scalar."""

    def __init__(
        self,
        *,
        source: EventSource | str,
        metric: str,
        value: float,
        env_steps: int,
    ) -> None:
        source_value = source.value if isinstance(source, EventSource) else str(source)
        metric_value = str(metric)
        scalar = float(value)
        _non_negative_integer("env_steps", env_steps)
        if not source_value or not metric_value:
            raise ValueError("non-finite metric source and name must be non-empty")
        if math.isfinite(scalar):
            raise ValueError("HumanoidNonFiniteMetricError requires a non-finite value")
        if math.isnan(scalar):
            safe_value = "nan"
        elif scalar > 0:
            safe_value = "inf"
        else:
            safe_value = "-inf"
        self.source = source_value
        self.metric = metric_value
        self.value = safe_value
        self.env_steps = env_steps
        super().__init__(
            f"non-finite Humanoid metric {metric_value}={safe_value} "
            f"from {source_value} at env_steps={env_steps}"
        )

    def to_dict(self) -> dict[str, str | int]:
        return {
            "type": "non_finite_metric",
            "source": self.source,
            "metric": self.metric,
            "value": self.value,
            "env_steps": self.env_steps,
            "message": str(self),
        }


@dataclass(frozen=True, slots=True)
class HumanoidPPOConfig:
    """Configuration shared by fresh teacher and latent PPO stages.

    ``training_eval_deterministic`` defaults to ``False`` to retain the
    stochastic Brax evaluation used by the historical Humanoid experiments.
    Final report evaluation is a separate deterministic API.
    """

    task: str
    seed: int
    requested_environment_steps: int
    stage_index: int = 0
    num_evals: int = 10
    num_eval_envs: int = 128
    training_eval_deterministic: bool = False
    ppo_overrides: Mapping[str, object] = field(default_factory=dict)
    execution_profile: HumanoidPPOExecutionProfile = HumanoidPPOExecutionProfile.NATIVE

    def __post_init__(self) -> None:
        try:
            task = HumanoidTask(self.task).value
        except ValueError as error:
            supported = ", ".join(SUPPORTED_TASKS)
            raise ValueError(
                f"Humanoid backend only supports {supported}; got {self.task!r}"
            ) from error
        object.__setattr__(self, "task", task)
        _non_negative_integer("seed", self.seed)
        _positive_integer(
            "requested_environment_steps",
            self.requested_environment_steps,
        )
        _non_negative_integer("stage_index", self.stage_index)
        _positive_integer("num_evals", self.num_evals)
        _positive_integer("num_eval_envs", self.num_eval_envs)
        try:
            execution_profile = HumanoidPPOExecutionProfile(self.execution_profile)
        except (TypeError, ValueError) as error:
            choices = ", ".join(
                profile.value for profile in HumanoidPPOExecutionProfile
            )
            raise ValueError(
                "unsupported Humanoid PPO execution profile "
                f"{self.execution_profile!r}; expected one of {choices}"
            ) from error
        object.__setattr__(self, "execution_profile", execution_profile)

        overrides = dict(self.ppo_overrides)
        unknown = sorted(set(overrides).difference(PPO_OVERRIDE_FIELDS))
        if unknown:
            names = ", ".join(unknown)
            raise ValueError(f"unsupported or controlled PPO override(s): {names}")
        object.__setattr__(self, "ppo_overrides", overrides)


@dataclass(frozen=True, slots=True)
class HumanoidLatentStageConfig:
    ppo: HumanoidPPOConfig
    decoder_kind: DecoderKind
    latent_size: int | None = None
    record_decoded_actions: bool = False

    def __post_init__(self) -> None:
        kind = DecoderKind(self.decoder_kind)
        if kind is DecoderKind.IDENTITY:
            raise ValueError("latent encoder stages require an FM or diffusion decoder")
        object.__setattr__(self, "decoder_kind", kind)
        if self.latent_size is not None:
            _positive_integer("latent_size", self.latent_size)


@dataclass(frozen=True, slots=True)
class HumanoidPPOState:
    """Portable policy state used for checkpoints and stage warm starts."""

    task: str
    seed: int
    stage_index: int
    decoder_kind: DecoderKind
    action_size: int
    latent_size: int
    requested_environment_steps: int
    actual_environment_steps: int
    ppo_parameters: Mapping[str, object]
    normalizer_params: Any = field(repr=False)
    policy_params: Any = field(repr=False)
    value_params: Any = field(repr=False)
    training_eval_deterministic: bool = False
    action_semantics: str = "bounded"
    schema_version: int = STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        try:
            task = HumanoidTask(self.task).value
        except ValueError as error:
            raise ValueError(f"unsupported Humanoid task: {self.task!r}") from error
        object.__setattr__(self, "task", task)
        object.__setattr__(self, "decoder_kind", DecoderKind(self.decoder_kind))
        _non_negative_integer("seed", self.seed)
        _non_negative_integer("stage_index", self.stage_index)
        _positive_integer("action_size", self.action_size)
        _positive_integer("latent_size", self.latent_size)
        _positive_integer(
            "requested_environment_steps",
            self.requested_environment_steps,
        )
        _non_negative_integer(
            "actual_environment_steps",
            self.actual_environment_steps,
        )
        if self.action_semantics != "bounded":
            raise ValueError(
                "Humanoid profile checkpoints require bounded action semantics"
            )
        if self.schema_version != STATE_SCHEMA_VERSION:
            raise ValueError(f"unsupported state schema version: {self.schema_version}")
        object.__setattr__(self, "ppo_parameters", dict(self.ppo_parameters))

    @property
    def brax_params(self) -> tuple[Any, Any, Any]:
        return self.normalizer_params, self.policy_params, self.value_params


@dataclass(frozen=True, slots=True)
class HumanoidWarmStart:
    """Select which checkpoint components initialize the next PPO stage."""

    state: HumanoidPPOState
    restore_policy: bool = True
    restore_value: bool = True
    restore_normalizer: bool = True

    def __post_init__(self) -> None:
        if not (self.restore_policy or self.restore_value or self.restore_normalizer):
            raise ValueError("warm-start must restore at least one component")


@dataclass(frozen=True, slots=True)
class HumanoidMetricEvent:
    source: EventSource
    task: str
    seed: int
    stage_index: int
    evaluation_index: int
    requested_environment_steps: int
    actual_environment_steps: int
    deterministic: bool
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", EventSource(self.source))
        try:
            task = HumanoidTask(self.task).value
        except ValueError as error:
            raise ValueError(f"unsupported Humanoid task: {self.task!r}") from error
        object.__setattr__(self, "task", task)
        _non_negative_integer("seed", self.seed)
        _non_negative_integer("stage_index", self.stage_index)
        _non_negative_integer("evaluation_index", self.evaluation_index)
        _positive_integer(
            "requested_environment_steps",
            self.requested_environment_steps,
        )
        _non_negative_integer(
            "actual_environment_steps",
            self.actual_environment_steps,
        )
        metrics = {str(key): float(value) for key, value in self.metrics.items()}
        if any(not math.isfinite(value) for value in metrics.values()):
            raise ValueError("metric events cannot contain non-finite values")
        object.__setattr__(self, "metrics", metrics)

    @property
    def return_mean(self) -> float | None:
        return self.metrics.get("eval/episode_reward")

    @property
    def return_std(self) -> float | None:
        return self.metrics.get("eval/episode_reward_std")


MetricCallback = Callable[[HumanoidMetricEvent], None]


@dataclass(frozen=True, slots=True)
class HumanoidEvaluationConfig:
    seed: int = 0
    num_envs: int = 128
    episode_length: int | None = None

    def __post_init__(self) -> None:
        _non_negative_integer("seed", self.seed)
        _positive_integer("num_envs", self.num_envs)
        if self.episode_length is not None:
            _positive_integer("episode_length", self.episode_length)


@dataclass(frozen=True, slots=True)
class HumanoidEvaluationResult:
    task: str
    seed: int
    stage_index: int
    deterministic: bool
    episode_returns: tuple[float, ...]
    episode_lengths: tuple[int, ...]
    event: HumanoidMetricEvent

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "episode_returns",
            tuple(float(value) for value in self.episode_returns),
        )
        object.__setattr__(
            self,
            "episode_lengths",
            tuple(int(value) for value in self.episode_lengths),
        )
        if not self.deterministic:
            raise ValueError("post-training Humanoid reports must be deterministic")
        if not self.episode_returns:
            raise ValueError("evaluation requires at least one episode")
        if len(self.episode_returns) != len(self.episode_lengths):
            raise ValueError("return and episode-length counts do not match")
        if any(not math.isfinite(value) for value in self.episode_returns):
            raise ValueError("evaluation returns must be finite")
        if any(value < 0 for value in self.episode_lengths):
            raise ValueError("evaluation lengths must be non-negative")

    @property
    def return_mean(self) -> float:
        return self.event.metrics["eval/episode_reward"]

    @property
    def return_std(self) -> float:
        return self.event.metrics["eval/episode_reward_std"]

    @property
    def actual_environment_steps(self) -> int:
        return sum(self.episode_lengths)


@dataclass(frozen=True, slots=True)
class HumanoidTrainingResult:
    initialization: Initialization
    state: HumanoidPPOState
    best_state: HumanoidPPOState | None
    events: tuple[HumanoidMetricEvent, ...]
    final_return: float | None
    best_return: float | None
    report: HumanoidEvaluationResult | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "initialization", Initialization(self.initialization))
        object.__setattr__(self, "events", tuple(self.events))
        _finite_optional("final_return", self.final_return)
        _finite_optional("best_return", self.best_return)
        if (
            self.final_return is not None
            and self.best_return is not None
            and self.best_return < self.final_return
        ):
            raise ValueError("best_return cannot be smaller than final_return")

    @property
    def requested_environment_steps(self) -> int:
        return self.state.requested_environment_steps

    @property
    def actual_environment_steps(self) -> int:
        return self.state.actual_environment_steps

    @property
    def from_scratch(self) -> bool:
        return self.initialization is Initialization.FROM_SCRATCH


@dataclass(frozen=True, slots=True)
class HumanoidBackendAvailability:
    available: bool
    missing_modules: tuple[str, ...]
    detail: str


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _non_negative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")


def _finite_optional(name: str, value: float | None) -> None:
    if value is not None and not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
