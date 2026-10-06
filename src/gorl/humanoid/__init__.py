"""Dependency-light contracts for the internal Humanoid backend.

Training is intentionally exposed through ``gorl train`` rather than package-level
backend functions.
"""

from .types import (
    STATE_SCHEMA_VERSION,
    SUPPORTED_TASKS,
    DecoderKind,
    EventSource,
    HumanoidBackendAvailability,
    HumanoidEvaluationConfig,
    HumanoidEvaluationResult,
    HumanoidLatentStageConfig,
    HumanoidMetricEvent,
    HumanoidNonFiniteMetricError,
    HumanoidPPOConfig,
    HumanoidPPOExecutionProfile,
    HumanoidPPOPhase,
    HumanoidPPOState,
    HumanoidTrainingResult,
    HumanoidWarmStart,
    Initialization,
    MetricCallback,
)

__all__ = [
    "STATE_SCHEMA_VERSION",
    "SUPPORTED_TASKS",
    "DecoderKind",
    "EventSource",
    "HumanoidBackendAvailability",
    "HumanoidEvaluationConfig",
    "HumanoidEvaluationResult",
    "HumanoidLatentStageConfig",
    "HumanoidMetricEvent",
    "HumanoidNonFiniteMetricError",
    "HumanoidPPOConfig",
    "HumanoidPPOExecutionProfile",
    "HumanoidPPOPhase",
    "HumanoidPPOState",
    "HumanoidTrainingResult",
    "HumanoidWarmStart",
    "Initialization",
    "MetricCallback",
]
