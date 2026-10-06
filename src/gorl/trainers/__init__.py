"""Dependency-light contracts used by the unified ``gorl train`` command."""

from .types import (
    Method,
    PPOProfile,
    RunStatus,
    TrainingRequest,
    TrainingResult,
)

__all__ = [
    "Method",
    "PPOProfile",
    "RunStatus",
    "TrainingRequest",
    "TrainingResult",
]
