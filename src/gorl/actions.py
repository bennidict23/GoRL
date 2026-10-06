"""Action-space conventions shared by trainers and decoders."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


class ActionSemantics(StrEnum):
    """Meaning of the action tensor produced by a policy or decoder."""

    LEGACY_PRE_TANH = "legacy_pre_tanh"
    BOUNDED = "bounded"

    @property
    def environment_transform(self) -> str:
        """Transform applied immediately before stepping the environment."""

        if self is ActionSemantics.LEGACY_PRE_TANH:
            return "tanh"
        return "clip"

    @property
    def legacy_cli_value(self) -> str:
        """Value understood by the existing component scripts."""

        if self is ActionSemantics.LEGACY_PRE_TANH:
            return "pre_tanh"
        return "bounded"


@dataclass(frozen=True, slots=True)
class ActionSpec:
    """Action-domain contract at the policy/decoder boundary.

    ``legacy_pre_tanh`` stores and models unbounded actions. The rollout code
    applies ``tanh`` before calling the environment. ``bounded`` models the
    environment action directly and clips only as a final safety guard.
    """

    semantics: ActionSemantics
    dimension: int | None = None
    minimum: float = -1.0
    maximum: float = 1.0

    def __post_init__(self) -> None:
        if self.dimension is not None and self.dimension <= 0:
            raise ValueError(f"action dimension must be positive, got {self.dimension}")
        if not math.isfinite(self.minimum) or not math.isfinite(self.maximum):
            raise ValueError("action bounds must be finite")
        if self.minimum >= self.maximum:
            raise ValueError(
                "action minimum must be smaller than maximum, "
                f"got [{self.minimum}, {self.maximum}]"
            )
        if self.semantics is ActionSemantics.LEGACY_PRE_TANH and (
            self.minimum != -1.0 or self.maximum != 1.0
        ):
            raise ValueError(
                "legacy_pre_tanh requires environment bounds [-1, 1], "
                f"got [{self.minimum}, {self.maximum}]"
            )

    @property
    def decoder_domain(self) -> str:
        if self.semantics is ActionSemantics.LEGACY_PRE_TANH:
            return "unbounded"
        return "bounded"

    @property
    def requires_environment_tanh(self) -> bool:
        return self.semantics is ActionSemantics.LEGACY_PRE_TANH
