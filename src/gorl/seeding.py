"""Deterministic component-seed resolution and provenance."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


SeedRole = Literal["primary", "teacher", "decoder", "encoder", "evaluation"]
SeedSchedule = Literal["stage_offset", "fixed"]


@dataclass(frozen=True, slots=True)
class SeedResolution:
    """A concrete seed together with the rule that produced it."""

    seed: int
    role: SeedRole
    schedule: SeedSchedule
    stage: int | None
    base_seed: int
    namespace_offset: int
    stage_offset: int

    def to_dict(self) -> dict[str, int | str | None]:
        return asdict(self)


def resolve_seed(
    *,
    base_seed: int,
    role: SeedRole,
    schedule: SeedSchedule,
    stage: int | None,
    namespace_offset: int = 0,
    stage_stride: int = 1,
) -> SeedResolution:
    """Resolve a fixed or stage-offset seed without hiding its provenance."""

    if base_seed < 0:
        raise ValueError("base_seed must be non-negative")
    if role not in {"primary", "teacher", "decoder", "encoder", "evaluation"}:
        raise ValueError(f"unsupported seed role: {role!r}")
    if schedule not in {"stage_offset", "fixed"}:
        raise ValueError(f"unsupported seed schedule: {schedule!r}")
    if stage is not None and stage < 0:
        raise ValueError("stage must be non-negative when set")
    if namespace_offset < 0:
        raise ValueError("namespace_offset must be non-negative")
    if stage_stride < 0:
        raise ValueError("stage_stride must be non-negative")
    if schedule == "stage_offset" and stage is None:
        raise ValueError("stage_offset schedule requires a stage")

    stage_offset = 0 if schedule == "fixed" else int(stage) * stage_stride
    seed = base_seed + namespace_offset + stage_offset
    return SeedResolution(
        seed=seed,
        role=role,
        schedule=schedule,
        stage=stage,
        base_seed=base_seed,
        namespace_offset=namespace_offset,
        stage_offset=stage_offset,
    )
