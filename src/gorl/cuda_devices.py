"""Resolve CUDA device selections without importing a compute framework."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


class CudaDeviceSelectionError(ValueError):
    """Raised when a requested CUDA device cannot be selected safely."""


@dataclass(frozen=True, slots=True)
class CudaDeviceSelection:
    """Requested, inherited, and worker-visible CUDA device identifiers."""

    requested: tuple[str, ...]
    inherited: str | None
    resolved: tuple[str, ...]


def _device_token(value: object, *, label: str) -> str:
    if isinstance(value, bool):
        raise CudaDeviceSelectionError(f"{label} entries must be CUDA device IDs")
    token = str(value)
    if not token or token.isspace():
        raise CudaDeviceSelectionError(f"{label} entries must be non-empty")
    if "," in token or any(character.isspace() for character in token):
        raise CudaDeviceSelectionError(
            f"invalid {label} entry {token!r}; pass IDs as separate arguments"
        )
    if token == "-1":
        raise CudaDeviceSelectionError(f"{label} cannot contain -1")
    if token.startswith("-") or (
        not token.isdigit()
        and not token.startswith("GPU-")
        and not token.startswith("MIG-")
    ):
        raise CudaDeviceSelectionError(
            f"invalid {label} entry {token!r}; use a non-negative ordinal, "
            "GPU UUID, or MIG UUID"
        )
    return token


def _unique(values: Sequence[str], *, label: str) -> tuple[str, ...]:
    if not values:
        raise CudaDeviceSelectionError(f"at least one {label} entry is required")
    seen: set[str] = set()
    selected: list[str] = []
    for value in values:
        if value in seen:
            raise CudaDeviceSelectionError(f"duplicate {label} entry: {value}")
        seen.add(value)
        selected.append(value)
    return tuple(selected)


def _unique_mappings(values: Sequence[str], *, label: str) -> tuple[str, ...]:
    seen: set[tuple[str, str | int]] = set()
    for value in values:
        identity: tuple[str, str | int] = (
            ("index", int(value)) if value.isdigit() else ("token", value)
        )
        if identity in seen:
            raise CudaDeviceSelectionError(f"duplicate {label} mapping: {value}")
        seen.add(identity)
    return tuple(values)


def resolve_cuda_devices(
    requested: Sequence[object],
    inherited: str | None,
) -> CudaDeviceSelection:
    """Resolve worker devices inside an optional parent CUDA allocation.

    With no inherited ``CUDA_VISIBLE_DEVICES``, requested system indices and
    UUIDs pass through unchanged. With an inherited list, numeric requests are
    logical ordinals into that list and UUID requests must match an inherited
    token exactly. The function performs no CUDA, JAX, or NVML calls.
    """

    requested_tokens = _unique(
        tuple(_device_token(value, label="requested GPU") for value in requested),
        label="requested GPU",
    )
    if inherited is None:
        resolved_tokens = _unique_mappings(
            requested_tokens,
            label="resolved GPU",
        )
        return CudaDeviceSelection(
            requested=requested_tokens,
            inherited=None,
            resolved=resolved_tokens,
        )

    raw_inherited_tokens = tuple(token.strip() for token in inherited.split(","))
    if not raw_inherited_tokens or any(not token for token in raw_inherited_tokens):
        raise CudaDeviceSelectionError(
            "CUDA_VISIBLE_DEVICES does not contain any usable CUDA devices"
        )
    inherited_tokens = _unique(
        tuple(
            _device_token(token, label="CUDA_VISIBLE_DEVICES")
            for token in raw_inherited_tokens
        ),
        label="CUDA_VISIBLE_DEVICES",
    )
    _unique_mappings(inherited_tokens, label="CUDA_VISIBLE_DEVICES")

    resolved: list[str] = []
    for token in requested_tokens:
        if token.isdigit():
            ordinal = int(token)
            if ordinal >= len(inherited_tokens):
                raise CudaDeviceSelectionError(
                    f"logical ordinal {token} is outside the inherited "
                    f"CUDA_VISIBLE_DEVICES allocation of {len(inherited_tokens)} "
                    "device(s)"
                )
            resolved.append(inherited_tokens[ordinal])
            continue
        if token not in inherited_tokens:
            raise CudaDeviceSelectionError(
                f"requested GPU UUID {token!r} is not present in the inherited "
                "CUDA_VISIBLE_DEVICES allocation"
            )
        resolved.append(token)

    resolved_tokens = _unique_mappings(
        _unique(tuple(resolved), label="resolved GPU"),
        label="resolved GPU",
    )
    return CudaDeviceSelection(
        requested=requested_tokens,
        inherited=inherited,
        resolved=resolved_tokens,
    )
