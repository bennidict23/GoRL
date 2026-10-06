"""Decoder identifiers shared by dependency-light pipeline contracts."""

from __future__ import annotations

from enum import StrEnum


class DecoderKind(StrEnum):
    IDENTITY = "identity"
    FM = "fm"
    DIFFUSION = "diffusion"
