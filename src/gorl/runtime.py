"""Dependency-light process runtime configuration."""

from __future__ import annotations

import os
from typing import Any, Mapping, MutableMapping

from gorl.humanoid.execution import resolve_humanoid_ppo_execution_contract
from gorl.humanoid.types import HumanoidPPOExecutionProfile


_TRITON_FLAG = "--xla_gpu_triton_gemm_any"
_AUTOTUNE_FLAG = "--xla_gpu_autotune_level"


def _without_owned_flag(raw_flags: str | None, flag: str) -> list[str]:
    prefix = f"{flag}="
    return [
        token
        for token in (raw_flags or "").split()
        if token != flag and not token.startswith(prefix)
    ]


def xla_flags_with_triton(
    raw_flags: str | None,
    value: bool | None,
) -> str | None:
    """Replace the GoRL-owned Triton flag while retaining unrelated flags."""

    flags = _without_owned_flag(raw_flags, _TRITON_FLAG)
    if value is not None:
        flags.append(f"{_TRITON_FLAG}={value}")
    return " ".join(flags) or None


def xla_flags_with_autotune(
    raw_flags: str | None,
    value: int | None,
) -> str | None:
    """Replace the GoRL-owned GPU autotune flag while retaining other flags."""

    flags = _without_owned_flag(raw_flags, _AUTOTUNE_FLAG)
    if value is not None:
        flags.append(f"{_AUTOTUNE_FLAG}={value}")
    return " ".join(flags) or None


def apply_process_runtime(
    environment: MutableMapping[str, str],
    *,
    matmul_precision: str | None,
    triton_gemm: bool | None,
    autotune_level: int | None = None,
) -> None:
    """Apply runtime values before a process imports JAX."""

    flags = xla_flags_with_triton(environment.get("XLA_FLAGS"), triton_gemm)
    flags = xla_flags_with_autotune(flags, autotune_level)
    if flags is None:
        environment.pop("XLA_FLAGS", None)
    else:
        environment["XLA_FLAGS"] = flags

    if matmul_precision in {None, "default"}:
        environment.pop("JAX_DEFAULT_MATMUL_PRECISION", None)
    else:
        environment["JAX_DEFAULT_MATMUL_PRECISION"] = str(matmul_precision)


def humanoid_gorl_runtime_plan(config: Any) -> dict[str, Any]:
    """Resolve teacher and downstream runtime values for Humanoid GoRL.

    Downstream omission inherits the task runtime. Triton accepts a boolean or
    ``"unset"``; GPU autotune accepts a non-negative integer or ``"unset"``;
    matmul precision accepts one of the validated JAX precision names.
    """

    runtime = config.section("runtime")
    training = config.section("training")
    execution_profile = HumanoidPPOExecutionProfile(
        runtime.get("humanoid_ppo_execution_profile", "native")
    )
    execution_contract = resolve_humanoid_ppo_execution_contract(execution_profile)
    teacher_precision = str(runtime.get("jax_default_matmul_precision", "default"))
    downstream_precision = str(
        training.get("downstream_jax_default_matmul_precision", teacher_precision)
    )
    teacher_triton = runtime.get("xla_gpu_triton_gemm_any")
    if teacher_triton is not None:
        teacher_triton = bool(teacher_triton)
    teacher_autotune = runtime.get("xla_gpu_autotune_level")

    has_downstream_triton = "downstream_xla_gpu_triton_gemm_any" in training
    configured_downstream_triton = training.get(
        "downstream_xla_gpu_triton_gemm_any",
        teacher_triton,
    )
    if has_downstream_triton:
        downstream_triton = (
            None
            if configured_downstream_triton == "unset"
            else bool(configured_downstream_triton)
        )
        downstream_source = "training_override"
    else:
        downstream_triton = teacher_triton
        downstream_source = "inherited_runtime"

    has_downstream_autotune = "downstream_xla_gpu_autotune_level" in training
    configured_downstream_autotune = training.get(
        "downstream_xla_gpu_autotune_level",
        teacher_autotune,
    )
    if has_downstream_autotune:
        downstream_autotune = (
            None
            if configured_downstream_autotune == "unset"
            else int(configured_downstream_autotune)
        )
        downstream_autotune_source = "training_override"
    else:
        downstream_autotune = teacher_autotune
        downstream_autotune_source = "inherited_runtime"

    runtime_separation_required = (
        downstream_precision != teacher_precision
        or downstream_triton != teacher_triton
        or downstream_autotune != teacher_autotune
    )
    teacher_process_required = (
        runtime_separation_required or execution_contract.teacher.process_required
    )
    latent_process_required = execution_contract.latent.process_required
    decoder_fit_process_required = (
        execution_profile is HumanoidPPOExecutionProfile.OFFICIAL_DIRECT_COMPAT
    )

    return {
        "schema_version": 1,
        "execution_profile": execution_profile.value,
        "execution_contract": execution_contract.to_dict(),
        "teacher": {
            "jax_default_matmul_precision": teacher_precision,
            "xla_gpu_triton_gemm_any": {
                "configured": teacher_triton,
                "effective": teacher_triton,
                "source": "runtime",
            },
            "xla_gpu_autotune_level": {
                "configured": teacher_autotune,
                "effective": teacher_autotune,
                "source": "runtime",
            },
        },
        "downstream": {
            "jax_default_matmul_precision": downstream_precision,
            "xla_gpu_triton_gemm_any": {
                "configured": configured_downstream_triton,
                "effective": downstream_triton,
                "source": downstream_source,
            },
            "xla_gpu_autotune_level": {
                "configured": configured_downstream_autotune,
                "effective": downstream_autotune,
                "source": downstream_autotune_source,
            },
        },
        "teacher_process_required": teacher_process_required,
        "latent_process_required": latent_process_required,
        "decoder_fit_process_required": decoder_fit_process_required,
        "process_requirements": {
            "teacher": {
                "required": teacher_process_required,
                "execution_contract": execution_contract.teacher.process_required,
                "runtime_separation": runtime_separation_required,
            },
            "latent": {
                "required": latent_process_required,
                "execution_contract": execution_contract.latent.process_required,
                "runtime_separation": False,
            },
            "decoder_fit": {
                "required": decoder_fit_process_required,
                "execution_profile": execution_profile.value,
                "runtime_separation": False,
            },
        },
    }


def runtime_values_for_main_process(
    config: Any,
) -> tuple[str | None, bool | None, int | None]:
    """Return the runtime that the public CLI must apply before imports."""

    runtime: Mapping[str, Any] = config.section("runtime")
    precision = runtime.get("jax_default_matmul_precision")
    triton = runtime.get("xla_gpu_triton_gemm_any")
    autotune = runtime.get("xla_gpu_autotune_level")
    if config.profile == "humanoid" and config.method in {
        "gorl_fm",
        "gorl_diffusion",
    }:
        plan = humanoid_gorl_runtime_plan(config)
        downstream = plan["downstream"]
        precision = downstream["jax_default_matmul_precision"]
        triton = downstream["xla_gpu_triton_gemm_any"]["effective"]
        autotune = downstream["xla_gpu_autotune_level"]["effective"]
    return precision, triton, autotune


def current_runtime_environment() -> dict[str, str | None]:
    return {
        "JAX_DEFAULT_MATMUL_PRECISION": os.environ.get("JAX_DEFAULT_MATMUL_PRECISION"),
        "XLA_FLAGS": os.environ.get("XLA_FLAGS"),
    }
