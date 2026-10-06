"""Dependency-light Humanoid PPO execution contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from gorl.humanoid.types import (
    HumanoidPPOExecutionProfile,
    HumanoidPPOPhase,
)


EXECUTION_CONTRACT_VERSION = 1


@dataclass(frozen=True, slots=True)
class HumanoidPPOPhaseExecutionContract:
    """Immutable orchestration semantics for one PPO phase."""

    phase: HumanoidPPOPhase
    process_required: bool
    import_surface: str
    xla_flag_timing: str
    evaluation_environment: str
    restore_api: str
    checkpoint_mode: str
    event_delivery: str
    brax_default_forwarding: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase", HumanoidPPOPhase(self.phase))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class HumanoidPPOExecutionContract:
    """Versioned atomic contract shared by teacher and latent PPO phases."""

    profile: HumanoidPPOExecutionProfile
    worker_model: str
    teacher: HumanoidPPOPhaseExecutionContract
    latent: HumanoidPPOPhaseExecutionContract
    contract_version: int = EXECUTION_CONTRACT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "profile",
            HumanoidPPOExecutionProfile(self.profile),
        )
        if self.teacher.phase is not HumanoidPPOPhase.TEACHER:
            raise ValueError("teacher execution contract must describe teacher phase")
        if self.latent.phase is not HumanoidPPOPhase.LATENT:
            raise ValueError("latent execution contract must describe latent phase")
        if self.contract_version != EXECUTION_CONTRACT_VERSION:
            raise ValueError(
                f"unsupported execution contract version: {self.contract_version}"
            )

    def for_phase(
        self,
        phase: HumanoidPPOPhase | str,
    ) -> HumanoidPPOPhaseExecutionContract:
        selected = HumanoidPPOPhase(phase)
        return self.teacher if selected is HumanoidPPOPhase.TEACHER else self.latent

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_humanoid_ppo_execution_contract(
    profile: HumanoidPPOExecutionProfile | str = HumanoidPPOExecutionProfile.NATIVE,
) -> HumanoidPPOExecutionContract:
    """Resolve a public profile into its complete, non-tunable contract."""

    selected = HumanoidPPOExecutionProfile(profile)
    if selected is HumanoidPPOExecutionProfile.NATIVE:
        return HumanoidPPOExecutionContract(
            profile=selected,
            worker_model="phase_runtime_dependent",
            teacher=HumanoidPPOPhaseExecutionContract(
                phase=HumanoidPPOPhase.TEACHER,
                process_required=False,
                import_surface="gorl_native",
                xla_flag_timing="prestart",
                evaluation_environment="shared_training_instance",
                restore_api="restore_params",
                checkpoint_mode="public_best_after_eval",
                event_delivery="callback_json",
                brax_default_forwarding="explicit_resolved_values",
            ),
            latent=HumanoidPPOPhaseExecutionContract(
                phase=HumanoidPPOPhase.LATENT,
                process_required=False,
                import_surface="gorl_native",
                xla_flag_timing="main_process_runtime",
                evaluation_environment="wrapped_training_instance",
                restore_api="restore_params",
                checkpoint_mode="public_best_after_eval",
                event_delivery="callback_json",
                brax_default_forwarding="explicit_resolved_values",
            ),
        )

    return HumanoidPPOExecutionContract(
        profile=selected,
        worker_model="per_ppo_phase_subprocess",
        teacher=HumanoidPPOPhaseExecutionContract(
            phase=HumanoidPPOPhase.TEACHER,
            process_required=True,
            import_surface="official_direct_teacher",
            xla_flag_timing="append_triton_after_import_surface",
            evaluation_environment="separate_registry_instance",
            restore_api="restore_checkpoint_path",
            checkpoint_mode="orbax_each_eval_before_eval",
            event_delivery="replay_after_process",
            brax_default_forwarding="omit_equal_official_defaults",
        ),
        latent=HumanoidPPOPhaseExecutionContract(
            phase=HumanoidPPOPhase.LATENT,
            process_required=True,
            import_surface="official_direct_latent",
            xla_flag_timing="owned_flags_unset",
            evaluation_environment="wrapped_training_instance",
            restore_api="restore_params",
            checkpoint_mode="latest_policy_best_after_eval",
            event_delivery="replay_after_process",
            brax_default_forwarding="omit_equal_official_defaults",
        ),
    )


def resolve_humanoid_ppo_phase_execution_contract(
    profile: HumanoidPPOExecutionProfile | str,
    phase: HumanoidPPOPhase | str,
) -> HumanoidPPOPhaseExecutionContract:
    return resolve_humanoid_ppo_execution_contract(profile).for_phase(phase)
