"""Checkpoint I/O for canonical Humanoid PPO state."""

from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

from .types import HumanoidPPOState, STATE_SCHEMA_VERSION


def save_state(state: HumanoidPPOState, path: str | Path) -> Path:
    """Atomically save a host-backed state.

    The pickle format matches the historical JAX checkpoint strategy. Only
    load checkpoints from trusted sources.
    """

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "gorl_humanoid_ppo_state",
        "schema_version": STATE_SCHEMA_VERSION,
        "state": _host_backed_state(state),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def load_state(path: str | Path) -> HumanoidPPOState:
    """Load a trusted canonical Humanoid state checkpoint."""

    source = Path(path).expanduser().resolve()
    with source.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid Humanoid checkpoint payload: {source}")
    if payload.get("format") != "gorl_humanoid_ppo_state":
        raise ValueError(f"not a canonical Humanoid checkpoint: {source}")
    if payload.get("schema_version") != STATE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported Humanoid checkpoint schema: {payload.get('schema_version')!r}"
        )
    state = payload.get("state")
    if not isinstance(state, HumanoidPPOState):
        raise ValueError(f"Humanoid checkpoint does not contain PPO state: {source}")
    return state


def _host_backed_state(state: HumanoidPPOState) -> HumanoidPPOState:
    try:
        import jax
    except ImportError:
        return state

    params = jax.tree_util.tree_map(
        _to_host_array,
        state.brax_params,
    )
    return HumanoidPPOState(
        task=state.task,
        seed=state.seed,
        stage_index=state.stage_index,
        decoder_kind=state.decoder_kind,
        action_size=state.action_size,
        latent_size=state.latent_size,
        requested_environment_steps=state.requested_environment_steps,
        actual_environment_steps=state.actual_environment_steps,
        ppo_parameters=state.ppo_parameters,
        normalizer_params=params[0],
        policy_params=params[1],
        value_params=params[2],
        training_eval_deterministic=state.training_eval_deterministic,
        action_semantics=state.action_semantics,
        schema_version=state.schema_version,
    )


def _to_host_array(value: Any) -> Any:
    try:
        import numpy as np

        return np.asarray(value)
    except (TypeError, ValueError):
        return value
