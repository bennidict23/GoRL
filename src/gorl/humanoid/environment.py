"""Frozen-decoder environment adapter for bounded Humanoid actions."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class FrozenDecoder(Protocol):
    """Minimum decoder surface consumed by the Humanoid PPO backend."""

    @property
    def action_size(self) -> int: ...

    def decode(self, observation: Any, latent: Any) -> Any: ...


class BoundedLatentActionEnvironment:
    """Duck-typed MuJoCo Playground wrapper with a frozen decoder.

    Brax optimizes only its policy and value parameter tuple. The decoder is
    captured by this environment object and therefore remains frozen. Both the
    latent policy and decoded environment action follow the historical bounded
    Humanoid recipe: Brax emits tanh-normal latent actions, then decoder output
    receives a final ``[-1, 1]`` safety clip.
    """

    def __init__(
        self,
        environment: Any,
        decoder: FrozenDecoder,
        *,
        latent_size: int | None = None,
        record_decoded_actions: bool = False,
    ) -> None:
        if not hasattr(decoder, "decode") or not hasattr(decoder, "action_size"):
            raise TypeError("decoder must expose action_size and decode(obs, latent)")
        action_size = int(decoder.action_size)
        if action_size <= 0:
            raise ValueError("decoder action_size must be positive")
        resolved_latent_size = action_size if latent_size is None else latent_size
        if resolved_latent_size <= 0:
            raise ValueError("latent_size must be positive")
        if resolved_latent_size != action_size:
            raise ValueError(
                "Humanoid profile currently requires latent_size == action_size; "
                f"got {resolved_latent_size} != {action_size}"
            )
        if int(environment.action_size) != action_size:
            raise ValueError(
                "decoder and environment action sizes differ: "
                f"{action_size} != {environment.action_size}"
            )
        self.env = environment
        self.decoder = decoder
        self._latent_size = int(resolved_latent_size)
        self._record_decoded_actions = bool(record_decoded_actions)

    @property
    def action_size(self) -> int:
        return self._latent_size

    @property
    def observation_size(self) -> Any:
        return self.env.observation_size

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped

    def __getattr__(self, name: str) -> Any:
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self.env, name)

    def reset(self, rng: Any) -> Any:
        return self.env.reset(rng)

    def decode_action(self, observation: Any, latent: Any) -> Any:
        return self.decoder.decode(observation, latent)

    def step(self, state: Any, latent: Any) -> Any:
        from jax import numpy as jnp

        decoded_raw = self.decode_action(state.obs, latent)
        bounded_action = jnp.clip(decoded_raw, -1.0, 1.0)
        next_state = self.env.step(state, bounded_action)

        if self._record_decoded_actions:
            info = dict(next_state.info)
            info["latent_action"] = latent
            info["decoded_action_raw"] = decoded_raw
            info["decoded_action"] = bounded_action
            next_state = next_state.replace(info=info)
        return next_state
