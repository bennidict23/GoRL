from __future__ import annotations

from typing import Any, Protocol, Self

import jax
import jax_dataclasses as jdc
import mujoco_playground as mjp
from jax import Array
from jax import numpy as jnp
from mujoco import mjx

from flow_policy import ppo, rollouts


class Decoder(Protocol):
    action_size: int

    def decode(self, obs: Array, latent: Array) -> Array: ...


@jdc.pytree_dataclass
class LatentPpoAgent:
    """PPO encoder plus decoder.

    Rollouts store latent samples as the PPO action while the environment sees
    decoded actions.
    """

    encoder: ppo.PpoState
    decoder: Any

    @property
    def env(self) -> mjp.MjxEnv:
        return self.encoder.env

    def sample_latent_and_action(
        self, obs: Array, prng: Array, deterministic: bool
    ) -> tuple[Array, Array, ppo.PpoActionInfo]:
        latent, action_info = self.encoder.sample_action(obs, prng, deterministic)
        decoded_action = self.decoder.decode(obs, latent)
        return latent, decoded_action, action_info

    def training_step(
        self, transitions: ppo.PpoTransition
    ) -> tuple[Self, dict[str, Array]]:
        encoder, metrics = self.encoder.training_step(transitions)
        with jdc.copy_and_mutate(self) as agent:
            agent.encoder = encoder
        return agent, metrics


@jdc.pytree_dataclass
class LatentEvalOutputs:
    scalar_metrics: dict[str, Array]
    latent_actions: Array
    decoded_actions: Array
    action_timestep_mask: Array


@jdc.pytree_dataclass
class LatentRolloutState:
    env: jdc.Static[mjp.MjxEnv]
    env_state: mjp.State
    first_obs: Array
    first_data: mjx.Data
    steps: Array
    num_envs: jdc.Static[int]
    prng: Array

    @staticmethod
    @jdc.jit
    def init(
        env: jdc.Static[mjp.MjxEnv],
        prng: Array,
        num_envs: jdc.Static[int],
    ) -> LatentRolloutState:
        prng, reset_prng = jax.random.split(prng, num=2)
        state = jax.vmap(env.reset)(jax.random.split(reset_prng, num=num_envs))
        return LatentRolloutState(
            env=env,
            env_state=state,
            first_obs=state.obs,  # type: ignore[arg-type]
            first_data=state.data,
            steps=jnp.zeros_like(state.done),
            num_envs=num_envs,
            prng=prng,
        )

    @jdc.jit
    def rollout(
        self,
        agent_state: LatentPpoAgent,
        episode_length: jdc.Static[int],
        iterations_per_env: jdc.Static[int],
        auto_reset: jdc.Static[bool] = True,
        deterministic: jdc.Static[bool] = False,
    ) -> tuple[Self, rollouts.TransitionStruct[ppo.PpoActionInfo], Array]:
        def env_step(carry: LatentRolloutState, _: Any):
            state = carry
            prng_act, prng_next = jax.random.split(state.prng)
            latent, decoded_action, action_info = agent_state.sample_latent_and_action(
                state.env_state.obs, prng_act, deterministic
            )

            next_env_state = jax.vmap(state.env.step)(
                state.env_state, jnp.tanh(decoded_action)
            )

            next_steps = state.steps + 1
            truncation = next_steps >= episode_length
            done_env = next_env_state.done.astype(bool)
            done_or_tr = jnp.logical_or(done_env, truncation)
            discount = 1.0 - done_env.astype(jnp.float32)

            transition = rollouts.TransitionStruct(
                obs=state.env_state.obs,
                next_obs=next_env_state.obs,
                action=latent,
                action_info=action_info,
                reward=next_env_state.reward,
                truncation=truncation.astype(jnp.float32),
                discount=discount,
            )

            next_state = state
            if auto_reset:

                def where_done(x: Array, y: Array) -> Array:
                    return jnp.where(
                        done_or_tr.reshape(
                            done_or_tr.shape + (1,) * (x.ndim - done_or_tr.ndim)
                        ),
                        x,
                        y,
                    )

                next_env_state = next_env_state.replace(  # type: ignore[attr-defined]
                    obs=jax.tree.map(where_done, state.first_obs, next_env_state.obs),
                    data=jax.tree.map(
                        where_done, state.first_data, next_env_state.data
                    ),
                    done=jnp.zeros_like(next_env_state.done),
                )
                with jdc.copy_and_mutate(next_state) as next_state:
                    next_state.env_state = next_env_state
                    next_state.steps = jnp.where(done_or_tr, 0, state.steps + 1)
                    next_state.prng = prng_next
            else:
                with jdc.copy_and_mutate(next_state) as next_state:
                    next_state.env_state = next_env_state
                    next_state.steps = next_steps
                    next_state.prng = prng_next

            return next_state, (transition, decoded_action)

        final_state, (traj, decoded_actions) = jax.lax.scan(
            env_step, self, (), length=iterations_per_env
        )
        return final_state, traj, decoded_actions


@jdc.jit
def eval_latent_policy(
    agent_state: LatentPpoAgent,
    prng: Array,
    num_envs: jdc.Static[int],
    max_episode_length: jdc.Static[int],
) -> LatentEvalOutputs:
    rollout_state = LatentRolloutState.init(agent_state.env, prng, num_envs)
    _, transitions, decoded_actions = rollout_state.rollout(
        agent_state,
        episode_length=max_episode_length,
        iterations_per_env=max_episode_length,
        auto_reset=False,
        deterministic=True,
    )
    done_after_step = transitions.discount <= 0.0
    done_before_step = jnp.concatenate(
        [
            jnp.zeros_like(done_after_step[:1], dtype=bool),
            jnp.maximum.accumulate(done_after_step[:-1], axis=0),
        ],
        axis=0,
    )
    valid_mask = jnp.logical_not(done_before_step)
    action_mask = valid_mask[..., None]
    action_mask_count = jnp.maximum(
        jnp.sum(valid_mask) * transitions.action.shape[-1],
        1.0,
    )
    rewards = jnp.sum(transitions.reward * valid_mask, axis=0)
    steps = jnp.sum(valid_mask, axis=0)
    decode_delta = decoded_actions - transitions.action

    def masked_abs_mean(values: Array) -> Array:
        return jnp.sum(jnp.abs(values) * action_mask) / action_mask_count

    def masked_abs_max(values: Array) -> Array:
        return jnp.max(jnp.where(action_mask, jnp.abs(values), 0.0))

    scalar_metrics = {
        "reward_mean": jnp.mean(rewards),
        "reward_min": jnp.min(rewards),
        "reward_max": jnp.max(rewards),
        "reward_std": jnp.std(rewards),
        "steps_mean": jnp.mean(steps),
        "steps_min": jnp.min(steps),
        "steps_max": jnp.max(steps),
        "steps_std": jnp.std(steps),
        "latent_abs_mean": masked_abs_mean(transitions.action),
        "decoded_abs_mean": masked_abs_mean(decoded_actions),
        "decode_delta_abs_mean": masked_abs_mean(decode_delta),
        "decode_delta_abs_max": masked_abs_max(decode_delta),
    }

    return LatentEvalOutputs(
        scalar_metrics=scalar_metrics,
        latent_actions=transitions.action,
        decoded_actions=decoded_actions,
        action_timestep_mask=valid_mask,
    )
