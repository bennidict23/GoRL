"""Trajectory collection for supervised GoRL decoder updates."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np

from .latent_ppo import LatentPpoAgent, LatentRolloutState


@dataclass(frozen=True, slots=True)
class DatasetStats:
    num_transitions: int
    num_episodes: int
    return_mean: float
    return_std: float
    return_min: float
    return_max: float
    length_mean: float
    length_min: int
    length_max: int


@dataclass(frozen=True, slots=True)
class CollectedDataset:
    arrays: dict[str, np.ndarray]
    stats: DatasetStats
    deterministic_policy: bool
    collection_seed: int
    transition_order: str = "batched_time_major"
    rng_schedule: str = "split_chain"

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".npz", dir=output.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            np.savez_compressed(temporary, **self.arrays)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, output)
        finally:
            if temporary.exists():
                temporary.unlink()
        return output


def collect_dataset(
    *,
    agent: LatentPpoAgent,
    num_episodes: int,
    num_envs: int = 64,
    episode_length: int,
    seed: int,
    deterministic_policy: bool = False,
) -> CollectedDataset:
    if num_episodes < 1 or num_envs < 1 or episode_length < 1:
        raise ValueError("num_episodes, num_envs, and episode_length must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    pieces: dict[str, list[np.ndarray]] = {}
    collection_prng = jax.random.key(seed)
    completed = 0
    while completed < num_episodes:
        batch_size = min(num_envs, num_episodes - completed)
        collection_prng, batch_prng = jax.random.split(collection_prng)
        rollout = LatentRolloutState.init(
            agent.env,
            prng=batch_prng,
            num_envs=batch_size,
        )
        _, transitions, decoded_actions = rollout.rollout(
            agent,
            episode_length=episode_length,
            iterations_per_env=episode_length,
            auto_reset=False,
            deterministic=deterministic_policy,
        )
        batch = flatten_valid_transitions(transitions, decoded_actions)
        batch["episode_id"] = batch["episode_id"] + completed
        for name, value in batch.items():
            if name == "episode_rank":
                continue
            pieces.setdefault(name, []).append(value)
        completed += batch_size

    arrays = {name: np.concatenate(values, axis=0) for name, values in pieces.items()}
    returns = arrays["episode_returns"]
    rank_order = np.argsort(-returns)
    episode_ranks = np.empty_like(rank_order)
    episode_ranks[rank_order] = np.arange(rank_order.size)
    arrays["episode_rank"] = episode_ranks[arrays["episode_id"]].astype(
        np.int32,
        copy=False,
    )
    return CollectedDataset(
        arrays=arrays,
        stats=dataset_stats(arrays),
        deterministic_policy=deterministic_policy,
        collection_seed=seed,
    )


def flatten_valid_transitions(
    transitions: Any,
    decoded_actions: Any,
) -> dict[str, np.ndarray]:
    obs = np.asarray(jax.device_get(transitions.obs))
    next_obs = np.asarray(jax.device_get(transitions.next_obs))
    latent = np.asarray(jax.device_get(transitions.action))
    action_pre_tanh = np.asarray(jax.device_get(decoded_actions))
    env_action = np.tanh(action_pre_tanh)
    reward = np.asarray(jax.device_get(transitions.reward))
    discount = np.asarray(jax.device_get(transitions.discount))
    truncation = np.asarray(jax.device_get(transitions.truncation))

    if reward.ndim != 2:
        raise ValueError(f"expected reward shape [time, episode], got {reward.shape}")
    if obs.shape[:2] != reward.shape or next_obs.shape[:2] != reward.shape:
        raise ValueError("observation leading dimensions do not match rewards")
    if latent.shape[:2] != reward.shape or action_pre_tanh.shape[:2] != reward.shape:
        raise ValueError("action leading dimensions do not match rewards")

    done_after_step = discount <= 0.0
    done_before_step = np.concatenate(
        (
            np.zeros_like(done_after_step[:1], dtype=bool),
            np.maximum.accumulate(done_after_step[:-1], axis=0),
        ),
        axis=0,
    )
    valid = ~done_before_step
    episode_returns = (reward * valid).sum(axis=0)
    episode_lengths = valid.sum(axis=0)
    rank_order = np.argsort(-episode_returns)
    episode_ranks = np.empty_like(rank_order)
    episode_ranks[rank_order] = np.arange(rank_order.size)

    steps = np.broadcast_to(np.arange(reward.shape[0])[:, None], reward.shape)
    episode_ids = np.broadcast_to(np.arange(reward.shape[1])[None, :], reward.shape)
    flat_episode_ids = episode_ids[valid]
    return {
        "obs": obs[valid].astype(np.float32, copy=False),
        "next_obs": next_obs[valid].astype(np.float32, copy=False),
        "latent_action": latent[valid].astype(np.float32, copy=False),
        "action_pre_tanh": action_pre_tanh[valid].astype(np.float32, copy=False),
        "env_action": env_action[valid].astype(np.float32, copy=False),
        "reward": reward[valid].astype(np.float32, copy=False),
        "discount": discount[valid].astype(np.float32, copy=False),
        "truncation": truncation[valid].astype(np.float32, copy=False),
        "episode_id": flat_episode_ids.astype(np.int32, copy=False),
        "step": steps[valid].astype(np.int32, copy=False),
        "episode_return": episode_returns[flat_episode_ids].astype(
            np.float32, copy=False
        ),
        "episode_rank": episode_ranks[flat_episode_ids].astype(np.int32, copy=False),
        "episode_returns": episode_returns.astype(np.float32, copy=False),
        "episode_lengths": episode_lengths.astype(np.int32, copy=False),
    }


def dataset_stats(arrays: dict[str, np.ndarray]) -> DatasetStats:
    returns = arrays["episode_returns"]
    lengths = arrays["episode_lengths"]
    if returns.size == 0:
        raise ValueError("dataset has no episodes")
    return DatasetStats(
        num_transitions=int(arrays["obs"].shape[0]),
        num_episodes=int(returns.shape[0]),
        return_mean=float(np.mean(returns)),
        return_std=float(np.std(returns)),
        return_min=float(np.min(returns)),
        return_max=float(np.max(returns)),
        length_mean=float(np.mean(lengths)),
        length_min=int(np.min(lengths)),
        length_max=int(np.max(lengths)),
    )
