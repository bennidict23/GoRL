"""PPO stage training for the dm_control GoRL profile."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Literal

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
from mujoco_playground import registry
from mujoco_playground.config import dm_control_suite_params

from flow_policy import ppo

from .latent_ppo import LatentPpoAgent, LatentRolloutState, eval_latent_policy


class EncoderInitialization(StrEnum):
    FRESH = "fresh"
    WARM_START = "warm_start"
    GAUSSIAN = "gaussian"
    GAUSSIAN_WARM_VALUE = "gaussian_warm_value"


@dataclass(frozen=True, slots=True)
class PPOStageConfig:
    task: str
    seed: int
    num_timesteps: int
    eval_seed: int | None = None
    num_envs: int = 2048
    num_eval_envs: int = 128
    episode_length: int = 1000
    batch_size: int = 1024
    num_minibatches: int = 32
    unroll_length: int = 30
    num_updates_per_batch: int = 16
    num_evals: int = 10
    action_repeat: int = 1
    learning_rate: float = 1e-3
    clipping_epsilon: float = 0.3
    entropy_cost: float = 0.01
    discounting: float = 0.995
    gae_lambda: float = 0.95
    reward_scaling: float = 10.0
    value_loss_coeff: float = 0.25
    normalize_observations: bool = True
    normalize_advantage: bool = True
    z_regularization: float = 1e-3
    max_grad_norm: float = 0.5
    max_policy_scale: float = 10.0
    tanh_entropy_correction: bool = False
    observation_stats_accumulation: Literal["cumulative", "legacy_batch_only"] = (
        "cumulative"
    )
    observation_stats_timing: Literal["rollout_consistent", "legacy_pre_update"] = (
        "rollout_consistent"
    )
    rollout_seed_offset: int = 1
    encoder_initialization: EncoderInitialization = EncoderInitialization.FRESH

    def __post_init__(self) -> None:
        if not self.task:
            raise ValueError("task must be non-empty")
        for name in (
            "seed",
            "num_timesteps",
            "num_envs",
            "num_eval_envs",
            "episode_length",
            "batch_size",
            "num_minibatches",
            "unroll_length",
            "num_updates_per_batch",
            "num_evals",
            "action_repeat",
        ):
            value = getattr(self, name)
            minimum = 0 if name == "seed" else 1
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}, got {value}")
        if self.eval_seed is not None and self.eval_seed < 0:
            raise ValueError("eval_seed must be non-negative when set")
        for name in (
            "learning_rate",
            "clipping_epsilon",
            "entropy_cost",
            "discounting",
            "gae_lambda",
            "z_regularization",
            "max_grad_norm",
            "max_policy_scale",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("reward_scaling", "value_loss_coeff"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.discounting > 1.0 or self.gae_lambda > 1.0:
            raise ValueError("discounting and gae_lambda must not exceed 1")
        if self.action_repeat != 1:
            raise ValueError("the dm_control profile requires action_repeat=1")
        if self.observation_stats_accumulation not in {
            "cumulative",
            "legacy_batch_only",
        }:
            raise ValueError(
                "observation_stats_accumulation must be 'cumulative' or "
                "'legacy_batch_only'"
            )
        if self.observation_stats_timing not in {
            "rollout_consistent",
            "legacy_pre_update",
        }:
            raise ValueError(
                "observation_stats_timing must be 'rollout_consistent' or "
                "'legacy_pre_update'"
            )
        if self.seed + self.rollout_seed_offset < 0:
            raise ValueError("seed + rollout_seed_offset must be non-negative")


@dataclass(frozen=True, slots=True)
class StageEvaluation:
    index: int
    env_steps: int
    return_mean: float
    return_std: float
    return_min: float
    return_max: float


@dataclass(frozen=True, slots=True)
class PPOStageOutput:
    agent: LatentPpoAgent
    best_agent: LatentPpoAgent
    evaluations: tuple[StageEvaluation, ...]
    requested_env_steps: int
    actual_env_steps: int
    steps_per_update: int
    last_train_metrics: dict[str, float]

    @property
    def final_return(self) -> float:
        return self.evaluations[-1].return_mean

    @property
    def best_return(self) -> float:
        return max(record.return_mean for record in self.evaluations)


EvalCallback = Callable[[StageEvaluation], None]
TrainCallback = Callable[[dict[str, float]], None]


def build_ppo_config(config: PPOStageConfig) -> ppo.PpoConfig:
    params = dm_control_suite_params.brax_ppo_config(config.task)
    params.action_repeat = config.action_repeat
    params.num_timesteps = config.num_timesteps
    params.num_envs = config.num_envs
    params.num_evals = config.num_evals
    params.episode_length = config.episode_length
    params.batch_size = config.batch_size
    params.num_minibatches = config.num_minibatches
    params.unroll_length = config.unroll_length
    params.num_updates_per_batch = config.num_updates_per_batch
    params.learning_rate = config.learning_rate
    params.entropy_cost = config.entropy_cost
    params.discounting = config.discounting
    params.gae_lambda = config.gae_lambda
    params.reward_scaling = config.reward_scaling
    params.value_loss_coeff = config.value_loss_coeff
    params.normalize_observations = config.normalize_observations
    params.normalize_advantage = config.normalize_advantage
    params.clipping_epsilon = config.clipping_epsilon
    params.z_regularization = config.z_regularization
    params.max_grad_norm = config.max_grad_norm
    params.max_policy_scale = config.max_policy_scale
    params.use_tanh_entropy_correction = config.tanh_entropy_correction
    params.observation_stats_accumulation = config.observation_stats_accumulation
    params.observation_stats_timing = config.observation_stats_timing
    return ppo.PpoConfig(**params)  # type: ignore[arg-type]


def gaussian_policy_params(policy_params: Any, action_size: int) -> Any:
    """Initialize the output projection so the policy starts as N(0, I)."""

    layers = list(policy_params)
    final_weights, final_bias = layers[-1]
    if final_bias.shape[-1] != action_size * 2:
        raise ValueError(
            "policy output size does not match the action dimension: "
            f"{final_bias.shape[-1]} != {action_size * 2}"
        )
    scale_raw = jnp.log(jnp.expm1(jnp.asarray(1.0 - 1e-3, dtype=final_bias.dtype)))
    gaussian_bias = jnp.concatenate(
        (
            jnp.zeros((action_size,), dtype=final_bias.dtype),
            jnp.full((action_size,), scale_raw, dtype=final_bias.dtype),
        )
    )
    layers[-1] = (jnp.zeros_like(final_weights), gaussian_bias)
    return tuple(layers)


def _validate_tree_compatible(saved: Any, fresh: Any, name: str) -> None:
    def check(saved_leaf: Any, fresh_leaf: Any) -> None:
        if getattr(saved_leaf, "shape", None) != getattr(fresh_leaf, "shape", None):
            raise ValueError(f"incompatible {name} leaf shapes")
        if getattr(saved_leaf, "dtype", None) != getattr(fresh_leaf, "dtype", None):
            raise ValueError(f"incompatible {name} leaf dtypes")

    jax.tree.map(check, saved, fresh)


def initialize_agent(
    *,
    env: Any,
    decoder: Any,
    config: PPOStageConfig,
    ppo_config: ppo.PpoConfig,
    previous_agent: LatentPpoAgent | None = None,
) -> LatentPpoAgent:
    decoder_action_size = int(getattr(decoder, "action_size"))
    if decoder_action_size != int(env.action_size):
        raise ValueError(
            "decoder action size does not match the environment: "
            f"{decoder_action_size} != {env.action_size}"
        )
    decoder_obs_size = int(getattr(decoder, "obs_size", env.observation_size))
    if decoder_obs_size != int(env.observation_size):
        raise ValueError(
            "decoder observation size does not match the environment: "
            f"{decoder_obs_size} != {env.observation_size}"
        )

    encoder = ppo.PpoState.init(
        prng=jax.random.key(config.seed),
        env=env,
        config=ppo_config,
    )
    mode = config.encoder_initialization
    if (
        mode
        in {
            EncoderInitialization.WARM_START,
            EncoderInitialization.GAUSSIAN_WARM_VALUE,
        }
        and previous_agent is None
    ):
        raise ValueError(f"{mode.value} requires a previous agent")
    if (
        mode
        in {
            EncoderInitialization.FRESH,
            EncoderInitialization.GAUSSIAN,
        }
        and previous_agent is not None
    ):
        raise ValueError(f"{mode.value} does not accept a previous agent")

    if mode is EncoderInitialization.WARM_START:
        assert previous_agent is not None
        saved = previous_agent.encoder
        _validate_tree_compatible(saved.params, encoder.params, "policy parameters")
        _validate_tree_compatible(
            saved.obs_stats, encoder.obs_stats, "observation statistics"
        )
        with jdc.copy_and_mutate(encoder) as encoder:
            encoder.params = saved.params
            encoder.obs_stats = saved.obs_stats
    elif mode is EncoderInitialization.GAUSSIAN:
        with jdc.copy_and_mutate(encoder) as encoder:
            encoder.params = ppo.ActorCriticParams(
                policy=gaussian_policy_params(
                    encoder.params.policy, int(env.action_size)
                ),
                value=encoder.params.value,
            )
    elif mode is EncoderInitialization.GAUSSIAN_WARM_VALUE:
        assert previous_agent is not None
        saved = previous_agent.encoder
        _validate_tree_compatible(saved.params, encoder.params, "policy parameters")
        _validate_tree_compatible(
            saved.obs_stats, encoder.obs_stats, "observation statistics"
        )
        with jdc.copy_and_mutate(encoder) as encoder:
            encoder.params = ppo.ActorCriticParams(
                policy=gaussian_policy_params(
                    encoder.params.policy, int(env.action_size)
                ),
                value=saved.params.value,
            )
            encoder.obs_stats = saved.obs_stats

    return LatentPpoAgent(encoder=encoder, decoder=decoder)


def _evaluate(
    agent: LatentPpoAgent,
    config: PPOStageConfig,
    eval_index: int,
    env_steps: int,
) -> StageEvaluation:
    eval_seed = config.seed if config.eval_seed is None else config.eval_seed
    output = eval_latent_policy(
        agent,
        prng=jax.random.fold_in(jax.random.key(eval_seed), eval_index),
        num_envs=config.num_eval_envs,
        max_episode_length=config.episode_length,
    )
    metrics = output.scalar_metrics
    return StageEvaluation(
        index=eval_index,
        env_steps=env_steps,
        return_mean=float(np.asarray(metrics["reward_mean"])),
        return_std=float(np.asarray(metrics["reward_std"])),
        return_min=float(np.asarray(metrics["reward_min"])),
        return_max=float(np.asarray(metrics["reward_max"])),
    )


def _mean_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {name: float(np.mean(np.asarray(value))) for name, value in metrics.items()}


def train_stage(
    *,
    decoder: Any,
    config: PPOStageConfig,
    previous_agent: LatentPpoAgent | None = None,
    eval_callback: EvalCallback | None = None,
    train_callback: TrainCallback | None = None,
) -> PPOStageOutput:
    """Train one PPO encoder stage and evaluate the post-update checkpoint."""

    env_config = registry.get_default_config(config.task)
    env = registry.load(config.task, config=env_config)
    ppo_config = build_ppo_config(config)
    steps_per_update = ppo_config.iterations_per_env * ppo_config.num_envs
    outer_updates = config.num_timesteps // steps_per_update
    if outer_updates < 1:
        raise ValueError(
            "num_timesteps is smaller than one PPO update: "
            f"{config.num_timesteps} < {steps_per_update}"
        )

    agent = initialize_agent(
        env=env,
        decoder=decoder,
        config=config,
        ppo_config=ppo_config,
        previous_agent=previous_agent,
    )
    rollout = LatentRolloutState.init(
        env,
        prng=jax.random.key(config.seed + config.rollout_seed_offset),
        num_envs=ppo_config.num_envs,
    )

    evaluations: list[StageEvaluation] = []
    initial = _evaluate(agent, config, eval_index=0, env_steps=0)
    evaluations.append(initial)
    best_agent = agent
    best_return = initial.return_mean
    if eval_callback is not None:
        eval_callback(initial)

    eval_updates = {
        int(index)
        for index in np.linspace(
            1,
            outer_updates,
            max(config.num_evals - 1, 1),
            dtype=int,
        )
    }
    eval_updates.add(outer_updates)
    last_train_metrics: dict[str, float] = {}

    for update in range(1, outer_updates + 1):
        rollout, transitions, decoded_actions = rollout.rollout(
            agent,
            episode_length=ppo_config.episode_length,
            iterations_per_env=ppo_config.iterations_per_env,
        )
        agent, raw_metrics = agent.training_step(transitions)
        env_steps = update * steps_per_update
        last_train_metrics = {
            "update": float(update),
            "env_steps": float(env_steps),
            "reward_mean": float(np.mean(np.asarray(transitions.reward))),
            "latent_abs_mean": float(np.mean(np.abs(np.asarray(transitions.action)))),
            "decoded_abs_mean": float(np.mean(np.abs(np.asarray(decoded_actions)))),
            **_mean_metrics(raw_metrics),
        }
        if train_callback is not None:
            train_callback(last_train_metrics)

        if update in eval_updates:
            record = _evaluate(
                agent,
                config,
                eval_index=update,
                env_steps=env_steps,
            )
            evaluations.append(record)
            if record.return_mean > best_return:
                best_return = record.return_mean
                best_agent = agent
            if eval_callback is not None:
                eval_callback(record)

    return PPOStageOutput(
        agent=agent,
        best_agent=best_agent,
        evaluations=tuple(evaluations),
        requested_env_steps=config.num_timesteps,
        actual_env_steps=outer_updates * steps_per_update,
        steps_per_update=steps_per_update,
        last_train_metrics=last_train_metrics,
    )
