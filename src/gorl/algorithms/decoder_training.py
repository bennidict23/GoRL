"""Dataset preparation and supervised decoder training for dm_control GoRL."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import optax
from jax import Array

from .decoders import (
    DiffusionParameterization,
    DiffusionSchedule,
    DiffusionDecoder,
    ObsNormalizedDecoder,
    ReverseTimeFlowMatchingDecoder,
)


DecoderMethod = Literal["fm", "diffusion"]
ActionSource = Literal["auto", "pre_tanh", "bounded"]
CheckpointMetric = Literal[
    "val_loss",
    "val_endpoint_mse",
    "val_endpoint_abs_mean",
]
DatasetSplitUnit = Literal["episode", "transition"]
BatchSampling = Literal["with_replacement", "epoch_permutation"]
RngSchedule = Literal[
    "independent_streams",
    "legacy_interleaved",
    "state_threaded",
]
ValidationSchedule = Literal["interval_sampled", "epoch_full_batches"]
WarmObservationStatsMode = Literal[
    "cumulative",
    "frozen",
    "legacy_new_batch",
]


@dataclass(frozen=True, slots=True)
class DecoderTrainingConfig:
    """Configuration shared by reverse-CFM and diffusion decoder training."""

    method: DecoderMethod
    seed: int = 0
    train_steps: int = 50_000
    batch_size: int = 8_192
    learning_rate: float = 3e-4
    validation_fraction: float = 0.1
    eval_interval: int = 100
    eval_batches: int = 8
    endpoint_eval_size: int = 4_096
    checkpoint_metric: CheckpointMetric = "val_loss"
    action_source: ActionSource = "auto"
    action_clip: float | None = None
    max_transitions: int | None = None
    normalize_observations: bool = True
    observation_epsilon: float = 1e-8
    hidden_dims: tuple[int, ...] = (128, 128, 128, 128)
    decoder_steps: int = 10
    timestep_embed_dim: int = 8
    output_scale: float | None = None
    n_samples_per_action: int = 8
    pairwise_loss_weight: float = 0.0
    zero_final_layer: bool = True
    diffusion_alpha_min: float = 1e-3
    diffusion_alpha_max: float = 1.0
    diffusion_parameterization: DiffusionParameterization = "epsilon_delta"
    diffusion_schedule: DiffusionSchedule = "linear_alpha_bar"
    split_unit: DatasetSplitUnit = "episode"
    batch_sampling: BatchSampling = "with_replacement"
    rng_schedule: RngSchedule = "independent_streams"
    validation_schedule: ValidationSchedule = "interval_sampled"
    early_stopping_patience: int | None = None
    warm_observation_stats_mode: WarmObservationStatsMode = "cumulative"

    def __post_init__(self) -> None:
        if self.method not in {"fm", "diffusion"}:
            raise ValueError(f"unsupported decoder method: {self.method!r}")
        if self.checkpoint_metric not in {
            "val_loss",
            "val_endpoint_mse",
            "val_endpoint_abs_mean",
        }:
            raise ValueError(
                f"unsupported checkpoint_metric: {self.checkpoint_metric!r}"
            )
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        for name in (
            "train_steps",
            "batch_size",
            "eval_interval",
            "eval_batches",
            "endpoint_eval_size",
            "decoder_steps",
            "n_samples_per_action",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < self.learning_rate:
            raise ValueError("learning_rate must be positive")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in (0, 1)")
        if self.action_source not in {"auto", "pre_tanh", "bounded"}:
            raise ValueError(f"unsupported action_source: {self.action_source!r}")
        if self.action_clip is not None and self.action_clip <= 0.0:
            raise ValueError("action_clip must be positive when set")
        if self.max_transitions is not None and self.max_transitions < 1:
            raise ValueError("max_transitions must be positive when set")
        if self.observation_epsilon <= 0.0:
            raise ValueError("observation_epsilon must be positive")
        if not self.hidden_dims or any(width < 1 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")
        if self.timestep_embed_dim < 2 or self.timestep_embed_dim % 2:
            raise ValueError("timestep_embed_dim must be a positive even number")
        if self.output_scale is not None and self.output_scale <= 0.0:
            raise ValueError("output_scale must be positive when set")
        if (
            not np.isfinite(self.pairwise_loss_weight)
            or self.pairwise_loss_weight < 0.0
        ):
            raise ValueError("pairwise_loss_weight must be finite and non-negative")
        if not (0.0 < self.diffusion_alpha_min < self.diffusion_alpha_max <= 1.0):
            raise ValueError(
                "diffusion alphas must satisfy 0 < alpha_min < alpha_max <= 1"
            )
        if self.diffusion_parameterization not in {"epsilon_delta", "epsilon"}:
            raise ValueError(
                "diffusion_parameterization must be 'epsilon_delta' or 'epsilon'"
            )
        if self.diffusion_schedule not in {
            "linear_alpha_bar",
            "cosine_alpha_bar",
        }:
            raise ValueError(
                "diffusion_schedule must be 'linear_alpha_bar' or 'cosine_alpha_bar'"
            )
        if self.split_unit not in {"episode", "transition"}:
            raise ValueError("split_unit must be 'episode' or 'transition'")
        if self.batch_sampling not in {
            "with_replacement",
            "epoch_permutation",
        }:
            raise ValueError(
                "batch_sampling must be 'with_replacement' or 'epoch_permutation'"
            )
        if self.rng_schedule not in {
            "independent_streams",
            "legacy_interleaved",
            "state_threaded",
        }:
            raise ValueError(
                "rng_schedule must be 'independent_streams', "
                "'legacy_interleaved', or 'state_threaded'"
            )
        if self.validation_schedule not in {
            "interval_sampled",
            "epoch_full_batches",
        }:
            raise ValueError(
                "validation_schedule must be 'interval_sampled' or 'epoch_full_batches'"
            )
        if self.warm_observation_stats_mode not in {
            "cumulative",
            "frozen",
            "legacy_new_batch",
        }:
            raise ValueError(
                "warm_observation_stats_mode must be 'cumulative', 'frozen', "
                "or 'legacy_new_batch'"
            )
        if (
            self.early_stopping_patience is not None
            and self.early_stopping_patience < 1
        ):
            raise ValueError("early_stopping_patience must be positive when set")
        state_threaded = self.rng_schedule == "state_threaded"
        epoch_validation = self.validation_schedule == "epoch_full_batches"
        if state_threaded != epoch_validation:
            raise ValueError(
                "state_threaded RNG and epoch_full_batches validation schedules "
                "must be enabled together"
            )
        if state_threaded:
            if self.split_unit != "transition":
                raise ValueError(
                    "state-threaded schedules require split_unit='transition'"
                )
            if self.batch_sampling != "epoch_permutation":
                raise ValueError(
                    "state-threaded schedules require "
                    "batch_sampling='epoch_permutation'"
                )
            if self.checkpoint_metric != "val_loss":
                raise ValueError(
                    "state-threaded schedules require checkpoint_metric='val_loss'"
                )
            if self.early_stopping_patience is None:
                raise ValueError(
                    "state-threaded schedules require early_stopping_patience"
                )
        if self.rng_schedule == "legacy_interleaved":
            if self.split_unit != "episode":
                raise ValueError("legacy_interleaved RNG requires split_unit='episode'")
            if self.batch_sampling != "with_replacement":
                raise ValueError(
                    "legacy_interleaved RNG requires batch_sampling='with_replacement'"
                )
            if self.validation_schedule != "interval_sampled":
                raise ValueError(
                    "legacy_interleaved RNG requires "
                    "validation_schedule='interval_sampled'"
                )

    @property
    def resolved_output_scale(self) -> float:
        if self.output_scale is not None:
            return self.output_scale
        return 1.0 if self.method == "fm" else 0.25


@dataclass(frozen=True, slots=True)
class DecoderDataset:
    """Canonical transition dataset used by both decoder objectives."""

    obs: np.ndarray
    action: np.ndarray
    episode_id: np.ndarray
    episode_return: np.ndarray
    action_key: str
    action_is_bounded: bool
    latent: np.ndarray | None = None
    latent_key: str | None = None

    def __post_init__(self) -> None:
        obs = np.asarray(self.obs, dtype=np.float32)
        action = np.asarray(self.action, dtype=np.float32)
        episode_id = np.asarray(self.episode_id)
        episode_return = np.asarray(self.episode_return, dtype=np.float32)
        latent = (
            None if self.latent is None else np.asarray(self.latent, dtype=np.float32)
        )

        if obs.ndim != 2 or action.ndim != 2:
            raise ValueError("obs and action must both have shape (transitions, dim)")
        if obs.shape[0] < 1 or obs.shape[0] != action.shape[0]:
            raise ValueError("obs and action must have the same non-zero length")
        if episode_id.ndim != 1 or episode_return.ndim != 1:
            raise ValueError("episode_id and episode_return must be one-dimensional")
        if episode_id.shape[0] != obs.shape[0]:
            raise ValueError("episode_id must have one value per transition")
        if episode_return.shape[0] != obs.shape[0]:
            raise ValueError("episode_return must have one value per transition")
        if latent is not None and latent.shape != action.shape:
            raise ValueError("latent and action must have identical shapes")
        if (latent is None) != (self.latent_key is None):
            raise ValueError(
                "latent and latent_key must either both be set or both be None"
            )
        if not np.issubdtype(episode_id.dtype, np.integer):
            if not np.all(np.equal(episode_id, np.floor(episode_id))):
                raise ValueError("episode_id must contain integer values")
        if not all(
            np.all(np.isfinite(value))
            for value in (obs, action, episode_return, latent)
            if value is not None
        ):
            raise ValueError("decoder dataset contains non-finite values")
        if self.action_is_bounded and np.max(np.abs(action)) > 1.0 + 1e-5:
            raise ValueError("bounded actions must lie in [-1, 1]")

        object.__setattr__(self, "obs", obs)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "episode_id", episode_id.astype(np.int64))
        object.__setattr__(self, "episode_return", episode_return)
        object.__setattr__(self, "latent", latent)

    @property
    def num_transitions(self) -> int:
        return self.obs.shape[0]

    @property
    def num_episodes(self) -> int:
        return np.unique(self.episode_id).size

    @property
    def obs_size(self) -> int:
        return self.obs.shape[1]

    @property
    def action_size(self) -> int:
        return self.action.shape[1]

    def take(self, indices: np.ndarray) -> "DecoderDataset":
        return DecoderDataset(
            obs=self.obs[indices],
            action=self.action[indices],
            episode_id=self.episode_id[indices],
            episode_return=self.episode_return[indices],
            action_key=self.action_key,
            action_is_bounded=self.action_is_bounded,
            latent=(None if self.latent is None else self.latent[indices]),
            latent_key=self.latent_key,
        )

    def with_action(self, action: np.ndarray) -> "DecoderDataset":
        return DecoderDataset(
            obs=self.obs,
            action=action,
            episode_id=self.episode_id,
            episode_return=self.episode_return,
            action_key=self.action_key,
            action_is_bounded=self.action_is_bounded,
            latent=self.latent,
            latent_key=self.latent_key,
        )


@dataclass(frozen=True, slots=True)
class DecoderDatasetSplit:
    train: DecoderDataset
    validation: DecoderDataset
    train_episode_ids: tuple[int, ...]
    validation_episode_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ObservationStats:
    mean: np.ndarray
    std: np.ndarray
    eps: float
    enabled: bool
    count: float = 0.0
    var_sum: np.ndarray | None = None

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        if mean.ndim != 1 or std.shape != mean.shape:
            raise ValueError("observation mean/std must be matching vectors")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
            raise ValueError("observation statistics must be finite")
        if np.any(std < 0.0) or self.eps <= 0.0:
            raise ValueError("observation std must be non-negative and eps positive")
        if not np.isfinite(self.count) or self.count < 0.0:
            raise ValueError("observation count must be finite and non-negative")
        var_sum = (
            np.square(std, dtype=np.float32) * np.float32(self.count)
            if self.var_sum is None
            else np.asarray(self.var_sum, dtype=np.float32)
        )
        if var_sum.shape != mean.shape or not np.all(np.isfinite(var_sum)):
            raise ValueError("observation var_sum must match mean and be finite")
        if np.any(var_sum < -1e-6):
            raise ValueError("observation var_sum must be non-negative")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)
        object.__setattr__(self, "count", float(self.count))
        object.__setattr__(self, "var_sum", np.maximum(var_sum, 0.0))

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        value = np.asarray(obs, dtype=np.float32)
        if not self.enabled:
            return value
        return ((value - self.mean) / (self.std + self.eps)).astype(np.float32)

    def update(self, obs: np.ndarray) -> "ObservationStats":
        """Merge a new observation batch using the historical Welford update."""

        if not self.enabled:
            return self
        value = np.asarray(obs, dtype=np.float32)
        if value.ndim != 2 or value.shape[1:] != self.mean.shape:
            raise ValueError("warm observation batch has incompatible shape")
        if value.shape[0] < 1 or not np.all(np.isfinite(value)):
            raise ValueError("warm observation batch must be non-empty and finite")

        old_mean = self.mean.astype(np.float64)
        new_count = self.count + value.shape[0]
        difference = value.astype(np.float64) - old_mean
        new_mean = old_mean + difference.sum(axis=0) / new_count
        new_var_sum = self.var_sum.astype(np.float64) + np.sum(
            difference * (value.astype(np.float64) - new_mean),
            axis=0,
        )
        new_var_sum = np.maximum(new_var_sum, 0.0)
        new_std = np.sqrt(new_var_sum / new_count)
        return ObservationStats(
            mean=new_mean.astype(np.float32),
            std=new_std.astype(np.float32),
            eps=self.eps,
            enabled=True,
            count=new_count,
            var_sum=new_var_sum.astype(np.float32),
        )

    def update_float32(self, obs: np.ndarray) -> "ObservationStats":
        """Merge observations using the decoder's state-threaded float32 path."""

        return self._update_float32(obs, include_previous_var_sum=True)

    def update_float32_legacy_new_batch(
        self,
        obs: np.ndarray,
    ) -> "ObservationStats":
        """Reproduce the historical warm-start variance update.

        The original Humanoid decoder trainer restored the previous count and
        mean but accidentally omitted its accumulated variance when merging a
        new stage.  Keeping this behavior behind an explicit compatibility mode
        lets historical recipes remain reproducible without making it the
        default for new experiments.
        """

        return self._update_float32(obs, include_previous_var_sum=False)

    def _update_float32(
        self,
        obs: np.ndarray,
        *,
        include_previous_var_sum: bool,
    ) -> "ObservationStats":
        """Merge observations with the selected float32 variance semantics."""

        if not self.enabled:
            return self
        value = jnp.asarray(obs, dtype=jnp.float32)
        if value.ndim != 2 or value.shape[1:] != self.mean.shape:
            raise ValueError("warm observation batch has incompatible shape")
        if value.shape[0] < 1 or not np.all(np.isfinite(np.asarray(value))):
            raise ValueError("warm observation batch must be non-empty and finite")

        old_mean = jnp.asarray(self.mean, dtype=jnp.float32)
        new_count = jnp.asarray(self.count, dtype=jnp.float32) + value.shape[0]
        difference = value - old_mean
        new_mean = old_mean + jnp.sum(difference, axis=0) / new_count
        batch_var_sum = jnp.sum(
            difference * (value - new_mean),
            axis=0,
        )
        new_var_sum = batch_var_sum
        if include_previous_var_sum:
            new_var_sum = jnp.asarray(self.var_sum, dtype=jnp.float32) + batch_var_sum
        new_std = jnp.sqrt(jnp.clip(new_var_sum / new_count, 1e-12, 1e12))
        return ObservationStats(
            mean=np.asarray(new_mean),
            std=np.asarray(new_std),
            eps=self.eps,
            enabled=True,
            count=float(np.asarray(new_count)),
            var_sum=np.asarray(new_var_sum),
        )


@dataclass(frozen=True, slots=True)
class PreparedDecoderData:
    split: DecoderDatasetSplit
    train_obs: np.ndarray
    validation_obs: np.ndarray
    observation_stats: ObservationStats
    action_clip_fraction: float


def train_steps_from_epochs(
    epochs: int,
    data: PreparedDecoderData | int,
    *,
    batch_size: int,
) -> int:
    """Convert legacy decoder epochs to their exact number of updates.

    The legacy diffusion trainer dropped each epoch's incomplete final batch
    and still performed one update when the dataset was smaller than a batch.
    This helper preserves that update count. ``batch_sampling`` controls whether
    ``fit_decoder`` traverses each epoch without replacement or only preserves
    the count while drawing batches with replacement.

    Args:
        epochs: Number of legacy training epochs.
        data: Prepared data, or its number of training transitions.
        batch_size: Requested training batch size.
    """

    if epochs < 1:
        raise ValueError("epochs must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    num_train_transitions = (
        data.split.train.num_transitions
        if isinstance(data, PreparedDecoderData)
        else data
    )
    if (
        isinstance(num_train_transitions, bool)
        or not isinstance(num_train_transitions, (int, np.integer))
        or num_train_transitions < 1
    ):
        raise ValueError("num_train_transitions must be a positive integer")
    effective_batch_size = min(batch_size, int(num_train_transitions))
    updates_per_epoch = max(
        1,
        int(num_train_transitions) // effective_batch_size,
    )
    return epochs * updates_per_epoch


def _iter_training_batch_indices(
    config: DecoderTrainingConfig,
    *,
    num_train_transitions: int,
    num_total_transitions: int,
    legacy_rng: np.random.Generator | None = None,
) -> Iterator[np.ndarray]:
    """Yield deterministic training batches for the configured traversal."""

    if config.batch_sampling == "with_replacement":
        if config.rng_schedule == "legacy_interleaved":
            if legacy_rng is None:
                raise ValueError(
                    "legacy_interleaved training batches require a shared RNG"
                )
            rng = legacy_rng
        else:
            if legacy_rng is not None:
                raise ValueError(
                    "a shared legacy RNG is valid only for legacy_interleaved"
                )
            rng = np.random.Generator(np.random.PCG64(config.seed + 2))
        for _ in range(config.train_steps):
            yield rng.integers(
                0,
                num_train_transitions,
                size=config.batch_size,
            )
        return

    rng = np.random.RandomState(config.seed)
    if config.split_unit == "transition":
        # Advance through the same split permutation used by
        # ``split_decoder_dataset`` before drawing the first epoch order.
        rng.permutation(num_total_transitions)
    effective_batch_size = min(config.batch_size, num_train_transitions)
    updates_per_epoch = max(1, num_train_transitions // effective_batch_size)
    permutation: np.ndarray | None = None
    for step in range(config.train_steps):
        batch_in_epoch = step % updates_per_epoch
        if batch_in_epoch == 0:
            permutation = rng.permutation(num_train_transitions)
        assert permutation is not None
        start = batch_in_epoch * effective_batch_size
        yield permutation[start : start + effective_batch_size]


@dataclass(frozen=True, slots=True)
class DecoderFitResult:
    """Best-validation decoder plus the final optimization state."""

    decoder: Any
    final_decoder: Any
    best_step: int
    best_metric: str
    best_value: float
    best_metrics: Mapping[str, float | int]
    final_metrics: Mapping[str, float | int]
    history: tuple[Mapping[str, float | int], ...]
    observation_stats: ObservationStats
    train_episode_ids: tuple[int, ...]
    validation_episode_ids: tuple[int, ...]
    action_key: str
    action_is_bounded: bool
    action_clip_fraction: float


@dataclass(frozen=True, slots=True)
class DecoderWarmStart:
    """Parameters and observation moments restored from a previous decoder."""

    method: DecoderMethod
    params: Any
    observation_stats: ObservationStats | None

    def __post_init__(self) -> None:
        if self.method not in {"fm", "diffusion"}:
            raise ValueError(f"unsupported warm-start method: {self.method!r}")
        leaves = jax.tree_util.tree_leaves(self.params)
        if not leaves:
            raise ValueError("warm-start params must be a non-empty pytree")
        if not all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves):
            raise ValueError("warm-start params contain non-finite values")

    @classmethod
    def from_decoder(
        cls,
        decoder: Any,
        *,
        observation_stats: ObservationStats | None,
    ) -> "DecoderWarmStart":
        raw_decoder = (
            decoder.decoder if isinstance(decoder, ObsNormalizedDecoder) else decoder
        )
        if isinstance(raw_decoder, ReverseTimeFlowMatchingDecoder):
            method: DecoderMethod = "fm"
        elif isinstance(raw_decoder, DiffusionDecoder):
            method = "diffusion"
        else:
            raise TypeError(
                "warm decoder must be ReverseTimeFlowMatchingDecoder, "
                "DiffusionDecoder, or ObsNormalizedDecoder wrapping one"
            )
        return cls(
            method=method,
            params=raw_decoder.params,
            observation_stats=observation_stats,
        )

    @classmethod
    def from_fit_result(
        cls,
        result: DecoderFitResult,
    ) -> "DecoderWarmStart":
        return cls.from_decoder(
            result.decoder,
            observation_stats=result.observation_stats,
        )


class DiffusionTrainingTargets(NamedTuple):
    """Noisy sample and identity-centered epsilon-delta target."""

    x_t: Array
    epsilon_delta: Array
    identity_epsilon: Array
    t_next: Array
    alpha_t: Array
    alpha_next: Array
    alpha_ratio: Array
    ddim_delta_scale: Array


class DirectEpsilonTrainingTargets(NamedTuple):
    """Noisy sample and direct epsilon target for standard DDPM training."""

    x_t: Array
    epsilon: Array
    alpha_t: Array


def _select_action_key(names: set[str], source: ActionSource) -> tuple[str, bool]:
    if source in {"auto", "pre_tanh"} and "action_pre_tanh" in names:
        return "action_pre_tanh", False
    if source == "pre_tanh":
        raise ValueError("action_source='pre_tanh' requires action_pre_tanh")

    bounded_keys = ("env_action", "action_bounded", "bounded_action", "action")
    for key in bounded_keys:
        if key in names:
            return key, True
    raise ValueError(
        "dataset must contain action_pre_tanh or a bounded action key "
        "(env_action, action_bounded, bounded_action, action)"
    )


def _select_latent_key(names: set[str]) -> str | None:
    for key in ("latent_action", "latent", "latents"):
        if key in names:
            return key
    return None


def _transition_returns(
    arrays: Mapping[str, np.ndarray],
    episode_id: np.ndarray,
    num_transitions: int,
) -> np.ndarray:
    for key in ("episode_return", "return"):
        if key in arrays:
            values = np.asarray(arrays[key], dtype=np.float32).reshape(-1)
            if values.shape[0] == num_transitions:
                return values
            return _expand_episode_returns(values, episode_id, key)
    if "episode_returns" in arrays:
        values = np.asarray(arrays["episode_returns"], dtype=np.float32).reshape(-1)
        return _expand_episode_returns(values, episode_id, "episode_returns")
    raise ValueError("dataset must contain episode_return, return, or episode_returns")


def _expand_episode_returns(
    values: np.ndarray, episode_id: np.ndarray, key: str
) -> np.ndarray:
    unique_ids = np.unique(episode_id)
    if values.shape[0] != unique_ids.shape[0]:
        raise ValueError(f"{key} must have one value per transition or one per episode")
    lookup = {episode: values[index] for index, episode in enumerate(unique_ids)}
    return np.asarray([lookup[episode] for episode in episode_id], np.float32)


def decoder_dataset_from_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    action_source: ActionSource = "auto",
    include_latent: bool = False,
) -> DecoderDataset:
    """Build canonical decoder data from an in-memory collector mapping."""

    names = set(arrays)
    if "obs" not in names or "episode_id" not in names:
        raise ValueError("dataset must contain obs and episode_id")
    action_key, action_is_bounded = _select_action_key(names, action_source)
    latent_key = _select_latent_key(names) if include_latent else None
    obs = np.asarray(arrays["obs"], dtype=np.float32)
    episode_id = np.asarray(arrays["episode_id"])
    if episode_id.ndim != 1:
        raise ValueError("episode_id must be one-dimensional")
    return DecoderDataset(
        obs=obs,
        action=np.asarray(arrays[action_key], dtype=np.float32),
        episode_id=episode_id,
        episode_return=_transition_returns(arrays, episode_id, obs.shape[0]),
        action_key=action_key,
        action_is_bounded=action_is_bounded,
        latent=(
            None
            if latent_key is None
            else np.asarray(arrays[latent_key], dtype=np.float32)
        ),
        latent_key=latent_key,
    )


def load_decoder_dataset(
    path: str | Path,
    *,
    action_source: ActionSource = "auto",
    max_transitions: int | None = None,
    include_latent: bool = False,
) -> DecoderDataset:
    """Load the canonical NPZ representation without changing its targets."""

    if max_transitions is not None and max_transitions < 1:
        raise ValueError("max_transitions must be positive when set")
    dataset_path = Path(path).expanduser()
    try:
        with np.load(dataset_path, allow_pickle=False) as archive:
            names = set(archive.files)
            if "obs" not in names or "episode_id" not in names:
                raise ValueError("dataset must contain obs and episode_id")
            action_key, action_is_bounded = _select_action_key(
                names,
                action_source,
            )
            latent_key = _select_latent_key(names) if include_latent else None
            return_key = next(
                (
                    candidate
                    for candidate in (
                        "episode_return",
                        "return",
                        "episode_returns",
                    )
                    if candidate in names
                ),
                None,
            )
            if return_key is None:
                raise ValueError(
                    "dataset must contain episode_return, return, or episode_returns"
                )

            # NPZ members decompress independently. Avoid materializing
            # collector diagnostics that the decoder never consumes.
            obs = np.asarray(archive["obs"], dtype=np.float32)
            action = np.asarray(archive[action_key], dtype=np.float32)
            latent = (
                None
                if latent_key is None
                else np.asarray(archive[latent_key], dtype=np.float32)
            )
            episode_id = np.asarray(archive["episode_id"])
            if episode_id.ndim != 1:
                raise ValueError("episode_id must be one-dimensional")
            return_values = np.asarray(
                archive[return_key],
                dtype=np.float32,
            )
    except (OSError, ValueError) as error:
        raise ValueError(
            f"could not load decoder dataset {dataset_path}: {error}"
        ) from error

    episode_return = _transition_returns(
        {return_key: return_values},
        episode_id,
        obs.shape[0],
    )
    if max_transitions is not None:
        limit = min(max_transitions, obs.shape[0])
        obs = obs[:limit]
        action = action[:limit]
        latent = None if latent is None else latent[:limit]
        episode_id = episode_id[:limit]
        episode_return = episode_return[:limit]
    return DecoderDataset(
        obs=obs,
        action=action,
        episode_id=episode_id,
        episode_return=episode_return,
        action_key=action_key,
        action_is_bounded=action_is_bounded,
        latent=latent,
        latent_key=latent_key,
    )


def split_decoder_dataset(
    dataset: DecoderDataset,
    *,
    validation_fraction: float,
    seed: int,
    split_unit: DatasetSplitUnit = "episode",
) -> DecoderDatasetSplit:
    """Split decoder data deterministically by whole episode or transition."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    if split_unit not in {"episode", "transition"}:
        raise ValueError("split_unit must be 'episode' or 'transition'")

    if split_unit == "transition":
        if dataset.num_transitions < 2:
            raise ValueError("transition-level split requires at least two samples")
        train_count = int(dataset.num_transitions * (1.0 - validation_fraction))
        train_count = min(max(train_count, 1), dataset.num_transitions - 1)
        # RandomState intentionally matches the NumPy permutation stream used
        # by the historical GoRL decoder trainer.
        shuffled = np.random.RandomState(seed).permutation(dataset.num_transitions)
        train_indices = shuffled[:train_count]
        validation_indices = shuffled[train_count:]
        train = dataset.take(train_indices)
        validation = dataset.take(validation_indices)
        return DecoderDatasetSplit(
            train=train,
            validation=validation,
            train_episode_ids=tuple(
                int(value) for value in np.unique(train.episode_id)
            ),
            validation_episode_ids=tuple(
                int(value) for value in np.unique(validation.episode_id)
            ),
        )

    episode_ids = np.unique(dataset.episode_id)
    if episode_ids.size < 2:
        raise ValueError("episode-level split requires at least two episodes")

    rng = np.random.Generator(np.random.PCG64(seed))
    shuffled = rng.permutation(episode_ids)
    validation_count = max(1, int(episode_ids.size * validation_fraction))
    validation_count = min(validation_count, episode_ids.size - 1)
    validation_ids = np.sort(shuffled[:validation_count])
    train_ids = np.sort(shuffled[validation_count:])
    validation_mask = np.isin(dataset.episode_id, validation_ids)
    train_indices = np.flatnonzero(~validation_mask)
    validation_indices = np.flatnonzero(validation_mask)

    return DecoderDatasetSplit(
        train=dataset.take(train_indices),
        validation=dataset.take(validation_indices),
        train_episode_ids=tuple(int(value) for value in train_ids),
        validation_episode_ids=tuple(int(value) for value in validation_ids),
    )


def compute_observation_stats(
    train_obs: np.ndarray,
    *,
    enabled: bool,
    eps: float,
) -> ObservationStats:
    obs = np.asarray(train_obs, dtype=np.float32)
    if obs.ndim != 2 or obs.shape[0] < 1:
        raise ValueError("train_obs must have shape (transitions, obs_dim)")
    if eps <= 0.0:
        raise ValueError("observation normalization epsilon must be positive")
    if enabled:
        mean = obs.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = obs.std(axis=0, dtype=np.float64).astype(np.float32)
        count = float(obs.shape[0])
        var_sum = (np.square(std.astype(np.float64)) * count).astype(np.float32)
    else:
        mean = np.zeros(obs.shape[1], dtype=np.float32)
        std = np.ones(obs.shape[1], dtype=np.float32)
        count = 0.0
        var_sum = np.zeros(obs.shape[1], dtype=np.float32)
    return ObservationStats(
        mean=mean,
        std=std,
        eps=eps,
        enabled=enabled,
        count=count,
        var_sum=var_sum,
    )


def compute_float32_observation_stats(
    train_obs: np.ndarray,
    *,
    enabled: bool,
    eps: float,
) -> ObservationStats:
    """Compute the reference FM running moments in float32."""

    obs = np.asarray(train_obs, dtype=np.float32)
    if obs.ndim != 2 or obs.shape[0] < 1:
        raise ValueError("train_obs must have shape (transitions, obs_dim)")
    if eps <= 0.0:
        raise ValueError("observation normalization epsilon must be positive")
    initial = ObservationStats(
        mean=np.zeros(obs.shape[1], dtype=np.float32),
        std=np.ones(obs.shape[1], dtype=np.float32),
        eps=eps,
        enabled=enabled,
        count=0.0,
        var_sum=np.zeros(obs.shape[1], dtype=np.float32),
    )
    return initial.update_float32(obs)


def prepare_decoder_data(
    source: str | Path | DecoderDataset | Any,
    config: DecoderTrainingConfig,
) -> PreparedDecoderData:
    if isinstance(source, (str, Path)):
        dataset = load_decoder_dataset(
            source,
            action_source=config.action_source,
            max_transitions=config.max_transitions,
            include_latent=config.pairwise_loss_weight > 0.0,
        )
    elif isinstance(source, DecoderDataset):
        dataset = source
    elif isinstance(getattr(source, "arrays", None), Mapping):
        dataset = decoder_dataset_from_arrays(
            source.arrays,
            action_source=config.action_source,
            include_latent=config.pairwise_loss_weight > 0.0,
        )
    else:
        raise TypeError(
            "decoder source must be an NPZ path, DecoderDataset, or collector "
            "object exposing an arrays mapping"
        )
    if config.max_transitions is not None and not isinstance(source, (str, Path)):
        limit = min(config.max_transitions, dataset.num_transitions)
        dataset = dataset.take(np.arange(limit))
    if config.pairwise_loss_weight > 0.0 and dataset.latent is None:
        raise ValueError(
            "pairwise_loss_weight > 0 requires latent_action, latent, or "
            "latents in the decoder dataset"
        )

    clip_fraction = 0.0
    if config.action_clip is not None:
        clip_fraction = float(np.mean(np.abs(dataset.action) > config.action_clip))
        dataset = dataset.with_action(
            np.clip(
                dataset.action,
                -config.action_clip,
                config.action_clip,
            ).astype(np.float32)
        )

    split = split_decoder_dataset(
        dataset,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
        split_unit=config.split_unit,
    )
    stats_fn = (
        compute_float32_observation_stats
        if config.rng_schedule == "state_threaded"
        else compute_observation_stats
    )
    stats = stats_fn(
        split.train.obs,
        enabled=config.normalize_observations,
        eps=config.observation_epsilon,
    )
    return PreparedDecoderData(
        split=split,
        train_obs=stats.normalize(split.train.obs),
        validation_obs=stats.normalize(split.validation.obs),
        observation_stats=stats,
        action_clip_fraction=clip_fraction,
    )


def reverse_cfm_targets(
    action: Array,
    noise: Array,
    t: Array,
) -> tuple[Array, Array]:
    """Return the exact legacy GoRL reverse-CFM path and velocity target."""

    action_samples = action[..., None, :]
    x_t = t * noise + (1.0 - t) * action_samples
    target_velocity = noise - action_samples
    return x_t, target_velocity


def identity_centered_diffusion_targets(
    decoder: DiffusionDecoder,
    action: Array,
    noise: Array,
    t: Array,
) -> DiffusionTrainingTargets:
    """Derive the epsilon delta used by the decoder's identity DDIM update.

    A standard deterministic DDIM step is ``r*x_t + c*epsilon``. The decoder
    implements the algebraically identical ``x_t + c*epsilon_delta`` so that a
    zero network is exactly identity. Therefore the supervised target is
    ``epsilon - ((1-r)/c)*x_t``.
    """

    step_size = 1.0 / decoder.diffusion_steps
    t_next = jnp.maximum(t - step_size, 0.0)
    alpha_t = decoder.alpha_bar(t)
    alpha_next = decoder.alpha_bar(t_next)
    sqrt_alpha_t = jnp.sqrt(alpha_t)
    sqrt_alpha_next = jnp.sqrt(alpha_next)
    sqrt_one_minus_t = jnp.sqrt(1.0 - alpha_t)
    sqrt_one_minus_next = jnp.sqrt(1.0 - alpha_next)
    alpha_ratio = sqrt_alpha_next / sqrt_alpha_t
    ddim_delta_scale = sqrt_one_minus_next - alpha_ratio * sqrt_one_minus_t

    action_samples = action[..., None, :]
    x_t = sqrt_alpha_t * action_samples + sqrt_one_minus_t * noise
    identity_epsilon = ((1.0 - alpha_ratio) / ddim_delta_scale) * x_t
    epsilon_delta = noise - identity_epsilon
    return DiffusionTrainingTargets(
        x_t=x_t,
        epsilon_delta=epsilon_delta,
        identity_epsilon=identity_epsilon,
        t_next=t_next,
        alpha_t=alpha_t,
        alpha_next=alpha_next,
        alpha_ratio=alpha_ratio,
        ddim_delta_scale=ddim_delta_scale,
    )


def direct_epsilon_diffusion_targets(
    decoder: DiffusionDecoder,
    action: Array,
    noise: Array,
    t: Array,
) -> DirectEpsilonTrainingTargets:
    """Return the historical direct-noise DDPM target at normalized time ``t``."""

    alpha_t = decoder.alpha_bar(t)
    action_samples = action[..., None, :]
    x_t = jnp.sqrt(alpha_t) * action_samples + jnp.sqrt(1.0 - alpha_t) * noise
    return DirectEpsilonTrainingTargets(
        x_t=x_t,
        epsilon=noise,
        alpha_t=alpha_t,
    )


def decoder_loss(
    decoder: ReverseTimeFlowMatchingDecoder | DiffusionDecoder,
    method: DecoderMethod,
    key: Array,
    obs: Array,
    action: Array,
    *,
    n_samples_per_action: int,
    latent: Array | None = None,
    pairwise_loss_weight: float = 0.0,
    time_key: Array | None = None,
    action_first_reduction: bool = False,
) -> tuple[Array, dict[str, Array]]:
    """Sample and evaluate one reverse-CFM or configured diffusion objective."""

    if time_key is None:
        key_noise, key_time = jax.random.split(key)
    else:
        key_noise, key_time = key, time_key
    sample_shape = (
        *action.shape[:-1],
        n_samples_per_action,
        action.shape[-1],
    )
    noise = jax.random.normal(key_noise, sample_shape)
    obs_samples = jnp.broadcast_to(
        obs[..., None, :],
        (*action.shape[:-1], n_samples_per_action, obs.shape[-1]),
    )

    if method == "fm":
        t = jax.random.uniform(
            key_time,
            (*action.shape[:-1], n_samples_per_action, 1),
        )
        x_t, target = reverse_cfm_targets(action, noise, t)
        prediction = decoder.velocity(obs_samples, x_t, t)
    elif method == "diffusion":
        if not isinstance(decoder, DiffusionDecoder):
            raise TypeError("diffusion objective requires DiffusionDecoder")
        if decoder.schedule == "cosine_alpha_bar":
            min_time_index = 0 if decoder.parameterization == "epsilon" else 1
            max_time_index = decoder.diffusion_steps
        else:
            min_time_index = 1
            max_time_index = decoder.diffusion_steps + 1
        if min_time_index >= max_time_index:
            raise ValueError(
                "cosine epsilon-delta diffusion requires at least two steps"
            )
        time_index = jax.random.randint(
            key_time,
            (*action.shape[:-1], n_samples_per_action, 1),
            minval=min_time_index,
            maxval=max_time_index,
        )
        t = time_index.astype(jnp.float32) / decoder.diffusion_steps
        if decoder.parameterization == "epsilon_delta":
            targets = identity_centered_diffusion_targets(
                decoder,
                action,
                noise,
                t,
            )
            x_t = targets.x_t
            target = targets.epsilon_delta
        else:
            direct_targets = direct_epsilon_diffusion_targets(
                decoder,
                action,
                noise,
                t,
            )
            x_t = direct_targets.x_t
            target = direct_targets.epsilon
        prediction = decoder.model_output(obs_samples, x_t, t)
    else:
        raise ValueError(f"unsupported decoder method: {method!r}")

    error = prediction - target
    squared_error = jnp.square(error)
    base_loss = (
        jnp.mean(jnp.mean(squared_error, axis=-1))
        if action_first_reduction
        else jnp.mean(squared_error)
    )
    pairwise_loss = jnp.zeros((), dtype=base_loss.dtype)
    if pairwise_loss_weight > 0.0:
        if latent is None:
            raise ValueError(
                "pairwise_loss_weight > 0 requires a latent for every action"
            )
        if latent.shape != action.shape:
            raise ValueError("pairwise latent and target action shapes differ")
        decoded_action = decoder.decode(obs, latent)
        pairwise_loss = jnp.mean(jnp.square(decoded_action - action))
    loss = base_loss + pairwise_loss_weight * pairwise_loss
    metrics = {
        "loss": loss,
        "base_loss": base_loss,
        "pairwise_loss": pairwise_loss,
        "prediction_abs_mean": jnp.mean(jnp.abs(prediction)),
        "target_abs_mean": jnp.mean(jnp.abs(target)),
        "action_abs_mean": jnp.mean(jnp.abs(action)),
        "noise_abs_mean": jnp.mean(jnp.abs(noise)),
    }
    return loss, metrics


def initialize_decoder(
    config: DecoderTrainingConfig,
    key: Array,
    *,
    obs_size: int,
    action_size: int,
) -> ReverseTimeFlowMatchingDecoder | DiffusionDecoder:
    common = {
        "prng": key,
        "obs_size": obs_size,
        "action_size": action_size,
        "hidden_dims": config.hidden_dims,
        "timestep_embed_dim": config.timestep_embed_dim,
        "output_scale": config.resolved_output_scale,
        "zero_final_layer": config.zero_final_layer,
    }
    if config.method == "fm":
        return ReverseTimeFlowMatchingDecoder.init(
            **common,
            flow_steps=config.decoder_steps,
        )
    return DiffusionDecoder.init(
        **common,
        diffusion_steps=config.decoder_steps,
        alpha_min=config.diffusion_alpha_min,
        alpha_max=config.diffusion_alpha_max,
        parameterization=config.diffusion_parameterization,
        schedule=config.diffusion_schedule,
    )


def restore_decoder_params(
    decoder: ReverseTimeFlowMatchingDecoder | DiffusionDecoder,
    warm_start: DecoderWarmStart,
    *,
    method: DecoderMethod,
) -> ReverseTimeFlowMatchingDecoder | DiffusionDecoder:
    """Restore compatible decoder parameters without restoring optimizer state."""

    if warm_start.method != method:
        raise TypeError(
            "warm-start decoder type mismatch: "
            f"checkpoint is {warm_start.method}, requested {method}"
        )
    expected_structure = jax.tree_util.tree_structure(decoder.params)
    restored_structure = jax.tree_util.tree_structure(warm_start.params)
    if expected_structure != restored_structure:
        raise ValueError(
            "warm-start parameter tree is incompatible with the requested "
            f"{method} decoder architecture"
        )
    expected_leaves = jax.tree_util.tree_leaves(decoder.params)
    restored_leaves = jax.tree_util.tree_leaves(warm_start.params)
    for index, (expected, restored) in enumerate(
        zip(expected_leaves, restored_leaves, strict=True)
    ):
        if expected.shape != restored.shape:
            raise ValueError(
                "warm-start parameter shape mismatch at leaf "
                f"{index}: {restored.shape} != {expected.shape}"
            )
        restored_dtype = np.asarray(restored).dtype
        expected_dtype = np.asarray(expected).dtype
        if restored_dtype != expected_dtype:
            raise TypeError(
                "warm-start parameter dtype mismatch at leaf "
                f"{index}: {restored_dtype} != {expected_dtype}"
            )
    return jdc.replace(decoder, params=_copy_params(warm_start.params))


def _apply_warm_observation_stats(
    data: PreparedDecoderData,
    config: DecoderTrainingConfig,
    warm_start: DecoderWarmStart,
) -> PreparedDecoderData:
    if not config.normalize_observations:
        return data
    previous = warm_start.observation_stats
    if previous is None or not previous.enabled or previous.count <= 0.0:
        raise ValueError(
            "normalized decoder warm-start requires previous observation "
            "moments with a positive count"
        )
    if previous.mean.shape != (data.split.train.obs_size,):
        raise ValueError(
            "warm-start observation statistics shape mismatch: "
            f"{previous.mean.shape} != {(data.split.train.obs_size,)}"
        )
    if previous.eps != config.observation_epsilon:
        raise ValueError(
            "warm-start observation epsilon mismatch: "
            f"{previous.eps} != {config.observation_epsilon}"
        )
    if config.warm_observation_stats_mode == "frozen":
        merged = previous
    elif config.warm_observation_stats_mode == "legacy_new_batch":
        merged = previous.update_float32_legacy_new_batch(data.split.train.obs)
    elif config.rng_schedule == "state_threaded":
        merged = previous.update_float32(data.split.train.obs)
    else:
        merged = previous.update(data.split.train.obs)
    return PreparedDecoderData(
        split=data.split,
        train_obs=merged.normalize(data.split.train.obs),
        validation_obs=merged.normalize(data.split.validation.obs),
        observation_stats=merged,
        action_clip_fraction=data.action_clip_fraction,
    )


def make_decoder_train_step(
    decoder: ReverseTimeFlowMatchingDecoder | DiffusionDecoder,
    optimizer: optax.GradientTransformation,
    config: DecoderTrainingConfig,
    *,
    observation_stats: ObservationStats | None = None,
) -> Callable[..., tuple[Any, Any, dict[str, Array]]]:
    """Build the parameter update used by both decoder variants."""

    def train_step(
        params: Any,
        opt_state: optax.OptState,
        key: Array,
        obs: Array,
        action: Array,
        latent: Array | None = None,
        time_key: Array | None = None,
    ) -> tuple[Any, optax.OptState, dict[str, Array]]:
        normalized_obs = _normalize_training_observation(obs, observation_stats)

        def loss_fn(current_params: Any) -> tuple[Array, dict[str, Array]]:
            current_decoder = jdc.replace(decoder, params=current_params)
            return decoder_loss(
                current_decoder,
                config.method,
                key,
                normalized_obs,
                action,
                n_samples_per_action=config.n_samples_per_action,
                latent=latent,
                pairwise_loss_weight=config.pairwise_loss_weight,
                time_key=time_key,
                action_first_reduction=config.rng_schedule == "state_threaded",
            )

        (_, metrics), gradients = jax.value_and_grad(
            loss_fn,
            has_aux=True,
        )(params)
        updates, next_opt_state = optimizer.update(
            gradients,
            opt_state,
            params,
        )
        next_params = optax.apply_updates(params, updates)
        metrics["grad_norm"] = optax.global_norm(gradients)
        return next_params, next_opt_state, metrics

    if config.rng_schedule == "state_threaded":
        return train_step
    return jax.jit(train_step)


def make_decoder_eval_step(
    decoder: ReverseTimeFlowMatchingDecoder | DiffusionDecoder,
    config: DecoderTrainingConfig,
    *,
    observation_stats: ObservationStats | None = None,
) -> Callable[..., dict[str, Array]]:
    def eval_step(
        params: Any,
        key: Array,
        obs: Array,
        action: Array,
        latent: Array | None = None,
        time_key: Array | None = None,
    ) -> dict[str, Array]:
        current_decoder = jdc.replace(decoder, params=params)
        normalized_obs = _normalize_training_observation(obs, observation_stats)
        _, metrics = decoder_loss(
            current_decoder,
            config.method,
            key,
            normalized_obs,
            action,
            n_samples_per_action=config.n_samples_per_action,
            latent=latent,
            pairwise_loss_weight=config.pairwise_loss_weight,
            time_key=time_key,
            action_first_reduction=config.rng_schedule == "state_threaded",
        )
        return metrics

    if config.rng_schedule == "state_threaded":
        return eval_step
    return jax.jit(eval_step)


def make_endpoint_eval_step(
    decoder: ReverseTimeFlowMatchingDecoder | DiffusionDecoder,
    *,
    observation_stats: ObservationStats | None = None,
) -> Callable[[Any, Array, Array, Array], dict[str, Array]]:
    @jax.jit
    def endpoint_step(
        params: Any,
        obs: Array,
        action: Array,
        latent: Array,
    ) -> dict[str, Array]:
        current_decoder = jdc.replace(decoder, params=params)
        normalized_obs = _normalize_training_observation(obs, observation_stats)
        decoded = current_decoder.decode(normalized_obs, latent)
        error = decoded - action
        return {
            "endpoint_mse": jnp.mean(jnp.square(error)),
            "endpoint_abs_mean": jnp.mean(jnp.abs(error)),
            "decoded_abs_mean": jnp.mean(jnp.abs(decoded)),
            "action_abs_mean": jnp.mean(jnp.abs(action)),
        }

    return endpoint_step


def _normalize_training_observation(
    obs: Array,
    stats: ObservationStats | None,
) -> Array:
    """Normalize inside the JAX graph for state-threaded compatibility.

    The original Humanoid decoder trainers normalized each raw minibatch in
    the differentiated JAX computation. Pre-normalizing the complete dataset
    with NumPy changes float32 rounding on GPU and can move a downstream PPO
    run onto a different trajectory. Other schedules retain their existing
    pre-normalized dataset path by passing ``None``.
    """

    if stats is None or not stats.enabled:
        return obs
    mean = jnp.asarray(stats.mean, dtype=jnp.float32)
    std = jnp.asarray(stats.std, dtype=jnp.float32)
    return (obs - mean) / (std + stats.eps)


def _evaluate_loss(
    eval_step: Callable[..., dict[str, Array]],
    params: Any,
    validation_obs: np.ndarray,
    validation_action: np.ndarray,
    validation_latent: np.ndarray | None,
    indices: np.ndarray,
    key: Array,
    *,
    time_key: Array | None = None,
    fold_in_batches: bool = True,
    split_chain_batches: bool = False,
) -> dict[str, float]:
    if fold_in_batches and split_chain_batches:
        raise ValueError("validation keys cannot use fold-in and split-chain together")
    records: list[dict[str, float]] = []
    split_chain_key = key
    for batch_index, batch_indices in enumerate(indices):
        if split_chain_batches:
            split_chain_key, batch_key = jax.random.split(split_chain_key)
        else:
            batch_key = jax.random.fold_in(key, batch_index) if fold_in_batches else key
        batch_time_key = (
            None
            if time_key is None
            else (
                jax.random.fold_in(time_key, batch_index)
                if fold_in_batches
                else time_key
            )
        )
        metrics = eval_step(
            params,
            batch_key,
            jnp.asarray(validation_obs[batch_indices]),
            jnp.asarray(validation_action[batch_indices]),
            (
                None
                if validation_latent is None
                else jnp.asarray(validation_latent[batch_indices])
            ),
            batch_time_key,
        )
        records.append(
            {name: float(np.asarray(value)) for name, value in metrics.items()}
        )
    return {
        name: float(np.mean([record[name] for record in records]))
        for name in records[0]
    }


def _validation_batch_indices(
    config: DecoderTrainingConfig,
    *,
    num_validation_transitions: int,
    legacy_rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Resolve validation batches for the configured random-number schedule."""

    if config.validation_schedule == "epoch_full_batches":
        batch_size = min(config.batch_size, num_validation_transitions)
        num_batches = min(
            50,
            max(1, num_validation_transitions // batch_size),
        )
        return np.arange(num_batches * batch_size, dtype=np.int64).reshape(
            num_batches,
            batch_size,
        )
    if config.rng_schedule == "legacy_interleaved":
        if legacy_rng is None:
            raise ValueError(
                "legacy_interleaved validation batches require a shared RNG"
            )
        rng = legacy_rng
    else:
        if legacy_rng is not None:
            raise ValueError("a shared legacy RNG is valid only for legacy_interleaved")
        rng = np.random.Generator(np.random.PCG64(config.seed + 1))
    return rng.integers(
        0,
        num_validation_transitions,
        size=(config.eval_batches, config.batch_size),
    )


def _endpoint_eval_indices(
    config: DecoderTrainingConfig,
    *,
    num_validation_transitions: int,
) -> np.ndarray:
    """Select the fixed endpoint-validation population."""

    endpoint_size = min(config.endpoint_eval_size, num_validation_transitions)
    if config.rng_schedule != "legacy_interleaved":
        return np.linspace(
            0,
            num_validation_transitions - 1,
            endpoint_size,
            dtype=np.int64,
        )

    edges = (
        np.arange(endpoint_size + 1, dtype=np.int64) * num_validation_transitions
    ) // endpoint_size
    rng = np.random.Generator(np.random.PCG64(config.seed))
    return rng.integers(
        low=edges[:-1],
        high=edges[1:],
        dtype=np.int64,
    )


def _advance_validation_key(
    config: DecoderTrainingConfig,
    key: Array,
) -> tuple[Array, Array]:
    """Return the retained and per-evaluation keys for validation."""

    if config.rng_schedule == "legacy_interleaved":
        retained_key, evaluation_key = jax.random.split(key)
        return retained_key, evaluation_key
    return key, key


def _copy_params(params: Any) -> Any:
    return jax.tree_util.tree_map(lambda value: jnp.array(value), params)


def _wrap_decoder(
    decoder: ReverseTimeFlowMatchingDecoder | DiffusionDecoder,
    params: Any,
    stats: ObservationStats,
) -> Any:
    current = jdc.replace(decoder, params=params)
    if not stats.enabled:
        return current
    return ObsNormalizedDecoder(
        decoder=current,
        obs_mean=jnp.asarray(stats.mean),
        obs_std=jnp.asarray(stats.std),
        eps=stats.eps,
    )


def fit_decoder(
    source: str | Path | DecoderDataset,
    config: DecoderTrainingConfig,
    *,
    warm_start: DecoderWarmStart | DecoderFitResult | None = None,
    metric_callback: Callable[[Mapping[str, float | int]], None] | None = None,
) -> DecoderFitResult:
    """Train a decoder and retain the lowest requested validation checkpoint."""

    resolved_warm_start = (
        DecoderWarmStart.from_fit_result(warm_start)
        if isinstance(warm_start, DecoderFitResult)
        else warm_start
    )
    data = prepare_decoder_data(source, config)
    if resolved_warm_start is not None:
        data = _apply_warm_observation_stats(
            data,
            config,
            resolved_warm_start,
        )
    train = data.split.train
    validation = data.split.validation
    state_threaded_schedule = config.rng_schedule == "state_threaded"
    legacy_interleaved_schedule = config.rng_schedule == "legacy_interleaved"
    graph_observation_stats = (
        data.observation_stats if state_threaded_schedule else None
    )
    training_obs = train.obs if state_threaded_schedule else data.train_obs
    validation_obs = validation.obs if state_threaded_schedule else data.validation_obs
    root_key = (
        jax.random.PRNGKey(config.seed)
        if state_threaded_schedule
        else jax.random.key(config.seed)
    )
    if state_threaded_schedule:
        init_key, train_key = jax.random.split(root_key, 2)
        eval_key = jax.random.fold_in(root_key, 1)
        endpoint_key = jax.random.fold_in(root_key, 2)
    else:
        init_key, train_key, eval_key, endpoint_key = jax.random.split(root_key, 4)
    decoder = initialize_decoder(
        config,
        init_key,
        obs_size=train.obs_size,
        action_size=train.action_size,
    )
    if resolved_warm_start is not None:
        decoder = restore_decoder_params(
            decoder,
            resolved_warm_start,
            method=config.method,
        )
    optimizer = optax.adam(config.learning_rate)
    params = decoder.params
    opt_state = optimizer.init(params)
    train_step = make_decoder_train_step(
        decoder,
        optimizer,
        config,
        observation_stats=graph_observation_stats,
    )
    eval_step = make_decoder_eval_step(
        decoder,
        config,
        observation_stats=graph_observation_stats,
    )
    endpoint_step = make_endpoint_eval_step(
        decoder,
        observation_stats=graph_observation_stats,
    )

    legacy_numpy_rng = (
        np.random.Generator(np.random.PCG64(config.seed))
        if legacy_interleaved_schedule
        else None
    )
    validation_indices = (
        None
        if legacy_interleaved_schedule
        else _validation_batch_indices(
            config,
            num_validation_transitions=validation.num_transitions,
        )
    )
    endpoint_indices = _endpoint_eval_indices(
        config,
        num_validation_transitions=validation.num_transitions,
    )
    endpoint_obs = jnp.asarray(validation_obs[endpoint_indices])
    endpoint_action = jnp.asarray(validation.action[endpoint_indices])
    endpoint_latent = jax.random.normal(endpoint_key, endpoint_action.shape)
    training_batches = _iter_training_batch_indices(
        config,
        num_train_transitions=train.num_transitions,
        num_total_transitions=(train.num_transitions + validation.num_transitions),
        legacy_rng=legacy_numpy_rng,
    )

    def evaluate(
        step: int,
        train_metrics: Mapping[str, Array] | None,
        *,
        loss_key: Array = eval_key,
        loss_time_key: Array | None = None,
        fold_in_batches: bool = True,
    ) -> dict[str, float | int]:
        nonlocal eval_key
        current_validation_indices = validation_indices
        split_chain_batches = False
        if legacy_interleaved_schedule:
            current_validation_indices = _validation_batch_indices(
                config,
                num_validation_transitions=validation.num_transitions,
                legacy_rng=legacy_numpy_rng,
            )
            eval_key, loss_key = _advance_validation_key(config, eval_key)
            fold_in_batches = False
            split_chain_batches = True
        assert current_validation_indices is not None
        loss_metrics = _evaluate_loss(
            eval_step,
            params,
            validation_obs,
            validation.action,
            validation.latent,
            current_validation_indices,
            loss_key,
            time_key=loss_time_key,
            fold_in_batches=fold_in_batches,
            split_chain_batches=split_chain_batches,
        )
        endpoint_metrics = endpoint_step(
            params,
            endpoint_obs,
            endpoint_action,
            endpoint_latent,
        )
        record: dict[str, float | int] = {
            "step": step,
            **{f"val_{name}": value for name, value in loss_metrics.items()},
            **{
                f"val_{name}": float(np.asarray(value))
                for name, value in endpoint_metrics.items()
            },
        }
        if train_metrics is not None:
            record.update(
                {
                    f"train_{name}": float(np.asarray(value))
                    for name, value in train_metrics.items()
                }
            )
        if not all(
            np.isfinite(value) for value in record.values() if isinstance(value, float)
        ):
            raise FloatingPointError(f"non-finite decoder metric at step {step}")
        if metric_callback is not None:
            metric_callback(dict(record))
        return record

    history: list[Mapping[str, float | int]] = []
    best_record: Mapping[str, float | int] | None = None
    best_value = float("inf")
    best_step = 0
    best_params = _copy_params(params)
    last_record: Mapping[str, float | int] | None = None
    if not state_threaded_schedule and not legacy_interleaved_schedule:
        initial_record = evaluate(0, None)
        history.append(initial_record)
        best_record = initial_record
        best_value = float(initial_record[config.checkpoint_metric])
        last_record = initial_record

    updates_per_epoch = max(
        1,
        train.num_transitions // min(config.batch_size, train.num_transitions),
    )
    epoch_train_metrics: list[Mapping[str, Array]] = []
    patience_counter = 0

    for step, batch_indices in enumerate(training_batches, start=1):
        if state_threaded_schedule:
            step_key, step_time_key, train_key = jax.random.split(train_key, 3)
        else:
            train_key, step_key = jax.random.split(train_key)
            step_time_key = None
        params, opt_state, train_metrics = train_step(
            params,
            opt_state,
            step_key,
            jnp.asarray(training_obs[batch_indices]),
            jnp.asarray(train.action[batch_indices]),
            (
                None
                if train.latent is None
                else jnp.asarray(train.latent[batch_indices])
            ),
            step_time_key,
        )
        if state_threaded_schedule:
            epoch_train_metrics.append(train_metrics)
            should_evaluate = (
                step % updates_per_epoch == 0 or step == config.train_steps
            )
        else:
            should_evaluate = (
                step == 1
                or step % config.eval_interval == 0
                or step == config.train_steps
            )
        if should_evaluate:
            reported_train_metrics = train_metrics
            loss_key = eval_key
            loss_time_key = None
            fold_in_batches = True
            if state_threaded_schedule:
                reported_train_metrics = {
                    name: jnp.mean(
                        jnp.stack([metrics[name] for metrics in epoch_train_metrics])
                    )
                    for name in epoch_train_metrics[0]
                }
                epoch_train_metrics.clear()
                loss_key, loss_time_key, _ = jax.random.split(train_key, 3)
                fold_in_batches = False
            last_record = evaluate(
                step,
                reported_train_metrics,
                loss_key=loss_key,
                loss_time_key=loss_time_key,
                fold_in_batches=fold_in_batches,
            )
            history.append(last_record)
            score = float(last_record[config.checkpoint_metric])
            if score < best_value:
                best_value = score
                best_step = step
                best_record = last_record
                best_params = _copy_params(params)
                patience_counter = 0
            elif config.early_stopping_patience is not None:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    break

    if best_record is None or last_record is None:
        raise RuntimeError("decoder training completed without validation")

    return DecoderFitResult(
        decoder=_wrap_decoder(decoder, best_params, data.observation_stats),
        final_decoder=_wrap_decoder(decoder, params, data.observation_stats),
        best_step=best_step,
        best_metric=config.checkpoint_metric,
        best_value=best_value,
        best_metrics=dict(best_record),
        final_metrics=dict(last_record),
        history=tuple(dict(record) for record in history),
        observation_stats=data.observation_stats,
        train_episode_ids=data.split.train_episode_ids,
        validation_episode_ids=data.split.validation_episode_ids,
        action_key=train.action_key,
        action_is_bounded=train.action_is_bounded,
        action_clip_fraction=data.action_clip_fraction,
    )
