from __future__ import annotations

from typing import Any, Literal, NamedTuple

import jax
import jax_dataclasses as jdc
import numpy as np
from jax import Array
from jax import numpy as jnp

from flow_policy import networks


DiffusionParameterization = Literal["epsilon_delta", "epsilon"]
DiffusionSchedule = Literal["linear_alpha_bar", "cosine_alpha_bar"]


class _DirectEpsilonSchedule(NamedTuple):
    t_current: Array
    t_next: Array
    alpha_t: Array
    alpha_next: Array


# Keep the host key form for compiled-trace parity with direct-epsilon checkpoints.
_DIRECT_EPSILON_KEY = np.zeros((2,), dtype=np.uint32)


@jdc.pytree_dataclass
class IdentityDecoder:
    """Stage0 decoder: latent action is the environment action pre-activation."""

    action_size: jdc.Static[int]

    def decode(self, obs: Array, latent: Array) -> Array:
        del obs
        assert latent.shape[-1] == self.action_size
        return latent


@jdc.pytree_dataclass
class FlowMatchingDecoder:
    """Conditional flow map from latent z to environment action.

    This is a normal mapping decoder: it starts from z and integrates a learned
    velocity field. With a zero final layer the initial velocity is exactly zero,
    so the initial flow map is identity.
    """

    params: networks.MlpWeights
    obs_size: jdc.Static[int]
    action_size: jdc.Static[int]
    flow_steps: jdc.Static[int] = 10
    timestep_embed_dim: jdc.Static[int] = 8
    output_scale: float = 0.25

    @staticmethod
    def init(
        prng: Array,
        obs_size: int,
        action_size: int,
        hidden_dims: tuple[int, ...] = (32, 32, 32, 32),
        flow_steps: int = 10,
        timestep_embed_dim: int = 8,
        output_scale: float = 0.25,
        zero_final_layer: bool = True,
    ) -> "FlowMatchingDecoder":
        assert timestep_embed_dim % 2 == 0
        dims = (obs_size + action_size + timestep_embed_dim, *hidden_dims, action_size)
        params = networks.mlp_init(prng, dims)
        if zero_final_layer:
            weights = tuple(params)
            final_w, final_b = weights[-1]
            params = networks.MlpWeights(
                (*weights[:-1], (jnp.zeros_like(final_w), jnp.zeros_like(final_b)))
            )
        return FlowMatchingDecoder(
            params=params,
            obs_size=obs_size,
            action_size=action_size,
            flow_steps=flow_steps,
            timestep_embed_dim=timestep_embed_dim,
            output_scale=output_scale,
        )

    def embed_timestep(self, t: Array) -> Array:
        assert t.shape[-1] == 1
        freqs = 2 ** jnp.arange(self.timestep_embed_dim // 2)
        scaled_t = t * freqs
        out = jnp.concatenate([jnp.cos(scaled_t), jnp.sin(scaled_t)], axis=-1)
        assert out.shape == (*t.shape[:-1], self.timestep_embed_dim)
        return out

    def velocity(self, obs: Array, x_t: Array, t: Array) -> Array:
        assert obs.shape[-1] == self.obs_size
        assert x_t.shape[-1] == self.action_size
        return (
            networks.flow_mlp_fwd(self.params, obs, x_t, self.embed_timestep(t))
            * self.output_scale
        )

    def decode(self, obs: Array, latent: Array) -> Array:
        assert obs.shape[-1] == self.obs_size
        assert latent.shape[-1] == self.action_size
        *batch_dims, _ = latent.shape
        ts = jnp.linspace(0.0, 1.0, self.flow_steps + 1)
        t_current = ts[:-1]
        t_next = ts[1:]

        def euler_step(x_t: Array, times: tuple[Array, Array]) -> tuple[Array, None]:
            t0, t1 = times
            t = jnp.full((*batch_dims, 1), t0)
            dt = t1 - t0
            x_next = x_t + dt * self.velocity(obs, x_t, t)
            assert x_next.shape == x_t.shape
            return x_next, None

        action, _ = jax.lax.scan(euler_step, latent, (t_current, t_next))
        assert action.shape == latent.shape
        return action


@jdc.pytree_dataclass
class ReverseTimeFlowMatchingDecoder:
    """Legacy FM convention: train action-to-noise, decode noise-to-action.

    The older GoRL code used x_t = t * z + (1 - t) * action and trained the
    velocity target z - action. Decoding starts at t=1 with latent z and
    integrates backward to t=0, so a learned constant target still maps z to
    action.
    """

    params: networks.MlpWeights
    obs_size: jdc.Static[int]
    action_size: jdc.Static[int]
    flow_steps: jdc.Static[int] = 10
    timestep_embed_dim: jdc.Static[int] = 8
    output_scale: float = 1.0

    @staticmethod
    def init(
        prng: Array,
        obs_size: int,
        action_size: int,
        hidden_dims: tuple[int, ...] = (64, 64, 64, 64),
        flow_steps: int = 10,
        timestep_embed_dim: int = 8,
        output_scale: float = 1.0,
        zero_final_layer: bool = False,
    ) -> "ReverseTimeFlowMatchingDecoder":
        assert timestep_embed_dim % 2 == 0
        dims = (obs_size + action_size + timestep_embed_dim, *hidden_dims, action_size)
        params = networks.mlp_init(prng, dims)
        if zero_final_layer:
            weights = tuple(params)
            final_w, final_b = weights[-1]
            params = networks.MlpWeights(
                (*weights[:-1], (jnp.zeros_like(final_w), jnp.zeros_like(final_b)))
            )
        return ReverseTimeFlowMatchingDecoder(
            params=params,
            obs_size=obs_size,
            action_size=action_size,
            flow_steps=flow_steps,
            timestep_embed_dim=timestep_embed_dim,
            output_scale=output_scale,
        )

    def embed_timestep(self, t: Array) -> Array:
        assert t.shape[-1] == 1
        freqs = 2 ** jnp.arange(self.timestep_embed_dim // 2)
        scaled_t = t * freqs
        out = jnp.concatenate([jnp.cos(scaled_t), jnp.sin(scaled_t)], axis=-1)
        assert out.shape == (*t.shape[:-1], self.timestep_embed_dim)
        return out

    def velocity(self, obs: Array, x_t: Array, t: Array) -> Array:
        assert obs.shape[-1] == self.obs_size
        assert x_t.shape[-1] == self.action_size
        return (
            networks.flow_mlp_fwd(self.params, obs, x_t, self.embed_timestep(t))
            * self.output_scale
        )

    def decode(self, obs: Array, latent: Array) -> Array:
        assert obs.shape[-1] == self.obs_size
        assert latent.shape[-1] == self.action_size
        *batch_dims, _ = latent.shape
        ts = jnp.linspace(1.0, 0.0, self.flow_steps + 1)
        t_current = ts[:-1]
        t_next = ts[1:]

        def euler_step(x_t: Array, times: tuple[Array, Array]) -> tuple[Array, None]:
            t0, t1 = times
            t = jnp.full((*batch_dims, 1), t0)
            dt = t1 - t0
            x_next = x_t + dt * self.velocity(obs, x_t, t)
            assert x_next.shape == x_t.shape
            return x_next, None

        action, _ = jax.lax.scan(euler_step, latent, (t_current, t_next))
        assert action.shape == latent.shape
        return action


@jdc.pytree_dataclass
class ObsNormalizedDecoder:
    """Wrap a decoder with fixed observation normalization.

    Decoder training may use train-set observation normalization. The wrapper
    stores those statistics with the checkpoint so later PPO stages can pass raw
    environment observations without a separate preprocessing path.
    """

    decoder: Any
    obs_mean: Array
    obs_std: Array
    eps: float = 1e-8

    @property
    def obs_size(self) -> int:
        return self.decoder.obs_size

    @property
    def action_size(self) -> int:
        return self.decoder.action_size

    def normalize_obs(self, obs: Array) -> Array:
        return (obs - self.obs_mean) / (self.obs_std + self.eps)

    def velocity(self, obs: Array, x_t: Array, t: Array) -> Array:
        return self.decoder.velocity(self.normalize_obs(obs), x_t, t)

    def decode(self, obs: Array, latent: Array) -> Array:
        return self.decoder.decode(self.normalize_obs(obs), latent)


@jdc.pytree_dataclass
class DiffusionDecoder:
    """Deterministic DDIM-style decoder from latent z to environment action.

    The sampler is a deterministic denoising chain with eta=0. It supports the
    modern identity-centered epsilon-delta objective and the direct-epsilon
    objective used by the historical Humanoid experiments. The former keeps a
    zero-initialized decoder exactly identity-like; the latter follows standard
    DDIM and normally uses a non-zero final-layer initialization.
    """

    params: networks.MlpWeights
    obs_size: jdc.Static[int]
    action_size: jdc.Static[int]
    diffusion_steps: jdc.Static[int] = 10
    timestep_embed_dim: jdc.Static[int] = 8
    output_scale: float = 0.25
    alpha_min: float = 1e-3
    alpha_max: float = 1.0
    parameterization: jdc.Static[DiffusionParameterization] = "epsilon_delta"
    schedule: jdc.Static[DiffusionSchedule] = "linear_alpha_bar"

    @staticmethod
    def init(
        prng: Array,
        obs_size: int,
        action_size: int,
        hidden_dims: tuple[int, ...] = (32, 32, 32, 32),
        diffusion_steps: int = 10,
        timestep_embed_dim: int = 8,
        output_scale: float = 0.25,
        alpha_min: float = 1e-3,
        alpha_max: float = 1.0,
        parameterization: DiffusionParameterization = "epsilon_delta",
        schedule: DiffusionSchedule = "linear_alpha_bar",
        zero_final_layer: bool = True,
    ) -> "DiffusionDecoder":
        assert timestep_embed_dim % 2 == 0
        assert diffusion_steps > 0
        assert 0.0 < alpha_min < alpha_max <= 1.0
        if parameterization not in {"epsilon_delta", "epsilon"}:
            raise ValueError(
                f"unsupported diffusion parameterization: {parameterization!r}"
            )
        if schedule not in {"linear_alpha_bar", "cosine_alpha_bar"}:
            raise ValueError(f"unsupported diffusion schedule: {schedule!r}")
        dims = (obs_size + action_size + timestep_embed_dim, *hidden_dims, action_size)
        params = networks.mlp_init(prng, dims)
        if zero_final_layer:
            weights = tuple(params)
            final_w, final_b = weights[-1]
            params = networks.MlpWeights(
                (*weights[:-1], (jnp.zeros_like(final_w), jnp.zeros_like(final_b)))
            )
        return DiffusionDecoder(
            params=params,
            obs_size=obs_size,
            action_size=action_size,
            diffusion_steps=diffusion_steps,
            timestep_embed_dim=timestep_embed_dim,
            output_scale=output_scale,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            parameterization=parameterization,
            schedule=schedule,
        )

    def embed_timestep(self, t: Array) -> Array:
        assert t.shape[-1] == 1
        freqs = 2 ** jnp.arange(self.timestep_embed_dim // 2)
        scaled_t = t * freqs
        out = jnp.concatenate([jnp.cos(scaled_t), jnp.sin(scaled_t)], axis=-1)
        assert out.shape == (*t.shape[:-1], self.timestep_embed_dim)
        return out

    def alpha_bar(self, t: Array) -> Array:
        if self.schedule == "linear_alpha_bar":
            return self.alpha_min + (self.alpha_max - self.alpha_min) * (1.0 - t)
        if self.parameterization == "epsilon":
            _, _, cumulative_alphas = self._direct_epsilon_beta_schedule()
        else:
            offset = 0.008
            grid = (
                jnp.linspace(0, self.diffusion_steps, self.diffusion_steps + 1)
                / self.diffusion_steps
            )
            angle = (grid + offset) / (1.0 + offset) * jnp.pi / 2.0
            normalizer = jnp.cos(offset / (1.0 + offset) * jnp.pi / 2.0) ** 2
            cumulative_alphas = (jnp.cos(angle) ** 2 / normalizer)[:-1]
        index = jnp.rint(t * self.diffusion_steps).astype(jnp.int32)
        return cumulative_alphas[jnp.clip(index, 0, self.diffusion_steps - 1)]

    def _direct_epsilon_beta_schedule(self) -> tuple[Array, Array, Array]:
        steps = self.diffusion_steps
        offset = 0.008
        time = jnp.linspace(0, steps, steps + 1) / steps
        cumulative_full = jnp.cos((time + offset) / (1.0 + offset) * jnp.pi / 2.0) ** 2
        cumulative_full = cumulative_full / cumulative_full[0]
        cumulative_alphas = cumulative_full[:-1]
        cumulative_previous = jnp.concatenate(
            [jnp.array([1.0]), cumulative_alphas[:-1]]
        )
        alphas = cumulative_alphas / cumulative_previous
        betas = jnp.clip(1.0 - alphas, 0.0001, 0.9999)
        return betas, alphas, cumulative_alphas

    def _direct_epsilon_schedule(self) -> _DirectEpsilonSchedule:
        _, _, cumulative_alphas = self._direct_epsilon_beta_schedule()
        timesteps = jnp.arange(self.diffusion_steps - 1, -1, -1)
        t_current = timesteps[:-1]
        t_next = timesteps[1:]
        alpha_t = cumulative_alphas[t_current]
        alpha_next_from_schedule = cumulative_alphas[t_next]
        alpha_next = jnp.where(t_next == 0, 1.0, alpha_next_from_schedule)
        return _DirectEpsilonSchedule(
            t_current=t_current.astype(jnp.float32),
            t_next=t_next.astype(jnp.float32),
            alpha_t=alpha_t,
            alpha_next=alpha_next,
        )

    def _embed_direct_epsilon_timestep(self, t: Array) -> Array:
        assert t.shape[-1] == 1
        normalized_timestep = t / self.diffusion_steps
        frequencies = jnp.arange(self.timestep_embed_dim // 2)
        scaled_timestep = normalized_timestep * (2 ** frequencies[None, :])
        return jnp.concatenate(
            [jnp.cos(scaled_timestep), jnp.sin(scaled_timestep)],
            axis=-1,
        )

    def model_output(self, obs: Array, x_t: Array, t: Array) -> Array:
        """Predict the configured diffusion target at normalized timestep ``t``."""

        assert obs.shape[-1] == self.obs_size
        assert x_t.shape[-1] == self.action_size
        return (
            networks.flow_mlp_fwd(self.params, obs, x_t, self.embed_timestep(t))
            * self.output_scale
        )

    def epsilon_delta(self, obs: Array, x_t: Array, t: Array) -> Array:
        return self.model_output(obs, x_t, t)

    def _decode_direct_epsilon_cosine(self, obs: Array, latent: Array) -> Array:
        """Decode direct-epsilon latents with the original DDIM arithmetic."""

        single_observation = obs.ndim == 1
        if single_observation:
            obs = obs[None, :]
            latent = latent[None, :]
        *batch_dims, _ = obs.shape
        action_size = self.params[-1][0].shape[-1]

        def ddim_step(
            x_t: Array,
            inputs: tuple[_DirectEpsilonSchedule, Array],
        ) -> tuple[Array, Array]:
            schedule, noise = inputs
            embedding = jnp.broadcast_to(
                self._embed_direct_epsilon_timestep(schedule.t_current[None, None]),
                (*batch_dims, self.timestep_embed_dim),
            )
            prediction = (
                networks.flow_mlp_fwd(self.params, obs, x_t, embedding)
                * self.output_scale
            )
            sqrt_alpha_t = jnp.sqrt(schedule.alpha_t)
            sqrt_one_minus_t = jnp.sqrt(1 - schedule.alpha_t)
            sqrt_alpha_next = jnp.sqrt(schedule.alpha_next)
            sqrt_one_minus_next = jnp.sqrt(1 - schedule.alpha_next)
            x0_prediction = (x_t - sqrt_one_minus_t * prediction) / sqrt_alpha_t
            x_next = sqrt_alpha_next * x0_prediction + sqrt_one_minus_next * prediction
            x_next = x_next + 0.0 * noise
            return x_next, x_t

        noise_key, _feather_key = jax.random.split(_DIRECT_EPSILON_KEY, num=2)
        noise_path = jax.random.normal(
            noise_key,
            (self.diffusion_steps - 1, *batch_dims, action_size),
        )
        action, _action_path = jax.lax.scan(
            ddim_step,
            latent,
            (self._direct_epsilon_schedule(), noise_path),
        )
        if single_observation:
            action = action.squeeze(0)
        return action

    def decode(self, obs: Array, latent: Array) -> Array:
        assert obs.shape[-1] == self.obs_size
        assert latent.shape[-1] == self.action_size
        if self.schedule == "cosine_alpha_bar" and self.parameterization == "epsilon":
            action = self._decode_direct_epsilon_cosine(obs, latent)
            assert action.shape == latent.shape
            return action
        *batch_dims, _ = latent.shape
        if self.schedule == "cosine_alpha_bar":
            # Historical GoRL Diffusion defines T cumulative-alpha entries at
            # indices [0, T), then denoises T-1 -> ... -> 0 in T-1 steps.
            indices = jnp.arange(self.diffusion_steps - 1, -1, -1)
            ts = indices.astype(jnp.float32) / self.diffusion_steps
        else:
            ts = jnp.linspace(1.0, 0.0, self.diffusion_steps + 1)
        t_current = ts[:-1]
        t_next = ts[1:]

        def ddim_step(x_t: Array, times: tuple[Array, Array]) -> tuple[Array, None]:
            t0, t1 = times
            t = jnp.full((*batch_dims, 1), t0)
            alpha_t = self.alpha_bar(t0)
            alpha_next = self.alpha_bar(t1)
            sqrt_alpha_t = jnp.sqrt(alpha_t)
            sqrt_alpha_next = jnp.sqrt(alpha_next)
            sqrt_one_minus_t = jnp.sqrt(1.0 - alpha_t)
            sqrt_one_minus_next = jnp.sqrt(1.0 - alpha_next)
            alpha_ratio = sqrt_alpha_next / sqrt_alpha_t
            ddim_delta_scale = sqrt_one_minus_next - alpha_ratio * sqrt_one_minus_t
            prediction = self.model_output(obs, x_t, t)
            if self.parameterization == "epsilon_delta":
                # Algebraically equivalent to a DDIM step with the analytic
                # identity epsilon baseline. This form avoids cancellation so
                # that a zero network preserves x_t bit-for-bit.
                x_next = x_t + ddim_delta_scale * prediction
            else:
                # Standard deterministic DDIM update for direct epsilon
                # prediction, matching the historical GoRL Diffusion decoder.
                x0_prediction = (x_t - sqrt_one_minus_t * prediction) / sqrt_alpha_t
                x_next = (
                    sqrt_alpha_next * x0_prediction + sqrt_one_minus_next * prediction
                )
            assert x_next.shape == x_t.shape
            return x_next, None

        action, _ = jax.lax.scan(ddim_step, latent, (t_current, t_next))
        assert action.shape == latent.shape
        return action
