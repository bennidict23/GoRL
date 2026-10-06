# Modified by the GoRL authors; see THIRD_PARTY_NOTICES.md.

from __future__ import annotations

from typing import NewType

import jax
from jax import Array, nn
from jax import numpy as jnp

from .math_utils import NormalDistribution

# Weights will be in the format ((linear, bias), ...).
MlpWeights = NewType("MlpWeights", tuple[tuple[Array, Array], ...])


def mlp_init(
    prng: Array,
    dims: tuple[int, ...],
    init_fn: nn.initializers.Initializer | None = None,
) -> MlpWeights:
    """Initialize MLP weights."""
    prngs = jax.random.split(prng, len(dims) - 1)
    shapes = zip(dims[:-1], dims[1:])
    init_fn = nn.initializers.lecun_uniform() if init_fn is None else init_fn
    return MlpWeights(
        tuple(
            (init_fn(prng, shape), jnp.zeros((shape[1],)))
            for prng, shape in zip(prngs, shapes)
        )
    )


def value_mlp_fwd(weights: MlpWeights, x: Array) -> Array:
    """Apply hidden layers, then output projection.

    Input: (*, obs_dim)
    Output: (*,)
    """
    for i in range(len(weights) - 1):
        linear, bias = weights[i]
        x = jnp.einsum("...i,ij->...j", x, linear) + bias
        x = nn.silu(x)

    linear, bias = weights[-1]
    x = jnp.einsum("...i,ij->...j", x, linear) + bias
    x = jnp.squeeze(x, axis=-1)
    return x


def flow_mlp_fwd(weights: MlpWeights, *inputs_to_concat: Array) -> Array:
    """Apply hidden layers, then output projection."""
    x = jnp.concatenate(inputs_to_concat, axis=-1)
    for i in range(len(weights) - 1):
        linear, bias = weights[i]
        x = jnp.einsum("...i,ij->...j", x, linear) + bias
        x = nn.silu(x)
    linear, bias = weights[-1]
    x = jnp.einsum("...i,ij->...j", x, linear) + bias
    return x


def gaussian_policy_fwd(
    weights: MlpWeights,
    x: Array,
    max_scale: float = 0.0,
) -> NormalDistribution:
    """Apply hidden layers, then output projection."""
    # Final layer is split into mean and scale.
    assert weights[-1][0].shape[-1] % 2 == 0

    for i in range(len(weights) - 1):
        linear, bias = weights[i]
        x = jnp.einsum("...i,ij->...j", x, linear) + bias
        x = nn.silu(x)

    linear, bias = weights[-1]
    x = jnp.einsum("...i,ij->...j", x, linear) + bias

    mean, scale = jnp.split(x, 2, axis=-1)
    scale = nn.softplus(scale) + 1e-3
    if max_scale > 0.0:
        scale = jnp.minimum(scale, max_scale)
    return NormalDistribution(mean, scale)
