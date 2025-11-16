import math

import jax
import jax.numpy as jnp
from jax import Array

from src.interpolants.stochastic_interpolant import LinearInterpolant

from .typing import Score, Velocity


# Inspired by https://github.com/malbergo/stochastic-interpolants/blob/main/interflow/fabrics.py
def make_gamma(gamma_type: str = "brownian", a: float | None = None):
    """
    returns callable functions for gamma, gamma_dot,
    and gamma(t)*gamma_dot(t) to avoid numerical divide by 0s,
    e.g. if one is using the brownian (default) gamma.
    """
    if gamma_type == "brownian":
        gamma = lambda t: jnp.sqrt(t * (1 - t))
        gamma_dot = lambda t: (1 / (2 * jnp.sqrt(t * (1 - t)))) * (1 - 2 * t)
        gg_dot = lambda t: (1 / 2) * (1 - 2 * t)

    elif gamma_type == "a-brownian":
        gamma = lambda t: jnp.sqrt(a * t * (1 - t))
        gamma_dot = lambda t: (1 / (2 * jnp.sqrt(a * t * (1 - t)))) * a * (1 - 2 * t)
        gg_dot = lambda t: (a / 2) * (1 - 2 * t)

    elif gamma_type == "zero":
        gamma = gamma_dot = gg_dot = lambda t: jnp.zeros_like(t)

    elif gamma_type == "bsquared":
        gamma = lambda t: t * (1 - t)
        gamma_dot = lambda t: 1 - 2 * t
        gg_dot = lambda t: gamma(t) * gamma_dot(t)

    elif gamma_type == "sinesquared":
        gamma = lambda t: jnp.sin(math.pi * t) ** 2
        gamma_dot = lambda t: 2 * math.pi * jnp.sin(math.pi * t) * jnp.cos(math.pi * t)
        gg_dot = lambda t: gamma(t) * gamma_dot(t)

    elif gamma_type == "sigmoid":
        f = jnp.tensor(10.0)
        gamma = (
            lambda t: jax.nn.sigmoid(f * (t - (1 / 2)) + 1)
            - jax.nn.sigmoid(f * (t - (1 / 2)) - 1)
            - jax.nn.sigmoid((-f / 2) + 1)
            + jax.nn.sigmoid((-f / 2) - 1)
        )
        gamma_dot = lambda t: (-f) * (1 - jax.nn.sigmoid(-1 + f * (t - (1 / 2)))) * jax.nn.sigmoid(
            -1 + f * (t - (1 / 2))
        ) + f * (1 - jax.nn.sigmoid(1 + f * (t - (1 / 2)))) * jax.nn.sigmoid(1 + f * (t - (1 / 2)))
        gg_dot = lambda t: gamma(t) * gamma_dot(t)

    elif gamma_type == None:
        gamma = lambda t: jnp.zeros(1)  ### no gamma
        gamma_dot = lambda t: jnp.zeros(1)  ### no gamma
        gg_dot = lambda t: jnp.zeros(1)  ### no gamma

    else:
        raise NotImplementedError("The gamma you specified is not implemented.")

    return gamma, gamma_dot, gg_dot


def build_t_grid(
    n_steps: int,
    endpoint_clip: float = 1e-12,
    schedule: str = "linear",
    power: float = 2.0,
) -> jnp.ndarray:
    if schedule == "linear":
        t = jnp.linspace(0.0 + endpoint_clip, 1.0 - endpoint_clip, n_steps)
    elif schedule == "cosine":
        s = jnp.linspace(0.0, 1.0, n_steps)
        t = 0.5 - 0.5 * jnp.cos(jnp.pi * s)
        t = jnp.clip(t, endpoint_clip, 1.0 - endpoint_clip)
    elif schedule == "power":
        s = jnp.linspace(0.0, 1.0, n_steps) ** power
        t = jnp.clip(t, endpoint_clip, 1.0 - endpoint_clip)
    else:
        raise ValueError("Unknown t schedule")
    return t


def epsilon_schedule(t: jnp.ndarray, eps0: float = 0.1, taper: float = 0.6) -> jnp.ndarray:
    schedule = eps0 * (t * (1.0 - t)) ** taper
    return schedule


def loss_per_sample_b(
    b: Velocity,
    x0: Array,
    x1: Array,
    t: Array,
    cosmos: Array,
    interpolant: LinearInterpolant,
    key: Array,
) -> Array:
    xtp, xtm, z = interpolant.calc_antithetic_xts(t, x0, x1, key)

    # Add fake batch dimension: [1, ...]
    xtp = xtp[jnp.newaxis, ...]
    xtm = xtm[jnp.newaxis, ...]
    t_vec = t[jnp.newaxis, ...].reshape((1,))
    cosmos_vec = cosmos[jnp.newaxis, ...]

    # These were called on the original (un-unsqueezed) tensors in your torch code
    dtIt = interpolant.dtIt(x0, x1, t)
    gamma_dot = interpolant.gamma_dot(t)

    # Drift evaluations (batched)
    btp = b(xtp, t_vec, cosmos_vec)
    btm = b(xtm, t_vec, cosmos_vec)

    # Same algebra as in PyTorch
    loss = 0.5 * jnp.sum(btp**2) - jnp.sum((dtIt + gamma_dot * z) * btp)
    loss += 0.5 * jnp.sum(btm**2) - jnp.sum((dtIt - gamma_dot * z) * btm)

    return loss


def loss_per_sample_s(
    s: Score,
    x0: Array,
    x1: Array,
    t: Array,
    cosmos: Array,
    interpolant: LinearInterpolant,
    key: Array,
) -> Array:
    """Compute the (variance-reduced) loss on an individual sample via antithetic sampling."""
    # JAX version of calc_antithetic_xts
    xtp, xtm, z = interpolant.calc_antithetic_xts(t, x0, x1, key)

    # Add fake batch dimension: [1, ...]
    xtp = xtp[jnp.newaxis, ...]
    xtm = xtm[jnp.newaxis, ...]
    t_vec = t[jnp.newaxis, ...].reshape((1,))
    cosmos_vec = cosmos[jnp.newaxis, ...]

    # Score evaluations
    stp = s(xtp, t_vec, cosmos_vec)
    stm = s(xtm, t_vec, cosmos_vec)

    gamma_t = interpolant.gamma(t)

    loss = 0.5 * jnp.sum(stp**2) + (1.0 / gamma_t) * jnp.sum(stp * z)
    loss += 0.5 * jnp.sum(stm**2) - (1.0 / gamma_t) * jnp.sum(stm * z)

    return loss


def batch_loss_b(
    b: Velocity,
    x0_batch: Array,
    x1_batch: Array,
    t_batch: Array,
    cosmos_batch: Array,
    interpolant: LinearInterpolant,
    key: Array,
) -> Array:
    """Compute mean loss over a batch for the b-network."""

    # Split RNG: one subkey per batch element
    keys = jax.random.split(key, x0_batch.shape[0])

    # Vectorize loss_per_sample_b over batch dimension
    loss_fn = jax.vmap(
        lambda x0, x1, t, c, k: loss_per_sample_b(b, x0, x1, t, c, interpolant, k),
        in_axes=(0, 0, 0, 0, 0),
    )

    losses = loss_fn(x0_batch, x1_batch, t_batch, cosmos_batch, keys)  # shape [B]
    return jnp.mean(losses)


def batch_loss_s(
    s: Score,
    x0_batch: Array,
    x1_batch: Array,
    t_batch: Array,
    cosmos_batch: Array,
    interpolant: LinearInterpolant,
    key: Array,
) -> Array:
    """Compute mean loss over a batch for the s-network."""

    keys = jax.random.split(key, x0_batch.shape[0])

    loss_fn = jax.vmap(
        lambda x0, x1, t, c, k: loss_per_sample_s(s, x0, x1, t, c, interpolant, k),
        in_axes=(0, 0, 0, 0, 0),
    )

    losses = loss_fn(x0_batch, x1_batch, t_batch, cosmos_batch, keys)
    return jnp.mean(losses)
