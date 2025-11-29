from dataclasses import dataclass
from typing import Callable, Literal, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from .typing import GammaType, InterpolantCoefficient, Score, TimeSchedule, Velocity
from .utils import make_gamma


def compute_div(
    f: Callable[[Array, Array], Array],
    x: Array,
    t: Array,  # [batch, ...]
) -> Array:
    """Compute the divergence of f(x, t) w.r.t. x for batched x.

    Assumes:
      - x has shape [bs, d]
      - f(x, t) returns an array of shape [bs, d]
    """

    # Make a per-sample version: (d,), (...) -> (d,)
    def f_single(x_i, t_i):
        # f expects batched inputs, so add batch dim and remove it afterwards
        return f(x_i[None, :], t_i[None, ...])[0]

    # Divergence for a single sample
    def div_single(x_i, t_i):
        # Jacobian of f_single w.r.t. x_i: shape [d, d]
        jac_x = jax.jacrev(f_single, argnums=0)(x_i, t_i)
        # Divergence = trace of Jacobian
        return jnp.trace(jac_x)

    # Vectorize over the batch dimension
    return jax.vmap(div_single)(x, t)


class LinearInterpolant:
    def __init__(
        self,
        alpha: InterpolantCoefficient,
        beta: InterpolantCoefficient,
        t_schedule: TimeSchedule,
        gamma_type: GammaType = "brownian",
    ):
        self.alpha: InterpolantCoefficient = alpha
        self.beta: InterpolantCoefficient = beta
        self.t_schedule: TimeSchedule = t_schedule
        self.gamma, self.gamma_dot, self.gg_dot = make_gamma(gamma_type=gamma_type)

    def interpolant(self, x0: Array, x1: Array, t: float) -> Array:
        t = jnp.broadcast_to(t, (x0.shape[0], 1, 1, 1))
        return self.alpha(t) * x0 + self.beta(t) * x1

    def calc_antithetic_xts(
        self, t: float, x0: Array, x1: Array, key: Array
    ) -> tuple[Array, Array, Array]:
        z = jax.random.normal(key, x0.shape)
        t = jnp.broadcast_to(t, (x0.shape[0], 1, 1, 1))
        x_interp = self.interpolant(x0, x1, t)
        return x_interp + self.gamma(t) * z, x_interp - self.gamma(t) * z, z

    def dtIt(self, x0: Array, x1: Array, t: float) -> Array:
        def _pure_interpolant_wrapper(t_inner: float) -> Array:
            return self.interpolant(x0, x1, t_inner)

        return jax.jacfwd(_pure_interpolant_wrapper)(t)

    def generate_xt(self, x0: Array, x1: Array, t: float, key: Array):
        z = jax.random.normal(key, x0.shape)
        t = jnp.broadcast_to(t, (x0.shape[0], 1, 1, 1))
        return self.interpolant(x0, x1, t) + self.gamma(t) * z, z


@dataclass
class SDEIntegrator:
    b: Velocity
    s: Score
    eps: Array  # or float
    interpolant: LinearInterpolant
    t_grid: Array  # shape [n_step + 1]
    n_save: int = 4
    start_end: Tuple[int, int] = (0, 1)  # optional, can be inferred from t_grid
    n_step: int = 3000  # optional, can be inferred from t_grid
    n_likelihood: int = 1

    # --- Drift and likelihood ---

    def bf(self, x: Array, t: Array) -> Array:
        """Forward drift. Assume x is batched, t broadcastable."""
        return self.b(x, t) + self.eps * self.s(x, t)

    def br(self, x: Array, t: Array) -> Array:
        """Backward drift. Assume x is batched, t broadcastable."""
        return self.b(x, t) - self.eps * self.s(x, t)

    def dt_logp(self, x: Array, t: Array) -> Array:
        """Time derivative of log-likelihood (integrating from 1 to 0)."""
        score = self.s(x, t)  # [B, ...]
        s_norm = jnp.linalg.norm(score, axis=-1) ** 2  # [B]
        return -(compute_div(self.bf, x, t) + self.eps * s_norm)

    # --- Integrators ---

    def step_forward_heun(self, x: Array, t: Array, key: Array, dt: Array) -> Array:
        """Forward-time Heun step (https://arxiv.org/pdf/2206.00364.pdf, Alg. 2)."""
        dW = jnp.sqrt(dt) * jax.random.normal(key, x.shape)
        xhat = x + jnp.sqrt(2.0 * self.eps) * dW
        K1 = self.bf(xhat, t + dt)
        xp = xhat + dt * K1
        K2 = self.bf(xp, t + dt)
        return xhat + 0.5 * dt * (K1 + K2)

    def step_reverse_heun(self, x: Array, t: Array, key: Array, dt: Array) -> Array:
        """Reverse-time Heun step."""
        dW = jnp.sqrt(dt) * jax.random.normal(key, x.shape)
        xhat = x + jnp.sqrt(2.0 * self.eps) * dW
        K1 = self.br(xhat, t - dt)
        xp = xhat - dt * K1
        K2 = self.br(xp, t - dt)
        return xhat - 0.5 * dt * (K1 + K2)

    # --- Rollout ---

    def forward_rollout(self, x0: Array, key: Array) -> Array:
        t_grid = self.t_grid

        def _integrator(i, state):
            x, key = state
            key, sub_key = jax.random.split(key)

            B = x.shape[0]
            t_i = jnp.broadcast_to(t_grid[i], (B, 1, 1, 1))
            t_ip1 = jnp.broadcast_to(t_grid[i + 1], (B, 1, 1, 1))
            dt = t_ip1 - t_i

            xn = self.step_forward_heun(x, t_i, sub_key, dt)
            return (xn, key)

        # If you trust t_grid, you can infer n_step from it:
        n_step = t_grid.shape[0] - 1
        X, _ = jax.lax.fori_loop(0, n_step, _integrator, (x0, key))
        return X

    def forward_rollout_trace(self, x0: Array, key: Array) -> list[Array]:
        """
        Slower, verbose rollout that returns the intermediate states for a single sample.
        Intended for debugging/visualization; not JIT-compiled.
        """
        states: list[Array] = []
        x = x0
        k = key
        t_grid = self.t_grid
        n_step = t_grid.shape[0] - 1

        for i in range(n_step):
            states.append(x)
            k, sub_key = jax.random.split(k)

            t_i = jnp.broadcast_to(t_grid[i], x.shape[:-1] + (1,))
            t_ip1 = jnp.broadcast_to(t_grid[i + 1], x.shape[:-1] + (1,))
            dt = t_ip1 - t_i

            x = self.step_forward_heun(x, t_i, sub_key, dt)

        states.append(x)
        return states
