from dataclasses import dataclass
from typing import Callable, Literal

import jax
import jax.numpy as jnp
from jax import Array

from .utils import make_gamma

InterpolantCoefficient = Callable[[float], Array]
TimeSchedule = Literal["linear", "cosine", "power"]
GammaType = Literal["brownian", "a-brownian", "zero", "bsquared", "sinesquared", "sigmoid"]
Velocity = jax.nnx.Module
Score = jax.nnx.Module


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

    def compute_derivative(self, x0: Array, x1: Array, t: float) -> Array:
        def _pure_interpolant_wrapper(t_inner: float) -> Array:
            return self.interpolant(x0, x1, t_inner)

        return jax.grad(_pure_interpolant_wrapper)(t)

    def generate_xt(self, x0: Array, x1: Array, t: float, key: Array):
        z = jax.random.normal(key, x0.shape[0])
        t = jnp.broadcast_to(t, (x0.shape[0], 1, 1, 1))
        return self.interpolant(x0, x1, t) + self.gamma(t) * z, z


@dataclass
class SDEIntegrator:
    b: Velocity
    s: Score
    eps: Array
    interpolant: LinearInterpolant
    n_save: int = 4
    start_end: tuple[int, int] = (0, 1)  # pyright: ignore[reportIndexIssue]
    n_step: int = 3000
    n_likelihood: int = 1

    def __post_init__(self) -> None:
        """Initialize forward dynamics, reverse dynamics, and likelihood."""

        def bf(x: Array, t: Array):
            """Forward drift. Assume x is batched but t is not."""
            self.b.to(x.device)
            self.s.to(x.device)
            return self.b(x, t) + self.eps * self.s(x, t)

        def br(x: Array, t: Array):
            """Backwards drift. Assume x is batched but t is not."""
            self.b.to(x.device)
            self.s.to(x.device)

            return self.b(x, t) - self.eps * self.s(x, t)

        def dt_logp(x: torch.tensor, t: torch.tensor):
            """Time derivative of the log-likelihood, assumed integrating from 1 to 0.
            Assume x is batched but t is not.
            """
            score = self.s(x, t)
            s_norm = jnp.linalg.norm(score, axis=-1) ** 2
            return -(compute_div(self.bf, x, t) + self.eps * s_norm)

        self.bf = bf
        self.br = br
        self.dt_logp = dt_logp
        self.start, self.end = self.start_end[0], self.start_end[1]
        self.ts = torch.linspace(self.start, self.end, self.n_step)
        self.dt = self.ts[1] - self.ts[0]
