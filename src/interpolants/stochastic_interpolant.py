from typing import Callable, Literal

import jax
import jax.numpy as jnp
from jax import Array

from .utils import make_gamma

InterpolantCoefficient = Callable[[float], Array]
TimeSchedule = Literal["linear", "cosine", "power"]
GammaType = Literal["brownian", "a-brownian", "zero", "bsquared", "sinesquared", "sigmoid"]


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
