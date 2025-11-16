from typing import Callable, Literal

import jax
from jax import Array

InterpolantCoefficient = Callable[[float], Array]
TimeSchedule = Literal["linear", "cosine", "power"]
GammaType = Literal["brownian", "a-brownian", "zero", "bsquared", "sinesquared", "sigmoid"]
Velocity = jax.nnx.Module
Score = jax.nnx.Module
Array = jax.Array
