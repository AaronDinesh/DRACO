from typing import Callable, Literal

import jax
from flax import nnx
from jax import Array

InterpolantCoefficient = Callable[[float], Array]
TimeSchedule = Literal["linear", "cosine", "power"]
GammaType = Literal["brownian", "a-brownian", "zero", "bsquared", "sinesquared", "sigmoid"]
Velocity = nnx.Module
Score = nnx.Module
Array = jax.Array
