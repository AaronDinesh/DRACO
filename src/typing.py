from collections.abc import Generator
from typing import Callable, Literal, TypedDict

from jax import Array

TransformName = Literal["none", "asinh", "signed_log1p"]


class Batch(TypedDict):
    inputs: Array
    targets: Array
    params: Array


# loader(key=None, drop_last=False) -> iterator of Batch
Loader = Callable[[Array | None, bool], Generator[Batch]]
