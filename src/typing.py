from collections.abc import Generator
from typing import Callable, TypedDict

from jax import Array


class Batch(TypedDict):
    inputs: Array
    targets: Array
    params: Array


# loader(key=None, drop_last=False) -> iterator of Batch
Loader = Callable[[Array | None, bool], Generator[Batch]]
