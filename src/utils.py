from pathlib import Path
from typing import Callable, Literal

import flax.nnx as nnx
import jax.numpy as jnp
import orbax.checkpoint as ocp

from src.typing import TransformName

_STD_CHKPTR = ocp.StandardCheckpointer()


def make_transform(
    name: TransformName,
    scale: float = 1.0,
) -> tuple[Callable[[jnp.ndarray], jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]]:
    """
    Returns (forward, inverse) intensity transforms.
    The 'scale' roughly sets the transition between linear and log-like behavior.
    """
    if name == "none":
        return (lambda x: x, lambda y: y)

    if name == "asinh":
        # y = asinh(x/s),  x = s*sinh(y)
        s = float(scale)

        def forward(x: jnp.ndarray):
            return jnp.arcsinh(x / s)

        def inverse(y: jnp.ndarray):
            return s * jnp.sinh(y)

        return forward, inverse

    if name == "signed_log1p":
        # y = sign(x) * log(1 + |x|/s),  x = sign(y)*s*(exp(|y|)-1)
        s = float(scale)

        def forward(x: jnp.ndarray):
            return jnp.sign(x) * jnp.log1p(jnp.abs(x) / s)

        def inverse(y: jnp.ndarray):
            return jnp.sign(y) * s * (jnp.expm1(jnp.abs(y)))

        return forward, inverse

    raise ValueError(f"Unknown transform name: {name!r}")  # pyright: ignore[reportUnreachable]


def save_checkpoint(
    checkpoint_dir: str,
    epoch: int = 0,
    step: int = 0,
    model: nnx.Module | None = None,
    optimizer: nnx.Optimizer | None = None,  # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType],
    model_name: str | None = None,
    alt_name: str | None = None,
) -> None:
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    _, model_state = nnx.split(model)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    _, opt_state = nnx.split(optimizer)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType, reportUnknownArgumentType]

    payload = {  # pyright: ignore[reportUnknownVariableType]
        "model_state": model_state,
        "opt_state": opt_state,
    }

    if alt_name is None:
        save_path = ckpt_dir / f"{model_name}_epoch_{epoch:07d}_step_{step:07d}"
    else:
        save_path = ckpt_dir / alt_name

    _STD_CHKPTR.save(str(save_path), payload)


def restore_checkpoint(checkpoint_path: str, model: nnx.Module, optimizer: nnx.Optimizer):  # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType]
    checkpoint = _STD_CHKPTR.restore(checkpoint_path)  # pyright: ignore[reportAny]

    nnx.update(model, checkpoint["model_state"])  # pyright: ignore[reportUnknownMemberType]
    nnx.update(optimizer, checkpoint["opt_state"])  # pyright: ignore[reportUnknownMemberType]
