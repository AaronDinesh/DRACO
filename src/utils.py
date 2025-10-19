import shutil
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
):
    """
    Returns (forward, inverse) intensity transforms.
    For 'asinh_viz': 2D map -> pretty grayscale [0,1] image using:
      - |B| magnitude
      - percentile clip [0.5, 99.5]
      - light Gaussian blur (sigma≈1.2 px)
      - asinh with robust 'scale' (if scale<=0 or non-finite, auto=p95(|blurred|))
      - min-max normalize to [0,1]
    NOTE: 'inverse' for 'asinh_viz' cannot undo clip/blur/normalize; it returns input.
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

    if name == "log10":
        ln10 = jnp.log(10.0)
        inv_ln10 = 1.0 / ln10

        def forward(x: jnp.ndarray):
            x_shape_dim = len(x.shape)

            if x_shape_dim == 2:
                reduce_axes = (0, 1)
            elif x_shape_dim == 3:
                reduce_axes = (1, 2)
            elif x_shape_dim == 4:
                reduce_axes = (1, 2)

            tiny = jnp.finfo(x.dtype).tiny
            g = jnp.log(jnp.maximum(x, tiny))  # (B,H,W,C)
            g_min = jnp.min(g, axis=reduce_axes, keepdims=True)  # (B,1,1,C) or (B,1,1,1)
            y = (g - g_min) * inv_ln10  # base-10, min->0
            return y

        def inverse(y: jnp.ndarray) -> jnp.ndarray:
            g = y * ln10
            return jnp.exp(g)

        return forward, inverse

    if name == "signed_log10":
        # y = sign(x) * log10(1 + |x|/s)
        # x = sign(y) * s * (10**|y| - 1)
        # Symmetric, zero-centered, reversible (no clipping/blur inside)
        s = float(scale)

        def forward(x: jnp.ndarray):
            ax = jnp.abs(x)
            return jnp.sign(x) * jnp.log10(1.0 + ax / s)

        def inverse(y: jnp.ndarray):
            ay = jnp.abs(y)
            return jnp.sign(y) * s * (jnp.power(10.0, ay) - 1.0)

    if name == "asinh_viz":
        # Self-contained helpers scoped inside the block to minimize globals.
        def _percentile(x, q):
            x = jnp.sort(x.reshape(-1))
            idx = (q / 100.0) * (x.size - 1)
            lo = jnp.floor(idx).astype(int)
            hi = jnp.ceil(idx).astype(int)
            w = idx - lo
            return (1.0 - w) * x[lo] + w * x[hi]

        def _gaussian_blur2d(img, sigma=1.2):
            r = int(jnp.ceil(3.0 * sigma))
            xk = jnp.arange(-r, r + 1)
            k = jnp.exp(-0.5 * (xk / sigma) ** 2)
            k = k / jnp.sum(k)

            def _conv1d(a, k, axis):
                pad = [(0, 0)] * a.ndim
                pad[axis] = (r, r)
                a = jnp.pad(a, pad, mode="reflect")
                idx = jnp.arange(a.shape[axis] - 2 * r)
                out = 0.0
                for i, kv in enumerate(k):
                    out = out + kv * jnp.take(a, idx + i, axis=axis)
                return out

            out = _conv1d(img, k, axis=0)
            out = _conv1d(out, k, axis=1)
            return out

        s_user = float(scale)

        def forward(x2d: jnp.ndarray) -> jnp.ndarray:
            # 1) magnitude (grayscale look)
            mag = jnp.abs(x2d)

            # 2) robust clip to suppress outliers
            lo = _percentile(mag, 0.5)
            hi = _percentile(mag, 99.5)
            mag = jnp.clip(mag, lo, hi)

            # 3) gentle blur to reveal filaments
            mag = _gaussian_blur2d(mag, sigma=1.2)

            # 4) robust asinh scale
            if not jnp.isfinite(s_user) or s_user <= 0.0:
                s = _percentile(jnp.abs(mag), 95.0)
            else:
                s = s_user
            s = jnp.maximum(s, 1e-12)

            y = jnp.arcsinh(mag / s)

            # 5) normalize to [0,1] for display
            ymin = jnp.min(y)
            ymax = jnp.max(y)
            y = (y - ymin) / (ymax - ymin + 1e-12)
            return y

        def inverse(y_disp: jnp.ndarray) -> jnp.ndarray:
            # Not invertible (clip/blur/normalize lose information).
            # Return input as a no-op to keep API-compatible.
            return y_disp

        return forward, inverse

    raise ValueError(f"Unknown transform name: {name!r}")


def save_checkpoint(
    checkpoint_dir: str,
    epoch: int = 0,
    step: int = 0,
    model: nnx.Module | None = None,
    optimizer: nnx.Optimizer | None = None,  # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType],
    model_name: str | None = None,
    alt_name: str | None = None,
    data_stats=None,
) -> None:
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    _, model_state = nnx.split(model)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    _, opt_state = nnx.split(optimizer)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType, reportUnknownArgumentType]

    payload = {  # pyright: ignore[reportUnknownVariableType]
        "model_state": model_state,
        "opt_state": opt_state,
    }

    if data_stats is not None:
        payload["data_stats"] = data_stats  # <— NEW

    if alt_name is None:
        save_path = ckpt_dir / f"{model_name}_epoch_{epoch:07d}_step_{step:07d}"
    else:
        save_path = ckpt_dir / alt_name

    _STD_CHKPTR.save(str(save_path), payload)


def restore_checkpoint(checkpoint_path: str, model: nnx.Module, optimizer: nnx.Optimizer):  # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType]
    checkpoint = _STD_CHKPTR.restore(checkpoint_path)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    nnx.update(model, checkpoint["model_state"])  # pyright: ignore[reportUnknownMemberType]
    nnx.update(optimizer, checkpoint["opt_state"])  # pyright: ignore[reportUnknownMemberType]

    return checkpoint.get("data_stats", None)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]


def delete_checkpoint(checkpoint_path: str, folder_name: str) -> None:
    checkpoint_folder = Path(checkpoint_path, folder_name)
    shutil.rmtree(checkpoint_folder)
