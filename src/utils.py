import jax.numpy as jnp


def to_log_space(
    x: jnp.ndarray, eps: float = 1e-6, offset: jnp.ndarray | None = None
) -> tuple[jnp.ndarray, jnp.ndarray]:
    # x: NHWC, can contain negatives if background-subtracted
    if offset is None:
        # choose a per-image/channel offset so min>=0
        minv = jnp.min(x, axis=(1, 2), keepdims=True)
        offset = jnp.maximum(-minv, 0.0) + eps
    x_shift = x + offset
    x_log = jnp.log1p(jnp.maximum(x_shift, eps))
    return x_log, offset  # keep offset to invert later


def from_log_space(x_log: jnp.ndarray, offset: jnp.ndarray) -> jnp.ndarray:
    return jnp.expm1(x_log) - offset
