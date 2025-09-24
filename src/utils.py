import jax.numpy as jnp


def to_log_space(x, eps=1e-6, offset=None):
    # x: NHWC, can contain negatives if background-subtracted
    if offset is None:
        # choose a per-image/channel offset so min>=0
        minv = jnp.min(x, axis=(1, 2), keepdims=True)
        offset = jnp.maximum(-minv, 0.0) + eps
    x_shift = x + offset
    x_log = jnp.log1p(jnp.maximum(x_shift, eps))
    return x_log, offset  # keep offset to invert later


def from_log_space(x_log, offset):
    return jnp.expm1(x_log) - offset
