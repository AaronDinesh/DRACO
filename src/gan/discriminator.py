from typing import Literal, Tuple, Union

import jax
import jax.numpy as jnp
from flax import nnx
from jax import lax, random

Array = jax.Array
Layer = Union[nnx.Linear, nnx.Conv]
Kind = Tuple[
    Literal["linear", "conv"], Tuple[int, ...]
]  # ("linear", (in,out)) or ("conv",(kh,kw,cin,cout))


class SpectralNorm(nnx.Module):
    """
    Spectral Normalization wrapper for nnx.Linear / nnx.Conv.
    - Keeps (u, v) as nnx state.
    - Uses reparameterization: W̄ = W / σ(W) for the forward pass.
    - Updates (u, v) via power iteration during training.
    """

    layer: Layer
    iters: int
    eps: float
    _kind: Kind
    u: nnx.Variable  # state: Array[o]
    v: nnx.Variable  # state: Array[k]

    def __init__(
        self,
        layer: Layer,
        *,
        iters: int = 10,
        eps: float = 1e-12,
        warmup: int = 15,
        rngs: nnx.Rngs,
    ):
        self.layer = layer
        self.iters = int(iters)
        self.eps = float(eps)

        # Flatten kernel to (out, in)
        k = layer.kernel
        if k.ndim == 2:  # Linear: (in, out) -> (out, in)
            W2d: Array = k.T
            self._kind = ("linear", tuple(k.shape))  # (in, out)
        else:  # Conv: (kh, kw, cin, cout) -> (out, kh*kw*cin)
            kh, kw, cin, cout = k.shape
            W2d = jnp.reshape(k, (kh * kw * cin, cout)).T
            self._kind = ("conv", (kh, kw, cin, cout))

        o, i = W2d.shape  # out_dim, in_dim
        k1, k2 = rngs()
        u: Array = random.normal(k1, (o,), dtype=W2d.dtype)
        v: Array = random.normal(k2, (i,), dtype=W2d.dtype)

        # Warmup power iterations (stabilize u,v)
        for _ in range(int(warmup)):
            v = (u @ W2d) / (jnp.linalg.norm(u @ W2d) + self.eps)
            u = (W2d @ v) / (jnp.linalg.norm(W2d @ v) + self.eps)

        self.u = nnx.Variable("state", u)
        self.v = nnx.Variable("state", v)

    def __call__(self, x: Array, *, training: bool = True) -> jnp.ndarray:
        u: Array = self.u.value
        v: Array = self.v.value

        # Rebuild flattened weight (out, in)
        k = self.layer.kernel
        if self._kind[0] == "linear":
            W2d: Array = k.T
        else:
            kh, kw, cin, cout = self._kind[1]
            W2d = jnp.reshape(k, (kh * kw * cin, cout)).T

        # Power iteration on stop-gradient weights
        if training:
            Wsg: Array = lax.stop_gradient(W2d)
            for _ in range(self.iters):
                v = (u @ Wsg) / (jnp.linalg.norm(u @ Wsg) + self.eps)
                u = (Wsg @ v) / (jnp.linalg.norm(Wsg @ v) + self.eps)
            self.u.value, self.v.value = u, v

        sigma: Array = jnp.einsum("o,ok,k->", u, W2d, v)  # scalar
        Wbar2d: Array = W2d / (sigma + self.eps)

        # Unflatten + forward
        if self._kind[0] == "linear":
            # Linear: Wbar (in, out)
            Wbar: Array = Wbar2d.T
            y: Array = jnp.tensordot(x, Wbar, axes=([-1], [0]))  # (..., in) @ (in, out)
            if self.layer.bias is not None:
                y = y + self.layer.bias
            return y
        else:
            kh, kw, cin, cout = self._kind[1]
            Wbar = jnp.reshape(Wbar2d.T, (kh, kw, cin, cout))  # HWIO

            # Ensure dilations are tuples (lax requires sequences)
            lhs_dil: Tuple[int, int] = getattr(self.layer, "lhs_dilation", (1, 1))
            rhs_dil: Tuple[int, int] = getattr(
                self.layer, "kernel_dilation", getattr(self.layer, "dilation", (1, 1))
            )
            y = lax.conv_general_dilated(
                lhs=x,
                rhs=Wbar,
                window_strides=self.layer.strides,  # Tuple[int, int]
                padding=self.layer.padding,  # str | sequence
                lhs_dilation=lhs_dil,
                rhs_dilation=rhs_dil,
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
                feature_group_count=getattr(self.layer, "feature_group_count", 1),
            )
            if self.layer.bias is not None:
                y = y + self.layer.bias[None, None, None, :]
            return y


class Discriminator(nnx.Module):
    condition_projection: nnx.Linear
    conv1: SpectralNorm
    conv2: SpectralNorm
    conv3: SpectralNorm
    conv4: SpectralNorm
    condition_dimension: int
    condition_proj_dim: int

    def __init__(
        self,
        key: Array,
        in_channels: int,
        condition_dim: int,
        condition_proj_dim: int,
        base_conv_dim: int = 64,
    ):
        (
            condition_projection_key,
            conv1_key,
            conv2_key,
            conv3_key,
            conv4_key,
        ) = random.split(key, 5)

        self.condition_projection = nnx.Linear(
            in_features=condition_dim,
            out_features=condition_proj_dim,
            rngs=nnx.Rngs(condition_projection_key),
        )

        self.conv1: SpectralNorm = SpectralNorm(
            nnx.Conv(
                in_features=2 * in_channels + condition_proj_dim,
                out_features=base_conv_dim,
                strides=2,
                kernel_size=(4, 4),
                padding="SAME",
                rngs=nnx.Rngs(conv1_key),
            ),
            iters=1,
            rngs=nnx.Rngs(conv1_key),
        )

        self.conv2: SpectralNorm = SpectralNorm(
            nnx.Conv(
                in_features=base_conv_dim,
                out_features=base_conv_dim * 2,
                strides=2,
                kernel_size=(4, 4),
                padding="SAME",
                rngs=nnx.Rngs(conv2_key),
            ),
            iters=1,
            rngs=nnx.Rngs(conv2_key),
        )

        self.conv3: SpectralNorm = SpectralNorm(
            nnx.Conv(
                in_features=base_conv_dim * 2,
                out_features=base_conv_dim * 4,
                strides=2,
                kernel_size=(4, 4),
                padding="SAME",
                rngs=nnx.Rngs(conv3_key),
            ),
            iters=1,
            rngs=nnx.Rngs(conv3_key),
        )

        self.conv4: SpectralNorm = SpectralNorm(
            nnx.Conv(
                in_features=base_conv_dim * 4,
                out_features=1,
                strides=1,
                kernel_size=(4, 4),
                padding="SAME",
                rngs=nnx.Rngs(conv4_key),
            ),
            iters=1,
            rngs=nnx.Rngs(conv4_key),
        )
        self.condition_proj_dim = condition_proj_dim

    def __call__(
        self,
        inputs: jnp.ndarray,
        output: jnp.ndarray,
        condition_params: jnp.ndarray,
        is_training: bool = True,
    ) -> jnp.ndarray:
        (batch, height, width, _) = inputs.shape

        # Project and turn in into [B, 1, 1, condition_proj_dim]
        condition_params_proj = self.condition_projection(condition_params)[:, None, None, :]

        condition_params_proj = jnp.broadcast_to(
            condition_params_proj, (batch, height, width, self.condition_proj_dim)
        )

        out = jnp.concatenate([inputs, output, condition_params_proj], axis=-1)

        out = nnx.leaky_relu(self.conv1(out, training=is_training), 0.2)
        out = nnx.leaky_relu(self.conv2(out, training=is_training), 0.2)
        out = nnx.leaky_relu(self.conv3(out, training=is_training), 0.2)
        logits = self.conv4(out, training=is_training)

        return logits
