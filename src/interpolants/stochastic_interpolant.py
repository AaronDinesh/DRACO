from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

# ---------------------------
# Embeddings
# ---------------------------


def sinusoidal_time_embed(t: jnp.ndarray, dim: int) -> jnp.ndarray:
    """t: (B,) in [0,1] -> (B, dim) sinusoidal embedding."""
    half = dim // 2
    freqs = jnp.exp(jnp.linspace(0.0, jnp.log(10000.0), half))
    t = t[:, None]
    angles = t * freqs[None, :]
    emb = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
    if dim % 2 == 1:
        emb = jnp.pad(emb, ((0, 0), (0, 1)))
    return emb


class MLP(nnx.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, *, rngs: nnx.Rngs):
        self.lin1 = nnx.Linear(in_dim, hidden, rngs=rngs)
        self.lin2 = nnx.Linear(hidden, out_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.lin1(x)
        x = jax.nn.silu(x)
        x = self.lin2(x)
        return x


# ---------------------------
# Attention block (uses nnx.MultiHeadAttention)
# ---------------------------


class AttentionBlock(nnx.Module):
    """
    Self-attention over spatial positions:
      (B,H,W,C) -> flatten to (B,HW,C) -> MHA -> reshape back.
    """

    def __init__(self, c: int, num_heads: int = 4, *, rngs: nnx.Rngs):
        # Match features in/out to channel dim
        self.mha = nnx.MultiHeadAttention(
            in_features=c,
            num_heads=num_heads,
            qkv_features=c,
            out_features=c,
            rngs=rngs,
        )
        # Normalize across channels (last axis); nnx handles NHWC just fine.
        self.norm = nnx.LayerNorm(
            num_features=c,
            feature_axes=(-1,),
            epsilon=1e-6,
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B,H,W,C)
        B, H, W, C = x.shape
        y = self.norm(x)  # (B,H,W,C)
        y = y.reshape(B, H * W, C)  # (B,N,C)
        # Self-attention: Q=K=V=y
        y = self.mha(y, y, y)  # (B,N,C)
        y = y.reshape(B, H, W, C)  # (B,H,W,C)
        return x + y  # residual


# ---------------------------
# FiLM Residual Block (uses nnx.LayerNorm)
# ---------------------------


class FiLMResBlock(nnx.Module):
    def __init__(self, c_in: int, c_out: int, cond_dim: int, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(c_in, c_out, (3, 3), padding="SAME", rngs=rngs)
        self.norm1 = nnx.LayerNorm(num_features=c_out, feature_axes=(-1,), epsilon=1e-6, rngs=rngs)
        self.conv2 = nnx.Conv(c_out, c_out, (3, 3), padding="SAME", rngs=rngs)
        self.norm2 = nnx.LayerNorm(num_features=c_out, feature_axes=(-1,), epsilon=1e-6, rngs=rngs)
        self.mod1 = nnx.Linear(cond_dim, 2 * c_out, rngs=rngs)
        self.mod2 = nnx.Linear(cond_dim, 2 * c_out, rngs=rngs)
        self.skip = (
            None if c_in == c_out else nnx.Conv(c_in, c_out, (1, 1), padding="SAME", rngs=rngs)
        )

    def _apply_mod(self, y: jnp.ndarray, m: jnp.ndarray) -> jnp.ndarray:
        s, b = jnp.split(m, 2, axis=-1)
        return y * (1 + s[:, None, None, :]) + b[:, None, None, :]

    def __call__(self, x: jnp.ndarray, cond_vec: jnp.ndarray) -> jnp.ndarray:
        h = self.conv1(x)
        h = self.norm1(h)
        h = self._apply_mod(h, self.mod1(cond_vec))
        h = jax.nn.silu(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = self._apply_mod(h, self.mod2(cond_vec))

        if self.skip is not None:
            x = self.skip(x)
        return jax.nn.silu(h + x)


# ---------------------------
# U-Net with nnx attention & norm
# ---------------------------


class DownBlock(nnx.Module):
    def __init__(
        self, c_in: int, c_out: int, cond_dim: int, attn: bool, num_heads: int, *, rngs: nnx.Rngs
    ):
        self.res1 = FiLMResBlock(c_in, c_out, cond_dim, rngs=rngs)
        self.res2 = FiLMResBlock(c_out, c_out, cond_dim, rngs=rngs)
        self.attn = AttentionBlock(c_out, num_heads=num_heads, rngs=rngs) if attn else None
        self.down = nnx.Conv(c_out, c_out, (3, 3), strides=(2, 2), padding="SAME", rngs=rngs)

    def __call__(self, x: jnp.ndarray, cond: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        x = self.res1(x, cond)
        x = self.res2(x, cond)
        if self.attn is not None:
            x = self.attn(x)
        skip = x
        x = self.down(x)
        return x, skip


class UpBlock(nnx.Module):
    def __init__(
        self, c_in: int, c_out: int, cond_dim: int, attn: bool, num_heads: int, *, rngs: nnx.Rngs
    ):
        # Up: resize (robust) + conv; avoids ConvTranspose checkerboard
        self.conv_up = nnx.Conv(c_in, c_out, (3, 3), padding="SAME", rngs=rngs)
        self.res1 = FiLMResBlock(c_out * 2, c_out, cond_dim, rngs=rngs)  # concat skip
        self.res2 = FiLMResBlock(c_out, c_out, cond_dim, rngs=rngs)
        self.attn = AttentionBlock(c_out, num_heads=num_heads, rngs=rngs) if attn else None

    def __call__(self, x: jnp.ndarray, skip: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
        B, Hs, Ws, _ = skip.shape
        x = jax.image.resize(x, (B, Hs, Ws, x.shape[-1]), method="bilinear")
        x = self.conv_up(x)
        x = jnp.concatenate([x, skip], axis=-1)
        x = self.res1(x, cond)
        x = self.res2(x, cond)
        if self.attn is not None:
            x = self.attn(x)
        return x


class StochasticInterpolantModel(nnx.Module):
    """
    U-Net with nnx.LayerNorm + nnx.MultiHeadAttention that predicts:
      - velocity b_hat(t, x_t, c)
      - denoiser eta_hat(t, x_t, c)
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        cond_in_dim: int,
        time_dim: int = 128,
        cond_dim: int = 256,
        num_heads: int = 4,
        *,
        rngs: nnx.Rngs,
    ):
        # embeddings
        self.time_mlp = MLP(time_dim, cond_dim, cond_dim, rngs=rngs)
        self.cond_mlp = MLP(cond_in_dim, cond_dim, cond_dim, rngs=rngs)

        # stem
        self.stem = nnx.Conv(in_channels, base_channels, (3, 3), padding="SAME", rngs=rngs)

        # encoder
        self.down1 = DownBlock(
            base_channels, base_channels, cond_dim, attn=False, num_heads=num_heads, rngs=rngs
        )
        self.down2 = DownBlock(
            base_channels, base_channels * 2, cond_dim, attn=True, num_heads=num_heads, rngs=rngs
        )
        self.down3 = DownBlock(
            base_channels * 2,
            base_channels * 4,
            cond_dim,
            attn=True,
            num_heads=num_heads,
            rngs=rngs,
        )

        # bottleneck
        self.mid_res1 = FiLMResBlock(base_channels * 4, base_channels * 4, cond_dim, rngs=rngs)
        self.mid_attn = AttentionBlock(base_channels * 4, num_heads=num_heads, rngs=rngs)
        self.mid_res2 = FiLMResBlock(base_channels * 4, base_channels * 4, cond_dim, rngs=rngs)

        # decoder
        self.up3 = UpBlock(
            base_channels * 4,
            base_channels * 2,
            cond_dim,
            attn=True,
            num_heads=num_heads,
            rngs=rngs,
        )
        self.up2 = UpBlock(
            base_channels * 2, base_channels, cond_dim, attn=True, num_heads=num_heads, rngs=rngs
        )
        self.up1 = UpBlock(
            base_channels, base_channels, cond_dim, attn=False, num_heads=num_heads, rngs=rngs
        )

        # heads
        self.head_b = nnx.Conv(base_channels, in_channels, (1, 1), padding="SAME", rngs=rngs)
        self.head_eta = nnx.Conv(base_channels, in_channels, (1, 1), padding="SAME", rngs=rngs)

        self._time_dim = time_dim

    def _fused_cond(self, t: jnp.ndarray, cond_vec: jnp.ndarray) -> jnp.ndarray:
        t_emb = sinusoidal_time_embed(t, self._time_dim)
        t_feat = self.time_mlp(t_emb)
        c_feat = self.cond_mlp(cond_vec)
        return jax.nn.silu(t_feat + c_feat)

    def __call__(
        self, x_t: jnp.ndarray, t: jnp.ndarray, cond_vec: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        fused = self._fused_cond(t, cond_vec)

        h0 = self.stem(x_t)
        h1, s1 = self.down1(h0, fused)
        h2, s2 = self.down2(h1, fused)
        h3, s3 = self.down3(h2, fused)

        h = self.mid_res1(h3, fused)
        h = self.mid_attn(h)
        h = self.mid_res2(h, fused)

        h = self.up3(h, s3, fused)
        h = self.up2(h, s2, fused)
        h = self.up1(h, s1, fused)

        b_hat = self.head_b(h)
        eta_hat = self.head_eta(h)
        return b_hat, eta_hat
