from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx


class AdaLN(nnx.Module):
    """
    Adaptive LayerNorm: y = LayerNorm(x) * (1 + scale(cond)) + bias(cond)
    cond -> Linear -> [scale, bias]
    """

    def __init__(self, cond_dim: int, num_channels: int, *, rngs: nnx.Rngs):
        self.norm = nnx.LayerNorm(
            num_features=num_channels, feature_axes=(-1,), epsilon=1e-6, rngs=rngs
        )
        self.to_modulation = nnx.Linear(cond_dim, 2 * num_channels, rngs=rngs)

    def __call__(self, x: jnp.ndarray, cond_features: jnp.ndarray) -> jnp.ndarray:
        # x: (B, H, W, C), cond_features: (B, cond_dim)
        x_norm = self.norm(x)
        modulation = self.to_modulation(cond_features)  # (B, 2*C)
        scale, bias = jnp.split(modulation, 2, axis=-1)  # (B, C), (B, C)
        return x_norm * (1.0 + scale[:, None, None, :]) + bias[:, None, None, :]


class ResBlock(nnx.Module):
    """
    DDPM-style residual block:
      x -> AdaLN -> SiLU -> Conv3x3 ->
           AdaLN -> SiLU -> Conv3x3 -> (+ skip 1x1 if channels change)
    """

    def __init__(
        self, in_channels: int, out_channels: int, cond_dim: int, *, rngs: nnx.Rngs
    ):
        self.adaln_1 = AdaLN(cond_dim, in_channels, rngs=rngs)
        self.conv_1 = nnx.Conv(
            in_channels, out_channels, (3, 3), padding="SAME", rngs=rngs
        )

        self.adaln_2 = AdaLN(cond_dim, out_channels, rngs=rngs)
        self.conv_2 = nnx.Conv(
            out_channels, out_channels, (3, 3), padding="SAME", rngs=rngs
        )

        # Match channels for residual if needed
        self.residual_proj = None
        if in_channels != out_channels:
            self.residual_proj = nnx.Conv(
                in_channels, out_channels, (1, 1), padding="SAME", rngs=rngs
            )

    def __call__(
        self,
        x: jnp.ndarray,
        cond_features: jnp.ndarray,
    ) -> jnp.ndarray:
        residual = x
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)

        hidden = self.adaln_1(x, cond_features)
        hidden = jax.nn.silu(hidden)
        hidden = self.conv_1(hidden)

        hidden = self.adaln_2(hidden, cond_features)
        hidden = jax.nn.silu(hidden)
        hidden = self.conv_2(hidden)

        return residual + hidden


def sinusoidal_time_embed(t: jnp.ndarray, dim: int) -> jnp.ndarray:
    """
    DDPM-style sinusoidal embedding.
    Args:
        t:   (B,) diffusion timesteps (float or int). Works with normalized t in [0,1] or raw.
        dim: embedding dimension (even).
    Returns:
        (B, dim) sinusoidal embedding [sin, cos] with exponentially spaced frequencies.
    """
    assert dim % 2 == 0, "sinusoidal_time_embed: dim must be even"
    half = dim // 2

    # Follow the transformer/DDPM frequency schedule:
    # freqs_k = exp(-ln(10000) * k / (half - 1)) for k in [0..half-1]
    # If you pass integer timesteps, treat t as float for continuous embedding.
    t = t.astype(jnp.float32)
    k = jnp.arange(half, dtype=jnp.float32)
    freqs = jnp.exp(-jnp.log(10000.0) * k / (half - 1.0))  # (half,)
    # (B,1) * (half,) -> (B, half)
    angles = t[:, None] * freqs[None, :]
    emb = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)  # (B, dim)
    return emb


class AttentionBlock(nnx.Module):
    def __init__(self, channels: int, *, rngs: nnx.Rngs):
        # Use NNX GroupNorm in channel-last mode
        self.norm = nnx.GroupNorm(
            num_groups=32,
            num_features=channels,
            epsilon=1e-6,
            rngs=rngs,
        )
        self.qkv = nnx.Conv(channels, 3 * channels, (1, 1), padding="SAME", rngs=rngs)
        self.proj_out = nnx.Conv(channels, channels, (1, 1), padding="SAME", rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        x_norm = self.norm(x)  # (B,H,W,C), GN32 over spatial+group channels
        qkv = self.qkv(x_norm)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        b, h, w, c = q.shape
        hw = h * w
        q = q.reshape(b, hw, c)
        k = k.reshape(b, hw, c)
        v = v.reshape(b, hw, c)
        scale = 1.0 / jnp.sqrt(c)

        # attn = jax.nn.softmax(jnp.einsum("bic,bjc->bij", q, k) * scale, axis=-1)
        # out = jnp.einsum("bij,bjc->bic", attn, v).reshape(b, h, w, c)

        out = jax.nn.dot_product_attention(q, k, v, scale=scale).reshape(b, h, w, c)
        out = self.proj_out(out)
        return residual + out


class DownBlock(nnx.Module):
    """
    Two ResBlocks (+ optional attention), then strided Conv downsample.
    Returns (downsampled, skip_connection).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        *,
        with_attn: bool,
        rngs: nnx.Rngs,
    ):
        self.resblock_1 = ResBlock(in_channels, out_channels, cond_dim, rngs=rngs)
        self.resblock_2 = ResBlock(out_channels, out_channels, cond_dim, rngs=rngs)
        self.attention = AttentionBlock(out_channels, rngs=rngs) if with_attn else None

        self.downsample = nnx.Conv(
            out_channels,
            out_channels,
            (3, 3),
            strides=(2, 2),
            padding="SAME",
            rngs=rngs,
        )

    def __call__(
        self,
        x: jnp.ndarray,
        cond_features: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        hidden = self.resblock_1(x, cond_features)
        hidden = self.resblock_2(hidden, cond_features)
        if self.attention is not None:
            hidden = self.attention(hidden)
        skip_connection = hidden
        hidden = self.downsample(hidden)
        return hidden, skip_connection


class UpBlock(nnx.Module):
    """
    Nearest/bilinear upsample -> Conv, concat skip -> two ResBlocks (+ optional attention).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        cond_dim: int,
        *,
        with_attn: bool,
        rngs: nnx.Rngs,
    ):
        self.reduce_after_resize = nnx.Conv(
            in_channels, out_channels, (3, 3), padding="SAME", rngs=rngs
        )
        self.resblock_1 = ResBlock(
            out_channels + skip_channels, out_channels, cond_dim, rngs=rngs
        )
        self.resblock_2 = ResBlock(out_channels, out_channels, cond_dim, rngs=rngs)
        self.attention = AttentionBlock(out_channels, rngs=rngs) if with_attn else None

    def __call__(
        self,
        x: jnp.ndarray,
        skip_connection: jnp.ndarray,
        cond_features: jnp.ndarray,
    ) -> jnp.ndarray:
        batch, height, width, _ = skip_connection.shape
        # Resize to skip spatial size
        x_resized = jax.image.resize(
            x, (batch, height, width, x.shape[-1]), method="bilinear"
        )
        hidden = self.reduce_after_resize(x_resized)
        hidden = jnp.concatenate([hidden, skip_connection], axis=-1)

        hidden = self.resblock_1(hidden, cond_features)
        hidden = self.resblock_2(hidden, cond_features)
        if self.attention is not None:
            hidden = self.attention(hidden)
        return hidden


class MLP(nnx.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, rngs: nnx.Rngs):
        self.lin1 = nnx.Linear(in_dim, hidden, rngs=rngs)
        self.lin2 = nnx.Linear(hidden, out_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.lin1(x)
        x = jax.nn.silu(x)
        x = self.lin2(x)
        return x


class StochasticInterpolantModel(nnx.Module):
    """
    DDPM-style U-Net backbone with proper AdaLN conditioning.
    Conditioning = fused(time_embedding, cosmology_embedding), injected in every ResBlock.
    Outputs:
        b_hat   : (B, H, W, C_in)
        eta_hat : (B, H, W, C_in)
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        cosmology_dim: int,
        *,
        time_embedding_dim: int = 128,
        fused_cond_dim: int = 256,
        rngs: nnx.Rngs,
    ):
        # Embeddings (reuse your helpers)
        self.time_mlp = MLP(
            time_embedding_dim, fused_cond_dim, fused_cond_dim, rngs=rngs
        )
        self.cosmo_mlp = MLP(cosmology_dim, fused_cond_dim, fused_cond_dim, rngs=rngs)

        # Stem
        self.stem = nnx.Conv(
            in_channels, base_channels, (3, 3), padding="SAME", rngs=rngs
        )

        # Encoder
        self.down_block_1 = DownBlock(
            base_channels,
            base_channels,
            fused_cond_dim,
            with_attn=False,
            rngs=rngs,
        )
        self.down_block_2 = DownBlock(
            base_channels,
            base_channels * 2,
            fused_cond_dim,
            with_attn=True,
            rngs=rngs,
        )
        self.down_block_3 = DownBlock(
            base_channels * 2,
            base_channels * 4,
            fused_cond_dim,
            with_attn=True,
            rngs=rngs,
        )

        # Bottleneck
        self.mid_resblock_1 = ResBlock(
            base_channels * 4, base_channels * 4, fused_cond_dim, rngs=rngs
        )
        self.mid_attention = AttentionBlock(base_channels * 4, rngs=rngs)
        self.mid_resblock_2 = ResBlock(
            base_channels * 4, base_channels * 4, fused_cond_dim, rngs=rngs
        )

        # Decoder
        self.up_block_3 = UpBlock(
            in_channels=base_channels * 4,
            out_channels=base_channels * 2,
            skip_channels=base_channels * 4,
            cond_dim=fused_cond_dim,
            with_attn=True,
            rngs=rngs,
        )
        self.up_block_2 = UpBlock(
            in_channels=base_channels * 2,
            out_channels=base_channels,
            skip_channels=base_channels * 2,
            cond_dim=fused_cond_dim,
            with_attn=True,
            rngs=rngs,
        )
        self.up_block_1 = UpBlock(
            in_channels=base_channels,
            out_channels=base_channels,
            skip_channels=base_channels,
            cond_dim=fused_cond_dim,
            with_attn=False,
            rngs=rngs,
        )

        # Heads for SI outputs
        self.head_b = nnx.Conv(
            base_channels, in_channels, (1, 1), padding="SAME", rngs=rngs
        )
        self.head_eta = nnx.Conv(
            base_channels, in_channels, (1, 1), padding="SAME", rngs=rngs
        )

        self._time_embedding_dim = time_embedding_dim

    def _fused_condition(
        self, time_scalar: jnp.ndarray, cosmology_vector: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Build fused conditioning features from time and cosmology vectors.
        time_scalar: (B,) in [0, 1] or diffusion time domain
        cosmology_vector: (B, D_cosmo)
        """
        time_positional = sinusoidal_time_embed(
            time_scalar, self._time_embedding_dim
        )  # (B, Tdim)
        time_features = self.time_mlp(time_positional)  # (B, Fdim)
        cosmo_features = self.cosmo_mlp(cosmology_vector)  # (B, Fdim)
        fused_features = jax.nn.silu(time_features + cosmo_features)  # (B, Fdim)
        return fused_features

    def __call__(
        self,
        x_t: jnp.ndarray,
        time_scalar: jnp.ndarray,
        cosmology_vector: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        x_t:               (B, H, W, C_in) noisy input at time t
        time_scalar:       (B,)
        cosmology_vector:  (B, cosmology_dim)
        """
        fused_cond = self._fused_condition(time_scalar, cosmology_vector)

        # Stem + Encoder
        hidden_0 = self.stem(x_t)
        hidden_1, skip_1 = self.down_block_1(hidden_0, fused_cond)
        hidden_2, skip_2 = self.down_block_2(hidden_1, fused_cond)
        hidden_3, skip_3 = self.down_block_3(hidden_2, fused_cond)

        # Bottleneck
        hidden_mid = self.mid_resblock_1(hidden_3, fused_cond)
        hidden_mid = self.mid_attention(hidden_mid)
        hidden_mid = self.mid_resblock_2(hidden_mid, fused_cond)

        # Decoder
        hidden = self.up_block_3(hidden_mid, skip_3, fused_cond)
        hidden = self.up_block_2(hidden, skip_2, fused_cond)
        hidden = self.up_block_1(hidden, skip_1, fused_cond)

        # SI heads
        b_hat = self.head_b(hidden)
        eta_hat = self.head_eta(hidden)
        return b_hat, eta_hat
