from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as random
from flax import nnx
from jax._src.typing import Array


def sinusoidal_time_embedding(
    timesteps: jnp.ndarray, dim: int, max_period: int = 10000
) -> jnp.ndarray:
    """Standard diffusion-style sinusoidal embedding for scalar timesteps."""
    if dim <= 0:
        raise ValueError("dim for time embeddings must be > 0")

    timesteps = jnp.asarray(timesteps, dtype=jnp.float32)
    if timesteps.ndim == 0:
        timesteps = timesteps[None]
    elif timesteps.ndim > 1:
        batch = timesteps.shape[0]
        timesteps = timesteps.reshape(batch, -1)
        timesteps = timesteps[:, 0]

    half_dim = dim // 2
    if half_dim == 0:
        return timesteps[:, None]

    frequencies = jnp.exp(
        -jnp.log(float(max_period)) * jnp.arange(half_dim, dtype=jnp.float32) / max(half_dim - 1, 1)
    )
    angles = timesteps[:, None] * frequencies[None, :]
    embeddings = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)

    if dim % 2 == 1:
        embeddings = jnp.pad(embeddings, ((0, 0), (0, 1)))
    return embeddings.astype(jnp.float32)


class TimeConditioner(nnx.Module):
    """Projects sinusoidal embeddings to a richer representation."""

    def __init__(self, key: Array, embed_dim: int):
        k1, k2 = random.split(key)
        hidden_dim = embed_dim * 2
        self.fc1 = nnx.Linear(
            in_features=embed_dim,
            out_features=hidden_dim,
            rngs=nnx.Rngs(k1),
        )
        self.fc2 = nnx.Linear(
            in_features=hidden_dim,
            out_features=embed_dim,
            rngs=nnx.Rngs(k2),
        )

    def __call__(self, embeddings: jnp.ndarray) -> jnp.ndarray:
        x = self.fc1(embeddings)
        x = nnx.silu(x)
        x = self.fc2(x)
        return x


def _group_calculation(channels: int) -> int:
    for g in (32, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class DownsampleBlock(nnx.Module):
    def __init__(
        self,
        key: Array,
        in_features: int,
        out_features: int,
        size: tuple[int, int],
        apply_groupnorm: bool = True,
    ):
        conv_key, norm_key = random.split(key)
        self.conv_block: nnx.Conv = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=size,
            strides=2,
            padding="SAME",
            use_bias=not apply_groupnorm,
            rngs=nnx.Rngs(conv_key),
        )
        self.group_norm: nnx.GroupNorm = nnx.GroupNorm(
            num_features=out_features,
            num_groups=_group_calculation(out_features),
            rngs=nnx.Rngs(norm_key),
        )
        self.apply_groupnorm = apply_groupnorm

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.conv_block(x)
        if self.apply_groupnorm:
            x = self.group_norm(x)
        return nnx.leaky_relu(x, 0.2)


class UpsampleBlock(nnx.Module):
    def __init__(
        self,
        key: Array,
        in_features: int,
        out_features: int,
        kernel_size: tuple[int, int],
        apply_groupnorm: bool = True,
    ):
        conv_key, norm_key = random.split(key)
        self.conv_block: nnx.Conv = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            strides=1,
            padding="SAME",
            rngs=nnx.Rngs(conv_key),
        )
        self.group_norm: nnx.GroupNorm = nnx.GroupNorm(
            num_features=out_features,
            num_groups=_group_calculation(out_features),
            rngs=nnx.Rngs(norm_key),
        )
        self.apply_groupnorm = apply_groupnorm

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch_size, height, width, channels = x.shape
        out_shape = (batch_size, height * 2, width * 2, channels)
        x = jax.image.resize(x, shape=out_shape, method="bilinear")
        x = self.conv_block(x)
        if self.apply_groupnorm:
            x = self.group_norm(x)
        return nnx.leaky_relu(x, 0.2)


class FiLM(nnx.Module):
    def __init__(self, key: Array, in_features: int, out_features: int):
        gamma_key, beta_key = random.split(key)
        self.gamma = nnx.Linear(
            in_features=in_features,
            out_features=out_features,
            rngs=nnx.Rngs(gamma_key),
        )
        self.beta = nnx.Linear(
            in_features=in_features,
            out_features=out_features,
            rngs=nnx.Rngs(beta_key),
        )

    def __call__(self, x: jnp.ndarray, conditioner: jnp.ndarray) -> jnp.ndarray:
        gamma = self.gamma(conditioner)[:, None, None, :]
        beta = self.beta(conditioner)[:, None, None, :]
        return x * gamma + beta


class StochasticInterpolantUNet(nnx.Module):
    """UNet that predicts the stochastic interpolant velocity field."""

    def __init__(
        self,
        key: Array,
        in_features: int,
        out_features: int,
        len_cosmos_params: int,
        time_embed_dim: int = 256,
    ):
        key_seq = iter(random.split(key, 64))

        def next_key() -> Array:
            return next(key_seq)

        self.time_embed_dim = time_embed_dim
        self.time_conditioner = TimeConditioner(next_key(), time_embed_dim)
        self.len_cosmos_params = len_cosmos_params
        conditioner_dim = len_cosmos_params + time_embed_dim

        down_channels = [64, 128, 256, 512, 512, 512, 512, 512]
        self.downsample_blocks: list[DownsampleBlock] = []
        in_ch = in_features
        for idx, out_ch in enumerate(down_channels):
            self.downsample_blocks.append(
                DownsampleBlock(
                    key=next_key(),
                    in_features=in_ch,
                    out_features=out_ch,
                    size=(4, 4),
                    apply_groupnorm=idx != 0,
                )
            )
            in_ch = out_ch

        upsample_layout = [
            (512, 512, True),
            (1024, 512, True),
            (1024, 512, True),
            (1024, 512, True),
            (1024, 256, True),
            (512, 128, True),
            (256, 64, True),
            (128, 64, False),
        ]
        self.upsample_blocks: list[UpsampleBlock] = [
            UpsampleBlock(
                key=next_key(),
                in_features=in_ch,
                out_features=upsample_layout[0][1],
                kernel_size=(4, 4),
                apply_groupnorm=upsample_layout[0][2],
            )
        ]

        for in_ch, out_ch, use_norm in upsample_layout[1:]:
            self.upsample_blocks.append(
                UpsampleBlock(
                    key=next_key(),
                    in_features=in_ch,
                    out_features=out_ch,
                    kernel_size=(4, 4),
                    apply_groupnorm=use_norm,
                )
            )

        skip_channels = down_channels[:-1]
        self.skip_conditioners: list[FiLM] = [
            FiLM(key=next_key(), in_features=conditioner_dim, out_features=c) for c in skip_channels
        ]
        self.bottleneck_conditioner = FiLM(
            key=next_key(),
            in_features=conditioner_dim,
            out_features=down_channels[-1],
        )
        self.output_stage = nnx.Conv(
            in_features=upsample_layout[-1][1],
            out_features=out_features,
            kernel_size=(3, 3),
            padding="SAME",
            rngs=nnx.Rngs(next_key()),
        )

    def _prepare_conditioning(
        self, cosmos_params: jnp.ndarray, timesteps: jnp.ndarray
    ) -> jnp.ndarray:
        cosmos_params = jnp.asarray(cosmos_params, dtype=jnp.float32)
        if cosmos_params.ndim == 1:
            if cosmos_params.shape[0] != self.len_cosmos_params:
                raise ValueError(
                    f"Expected cosmology vector of length {self.len_cosmos_params}, "
                    f"got {cosmos_params.shape[0]}"
                )
            cosmos_params = cosmos_params[None, :]
        elif cosmos_params.ndim > 2:
            batch = cosmos_params.shape[0]
            cosmos_params = cosmos_params.reshape(batch, -1)

        if cosmos_params.shape[-1] != self.len_cosmos_params:
            raise ValueError(
                f"Expected cosmology vector of length {self.len_cosmos_params}, "
                f"got {cosmos_params.shape[-1]}"
            )

        t_embed = sinusoidal_time_embedding(timesteps, self.time_embed_dim)
        t_embed = self.time_conditioner(t_embed)

        if cosmos_params.shape[0] != t_embed.shape[0]:
            raise ValueError(
                "Batch size for cosmology parameters and timesteps must match "
                f"({cosmos_params.shape[0]} vs {t_embed.shape[0]})."
            )
        return jnp.concatenate([cosmos_params, t_embed], axis=-1)

    def __call__(
        self,
        x: jnp.ndarray,
        cosmos_params: jnp.ndarray,
        timesteps: jnp.ndarray,
    ) -> jnp.ndarray:
        cond_vector = self._prepare_conditioning(cosmos_params, timesteps)

        skip_connections: list[jnp.ndarray] = []
        for block, conditioner in zip(self.downsample_blocks[:-1], self.skip_conditioners):
            x = block(x)
            skip_connections.append(conditioner(x, cond_vector))

        x = self.downsample_blocks[-1](x)
        x = self.bottleneck_conditioner(x, cond_vector)

        x = self.upsample_blocks[0](x)
        skip_connections = list(reversed(skip_connections))

        for up_block, skip in zip(self.upsample_blocks[1:], skip_connections):
            x = jnp.concatenate([x, skip], axis=-1)
            x = up_block(x)

        return self.output_stage(x)
