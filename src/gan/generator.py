import jax
import jax.numpy as jnp
import jax.random as random
from flax import nnx
from jax._src.typing import Array

# Need to impliment a UNet here


class downsample_block(nnx.Module):
    def __init__(
        self,
        key: Array,
        in_features: int,
        out_features: int,
        size: tuple[int, int],
        apply_groupnorm: bool = True,
    ):
        self.conv_key, self.groupnorm_key = random.split(key)
        self.conv_block: nnx.Conv = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=size,
            strides=2,
            padding="SAME",
            use_bias=not apply_groupnorm,
            rngs=nnx.Rngs(self.conv_key),
        )

        def _group_calculation(c: int) -> int:
            for g in (32, 16, 8, 4, 2, 1):
                if c % g == 0:
                    return g
            return 1  # fallback

        self.group_norm: nnx.GroupNorm = nnx.GroupNorm(
            num_features=out_features,
            num_groups=_group_calculation(out_features),
            rngs=nnx.Rngs(self.groupnorm_key),
        )
        self.apply_groupnorm: bool = apply_groupnorm

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.conv_block(x)
        if self.apply_groupnorm:
            x = self.group_norm(x)
        return nnx.leaky_relu(x, 0.2)


class upsample_block(nnx.Module):
    def __init__(
        self,
        key: Array,
        in_features: int,
        out_features: int,
        kernel_size: tuple[int, int],
        apply_groupnorm: bool = True,
    ):
        (
            self.conv_key,
            self.groupnorm_key,
        ) = random.split(key)

        self.conv_block: nnx.Conv = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            strides=1,
            padding="SAME",
            rngs=nnx.Rngs(self.conv_key),
        )

        def _group_calculation(c: int) -> int:
            for g in (32, 16, 8, 4, 2, 1):
                if c % g == 0:
                    return g
            return 1  # fallback

        self.group_norm: nnx.GroupNorm = nnx.GroupNorm(
            num_features=out_features,
            num_groups=_group_calculation(out_features),
            rngs=nnx.Rngs(self.groupnorm_key),
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
        self.gamma_key, self.beta_key = random.split(key)
        self.gamma = nnx.Linear(
            in_features=in_features, out_features=out_features, rngs=nnx.Rngs(self.gamma_key)
        )
        self.beta = nnx.Linear(
            in_features=in_features, out_features=out_features, rngs=nnx.Rngs(self.beta_key)
        )

    def __call__(self, x: jnp.ndarray, cosmos_params: jnp.ndarray) -> jnp.ndarray:
        gamma = self.gamma(cosmos_params)[:, None, None, :]
        beta = self.beta(cosmos_params)[:, None, None, :]
        return x * gamma + beta


class Generator(nnx.Module):
    def __init__(self, key: Array, in_features: int, out_features: int, len_cosmos_params: int):
        keys: jax.Array = random.split(key, 25)

        # fmt: off
        self.downsample_blocks: list[downsample_block] = [
            downsample_block(keys[0], in_features=in_features, out_features=64, size=(4, 4), apply_groupnorm=False),
            downsample_block(keys[1], in_features=64, out_features=128,  size=(4, 4)),
            downsample_block(keys[2], in_features=128, out_features=256, size=(4, 4)),
            downsample_block(keys[3], in_features=256, out_features=512, size=(4, 4)),
            downsample_block(keys[4], in_features=512, out_features=512, size=(4, 4)),
            downsample_block(keys[5], in_features=512, out_features=512, size=(4, 4)),
            downsample_block(keys[6], in_features=512, out_features=512, size=(4, 4)),
            downsample_block(keys[7], in_features=512, out_features=512, size=(4, 4)),
        ]

        self.upsample_blocks: list[upsample_block] = [
            upsample_block(keys[8], in_features=512, out_features=512, kernel_size=(4, 4)),
            upsample_block(keys[9], in_features=1024, out_features=512, kernel_size=(4, 4)),
            upsample_block(keys[10], in_features=1024, out_features=512, kernel_size=(4, 4)),
            upsample_block(keys[11], in_features=1024, out_features=512, kernel_size=(4, 4)),
            upsample_block(keys[12], in_features=1024, out_features=256, kernel_size=(4, 4)),
            upsample_block(keys[13], in_features=512, out_features=128, kernel_size=(4, 4)),
            upsample_block(keys[14], in_features=256, out_features=64, kernel_size=(4, 4)),
            upsample_block(keys[15], in_features=128, out_features=4, kernel_size=(4, 4), apply_groupnorm=False),
        ]
        # fmt: on
        self.skip_connection_conditioners: list[FiLM] = [
            FiLM(in_features=len_cosmos_params, out_features=64, key=keys[17]),
            FiLM(in_features=len_cosmos_params, out_features=128, key=keys[18]),
            FiLM(in_features=len_cosmos_params, out_features=256, key=keys[19]),
            FiLM(in_features=len_cosmos_params, out_features=512, key=keys[20]),
            FiLM(in_features=len_cosmos_params, out_features=512, key=keys[21]),
            FiLM(in_features=len_cosmos_params, out_features=512, key=keys[22]),
            FiLM(in_features=len_cosmos_params, out_features=512, key=keys[23]),
        ]

        self.output_stage: nnx.ConvTranspose = nnx.ConvTranspose(
            in_features=4,
            out_features=out_features,
            kernel_size=(4, 4),
            padding="SAME",
            rngs=nnx.Rngs(keys[16]),
        )

    def __call__(self, x: jnp.ndarray, cosmos_params: jnp.ndarray) -> jnp.ndarray:
        skip_cons: list[jnp.ndarray] = []

        for i in range(len(self.downsample_blocks) - 1):
            x = self.downsample_blocks[i](x)
            skip_cons.append(self.skip_connection_conditioners[i](x, cosmos_params))

        x = self.downsample_blocks[-1](x)

        # Upsample stage

        skip_cons = list(reversed(skip_cons))
        x = self.upsample_blocks[0](x)

        x = jnp.concatenate([skip_cons[0], x], axis=-1)

        for up, skip in zip(self.upsample_blocks[1:-1], skip_cons[1:]):
            x = up(x)  # upsample
            x = jnp.concatenate([skip, x], axis=-1)  # concat along channels (NHWC)

        x = self.upsample_blocks[-1](x)
        return self.output_stage(x)
