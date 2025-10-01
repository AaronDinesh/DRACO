import jax
import jax.numpy as jnp
import jax.random as random
from flax import nnx
from jax._src.typing import Array


class FiLM(nnx.Module):
    def __init__(self, theta_dim: int, out_features: int, key: Array):
        self.gamma_key, self.beta_key = random.split(key)  # pyright: ignore[reportAny, reportUnannotatedClassAttribute]
        self.gamma: nnx.Linear = nnx.Linear(theta_dim, out_features, rngs=nnx.Rngs(self.gamma_key))  # pyright: ignore[reportAny]
        self.beta: nnx.Linear = nnx.Linear(theta_dim, out_features, rngs=nnx.Rngs(self.beta_key))  # pyright: ignore[reportAny]

    def __call__(self, input: jnp.ndarray, condition_param: jnp.ndarray) -> jnp.ndarray:
        # This is because my tensor is of shape (B, H, W, C)
        gamma = self.gamma(condition_param)[:, None, None, :]
        beta = self.beta(condition_param)[:, None, None, :]
        return input * (1.0 + gamma) + beta


class downsample(nnx.Module):
    def __init__(
        self,
        key: Array,
        in_features: int,
        out_features: int,
        size: tuple[int, int],
        apply_BatchNorm: bool = True,
    ):
        self.conv_key, self.batch_norm_key = random.split(key)  # pyright: ignore[reportAny, reportUnannotatedClassAttribute]
        self.conv_block: nnx.Conv = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=size,
            strides=2,
            padding="SAME",
            use_bias=False,
            rngs=nnx.Rngs(self.conv_key),  # pyright: ignore[reportAny]
        )
        self.batch_norm: nnx.BatchNorm = nnx.BatchNorm(
            num_features=out_features,
            rngs=nnx.Rngs(self.batch_norm_key),  # pyright: ignore[reportAny]
        )
        self.apply_BatchNorm: bool = apply_BatchNorm

    def __call__(self, x: jnp.ndarray, is_training: bool):
        x = self.conv_block(x)
        if self.apply_BatchNorm:
            x = self.batch_norm(x, use_running_average=not is_training)
        return nnx.leaky_relu(x)


class upsample(nnx.Module):
    def __init__(
        self,
        key: Array,
        in_features: int,
        out_features: int,
        size: tuple[int, int],
        apply_Dropout: bool = True,
    ):
        self.conv_key, self.batch_norm_key, self.dropout_key = random.split(key, 3)  # pyright: ignore[reportAny, reportUnannotatedClassAttribute]
        self.conv_block: nnx.ConvTranspose = nnx.ConvTranspose(
            in_features=in_features,
            out_features=out_features,
            kernel_size=size,
            strides=2,
            padding="SAME",
            use_bias=False,
            rngs=nnx.Rngs(self.conv_key),  # pyright: ignore[reportAny]
        )
        self.batch_norm: nnx.BatchNorm = nnx.BatchNorm(
            num_features=out_features,
            rngs=nnx.Rngs(self.batch_norm_key),  # pyright: ignore[reportAny]
        )

        self.dropout: nnx.Dropout = nnx.Dropout(rate=0.5, rngs=nnx.Rngs(self.dropout_key))  # pyright: ignore[reportAny]

        self.apply_Dropout: bool = apply_Dropout

    def __call__(self, x: jnp.ndarray, is_training: bool = False) -> jnp.ndarray:
        x = self.conv_block(x)
        x = self.batch_norm(x, use_running_average=not is_training)
        if self.apply_Dropout and is_training:
            x = self.dropout(x)
        return nnx.relu(x)


class Generator(nnx.Module):
    key: Array
    in_features: int
    out_features: int
    len_condition_params: int

    def __init__(
        self, key: Array, in_features: int, out_features: int, len_condition_params: int
    ) -> None:
        keys: jax.Array = random.split(key, 25)

        # fmt: off
        self.downsample_blocks: list[downsample] = [
            downsample(keys[0], in_features=in_features, out_features=64, size=(4, 4), apply_BatchNorm=False),
            downsample(keys[1], in_features=64, out_features=128, size=(4, 4)),
            downsample(keys[2], in_features=128, out_features=256, size=(4, 4)),
            downsample(keys[3], in_features=256, out_features=512, size=(4, 4)),
            downsample(keys[4], in_features=512, out_features=512, size=(4, 4)),
            downsample(keys[5], in_features=512, out_features=512, size=(4, 4)),
            downsample(keys[6], in_features=512, out_features=512, size=(4, 4)),
            downsample(keys[7], in_features=512, out_features=512, size=(4, 4)),
        ]

        self.upsample_blocks: list[upsample] = [
            upsample(keys[8], in_features=512, out_features=512, size=(4, 4)),
            upsample(keys[9], in_features=1024, out_features=512, size=(4, 4)),
            upsample(keys[10], in_features=1024, out_features=512, size=(4, 4)),
            upsample(keys[11], in_features=1024, out_features=512, size=(4, 4), apply_Dropout=False),
            upsample(keys[12], in_features=1024, out_features=256, size=(4, 4), apply_Dropout=False),
            upsample(keys[13], in_features=512, out_features=128, size=(4, 4), apply_Dropout=False),
            upsample(keys[14], in_features=256, out_features=64, size=(4, 4), apply_Dropout=False),
            upsample(keys[15], in_features=128, out_features=4, size=(4, 4), apply_Dropout=False),
        ]
        # fmt: on

        self.skip_connection_conditioners: list[FiLM] = [
            FiLM(theta_dim=len_condition_params, out_features=64, key=keys[17]),
            FiLM(theta_dim=len_condition_params, out_features=128, key=keys[18]),
            FiLM(theta_dim=len_condition_params, out_features=256, key=keys[19]),
            FiLM(theta_dim=len_condition_params, out_features=512, key=keys[20]),
            FiLM(theta_dim=len_condition_params, out_features=512, key=keys[21]),
            FiLM(theta_dim=len_condition_params, out_features=512, key=keys[22]),
            FiLM(theta_dim=len_condition_params, out_features=512, key=keys[23]),
        ]

        self.output_stage: nnx.ConvTranspose = nnx.ConvTranspose(
            in_features=4,
            out_features=out_features,
            kernel_size=(4, 4),
            padding="same",
            rngs=nnx.Rngs(keys[16]),
        )

    def __call__(
        self, x: jnp.ndarray, condition_params: jnp.ndarray, is_training: bool = False
    ) -> jnp.ndarray:
        # Downsample stage
        skip_cons: list[jnp.ndarray] = []
        for i in range(len(self.downsample_blocks) - 1):
            x = self.downsample_blocks[i](x, is_training)
            skip_cons.append(self.skip_connection_conditioners[i](x, condition_params))

        x = self.downsample_blocks[-1](x, is_training)

        # Upsample stage
        x = self.upsample_blocks[0](x, is_training)
        skip_cons = list(reversed(skip_cons))

        x = jnp.concatenate([skip_cons[0], x], axis=-1)

        for up, skip in zip(self.upsample_blocks[1:-1], skip_cons[1:]):
            x = up(x, is_training)  # upsample
            x = jnp.concatenate([skip, x], axis=-1)  # concat along channels (NHWC)

        x = self.upsample_blocks[-1](x, is_training)
        return self.output_stage(x)
