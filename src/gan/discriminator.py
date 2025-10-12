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
    """Spectral Normalization wrapper for NNX modules.

    Applies spectral normalization to the weight matrix of wrapped modules
    (e.g., Linear, Conv) by dividing weights by their largest singular value,
    approximated via power iteration.

    Args:
        module: The NNX module to wrap (e.g., nnx.Linear, nnx.Conv)
        n_power_iterations: Number of power iterations for approximating
            the spectral norm (default: 1)
        eps: Small constant for numerical stability (default: 1e-12)

    Example:
        >>> # Wrap a Linear layer
        >>> linear = nnx.Linear(128, 64, rngs=nnx.Rngs(0))
        >>> sn_linear = SpectralNorm(linear, rngs=nnx.Rngs(1))
        >>>
        >>> # Wrap a Conv layer
        >>> conv = nnx.Conv(3, 64, kernel_size=(3, 3), rngs=nnx.Rngs(0))
        >>> sn_conv = SpectralNorm(conv, rngs=nnx.Rngs(2), n_power_iterations=3)
    """

    def __init__(
        self,
        module: nnx.Module,
        *,
        rngs: nnx.Rngs,
        n_power_iterations: int = 1,
        eps: float = 1e-12,
    ):
        self.module = module
        self.n_power_iterations = n_power_iterations
        self.eps = eps

        # Get the weight parameter from the module
        if hasattr(module, "kernel"):
            weight = module.kernel.value
        elif hasattr(module, "weight"):
            weight = module.weight.value
        else:
            raise ValueError("Module must have either 'kernel' or 'weight' attribute")

        # Reshape weight to 2D for spectral norm computation
        self.weight_shape = weight.shape
        weight_mat = self._reshape_weight(weight)

        # Initialize u vector for power iteration
        u_shape = (weight_mat.shape[0],)
        u_init = jax.random.normal(rngs(), u_shape)
        u_init = u_init / jnp.linalg.norm(u_init)

        self.u = nnx.Variable(u_init)

    def _reshape_weight(self, weight: jnp.ndarray) -> jnp.ndarray:
        """Reshape weight tensor to 2D matrix."""
        if weight.ndim == 1:
            return weight.reshape(1, -1)
        else:
            return weight.reshape(weight.shape[0], -1)

    def _compute_spectral_norm(
        self, weight: jnp.ndarray, update_u: bool = True
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute spectral norm using power iteration.

        Args:
            weight: Weight matrix (2D)
            update_u: Whether to update the u vector

        Returns:
            Tuple of (sigma, u) where sigma is the spectral norm
        """
        weight_mat = self._reshape_weight(weight)
        u = self.u.value

        # Power iteration
        for _ in range(self.n_power_iterations):
            # v = W^T u / ||W^T u||
            v = jnp.matmul(weight_mat.T, u)
            v = v / (jnp.linalg.norm(v) + self.eps)

            # u = W v / ||W v||
            u = jnp.matmul(weight_mat, v)
            u = u / (jnp.linalg.norm(u) + self.eps)

        # Compute spectral norm: sigma = u^T W v
        sigma = jnp.dot(u, jnp.matmul(weight_mat, v))

        if update_u:
            self.u.value = u

        return sigma, u

    def __call__(self, *args, is_training: bool = True, **kwargs):
        """Forward pass with spectral normalization applied.

        Args:
            training: If True, compute and update spectral norm. If False,
                use cached spectral norm from training.
        """
        # Get the weight parameter
        if hasattr(self.module, "kernel"):
            weight_param = self.module.kernel
        else:
            weight_param = self.module.weight

        original_weight = weight_param.value

        # During training: compute spectral norm and update u
        # During inference: use cached u vector without updating
        if is_training:
            sigma, _ = self._compute_spectral_norm(
                original_weight, update_u=not nnx.is_initializing()
            )
        else:
            # Use cached u vector for inference (no updates)
            sigma, _ = self._compute_spectral_norm(original_weight, update_u=False)

        normalized_weight = original_weight / (sigma + self.eps)

        # Temporarily replace the weight
        weight_param.value = normalized_weight

        try:
            # Forward pass through wrapped module
            output = self.module(*args, **kwargs)
        finally:
            # Restore original weight
            weight_param.value = original_weight

        return output


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
            module=nnx.Conv(
                in_features=2 * in_channels + condition_proj_dim,
                out_features=base_conv_dim,
                strides=2,
                kernel_size=(4, 4),
                padding="SAME",
                rngs=nnx.Rngs(conv1_key),
            ),
            n_power_iterations=1,
            rngs=nnx.Rngs(conv1_key),
        )

        self.conv2: SpectralNorm = SpectralNorm(
            module=nnx.Conv(
                in_features=base_conv_dim,
                out_features=base_conv_dim * 2,
                strides=2,
                kernel_size=(4, 4),
                padding="SAME",
                rngs=nnx.Rngs(conv2_key),
            ),
            n_power_iterations=1,
            rngs=nnx.Rngs(conv2_key),
        )

        self.conv3: SpectralNorm = SpectralNorm(
            module=nnx.Conv(
                in_features=base_conv_dim * 2,
                out_features=base_conv_dim * 4,
                strides=2,
                kernel_size=(4, 4),
                padding="SAME",
                rngs=nnx.Rngs(conv3_key),
            ),
            n_power_iterations=1,
            rngs=nnx.Rngs(conv3_key),
        )

        self.conv4: SpectralNorm = SpectralNorm(
            module=nnx.Conv(
                in_features=base_conv_dim * 4,
                out_features=1,
                strides=1,
                kernel_size=(4, 4),
                padding="SAME",
                rngs=nnx.Rngs(conv4_key),
            ),
            n_power_iterations=1,
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
        condition_params_proj = self.condition_projection(condition_params)[
            :, None, None, :
        ]

        condition_params_proj = jnp.broadcast_to(
            condition_params_proj, (batch, height, width, self.condition_proj_dim)
        )

        out = jnp.concatenate([inputs, output, condition_params_proj], axis=-1)

        out = nnx.leaky_relu(self.conv1(x=out, training=is_training), 0.2)
        out = nnx.leaky_relu(self.conv2(x=out, training=is_training), 0.2)
        out = nnx.leaky_relu(self.conv3(x=out, training=is_training), 0.2)
        logits = self.conv4(out, training=is_training)

        return logits
