import jax.numpy as jnp
import jax.random as random
from flax import nnx
from jax.typing import ArrayLike


class SpectralNorm(nnx.Module):
    """Applies spectral normalization to a module's kernel."""

    def __init__(self, module: nnx.Module, *, rngs: nnx.Rngs):
        self.module = module

        if not hasattr(self.module, "kernel"):
            raise ValueError("Wrapped module must have a 'kernel' attribute.")

        kernel_shape = self.module.kernel.shape
        # Reshape kernel to a 2D matrix for spectral norm calculation
        # For Conv: (H, W, C_in, C_out) -> (C_out, H * W * C_in)
        # For Linear: (C_in, C_out) -> (C_out, C_in)
        if len(kernel_shape) == 4:  # Conv
            self.c_out = kernel_shape[3]
            self.get_w_reshaped = lambda w: w.reshape(-1, self.c_out).T
        elif len(kernel_shape) == 2:  # Linear
            self.c_out = kernel_shape[1]
            self.get_w_reshaped = lambda w: w.T
        else:
            raise ValueError(f"Unsupported kernel shape: {kernel_shape}")

        # Initialize the power iteration vector 'u'
        key = rngs.params()
        u_shape = (1, self.c_out)
        u_init = random.normal(key, u_shape)
        u_init = u_init / jnp.linalg.norm(u_init)
        self.u = nnx.Variable(u_init, collection="spectral_stats")

    def __call__(self, x: jnp.ndarray, *, update_stats: bool = False) -> jnp.ndarray:
        w_orig = self.module.kernel.value
        w_reshaped = self.get_w_reshaped(w_orig)
        u = self.u.value

        if update_stats:
            # Power iteration to update u
            v = w_reshaped.T @ u.T
            v = v / jnp.linalg.norm(v)
            u_new = w_reshaped @ v
            u_new = u_new / jnp.linalg.norm(u_new)
            self.u.value = u_new.T
            u = u_new.T  # Use updated u for sigma calculation

        # Estimate sigma using the current u
        v = w_reshaped.T @ u.T
        v = v / jnp.linalg.norm(v)
        u_new = w_reshaped @ v
        sigma = u @ u_new

        # Normalize the kernel
        w_sn = w_orig / sigma

        # Temporarily replace the kernel for the forward pass
        self.module.kernel.value = w_sn
        out = self.module(x)
        # Restore the original kernel so the optimizer updates the unnormalized weights
        self.module.kernel.value = w_orig

        return out


class conv_block(nnx.Module):
    def __init__(
        self,
        key: ArrayLike,
        in_features: int,
        out_features: int,
        size: tuple[int, int] = (3, 3),
        stride: int = 1,
        apply_BatchNorm: bool = False,
        apply_spectral_norm: bool = True,
        activation: str = "leaky_relu",
    ):
        k1, k2, k3 = random.split(key, 3)  # type: ignore
        self.apply_spectral_norm = apply_spectral_norm

        conv_layer = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=size,
            strides=stride,
            padding="same",
            use_bias=not apply_BatchNorm,
            rngs=nnx.Rngs(k1),
        )
        if self.apply_spectral_norm:
            self.conv = SpectralNorm(conv_layer, rngs=nnx.Rngs(params=k3))
        else:
            self.conv = conv_layer

        self.apply_BatchNorm: bool = apply_BatchNorm
        if apply_BatchNorm:
            self.batch_norm: nnx.BatchNorm = nnx.BatchNorm(
                num_features=out_features, rngs=nnx.Rngs(k2)
            )
        self.activation: str = activation

    def __call__(self, x: jnp.ndarray, is_training: bool) -> jnp.ndarray:
        if self.apply_spectral_norm:
            x = self.conv(x, update_stats=is_training)
        else:
            x = self.conv(x)

        if self.apply_BatchNorm:
            x = self.batch_norm(x, use_running_average=not is_training)

        if self.activation == "leaky_relu":
            return nnx.leaky_relu(x)
        elif self.activation == "relu":
            return nnx.relu(x)
        return x


class film(nnx.Module):
    """FiLM: condition_params -> γ, β; returns x * (1+γ) + β"""

    def __init__(self, key: ArrayLike, theta_dim: int, out_features: int):
        k1, k2 = random.split(key)  # type: ignore
        self.gamma: nnx.Linear = nnx.Linear(theta_dim, out_features, rngs=nnx.Rngs(k1))
        self.beta: nnx.Linear = nnx.Linear(theta_dim, out_features, rngs=nnx.Rngs(k2))

    def __call__(self, x: jnp.ndarray, theta: jnp.ndarray):
        gamma = self.gamma(theta)[:, None, None, :]  # [B,1,1,C]
        beta = self.beta(theta)[:, None, None, :]  # [B,1,1,C]
        return x * (1.0 + gamma) + beta


class sle_block(nnx.Module):
    """Skip-Layer Excitation: low-res gates high-res (channel-wise)."""

    def __init__(self, key: ArrayLike, low_features: int, high_features: int, hidden: int = 128):
        k1, k2 = random.split(key)  # type: ignore
        # Use global avg pool on low-res, then 1x1 convs to produce per-channel gate for high
        self.conv_block1: nnx.Conv = nnx.Conv(
            in_features=low_features,
            out_features=hidden,
            kernel_size=(1, 1),
            strides=1,
            padding="valid",
            rngs=nnx.Rngs(k1),
        )
        self.conv_block2: nnx.Conv = nnx.Conv(
            in_features=hidden,
            out_features=high_features,
            kernel_size=(1, 1),
            strides=1,
            padding="valid",
            rngs=nnx.Rngs(k2),
        )

    def __call__(self, low: jnp.ndarray, high: jnp.ndarray) -> jnp.ndarray:
        # Global avg pool over H,W on 'low'
        g = jnp.mean(low, axis=(1, 2), keepdims=True)  # [B,1,1,Clow]
        g = nnx.relu(self.conv_block1(g))  # [B,1,1,Hid]
        g = self.conv_block2(g)  # [B,1,1,Chigh]
        return high * g  # broadcast over H,W


class Discriminator(nnx.Module):
    """Returns raw logits (not sigmoid). Use BCE-with-logits in training."""

    def __init__(self, key: ArrayLike, in_features: int, len_condition_params: int):
        keys = random.split(key, 18)  # type: ignore

        # 256-branch
        self.d256 = conv_block(keys[0], in_features, 64, stride=2)  # 256 -> 128
        self.film128 = film(keys[1], len_condition_params, 64)

        self.d128 = conv_block(keys[2], 64, 128, stride=2)  # 128 -> 64
        self.film64a = film(keys[3], len_condition_params, 128)

        # 128-branch
        self.in128 = conv_block(keys[4], in_features, 64, stride=2)  # 128 -> 64
        self.film64b = film(keys[5], len_condition_params, 64)

        # Merge at 64×64: concat [h64a (128ch), h64b (64ch)] -> 192ch -> 256ch
        self.merge64 = conv_block(keys[6], 128 + 64, 256, stride=1)
        self.film64 = film(keys[7], len_condition_params, 256)

        # Deeper downs: 64 -> 32 -> 16 -> 8
        self.d32 = conv_block(keys[8], 256, 512, stride=2)  # 64 -> 32
        self.film32 = film(keys[9], len_condition_params, 512)

        self.d16 = conv_block(keys[10], 512, 512, stride=2)  # 32 -> 16
        self.film16 = film(keys[11], len_condition_params, 512)

        self.d8 = conv_block(keys[12], 512, 512, stride=2)  # 16 -> 8
        self.film8 = film(keys[13], len_condition_params, 512)

        # SLE gates: low (16->) gate high (128), and low (8->) gate high (64)
        self.sle16_128 = sle_block(keys[14], low_features=512, high_features=64)
        self.sle8_64 = sle_block(keys[15], low_features=512, high_features=256)

        # Heads
        self.head_fc = nnx.Linear(512, 1, rngs=nnx.Rngs(keys[16]))
        self.patch_head = nnx.Conv(
            in_features=512,
            out_features=1,
            kernel_size=(1, 1),
            strides=1,
            padding="same",
            rngs=nnx.Rngs(keys[17]),
        )

    def __call__(
        self,
        x256: jnp.ndarray,
        x128: jnp.ndarray,
        condition_params: jnp.ndarray,
        is_training: bool = False,
    ) -> dict[str, jnp.ndarray]:
        # 256 branch
        h128 = self.d256(x256, is_training)  # [B,128,128,64]
        h128 = self.film128(h128, condition_params)

        h64a = self.d128(h128, is_training)  # [B,64,64,128]
        h64a = self.film64a(h64a, condition_params)

        # 128 branch
        h64b = self.in128(x128, is_training)  # [B,64,64,64]
        h64b = self.film64b(h64b, condition_params)

        # Merge at 64
        h64 = jnp.concatenate([h64a, h64b], axis=-1)  # [B,64,64,192]
        h64 = self.merge64(h64, is_training)  # [B,64,64,256]
        h64 = self.film64(h64, condition_params)

        # Down to 32, 16, 8 with FiLM each time
        h32 = self.d32(h64, is_training)  # [B,32,32,512]
        h32 = self.film32(h32, condition_params)

        h16 = self.d16(h32, is_training)  # [B,16,16,512]
        h16 = self.film16(h16, condition_params)

        h8 = self.d8(h16, is_training)  # [B,8,8,512]
        h8 = self.film8(h8, condition_params)

        # SLE: 16->128 and 8->64 (optionally use these gated features for aux losses)
        h128_gated = self.sle16_128(h16, h128)  # [B,128,128,64]
        h64_gated = self.sle8_64(h8, h64)  # [B,64,64,256]

        # Heads
        pooled = jnp.mean(h8, axis=(1, 2))  # global average pool -> [B,512]
        logits = self.head_fc(pooled).squeeze(-1)  # [B]
        patch = self.patch_head(h8)  # [B,8,8,1]

        return {
            # IMPORTANT: return raw logits; apply sigmoid in loss/metrics if needed
            "logits": logits,
            "patch": patch,
            "h128_gated": h128_gated,
            "h64_gated": h64_gated,
        }
