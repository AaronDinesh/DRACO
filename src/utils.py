import collections.abc
import shutil
from pathlib import Path
from typing import Callable, Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pandas as pd
from jax import Array, lax

from src.typing import Batch, TransformName

_STD_CHKPTR = ocp.StandardCheckpointer()


def make_train_test_loaders(
    key: Array,
    batch_size: int,
    input_data_path: str,
    output_data_path: str,
    csv_path: str,
    test_ratio: float = 0.2,
    transform_name: TransformName = "signed_log1p",
):
    input_maps = jnp.load(input_data_path, mmap_mode="r")
    output_maps = jnp.load(output_data_path, mmap_mode="r")
    cosmos_params = pd.read_csv(csv_path, header=None, sep=" ")

    repeat_factor: int = input_maps.shape[0] // len(cosmos_params)
    # fmt: off
    cosmos_params = cosmos_params.loc[cosmos_params.index.repeat(repeat_factor)].reset_index(drop=True)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    # fmt: on
    cosmos_params = jnp.asarray(cosmos_params.to_numpy())  # pyright: ignore[reportUnknownMemberType]

    assert len(input_maps) == len(cosmos_params), (
        f"The length of the input maps {input_maps.shape} do not match the number of cosmos params entries {cosmos_params.shape}"
    )
    assert len(input_maps) == len(output_maps), (
        "The number of input maps does not match the number of output maps"
    )

    dataset_len = len(input_maps)

    n_test = max(1, int(round(test_ratio * dataset_len)))

    random_shuffle = jax.random.permutation(key=key, x=dataset_len)

    test_idx = random_shuffle[:n_test]
    train_idx = random_shuffle[n_test:]

    # after train_idx/test_idx are defined
    mu = jnp.mean(cosmos_params[train_idx], axis=0)
    sigma = jnp.std(cosmos_params[train_idx], axis=0) + 1e-6

    forward_transform, _ = make_transform(name=transform_name)

    # This ensures that there is no data leakage
    assert len(jnp.intersect1d(train_idx, test_idx)) == 0

    def _standardize_params(x: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
        return (x - mu) / sigma

    def _add_channel_last(x: jnp.ndarray):
        # x is (N,H,W) or (N,H,W,C); return (N,H,W,1) or (N,H,W,C)
        return x[..., None] if x.ndim == 3 else x

    def _run_epoch(
        shuffled_idx: jnp.ndarray, key: Array | None, drop_last: bool = False
    ) -> collections.abc.Generator[Batch]:
        if key is not None:
            shuffled_idx = jax.random.permutation(key=key, x=shuffled_idx)

        n = len(shuffled_idx)
        stop = (n // batch_size) * batch_size if drop_last else n

        for s in range(0, stop, batch_size):
            batch = shuffled_idx[s : s + batch_size]

            yield {
                "inputs": forward_transform(_add_channel_last(input_maps[batch])),
                "targets": forward_transform(_add_channel_last(output_maps[batch])),
                "params": _standardize_params(cosmos_params[batch], mu=mu, sigma=sigma),
            }

    def train_loader(key: Array | None = None, drop_last: bool = False):
        return _run_epoch(train_idx, key, drop_last)

    def test_loader(key: Array | None = None, drop_last: bool = False):
        return _run_epoch(test_idx, key, drop_last)

    return (
        train_loader,
        test_loader,
        len(train_idx),
        len(test_idx),
        input_maps[0].shape[1],
        cosmos_params[0].shape[0],
        mu,
        sigma,
    )


def make_transform(
    name: TransformName,
    scale: float = 1.0,
):
    """
    Returns (forward, inverse) intensity transforms.
    For 'asinh_viz': 2D map -> pretty grayscale [0,1] image using:
      - |B| magnitude
      - percentile clip [0.5, 99.5]
      - light Gaussian blur (sigma≈1.2 px)
      - asinh with robust 'scale' (if scale<=0 or non-finite, auto=p95(|blurred|))
      - min-max normalize to [0,1]
    NOTE: 'inverse' for 'asinh_viz' cannot undo clip/blur/normalize; it returns input.
    """

    if name == "none":
        return (lambda x: x, lambda y: y)

    if name == "asinh":
        # y = asinh(x/s),  x = s*sinh(y)
        s = float(scale)

        def forward(x: jnp.ndarray):
            return jnp.arcsinh(x / s)

        def inverse(y: jnp.ndarray):
            return s * jnp.sinh(y)

        return forward, inverse

    if name == "signed_log1p":
        # y = sign(x) * log(1 + |x|/s),  x = sign(y)*s*(exp(|y|)-1)
        s = float(scale)

        def forward(x: jnp.ndarray):
            return jnp.sign(x) * jnp.log1p(jnp.abs(x) / s)

        def inverse(y: jnp.ndarray):
            return jnp.sign(y) * s * (jnp.expm1(jnp.abs(y)))

        return forward, inverse

    if name == "log10":
        ln10 = jnp.log(10.0)
        inv_ln10 = 1.0 / ln10

        def forward(x: jnp.ndarray):
            x_shape_dim = len(x.shape)

            if x_shape_dim == 2:
                reduce_axes = (0, 1)
            elif x_shape_dim == 3:
                reduce_axes = (1, 2)
            elif x_shape_dim == 4:
                reduce_axes = (1, 2)

            tiny = jnp.finfo(x.dtype).tiny
            g = jnp.log(jnp.maximum(x, tiny))  # (B,H,W,C)
            g_min = jnp.min(g, axis=reduce_axes, keepdims=True)  # (B,1,1,C) or (B,1,1,1)
            y = (g - g_min) * inv_ln10  # base-10, min->0
            return y

        def inverse(y: jnp.ndarray) -> jnp.ndarray:
            g = y * ln10
            return jnp.exp(g)

        return forward, inverse

    if name == "signed_log10":
        # y = sign(x) * log10(1 + |x|/s)
        # x = sign(y) * s * (10**|y| - 1)
        # Symmetric, zero-centered, reversible (no clipping/blur inside)
        s = float(scale)

        def forward(x: jnp.ndarray):
            ax = jnp.abs(x)
            return jnp.sign(x) * jnp.log10(1.0 + ax / s)

        def inverse(y: jnp.ndarray):
            ay = jnp.abs(y)
            return jnp.sign(y) * s * (jnp.power(10.0, ay) - 1.0)

        return forward, inverse

    if name == "asinh_viz":
        # Self-contained helpers scoped inside the block to minimize globals.
        def _percentile(x, q):
            x = jnp.sort(x.reshape(-1))
            idx = (q / 100.0) * (x.size - 1)
            lo = jnp.floor(idx).astype(int)
            hi = jnp.ceil(idx).astype(int)
            w = idx - lo
            return (1.0 - w) * x[lo] + w * x[hi]

        def _gaussian_blur2d(img, sigma=1.2):
            r = int(jnp.ceil(3.0 * sigma))
            xk = jnp.arange(-r, r + 1)
            k = jnp.exp(-0.5 * (xk / sigma) ** 2)
            k = k / jnp.sum(k)

            def _conv1d(a, k, axis):
                pad = [(0, 0)] * a.ndim
                pad[axis] = (r, r)
                a = jnp.pad(a, pad, mode="reflect")
                idx = jnp.arange(a.shape[axis] - 2 * r)
                out = 0.0
                for i, kv in enumerate(k):
                    out = out + kv * jnp.take(a, idx + i, axis=axis)
                return out

            out = _conv1d(img, k, axis=0)
            out = _conv1d(out, k, axis=1)
            return out

        s_user = float(scale)

        def forward(x2d: jnp.ndarray) -> jnp.ndarray:
            # 1) magnitude (grayscale look)
            mag = jnp.abs(x2d)

            # 2) robust clip to suppress outliers
            lo = _percentile(mag, 0.5)
            hi = _percentile(mag, 99.5)
            mag = jnp.clip(mag, lo, hi)

            # 3) gentle blur to reveal filaments
            mag = _gaussian_blur2d(mag, sigma=1.2)

            # 4) robust asinh scale
            if not jnp.isfinite(s_user) or s_user <= 0.0:
                s = _percentile(jnp.abs(mag), 95.0)
            else:
                s = s_user
            s = jnp.maximum(s, 1e-12)

            y = jnp.arcsinh(mag / s)

            # 5) normalize to [0,1] for display
            ymin = jnp.min(y)
            ymax = jnp.max(y)
            y = (y - ymin) / (ymax - ymin + 1e-12)
            return y

        def inverse(y_disp: jnp.ndarray) -> jnp.ndarray:
            # Not invertible (clip/blur/normalize lose information).
            # Return input as a no-op to keep API-compatible.
            return y_disp

        return forward, inverse

    raise ValueError(f"Unknown transform name: {name!r}")


def save_checkpoint(
    checkpoint_dir: str,
    epoch: int = 0,
    step: int = 0,
    model: nnx.Module | None = None,
    optimizer: nnx.Optimizer | None = None,  # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType],
    model_name: str | None = None,
    alt_name: str | None = None,
    data_stats=None,
) -> None:
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    _, model_state = nnx.split(model)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    _, opt_state = nnx.split(optimizer)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType, reportUnknownArgumentType]

    payload = {  # pyright: ignore[reportUnknownVariableType]
        "model_state": model_state,
        "opt_state": opt_state,
    }

    if data_stats is not None:
        payload["data_stats"] = data_stats  # <— NEW

    if alt_name is None:
        save_path = ckpt_dir / f"{model_name}_epoch_{epoch:07d}_step_{step:07d}"
    else:
        save_path = ckpt_dir / alt_name

    _STD_CHKPTR.save(str(save_path), payload)


def restore_checkpoint(checkpoint_path: str, model: nnx.Module, optimizer: nnx.Optimizer):  # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType]
    checkpoint = _STD_CHKPTR.restore(checkpoint_path)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    nnx.update(model, checkpoint["model_state"])  # pyright: ignore[reportUnknownMemberType]
    nnx.update(optimizer, checkpoint["opt_state"])  # pyright: ignore[reportUnknownMemberType]

    return checkpoint.get("data_stats", None)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]


def delete_checkpoint(checkpoint_path: str, folder_name: str) -> None:
    checkpoint_folder = Path(checkpoint_path, folder_name)
    shutil.rmtree(checkpoint_folder)


##### Stochastic Interpolant helpers


def gamma_and_deriv(
    t: jnp.ndarray, a: float = 1.0, eps: float = 1e-6
) -> tuple[jnp.ndarray, jnp.ndarray]:
    t = jnp.clip(t, eps, 1.0 - eps)
    num = 2.0 * a * t * (1.0 - t)
    gamma = jnp.sqrt(num)
    gamma_dot = a * (1.0 - 2.0 * t) / jnp.maximum(gamma, eps)
    return gamma, gamma_dot


def make_xt_and_targets(
    x0: jnp.ndarray, x1: jnp.ndarray, z: jnp.ndarray, time: jnp.ndarray, a: float = 1.0
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    interpolant = (1.0 - time)[:, None, None, None] * x0 + time[:, None, None, None] * x1
    gamma, gamma_dot = gamma_and_deriv(time, a=a)
    x_t = interpolant + gamma[:, None, None, None] * z
    dInterpolant = x1 - x0
    gdot_z = gamma_dot[:, None, None, None] * z
    return x_t, dInterpolant, gdot_z


def build_t_grid(
    n_steps: int, endpoint_clip: float = 1e-3, schedule: str = "cosine", power: float = 2.0
) -> jnp.ndarray:
    if schedule == "linear":
        t = jnp.linspace(0.0 + endpoint_clip, 1.0 - endpoint_clip, n_steps)
    elif schedule == "cosine":
        s = jnp.linspace(0.0, 1.0, n_steps)
        t = 0.5 - 0.5 * jnp.cos(jnp.pi * s)
        t = t * (1.0 - 2 * endpoint_clip) + endpoint_clip
    elif schedule == "power":
        s = jnp.linspace(0.0, 1.0, n_steps) ** power
        t = s * (1.0 - 2 * endpoint_clip) + endpoint_clip
    else:
        raise ValueError("Unknown t schedule")
    return t


def epsilon_schedule(t: jnp.ndarray, eps0: float = 0.1, taper: float = 0.6) -> jnp.ndarray:
    return eps0 * (t * (1.0 - t)) ** taper


def sde_sample_forward_conditional(
    model,
    x0: jnp.ndarray,
    cond_vec: jnp.ndarray,
    n_infer_steps: int = 250,
    a_gamma: float = 1.0,
    eps0: float = 0.1,
    eps_taper: float = 0.6,
    endpoint_clip: float = 1e-3,
    t_schedule: str = "cosine",
    t_power: float = 2.0,
    key: jax.Array | None = None,
) -> jnp.ndarray:
    key = key or jax.random.PRNGKey(0)
    B = x0.shape[0]
    X = x0
    t_grid = build_t_grid(n_infer_steps, endpoint_clip, t_schedule, t_power)
    dt = 1.0 / n_infer_steps

    # Makes use of Euler–Maruyama to integrate the SDE
    def euler_maruyama(i, state):
        X, key = state
        key, sub = jax.random.split(key)
        t_i = jnp.broadcast_to(t_grid[i], (B,))
        eps_i = epsilon_schedule(t_i, eps0, eps_taper)
        gamma_i, _ = gamma_and_deriv(t_i, a=a_gamma)
        b_hat, eta_hat = model(X, t_i, cond_vec)
        s_hat = -eta_hat / gamma_i[:, None, None, None]
        bF = b_hat + eps_i[:, None, None, None] * s_hat
        noise = jax.random.normal(sub, X.shape)
        X_next = X + bF * dt + jnp.sqrt(2.0 * eps_i)[:, None, None, None] * noise * jnp.sqrt(dt)
        return (X_next, key)

    X, _ = jax.lax.fori_loop(0, n_infer_steps, euler_maruyama, (X, key))
    return X


def sde_sample_forward_cfg(
    model,
    x0: jnp.ndarray,
    cond_vec: jnp.ndarray,
    guidance_scale: float = 1.5,
    n_infer_steps: int = 250,
    a_gamma: float = 1.0,
    eps0: float = 0.1,
    eps_taper: float = 0.6,
    endpoint_clip: float = 1e-3,
    t_schedule: str = "cosine",
    t_power: float = 2.0,
    key: jax.Array | None = None,
) -> jnp.ndarray:
    key = key or jax.random.PRNGKey(0)
    B = x0.shape[0]
    X = x0
    t_grid = build_t_grid(n_infer_steps, endpoint_clip, t_schedule, t_power)
    dt = 1.0 / n_infer_steps

    # Makes use of Euler–Maruyama to integrate the SDE
    def euler_maruyama(i, state):
        X, key = state
        key, sub = jax.random.split(key)
        t_i = jnp.broadcast_to(t_grid[i], (B,))
        eps_i = epsilon_schedule(t_i, eps0, eps_taper)
        gamma_i, _ = gamma_and_deriv(t_i, a=a_gamma)

        b_u, eta_u = model(X, t_i, jnp.zeros_like(cond_vec))
        b_c, eta_c = model(X, t_i, cond_vec)

        s = guidance_scale
        b_hat = b_u + s * (b_c - b_u)
        eta_hat = eta_u + s * (eta_c - eta_u)

        s_hat = -eta_hat / gamma_i[:, None, None, None]
        bF = b_hat + eps_i[:, None, None, None] * s_hat

        noise = jax.random.normal(sub, X.shape)
        X_next = X + bF * dt + jnp.sqrt(2.0 * eps_i)[:, None, None, None] * noise * jnp.sqrt(dt)
        return (X_next, key)

    X, _ = jax.lax.fori_loop(0, n_infer_steps, euler_maruyama, (X, key))
    return X


def random_crops(x: jnp.ndarray, crop: int, key: jax.Array) -> jnp.ndarray:
    """
    JIT-safe random crop.
    x: (B, H, W, C)
    crop: Python int (static)
    returns: (B, crop, crop, C)
    """
    B, H, W, C = x.shape

    # ensure sizes are static Python ints for dynamic_slice
    crop = int(crop)
    if crop > H or crop > W:
        raise ValueError(f"crop={crop} exceeds input size {(H, W)}")

    key_h, key_w = jax.random.split(key)
    # dynamic starts are fine
    hs = jax.random.randint(key_h, (B,), 0, H - crop + 1)
    ws = jax.random.randint(key_w, (B,), 0, W - crop + 1)

    def do_crop(xi, h0, w0):
        # dynamic starts; static sizes
        return lax.dynamic_slice(xi, (h0, w0, 0), (crop, crop, C))

    # vmap over batch
    return jax.vmap(do_crop, in_axes=(0, 0, 0))(x, hs, ws)


def maybe_hflip(x: jnp.ndarray, p: float, key: jax.Array) -> jnp.ndarray:
    flip = jax.random.bernoulli(key, p, (x.shape[0],))
    flipped = lax.rev(x, dimensions=(2,))  # reverse width axis
    return jnp.where(flip[:, None, None, None], flipped, x)


def wandb_image_panel(
    wandb, inputs: jnp.ndarray, targets: jnp.ndarray, preds: jnp.ndarray, max_items: int = 3
):
    def _to_uint8_linear(x: jnp.ndarray):
        lo = jnp.percentile(x, 1.0)
        hi = jnp.percentile(x, 99.0)
        x01 = jnp.clip((x - lo) / jnp.maximum(hi - lo, 1e-8), 0.0, 1.0)
        img = (x01 * 255.0).astype(jnp.uint8)
        img = jax.device_get(img)
        img = np.asarray(img)
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img[..., 0]
        return img

    imgs: list[np.ndarray] = []
    B = min(max_items, inputs.shape[0])
    for i in range(B):
        imgs.append(wandb.Image(_to_uint8_linear(inputs[i]), caption=f"in[{i}]"))
        imgs.append(wandb.Image(_to_uint8_linear(targets[i]), caption=f"tgt[{i}]"))
        imgs.append(wandb.Image(_to_uint8_linear(preds[i]), caption=f"gen[{i}]"))
    return imgs


def mae(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.abs(x - y))


def mse(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((x - y) ** 2)


def rmse(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(mse(x, y))


def psnr(x: jnp.ndarray, y: jnp.ndarray, data_range: float | None = None) -> jnp.ndarray:
    # If inputs already normalized, set data_range=1.0
    if data_range is None:
        mx = jnp.max(jnp.stack([x, y]))
        mn = jnp.min(jnp.stack([x, y]))
        data_range = jnp.maximum(mx - mn, 1e-8)
    err = mse(x, y)
    return 20.0 * jnp.log10(data_range) - 10.0 * jnp.log10(jnp.maximum(err, 1e-12))


def ssim_2d(
    x: jnp.ndarray, y: jnp.ndarray, C1: float = 0.01**2, C2: float = 0.03**2
) -> jnp.ndarray:
    # x,y: (H,W) or (H,W,1). Simple global-mean SSIM (no window) for speed.
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
        y = y[..., 0]
    mu_x = jnp.mean(x)
    mu_y = jnp.mean(y)
    sigma_x = jnp.var(x)
    sigma_y = jnp.var(y)
    sigma_xy = jnp.mean((x - mu_x) * (y - mu_y))
    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    return num / jnp.maximum(den, 1e-12)


def batch_metrics(pred: jnp.ndarray, tgt: jnp.ndarray) -> dict[str, jnp.ndarray]:
    # pred,tgt: (B,H,W,1)
    ps = jax.vmap(psnr)(pred, tgt)
    ss = jax.vmap(ssim_2d)(pred[..., 0], tgt[..., 0])
    me = jax.vmap(mse)(pred, tgt)
    ma = jax.vmap(mae)(pred, tgt)
    return {
        "psnr": jnp.mean(ps),
        "ssim": jnp.mean(ss),
        "mse": jnp.mean(me),
        "mae": jnp.mean(ma),
    }


def _fft_power_2d(x: jnp.ndarray) -> jnp.ndarray:
    """x: (H,W) or (H,W,1) real → return |FFT|^2 over full 2D (H,W)."""
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    # remove mean to avoid an overwhelming DC spike
    x = x - jnp.mean(x)
    F = jnp.fft.fftn(x, s=x.shape, norm="ortho")
    P = F.real**2 + F.imag**2
    return P  # (H,W)


def _radial_bins(h: int, w: int, nbins: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Precompute integer bin ids per (i,j) pixel for radial averaging."""
    # frequency coordinates on [-0.5, 0.5) after fftshift
    ky = jnp.fft.fftfreq(h)  # shape (H,)
    kx = jnp.fft.fftfreq(w)  # shape (W,)
    KX, KY = jnp.meshgrid(kx, ky, indexing="xy")
    r = jnp.sqrt(KX**2 + KY**2)  # radial frequency
    r = jnp.fft.fftshift(r)  # match power after fftshift

    # uniform bins from 0 to Nyquist (max radius)
    rmax = jnp.max(r)
    edges = jnp.linspace(0.0, rmax, nbins + 1)
    # digitize → bin ids in [0, nbins-1]
    ids = jnp.digitize(r.ravel(), edges) - 1
    ids = jnp.clip(ids, 0, nbins - 1)
    return ids.astype(jnp.int32), edges


def radial_power_spectrum(x: jnp.ndarray, nbins: int = 64) -> jnp.ndarray:
    """
    Compute isotropic 1D power spectrum by radial binning of |FFT|^2.
    x: (H,W,1) or (H,W)
    returns: (nbins,) spectrum (mean power per radial bin), normalized.
    """
    H, W = x.shape[:2]
    P = jnp.fft.fftshift(_fft_power_2d(x))  # (H,W)
    ids, _ = _radial_bins(H, W, nbins)  # (H*W,), static given H,W,nbins
    vals = P.ravel()
    # bin means using segment sums
    counts = jnp.bincount(ids, length=nbins)
    sums = jnp.bincount(ids, weights=vals, length=nbins)
    spec = sums / jnp.maximum(counts, 1)
    # stabilize dynamic range: log-spectrum and mean-normalize
    spec = jnp.log1p(spec)
    spec = spec / jnp.maximum(jnp.mean(spec), 1e-8)
    return spec  # (nbins,)


def batch_spectrum_loss(pred: jnp.ndarray, target: jnp.ndarray, nbins: int = 64) -> jnp.ndarray:
    """
    pred/target: (B,H,W,1) or (B,H,W,C) → average spectral MSE over channels.
    """
    B, H, W, C = pred.shape

    def spec_ch(p, t):
        # average channels’ spectra (or sum and normalize equivalently)
        def per_ch(pc, tc):
            sp = radial_power_spectrum(pc, nbins)
            st = radial_power_spectrum(tc, nbins)
            return jnp.mean((sp - st) ** 2)

        return jnp.mean(jax.vmap(per_ch)(jnp.moveaxis(p, -1, 0), jnp.moveaxis(t, -1, 0)))

    return jnp.mean(jax.vmap(spec_ch)(pred, target))  # scalar
