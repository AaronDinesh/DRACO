import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Tuple

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import optax
import pandas as pd
from dotenv import load_dotenv
from flax import nnx
from PIL import Image
from tqdm import tqdm

from src.interpolants import LinearInterpolant, SDEIntegrator, StochasticInterpolantUNet, make_gamma
from src.utils import make_transform, power_spectrum, restore_checkpoint


def _add_channel_last(x: jnp.ndarray) -> jnp.ndarray:
    return x[..., None] if x.ndim == 2 else x


def _power_spectrum_curve(field: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    mesh = jnp.asarray(field)
    if mesh.ndim == 3 and mesh.shape[-1] == 1:
        mesh = mesh[..., 0]
    k_vals, pk = power_spectrum(mesh, kedges=bins)
    return np.asarray(jax.device_get(k_vals)), np.asarray(jax.device_get(pk))


def _spectrum_mse(pred: np.ndarray, target: np.ndarray) -> float:
    min_len = min(len(pred), len(target))
    if min_len == 0:
        return 0.0
    diff = pred[:min_len] - target[:min_len]
    return float(np.mean(np.square(diff)))


def _load_input(
    input_maps: np.ndarray,
    cosmos_params: jnp.ndarray,
    sample_idx: int,
    transform_name: str,
    cosmos_mu: jnp.ndarray,
    cosmos_sigma: jnp.ndarray,
    target_maps_path: str | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, np.ndarray | None]:
    if sample_idx < 0 or sample_idx >= len(input_maps):
        raise ValueError(
            f"sample_idx {sample_idx} out of range for dataset of size {len(input_maps)}"
        )

    forward_transform, _ = make_transform(name=transform_name)

    x0 = jnp.asarray(
        forward_transform(_add_channel_last(jnp.asarray(input_maps[sample_idx]))), dtype=jnp.float32
    )
    cosmos = jnp.asarray(
        (cosmos_params[sample_idx] - cosmos_mu) / cosmos_sigma,
        dtype=jnp.float32,
    )

    target: np.ndarray | None = None
    if target_maps_path:
        target_maps = np.load(target_maps_path, mmap_mode="r")
        if sample_idx < 0 or sample_idx >= len(target_maps):
            raise ValueError(
                f"sample_idx {sample_idx} out of range for target dataset of size {len(target_maps)}"
            )
        target = np.asarray(
            forward_transform(_add_channel_last(np.asarray(target_maps[sample_idx]))),
            dtype=np.float32,
        )

    return x0, cosmos, target


def _build_models(
    key: jax.Array,
    img_channels: int,
    cosmos_params_len: int,
    time_embed_dim: int,
    velocity_checkpoint: str,
    score_checkpoint: str,
) -> Tuple[StochasticInterpolantUNet, StochasticInterpolantUNet, dict[str, Any], dict[str, Any]]:
    vel_key, score_key = random.split(key)
    vel_model = StochasticInterpolantUNet(
        key=vel_key,
        in_features=img_channels,
        out_features=img_channels,
        len_cosmos_params=cosmos_params_len,
        time_embed_dim=time_embed_dim,
    )
    score_model = StochasticInterpolantUNet(
        key=score_key,
        in_features=img_channels,
        out_features=img_channels,
        len_cosmos_params=cosmos_params_len,
        time_embed_dim=time_embed_dim,
    )

    # Optimizers mirror training definitions so checkpoint restore matches shapes.
    vel_opt = nnx.Optimizer(
        vel_model,
        optax.adam(2e-4, b1=0.9, b2=0.999),
        wrt=nnx.Param,
    )
    score_opt = nnx.Optimizer(
        score_model,
        optax.adam(2e-4, b1=0.9, b2=0.999),
        wrt=nnx.Param,
    )

    vel_meta = restore_checkpoint(velocity_checkpoint, vel_model, vel_opt)
    score_meta = restore_checkpoint(score_checkpoint, score_model, score_opt)

    return vel_model, score_model, vel_meta, score_meta


def _create_integrator(
    interpolant: LinearInterpolant,
    eps: float,
    t_grid: jnp.ndarray,
    n_save: int,
    n_likelihood: int,
    vel_model: StochasticInterpolantUNet,
    score_model: StochasticInterpolantUNet,
    cosmos: jnp.ndarray,
) -> SDEIntegrator:
    def _prepare_t(t: jnp.ndarray) -> jnp.ndarray:
        t = jnp.reshape(t, (t.shape[0],))
        return t

    def b_fn(x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        return vel_model(x, cosmos, _prepare_t(t))

    def s_fn(x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        return score_model(x, cosmos, _prepare_t(t))

    return SDEIntegrator(
        b=b_fn,
        s=s_fn,
        eps=eps,
        interpolant=interpolant,
        t_grid=t_grid,
        n_save=n_save,
        n_step=t_grid.shape[0] - 1,
        n_likelihood=n_likelihood,
    )


def generate_samples(
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, Iterable[np.ndarray]]:
    _ = load_dotenv()

    master_key = random.key(args.seed)
    model_key, sample_key, _aux_key, train_test_key = random.split(master_key, 4)

    input_maps = np.load(args.input_maps, mmap_mode="r")
    cosmos_params_df = pd.read_csv(args.cosmos_params, header=None, sep=" ")
    repeat_factor: int = input_maps.shape[0] // len(cosmos_params_df)
    cosmos_params_df = cosmos_params_df.loc[
        cosmos_params_df.index.repeat(repeat_factor)
    ].reset_index(drop=True)
    cosmos_params = jnp.asarray(cosmos_params_df.to_numpy(), dtype=jnp.float32)
    if len(input_maps) != len(cosmos_params):
        raise ValueError(
            f"Input maps ({len(input_maps)}) and cosmos params ({len(cosmos_params)}) differ in length"
        )
    dataset_len = len(input_maps)

    # Deterministic train/test split matching evaluation: use the 4th split key from the seed.
    perm = np.asarray(random.permutation(train_test_key, dataset_len))
    n_test = max(1, int(round(args.test_ratio * dataset_len)))
    test_indices = perm[:n_test]
    train_indices = perm[n_test:]

    if args.split not in {"train", "test", "all"}:
        raise ValueError("split must be one of: train, test, all")
    if args.split == "train":
        subset_indices = train_indices
    elif args.split == "test":
        subset_indices = test_indices
    else:
        subset_indices = perm

    if len(subset_indices) == 0:
        raise ValueError(f"No samples available for split '{args.split}'")
    if args.sample_idx < 0 or args.sample_idx >= len(subset_indices):
        raise ValueError(
            f"sample_idx {args.sample_idx} out of range for split '{args.split}' of size {len(subset_indices)}"
        )
    selected_idx = int(subset_indices[args.sample_idx])

    # Use train-split statistics to mirror training normalization (unless checkpoint overrides).
    stats_indices = train_indices if len(train_indices) > 0 else np.arange(dataset_len)
    cosmos_mu = jnp.mean(cosmos_params[stats_indices], axis=0)
    cosmos_sigma = jnp.std(cosmos_params[stats_indices], axis=0) + 1e-6

    vel_model, score_model, vel_meta, score_meta = _build_models(
        key=model_key,
        img_channels=args.img_channels,
        cosmos_params_len=cosmos_params.shape[-1],
        time_embed_dim=args.time_embed_dim,
        velocity_checkpoint=args.velocity_checkpoint_path,
        score_checkpoint=args.score_checkpoint_path,
    )

    stored_data_stats: dict[str, jnp.ndarray] | None = None
    for meta in (vel_meta, score_meta):
        if meta.get("data_stats") is not None:
            stored_data_stats = meta["data_stats"]
            break
    if stored_data_stats is not None:
        mu_override = stored_data_stats.get("cosmos_params_mu")
        sigma_override = stored_data_stats.get("cosmos_params_sigma")
        if mu_override is not None and sigma_override is not None:
            cosmos_mu = jnp.asarray(mu_override)
            cosmos_sigma = jnp.asarray(sigma_override)

    x0, cosmos, target = _load_input(
        input_maps=input_maps,
        cosmos_params=cosmos_params,
        sample_idx=selected_idx,
        transform_name=args.transform_name,
        cosmos_mu=cosmos_mu,
        cosmos_sigma=cosmos_sigma,
        target_maps_path=args.target_maps,
    )
    cosmos = cosmos[None, ...]  # add batch axis
    x0 = x0[None, ...]  # add batch axis

    t_grid = jnp.linspace(args.t_min, args.t_max, args.integrator_steps + 1)
    interpolant = LinearInterpolant(
        alpha=lambda t: 1.0 - t,
        beta=lambda t: t,
        t_schedule="linear",
        gamma_type=args.gamma_type,
    )
    gamma_fn, gamma_dot_fn, gg_dot_fn = make_gamma(gamma_type=args.gamma_type, a=args.gamma_a)
    interpolant.gamma = gamma_fn
    interpolant.gamma_dot = gamma_dot_fn
    interpolant.gg_dot = gg_dot_fn

    integrator = _create_integrator(
        interpolant=interpolant,
        eps=args.eps,
        t_grid=t_grid,
        n_save=args.n_save,
        n_likelihood=args.n_likelihood,
        vel_model=vel_model,
        score_model=score_model,
        cosmos=cosmos,
    )

    # Run stochastic rollouts
    rollout = jax.jit(lambda x, key: integrator.forward_rollout(x, key))

    def _pred_iter():
        nonlocal sample_key
        for _ in tqdm(range(args.n_samples), desc="Generating samples", unit="sample"):
            sample_key, sub = random.split(sample_key)
            preds = rollout(x0, sub)
            yield np.asarray(preds[0])

    return target, _pred_iter()


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    lo = np.percentile(arr, 1.0)
    hi = np.percentile(arr, 99.0)
    if np.isclose(hi, lo):
        hi = lo + 1e-6
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    arr_uint8 = (arr * 255.0).astype(np.uint8)
    if arr_uint8.ndim == 3 and arr_uint8.shape[-1] == 1:
        arr_uint8 = arr_uint8[..., 0]
    return arr_uint8


def save_outputs(
    outputs: Iterable[np.ndarray], output_dir: Path, target: np.ndarray | None = None
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if target is not None:
        target_np = np.asarray(target)
        target_uint8 = _to_uint8(target_np)
        Image.fromarray(target_uint8).save(output_dir / "target.png")
        np.save(raw_dir / "target.npy", target_np)

    for idx, arr in enumerate(outputs):
        arr_np = np.asarray(arr)
        arr_uint8 = _to_uint8(arr_np)
        img = Image.fromarray(arr_uint8)
        out_path = output_dir / f"sample_{idx:03d}.png"
        img.save(out_path)
        np.save(raw_dir / f"sample_{idx:03d}.npy", arr_np)


def main():
    parser = argparse.ArgumentParser("SI sampler: generate N outputs from one input")
    # fmt: off
    parser.add_argument("--input-maps", required=True, help="Path to input .npy array (N,H,W[,C])")
    parser.add_argument("--cosmos-params", required=True, help="Path to cosmos params txt/csv")
    parser.add_argument("--sample-idx", type=int, default=0, help="Index of the input to generate for")
    parser.add_argument("--target-maps", help="Optional path to target/output .npy array for visualization")
    parser.add_argument("--velocity-checkpoint-path", required=True, help="Velocity model checkpoint")
    parser.add_argument("--score-checkpoint-path", required=True, help="Score model checkpoint")
    parser.add_argument("--output-dir", required=True, help="Directory to write generated samples")
    parser.add_argument("--n-samples", type=int, default=15, help="Number of stochastic generations")
    parser.add_argument("--img-channels", type=int, default=1)
    parser.add_argument("--transform-name", default="log10")
    parser.add_argument("--split", choices=["train", "test", "all"], default="test", help="Which split to draw samples from; sample_idx is local to that split.")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Test split ratio for selecting train/test indices.")
    parser.add_argument("--power-spectrum-bins", type=int, default=64, help="Number of k-bins for power spectrum outputs.")
    parser.add_argument("--eps", type=float, default=5e-3)
    parser.add_argument("--t-min", type=float, default=1e-9)
    parser.add_argument("--t-max", type=float, default=1.0 - 1e-9)
    parser.add_argument("--integrator-steps", type=int, default=10000)
    parser.add_argument("--n-save", type=int, default=1)
    parser.add_argument("--n-likelihood", type=int, default=1)
    parser.add_argument("--gamma-type", type=str, default="brownian")
    parser.add_argument("--gamma-a", type=float, default=1.0)
    parser.add_argument("--time-embed-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    # fmt: on
    args = parser.parse_args()

    target, outputs = generate_samples(args)
    outputs_list = list(outputs)
    output_dir = Path(args.output_dir)
    save_outputs(outputs_list, output_dir, target=target)

    # Power spectrum outputs (compatible with plotting/plot_evaluation_si.py).
    # That plotting code expects:
    # - `k_vals.npy`: a normal `.npy` (via `np.save`)
    # - `target_pk.npy` and `si_pk.npy`: raw float32 buffers WITHOUT `.npy` headers
    if target is None:
        raise ValueError(
            "Power spectrum outputs require --target-maps so `target_pk.npy` is available for plotting."
        )
    if len(outputs_list) == 0:
        raise ValueError("No generated samples available to compute a power spectrum.")

    k_vals, target_pk_1d = _power_spectrum_curve(np.asarray(target), args.power_spectrum_bins)
    k_pred, pk_pred = _power_spectrum_curve(np.asarray(outputs_list[0]), args.power_spectrum_bins)
    if len(k_pred) != len(k_vals) or not np.allclose(k_pred, k_vals):
        raise ValueError("Inconsistent k-binning between target and generated sample.")
    pk_mse = _spectrum_mse(pk_pred, target_pk_1d)

    k_vals = np.asarray(k_vals)
    np.save(output_dir / "k_vals.npy", k_vals)

    pred_pk_arr = np.asarray(pk_pred, dtype=np.float32)[None, :]
    target_pk_arr = np.asarray(target_pk_1d, dtype=np.float32)[None, :]

    # Write raw float32 buffers (no `.npy` header) to match evaluation scripts / plotting loaders.
    target_pk_arr.tofile(output_dir / "target_pk.npy")
    pred_pk_arr.tofile(output_dir / "si_pk.npy")

    meta_path = output_dir / "power_spectra_metadata.csv"
    with meta_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_idx", "spectra_row", "pk_mse"])
        writer.writerow([0, 0, float(pk_mse)])

    metrics = {
        "power_spectrum_bins": int(args.power_spectrum_bins),
        "n_generated_total": int(len(outputs_list)),
        "n_power_spectra": int(pred_pk_arr.shape[0]),
        "pk_mse": float(pk_mse),
    }
    with (output_dir / "metrics_power_spectrum.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved {args.n_samples} samples to {args.output_dir}")


if __name__ == "__main__":
    main()
