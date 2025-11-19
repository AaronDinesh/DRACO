import argparse
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import optax
from dotenv import load_dotenv
from flax import nnx
from tqdm import tqdm

from src.interpolants import LinearInterpolant, SDEIntegrator, StochasticInterpolantUNet, make_gamma
from src.typing import Batch
from src.utils import batch_metrics, make_train_test_loaders, power_spectrum, restore_checkpoint


def rollout(
    vel_model: StochasticInterpolantUNet,
    score_model: StochasticInterpolantUNet,
    batch: Batch,
    eval_key: jnp.ndarray,
    interpolant: LinearInterpolant,
    eps: float,
    t_grid: jnp.ndarray,
    n_save: int,
    n_likelihood: int,
) -> tuple[dict[str, jnp.ndarray], jnp.ndarray]:
    cosmos = batch["params"]
    x0 = batch["inputs"]
    x1 = batch["targets"]

    def _prepare_t(t: jnp.ndarray) -> jnp.ndarray:
        t = jnp.reshape(t, (t.shape[0],))
        return t

    def b_fn(x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        return vel_model(x, cosmos, _prepare_t(t))

    def s_fn(x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        return score_model(x, cosmos, _prepare_t(t))

    integrator = SDEIntegrator(
        b=b_fn,
        s=s_fn,
        eps=eps,
        interpolant=interpolant,
        t_grid=t_grid,
        n_save=n_save,
        n_step=t_grid.shape[0] - 1,
        n_likelihood=n_likelihood,
    )
    preds = integrator.forward_rollout(x0, eval_key)
    metrics = batch_metrics(preds, x1)
    return metrics, preds


def _power_spectrum_values(final_img: jnp.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    mesh = final_img
    if mesh.ndim == 3 and mesh.shape[-1] == 1:
        mesh = mesh[..., 0]
    k_vals, pk = power_spectrum(mesh, kedges=bins)
    return np.asarray(jax.device_get(k_vals)), np.asarray(jax.device_get(pk))


def _power_spectrum_metrics(
    preds: jnp.ndarray, targets: jnp.ndarray, bins: int
) -> tuple[float, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    batch_mses: list[float] = []
    spectra: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for pred, target in zip(preds, targets):
        k_pred, pk_pred = _power_spectrum_values(pred, bins)
        _, pk_target = _power_spectrum_values(target, bins)
        min_len = min(len(pk_pred), len(pk_target))
        spectra.append((k_pred[:min_len], pk_pred[:min_len], pk_target[:min_len]))
        if min_len == 0:
            continue
        diff = pk_pred[:min_len] - pk_target[:min_len]
        batch_mses.append(float(np.mean(np.square(diff))))

    mse = float(np.mean(batch_mses)) if batch_mses else 0.0
    return mse, spectra


def _plot_power_spectrum(
    data: tuple[np.ndarray, np.ndarray, np.ndarray], sample_idx: int
):
    k_vals, pred_spectra, target_spectra = data
    if len(k_vals) == 0:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(k_vals, pred_spectra, label="generated")
    ax.loglog(k_vals, target_spectra, label="target")
    ax.set_title(f"Power Spectrum Sample {sample_idx:05d}")
    ax.set_xlabel("Wave number k [h/Mpc]")
    ax.set_ylabel("P(k)")
    ax.legend()
    fig.tight_layout()
    return fig


def _accumulate(metrics: dict[str, float], batch_metrics: dict[str, float], weight: int):
    for key, value in batch_metrics.items():
        metrics[key] = metrics.get(key, 0.0) + value * weight


def _to_float_dict(metrics: dict[str, jnp.ndarray | float]) -> dict[str, float]:
    return {k: float(jax.device_get(v)) for k, v in metrics.items()}


def evaluate(args: argparse.Namespace) -> None:
    _ = load_dotenv()

    output_dir = Path(args.output_dir)
    spectra_dir = output_dir / "power_spectra"
    spectra_dir.mkdir(parents=True, exist_ok=True)

    master_key = random.key(args.seed)
    vel_key, score_key, data_key, train_test_key = random.split(master_key, 4)

    (
        _train_loader,
        test_loader,
        _n_train,
        n_test,
        _img_size,
        cosmos_params_len,
        _cosmos_mu,
        _cosmos_sigma,
    ) = make_train_test_loaders(
        key=train_test_key,
        batch_size=args.batch_size,
        input_data_path=args.input_maps,
        output_data_path=args.output_maps,
        csv_path=args.cosmos_params,
        test_ratio=args.test_ratio,
        transform_name=args.transform_name,
    )

    vel_model = StochasticInterpolantUNet(
        key=vel_key,
        in_features=args.img_channels,
        out_features=args.img_channels,
        len_cosmos_params=cosmos_params_len,
        time_embed_dim=args.time_embed_dim,
    )
    score_model = StochasticInterpolantUNet(
        key=score_key,
        in_features=args.img_channels,
        out_features=args.img_channels,
        len_cosmos_params=cosmos_params_len,
        time_embed_dim=args.time_embed_dim,
    )

    vel_opt = nnx.Optimizer(
        vel_model,
        optax.adam(args.vel_lr, b1=args.beta1, b2=args.beta2),
        wrt=nnx.Param,
    )
    score_opt = nnx.Optimizer(
        score_model,
        optax.adam(args.score_lr, b1=args.beta1, b2=args.beta2),
        wrt=nnx.Param,
    )

    restore_checkpoint(args.velocity_checkpoint_path, vel_model, vel_opt)
    restore_checkpoint(args.score_checkpoint_path, score_model, score_opt)

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

    _train_key, test_key = random.split(data_key)

    total_steps = max(1, math.ceil(n_test / args.batch_size))
    total_weight = 0
    sample_idx = 0
    metrics_sum: dict[str, float] = {}

    eval_iter = test_loader(key=test_key, drop_last=False)
    for batch in tqdm(eval_iter, total=total_steps, desc="Evaluating SI", unit="batch"):
        test_key, eval_subkey = random.split(test_key)
        metrics, preds = rollout(
            vel_model,
            score_model,
            batch,
            eval_subkey,
            interpolant,
            args.eps,
            t_grid,
            args.n_save,
            args.n_likelihood,
        )
        power_mse, batch_spectra = _power_spectrum_metrics(
            preds,
            batch["targets"],
            args.power_spectrum_bins,
        )
        merged_metrics = {**_to_float_dict(metrics), "power_spectrum_mse": float(power_mse)}
        batch_weight = int(batch["inputs"].shape[0])
        _accumulate(metrics_sum, merged_metrics, batch_weight)
        total_weight += batch_weight

        for offset, spectra in enumerate(batch_spectra):
            fig = _plot_power_spectrum(spectra, sample_idx + offset)
            if fig is None:
                continue
            save_path = spectra_dir / f"sample_{sample_idx + offset:05d}.png"
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
        sample_idx += batch_weight

    if total_weight == 0:
        raise RuntimeError("Evaluation dataset is empty")

    final_metrics = {k: v / total_weight for k, v in metrics_sum.items()}
    output_path = output_dir / "metrics.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    print("Evaluation complete. Aggregated metrics:")
    for key, value in final_metrics.items():
        print(f"  {key}: {value:.6f}")
    print(f"Saved per-sample power spectra to {spectra_dir}")
    print(f"Metrics JSON saved to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stochastic Interpolant Evaluation Script")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--input-maps", type=str, required=True)
    parser.add_argument("--output-maps", type=str, required=True)
    parser.add_argument("--cosmos-params", type=str, required=True)
    parser.add_argument("--img-channels", type=int, default=1)
    parser.add_argument("--transform-name", type=str, default="log10")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--vel-lr", type=float, default=2e-4)
    parser.add_argument("--score-lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--t-min", type=float, default=1e-3)
    parser.add_argument("--t-max", type=float, default=1.0 - 1e-3)
    parser.add_argument("--gamma-type", type=str, default="brownian")
    parser.add_argument("--gamma-a", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-2)
    parser.add_argument("--integrator-steps", type=int, default=500)
    parser.add_argument("--n-save", type=int, default=4)
    parser.add_argument("--n-likelihood", type=int, default=1)
    parser.add_argument("--power-spectrum-bins", type=int, default=64)
    parser.add_argument("--time-embed-dim", type=int, default=256)
    parser.add_argument("--velocity-checkpoint-path", type=str, required=True)
    parser.add_argument("--score-checkpoint-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="evaluations/si")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
