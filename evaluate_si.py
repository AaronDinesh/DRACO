import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import optax
from dotenv import load_dotenv
from flax import nnx
from jax._src.typing import Array
from tqdm import tqdm

from src.interpolants import LinearInterpolant, SDEIntegrator, StochasticInterpolantUNet, make_gamma
from src.typing import Batch
from src.utils import batch_metrics, make_train_test_loaders, power_spectrum, restore_checkpoint


@nnx.jit(static_argnums=(5,))
def rollout(
    vel_model: StochasticInterpolantUNet,
    score_model: StochasticInterpolantUNet,
    batch: Batch,
    eval_key: Array,
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


def _accumulate(metrics: dict[str, float], batch_metrics: dict[str, float], weight: int):
    for key, value in batch_metrics.items():
        metrics[key] = metrics.get(key, 0.0) + value * weight


def _to_float_dict(metrics: dict[str, jnp.ndarray | float]) -> dict[str, float]:
    return {k: float(jax.device_get(v)) for k, v in metrics.items()}


def _power_spectrum_curve(field: jnp.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    mesh = field
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


def _plot_and_save_power_spectrum(
    target: jnp.ndarray,
    si_pred: jnp.ndarray,
    bins: int,
    sample_idx: int,
    spectra_dir: Path,
    field_name: str,
) -> float:
    curves = {
        "Target": _power_spectrum_curve(target, bins),
        "SI": _power_spectrum_curve(si_pred, bins),
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    for label, (k_vals, pk_vals) in curves.items():
        ax.loglog(k_vals, pk_vals, label=label)
    ax.set_title(f"SI Power Spectrum ({field_name}) Sample {sample_idx:05d}")
    ax.set_xlabel("Wave number k [h/Mpc]")
    ax.set_ylabel("P(k)")
    ax.legend()
    fig.tight_layout()
    save_path = spectra_dir / f"sample_{sample_idx:05d}.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    si_mse = _spectrum_mse(curves["SI"][1], curves["Target"][1])
    return si_mse


def evaluate(args: argparse.Namespace) -> None:
    _ = load_dotenv()

    output_dir = Path(args.output_dir)
    spectra_dir = output_dir / "power_spectra"
    spectra_dir.mkdir(parents=True, exist_ok=True)

    master_key = random.key(args.seed)
    si_vel_key, si_score_key, data_key, train_test_key = random.split(master_key, 4)
    base_train_test_key = train_test_key

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
        key=si_vel_key,
        in_features=args.img_channels,
        out_features=args.img_channels,
        len_cosmos_params=cosmos_params_len,
        time_embed_dim=args.time_embed_dim,
    )
    score_model = StochasticInterpolantUNet(
        key=si_score_key,
        in_features=args.img_channels,
        out_features=args.img_channels,
        len_cosmos_params=cosmos_params_len,
        time_embed_dim=args.time_embed_dim,
    )
    vel_opt = nnx.Optimizer(
        vel_model,
        optax.adam(args.si_vel_lr, b1=args.si_beta1, b2=args.si_beta2),
        wrt=nnx.Param,
    )
    score_opt = nnx.Optimizer(
        score_model,
        optax.adam(args.si_score_lr, b1=args.si_beta1, b2=args.si_beta2),
        wrt=nnx.Param,
    )
    stored_data_stats: dict[str, jnp.ndarray] | None = None
    vel_ckpt = restore_checkpoint(args.velocity_checkpoint_path, vel_model, vel_opt)
    if stored_data_stats is None and vel_ckpt.get("data_stats") is not None:
        stored_data_stats = vel_ckpt["data_stats"]
    score_ckpt = restore_checkpoint(args.score_checkpoint_path, score_model, score_opt)
    if stored_data_stats is None and score_ckpt.get("data_stats") is not None:
        stored_data_stats = score_ckpt["data_stats"]
    del vel_opt
    del score_opt

    if stored_data_stats is not None:
        mu_override = stored_data_stats.get("cosmos_params_mu")
        sigma_override = stored_data_stats.get("cosmos_params_sigma")
        if mu_override is not None and sigma_override is not None:
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
                key=base_train_test_key,
                batch_size=args.batch_size,
                input_data_path=args.input_maps,
                output_data_path=args.output_maps,
                csv_path=args.cosmos_params,
                test_ratio=args.test_ratio,
                transform_name=args.transform_name,
                mu_override=mu_override,
                sigma_override=sigma_override,
            )

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

    total_steps = max(1, math.ceil(n_test / args.batch_size))
    total_weight = 0
    sample_idx = 0
    si_metrics_sum: dict[str, float] = {}
    si_power_error = 0.0

    eval_iter: Iterable[Batch] = test_loader(key=data_key, drop_last=False)
    sample_pbar = tqdm(total=n_test, desc="SI power spectra", unit="sample")
    for batch in tqdm(eval_iter, total=total_steps, desc="Evaluating SI", unit="batch"):
        batch_size = int(batch["inputs"].shape[0])
        data_key, rollout_key = random.split(data_key)
        si_metrics, si_preds = rollout(
            vel_model,
            score_model,
            batch,
            rollout_key,
            interpolant,
            args.eps,
            t_grid,
            args.n_save,
            args.n_likelihood,
        )
        si_log = _to_float_dict(si_metrics)

        _accumulate(si_metrics_sum, si_log, batch_size)
        total_weight += batch_size

        for offset in range(batch_size):
            si_mse = _plot_and_save_power_spectrum(
                target=batch["targets"][offset],
                si_pred=si_preds[offset],
                bins=args.power_spectrum_bins,
                sample_idx=sample_idx + offset,
                spectra_dir=spectra_dir,
                field_name=args.field_name,
            )
            si_power_error += si_mse
        sample_idx += batch_size
        sample_pbar.update(batch_size)
    sample_pbar.close()

    if total_weight == 0:
        raise RuntimeError("Evaluation dataset is empty")

    si_metrics_avg = {k: v / total_weight for k, v in si_metrics_sum.items()}
    divisor = max(sample_idx, 1)
    si_metrics_avg["power_spectrum_mse"] = si_power_error / divisor

    results = {"si": si_metrics_avg}
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("SI evaluation complete. Aggregated metrics:")
    for key, value in si_metrics_avg.items():
        print(f"  {key}: {value:.6f}")
    print(f"Saved per-sample power spectra to {spectra_dir}")
    print(f"Metrics JSON saved to {metrics_path}")

    del vel_model
    del score_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stochastic Interpolant Evaluation")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--input-maps", type=str, required=True)
    parser.add_argument("--output-maps", type=str, required=True)
    parser.add_argument("--cosmos-params", type=str, required=True)
    parser.add_argument("--img-channels", type=int, default=1)
    parser.add_argument("--transform-name", type=str, default="log10")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--power-spectrum-bins", type=int, default=64)
    parser.add_argument("--field-name", type=str, default="Field")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="evaluations/si")

    # SI-specific options
    parser.add_argument("--velocity-checkpoint-path", type=str, required=True)
    parser.add_argument("--score-checkpoint-path", type=str, required=True)
    parser.add_argument("--si-vel-lr", type=float, default=2e-4)
    parser.add_argument("--si-score-lr", type=float, default=2e-4)
    parser.add_argument("--si-beta1", type=float, default=0.9)
    parser.add_argument("--si-beta2", type=float, default=0.999)
    parser.add_argument("--t-min", type=float, default=1e-3)
    parser.add_argument("--t-max", type=float, default=1.0 - 1e-3)
    parser.add_argument("--gamma-type", type=str, default="brownian")
    parser.add_argument("--gamma-a", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-2)
    parser.add_argument("--integrator-steps", type=int, default=500)
    parser.add_argument("--n-save", type=int, default=4)
    parser.add_argument("--n-likelihood", type=int, default=1)
    parser.add_argument("--time-embed-dim", type=int, default=256)

    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
