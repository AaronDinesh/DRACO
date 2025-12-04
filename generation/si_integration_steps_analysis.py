import argparse
import json
from pathlib import Path
from typing import Dict, List

import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib

matplotlib.use("Agg")  # headless backend for batch plotting
import numpy as np
import optax
from flax import nnx
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from tqdm import tqdm

from src.interpolants import LinearInterpolant, SDEIntegrator, StochasticInterpolantUNet, make_gamma
from src.utils import mae, make_train_test_loaders, mse, psnr, restore_checkpoint


def _prepare_t(t: jnp.ndarray) -> jnp.ndarray:
    if t.ndim == 1:
        return t
    return jnp.reshape(t, (t.shape[0], -1))[:, 0]


def _load_models(
    key: jax.Array,
    img_channels: int,
    cosmos_params_len: int,
    time_embed_dim: int,
    velocity_checkpoint: str,
    score_checkpoint: str,
) -> tuple[StochasticInterpolantUNet, StochasticInterpolantUNet, dict[str, jnp.ndarray] | None]:
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

    stored_data_stats: dict[str, jnp.ndarray] | None = None
    vel_ckpt = restore_checkpoint(velocity_checkpoint, vel_model, vel_opt)
    if stored_data_stats is None and vel_ckpt.get("data_stats") is not None:
        stored_data_stats = vel_ckpt["data_stats"]
    score_ckpt = restore_checkpoint(score_checkpoint, score_model, score_opt)
    if stored_data_stats is None and score_ckpt.get("data_stats") is not None:
        stored_data_stats = score_ckpt["data_stats"]
    return vel_model, score_model, stored_data_stats


def _load_sample_and_models(args: argparse.Namespace):
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
        batch_size=1,
        input_data_path=args.input_maps,
        output_data_path=args.output_maps,
        csv_path=args.cosmos_params,
        test_ratio=args.test_ratio,
        transform_name=args.transform_name,
    )

    vel_model, score_model, stored_data_stats = _load_models(
        key=vel_key,
        img_channels=args.img_channels,
        cosmos_params_len=cosmos_params_len,
        time_embed_dim=args.time_embed_dim,
        velocity_checkpoint=args.velocity_checkpoint_path,
        score_checkpoint=args.score_checkpoint_path,
    )

    # Rebuild loaders if checkpoints include data stats
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
                key=train_test_key,
                batch_size=1,
                input_data_path=args.input_maps,
                output_data_path=args.output_maps,
                csv_path=args.cosmos_params,
                test_ratio=args.test_ratio,
                transform_name=args.transform_name,
                mu_override=mu_override,
                sigma_override=sigma_override,
            )

    if args.sample_idx < 0 or args.sample_idx >= n_test:
        raise ValueError(f"sample_idx {args.sample_idx} out of range (0..{n_test - 1})")

    # Locate the requested sample in the test loader
    target_batch = None
    eval_iter = test_loader(key=data_key, drop_last=False)
    for idx, batch in enumerate(
        tqdm(eval_iter, total=n_test, desc="Scanning samples", unit="sample")
    ):
        if idx == args.sample_idx:
            target_batch = batch
            break

    if target_batch is None:
        raise RuntimeError("Sample index not found in test loader iteration.")

    inputs = target_batch["inputs"]
    targets = target_batch["targets"]
    cosmos_params = target_batch["params"]

    return inputs, targets, cosmos_params, vel_model, score_model


def _build_integrator(
    vel_model: StochasticInterpolantUNet,
    score_model: StochasticInterpolantUNet,
    cosmos: jnp.ndarray,
    args: argparse.Namespace,
    n_steps: int,
) -> SDEIntegrator:
    t_grid = jnp.linspace(args.t_min, args.t_max, n_steps + 1)
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

    def b_fn_single(x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        return vel_model(x, cosmos, _prepare_t(t))

    def s_fn_single(x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        return score_model(x, cosmos, _prepare_t(t))

    return SDEIntegrator(
        b=b_fn_single,
        s=s_fn_single,
        eps=args.eps,
        interpolant=interpolant,
        t_grid=t_grid,
        n_save=args.n_save,
        n_step=t_grid.shape[0] - 1,
        n_likelihood=args.n_likelihood,
    )


def _generate_for_steps(
    integrator: SDEIntegrator,
    x0: jnp.ndarray,
    n_samples: int,
    key: jax.Array,
) -> list[jnp.ndarray]:
    rollout = jax.jit(lambda x, k: integrator.forward_rollout(x, k))
    preds: list[jnp.ndarray] = []
    for _ in range(n_samples):
        key, sub = random.split(key)
        preds.append(rollout(x0, sub))
    return preds


def _compute_metrics(preds: list[jnp.ndarray], target: jnp.ndarray) -> dict[str, object]:
    pred_stack = jnp.stack([p[0] for p in preds], axis=0)  # drop batch axis
    target_single = target[0]

    mse_vals = jax.vmap(mse)(pred_stack, target_single)
    mae_vals = jax.vmap(mae)(pred_stack, target_single)
    psnr_vals = jax.vmap(psnr)(pred_stack, target_single)

    return {
        "mse_values": np.asarray(mse_vals),
        "mae_values": np.asarray(mae_vals),
        "psnr_values": np.asarray(psnr_vals),
        "mse_mean": float(jnp.mean(mse_vals)),
        "mae_mean": float(jnp.mean(mae_vals)),
        "psnr_mean": float(jnp.mean(psnr_vals)),
    }


def _psnr_distribution_difference(psnr_vals: np.ndarray, target: jnp.ndarray) -> float:
    """
    Compare the PSNR distribution of generated rollouts to the degenerate distribution
    of the ground-truth sample (PSNR of target vs itself). This reduces to the average
    absolute gap between generated PSNR values and the target's self-PSNR.
    """
    base = float(psnr(target, target))
    return float(np.mean(np.abs(psnr_vals - base)))


def _plot_line(x: List[int], y: List[float], ylabel: str, title: str, out_path: Path):
    fig = Figure(figsize=(6, 4))
    FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.plot(x, y, marker="o")
    ax.set_xlabel("Integration steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def main():
    parser = argparse.ArgumentParser("Sweep SI integration steps for a single sample")
    # fmt: off
    parser.add_argument("--input-maps", type=str, required=True)
    parser.add_argument("--output-maps", type=str, required=True)
    parser.add_argument("--cosmos-params", type=str, required=True)
    parser.add_argument("--sample-idx", type=int, required=True, help="Index within the test split")
    parser.add_argument("--velocity-checkpoint-path", type=str, required=True)
    parser.add_argument("--score-checkpoint-path", type=str, required=True)
    parser.add_argument("--integration-steps", type=int, nargs="+", required=True,
                        help="List of integration step counts to test")
    parser.add_argument("--n-samples", type=int, default=8, help="Number of stochastic rollouts per setting")
    parser.add_argument("--output-dir", type=str, default="generation/si_integration_analysis")
    parser.add_argument("--img-channels", type=int, default=1)
    parser.add_argument("--transform-name", type=str, default="log10")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--t-min", type=float, default=1e-3)
    parser.add_argument("--t-max", type=float, default=1.0 - 1e-3)
    parser.add_argument("--gamma-type", type=str, default="brownian")
    parser.add_argument("--gamma-a", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-2)
    parser.add_argument("--n-save", type=int, default=1)
    parser.add_argument("--n-likelihood", type=int, default=1)
    parser.add_argument("--time-embed-dim", type=int, default=256)
    # fmt: on
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x0, target, cosmos, vel_model, score_model = _load_sample_and_models(args)

    # Sweep integration steps
    results: Dict[int, dict] = {}
    sweep_key = random.key(args.seed + 1)
    for steps in tqdm(args.integration_steps, desc="Integration step sweep", unit="setting"):
        if steps <= 0:
            raise ValueError("integration-steps must be positive")

        integrator = _build_integrator(
            vel_model=vel_model,
            score_model=score_model,
            cosmos=cosmos,
            args=args,
            n_steps=steps,
        )
        preds = _generate_for_steps(
            integrator=integrator,
            x0=x0,
            n_samples=args.n_samples,
            key=sweep_key,
        )
        sweep_key, _ = random.split(sweep_key)  # advance key between settings
        metrics = _compute_metrics(preds, target)
        metrics["psnr_distribution_diff"] = _psnr_distribution_difference(
            metrics["psnr_values"], target
        )
        results[steps] = metrics

    # Save metrics JSON (arrays converted to lists)
    metrics_path = output_dir / "integration_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=2,
            default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x,
        )

    # Plots
    sorted_steps = sorted(results.keys())
    mse_vals = [results[s]["mse_mean"] for s in sorted_steps]
    mae_vals = [results[s]["mae_mean"] for s in sorted_steps]
    psnr_diff_vals = [results[s]["psnr_distribution_diff"] for s in sorted_steps]

    _plot_line(
        sorted_steps,
        mse_vals,
        ylabel="MSE",
        title="MSE vs integration steps",
        out_path=output_dir / "mse_vs_steps.png",
    )
    _plot_line(
        sorted_steps,
        mae_vals,
        ylabel="MAE",
        title="MAE vs integration steps",
        out_path=output_dir / "mae_vs_steps.png",
    )
    _plot_line(
        sorted_steps,
        psnr_diff_vals,
        ylabel="PSNR distribution difference",
        title="PSNR distribution difference vs steps",
        out_path=output_dir / "psnr_distribution_diff.png",
    )

    print(f"Finished integration sweep. Metrics saved to {metrics_path}")
    print(f"Plots written to {output_dir}")


if __name__ == "__main__":
    main()
