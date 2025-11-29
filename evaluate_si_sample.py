import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import optax
from dotenv import load_dotenv
from flax import nnx
from jax._src.typing import Array
from matplotlib import pyplot as plt

from src.interpolants import LinearInterpolant, SDEIntegrator, StochasticInterpolantUNet, make_gamma
from src.utils import make_train_test_loaders, restore_checkpoint


def _prepare_t(t: jnp.ndarray) -> jnp.ndarray:
    if t.ndim == 1:
        return t
    return jnp.reshape(t, (t.shape[0], -1))[:, 0]


def save_trace_images(states: list[jnp.ndarray], target: jnp.ndarray, out_dir: Path):
    pred_dir = out_dir / "pred_steps"
    target_dir = out_dir / "target"
    pred_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    target_np = np.asarray(jax.device_get(target))
    target_img = (
        target_np[..., 0] if target_np.ndim == 3 and target_np.shape[-1] == 1 else target_np
    )
    plt.imsave(target_dir / "target.png", target_img, cmap="gray")

    for step_idx, state in enumerate(states):
        state_np = np.asarray(jax.device_get(state))
        state_img = state_np[..., 0] if state_np.ndim == 4 and state_np.shape[-1] == 1 else state_np
        plt.imsave(pred_dir / f"step_{step_idx:05d}.png", state_img[0], cmap="gray")


def evaluate_single(args: argparse.Namespace) -> None:
    _ = load_dotenv()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    master_key = random.key(args.seed)
    si_vel_key, si_score_key, data_key, train_test_key = random.split(master_key, 4)

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
        batch_size=1,  # force single-sample batches for tracing
        input_data_path=args.input_maps,
        output_data_path=args.output_maps,
        csv_path=args.cosmos_params,
        test_ratio=args.test_ratio,
        transform_name=args.transform_name,
    )

    if args.sample_idx < 0 or args.sample_idx >= n_test:
        raise ValueError(f"sample_idx {args.sample_idx} out of range (0..{n_test - 1})")

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

    # Rebuild loaders if data stats were stored
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
                batch_size=args.batch_size,
                input_data_path=args.input_maps,
                output_data_path=args.output_maps,
                csv_path=args.cosmos_params,
                test_ratio=args.test_ratio,
                transform_name=args.transform_name,
                mu_override=mu_override,
                sigma_override=sigma_override,
            )

    # Locate the sample batch containing sample_idx (batch_size is fixed to 1)
    eval_iter = test_loader(key=data_key, drop_last=False)
    target_inputs = None
    target_targets = None
    target_params = None
    for idx, batch in enumerate(eval_iter):
        if idx == args.sample_idx:
            target_inputs = batch["inputs"]
            target_targets = batch["targets"]
            target_params = batch["params"]
            break

    if target_inputs is None:
        raise RuntimeError("Sample index not found in test loader iteration.")

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

    def b_fn_single(x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        return vel_model(x, target_params, _prepare_t(t))

    def s_fn_single(x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        return score_model(x, target_params, _prepare_t(t))

    trace_integrator = SDEIntegrator(
        b=b_fn_single,
        s=s_fn_single,
        eps=args.eps,
        interpolant=interpolant,
        t_grid=t_grid,
        n_save=args.n_save,
        n_step=t_grid.shape[0] - 1,
        n_likelihood=args.n_likelihood,
    )

    _, rollout_key = random.split(data_key)
    trace_states = trace_integrator.forward_rollout_trace(target_inputs, rollout_key)

    trace_base = Path(args.trace_dir) if args.trace_dir is not None else output_dir
    trace_out = trace_base / f"trace_sample_{args.sample_idx:05d}"
    save_trace_images(trace_states, target_targets[0], trace_out)
    print(f"Trace for sample {args.sample_idx} saved to {trace_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stochastic Interpolant Single Sample Trace")
    parser.add_argument("--batch-size", type=int, default=1, help="Ignored; batch is fixed to 1")
    parser.add_argument("--input-maps", type=str, required=True)
    parser.add_argument("--output-maps", type=str, required=True)
    parser.add_argument("--cosmos-params", type=str, required=True)
    parser.add_argument("--img-channels", type=int, default=1)
    parser.add_argument("--transform-name", type=str, default="log10")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="evaluations/si_sample")
    parser.add_argument("--trace-dir", type=str, default=None)
    parser.add_argument("--sample-idx", type=int, required=True)

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
    evaluate_single(parse_args())
