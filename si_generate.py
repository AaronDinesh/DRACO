import argparse
from pathlib import Path
from typing import Iterable, Tuple

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
from src.utils import make_transform, restore_checkpoint


def _add_channel_last(x: jnp.ndarray) -> jnp.ndarray:
    return x[..., None] if x.ndim == 2 else x


def _load_input(
    input_maps_path: str,
    params_path: str,
    sample_idx: int,
    transform_name: str,
    target_maps_path: str | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, np.ndarray | None]:
    input_maps = np.load(input_maps_path, mmap_mode="r")
    cosmos_params = pd.read_csv(params_path, header=None, sep=" ")
    cosmos_params = jnp.asarray(cosmos_params.to_numpy(), dtype=jnp.float32)

    if sample_idx < 0 or sample_idx >= len(input_maps):
        raise ValueError(
            f"sample_idx {sample_idx} out of range for dataset of size {len(input_maps)}"
        )

    forward_transform, _ = make_transform(name=transform_name)
    cosmos_mu = jnp.mean(cosmos_params, axis=0)
    cosmos_sigma = jnp.std(cosmos_params, axis=0) + 1e-6

    x0 = jnp.asarray(
        forward_transform(_add_channel_last(jnp.asarray(input_maps[sample_idx]))), dtype=jnp.float32
    )
    cosmos = jnp.asarray((cosmos_params[sample_idx] - cosmos_mu) / cosmos_sigma, dtype=jnp.float32)

    target: np.ndarray | None = None
    if target_maps_path:
        target_maps = np.load(target_maps_path, mmap_mode="r")
        if sample_idx < 0 or sample_idx >= len(target_maps):
            raise ValueError(
                f"sample_idx {sample_idx} out of range for target dataset of size {len(target_maps)}"
            )
        target = np.asarray(
            forward_transform(_add_channel_last(np.asarray(target_maps[sample_idx]))), dtype=np.float32
        )

    return x0, cosmos, target


def _build_models(
    key: jax.Array,
    img_channels: int,
    cosmos_params_len: int,
    time_embed_dim: int,
    velocity_checkpoint: str,
    score_checkpoint: str,
) -> Tuple[StochasticInterpolantUNet, StochasticInterpolantUNet]:
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

    restore_checkpoint(velocity_checkpoint, vel_model, vel_opt)
    restore_checkpoint(score_checkpoint, score_model, score_opt)

    return vel_model, score_model


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
    model_key, sample_key = random.split(master_key)

    x0, cosmos, target = _load_input(
        args.input_maps,
        args.cosmos_params,
        args.sample_idx,
        args.transform_name,
        target_maps_path=args.target_maps,
    )
    cosmos = cosmos[None, ...]  # add batch axis
    x0 = x0[None, ...]  # add batch axis

    vel_model, score_model = _build_models(
        key=model_key,
        img_channels=args.img_channels,
        cosmos_params_len=cosmos.shape[-1],
        time_embed_dim=args.time_embed_dim,
        velocity_checkpoint=args.velocity_checkpoint_path,
        score_checkpoint=args.score_checkpoint_path,
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
    if target is not None:
        target_uint8 = _to_uint8(np.asarray(target))
        Image.fromarray(target_uint8).save(output_dir / "target.png")

    for idx, arr in enumerate(outputs):
        arr_uint8 = _to_uint8(np.asarray(arr))
        img = Image.fromarray(arr_uint8)
        out_path = output_dir / f"sample_{idx:03d}.png"
        img.save(out_path)


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
    parser.add_argument("--eps", type=float, default=5e-3)
    parser.add_argument("--t-min", type=float, default=1e-9)
    parser.add_argument("--t-max", type=float, default=1.0 - 1e-9)
    parser.add_argument("--integrator-steps", type=int, default=3000)
    parser.add_argument("--n-save", type=int, default=1)
    parser.add_argument("--n-likelihood", type=int, default=1)
    parser.add_argument("--gamma-type", type=str, default="brownian")
    parser.add_argument("--gamma-a", type=float, default=1.0)
    parser.add_argument("--time-embed-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    # fmt: on
    args = parser.parse_args()

    target, outputs = generate_samples(args)
    save_outputs(outputs, Path(args.output_dir), target=target)
    print(f"Saved {args.n_samples} samples to {args.output_dir}")


if __name__ == "__main__":
    main()
