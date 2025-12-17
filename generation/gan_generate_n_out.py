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

from src import Generator
from src.utils import make_transform, restore_checkpoint


def _add_channel_last(x: jnp.ndarray) -> jnp.ndarray:
    return x[..., None] if x.ndim == 2 else x


def _append_noise_channel(inputs: jnp.ndarray, key: jax.Array, sigma: float = 0.5) -> jnp.ndarray:
    noise = sigma * random.normal(key, shape=inputs.shape[:-1] + (1,), dtype=inputs.dtype)
    return jnp.concatenate((inputs, noise), axis=-1)


def _load_sample(
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

    inputs = jnp.asarray(
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

    return inputs, cosmos, target


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


def generate_samples(args: argparse.Namespace) -> tuple[np.ndarray | None, Iterable[np.ndarray]]:
    _ = load_dotenv()

    master_key = random.key(args.seed)
    gen_key, sample_key = random.split(master_key)

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

    cosmos_mu = jnp.mean(cosmos_params, axis=0)
    cosmos_sigma = jnp.std(cosmos_params, axis=0) + 1e-6

    # Build generator with feature size matching training (optionally with noise channel).
    input_features_size = args.img_channels + 1 if args.add_noise else args.img_channels
    generator = Generator(
        key=gen_key,
        in_features=input_features_size,
        out_features=int(args.img_channels),
        len_cosmos_params=cosmos_params.shape[-1],
    )
    opt_gen = nnx.Optimizer(generator, optax.adam(2e-4), wrt=nnx.Param)

    ckpt_path = (
        args.generator_noise_checkpoint_path
        if args.add_noise and args.generator_noise_checkpoint_path is not None
        else args.generator_checkpoint_path
    )
    if ckpt_path is None:
        raise ValueError("A generator checkpoint path must be provided.")

    # Restore checkpoint and use stored cosmos stats if available.
    meta = restore_checkpoint(ckpt_path, generator, opt_gen)
    if meta.get("data_stats") is not None:
        data_stats = meta["data_stats"]
        if (
            data_stats.get("cosmos_params_mu") is not None
            and data_stats.get("cosmos_params_sigma") is not None
        ):
            cosmos_mu = jnp.asarray(data_stats["cosmos_params_mu"])
            cosmos_sigma = jnp.asarray(data_stats["cosmos_params_sigma"])

    inputs, cosmos, target = _load_sample(
        input_maps=input_maps,
        cosmos_params=cosmos_params,
        sample_idx=args.sample_idx,
        transform_name=args.transform_name,
        cosmos_mu=cosmos_mu,
        cosmos_sigma=cosmos_sigma,
        target_maps_path=args.target_maps,
    )

    inputs = inputs[None, ...]  # add batch axis
    cosmos = cosmos[None, ...]  # add batch axis

    def _pred_iter():
        nonlocal sample_key
        for _ in tqdm(range(args.n_samples), desc="Generating samples", unit="sample"):
            if args.add_noise:
                sample_key, noise_key = random.split(sample_key)
                gan_inputs = _append_noise_channel(inputs, noise_key)
            else:
                gan_inputs = inputs
            preds = generator(gan_inputs, cosmos)
            yield np.asarray(preds[0])

    return target, _pred_iter()


def main():
    parser = argparse.ArgumentParser("GAN sampler: generate N outputs from one input")
    # fmt: off
    parser.add_argument("--input-maps", required=True, help="Path to input .npy array (N,H,W[,C])")
    parser.add_argument("--cosmos-params", required=True, help="Path to cosmos params txt/csv")
    parser.add_argument("--sample-idx", type=int, default=0, help="Index of the input to generate for")
    parser.add_argument("--target-maps", help="Optional path to target/output .npy array for visualization")
    parser.add_argument("--generator-checkpoint-path", required=False, help="Generator checkpoint path")
    parser.add_argument("--generator-noise-checkpoint-path", required=False, help="Optional noise-GAN generator checkpoint (used when --add-noise)")
    parser.add_argument("--output-dir", required=True, help="Directory to write generated samples")
    parser.add_argument("--n-samples", type=int, default=15, help="Number of generated outputs")
    parser.add_argument("--img-channels", type=int, default=1, help="Number of channels in the input/output images")
    parser.add_argument("--transform-name", default="signed_log1p", help="Transform applied during training (default: signed_log1p)")
    parser.add_argument("--add-noise", action="store_true", help="If set, append a noise channel (for noise-GAN checkpoints)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    # fmt: on
    args = parser.parse_args()

    target, outputs = generate_samples(args)
    save_outputs(outputs, Path(args.output_dir), target=target)
    print(f"Saved {args.n_samples} samples to {args.output_dir}")


if __name__ == "__main__":
    main()
