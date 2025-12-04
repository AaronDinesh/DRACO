import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import optax
from dotenv import load_dotenv
from flax import nnx
from GAN_train import make_train_test_loaders
from jax._src.typing import Array
from matplotlib import pyplot as plt
from tqdm import tqdm

from src import Discriminator, Generator
from src.typing import Batch
from src.utils import batch_metrics, power_spectrum, restore_checkpoint


@nnx.jit
def gan_eval_step(
    disc: Discriminator,
    gen: Generator,
    batch: Batch,
    l1_lambda: float = 100.0,
) -> dict[str, jnp.ndarray]:
    inputs, cosmos_params, targets = batch["inputs"], batch["params"], batch["targets"]

    out_real_logits = disc(
        inputs=inputs,
        output=targets,
        condition_params=cosmos_params,
        is_training=False,
    )
    fake_images = gen(inputs, cosmos_params)
    out_fake_logits = disc(
        inputs=inputs,
        output=fake_images,
        condition_params=cosmos_params,
        is_training=False,
    )

    def d_hinge_loss(
        real_logits: jnp.ndarray, fake_logits: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        real_loss = jnp.maximum(0.0, 1.0 - real_logits)
        fake_loss = jnp.maximum(0.0, 1.0 + fake_logits)
        return (jnp.mean(real_loss) + jnp.mean(fake_loss), real_loss, fake_loss)

    def g_hinge_l1_loss(
        fake_logits: jnp.ndarray,
        y_fake: jnp.ndarray,
        y_real: jnp.ndarray,
        l1_lambda: float = 100.0,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        adversarial = -jnp.mean(fake_logits)
        reconstruction_loss = jnp.mean(jnp.abs(y_real - y_fake))
        return (
            adversarial + l1_lambda * reconstruction_loss,
            adversarial,
            reconstruction_loss,
        )

    disc_loss, _, _ = d_hinge_loss(out_real_logits, out_fake_logits)
    gen_loss, adversarial_loss, reconstruction_loss = g_hinge_l1_loss(
        out_fake_logits, fake_images, targets, l1_lambda
    )

    d_real_acc = (out_real_logits > 0.0).mean()
    d_fake_acc = (out_fake_logits < 0.0).mean()
    d_acc = 0.5 * (d_real_acc + d_fake_acc)
    g_trick_acc = (out_fake_logits > 0.0).mean()

    return {
        "d_loss": disc_loss,
        "d_real_acc": d_real_acc,
        "d_fake_acc": d_fake_acc,
        "d_acc": d_acc,
        "g_loss": gen_loss,
        "g_adversarial": adversarial_loss,
        "g_reconstruct": reconstruction_loss,
        "g_trick_acc": g_trick_acc,
        "sample_fake": fake_images,
    }


@jax.jit
def _append_noise_channel(inputs: jnp.ndarray, key: Array) -> jnp.ndarray:
    sigma = 0.5
    noise = sigma * random.normal(key, shape=inputs.shape[:-1] + (1,), dtype=inputs.dtype)
    return jnp.concatenate((inputs, noise), axis=-1)


def _maybe_add_noise(inputs: jnp.ndarray, add_noise: bool, key: Array | None) -> jnp.ndarray:
    if not add_noise:
        return inputs
    if key is None:
        raise ValueError("Noise-enabled evaluation requires a PRNG key")
    return _append_noise_channel(inputs, key)


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


def evaluate(args: argparse.Namespace) -> None:
    _ = load_dotenv()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir: Path | None = None
    target_images_dir: Path | None = None
    pred_images_dir: Path | None = None
    if args.save_images:
        images_dir = output_dir / "images"
        target_images_dir = images_dir / "target"
        pred_images_dir = images_dir / "pred"
        target_images_dir.mkdir(parents=True, exist_ok=True)
        pred_images_dir.mkdir(parents=True, exist_ok=True)

    master_key = random.key(args.seed)
    gan_gen_key, gan_disc_key, data_key, train_test_key = random.split(master_key, 4)
    base_train_test_key = train_test_key

    (
        _train_loader,
        test_loader,
        _n_train,
        n_test,
        _img_size,
        cosmos_params_len,
        cosmos_mu,
        cosmos_sigma,
    ) = make_train_test_loaders(
        key=train_test_key,
        batch_size=args.batch_size,
        input_data_path=args.input_maps,
        output_data_path=args.output_maps,
        csv_path=args.cosmos_params,
        test_ratio=args.test_ratio,
        transform_name=args.transform_name,
    )

    in_channels = args.img_channels + (1 if args.add_noise else 0)
    generator = Generator(
        key=gan_gen_key,
        in_features=in_channels,
        out_features=args.img_channels,
        len_cosmos_params=cosmos_params_len,
    )
    discriminator = Discriminator(
        key=gan_disc_key,
        input_channels=in_channels,
        condition_dim=cosmos_params_len,
        condition_proj_dim=8,
        target_channels=args.img_channels,
    )

    stored_data_stats: dict[str, jnp.ndarray] | None = None

    opt_gen = nnx.Optimizer(
        generator,
        optax.adam(args.gan_g_lr, b1=args.gan_beta1, b2=args.gan_beta2),
        wrt=nnx.Param,
    )
    opt_disc = nnx.Optimizer(
        discriminator,
        optax.adam(args.gan_d_lr, b1=args.gan_beta1, b2=args.gan_beta2),
        wrt=nnx.Param,
    )
    gen_ckpt_path = (
        args.generator_noise_checkpoint_path
        if args.generator_noise_checkpoint_path is not None
        else args.generator_checkpoint_path
    )
    disc_ckpt_path = (
        args.discriminator_noise_checkpoint_path
        if args.discriminator_noise_checkpoint_path is not None
        else args.discriminator_checkpoint_path
    )
    if gen_ckpt_path is None or disc_ckpt_path is None:
        raise ValueError("Checkpoint paths must be provided for GAN evaluation")

    gen_ckpt = restore_checkpoint(gen_ckpt_path, generator, opt_gen)
    if stored_data_stats is None and gen_ckpt.get("data_stats") is not None:
        stored_data_stats = gen_ckpt["data_stats"]
    disc_ckpt = restore_checkpoint(disc_ckpt_path, discriminator, opt_disc)
    if stored_data_stats is None and disc_ckpt.get("data_stats") is not None:
        stored_data_stats = disc_ckpt["data_stats"]
    del opt_gen
    del opt_disc

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
                cosmos_mu,
                cosmos_sigma,
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

    total_steps = max(1, math.ceil(n_test / args.batch_size))
    total_weight = 0
    sample_idx = 0
    gan_metrics_sum: dict[str, float] = {}
    gan_power_error = 0.0
    spectra_k_ref: np.ndarray | None = None
    target_pk_mem: np.memmap | None = None
    gan_pk_mem: np.memmap | None = None
    cosmos_mem = np.memmap(
        output_dir / "cosmos_params.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n_test, cosmos_params_len),
    )

    # Keep test sample ordering fixed so SI and GAN evaluations align one-to-one.
    eval_iter: Iterable[Batch] = test_loader(key=None, drop_last=False)
    metadata_path = output_dir / "power_spectra_metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as metadata_file:
        writer = csv.writer(metadata_file)
        header = ["sample_idx", "spectra_row"] + [
            f"cosmo_param_{i}" for i in range(cosmos_params_len)
        ]
        writer.writerow(header)

        sample_pbar = tqdm(total=n_test, desc="GAN power spectra", unit="sample")
        for batch in tqdm(eval_iter, total=total_steps, desc="Evaluating GAN", unit="batch"):
            batch_size = int(batch["inputs"].shape[0])
            noise_key = None
            if args.add_noise:
                data_key, noise_key = random.split(data_key)
            gan_inputs = _maybe_add_noise(batch["inputs"], args.add_noise, noise_key)
            gan_batch = {**batch, "inputs": gan_inputs}

            gan_metrics = gan_eval_step(
                discriminator, generator, gan_batch, l1_lambda=args.l1_lambda
            )
            fake_images = gan_metrics.pop("sample_fake")
            recon_metrics = batch_metrics(fake_images, batch["targets"])
            gan_log = {**_to_float_dict(gan_metrics), **_to_float_dict(recon_metrics)}

            _accumulate(gan_metrics_sum, gan_log, batch_size)
            total_weight += batch_size

            cosmos_params_denorm = np.asarray(
                jax.device_get(batch["params"] * cosmos_sigma + cosmos_mu)
            )

            for offset in range(batch_size):
                k_target, pk_target = _power_spectrum_curve(
                    batch["targets"][offset], args.power_spectrum_bins
                )
                k_pred, pk_pred = _power_spectrum_curve(
                    fake_images[offset], args.power_spectrum_bins
                )

                if spectra_k_ref is None:
                    spectra_k_ref = k_target
                    k_len = len(k_target)
                    target_pk_mem = np.memmap(
                        output_dir / "target_pk.npy",
                        mode="w+",
                        dtype=np.float32,
                        shape=(n_test, k_len),
                    )
                    gan_pk_mem = np.memmap(
                        output_dir / "gan_pk.npy",
                        mode="w+",
                        dtype=np.float32,
                        shape=(n_test, k_len),
                    )
                    np.save(output_dir / "k_vals.npy", spectra_k_ref)
                elif len(k_target) != len(spectra_k_ref):
                    raise ValueError("Inconsistent k-binning encountered across samples.")
                if len(k_pred) != len(spectra_k_ref):
                    raise ValueError("Predicted spectrum k-binning differs from reference.")

                gan_mse = _spectrum_mse(pk_pred, pk_target)
                gan_power_error += gan_mse

                row = sample_idx + offset
                cosmos_mem[row] = cosmos_params_denorm[offset]
                target_pk_mem[row] = pk_target  # type: ignore[arg-type]
                gan_pk_mem[row] = pk_pred  # type: ignore[arg-type]
                writer.writerow([row, row, *cosmos_params_denorm[offset].tolist()])
                if target_images_dir is not None and pred_images_dir is not None:
                    target_np = np.asarray(jax.device_get(batch["targets"][offset]))
                    pred_np = np.asarray(jax.device_get(fake_images[offset]))
                    target_img = (
                        target_np[..., 0]
                        if target_np.ndim == 3 and target_np.shape[-1] == 1
                        else target_np
                    )
                    pred_img = (
                        pred_np[..., 0] if pred_np.ndim == 3 and pred_np.shape[-1] == 1 else pred_np
                    )
                    plt.imsave(target_images_dir / f"sample_{row:05d}.png", target_img, cmap="gray")
                    plt.imsave(pred_images_dir / f"sample_{row:05d}.png", pred_img, cmap="gray")
            sample_idx += batch_size
            sample_pbar.update(batch_size)
        sample_pbar.close()

    if total_weight == 0:
        raise RuntimeError("Evaluation dataset is empty")

    gan_metrics_avg = {k: v / total_weight for k, v in gan_metrics_sum.items()}
    divisor = max(sample_idx, 1)
    gan_metrics_avg["power_spectrum_mse"] = gan_power_error / divisor

    results = {"gan": gan_metrics_avg}
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    if target_pk_mem is not None:
        target_pk_mem.flush()
    if gan_pk_mem is not None:
        gan_pk_mem.flush()
    cosmos_mem.flush()

    print("GAN evaluation complete. Aggregated metrics:")
    for key, value in gan_metrics_avg.items():
        print(f"  {key}: {value:.6f}")
    print(f"Metrics JSON saved to {metrics_path}")
    print(f"Saved k values to {output_dir / 'k_vals.npy'}")
    print(f"Saved target spectra to {output_dir / 'target_pk.npy'}")
    print(f"Saved GAN spectra to {output_dir / 'gan_pk.npy'}")
    print(f"Saved cosmological parameters to {output_dir / 'cosmos_params.npy'}")
    print(f"Power spectrum metadata CSV saved to {metadata_path}")

    del discriminator
    del generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("GAN Evaluation")
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
    parser.add_argument("--add-noise", action="store_true")
    parser.add_argument("--output-dir", type=str, default="evaluations/gan")
    parser.add_argument("--save-images", action="store_true")

    # GAN-specific options
    parser.add_argument("--generator-checkpoint-path", type=str, default=None)
    parser.add_argument("--discriminator-checkpoint-path", type=str, default=None)
    parser.add_argument("--generator-noise-checkpoint-path", type=str, default=None)
    parser.add_argument("--discriminator-noise-checkpoint-path", type=str, default=None)
    parser.add_argument("--gan-g-lr", type=float, default=2e-4)
    parser.add_argument("--gan-d-lr", type=float, default=2e-4)
    parser.add_argument("--gan-beta1", type=float, default=0.5)
    parser.add_argument("--gan-beta2", type=float, default=0.999)
    parser.add_argument("--l1-lambda", type=float, default=100.0)

    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
