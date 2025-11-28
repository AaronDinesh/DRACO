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

from GAN_train import make_train_test_loaders
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


def _plot_and_save_power_spectrum(
    target: jnp.ndarray,
    gan_pred: jnp.ndarray,
    bins: int,
    sample_idx: int,
    spectra_dir: Path,
    field_name: str,
) -> float:
    curves = {
        "Target": _power_spectrum_curve(target, bins),
        "GAN": _power_spectrum_curve(gan_pred, bins),
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    for label, (k_vals, pk_vals) in curves.items():
        ax.loglog(k_vals, pk_vals, label=label)
    ax.set_title(f"GAN Power Spectrum ({field_name}) Sample {sample_idx:05d}")
    ax.set_xlabel("Wave number k [h/Mpc]")
    ax.set_ylabel("P(k)")
    ax.legend()
    fig.tight_layout()
    save_path = spectra_dir / f"sample_{sample_idx:05d}.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    gan_mse = _spectrum_mse(curves["GAN"][1], curves["Target"][1])
    return gan_mse


def evaluate(args: argparse.Namespace) -> None:
    _ = load_dotenv()

    output_dir = Path(args.output_dir)
    spectra_dir = output_dir / "power_spectra"
    spectra_dir.mkdir(parents=True, exist_ok=True)

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
        if args.add_noise and args.generator_noise_checkpoint_path
        else args.generator_checkpoint_path
    )
    disc_ckpt_path = (
        args.discriminator_noise_checkpoint_path
        if args.add_noise and args.discriminator_noise_checkpoint_path
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

    total_steps = max(1, math.ceil(n_test / args.batch_size))
    total_weight = 0
    sample_idx = 0
    gan_metrics_sum: dict[str, float] = {}
    gan_power_error = 0.0

    eval_iter: Iterable[Batch] = test_loader(key=data_key, drop_last=False)
    for batch in tqdm(eval_iter, total=total_steps, desc="Evaluating GAN", unit="batch"):
        batch_size = int(batch["inputs"].shape[0])
        noise_key = None
        if args.add_noise:
            data_key, noise_key = random.split(data_key)
        gan_inputs = _maybe_add_noise(batch["inputs"], args.add_noise, noise_key)
        gan_batch = {**batch, "inputs": gan_inputs}

        gan_metrics = gan_eval_step(discriminator, generator, gan_batch, l1_lambda=args.l1_lambda)
        fake_images = gan_metrics.pop("sample_fake")
        recon_metrics = batch_metrics(fake_images, batch["targets"])
        gan_log = {**_to_float_dict(gan_metrics), **_to_float_dict(recon_metrics)}

        _accumulate(gan_metrics_sum, gan_log, batch_size)
        total_weight += batch_size

        for offset in range(batch_size):
            gan_mse = _plot_and_save_power_spectrum(
                target=batch["targets"][offset],
                gan_pred=fake_images[offset],
                bins=args.power_spectrum_bins,
                sample_idx=sample_idx + offset,
                spectra_dir=spectra_dir,
                field_name=args.field_name,
            )
            gan_power_error += gan_mse
        sample_idx += batch_size

    if total_weight == 0:
        raise RuntimeError("Evaluation dataset is empty")

    gan_metrics_avg = {k: v / total_weight for k, v in gan_metrics_sum.items()}
    divisor = max(sample_idx, 1)
    gan_metrics_avg["power_spectrum_mse"] = gan_power_error / divisor

    results = {"gan": gan_metrics_avg}
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("GAN evaluation complete. Aggregated metrics:")
    for key, value in gan_metrics_avg.items():
        print(f"  {key}: {value:.6f}")
    print(f"Saved per-sample power spectra to {spectra_dir}")
    print(f"Metrics JSON saved to {metrics_path}")

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

    # GAN-specific options
    parser.add_argument("--generator-checkpoint-path", type=str, required=True)
    parser.add_argument("--discriminator-checkpoint-path", type=str, required=True)
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
