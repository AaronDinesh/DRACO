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
    gen_key, disc_key, data_key, train_test_key = random.split(master_key, 4)

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
        add_noise=args.add_noise,
    )

    in_channels = args.img_channels + (1 if args.add_noise else 0)
    generator = Generator(
        key=gen_key,
        in_features=in_channels,
        out_features=args.img_channels,
        len_cosmos_params=cosmos_params_len,
    )
    discriminator = Discriminator(
        key=disc_key,
        input_channels=in_channels,
        condition_dim=cosmos_params_len,
        condition_proj_dim=8,
        target_channels=args.img_channels,
    )

    opt_gen = nnx.Optimizer(
        generator,
        optax.adam(args.g_lr, b1=args.beta1, b2=args.beta2),
        wrt=nnx.Param,
    )
    opt_disc = nnx.Optimizer(
        discriminator,
        optax.adam(args.d_lr, b1=args.beta1, b2=args.beta2),
        wrt=nnx.Param,
    )

    restore_checkpoint(args.generator_checkpoint_path, generator, opt_gen)
    restore_checkpoint(args.discriminator_checkpoint_path, discriminator, opt_disc)

    _train_key, test_key = random.split(data_key)

    total_weight = 0
    metrics_sum: dict[str, float] = {}
    sample_idx = 0

    total_steps = max(1, math.ceil(n_test / args.batch_size))
    eval_iter: Iterable[Batch] = test_loader(key=test_key, drop_last=False)
    for batch in tqdm(eval_iter, total=total_steps, desc="Evaluating GAN", unit="batch"):
        metrics = gan_eval_step(discriminator, generator, batch, l1_lambda=args.l1_lambda)
        fake_images = metrics.pop("sample_fake")
        recon_metrics = batch_metrics(fake_images, batch["targets"])
        power_mse, batch_spectra = _power_spectrum_metrics(
            fake_images,
            batch["targets"],
            args.power_spectrum_bins,
        )
        merged_metrics = {
            **_to_float_dict({k: v for k, v in metrics.items()}),
            **_to_float_dict(recon_metrics),
            "power_spectrum_mse": float(power_mse),
        }

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
    parser = argparse.ArgumentParser("GAN Evaluation Script")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--input-maps", type=str, required=True)
    parser.add_argument("--output-maps", type=str, required=True)
    parser.add_argument("--cosmos-params", type=str, required=True)
    parser.add_argument("--img-channels", type=int, default=1)
    parser.add_argument("--transform-name", type=str, default="log10")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--add-noise", action="store_true")
    parser.add_argument("--generator-checkpoint-path", type=str, required=True)
    parser.add_argument("--discriminator-checkpoint-path", type=str, required=True)
    parser.add_argument("--power-spectrum-bins", type=int, default=64)
    parser.add_argument("--l1-lambda", type=float, default=100.0)
    parser.add_argument("--g-lr", type=float, default=2e-4)
    parser.add_argument("--d-lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--output-dir", type=str, default="evaluations/gan")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
