import argparse
import collections.abc
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import optax
import pandas as pd
from dotenv import load_dotenv
from flax import nnx
from jax._src.typing import Array
from tqdm.auto import tqdm

import src
import wandb
from src import Discriminator, Generator
from src.typing import Batch, Loader, TransformName
from src.utils import (
    delete_checkpoint,
    make_train_test_loaders,
    make_transform,
    restore_checkpoint,
    save_checkpoint,
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


@nnx.jit
def disc_step(
    disc: Discriminator, opt_disc, gen: Generator, batch: dict[str, jnp.ndarray]
) -> dict[str, jnp.ndarray]:
    inputs, cosmos_params, targets = batch["inputs"], batch["params"], batch["targets"]

    def loss_fn(disc: Discriminator, gen: Generator) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        out_real_logits = disc(
            inputs=inputs,
            output=targets,
            condition_params=cosmos_params,
            is_training=True,
        )

        out_fake_images = gen(inputs, cosmos_params)
        out_fake_images = jax.lax.stop_gradient(out_fake_images)
        out_fake_logits = disc(
            inputs=inputs,
            output=out_fake_images,
            condition_params=cosmos_params,
            is_training=True,
        )
        disc_loss, _, _ = d_hinge_loss(out_real_logits, out_fake_logits)

        disc_real_accuracy = jnp.mean(out_real_logits > 0.0)
        disc_fake_accuracy = jnp.mean(out_fake_logits < 0.0)

        avg_disc_accuracy = (disc_real_accuracy + disc_fake_accuracy) * 0.5

        metrics = {
            "d_loss": disc_loss,
            "d_real_acc": disc_real_accuracy,
            "d_fake_acc": disc_fake_accuracy,
            "d_acc": avg_disc_accuracy,
        }

        return disc_loss, metrics

    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(disc, gen)
    opt_disc.update(model=disc, grads=grads)
    return metrics


@nnx.jit
def gen_step(
    gen: Generator,
    opt_gen,
    disc: Discriminator,
    batch: dict[str, jnp.ndarray],
    l1_lambda: float = 100.0,
) -> dict[str, jnp.ndarray]:
    inputs, cosmos_params, targets = batch["inputs"], batch["params"], batch["targets"]

    def loss_fn(gen: Generator, disc: Discriminator) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        fake_images = gen(inputs, cosmos_params)
        out_fake_logits = disc(
            inputs=inputs,
            output=fake_images,
            condition_params=cosmos_params,
            is_training=True,
        )

        gen_loss, adversarial_loss, reconstruction_loss = g_hinge_l1_loss(
            out_fake_logits, fake_images, targets, l1_lambda
        )

        g_trick_acc = (out_fake_logits > 0.0).mean()

        metrics = {
            "g_loss": gen_loss,
            "g_adversarial": adversarial_loss,
            "g_reconstruct": reconstruction_loss,
            "g_trick_acc": g_trick_acc,
        }
        return gen_loss, metrics

    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(gen, disc)
    opt_gen.update(model=gen, grads=grads)
    return metrics


@nnx.jit
def eval_step(
    disc: Discriminator,
    gen: Generator,
    batch: dict[str, jnp.ndarray],
    l1_lambda: float = 100.0,
) -> dict[str, jnp.ndarray]:
    inputs, cosmos_params, targets = batch["inputs"], batch["params"], batch["targets"]

    out_real_logits = disc(
        inputs=inputs, output=targets, condition_params=cosmos_params, is_training=False
    )
    fake_images = gen(inputs, cosmos_params)
    out_fake_logits = disc(
        inputs=inputs,
        output=fake_images,
        condition_params=cosmos_params,
        is_training=False,
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


def _to_float_dict(metrics: dict[str, jnp.ndarray]) -> dict[str, float]:
    out = {}
    for k, v in metrics.items():
        if isinstance(v, jnp.ndarray):
            out[k] = float(jax.device_get(v))
    return out


def _wandb_images(
    wandb,
    batch: dict[str, jnp.ndarray],
    fake: jnp.ndarray,
    max_items: int = 3,
    transform_name: TransformName = "signed_log1p",
):
    _, inverse_transform = make_transform(name=transform_name)

    def _to_uint8_linear(x: jnp.ndarray):
        # x = inverse_transform(x)
        lo = jnp.percentile(x, 1.0)
        hi = jnp.percentile(x, 99.0)
        x01 = jnp.clip((x - lo) / jnp.maximum(hi - lo, 1e-8), 0.0, 1.0)
        img = (x01 * 255.0).astype(jnp.uint8)

        # --- NEW: JAX -> NumPy for wandb.Image/PIL ---
        img = jax.device_get(img)
        img = np.asarray(img)

        # If it’s HxWx1, make it HxW so PIL treats it as grayscale
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img[..., 0]

        return img

    inputs = batch["inputs"][:max_items]
    targets = batch["targets"][:max_items]
    fake = fake[:max_items]

    imgs = []
    for i in range(min(max_items, inputs.shape[0])):
        imgs.append(wandb.Image(_to_uint8_linear(inputs[i]), caption=f"inputs[{i}] lin"))
        imgs.append(wandb.Image(_to_uint8_linear(targets[i]), caption=f"target[{i}] lin"))
        imgs.append(wandb.Image(_to_uint8_linear(fake[i]), caption=f"fake[{i}] lin"))
    return imgs


def train(
    num_epochs: int,
    batch_size: int,
    train_loader: Loader,
    n_train: int,
    n_test: int,
    test_loader: Loader,
    img_size: int,
    img_channels: int,
    cosmos_params_len: int,
    log_every: int,
    generator: Generator,
    discriminator: Discriminator,
    opt_gen,  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
    opt_disc,  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
    data_key: Array,
    use_wandb: bool,
    args: argparse.Namespace,
    **kwargs,
) -> None:
    train_steps_per_epoch = math.ceil(n_train / batch_size)
    eval_steps_per_epoch = math.ceil(n_test / batch_size)

    cosmos_params_mu = kwargs["cosmos_params_mu"]
    cosmos_params_sigma = kwargs["cosmos_params_sigma"]

    best_disc_acc: float = 0.0
    best_gan_loss: float = jnp.inf

    if use_wandb:
        print("----- Setting up WANDB -----")
        _ = wandb.login()
        wandb.require("core")
        wandb.init(  # pyright: ignore[reportUnusedCallResult]
            entity="aarondinesh2002-epfl",
            project=args.wandb_proj_name,  # pyright: ignore[reportAny]
            name=args.wandb_run_name,  # pyright: ignore[reportAny]
            mode="online",
            config=dict(
                batch_size=batch_size,
                g_lr=args.g_lr,  # pyright: ignore[reportAny]
                d_lr=args.d_lr,  # pyright: ignore[reportAny]
                beta1=args.beta1,  # pyright: ignore[reportAny]
                beta2=args.beta2,  # pyright: ignore[reportAny]
                n_critic=args.n_critic,  # pyright: ignore[reportAny]
                l1_lambda=args.l1_lambda,
                epochs=args.epochs,
                transform_name=args.transform_name,
                img_size=img_size,
                cond_dim=cosmos_params_len,
                train_steps=train_steps_per_epoch,
                eval_steps=eval_steps_per_epoch,
                cosmos_params_mu=cosmos_params_mu,
                cosmos_params_sigma=cosmos_params_sigma,
            ),
        )

    train_key, test_key = random.split(data_key)

    global_step = 0
    for epoch in tqdm(
        range(1, num_epochs + 1),
        total=num_epochs,
        leave=False,
        position=0,
        desc="Running Epoch",
    ):
        # ---------------- TRAIN ----------------
        step = 0
        train_key = random.fold_in(train_key, epoch)
        test_key = random.fold_in(test_key, epoch)

        for batch in tqdm(  # pyright: ignore[reportUnknownVariableType]
            train_loader(key=train_key, drop_last=True),  # pyright: ignore[reportUnknownArgumentType, reportCallIssue]
            total=train_steps_per_epoch,
            leave=False,
            position=1,
            desc=f"Epoch: {epoch:03d} - Training Batch",
        ):
            d_metrics = disc_step(discriminator, opt_disc, generator, batch)  # pyright: ignore[reportUnknownArgumentType, reportAny]

            if (step % args.n_critic) == 0:
                g_metrics = gen_step(
                    generator,
                    opt_gen,  # pyright: ignore[reportUnknownArgumentType]
                    discriminator,
                    batch,  # pyright: ignore[reportUnknownArgumentType]
                    l1_lambda=args.l1_lambda,  # pyright: ignore[reportAny]
                )

            global_step += 1
            step += 1
            if step % log_every == 0 or step == train_steps_per_epoch:
                d_log = {f"train/{k}": v for k, v in _to_float_dict(d_metrics).items()}
                g_log = {f"train/{k}": v for k, v in _to_float_dict(g_metrics).items()}
                log = {"epoch": epoch, "step": global_step} | d_log | g_log
                print(
                    f"\n[Epoch {epoch:03d} Step {step:04d}] "
                    + f"d_loss={d_log.get('train/d_loss', 0.0):.4f} "
                    + f"d_acc={d_log.get('train/d_acc', 0.0):.3f} "
                    + f"| g_loss={g_log.get('train/g_loss', 0.0):.4f} "
                    + f"g_trick_acc={g_log.get('train/g_trick_acc', 0.0):.3f}"
                )
                if use_wandb:
                    wandb.log(log, step=global_step)

        if epoch % 10 == 0:
            save_checkpoint(
                args.checkpoint_dir,
                model=generator,
                optimizer=opt_gen,
                alt_name=f"generator_epoch_{epoch:03d}_gloss_{g_metrics['g_loss']:.03f}",
            )
            save_checkpoint(
                args.checkpoint_dir,
                model=discriminator,
                optimizer=opt_disc,
                alt_name=f"discriminator_epoch_{epoch:03d}_dacc_{d_metrics['d_acc']:.03f}",
            )

        # Accumulate averages across eval steps
        eval_sums = {}
        first_fake = None
        first_batch = None

        for batch in tqdm(
            test_loader(key=test_key, drop_last=True),
            total=eval_steps_per_epoch,
            leave=False,
            position=1,
            desc=f"Epoch {epoch:03d} - Running Eval Batch",
        ):
            metrics = eval_step(discriminator, generator, batch, l1_lambda=args.l1_lambda)  # pyright: ignore[reportAny, reportUnknownArgumentType]
            if first_fake is None:
                first_fake = metrics["sample_fake"]  # pyright: ignore[reportAny]
                first_batch = batch
            # Remove large tensors before averaging
            m = {k: v for k, v in metrics.items() if k != "sample_fake"}  # pyright: ignore[reportAny]
            for k, v in m.items():  # pyright: ignore[reportAny]
                eval_sums[k] = eval_sums.get(k, 0.0) + float(jax.device_get(v))  # pyright: ignore[reportUnknownMemberType, reportAny]

        eval_avg = {k: v / eval_steps_per_epoch for k, v in eval_sums.items()}  # pyright: ignore[reportUnknownVariableType]

        if eval_avg["d_acc"] > best_disc_acc:
            if Path(args.checkpoint_dir, f"BEST_DISC_ACC_acc_{best_disc_acc:.04f}").exists():
                delete_checkpoint(args.checkpoint_dir, f"BEST_DISC_ACC_acc_{best_disc_acc:.04f}")
            best_disc_acc = eval_avg["d_acc"]
            save_checkpoint(
                args.checkpoint_dir,
                epoch,
                global_step,
                discriminator,
                opt_disc,
                alt_name=f"BEST_DISC_ACC_acc_{best_disc_acc:.04f}",
            )

        if eval_avg["g_loss"] < best_gan_loss:
            if Path(args.checkpoint_dir, f"BEST_GAN_LOSS_loss_{best_gan_loss:.04f}").exists():
                delete_checkpoint(args.checkpoint_dir, f"BEST_GAN_LOSS_loss_{best_gan_loss:.04f}")
            best_gan_loss = eval_avg["g_loss"]
            save_checkpoint(
                args.checkpoint_dir,
                epoch,
                global_step,
                generator,
                opt_gen,
                alt_name=f"BEST_GAN_LOSS_loss_{best_gan_loss:.04f}",
            )

        print(
            f"[Eval {epoch:02d}] d_loss={eval_avg['d_loss']:.4f} d_acc={eval_avg['d_acc']:.3f} "
            + f"| g_loss={eval_avg['g_loss']:.4f} g_trick_acc={eval_avg['g_trick_acc']:.3f}"
        )

        if use_wandb:
            wandb.log({f"val/{k}": v for k, v in eval_avg.items()}, step=global_step)
            # Log a small image panel from the first eval batch
            if first_fake is not None and first_batch is not None:
                imgs = _wandb_images(
                    wandb, first_batch, first_fake, transform_name=args.transform_name
                )  # pyright: ignore[reportUnknownVariableType, reportAny]
                wandb.log({"val/examples": imgs}, step=global_step)

    if use_wandb:
        wandb.finish()

    print("Training loop finished.")


def main(parser: argparse.ArgumentParser):
    _ = load_dotenv()
    args = parser.parse_args()

    master_key = random.key(0)
    gen_key, disc_key, data_key, train_test_key = random.split(master_key, 4)  # pyright: ignore[reportAny]

    print("----- Creating Dataset Loaders -----")
    (
        train_loader,
        test_loader,
        n_train,
        n_test,
        img_size,
        cosmos_params_len,
        cosmos_params_mu,
        cosmos_params_sigma,
    ) = make_train_test_loaders(
        key=train_test_key,  # pyright: ignore[reportAny]
        batch_size=args.batch_size,  # pyright: ignore[reportAny]
        input_data_path=args.input_maps,  # pyright: ignore[reportAny]
        output_data_path=args.output_maps,  # pyright: ignore[reportAny]
        csv_path=args.cosmos_params,  # pyright: ignore[reportAny]
        transform_name=args.transform_name,  # pyright: ignore[reportAny]
    )

    print("----- Creating Generator -----")
    generator = src.Generator(
        key=gen_key,
        in_features=args.img_channels,  # pyright: ignore[reportAny]
        out_features=args.img_channels,  # p# pyright: ignore[reportUnusedCallResult]yright: ignore[reportAny]
        len_cosmos_params=cosmos_params_len,
    )

    print("----- Creating Discriminator -----")
    discriminator = src.Discriminator(
        key=disc_key,
        in_channels=args.img_channels,
        condition_dim=cosmos_params_len,
        condition_proj_dim=8,
    )


    params = nnx.state(generator, nnx.Param)
    total_params_gen = sum(jnp.prod(x.shape) for x in jax.tree_util.tree_leaves(params))

    params_disc = nnx.state(discriminator, nnx.Param)
    total_params_disc = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params_disc))

    print(f"Total Gen: {total_params_gen}")
    print(f"Total Disc: {total_params_disc}")

    return

    print("----- Creating Optimizers -----")
    opt_gen = nnx.Optimizer(
        generator,
        optax.adam(args.g_lr, b1=args.beta1, b2=args.beta2),  # pyright: ignore[ reportAny]
        wrt=nnx.Param,
    )
    opt_disc = nnx.Optimizer(
        discriminator,
        optax.adam(args.d_lr, b1=args.beta1, b2=args.beta2),  # pyright: ignore[reportAny]
        wrt=nnx.Param,
    )

    if args.generator_checkpoint_path:
        print(f"----- Loading Generator from {args.generator_checkpoint_path} -----")
        restore_checkpoint(args.generator_checkpoint_path, generator, opt_gen)

    if args.discriminator_checkpoint_path:
        print(f"----- Loading Discriminator from {args.discriminator_checkpoint_path} -----")
        restore_checkpoint(args.discriminator_checkpoint_path, discriminator, opt_disc)

    print("----- Begining Training Run -----")
    train(
        num_epochs=args.epochs,  # pyright: ignore[reportAny]
        batch_size=args.batch_size,  # pyright: ignore[reportAny]
        train_loader=train_loader,
        n_train=n_train,
        test_loader=test_loader,
        n_test=n_test,
        img_size=img_size,
        img_channels=args.img_channels,  # pyright: ignore[reportAny]
        cosmos_params_len=cosmos_params_len,
        log_every=args.log_rate,  # pyright: ignore[reportAny]
        generator=generator,
        discriminator=discriminator,
        opt_gen=opt_gen,
        opt_disc=opt_disc,
        data_key=data_key,  # pyright: ignore[reportAny]
        use_wandb=args.use_wandb,  # pyright: ignore[reportAny]
        args=args,
        cosmos_params_mu=cosmos_params_mu,
        cosmos_params_sigma=cosmos_params_sigma,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training Script")

    parser.add_argument("--batch-size", default=128)  # pyright: ignore[reportUnusedCallResult]
    parser.add_argument("--g-lr", type=float, default=2e-4)  # pyright: ignore[reportUnusedCallResult]
    parser.add_argument("--d-lr", type=float, default=2e-4)  # pyright: ignore[reportUnusedCallResult]
    parser.add_argument("--beta1", type=float, default=0.5)  # pyright: ignore[reportUnusedCallResult]
    parser.add_argument("--beta2", type=float, default=0.999)  # pyright: ignore[reportUnusedCallResult]
    parser.add_argument("--n-critic", type=int, default=1)  # pyright: ignore[reportUnusedCallResult]
    parser.add_argument("--l1-lambda", type=float, default=100)  # pyright: ignore[reportUnusedCallResult]
    parser.add_argument("--transform-name", default="log10")  # pyright: ignore[reportUnusedCallResult]
    parser.add_argument("--epochs", default=150)  # pyright: ignore[reportUnusedCallResult]
    parser.add_argument("--log-rate", default=5)  # pyright: ignore[reportUnusedCallResult]
    parser.add_argument("--input-maps")  # pyright: ignore[reportUnusedCallResult]
    parser.add_argument("--output-maps")  # pyright: ignore[reportUnusedCallResult]
    parser.add_argument("--cosmos-params")  # pyright: ignore[reportUnusedCallResult]
    parser.add_argument("--checkpoint-dir")  # pyright: ignore[reportUnusedCallResult]
    parser.add_argument("--img-channels", default=1)  # pyright: ignore[reportUnusedCallResult]
    parser.add_argument("--use-wandb", action="store_true", default=True)  # pyright: ignore[reportUnusedCallResult]
    parser.add_argument("--wandb-proj-name", default="DRACO")  # pyright: ignore[reportUnusedCallResult]
    parser.add_argument("--wandb-run-name", default="nnx-cgan-256-run-4")  # pyright: ignore[reportUnusedCallResult]
    parser.add_argument("--generator-checkpoint-path", default=None)  # pyright: ignore[reportUnusedCallResult]
    parser.add_argument("--discriminator-checkpoint-path", default=None)  # pyright: ignore[reportUnusedCallResult]

    main(parser)
