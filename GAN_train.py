import argparse
import collections.abc
import math
from collections import deque
from pathlib import Path
from typing import Any

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
    batch_metrics,
    delete_checkpoint,
    make_transform,
    restore_checkpoint,
    save_checkpoint,
)


@jax.jit
def _append_noise_channel(inputs: jnp.ndarray, key: Array) -> jnp.ndarray:
    sigma = 0.5
    noise = sigma * random.normal(key, shape=inputs.shape[:-1] + (1,), dtype=inputs.dtype)
    return jnp.concatenate((inputs, noise), axis=-1)


def _maybe_add_noise(inputs: jnp.ndarray, add_noise: bool, key: Array | None) -> jnp.ndarray:
    if not add_noise:
        return inputs
    if key is None:
        raise ValueError("Noise-enabled training requires a PRNG key")
    return _append_noise_channel(inputs, key)


def _prefetch_to_device(
    iterator: collections.abc.Iterator[Batch], prefetch_size: int
) -> collections.abc.Iterator[Batch]:
    if prefetch_size <= 0:
        yield from iterator
        return

    queue: deque[Batch] = deque()
    try:
        for _ in range(prefetch_size):
            queue.append(jax.device_put(next(iterator)))
    except StopIteration:
        pass

    while queue:
        batch = queue.popleft()
        yield batch
        try:
            queue.append(jax.device_put(next(iterator)))
        except StopIteration:
            continue


def make_train_test_loaders(
    key: Array,
    batch_size: int,
    input_data_path: str,
    output_data_path: str,
    csv_path: str,
    test_ratio: float = 0.2,
    transform_name: TransformName = "signed_log1p",
):
    input_maps_np = np.load(input_data_path, mmap_mode="r")
    output_maps_np = np.load(output_data_path, mmap_mode="r")
    cosmos_params_df = pd.read_csv(csv_path, header=None, sep=" ")

    repeat_factor: int = input_maps_np.shape[0] // len(cosmos_params_df)
    cosmos_params_df = cosmos_params_df.loc[
        cosmos_params_df.index.repeat(repeat_factor)
    ].reset_index(drop=True)
    cosmos_params = jnp.asarray(cosmos_params_df.to_numpy(), dtype=jnp.float32)

    assert len(input_maps_np) == len(cosmos_params), (
        f"The length of the input maps {input_maps_np.shape} do not match the number of cosmos params entries {cosmos_params.shape}"
    )
    assert len(input_maps_np) == len(output_maps_np), (
        "The number of input maps does not match the number of output maps"
    )

    def _add_channel_last(x: np.ndarray):
        return x[..., None] if x.ndim == 3 else x

    forward_transform, _ = make_transform(name=transform_name)
    input_maps = forward_transform(jnp.asarray(_add_channel_last(np.asarray(input_maps_np))))
    output_maps = forward_transform(jnp.asarray(_add_channel_last(np.asarray(output_maps_np))))
    input_maps = input_maps.astype(jnp.float32)
    output_maps = output_maps.astype(jnp.float32)

    dataset_len = input_maps.shape[0]
    n_test = max(1, int(round(test_ratio * dataset_len)))

    random_shuffle = jax.random.permutation(key=key, x=dataset_len)
    test_idx = random_shuffle[:n_test]
    train_idx = random_shuffle[n_test:]

    mu = jnp.mean(cosmos_params[train_idx], axis=0)
    sigma = jnp.std(cosmos_params[train_idx], axis=0) + 1e-6
    standardized_params = ((cosmos_params - mu) / sigma).astype(jnp.float32)

    assert len(jnp.intersect1d(train_idx, test_idx)) == 0

    def _run_epoch(
        shuffled_idx: jnp.ndarray, key: Array | None, drop_last: bool = False
    ) -> collections.abc.Generator[Batch]:
        if key is not None:
            key, perm_key = random.split(key)
            shuffled_idx = random.permutation(perm_key, shuffled_idx)

        n = len(shuffled_idx)
        stop = (n // batch_size) * batch_size if drop_last else n

        for s in range(0, stop, batch_size):
            batch = shuffled_idx[s : s + batch_size]
            yield {
                "inputs": input_maps[batch],
                "targets": output_maps[batch],
                "params": standardized_params[batch],
            }

    def train_loader(key: Array | None = None, drop_last: bool = False):
        return _run_epoch(train_idx, key, drop_last)

    def test_loader(key: Array | None = None, drop_last: bool = False):
        return _run_epoch(test_idx, key, drop_last)

    return (
        train_loader,
        test_loader,
        len(train_idx),
        len(test_idx),
        input_maps.shape[1],
        cosmos_params.shape[1],
        mu,
        sigma,
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
        metrics["sample_fake"] = jax.lax.stop_gradient(fake_images)
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
    out: dict[str, float] = {}
    for k, v in metrics.items():
        if isinstance(v, jnp.ndarray):
            out[k] = float(jax.device_get(v))
        elif isinstance(v, (float, int)):
            out[k] = float(v)
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

    # If inputs have multiple channels (e.g., map + noise),
    # visualize only the first channel for interpretability.
    if inputs.ndim == 4 and inputs.shape[-1] > 1:
        inputs_vis = inputs[..., :1]
    else:
        inputs_vis = inputs

    imgs = []
    for i in range(min(max_items, inputs_vis.shape[0])):
        imgs.append(wandb.Image(_to_uint8_linear(inputs_vis[i]), caption=f"inputs[{i}] lin"))
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
    start_epoch: int = 1,
    start_step: int = 0,
    resume_wandb_id: str | None = None,
    **kwargs,
) -> None:
    # With drop_last=True in the loaders, the true number of steps is
    # floor(n / batch_size); use that so progress bars and eval
    # averaging match the actual number of batches.
    train_steps_per_epoch = max(1, n_train // batch_size)
    eval_steps_per_epoch = max(1, n_test // batch_size)

    cosmos_params_mu = kwargs["cosmos_params_mu"]
    cosmos_params_sigma = kwargs["cosmos_params_sigma"]
    data_stats = {
        "cosmos_params_mu": cosmos_params_mu,
        "cosmos_params_sigma": cosmos_params_sigma,
    }

    best_disc_acc: float = 0.0
    best_gan_loss: float = jnp.inf

    run_id = resume_wandb_id
    if use_wandb:
        print("----- Setting up WANDB -----")
        _ = wandb.login()
        wandb.require("core")
        wandb_init_kwargs = dict(
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
        if run_id is not None:
            wandb_init_kwargs["id"] = run_id
            wandb_init_kwargs["resume"] = "allow"

        run = wandb.init(**wandb_init_kwargs)  # pyright: ignore[reportUnusedCallResult]
        run_id = run.id if run is not None else run_id

    train_key, test_key = random.split(data_key)

    global_step = start_step
    last_epoch = start_epoch - 1
    for epoch in tqdm(
        range(start_epoch, num_epochs + 1),
        total=max(0, num_epochs - start_epoch + 1),
        leave=False,
        position=0,
        desc="Running Epoch",
    ):
        last_epoch = epoch
        # ---------------- TRAIN ----------------
        step = 0
        train_key = random.fold_in(train_key, epoch)
        test_key = random.fold_in(test_key, epoch)

        train_key, loader_key = random.split(train_key)
        train_iterator = _prefetch_to_device(
            train_loader(key=loader_key, drop_last=True),
            int(args.prefetch_size),
        )

        for batch in tqdm(
            train_iterator,
            total=train_steps_per_epoch,
            leave=False,
            position=1,
            desc=f"Epoch: {epoch:03d} - Training Batch",
        ):
            noise_key = None
            if args.add_noise:
                train_key, noise_key = random.split(train_key)
            prepared_inputs = _maybe_add_noise(batch["inputs"], args.add_noise, noise_key)
            batch = {**batch, "inputs": prepared_inputs}

            d_metrics = disc_step(discriminator, opt_disc, generator, batch)  # pyright: ignore[reportUnknownArgumentType, reportAny]

            if (step % args.n_critic) == 0:
                g_metrics = gen_step(
                    generator,
                    opt_gen,  # pyright: ignore[reportUnknownArgumentType]
                    discriminator,
                    batch,  # pyright: ignore[reportUnknownArgumentType]
                    l1_lambda=float(args.l1_lambda),  # pyright: ignore[reportAny]
                )
                fake_images = g_metrics.pop("sample_fake", None)
                if fake_images is not None:
                    recon_metrics = batch_metrics(fake_images, batch["targets"])
                    g_metrics = {
                        **g_metrics,
                        **recon_metrics,
                    }

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
                epoch=epoch,
                step=global_step,
                model=generator,
                optimizer=opt_gen,
                alt_name=f"generator_epoch_{epoch:03d}_gloss_{g_metrics['g_loss']:.03f}",
                data_stats=data_stats,
                wandb_run_id=run_id,
            )
            save_checkpoint(
                args.checkpoint_dir,
                epoch=epoch,
                step=global_step,
                model=discriminator,
                optimizer=opt_disc,
                alt_name=f"discriminator_epoch_{epoch:03d}_dacc_{d_metrics['d_acc']:.03f}",
                data_stats=data_stats,
                wandb_run_id=run_id,
            )

        # Accumulate averages across eval steps
        eval_sums = {}
        first_fake = None
        first_batch = None

        test_key, eval_loader_key = random.split(test_key)
        eval_iterator = _prefetch_to_device(
            test_loader(key=eval_loader_key, drop_last=True),
            int(args.prefetch_size),
        )

        for batch in tqdm(
            eval_iterator,
            total=eval_steps_per_epoch,
            leave=False,
            position=1,
            desc=f"Epoch {epoch:03d} - Running Eval Batch",
        ):
            eval_noise_key = None
            if args.add_noise:
                test_key, eval_noise_key = random.split(test_key)
            prepared_inputs = _maybe_add_noise(batch["inputs"], args.add_noise, eval_noise_key)
            batch = {**batch, "inputs": prepared_inputs}
            metrics = eval_step(discriminator, generator, batch, l1_lambda=float(args.l1_lambda))  # pyright: ignore[reportAny, reportUnknownArgumentType]
            if first_fake is None:
                first_fake = metrics["sample_fake"]  # pyright: ignore[reportAny]
                first_batch = batch
            fake_images = metrics["sample_fake"]  # pyright: ignore[reportAny]

            recon_metrics = batch_metrics(fake_images, batch["targets"])
            # Remove large tensors before averaging
            m = {
                **{k: v for k, v in metrics.items() if k != "sample_fake"},  # pyright: ignore[reportAny]
                **recon_metrics,
            }
            for k, v in m.items():  # pyright: ignore[reportAny]
                eval_sums[k] = eval_sums.get(k, 0.0) + float(jax.device_get(v))  # pyright: ignore[reportUnknownMemberType, reportAny]

        eval_avg = {k: v / eval_steps_per_epoch for k, v in eval_sums.items()}  # pyright: ignore[reportUnknownVariableType]

        if eval_avg["d_acc"] > best_disc_acc:
            if Path(args.checkpoint_dir, f"BEST_DISC_ACC_acc_{best_disc_acc:.04f}").exists():
                delete_checkpoint(args.checkpoint_dir, f"BEST_DISC_ACC_acc_{best_disc_acc:.04f}")
            best_disc_acc = eval_avg["d_acc"]
            save_checkpoint(
                args.checkpoint_dir,
                epoch=epoch,
                step=global_step,
                model=discriminator,
                optimizer=opt_disc,
                alt_name=f"BEST_DISC_ACC_acc_{best_disc_acc:.04f}",
                data_stats=data_stats,
                wandb_run_id=run_id,
            )

        if eval_avg["g_loss"] < best_gan_loss:
            if Path(args.checkpoint_dir, f"BEST_GAN_LOSS_loss_{best_gan_loss:.04f}").exists():
                delete_checkpoint(args.checkpoint_dir, f"BEST_GAN_LOSS_loss_{best_gan_loss:.04f}")
            best_gan_loss = eval_avg["g_loss"]
            save_checkpoint(
                args.checkpoint_dir,
                epoch=epoch,
                step=global_step,
                model=generator,
                optimizer=opt_gen,
                alt_name=f"BEST_GAN_LOSS_loss_{best_gan_loss:.04f}",
                data_stats=data_stats,
                wandb_run_id=run_id,
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
                    wandb,
                    first_batch,
                    first_fake,
                    transform_name=args.transform_name,
                )  # pyright: ignore[reportUnknownVariableType, reportAny]
                wandb.log({"val/examples": imgs}, step=global_step)

    if use_wandb:
        wandb.finish()

    save_checkpoint(
        args.checkpoint_dir,
        epoch=last_epoch,
        step=global_step,
        model=generator,
        optimizer=opt_gen,
        alt_name=f"Final_Generator",
        data_stats=data_stats,
        wandb_run_id=run_id,
        wait=True,
    )

    save_checkpoint(
        args.checkpoint_dir,
        epoch=last_epoch,
        step=global_step,
        model=discriminator,
        optimizer=opt_disc,
        alt_name=f"Final_Discriminator",
        data_stats=data_stats,
        wandb_run_id=run_id,
        wait=True,
    )

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
        batch_size=int(args.batch_size),  # pyright: ignore[reportAny]
        input_data_path=args.input_maps,  # pyright: ignore[reportAny]
        output_data_path=args.output_maps,  # pyright: ignore[reportAny]
        csv_path=args.cosmos_params,  # pyright: ignore[reportAny]
        transform_name=args.transform_name,  # pyright: ignore[reportAny]
    )

    print("----- Creating Generator -----")
    input_features_size = args.img_channels + 1 if args.add_noise else args.img_channels
    generator = src.Generator(
        key=gen_key,
        in_features=input_features_size,  # pyright: ignore[reportAny]
        out_features=int(
            args.img_channels
        ),  # p# pyright: ignore[reportUnusedCallResult]yright: ignore[reportAny]
        len_cosmos_params=cosmos_params_len,
    )

    print("----- Creating Discriminator -----")
    discriminator = src.Discriminator(
        key=disc_key,
        input_channels=input_features_size,
        condition_dim=cosmos_params_len,
        condition_proj_dim=8,
        target_channels=int(args.img_channels),
    )

    # params = nnx.state(generator, nnx.Param)
    # total_params_gen = sum(jnp.prod(x.shape) for x in jax.tree_util.tree_leaves(params))

    # params_disc = nnx.state(discriminator, nnx.Param)
    # total_params_disc = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params_disc))

    # print(f"Total Gen: {total_params_gen}")
    # print(f"Total Disc: {total_params_disc}")

    print("----- Creating Optimizers -----")
    opt_gen = nnx.Optimizer(
        generator,
        optax.adam(float(args.g_lr), b1=float(args.beta1), b2=float(args.beta2)),  # pyright: ignore[ reportAny]
        wrt=nnx.Param,
    )
    opt_disc = nnx.Optimizer(
        discriminator,
        optax.adam(float(args.d_lr), b1=float(args.beta1), b2=float(args.beta2)),  # pyright: ignore[reportAny]
        wrt=nnx.Param,
    )

    start_epoch = 1
    start_step = 0
    resume_wandb_id: str | None = None

    if args.generator_checkpoint_path:
        print(f"----- Loading Generator from {args.generator_checkpoint_path} -----")
        gen_ckpt = restore_checkpoint(args.generator_checkpoint_path, generator, opt_gen)
        gen_epoch = gen_ckpt.get("epoch")
        gen_step = gen_ckpt.get("step")
        start_epoch = max(start_epoch, (int(gen_epoch) if gen_epoch is not None else 0) + 1)
        start_step = max(start_step, int(gen_step) if gen_step is not None else 0)
        resume_wandb_id = resume_wandb_id or gen_ckpt.get("wandb_run_id")

    if args.discriminator_checkpoint_path:
        print(f"----- Loading Discriminator from {args.discriminator_checkpoint_path} -----")
        disc_ckpt = restore_checkpoint(args.discriminator_checkpoint_path, discriminator, opt_disc)
        disc_epoch = disc_ckpt.get("epoch")
        disc_step = disc_ckpt.get("step")
        start_epoch = max(start_epoch, (int(disc_epoch) if disc_epoch is not None else 0) + 1)
        start_step = max(start_step, int(disc_step) if disc_step is not None else 0)
        resume_wandb_id = resume_wandb_id or disc_ckpt.get("wandb_run_id")

    if start_epoch > 1 or start_step > 0:
        print(f"----- Resuming from epoch {start_epoch} (global step {start_step}) -----")

    print("----- Begining Training Run -----")
    train(
        num_epochs=int(args.epochs),  # pyright: ignore[reportAny]
        batch_size=int(args.batch_size),  # pyright: ignore[reportAny]
        train_loader=train_loader,
        n_train=n_train,
        test_loader=test_loader,
        n_test=n_test,
        img_size=img_size,
        img_channels=int(args.img_channels),  # pyright: ignore[reportAny]
        cosmos_params_len=cosmos_params_len,
        log_every=int(args.log_rate),  # pyright: ignore[reportAny]
        generator=generator,
        discriminator=discriminator,
        opt_gen=opt_gen,
        opt_disc=opt_disc,
        data_key=data_key,  # pyright: ignore[reportAny]
        use_wandb=args.use_wandb,  # pyright: ignore[reportAny]
        args=args,
        cosmos_params_mu=cosmos_params_mu,
        cosmos_params_sigma=cosmos_params_sigma,
        start_epoch=start_epoch,
        start_step=start_step,
        resume_wandb_id=resume_wandb_id,
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
    parser.add_argument("--prefetch-size", type=int, default=2)  # pyright: ignore[reportUnusedCallResult]
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
    parser.add_argument("--add-noise", action="store_true", default=False)  # pyright: ignore[reportUnusedCallResult]
    main(parser)
