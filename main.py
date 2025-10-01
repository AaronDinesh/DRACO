import argparse
import collections.abc
import math
import os

import jax
import jax.numpy as jnp
import jax.random as random
import optax
import pandas as pd
import wandb
from dotenv import load_dotenv
from flax import nnx
from jax._src.typing import Array

from src.gan import Discriminator, Generator
from src.typing import Batch, Loader


def avg_pool_2x(x: jnp.ndarray) -> jnp.ndarray:
    """2x2 average pooling with stride 2 for NHWC images."""
    window = (1, 2, 2, 1)
    stride = (1, 2, 2, 1)
    x_sum = jax.lax.reduce_window(x, 0.0, jax.lax.add, window, stride, padding="SAME")
    return x_sum / 4.0


def make_train_test_loaders(
    key: Array,
    batch_size: int,
    input_data_path: str,
    output_data_path: str,
    csv_path: str,
    test_ratio: float = 0.2,
) -> tuple[Loader, Loader, int, int, int, int]:
    input_maps = jnp.load(input_data_path, mmap_mode="r")
    output_maps = jnp.load(output_data_path, mmap_mode="r")
    cosmos_params = jnp.asarray(pd.read_csv(csv_path, header=None).to_numpy())  # pyright: ignore[reportUnknownMemberType]

    assert len(input_maps) == len(cosmos_params), (
        "The length of the input maps do not match the number of cosmos params entried"
    )
    assert len(input_maps) == len(output_maps), (
        "The number of input maps does not match the number of output maps"
    )

    dataset_len = len(input_maps)

    n_test = max(1, int(round(test_ratio * dataset_len)))

    random_shuffle = jax.random.permutation(key=key, x=dataset_len)

    test_idx = random_shuffle[:n_test]
    train_idx = random_shuffle[n_test:]

    # This ensures that there is no data leakage
    assert len(jnp.intersect1d(train_idx, test_idx)) == 0

    def _run_epoch(
        shuffled_idx: jnp.ndarray, key: Array | None, drop_last: bool = False
    ) -> collections.abc.Generator[Batch]:
        if key is not None:
            shuffled_idx = jax.random.permutation(key=key, x=shuffled_idx)

        n = len(shuffled_idx)
        stop = (n // batch_size) * batch_size if drop_last else n

        for s in range(0, stop, batch_size):
            batch = shuffled_idx[s : s + batch_size]

            yield {
                "inputs": input_maps[batch],
                "targets": output_maps[batch],
                "params": cosmos_params[batch],
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
        input_maps[0].shape[1],
        cosmos_params[0].shape[0],
    )


def bce_with_logits(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Mean BCE with logits."""
    return optax.sigmoid_binary_cross_entropy(logits, labels).mean()


@nnx.jit
def d_step(
    disc,
    opt_disc,
    gen,
    batch: dict[str, jnp.ndarray],
) -> dict[str, jnp.ndarray]:
    inputs, cosmos_params, targets = batch["inputs"], batch["params"], batch["targets"]

    def loss_fn(disc) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        # Real
        real256 = targets
        real128 = avg_pool_2x(real256)
        out_real = disc(real256, real128, cosmos_params, is_training=True)
        logits_real = out_real["logits"]  # [B]

        # Fake (stopgrad on G)
        fake256 = gen(inputs, cosmos_params, is_training=True)
        fake256 = jax.lax.stop_gradient(fake256)
        fake128 = avg_pool_2x(fake256)
        out_fake = disc(fake256, fake128, cosmos_params, is_training=True)
        logits_fake = out_fake["logits"]

        loss_real = bce_with_logits(logits_real, jnp.ones_like(logits_real))
        loss_fake = bce_with_logits(logits_fake, jnp.zeros_like(logits_fake))
        d_loss = loss_real + loss_fake

        # accuracies for D
        d_real_acc = (logits_real > 0.0).mean()
        d_fake_acc = (logits_fake < 0.0).mean()
        d_acc = 0.5 * (d_real_acc + d_fake_acc)

        metrics = {
            "d_loss": d_loss,
            "d_real_acc": d_real_acc,
            "d_fake_acc": d_fake_acc,
            "d_acc": d_acc,
            "D_real": jax.nn.sigmoid(logits_real).mean(),
            "D_fake": jax.nn.sigmoid(logits_fake).mean(),
        }
        return d_loss, metrics

    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(disc)
    opt_disc.update(grads)
    return metrics


@nnx.jit
def g_step(
    gen,
    opt_gen,
    disc,
    batch: dict[str, jnp.ndarray],
    l1_lambda: float = 0.0,  # set >0 for pix2pix-like reconstruction
) -> dict[str, jnp.ndarray]:
    inputs, cosmos_params, targets = batch["inputs"], batch["params"], batch["targets"]

    def loss_fn(gen) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        fake256 = gen(inputs, cosmos_params, is_training=True)
        fake128 = avg_pool_2x(fake256)
        out_fake = disc(fake256, fake128, cosmos_params, is_training=True)
        logits_fake = out_fake["logits"]

        adv = bce_with_logits(logits_fake, jnp.ones_like(logits_fake))
        rec = jnp.mean(jnp.abs(fake256 - targets))
        g_loss = adv + l1_lambda * rec

        # "accuracy" proxy for G: how often D predicts fake as real
        g_trick_acc = (logits_fake > 0.0).mean()

        metrics = {"g_loss": g_loss, "g_adv": adv, "g_l1": rec, "g_trick_acc": g_trick_acc}
        return g_loss, metrics

    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(gen)
    opt_gen.update(grads)
    return metrics


@nnx.jit
def eval_step(
    disc: Discriminator,
    gen: Generator,
    batch: dict[str, jnp.ndarray],
    l1_lambda: float = 0.0,
) -> dict[str, jnp.ndarray]:
    inputs, cosmos_params, targets = batch["inputs"], batch["params"], batch["targets"]

    # Real branch
    real256 = targets
    real128 = avg_pool_2x(real256)
    disc_out_real = disc(real256, real128, cosmos_params, is_training=False)
    logits_real = disc_out_real["logits"]

    # Fake branch
    fake256 = gen(inputs, cosmos_params, is_training=False)
    fake128 = avg_pool_2x(fake256)
    disc_out_fake = disc(fake256, fake128, cosmos_params, is_training=False)
    logits_fake = disc_out_fake["logits"]

    loss_real = bce_with_logits(logits_real, jnp.ones_like(logits_real))
    loss_fake = bce_with_logits(logits_fake, jnp.zeros_like(logits_fake))
    d_loss = loss_real + loss_fake

    adv = bce_with_logits(logits_fake, jnp.ones_like(logits_fake))
    rec = jnp.mean(jnp.abs(fake256 - targets))
    g_loss = adv + l1_lambda * rec

    d_real_acc = (logits_real > 0.0).mean()
    d_fake_acc = (logits_fake < 0.0).mean()
    d_acc = 0.5 * (d_real_acc + d_fake_acc)
    g_trick_acc = (logits_fake > 0.0).mean()

    return {
        "d_loss": d_loss,
        "d_real_acc": d_real_acc,
        "d_fake_acc": d_fake_acc,
        "d_acc": d_acc,
        "D_real": jax.nn.sigmoid(logits_real).mean(),
        "D_fake": jax.nn.sigmoid(logits_fake).mean(),
        "g_loss": g_loss,
        "g_adv": adv,
        "g_l1": rec,
        "g_trick_acc": g_trick_acc,
        "sample_fake": fake256,
    }


def _to_float_dict(metrics: dict[str, jnp.ndarray]) -> dict[str, float]:
    out = {}
    for k, v in metrics.items():
        if isinstance(v, jnp.ndarray):
            out[k] = float(jax.device_get(v))
    return out


def _wandb_images(wandb, batch: dict[str, jnp.ndarray], fake: jnp.ndarray, max_items: int = 3):
    """Log a few image triplets (dm, target, fake). Expect values in [-1,1]."""
    inputs = batch["inputs"]
    targets = batch["targets"]
    inputs_np = jax.device_get(inputs[:max_items])
    targets_np = jax.device_get(targets[:max_items])
    fake_np = jax.device_get(fake[:max_items])

    def _to_uint8(x):
        x = (x * 0.5 + 0.5).clip(0.0, 1.0)  # [-1,1] -> [0,1]
        x = (x * 255.0).astype(jnp.uint8)
        return x

    imgs = []
    for i in range(min(max_items, inputs_np.shape[0])):
        imgs.append(wandb.Image(_to_uint8(inputs_np[i]), caption=f"inputs[{i}]"))
        imgs.append(wandb.Image(_to_uint8(targets_np[i]), caption=f"target[{i}]"))
        imgs.append(wandb.Image(_to_uint8(fake_np[i]), caption=f"fake[{i}]"))
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
    generator,  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
    discriminator,  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
    opt_gen,  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
    opt_disc,  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
    data_key: Array,
    use_wandb: bool,
    args: argparse.Namespace,
) -> None:
    train_steps_per_epoch = math.ceil(n_train / batch_size)
    eval_steps_per_epoch = math.ceil(n_test / batch_size)

    if use_wandb:
        _ = wandb.login()
        wandb.require("core")
        wandb.init(  # pyright: ignore[reportUnusedCallResult]
            project=args.wandb_proj_name,  # pyright: ignore[reportAny]
            name=args.wandb_run_name,  # pyright: ignore[reportAny]
            mode="online",
            config=dict(
                batch_size=batch_size,
                lr=args.learning_rate,  # pyright: ignore[reportAny]
                beta1=args.beta_1,  # pyright: ignore[reportAny]
                img_size=img_size,
                cond_dim=cosmos_params_len,
                train_steps=train_steps_per_epoch,
                eval_steps=eval_steps_per_epoch,
            ),
        )

    train_key, test_key = random.split(data_key)

    global_step = 0
    for epoch in range(1, num_epochs + 1):
        # ---------------- TRAIN ----------------
        step = 0
        train_key = random.fold_in(train_key, epoch)
        test_key = random.fold_in(test_key, epoch)

        for batch in train_loader(key=train_key, drop_last=False):  # pyright: ignore[reportCallIssue]
            d_metrics = d_step(discriminator, opt_disc, generator, batch)  # pyright: ignore[reportUnknownArgumentType, reportAny]
            g_metrics = g_step(generator, opt_gen, discriminator, batch)  # pyright: ignore[reportUnknownArgumentType, reportAny]

            global_step += 1
            step += 1
            if step % log_every == 0 or step == train_steps_per_epoch:
                d_log = {f"train/{k}": v for k, v in _to_float_dict(d_metrics).items()}  # pyright: ignore[reportAny]
                g_log = {f"train/{k}": v for k, v in _to_float_dict(g_metrics).items()}  # pyright: ignore[reportAny]
                log = {"epoch": epoch, "step": global_step} | d_log | g_log
                print(
                    f"[Epoch {epoch:02d} Step {step:04d}] "
                    + f"d_loss={d_log.get('train/d_loss', 0.0):.4f} "
                    + f"d_acc={d_log.get('train/d_acc', 0.0):.3f} "
                    + f"| g_loss={g_log.get('train/g_loss', 0.0):.4f} "
                    + f"g_trick_acc={g_log.get('train/g_trick_acc', 0.0):.3f}"
                )
                if use_wandb:
                    wandb.log(log, step=global_step)

        # Accumulate averages across eval steps
        eval_sums = {}
        first_fake = None
        first_batch = None

        for batch in test_loader(key=test_key, drop_last=False):  # pyright: ignore[reportCallIssue]
            metrics = eval_step(discriminator, generator, batch, l1_lambda=0.0)  # pyright: ignore[reportAny, reportUnknownArgumentType]
            if first_fake is None:
                first_fake = metrics["sample_fake"]  # pyright: ignore[reportAny]
                first_batch = batch
            # Remove large tensors before averaging
            m = {k: v for k, v in metrics.items() if k != "sample_fake"}  # pyright: ignore[reportAny]
            for k, v in m.items():  # pyright: ignore[reportAny]
                eval_sums[k] = eval_sums.get(k, 0.0) + float(jax.device_get(v))  # pyright: ignore[reportUnknownMemberType, reportAny]

        eval_avg = {k: v / eval_steps_per_epoch for k, v in eval_sums.items()}  # pyright: ignore[reportUnknownVariableType]

        print(
            f"[Eval {epoch:02d}] d_loss={eval_avg['d_loss']:.4f} d_acc={eval_avg['d_acc']:.3f} "
            + f"| g_loss={eval_avg['g_loss']:.4f} g_trick_acc={eval_avg['g_trick_acc']:.3f}"
        )

        if use_wandb:
            wandb.log({f"val/{k}": v for k, v in eval_avg.items()}, step=global_step)
            # Log a small image panel from the first eval batch
            if first_fake is not None and first_batch is not None:
                imgs = _wandb_images(wandb, first_batch, first_fake)  # pyright: ignore[reportUnknownVariableType, reportAny]
                wandb.log({"val/examples": imgs}, step=global_step)

    if use_wandb:
        wandb.finish()

    print("Training loop finished.")


def main(parser: argparse.ArgumentParser):
    _ = load_dotenv()
    args = parser.parse_args()

    master_key = random.key(0)
    gen_key, disc_key, data_key, train_test_key = random.split(master_key, 4)  # pyright: ignore[reportAny]

    train_loader, test_loader, n_train, n_test, img_size, cosmos_params_len = (
        make_train_test_loaders(
            key=train_test_key,  # pyright: ignore[reportAny]
            batch_size=args.batch_size,  # pyright: ignore[reportAny]
            input_data_path=args.input_maps,  # pyright: ignore[reportAny]
            output_data_path=args.output_maps,  # pyright: ignore[reportAny]
            csv_path=args.cosmos_params,  # pyright: ignore[reportAny]
        )
    )

    generator: Generator = Generator(  # pyright: ignore[reportUnknownVariableType]
        key=gen_key,
        in_features=args.img_channels,  # pyright: ignore[reportAny]
        out_features=args.img_channels,  # pyright: ignore[reportAny]
        len_condition_params=cosmos_params_len,
    )
    discriminator = Discriminator(  # pyright: ignore[reportUnknownVariableType]
        key=disc_key,
        in_features=args.img_channels,  # pyright: ignore[reportAny]
        len_condition_params=cosmos_params_len,
    )

    opt_gen = nnx.Optimizer(  # pyright: ignore[reportUnknownVariableType]
        generator,  # pyright: ignore[reportUnknownArgumentType]
        optax.adam(args.learning_rate, b1=args.beta_1),  # pyright: ignore[ reportAny]
        wrt=nnx.Param,
    )
    opt_disc = nnx.Optimizer(  # pyright: ignore[reportUnknownVariableType]
        discriminator,  # pyright: ignore[reportUnknownArgumentType]
        optax.adam(args.learning_rate, b1=args.beta_1),  # pyright: ignore[reportAny]
        wrt=nnx.Param,
    )

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
        log_every=args.log_every,  # pyright: ignore[reportAny]
        generator=generator,  # pyright: ignore[reportUnknownArgumentType]
        discriminator=discriminator,  # pyright: ignore[reportUnknownArgumentType]
        opt_gen=opt_gen,
        opt_disc=opt_disc,
        data_key=data_key,  # pyright: ignore[reportAny]
        use_wandb=args.use_wandb,  # pyright: ignore[reportAny]
        args=args,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training Script")

    parser.add_argument("--batch-size", default=8)  # pyright: ignore[reportUnusedCallResult]
    parser.add_argument("--learning-rate", deafult=2e-4)  # pyright: ignore[reportUnusedCallResult]
    parser.add_argument("--beta-1", default=0.5)  # pyright: ignore[reportUnusedCallResult]
    parser.add_argument("--epochs", default=5)  # pyright: ignore[reportUnusedCallResult]
    parser.add_argument("--log-rate", default=20)  # pyright: ignore[reportUnusedCallResult]
    parser.add_argument("--input-maps")  # pyright: ignore[reportUnusedCallResult]
    parser.add_argument("--output-maps")  # pyright: ignore[reportUnusedCallResult]
    parser.add_argument("--cosmos-params")  # pyright: ignore[reportUnusedCallResult]
    parser.add_argument("--img-channels", default=1)  # pyright: ignore[reportUnusedCallResult]
    parser.add_argument("--use-wandb", action="store_true", default=True)  # pyright: ignore[reportUnusedCallResult]
    parser.add_argument("--wandb-proj-name", default="astro-gan")  # pyright: ignore[reportUnusedCallResult]
    parser.add_argument("--wandb-run-name", default="nnx-cgan-256")  # pyright: ignore[reportUnusedCallResult]

    main(parser)
