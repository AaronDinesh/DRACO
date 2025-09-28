from __future__ import annotations

from collections.abc import Iterator

import jax
import jax.numpy as jnp
import jax.random as random
import optax
from flax import nnx

from src.gan import Discriminator, Generator

# -------------------
# 1. Hyperparameters
# -------------------
BATCH_SIZE = 8
LEARNING_RATE = 2e-4
BETA_1 = 0.5  # Recommended for Adam in GANs
NUM_EPOCHS = 5
TRAIN_STEPS_PER_EPOCH = 200
EVAL_STEPS_PER_EPOCH = 50
LOG_EVERY = 20

# Field properties
IMG_SIZE = 256
IMG_CHANNELS = 1
CONDITION_PARAMS_LEN = 10

# Logging
USE_WANDB = True
WANDB_PROJECT = "astro-gan"
WANDB_RUN_NAME = "nnx-cgan-256"
WANDB_MODE = "online"  # "disabled" for no-upload environments


# -------------------
# 2. Utilities
# -------------------


def avg_pool_2x(x: jnp.ndarray) -> jnp.ndarray:
    """2x2 average pooling with stride 2 for NHWC images."""
    window = (1, 2, 2, 1)
    stride = (1, 2, 2, 1)
    x_sum = jax.lax.reduce_window(x, 0.0, jax.lax.add, window, stride, padding="SAME")
    return x_sum / 4.0


def make_dummy_loaders(
    key: jax.Array,
    batch_size: int,
    steps_per_epoch: int,
    img_size: int,
    channels: int,
    cond_dim: int,
) -> tuple[Iterator[dict[str, jnp.ndarray]], Iterator[dict[str, jnp.ndarray]]]:
    """
    Returns (train_loader, eval_loader) that yield dicts with:
      - 'dm': input dark-matter map in [-1,1], shape [B,H,W,C]
      - 'theta': conditioning vector, shape [B,cond_dim]
      - 'target': target astro/baryon map in [-1,1], shape [B,H,W,C]
    """

    def make_epoch(seed: int, steps: int) -> Iterator[dict[str, jnp.ndarray]]:
        k = random.PRNGKey(seed)
        for _ in range(steps):
            k, k_dm, k_t, k_theta = random.split(k, 4)
            dm = random.uniform(
                k_dm, (batch_size, img_size, img_size, channels), minval=-1.0, maxval=1.0
            )
            theta = random.normal(k_theta, (batch_size, cond_dim))
            # dummy target: blurred version + noise to mimic structure
            noise = 0.05 * random.normal(k_t, (batch_size, img_size, img_size, channels))
            lp = avg_pool_2x(dm)
            lp = jnp.repeat(jnp.repeat(lp, 2, axis=1), 2, axis=2)
            target = jnp.clip(dm * 0.6 + lp * 0.4 + noise, -1.0, 1.0)
            yield {"dm": dm, "theta": theta, "target": target}

    train_loader = make_epoch(seed=0, steps=steps_per_epoch)
    eval_loader = make_epoch(seed=1, steps=EVAL_STEPS_PER_EPOCH)
    return train_loader, eval_loader


# -------------------
# 3. Model and Optimizers
# -------------------

master_key = random.key(0)
gen_key, disc_key, data_key = random.split(master_key, 3)

generator = Generator(
    key=gen_key,
    in_features=IMG_CHANNELS,
    out_features=IMG_CHANNELS,
    len_condition_params=CONDITION_PARAMS_LEN,
)
discriminator = Discriminator(
    key=disc_key, in_features=IMG_CHANNELS, len_condition_params=CONDITION_PARAMS_LEN
)

opt_gen = nnx.Optimizer(generator, optax.adam(LEARNING_RATE, b1=BETA_1), wrt=nnx.Param)
opt_disc = nnx.Optimizer(discriminator, optax.adam(LEARNING_RATE, b1=BETA_1), wrt=nnx.Param)


def bce_with_logits(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Mean BCE with logits."""
    return optax.sigmoid_binary_cross_entropy(logits, labels).mean()


@nnx.jit
def d_step(
    disc: Discriminator,
    opt_disc: nnx.Optimizer,
    gen: Generator,
    batch: dict[str, jnp.ndarray],
) -> dict[str, jnp.ndarray]:
    dm, theta, target = batch["dm"], batch["theta"], batch["target"]

    def loss_fn(disc: Discriminator) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        # Real
        real256 = target
        real128 = avg_pool_2x(real256)
        out_real = disc(real256, real128, theta, is_training=True)
        logits_real = out_real["logits"]  # [B]

        # Fake (stopgrad on G)
        fake256 = gen(dm, theta, is_training=True)
        fake256 = jax.lax.stop_gradient(fake256)
        fake128 = avg_pool_2x(fake256)
        out_fake = disc(fake256, fake128, theta, is_training=True)
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
    gen: Generator,
    opt_gen: nnx.Optimizer,
    disc: Discriminator,
    batch: dict[str, jnp.ndarray],
    l1_lambda: float = 0.0,  # set >0 for pix2pix-like reconstruction
) -> dict[str, jnp.ndarray]:
    dm, theta, target = batch["dm"], batch["theta"], batch["target"]

    def loss_fn(gen: Generator) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        fake256 = gen(dm, theta, is_training=True)
        fake128 = avg_pool_2x(fake256)
        out_fake = disc(fake256, fake128, theta, is_training=True)
        logits_fake = out_fake["logits"]

        adv = bce_with_logits(logits_fake, jnp.ones_like(logits_fake))
        rec = jnp.mean(jnp.abs(fake256 - target))
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
    dm, theta, target = batch["dm"], batch["theta"], batch["target"]

    # Real branch
    real256 = target
    real128 = avg_pool_2x(real256)
    o_r = disc(real256, real128, theta, is_training=False)
    lr = o_r["logits"]

    # Fake branch
    fake256 = gen(dm, theta, is_training=False)
    fake128 = avg_pool_2x(fake256)
    o_f = disc(fake256, fake128, theta, is_training=False)
    lf = o_f["logits"]

    loss_real = bce_with_logits(lr, jnp.ones_like(lr))
    loss_fake = bce_with_logits(lf, jnp.zeros_like(lf))
    d_loss = loss_real + loss_fake

    adv = bce_with_logits(lf, jnp.ones_like(lf))
    rec = jnp.mean(jnp.abs(fake256 - target))
    g_loss = adv + l1_lambda * rec

    d_real_acc = (lr > 0.0).mean()
    d_fake_acc = (lf < 0.0).mean()
    d_acc = 0.5 * (d_real_acc + d_fake_acc)
    g_trick_acc = (lf > 0.0).mean()

    return {
        "d_loss": d_loss,
        "d_real_acc": d_real_acc,
        "d_fake_acc": d_fake_acc,
        "d_acc": d_acc,
        "D_real": jax.nn.sigmoid(lr).mean(),
        "D_fake": jax.nn.sigmoid(lf).mean(),
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


def _maybe_wandb_init():
    if not USE_WANDB:
        return None
    try:
        import wandb  # type: ignore

        wandb.require("core")
        wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_RUN_NAME,
            mode=WANDB_MODE,
            config=dict(
                batch_size=BATCH_SIZE,
                lr=LEARNING_RATE,
                beta1=BETA_1,
                img_size=IMG_SIZE,
                cond_dim=CONDITION_PARAMS_LEN,
                train_steps=TRAIN_STEPS_PER_EPOCH,
                eval_steps=EVAL_STEPS_PER_EPOCH,
            ),
        )
        return wandb
    except Exception as e:
        print(f"[WARN] wandb not available or failed to init: {e}")
        return None


def _wandb_images(wandb, batch: dict[str, jnp.ndarray], fake: jnp.ndarray, max_items: int = 3):
    """Log a few image triplets (dm, target, fake). Expect values in [-1,1]."""
    dm = batch["dm"]
    tgt = batch["target"]
    dm_np = jax.device_get(dm[:max_items])
    tgt_np = jax.device_get(tgt[:max_items])
    fake_np = jax.device_get(fake[:max_items])

    def _to_uint8(x):
        x = (x * 0.5 + 0.5).clip(0.0, 1.0)  # [-1,1] -> [0,1]
        x = (x * 255.0).astype(jnp.uint8)
        return x

    imgs = []
    for i in range(min(max_items, dm_np.shape[0])):
        imgs.append(wandb.Image(_to_uint8(dm_np[i]), caption=f"dm[{i}]"))
        imgs.append(wandb.Image(_to_uint8(tgt_np[i]), caption=f"target[{i}]"))
        imgs.append(wandb.Image(_to_uint8(fake_np[i]), caption=f"fake[{i}]"))
    return imgs


def train(
    num_epochs: int = NUM_EPOCHS,
    train_steps_per_epoch: int = TRAIN_STEPS_PER_EPOCH,
    eval_steps_per_epoch: int = EVAL_STEPS_PER_EPOCH,
) -> None:
    wandb = _maybe_wandb_init()

    train_loader, eval_loader = make_dummy_loaders(
        data_key, BATCH_SIZE, train_steps_per_epoch, IMG_SIZE, IMG_CHANNELS, CONDITION_PARAMS_LEN
    )

    global_step = 0
    for epoch in range(1, num_epochs + 1):
        # ---------------- TRAIN ----------------
        for step, batch in enumerate(train_loader, start=1):
            d_metrics = d_step(discriminator, opt_disc, generator, batch)
            g_metrics = g_step(generator, opt_gen, discriminator, batch)

            global_step += 1

            if step % LOG_EVERY == 0 or step == train_steps_per_epoch:
                d_log = {f"train/{k}": v for k, v in _to_float_dict(d_metrics).items()}
                g_log = {f"train/{k}": v for k, v in _to_float_dict(g_metrics).items()}
                log = {"epoch": epoch, "step": global_step} | d_log | g_log
                print(
                    f"[Epoch {epoch:02d} Step {step:04d}] "
                    f"d_loss={d_log.get('train/d_loss', 0.0):.4f} "
                    f"d_acc={d_log.get('train/d_acc', 0.0):.3f} "
                    f"| g_loss={g_log.get('train/g_loss', 0.0):.4f} "
                    f"g_trick_acc={g_log.get('train/g_trick_acc', 0.0):.3f}"
                )
                if wandb is not None:
                    wandb.log(log, step=global_step)

        # Accumulate averages across eval steps
        eval_sums = {}
        first_fake = None
        first_batch = None
        for step, batch in enumerate(eval_loader, start=1):
            metrics = eval_step(discriminator, generator, batch, l1_lambda=0.0)
            if first_fake is None:
                first_fake = metrics["sample_fake"]
                first_batch = batch
            # Remove large tensors before averaging
            m = {k: v for k, v in metrics.items() if k != "sample_fake"}
            for k, v in m.items():
                eval_sums[k] = eval_sums.get(k, 0.0) + float(jax.device_get(v))

        eval_avg = {k: v / eval_steps_per_epoch for k, v in eval_sums.items()}

        print(
            f"[Eval {epoch:02d}] d_loss={eval_avg['d_loss']:.4f} d_acc={eval_avg['d_acc']:.3f} "
            f"| g_loss={eval_avg['g_loss']:.4f} g_trick_acc={eval_avg['g_trick_acc']:.3f}"
        )

        if wandb is not None:
            wandb.log({f"val/{k}": v for k, v in eval_avg.items()}, step=global_step)
            # Log a small image panel from the first eval batch
            if first_fake is not None and first_batch is not None:
                imgs = _wandb_images(wandb, first_batch, first_fake)
                wandb.log({"val/examples": imgs}, step=global_step)

    if USE_WANDB and (wandb is not None):
        wandb.finish()

    print("Training loop finished.")


if __name__ == "__main__":
    train()
