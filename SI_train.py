from __future__ import annotations

import argparse
import math
from collections import deque
from functools import partial
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jax.random as random
import optax
from dotenv import load_dotenv
from tqdm.auto import tqdm

import wandb

# Project imports
from src.interpolants import StochasticInterpolantModel
from src.utils import (
    batch_metrics,
    delete_checkpoint,
    make_train_test_loaders,
    make_xt_and_targets,
    maybe_hflip,
    random_crops,
    restore_checkpoint,
    save_checkpoint,
    sde_sample_forward_cfg,
    sde_sample_forward_conditional,
    wandb_image_panel,
)

# ---------------------------
# Loss (velocity + denoiser)
# ---------------------------


def ema_update(prev: float | None, x: float, beta: float = 0.98) -> float:
    return x if prev is None else (beta * prev + (1 - beta) * x)


def si_losses(
    model: StochasticInterpolantModel,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    cond_vec: jnp.ndarray,
    key: jax.Array,
    a_gamma: float,
    cfg_drop_p: float,
) -> dict[str, jnp.ndarray]:
    """Stochastic interpolant losses: velocity (b) and denoiser (eta)."""
    B = x0.shape[0]
    key_t, key_z, key_drop = random.split(key, 3)
    t = random.uniform(key_t, (B,), minval=0.0, maxval=1.0)
    z = random.normal(key_z, x0.shape)

    x_t, dI, gdot_z = make_xt_and_targets(x0, x1, z, t, a=a_gamma)

    # Classifier-free guidance training drop (optional)
    if cfg_drop_p > 0.0:
        keep = random.bernoulli(key_drop, 1.0 - cfg_drop_p, (B, 1))
        cond_use = cond_vec * keep
    else:
        cond_use = cond_vec

    b_hat, eta_hat = model(x_t, t, cond_use)
    assert b_hat.shape[-1] == 1
    assert eta_hat.shape[-1] == 1
    reduce = lambda y: jnp.sum(y, axis=(1, 2, 3))  # NHWC
    lb = 0.5 * reduce(b_hat**2) - reduce((dI + gdot_z) * b_hat)
    leta = 0.5 * reduce(eta_hat**2) - reduce(z * eta_hat)

    loss = jnp.mean(lb + leta)

    return {"loss": loss, "lb": jnp.mean(lb), "leta": jnp.mean(leta)}


# ---------------------------
# Optimizer and JITted steps
# ---------------------------


def make_optim_and_steps(args: argparse.Namespace, n_train: int):
    total_steps = math.ceil(n_train / args.batch_size) * args.epochs
    warmup_steps = int(total_steps * 0.1)
    # tx = optax.adamw(learning_rate=args.g_lr, weight_decay=args.weight_decay)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.g_lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps - warmup_steps,
        end_value=1e-6,
    )

    tx = optax.adafactor(
        learning_rate=schedule,  # schedule or a fixed float
        multiply_by_parameter_scale=False,
        weight_decay_rate=args.weight_decay,
    )

    @jax.jit
    def train_step(
        optimizer: nnx.Optimizer,
        model: StochasticInterpolantModel,
        batch: dict[str, jnp.ndarray],
        key: jax.Array,
    ):
        x0, x1, cond = batch["inputs"], batch["targets"], batch["params"]

        # Local-fidelity augs (random crops + flips); keep cond unchanged (global)
        key0, key1, keyf, keytz = random.split(key, 4)
        # x0 = random_crops(x0, args.crop_size, key0)
        # x1 = random_crops(x1, args.crop_size, key1)
        # x0 = maybe_hflip(x0, 0.5, keyf)
        # x1 = maybe_hflip(x1, 0.5, keyf)

        def loss_fn(m: StochasticInterpolantModel) -> jnp.ndarray:
            metrics = si_losses(
                m, x0, x1, cond, keytz, a_gamma=args.a_gamma, cfg_drop_p=args.cfg_drop_p
            )

            return metrics["loss"], metrics  # Return loss and then metrics as auxiliary return

        (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        optimizer.update(model, grads)

        return optimizer, model, metrics

    @jax.jit
    def eval_step(
        model: StochasticInterpolantModel,
        batch: dict[str, jnp.ndarray],
        key: jax.Array,
    ):
        x0, x1, cond = batch["inputs"], batch["targets"], batch["params"]
        return si_losses(model, x0, x1, cond, key, a_gamma=args.a_gamma, cfg_drop_p=args.cfg_drop_p)

    return tx, train_step, eval_step


# ---------------------------
# Training loop
# ---------------------------


def train(args: argparse.Namespace):
    # RNG
    master_key = random.key(args.seed)
    data_key, split_key = random.split(master_key)

    # Data (YOUR loader)
    (
        train_loader,
        test_loader,
        n_train,
        n_test,
        img_size,  # from your utils: input_maps[0].shape[1]
        cond_dim,  # from your utils: cosmos_params[0].shape[0]
        cosmos_mu,
        cosmos_sigma,
    ) = make_train_test_loaders(
        key=split_key,
        batch_size=args.batch_size,
        input_data_path=args.input_maps,
        output_data_path=args.output_maps,
        csv_path=args.cosmos_params,
        test_ratio=args.test_ratio,
        transform_name=args.transform_name,
    )

    # Model + optimizer
    rngs = nnx.Rngs(args.seed)

    model = StochasticInterpolantModel(
        in_channels=args.img_channels,
        base_channels=args.base_channels,
        cosmology_dim=int(cond_dim),
        time_embedding_dim=args.time_dim,
        fused_cond_dim=args.cond_dim,
        rngs=rngs,
    )

    tx, train_step, eval_step = make_optim_and_steps(args, n_train)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    # Optional restore
    if args.checkpoint_path:
        _ = restore_checkpoint(args.checkpoint_path, model, optimizer)

    # W&B
    if args.use_wandb:
        wandb.require("core")
        wandb.init(
            project=args.wandb_proj_name,
            entity="aarondinesh2002-epfl",
            name=args.wandb_run_name,
            mode="online",
            config=dict(
                batch_size=args.batch_size,
                g_lr=args.g_lr,
                weight_decay=args.weight_decay,
                epochs=args.epochs,
                img_size=int(img_size),
                img_channels=args.img_channels,
                cond_dim=int(cond_dim),
                crop_size=args.crop_size,
                a_gamma=args.a_gamma,
                eps0=args.eps0,
                eps_taper=args.eps_taper,
                t_schedule=args.t_schedule,
                guidance_scale=args.guidance_scale,
                cfg_drop_p=args.cfg_drop_p,
                transform_name=args.transform_name,
            ),
        )

    train_steps_per_epoch = math.ceil(n_train / args.batch_size)
    eval_steps_per_epoch = math.ceil(n_test / args.batch_size)
    global_step = 0
    best_val = float("inf")
    _loss_ema = None

    _loss_buffer = deque(maxlen=max(2, args.plateau_window + 1))  # store (global_step, loss)
    _plateau_triggered = False

    # Epochs
    train_key = random.fold_in(data_key, 0)
    for epoch in tqdm(range(1, args.epochs + 1), total=args.epochs, desc="Epochs"):
        # TRAIN
        step = 0
        train_key = random.fold_in(train_key, epoch)
        for batch in tqdm(
            train_loader(key=train_key, drop_last=True),
            total=train_steps_per_epoch,
            leave=False,
            desc=f"Train {epoch:03d}",
        ):
            break  ### REMOVE
            train_key, sub = random.split(train_key)
            optimizer, model, metrics = train_step(optimizer, model, batch, sub)
            step += 1
            global_step += 1

            if step % args.log_rate == 0 or step == train_steps_per_epoch:
                log = {f"train/{k}": float(jax.device_get(v)) for k, v in metrics.items()}
                log.update({"epoch": epoch, "step": global_step})
                print(
                    f"[Train e{epoch:03d} s{step:04d}] "
                    f"loss={log['train/loss']:.4f} lb={log['train/lb']:.4f} leta={log['train/leta']:.4f}"
                )
                if args.use_wandb:
                    wandb.log(log, step=global_step)

            if args.use_plateau:
                # Add the loss to the queue for plateau detection
                _loss_ema = ema_update(_loss_ema, float(jax.device_get(metrics["loss"])), beta=0.98)
                _loss_buffer.append((global_step, _loss_ema))

                if (
                    global_step >= args.plateau_warmup
                    and epoch >= args.plateau_min_epochs
                    and len(_loss_buffer) > args.plateau_window
                ):
                    # Current loss
                    cur_step, cur_loss = _loss_buffer[-1]

                    # Find the oldest entry that is at least plateau_window steps behind
                    old_idx = -1
                    for i in range(len(_loss_buffer) - 1, -1, -1):
                        s, _ = _loss_buffer[i]
                        if cur_step - s >= args.plateau_window:
                            old_idx = i
                            break  # Found it, exit the search loop

                    # If we found a valid old entry, check for plateau
                    if old_idx != -1:
                        old_step, old_loss = _loss_buffer[old_idx]
                        denom = abs(old_loss) + 1e-12
                        rel_impr = abs(cur_loss - old_loss) / denom

                        # Log the plateau signal
                        if args.use_wandb:
                            wandb.log({"train/plateau_rel_impr": rel_impr}, step=global_step)

                        # Check if relative improvement is below threshold
                        if rel_impr < args.plateau_threshold:
                            print(
                                f"[PlateauStop] No sufficient improvement over "
                                f"{args.plateau_window} steps (Δrel={rel_impr:.6g} < {args.plateau_threshold}). "
                                f"Stopping at epoch {epoch}, step {global_step}."
                            )
                            _plateau_triggered = True

                # Check if plateau was triggered (outside the condition block)
                if _plateau_triggered:
                    if args.use_wandb:
                        wandb.finish()

                    save_checkpoint(
                        checkpoint_dir=args.checkpoint_dir,
                        epoch=None,
                        step=None,
                        model=model,
                        optimizer=optimizer,
                        alt_name=f"Final_model",
                        data_stats={
                            "cosmos_mu": cosmos_mu,
                            "cosmos_sigma": cosmos_sigma,
                        },
                        wait=True,
                    )
                    return (
                        model,
                        optimizer,
                        cosmos_mu,
                        cosmos_sigma,
                    )  # exit train() early

        # CHECKPOINT (store cosmos stats too, for completeness)
        if epoch % args.ckpt_every == 0:
            save_checkpoint(
                checkpoint_dir=args.checkpoint_dir,
                epoch=epoch,
                step=global_step,
                model=model,
                optimizer=optimizer,
                model_name="SI",
                data_stats={"cosmos_mu": cosmos_mu, "cosmos_sigma": cosmos_sigma},
            )

        # EVAL
        eval_key = random.fold_in(random.key(args.seed + 999), epoch)
        eval_sums: dict[str, float] = {}
        sample_logged = False
        for batch in tqdm(
            test_loader(key=eval_key, drop_last=True),
            total=eval_steps_per_epoch,
            leave=False,
            desc=f"Eval  {epoch:03d}",
        ):
            eval_key, sub = random.split(eval_key)
            m = eval_step(model, batch, sub)
            for k, v in m.items():
                eval_sums[k] = eval_sums.get(k, 0.0) + float(jax.device_get(v))

            # Log a quick sample panel once per epoch
            if not sample_logged:
                x0, x1, c = batch["inputs"], batch["targets"], batch["params"]
                if args.guidance_scale > 1.0 and args.cfg_drop_p > 0.0:
                    gen = sde_sample_forward_cfg(
                        model,
                        x0,
                        c,
                        guidance_scale=args.guidance_scale,
                        n_infer_steps=args.eval_infer_steps,
                        a_gamma=args.a_gamma,
                        eps0=args.eps0,
                        eps_taper=args.eps_taper,
                        endpoint_clip=args.endpoint_clip,
                        t_schedule=args.t_schedule,
                        t_power=args.t_power,
                        key=sub,
                    )
                else:
                    gen = sde_sample_forward_conditional(
                        model,
                        x0,
                        c,
                        n_infer_steps=args.eval_infer_steps,
                        a_gamma=args.a_gamma,
                        eps0=args.eps0,
                        eps_taper=args.eps_taper,
                        endpoint_clip=args.endpoint_clip,
                        t_schedule=args.t_schedule,
                        t_power=args.t_power,
                        key=sub,
                    )
                if args.use_wandb:
                    imgs = wandb_image_panel(wandb, x0[:3], x1[:3], gen[:3])
                    wandb.log({"val/examples": imgs}, step=global_step)
                sample_logged = True

        eval_avg = {k: v / eval_steps_per_epoch for k, v in eval_sums.items()}
        print(
            f"[Eval  e{epoch:03d}] loss={eval_avg['loss']:.4f} "
            f"lb={eval_avg['lb']:.4f} leta={eval_avg['leta']:.4f}"
        )

        if args.use_wandb:
            wandb.log({f"val/{k}": v for k, v in eval_avg.items()}, step=global_step)

        mets = batch_metrics(gen, x1)
        if args.use_wandb:
            wandb.log({f"val/{k}": float(v) for k, v in mets.items()}, step=global_step)
        print(
            f"PSNR={float(mets['psnr']):.2f}dB  SSIM={float(mets['ssim']):.3f}  RMSE={float(jnp.sqrt(mets['mse'])):.4f}"
        )

        # Save best-by-loss
        if eval_avg["loss"] < best_val:
            prev_name = f"BEST_LOSS_{best_val:.06f}"
            try:  # keep dir tidy (ignore if missing)
                delete_checkpoint(args.checkpoint_dir, prev_name)
            except Exception:
                pass
            best_val = eval_avg["loss"]
            save_checkpoint(
                checkpoint_dir=args.checkpoint_dir,
                epoch=epoch,
                step=global_step,
                model=model,
                optimizer=optimizer,
                alt_name=f"BEST_LOSS_{best_val:.06f}",
                data_stats={"cosmos_mu": cosmos_mu, "cosmos_sigma": cosmos_sigma},
            )

    if args.use_wandb:
        wandb.finish()

    save_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        epoch=None,
        step=None,
        model=model,
        optimizer=optimizer,
        alt_name=f"Final_model",
        data_stats={"cosmos_mu": cosmos_mu, "cosmos_sigma": cosmos_sigma},
        wait=True,  # Since save_checkpoint is async we specifically need to tell it to block here
    )
    return model, optimizer, cosmos_mu, cosmos_sigma


# ---------------------------
# CLI
# ---------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Stochastic Interpolants Training")

    # data
    p.add_argument("--input-maps", required=True)
    p.add_argument("--output-maps", required=True)
    p.add_argument("--cosmos-params", required=True)
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--transform-name", default="log10")

    # model
    p.add_argument("--img-channels", type=int, default=1)
    p.add_argument("--base-channels", type=int, default=96)
    p.add_argument("--time-dim", type=int, default=128)
    p.add_argument("--cond-dim", type=int, default=256)

    # SI
    p.add_argument("--a-gamma", type=float, default=1.0)

    # training
    p.add_argument("--epochs", type=int, default=150000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--g-lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--cfg-drop-p", type=float, default=0.1)
    p.add_argument("--log-rate", type=int, default=5)
    p.add_argument("--ckpt-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)

    # fmt: off
    # Early Stopping (Loss Plateau Detection)
    p.add_argument("--use-plateau", action="store_true", default=False)
    p.add_argument("--plateau-window", type=int, default=1000,
        help="k: compare current loss to loss k steps ago")
    p.add_argument("--plateau-threshold", type=float, default=1e-3,
        help="relative improvement threshold; e.g. 1e-3 = 0.1%")
    p.add_argument("--plateau-warmup", type=int, default=2000,
        help="don’t check plateau until this many global steps")
    p.add_argument("--plateau-min-epochs", type=int, default=3,
        help="require at least this many epochs before allowing plateau stop")
    # fmt: on

    # sampler
    p.add_argument("--n-infer-steps", type=int, default=300)
    p.add_argument("--eval-infer-steps", type=int, default=300)
    p.add_argument("--guidance-scale", type=float, default=1.5)
    p.add_argument("--eps0", type=float, default=0.1)
    p.add_argument("--eps-taper", type=float, default=0.6)
    p.add_argument("--endpoint-clip", type=float, default=1e-12)
    p.add_argument("--t-schedule", choices=["linear", "cosine", "power"], default="cosine")
    p.add_argument("--t-power", type=float, default=2.0)

    # checkpoints & wandb
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--checkpoint-path", default=None)
    p.add_argument("--use-wandb", action="store_true", default=True)
    p.add_argument("--wandb-proj-name", default="DRACO-SI")
    p.add_argument("--wandb-run-name", default="si-nnx-unet-attn")
    return p


if __name__ == "__main__":
    load_dotenv()
    parser = build_argparser()
    args = parser.parse_args()
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    model, optimizer, cosmos_mu, cosmos_sigma = train(args)
