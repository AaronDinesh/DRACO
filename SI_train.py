import argparse
import math
from typing import Iterator

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import optax
from dotenv import load_dotenv
from flax import nnx
from jax._src.typing import Array
from tqdm.auto import tqdm

import wandb
from src.interpolants import (
    LinearInterpolant,
    SDEIntegrator,
    StochasticInterpolantUNet,
    batch_loss_b,
    batch_loss_s,
    make_gamma,
)
from src.typing import Batch, Loader, TransformName
from src.utils import (
    batch_metrics,
    make_train_test_loaders,
    restore_checkpoint,
    save_checkpoint,
    wandb_image_panel,
)


def _to_float_dict(log: dict[str, jnp.ndarray | float]) -> dict[str, float]:
    return {k: float(v) for k, v in log.items()}


def _velocity_loss(
    model: StochasticInterpolantUNet,
    batch: Batch,
    t_batch: jnp.ndarray,
    interpolant: LinearInterpolant,
    key: Array,
) -> jnp.ndarray:
    x0 = batch["inputs"]
    x1 = batch["targets"]
    cosmos = batch["params"]

    def conditioned_b(x: jnp.ndarray, t: jnp.ndarray, cosmos_vec: jnp.ndarray) -> jnp.ndarray:
        return model(x, cosmos_vec, t)

    return batch_loss_b(conditioned_b, x0, x1, t_batch, cosmos, interpolant, key)


def _score_loss(
    model: StochasticInterpolantUNet,
    batch: Batch,
    t_batch: jnp.ndarray,
    interpolant: LinearInterpolant,
    key: Array,
) -> jnp.ndarray:
    x0 = batch["inputs"]
    x1 = batch["targets"]
    cosmos = batch["params"]

    def conditioned_s(x: jnp.ndarray, t: jnp.ndarray, cosmos_vec: jnp.ndarray) -> jnp.ndarray:
        return model(x, cosmos_vec, t)

    return batch_loss_s(conditioned_s, x0, x1, t_batch, cosmos, interpolant, key)


@nnx.jit(static_argnums=(5,))
def _velocity_step(
    model: StochasticInterpolantUNet,
    optimizer,
    batch: Batch,
    t_batch: jnp.ndarray,
    key: Array,
    interpolant: LinearInterpolant,
) -> dict[str, jnp.ndarray]:
    def loss_fn(model: StochasticInterpolantUNet) -> jnp.ndarray:
        return _velocity_loss(model, batch, t_batch, interpolant, key)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model=model, grads=grads)
    return {
        "b_loss": loss,
        "b_t_mean": jnp.mean(t_batch),
    }


@nnx.jit(static_argnums=(5,))
def _score_step(
    model: StochasticInterpolantUNet,
    optimizer,
    batch: Batch,
    t_batch: jnp.ndarray,
    key: Array,
    interpolant: LinearInterpolant,
) -> dict[str, jnp.ndarray]:
    def loss_fn(model: StochasticInterpolantUNet) -> jnp.ndarray:
        return _score_loss(model, batch, t_batch, interpolant, key)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model=model, grads=grads)
    return {
        "s_loss": loss,
        "s_t_mean": jnp.mean(t_batch),
    }


def eval_step(
    vel_model: StochasticInterpolantUNet,
    score_model: StochasticInterpolantUNet,
    batch: Batch,
    eval_key: Array,
    interpolant: LinearInterpolant,
    eps: float,
    t_grid: jnp.ndarray,
    n_save: int,
    n_likelihood: int,
) -> tuple[dict[str, jnp.ndarray], jnp.ndarray]:
    cosmos = batch["params"]
    x0 = batch["inputs"]
    x1 = batch["targets"]

    def _prepare_t(t: jnp.ndarray) -> jnp.ndarray:
        t = jnp.reshape(t, (t.shape[0],))
        return t

    def b_fn(x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        return vel_model(x, cosmos, _prepare_t(t))

    def s_fn(x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        return score_model(x, cosmos, _prepare_t(t))

    integrator = SDEIntegrator(
        b=b_fn,
        s=s_fn,
        eps=eps,
        interpolant=interpolant,
        t_grid=t_grid,
        n_save=n_save,
        n_step=t_grid.shape[0] - 1,
        n_likelihood=n_likelihood,
    )
    preds = integrator.forward_rollout(x0, eval_key)
    metrics = batch_metrics(preds, x1)
    return metrics, preds


def train(
    num_epochs: int,
    batch_size: int,
    train_loader: Loader,
    n_train: int,
    test_loader: Loader,
    n_test: int,
    img_size: int,
    img_channels: int,
    cosmos_params_len: int,
    log_every: int,
    eval_every: int,
    checkpoint_every: int,
    vel_model: StochasticInterpolantUNet,
    score_model: StochasticInterpolantUNet,
    vel_opt,
    score_opt,
    data_key: Array,
    use_wandb: bool,
    args: argparse.Namespace,
    interpolant: LinearInterpolant,
    t_grid: jnp.ndarray,
    cosmos_params_mu: jnp.ndarray,
    cosmos_params_sigma: jnp.ndarray,
    t_min: float,
    t_max: float,
    start_epoch: int = 1,
    start_step: int = 0,
    resume_wandb_id: str | None = None,
) -> None:
    train_steps_per_epoch = math.ceil(n_train / batch_size)
    eval_steps_per_epoch = math.ceil(n_test / batch_size)
    data_stats = {
        "cosmos_params_mu": cosmos_params_mu,
        "cosmos_params_sigma": cosmos_params_sigma,
    }

    run_id = resume_wandb_id
    if use_wandb:
        print("----- Setting up WANDB -----")
        _ = wandb.login()
        wandb.require("core")
        wandb_kwargs = dict(
            entity="aarondinesh2002-epfl",
            project=args.wandb_proj_name,
            name=args.wandb_run_name,
            mode="online",
            config=dict(
                batch_size=batch_size,
                vel_lr=args.vel_lr,
                score_lr=args.score_lr,
                beta1=args.beta1,
                beta2=args.beta2,
                epochs=args.epochs,
                transform_name=args.transform_name,
                img_size=img_size,
                img_channels=img_channels,
                cond_dim=cosmos_params_len,
                train_steps=train_steps_per_epoch,
                eval_steps=eval_steps_per_epoch,
                t_min=args.t_min,
                t_max=args.t_max,
                gamma_type=args.gamma_type,
                eps=args.eps,
                integrator_steps=args.integrator_steps,
                cosmos_params_mu=np.asarray(cosmos_params_mu),
                cosmos_params_sigma=np.asarray(cosmos_params_sigma),
            ),
        )
        if run_id is not None:
            wandb_kwargs["id"] = run_id
            wandb_kwargs["resume"] = "allow"

        run = wandb.init(**wandb_kwargs)  # pyright: ignore[reportUnusedCallResult]
        run_id = run.id if run is not None else run_id
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("eval/*", step_metric="epoch")

    train_key, test_key = random.split(data_key)
    global_step = start_step

    epoch_loop = tqdm(
        range(start_epoch, num_epochs + 1),
        total=max(0, num_epochs - start_epoch + 1),
        leave=False,
        position=0,
        desc="Epoch",
    )
    last_epoch = start_epoch - 1
    for epoch in epoch_loop:
        last_epoch = epoch
        step = 0
        train_key = random.fold_in(train_key, epoch)
        test_key = random.fold_in(test_key, epoch)

        train_bar = tqdm(
            train_loader(key=train_key, drop_last=True),
            total=train_steps_per_epoch,
            leave=False,
            position=1,
            desc=f"Epoch {epoch:03d} Train",
        )
        last_vel_log: dict[str, float] = {}
        last_score_log: dict[str, float] = {}
        for batch in train_bar:
            train_key, vel_key, score_key = random.split(train_key, 3)
            vel_t_key, vel_loss_key = random.split(vel_key)
            score_t_key, score_loss_key = random.split(score_key)

            batch_size_actual = batch["inputs"].shape[0]
            vel_t_batch = random.uniform(
                vel_t_key,
                shape=(batch_size_actual,),
                minval=t_min,
                maxval=t_max,
                dtype=batch["inputs"].dtype,
            )
            score_t_batch = random.uniform(
                score_t_key,
                shape=(batch_size_actual,),
                minval=t_min,
                maxval=t_max,
                dtype=batch["inputs"].dtype,
            )

            vel_metrics = _velocity_step(
                vel_model,
                vel_opt,
                batch,
                vel_t_batch,
                vel_loss_key,
                interpolant,
            )

            score_metrics = _score_step(
                score_model,
                score_opt,
                batch,
                score_t_batch,
                score_loss_key,
                interpolant,
            )

            step += 1
            global_step += 1

            if step % log_every == 0 or step == train_steps_per_epoch:
                vel_log = {f"train/{k}": v for k, v in _to_float_dict(vel_metrics).items()}
                score_log = {f"train/{k}": v for k, v in _to_float_dict(score_metrics).items()}
                log = {"epoch": epoch, "step": global_step} | vel_log | score_log
                last_vel_log = vel_log
                last_score_log = score_log
                print(
                    f"[Epoch {epoch:03d} Step {step:04d}] "
                    + f"b_loss={vel_log.get('train/b_loss', 0.0):.4f} "
                    + f"s_loss={score_log.get('train/s_loss', 0.0):.4f}"
                )
                if use_wandb:
                    wandb.log(log, step=global_step)
            train_bar.set_postfix(
                b_loss=last_vel_log.get("train/b_loss", 0.0),
                s_loss=last_score_log.get("train/s_loss", 0.0),
            )
        train_bar.close()
        if epoch % checkpoint_every == 0:
            save_checkpoint(
                args.checkpoint_dir,
                epoch=epoch,
                step=global_step,
                model=vel_model,
                optimizer=vel_opt,
                alt_name=f"velocity_epoch_{epoch:03d}",
                data_stats=data_stats,
                wandb_run_id=run_id,
            )
            save_checkpoint(
                args.checkpoint_dir,
                epoch=epoch,
                step=global_step,
                model=score_model,
                optimizer=score_opt,
                alt_name=f"score_epoch_{epoch:03d}",
                data_stats=data_stats,
                wandb_run_id=run_id,
            )

        if epoch % eval_every == 0:
            eval_batches = min(args.eval_batches, eval_steps_per_epoch)
            eval_loader: Iterator[Batch] = test_loader(key=test_key, drop_last=False)
            eval_metrics_accum: dict[str, float] = {}
            eval_count = 0
            final_preds = None
            last_eval_batch: Batch | None = None
            eval_bar = tqdm(
                range(eval_batches),
                total=eval_batches,
                leave=False,
                position=2,
                desc=f"Epoch {epoch:03d} Eval",
            )
            for _ in eval_bar:
                try:
                    batch = next(eval_loader)
                except StopIteration:
                    break
                test_key, eval_subkey = random.split(test_key)
                metrics, preds = eval_step(
                    vel_model,
                    score_model,
                    batch,
                    eval_subkey,
                    interpolant,
                    args.eps,
                    t_grid,
                    args.n_save,
                    args.n_likelihood,
                )
                final_preds = preds
                last_eval_batch = batch
                metrics = dict(metrics)
                eval_metrics_accum = (
                    {k: eval_metrics_accum.get(k, 0.0) + float(v) for k, v in metrics.items()}
                    if eval_metrics_accum
                    else {k: float(v) for k, v in metrics.items()}
                )

                eval_count += 1
                eval_bar.set_postfix(
                    psnr=float(metrics.get("psnr", 0.0)),
                    mse=float(metrics.get("mse", 0.0)),
                )
            eval_bar.close()

            if eval_count > 0 and final_preds is not None and last_eval_batch is not None:
                avg_eval_metrics = {k: v / eval_count for k, v in eval_metrics_accum.items()}
                eval_log = {f"eval/{k}": v for k, v in avg_eval_metrics.items()}
                print(
                    f"[Epoch {epoch:03d} Eval] "
                    + ", ".join(f"{k}={v:.4f}" for k, v in avg_eval_metrics.items())
                )

                if use_wandb:
                    wandb.log(eval_log, step=global_step)

                    panel = wandb_image_panel(
                        wandb,
                        inputs=last_eval_batch["inputs"],
                        targets=last_eval_batch["targets"],
                        preds=final_preds,
                        max_items=final_preds.shape[0],
                    )
                    wandb.log({"eval/images": panel}, step=global_step)

    save_checkpoint(
        args.checkpoint_dir,
        epoch=last_epoch,
        step=global_step,
        model=vel_model,
        optimizer=vel_opt,
        alt_name="velocity_final",
        data_stats=data_stats,
        wandb_run_id=run_id,
        wait=True,
    )
    save_checkpoint(
        args.checkpoint_dir,
        epoch=last_epoch,
        step=global_step,
        model=score_model,
        optimizer=score_opt,
        alt_name="score_final",
        data_stats=data_stats,
        wandb_run_id=run_id,
        wait=True,
    )
    print("Training finished.")


def main(parser: argparse.ArgumentParser):
    _ = load_dotenv()
    args = parser.parse_args()

    master_key = random.key(args.seed)
    vel_key, score_key, data_key, train_test_key = random.split(master_key, 4)

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
        key=train_test_key,
        batch_size=args.batch_size,
        input_data_path=args.input_maps,
        output_data_path=args.output_maps,
        csv_path=args.cosmos_params,
        test_ratio=args.test_ratio,
        transform_name=args.transform_name,
    )

    print("----- Creating UNet Models -----")
    vel_model = StochasticInterpolantUNet(
        key=vel_key,
        in_features=args.img_channels,
        out_features=args.img_channels,
        len_cosmos_params=cosmos_params_len,
        time_embed_dim=args.time_embed_dim,
    )
    score_model = StochasticInterpolantUNet(
        key=score_key,
        in_features=args.img_channels,
        out_features=args.img_channels,
        len_cosmos_params=cosmos_params_len,
        time_embed_dim=args.time_embed_dim,
    )

    # vel_params = nnx.state(vel_model, nnx.Param)
    # vel_param_count = sum(jnp.prod(x.shape) for x in jax.tree_util.tree_leaves(vel_params))
    # score_params = nnx.state(score_model, nnx.Param)
    # score_param_count = sum(jnp.prod(x.shape) for x in jax.tree_util.tree_leaves(score_params))
    # print(f"Velocity params: {vel_param_count}")
    # print(f"Score params: {score_param_count}")

    print("----- Creating Optimizers -----")
    vel_opt = nnx.Optimizer(
        vel_model,
        optax.adam(args.vel_lr, b1=args.beta1, b2=args.beta2),
        wrt=nnx.Param,
    )
    score_opt = nnx.Optimizer(
        score_model,
        optax.adam(args.score_lr, b1=args.beta1, b2=args.beta2),
        wrt=nnx.Param,
    )

    start_epoch = 1
    start_step = 0
    resume_wandb_id: str | None = None

    if args.velocity_checkpoint_path:
        print(f"----- Loading Velocity Model from {args.velocity_checkpoint_path} -----")
        vel_ckpt = restore_checkpoint(args.velocity_checkpoint_path, vel_model, vel_opt)
        vel_epoch = vel_ckpt.get("epoch")
        vel_step = vel_ckpt.get("step")
        start_epoch = max(start_epoch, (int(vel_epoch) if vel_epoch is not None else 0) + 1)
        start_step = max(start_step, int(vel_step) if vel_step is not None else 0)
        resume_wandb_id = resume_wandb_id or vel_ckpt.get("wandb_run_id")

    if args.score_checkpoint_path:
        print(f"----- Loading Score Model from {args.score_checkpoint_path} -----")
        score_ckpt = restore_checkpoint(args.score_checkpoint_path, score_model, score_opt)
        score_epoch = score_ckpt.get("epoch")
        score_step = score_ckpt.get("step")
        start_epoch = max(start_epoch, (int(score_epoch) if score_epoch is not None else 0) + 1)
        start_step = max(start_step, int(score_step) if score_step is not None else 0)
        resume_wandb_id = resume_wandb_id or score_ckpt.get("wandb_run_id")

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

    if start_epoch > 1 or start_step > 0:
        print(f"----- Resuming from epoch {start_epoch} (global step {start_step}) -----")

    print("----- Beginning Training -----")
    train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        train_loader=train_loader,
        n_train=n_train,
        test_loader=test_loader,
        n_test=n_test,
        img_size=img_size,
        img_channels=args.img_channels,
        cosmos_params_len=cosmos_params_len,
        log_every=args.log_rate,
        eval_every=args.eval_rate,
        checkpoint_every=args.checkpoint_rate,
        vel_model=vel_model,
        score_model=score_model,
        vel_opt=vel_opt,
        score_opt=score_opt,
        data_key=data_key,
        use_wandb=args.use_wandb,
        args=args,
        interpolant=interpolant,
        t_grid=t_grid,
        cosmos_params_mu=cosmos_params_mu,
        cosmos_params_sigma=cosmos_params_sigma,
        t_min=args.t_min,
        t_max=args.t_max,
        start_epoch=start_epoch,
        start_step=start_step,
        resume_wandb_id=resume_wandb_id,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Stochastic Interpolant Training Script")

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--log-rate", type=int, default=5)
    parser.add_argument("--eval-rate", type=int, default=5)
    parser.add_argument("--checkpoint-rate", type=int, default=10)
    parser.add_argument("--input-maps", type=str, required=True)
    parser.add_argument("--output-maps", type=str, required=True)
    parser.add_argument("--cosmos-params", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--img-channels", type=int, default=1)
    parser.add_argument("--transform-name", default="log10")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--vel-lr", type=float, default=2e-4)
    parser.add_argument("--score-lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--t-min", type=float, default=1e-9)
    parser.add_argument("--t-max", type=float, default=1.0 - 1e-9)
    parser.add_argument("--gamma-type", type=str, default="brownian")
    parser.add_argument("--gamma-a", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=5e-3)
    parser.add_argument("--integrator-steps", type=int, default=3000)
    parser.add_argument("--n-save", type=int, default=1)
    parser.add_argument("--n-likelihood", type=int, default=1)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--time-embed-dim", type=int, default=256)
    parser.add_argument("--velocity-checkpoint-path", type=str, default=None)
    parser.add_argument("--score-checkpoint-path", type=str, default=None)
    parser.add_argument("--use-wandb", action="store_true", default=True)
    parser.add_argument("--wandb-proj-name", type=str, default="DRACO-SI")
    parser.add_argument("--wandb-run-name", type=str, default="si-training-run")
    parser.add_argument("--seed", type=int, default=0)

    main(parser)
