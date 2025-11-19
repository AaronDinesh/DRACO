import argparse
import math
from typing import Any, Iterator

import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
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
    power_spectrum,
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


def _power_spectrum_values(final_img: jnp.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    mesh = final_img
    if mesh.ndim == 3 and mesh.shape[-1] == 1:
        mesh = mesh[..., 0]
    k_vals, pk = power_spectrum(mesh, kedges=bins)
    return np.asarray(k_vals), np.asarray(pk)


def _power_spectrum_metrics(
    preds: jnp.ndarray, targets: jnp.ndarray, bins: int
) -> tuple[
    float,
    tuple[np.ndarray, np.ndarray, np.ndarray],
    list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None,
]:
    batch_mses: list[float] = []
    spectra = []
    representative: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    for pred, target in zip(preds, targets):
        k_pred, pk_pred = _power_spectrum_values(pred, bins)
        _, pk_target = _power_spectrum_values(target, bins)
        min_len = min(len(pk_pred), len(pk_target))
        spectra.append((k_pred[:min_len], pk_pred[:min_len], pk_target[:min_len]))
        if min_len == 0:
            continue
        diff = pk_pred[:min_len] - pk_target[:min_len]
        batch_mses.append(float(np.mean(np.square(diff))))
        if representative is None:
            representative = (k_pred[:min_len], pk_pred[:min_len], pk_target[:min_len])

    mse = float(np.mean(batch_mses)) if batch_mses else 0.0
    return mse, representative, spectra


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
) -> None:
    train_steps_per_epoch = math.ceil(n_train / batch_size)
    eval_steps_per_epoch = math.ceil(n_test / batch_size)

    if use_wandb:
        print("----- Setting up WANDB -----")
        _ = wandb.login()
        wandb.require("core")
        wandb.init(  # pyright: ignore[reportUnusedCallResult]
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

    train_key, test_key = random.split(data_key)
    global_step = 0

    epoch_loop = tqdm(
        range(1, num_epochs + 1),
        total=num_epochs,
        leave=False,
        position=0,
        desc="Epoch",
    )
    for epoch in epoch_loop:
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
                model=vel_model,
                optimizer=vel_opt,
                alt_name=f"velocity_epoch_{epoch:03d}",
            )
            save_checkpoint(
                args.checkpoint_dir,
                model=score_model,
                optimizer=score_opt,
                alt_name=f"score_epoch_{epoch:03d}",
            )

        if epoch % eval_every == 0:
            eval_batches = min(args.eval_batches, eval_steps_per_epoch)
            eval_loader: Iterator[Batch] = test_loader(key=test_key, drop_last=False)
            eval_metrics_accum: dict[str, float] = {}
            eval_count = 0
            final_preds = None
            last_eval_batch: Batch | None = None
            last_spectrum: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
            eval_spectrum_figs: list[Any] = []
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
                ps_mse, spectrum, all_spectra = _power_spectrum_metrics(
                    preds, batch["targets"], args.power_spectrum_bins
                )
                metrics = dict(metrics)
                metrics["power_spectrum_mse"] = ps_mse
                if spectrum is not None:
                    last_spectrum = spectrum
                eval_metrics_accum = (
                    {k: eval_metrics_accum.get(k, 0.0) + float(v) for k, v in metrics.items()}
                    if eval_metrics_accum
                    else {k: float(v) for k, v in metrics.items()}
                )

                if use_wandb and all_spectra:
                    batch_figs: list[Any] = []
                    for i, item in enumerate(all_spectra):
                        x, pred_spectra, target_spectra = item
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.loglog(x, pred_spectra, label="generated")
                        ax.loglog(x, target_spectra, label="target")
                        ax.legend()
                        ax.set_title(f"Power Spectrum of Eval {i} at Epoch {epoch}")
                        ax.set_xlabel("Wave number k [h/Mpc]")
                        ax.set_ylabel("P(k)")
                        batch_figs.append(fig)
                        plt.close(fig)
                    eval_spectrum_figs.extend(batch_figs)

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

                    if eval_spectrum_figs:
                        wandb.log(
                            {"eval/power_spectra": [wandb.Image(f) for f in eval_spectrum_figs]},
                            step=global_step,
                        )

                    panel = wandb_image_panel(
                        wandb,
                        inputs=last_eval_batch["inputs"],
                        targets=last_eval_batch["targets"],
                        preds=final_preds,
                        max_items=final_preds.shape[0],
                    )
                    wandb.log({"eval/images": panel}, step=global_step)

                    # final_img = final_preds[0]
                    # final_target = last_eval_batch["targets"][0]
                    # if last_spectrum is not None:
                    #     k_vals, pk_pred, pk_target = last_spectrum
                    #     data_rows = [
                    #         [float(k), float(pred), float(tgt)]
                    #         for k, pred, tgt in zip(k_vals, pk_pred, pk_target)
                    #     ]
                    #     ps_table = wandb.Table(
                    #         data=data_rows,
                    #         columns=["k", "P_pred(k)", "P_target(k)"],
                    #     )
                    #     wandb.log(
                    #         {
                    #             "eval/final_image": wandb.Image(
                    #                 np.asarray(final_img[..., 0]),
                    #                 caption=f"Epoch {epoch:03d} rollout",
                    #             ),
                    #             "eval/final_target": wandb.Image(
                    #                 np.asarray(final_target[..., 0]),
                    #                 caption=f"Epoch {epoch:03d} target",
                    #             ),
                    #             # "eval/power_spectrum": ps_table,
                    #         },
                    #         step=global_step,
                    #     )
                    # else:
                    #     wandb.log(
                    #         {
                    #             "eval/final_image": wandb.Image(
                    #                 np.asarray(final_img[..., 0]),
                    #                 caption=f"Epoch {epoch:03d} rollout",
                    #             ),
                    #             "eval/final_target": wandb.Image(
                    #                 np.asarray(final_target[..., 0]),
                    #                 caption=f"Epoch {epoch:03d} target",
                    #             ),
                    #         },
                    #         step=global_step,
                    #     )

    save_checkpoint(
        args.checkpoint_dir,
        model=vel_model,
        optimizer=vel_opt,
        alt_name="velocity_final",
        wait=True,
    )
    save_checkpoint(
        args.checkpoint_dir,
        model=score_model,
        optimizer=score_opt,
        alt_name="score_final",
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

    if args.velocity_checkpoint_path:
        print(f"----- Loading Velocity Model from {args.velocity_checkpoint_path} -----")
        restore_checkpoint(args.velocity_checkpoint_path, vel_model, vel_opt)

    if args.score_checkpoint_path:
        print(f"----- Loading Score Model from {args.score_checkpoint_path} -----")
        restore_checkpoint(args.score_checkpoint_path, score_model, score_opt)

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
    parser.add_argument("--t-min", type=float, default=1e-3)
    parser.add_argument("--t-max", type=float, default=1.0 - 1e-3)
    parser.add_argument("--gamma-type", type=str, default="brownian")
    parser.add_argument("--gamma-a", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-2)
    parser.add_argument("--integrator-steps", type=int, default=1000)
    parser.add_argument("--n-save", type=int, default=4)
    parser.add_argument("--n-likelihood", type=int, default=1)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--power-spectrum-bins", type=int, default=64)
    parser.add_argument("--time-embed-dim", type=int, default=256)
    parser.add_argument("--velocity-checkpoint-path", type=str, default=None)
    parser.add_argument("--score-checkpoint-path", type=str, default=None)
    parser.add_argument("--use-wandb", action="store_true", default=True)
    parser.add_argument("--wandb-proj-name", type=str, default="DRACO-SI")
    parser.add_argument("--wandb-run-name", type=str, default="si-training-run")
    parser.add_argument("--seed", type=int, default=0)

    main(parser)
