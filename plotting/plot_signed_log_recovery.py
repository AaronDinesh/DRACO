import argparse
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.random as random
import matplotlib

matplotlib.use("Agg")  # headless backend for batch plotting
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

_TARGETS_MM: np.memmap | None = None
_TEST_INDICES_ARR: np.ndarray | None = None
_CALIB_CONST: float | None = None
_OUTPUT_DIR: Path | None = None
_CALIB_OUTPUT_DIR: Path | None = None
_PRIMARY_CMAP: str | None = None
_RESIDUAL_CMAP: str | None = None
_INCLUDE_CALIBRATED: bool = False
_DPI: int = 150


def _init_worker(
    target_data_path_str: str,
    test_indices_arr: np.ndarray,
    calib_const: float | None,
    output_dir_str: str,
    calibrated_output_dir_str: str | None,
    cmap_str: str,
    residual_cmap_str: str,
    include_calibrated: bool,
    dpi: int,
):
    """Initializer for ProcessPool workers."""
    global _TARGETS_MM, _TEST_INDICES_ARR, _CALIB_CONST
    global _OUTPUT_DIR, _CALIB_OUTPUT_DIR, _PRIMARY_CMAP, _RESIDUAL_CMAP, _INCLUDE_CALIBRATED
    global _DPI
    _TARGETS_MM = np.load(Path(target_data_path_str), mmap_mode="r")
    _TEST_INDICES_ARR = np.asarray(test_indices_arr)
    _CALIB_CONST = calib_const
    _OUTPUT_DIR = Path(output_dir_str)
    _CALIB_OUTPUT_DIR = Path(calibrated_output_dir_str) if calibrated_output_dir_str else None
    _PRIMARY_CMAP = cmap_str
    _RESIDUAL_CMAP = residual_cmap_str
    _INCLUDE_CALIBRATED = include_calibrated
    _DPI = int(dpi)
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if _INCLUDE_CALIBRATED and _CALIB_OUTPUT_DIR is not None:
        _CALIB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _proc_task(sample_id: int, pred_path_str: str):
    if (
        _TARGETS_MM is None
        or _TEST_INDICES_ARR is None
        or _OUTPUT_DIR is None
        or _PRIMARY_CMAP is None
        or _RESIDUAL_CMAP is None
    ):
        return None
    if sample_id >= len(_TEST_INDICES_ARR):
        return "skip"

    dataset_idx = int(_TEST_INDICES_ARR[sample_id])
    target_raw = np.asarray(_TARGETS_MM[dataset_idx])
    pred_sample = np.load(pred_path_str)
    _generate_plots(
        sample_id=sample_id,
        target_raw=target_raw,
        pred_phi=pred_sample,
        calibration_constant=_CALIB_CONST,
        output_dir=_OUTPUT_DIR,
        calibrated_output_dir=_CALIB_OUTPUT_DIR,
        cmap=_PRIMARY_CMAP,
        residual_cmap=_RESIDUAL_CMAP,
        include_calibrated=_INCLUDE_CALIBRATED,
        dpi=_DPI,
    )
    return "ok"


def _split_master_key(seed: int):
    """
    Mirror the training/evaluation key splits:
    master -> gen_key, disc_key, data_key, train_test_key
    """
    master_key = random.key(seed)
    return random.split(master_key, 4)


def _make_target_loader(
    key,
    target_data_path: Path,
    batch_size: int,
    test_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.memmap]:
    """Minimal target-only loader with the same permutation logic as training."""
    targets = np.load(target_data_path, mmap_mode="r")
    dataset_len = int(targets.shape[0])
    n_test = max(1, int(round(test_ratio * dataset_len)))

    random_shuffle = random.permutation(key=key, x=dataset_len)
    test_idx = random_shuffle[:n_test]
    train_idx = random_shuffle[n_test:]

    return np.asarray(train_idx), np.asarray(test_idx), targets


def _compute_log10_min_constant(
    targets: np.memmap, indices: np.ndarray, desc: str = "Calibrating"
) -> float:
    """Dataset-average minimum log10 used for the calibrated comparison."""
    mins = []
    pbar = tqdm(indices, desc=desc, unit="sample")
    for idx in pbar:
        x = targets[int(idx)]
        logx = _safe_log10(x)
        mins.append(float(logx.min()))
    if not mins:
        raise RuntimeError("No samples found to compute calibration constant.")
    return float(np.mean(mins))


def _safe_log10(arr: np.ndarray) -> np.ndarray:
    """Compute log10 with a tiny floor to avoid log(0)."""
    tiny = np.finfo(arr.dtype).tiny
    return np.log10(np.maximum(arr, tiny))


def _phi_transform(arr: np.ndarray) -> np.ndarray:
    """Training-space transform phi(x) = log10(x) - min(log10(x))."""
    log_arr = _safe_log10(arr)
    return log_arr - float(log_arr.min())


def _to_2d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr[..., 0]
    if arr.ndim == 4:
        return arr[0, ..., 0]
    raise ValueError(f"Cannot convert array with shape {arr.shape} to 2D")


def _shared_limits(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Shared color limits for GT and prediction panels."""
    vmin = float(min(np.min(a), np.min(b)))
    vmax = float(max(np.max(a), np.max(b)))
    if np.isclose(vmin, vmax):
        span = 1e-6 if vmax == 0 else 1e-3 * abs(vmax)
        vmin -= span
        vmax += span
    return vmin, vmax


def _symmetric_limits(arr: np.ndarray) -> tuple[float, float]:
    max_abs = float(np.max(np.abs(arr)))
    if max_abs == 0.0:
        max_abs = 1e-6
    return -max_abs, max_abs


def _plot_training_space(
    sample_id: int,
    gt_phi: np.ndarray,
    pred_phi: np.ndarray,
    output_dir: Path,
    cmap: str,
    residual_cmap: str,
    dpi: int,
):
    gt_2d = _to_2d(gt_phi)
    pred_2d = _to_2d(pred_phi)
    vmin, vmax = _shared_limits(gt_2d, pred_2d)
    residual = pred_2d - gt_2d
    res_vmin, res_vmax = _symmetric_limits(residual)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16.0, 4.8),
        gridspec_kw={"wspace": 0.25},
    )
    im_gt = axes[0].imshow(gt_2d, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title(r"GT $\phi(x)$")
    axes[1].imshow(pred_2d, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title(r"Prediction ($\phi$ space)")
    im_res = axes[2].imshow(residual, cmap=residual_cmap, vmin=res_vmin, vmax=res_vmax)
    axes[2].set_title(r"Residual ($\hat{\phi} - \phi$)")

    for ax in axes:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    cb_phi = fig.colorbar(im_gt, ax=axes[:2], fraction=0.035, pad=0.02)
    cb_phi.set_label(r"$\phi(x) = \log_{10}(x) - \min(\log_{10}(x))$", rotation=90)
    cb_res = fig.colorbar(im_res, ax=axes[2], fraction=0.035, pad=0.05)
    cb_res.set_label(r"Residual ($\log_{10}$ units)", rotation=90)
    fig.tight_layout()

    save_path = output_dir / f"sample_{sample_id:05d}_training_space.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


def _plot_calibrated_space(
    sample_id: int,
    log_gt: np.ndarray,
    pred_calibrated: np.ndarray,
    output_dir: Path,
    cmap: str,
    residual_cmap: str,
    dpi: int,
):
    log_gt_2d = _to_2d(log_gt)
    pred_cal_2d = _to_2d(pred_calibrated)
    vmin, vmax = _shared_limits(log_gt_2d, pred_cal_2d)
    residual = pred_cal_2d - log_gt_2d
    res_vmin, res_vmax = _symmetric_limits(residual)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16.0, 4.8),
        gridspec_kw={"wspace": 0.25},
    )
    im_gt = axes[0].imshow(log_gt_2d, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title(r"GT $\log_{10}(x)$")
    axes[1].imshow(pred_cal_2d, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title(r"Prediction + C (dataset-calibrated)")
    im_res = axes[2].imshow(residual, cmap=residual_cmap, vmin=res_vmin, vmax=res_vmax)
    axes[2].set_title("Residual (calibrated)")

    for ax in axes:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    cb_phi = fig.colorbar(im_gt, ax=axes[:2], fraction=0.035, pad=0.02)
    cb_phi.set_label(r"$\log_{10}(x)$", rotation=90)
    cb_res = fig.colorbar(im_res, ax=axes[2], fraction=0.035, pad=0.05)
    cb_res.set_label(r"Residual ($\log_{10}$ units)", rotation=90)
    fig.tight_layout()

    save_path = output_dir / f"sample_{sample_id:05d}_dataset_calibrated.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


def _generate_plots(
    sample_id: int,
    target_raw: np.ndarray,
    pred_phi: np.ndarray,
    calibration_constant: float | None,
    output_dir: Path,
    calibrated_output_dir: Path | None,
    cmap: str,
    residual_cmap: str,
    include_calibrated: bool,
    dpi: int,
):
    gt_phi = _phi_transform(target_raw)
    _plot_training_space(
        sample_id=sample_id,
        gt_phi=gt_phi,
        pred_phi=pred_phi,
        output_dir=output_dir,
        cmap=cmap,
        residual_cmap=residual_cmap,
        dpi=dpi,
    )

    if include_calibrated:
        if calibration_constant is None:
            raise RuntimeError("Calibration constant is required for calibrated plots.")
        if calibrated_output_dir is None:
            raise RuntimeError("Calibrated output directory is not set.")
        log_gt = _safe_log10(target_raw)
        pred_calibrated = np.asarray(pred_phi) + calibration_constant
        _plot_calibrated_space(
            sample_id=sample_id,
            log_gt=log_gt,
            pred_calibrated=pred_calibrated,
            output_dir=calibrated_output_dir,
            cmap=cmap,
            residual_cmap=residual_cmap,
            dpi=dpi,
        )


def _gather_pred_samples(pred_dir: Path, sample_number: int | None) -> list[tuple[int, Path]]:
    if sample_number is not None:
        path = pred_dir / f"sample_{sample_number:05d}.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"No prediction file found for sample {sample_number} at {path}"
            )
        return [(sample_number, path)]

    items: list[tuple[int, Path]] = []
    for file in sorted(pred_dir.glob("sample_*.npy")):
        try:
            idx = int(file.stem.split("_")[-1])
        except ValueError:
            continue
        items.append((idx, file))
    if not items:
        raise RuntimeError(f"No prediction .npy files found in {pred_dir}")
    return items


def process_samples(
    run_images_dir: Path,
    pred_subdir: str,
    target_data_path: Path,
    seed: int,
    batch_size: int,
    test_ratio: float,
    cmap: str,
    residual_cmap: str,
    sample_number: int | None,
    output_subdir: str,
    calibrated_output_subdir: str,
    include_calibrated: bool,
    calibration_constant: float | None,
    calibration_split: str,
    num_workers: int | None,
    start_method: str,
    dpi: int,
    limit: int | None,
):
    pred_dir = run_images_dir / pred_subdir
    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
    output_dir = pred_dir.parent / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    calibrated_output_dir: Path | None = None
    if include_calibrated:
        calibrated_output_dir = pred_dir.parent / calibrated_output_subdir
        calibrated_output_dir.mkdir(parents=True, exist_ok=True)

    _, _, _, train_test_key = _split_master_key(seed)
    train_indices, test_indices, target_maps = _make_target_loader(
        key=train_test_key,
        target_data_path=target_data_path,
        batch_size=batch_size,
        test_ratio=test_ratio,
    )

    calibration_constant_value = calibration_constant
    if calibration_constant_value is None:
        if calibration_split == "train":
            calib_indices = train_indices
        elif calibration_split == "test":
            calib_indices = test_indices
        else:
            calib_indices = np.arange(target_maps.shape[0])
        calibration_constant_value = _compute_log10_min_constant(
            targets=target_maps, indices=calib_indices, desc=f"Calibrating ({calibration_split})"
        )
    print(f"Using calibration constant C={calibration_constant_value:.6f}")

    pred_items = _gather_pred_samples(pred_dir, sample_number)
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be a positive integer.")
        pred_items = pred_items[:limit]
        print(f"Limiting to first {len(pred_items)} prediction samples (limit={limit}).")

    def _process_single(sample_id: int, pred_path: Path):
        if sample_id >= len(test_indices):
            return f"skip_{sample_id}"
        dataset_idx = int(test_indices[sample_id])
        target_raw = np.asarray(target_maps[dataset_idx])
        pred_sample = np.load(pred_path)
        _generate_plots(
            sample_id=sample_id,
            target_raw=target_raw,
            pred_phi=pred_sample,
            calibration_constant=calibration_constant_value,
            output_dir=output_dir,
            calibrated_output_dir=calibrated_output_dir,
            cmap=cmap,
            residual_cmap=residual_cmap,
            include_calibrated=include_calibrated,
            dpi=dpi,
        )
        return "ok"

    max_workers = num_workers if num_workers and num_workers > 0 else None
    if max_workers == 1:
        for sample_id, pred_path in tqdm(pred_items, desc="Processing samples", unit="sample"):
            _process_single(sample_id, pred_path)
        return

    ctx = mp.get_context(start_method)
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(
            str(target_data_path),
            test_indices,
            calibration_constant_value,
            str(output_dir),
            str(calibrated_output_dir) if calibrated_output_dir is not None else None,
            cmap,
            residual_cmap,
            include_calibrated,
            dpi,
        ),
    ) as executor:
        futures = [
            executor.submit(_proc_task, sample_id, str(pred_path))
            for sample_id, pred_path in pred_items
        ]
        for fut in tqdm(
            as_completed(futures), total=len(futures), desc="Processing samples", unit="sample"
        ):
            fut.result()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare predictions to ground truth in the training space and optionally with a "
            "dataset-level calibrated log10 view."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to the evaluation run directory that contains an images/ folder.",
    )
    parser.add_argument(
        "--pred-subdir",
        type=str,
        default="pred_raw",
        help="Subdirectory under images/ that holds the raw prediction .npy files.",
    )
    parser.add_argument(
        "--target-data",
        type=Path,
        required=True,
        help="Path to the target .npy file from the data directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used during training/evaluation (controls the PRNG split for dataset partitioning).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size used for the original loader (keeps epoch slicing identical).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Test split ratio to match training/evaluation.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional sample number to process (e.g., 12). If omitted, process all predictions.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of prediction samples to process (e.g., 300).",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="magma",
        help="Matplotlib colormap to use for GT and prediction panels.",
    )
    parser.add_argument(
        "--residual-cmap",
        type=str,
        default="coolwarm",
        help="Matplotlib colormap to use for residual panels.",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="training_space_comparison",
        help="Name of the output folder to create under the images/ directory for primary plots.",
    )
    parser.add_argument(
        "--calibrated-output-subdir",
        type=str,
        default="dataset_calibrated_comparison",
        help="Name of the output folder for dataset-level calibrated plots.",
    )
    parser.add_argument(
        "--include-calibrated",
        action="store_true",
        help="Generate the optional dataset-level calibrated comparison plot.",
    )
    parser.add_argument(
        "--calibration-constant",
        type=float,
        default=None,
        help="Precomputed log10 min calibration constant C. If omitted, it will be computed from the dataset.",
    )
    parser.add_argument(
        "--calibration-split",
        type=str,
        default="train",
        choices=["train", "test", "all"],
        help="Split used to compute the calibration constant when not provided.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker processes to use. Set to 1 to disable multiprocessing.",
    )
    parser.add_argument(
        "--start-method",
        type=str,
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="Multiprocessing start method (spawn avoids JAX/fork deadlocks).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved figures (lower for faster rendering/smaller files).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    process_samples(
        run_images_dir=args.run_dir / "images",
        pred_subdir=args.pred_subdir,
        target_data_path=args.target_data,
        seed=args.seed,
        batch_size=args.batch_size,
        test_ratio=args.test_ratio,
        cmap=args.cmap,
        residual_cmap=args.residual_cmap,
        sample_number=args.sample,
        output_subdir=args.output_subdir,
        calibrated_output_subdir=args.calibrated_output_subdir,
        include_calibrated=args.include_calibrated,
        calibration_constant=args.calibration_constant,
        calibration_split=args.calibration_split,
        num_workers=args.num_workers,
        start_method=args.start_method,
        dpi=args.dpi,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
