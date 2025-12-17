import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend for batch plotting
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from tqdm import tqdm


def load_memmaps(run_dir: Path):
    k_vals = np.load(run_dir / "k_vals.npy", mmap_mode="r")

    def _memmap_without_header(path: Path, dtype: np.dtype, cols: int) -> np.memmap:
        bytes_per_elem = np.dtype(dtype).itemsize
        total_bytes = path.stat().st_size
        rows = total_bytes // (bytes_per_elem * cols)
        return np.memmap(path, mode="r", dtype=dtype, shape=(rows, cols))

    k_len = int(k_vals.shape[0])
    target_pk = _memmap_without_header(run_dir / "target_pk.npy", np.float32, k_len)
    si_pk = _memmap_without_header(run_dir / "si_pk.npy", np.float32, k_len)
    return k_vals, target_pk, si_pk


def load_metadata(run_dir: Path, spectra_len: int) -> dict[int, int]:
    metadata_path = run_dir / "power_spectra_metadata.csv"
    if not metadata_path.exists():
        return {i: i for i in range(spectra_len)}

    mapping: dict[int, int] = {}
    with metadata_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 2:
                continue
            sample_idx, spectra_row = int(row[0]), int(row[1])
            mapping[sample_idx] = spectra_row
    return mapping


def plot_sample(
    k_vals: np.ndarray,
    target_pk: np.ndarray,
    si_pk: np.ndarray,
    idx: int,
    save_dir: Path,
    title_prefix: str,
):
    fig = Figure(figsize=(6, 4))
    FigureCanvas(fig)  # attach canvas for Agg backend
    ax = fig.add_subplot(111)
    ax.loglog(k_vals, target_pk, label="Target", linewidth=2.0)
    ax.loglog(k_vals, si_pk, label="SI", linewidth=1.8)
    ax.set_title(f"{title_prefix} Sample {idx:05d}")
    ax.set_xlabel("Wave number k [h/Mpc]")
    ax.set_ylabel("P(k)")
    ax.legend()
    fig.tight_layout()
    save_path = save_dir / f"sample_{idx:05d}.png"
    fig.savefig(save_path, dpi=150)
    return save_path


def plot_mean_and_iqr(
    k_vals: np.ndarray,
    target_pk: np.ndarray,
    si_pk: np.ndarray,
    save_dir: Path,
    title_prefix: str,
):
    """Plot median spectra with interquartile bands for target and SI outputs."""
    target_median = np.median(target_pk, axis=0)
    si_median = np.median(si_pk, axis=0)
    target_q25 = np.percentile(target_pk, 25, axis=0)
    target_q75 = np.percentile(target_pk, 75, axis=0)
    si_q25 = np.percentile(si_pk, 25, axis=0)
    si_q75 = np.percentile(si_pk, 75, axis=0)

    fig = Figure(figsize=(6, 4))
    FigureCanvas(fig)  # attach canvas for Agg backend
    ax = fig.add_subplot(111)
    target_color = "dodgerblue"
    si_color = "seagreen"

    ax.fill_between(k_vals, target_q25, target_q75, color=target_color, alpha=0.25, label="Target IQR")
    ax.fill_between(k_vals, si_q25, si_q75, color=si_color, alpha=0.25, label="SI IQR")
    ax.plot(k_vals, target_median, color=target_color, linewidth=2.2, label="Target median")
    ax.plot(k_vals, si_median, color=si_color, linewidth=2.0, label="SI median")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"{title_prefix} Median with IQR")
    ax.set_xlabel("Wave number k [h/Mpc]")
    ax.set_ylabel("P(k)")
    ax.legend()
    fig.tight_layout()
    save_path = save_dir / "power_spectrum_median_iqr.png"
    fig.savefig(save_path, dpi=150)
    return save_path


_K_VALS: np.ndarray | None = None
_TARGET_PK: np.memmap | None = None
_SI_PK: np.memmap | None = None
_OUT_DIR_STR: str | None = None
_TITLE_PREFIX: str | None = None


def _init_worker(run_dir_str: str, out_dir_str: str, title_prefix: str):
    """Initialize per-process globals for multiprocessing."""
    global _K_VALS, _TARGET_PK, _SI_PK, _OUT_DIR_STR, _TITLE_PREFIX
    rd = Path(run_dir_str)
    kv, tp, sp = load_memmaps(rd)
    _K_VALS = np.asarray(kv)
    _TARGET_PK = tp
    _SI_PK = sp
    _OUT_DIR_STR = out_dir_str
    _TITLE_PREFIX = title_prefix


def _proc_task(sample_idx: int, spectra_row: int):
    if (
        _K_VALS is None
        or _TARGET_PK is None
        or _SI_PK is None
        or _OUT_DIR_STR is None
        or _TITLE_PREFIX is None
    ):
        return None
    if spectra_row >= _TARGET_PK.shape[0]:
        return None
    return plot_sample(
        k_vals=_K_VALS,
        target_pk=np.asarray(_TARGET_PK[spectra_row]),
        si_pk=np.asarray(_SI_PK[spectra_row]),
        idx=sample_idx,
        save_dir=Path(_OUT_DIR_STR),
        title_prefix=_TITLE_PREFIX,
    )


def generate_plots(
    run_dir: Path,
    output_dir: Path | None,
    title_prefix: str,
    num_workers: int | None = None,
):
    k_vals, target_pk, si_pk = load_memmaps(run_dir)
    n_samples = target_pk.shape[0]
    if k_vals.size == 0 or n_samples == 0:
        raise RuntimeError("No spectra data found in the provided run directory.")

    mapping = load_metadata(run_dir, n_samples)

    out_dir = output_dir or (run_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[int, int]] = []
    for sample_idx, spectra_row in mapping.items():
        if spectra_row >= n_samples:
            continue
        jobs.append((sample_idx, spectra_row))
    if not jobs:
        jobs = [(i, i) for i in range(n_samples)]
    jobs.sort(key=lambda x: x[0])
    spectra_rows = [row for _, row in jobs]
    k_vals_array = np.asarray(k_vals)

    # Single-process path (no pool) for easiest debugging or when requested.
    max_workers = num_workers if num_workers and num_workers > 0 else None
    if max_workers == 1:
        for sample_idx, spectra_row in tqdm(
            jobs, total=len(jobs), desc="Plotting spectra", unit="plot"
        ):
            plot_sample(
                k_vals=k_vals_array,
                target_pk=np.asarray(target_pk[spectra_row]),
                si_pk=np.asarray(si_pk[spectra_row]),
                idx=sample_idx,
                save_dir=out_dir,
                title_prefix=title_prefix,
            )
        plot_mean_and_iqr(
            k_vals=k_vals_array,
            target_pk=np.asarray(target_pk[spectra_rows]),
            si_pk=np.asarray(si_pk[spectra_rows]),
            save_dir=out_dir,
            title_prefix=title_prefix,
        )
        return out_dir

    # Multiprocessing path: load memmaps per worker via initializer to avoid large pickles.
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(str(run_dir), str(out_dir), title_prefix),
    ) as executor:
        futures = [
            executor.submit(_proc_task, sample_idx, spectra_row) for sample_idx, spectra_row in jobs
        ]
        with tqdm(total=len(futures), desc="Plotting spectra", unit="plot") as pbar:
            for fut in as_completed(futures):
                fut.result()
                pbar.update(1)
    plot_mean_and_iqr(
        k_vals=k_vals_array,
        target_pk=np.asarray(target_pk[spectra_rows]),
        si_pk=np.asarray(si_pk[spectra_rows]),
        save_dir=out_dir,
        title_prefix=title_prefix,
    )
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Plot SI evaluation spectra")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to an evaluation run directory produced by evaluate_si.py",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots (default: <run-dir>/plots)",
    )
    parser.add_argument(
        "--title-prefix",
        type=str,
        default="SI Power Spectrum",
        help="Prefix for the plot titles",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of threads to use (default: Python-chosen max). Use 1 to disable threading.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None
    saved_dir = generate_plots(
        run_dir=run_dir,
        output_dir=output_dir,
        title_prefix=args.title_prefix,
        num_workers=args.num_workers,
    )
    print(f"Saved plots to {saved_dir}")


if __name__ == "__main__":
    main()
