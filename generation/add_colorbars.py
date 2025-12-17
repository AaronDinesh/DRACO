import argparse
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib
from matplotlib import ticker

# Use non-interactive backend for headless environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

# Reduce matplotlib clipping spam when values sit outside default ranges.
warnings.filterwarnings("ignore", message="Clipping input data to the valid range")

_TARGETS_CACHE: dict[str, np.ndarray] = {}


def _load_targets(path: Path) -> np.ndarray:
    """Load target npy with mmap, cached per process to avoid repeated IO."""
    key = str(path.resolve())
    if key not in _TARGETS_CACHE:
        _TARGETS_CACHE[key] = np.load(path, mmap_mode="r")
    return _TARGETS_CACHE[key]


def _dtype_limits(arr: np.ndarray) -> tuple[float, float] | None:
    """Return dtype min/max for integer types; otherwise None."""
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        return float(info.min), float(info.max)
    return None


def _compute_limits(
    arr: np.ndarray,
    vmin: float | None,
    vmax: float | None,
    percentiles: Tuple[float, float] | None,
    use_dtype_range: bool,
    dtype_limits: tuple[float, float] | None,
) -> tuple[float, float]:
    """Return lower/upper bounds for the colormap without rescaling data."""
    if vmin is not None and vmax is not None:
        return vmin, vmax
    if use_dtype_range and dtype_limits is not None:
        return dtype_limits
    if percentiles is not None:
        lo, hi = np.percentile(arr, percentiles)
        if np.isclose(lo, hi):
            hi = lo + 1e-6
        return float(lo), float(hi)
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    if np.isclose(min_val, max_val):
        max_val = min_val + 1e-6
    return min_val, max_val


def _add_colorbar_to_image(
    src_path: Path,
    dst_dir: Path,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    percentiles: Tuple[float, float] | None,
    dpi: int,
    use_dtype_range: bool,
    colorbar_label: str | None,
    log10_ticks: bool,
    targets_npy: Path | None,
    transform: str,
    sample_idx: int,
) -> None:
    dst_path = dst_dir / src_path.name
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src_path) as im:
        arr_raw = np.array(im, dtype=np.uint8)

    if arr_raw.ndim == 3 and arr_raw.shape[-1] == 1:
        arr_raw = arr_raw[..., 0]

    dtype_limits = _dtype_limits(arr_raw)
    arr = arr_raw.astype(np.float32)

    vmin_img, vmax_img = _compute_limits(
        arr, vmin, vmax, percentiles, use_dtype_range, dtype_limits
    )

    # If we have ground-truth targets and the images are in relative log10 space,
    # recover the per-sample log minimum so we can show colorbar labels in true scale.
    ln_min: float | None = None
    if targets_npy is not None and transform == "log10":
        targets = _load_targets(targets_npy)
        if sample_idx < 0 or sample_idx >= len(targets):
            raise IndexError(
                f"sample_idx {sample_idx} out of range for targets of length {len(targets)}"
            )
        target = np.asarray(targets[sample_idx])
        tiny = np.finfo(
            target.dtype if np.issubdtype(target.dtype, np.floating) else np.float32
        ).tiny
        ln_min = float(np.log(np.maximum(target, tiny)).min())
    elif targets_npy is not None and transform != "none":
        warnings.warn(
            f"targets_npy provided but transform '{transform}' is not handled; proceeding without inverse."
        )

    height, width = arr.shape[0], arr.shape[1]
    fig_w = width / dpi
    fig_h = height / dpi
    fig, ax = plt.subplots(figsize=(6, 4))
    im_artist = ax.imshow(arr.astype(np.uint8), cmap=cmap, vmin=0, vmax=int(np.max(arr)))
    ax.axis("on")
    cbar = plt.colorbar(im_artist, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    if colorbar_label:
        cbar.set_label(colorbar_label, fontsize=9)
    if ln_min is not None:
        ln10 = np.log(10.0)

        def _fmt(val: float, _: object) -> str:
            return f"{np.exp(val * ln10 + ln_min):.3g}"

        cbar.formatter = ticker.FuncFormatter(_fmt)
        cbar.update_ticks()
    elif log10_ticks:
        cbar.formatter = ticker.FuncFormatter(lambda val, _: rf"$10^{{{val:g}}}$")
        cbar.update_ticks()

    fig.tight_layout(pad=0.1)
    fig.savefig(dst_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _process_images(
    paths: Iterable[Path],
    output_dir: Path,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    percentiles: Tuple[float, float] | None,
    dpi: int,
    workers: int,
    use_dtype_range: bool,
    colorbar_label: str | None,
    log10_ticks: bool,
    targets_npy: Path | None,
    transform: str,
) -> None:
    paths_list = list(paths)
    total = len(paths_list)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for _ in tqdm(
            executor.map(
                _add_colorbar_to_image,
                paths_list,
                repeat(output_dir, total),
                repeat(cmap, total),
                repeat(vmin, total),
                repeat(vmax, total),
                repeat(percentiles, total),
                repeat(dpi, total),
                repeat(use_dtype_range, total),
                repeat(colorbar_label, total),
                repeat(log10_ticks, total),
                repeat(targets_npy, total),
                repeat(transform, total),
                range(total),
            ),
            total=total,
            desc="Adding colorbars",
        ):
            pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add matplotlib colorbars to existing PNG images in bulk."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing the source images (e.g., generated samples).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write colorbar-augmented images.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.png",
        help="Glob pattern for images to process (default: *.png).",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="gray",
        help="Matplotlib colormap name to apply (default: gray for black/white inputs).",
    )
    parser.add_argument(
        "--percentiles",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="Optional lower/upper percentiles to set color limits (e.g., 1 99).",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        help="Optional fixed lower bound for color scaling (overrides percentiles).",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        help="Optional fixed upper bound for color scaling (overrides percentiles).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for saved figures; higher values increase output resolution.",
    )
    parser.add_argument(
        "--use-dtype-range",
        action="store_true",
        help="If set, use the full integer dtype range (e.g., 0-255 for uint8) for color limits.",
    )
    parser.add_argument(
        "--label",
        type=str,
        help="Optional text/LaTeX label to display alongside the colorbar.",
    )
    parser.add_argument(
        "--log10-ticks",
        action="store_true",
        help="Format colorbar ticks as powers of ten (values represent log10 magnitudes).",
    )
    parser.add_argument(
        "--targets-npy",
        type=Path,
        help=(
            "Optional npy/npz of target images aligned with inputs. "
            "If provided with --transform log10, colorbar labels are restored to true scale."
        ),
    )
    parser.add_argument(
        "--transform",
        type=str,
        default="none",
        choices=["none", "log10"],
        help="Name of transform applied to images; log10 enables per-sample inverse using targets.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="Number of parallel processes to use (default: all available cores).",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    percentiles: Tuple[float, float] | None = None
    if args.percentiles is not None:
        if len(args.percentiles) != 2:
            raise ValueError("Provide exactly two percentile values: LOW HIGH.")
        percentiles = (float(args.percentiles[0]), float(args.percentiles[1]))

    paths = sorted(args.input_dir.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No images matching '{args.pattern}' found in {args.input_dir}")

    if args.transform == "log10" and args.targets_npy is None:
        raise ValueError("--transform log10 requires --targets-npy to recover true scale.")

    workers = args.workers or 1
    _process_images(
        paths=paths,
        output_dir=args.output_dir,
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
        percentiles=percentiles,
        dpi=args.dpi,
        workers=workers,
        use_dtype_range=args.use_dtype_range,
        colorbar_label=args.label,
        log10_ticks=args.log10_ticks,
        targets_npy=args.targets_npy,
        transform=args.transform,
    )


if __name__ == "__main__":
    main()
