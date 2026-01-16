import argparse
import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from matplotlib.colors import PowerNorm, SymLogNorm
from PIL import Image
from tqdm import tqdm

from src.utils import make_transform


def percentile_abs_histogram_jnp(
    data_mmap,
    q: float = 50.0,
    bins: int = 65536,
    chunk_slices: int = 4,
    eps: float = 0.0,
    dtype=jnp.float32,
) -> float:
    """
    Approximate percentile of |x| over ALL pixels in a (N,H,W) memmap/ndarray,
    using a streaming histogram (deterministic, no sampling).

    Uses only jax.numpy ops for computation, but still reads chunks from the
    numpy memmap in Python and converts each chunk to jnp arrays.

    Returns a Python float.
    """
    n = int(data_mmap.shape[0])

    # Pass 1: find max(|x|) (optionally ignoring values <= eps)
    max_abs = jnp.array(0.0, dtype=dtype)

    num_steps = math.ceil(n / chunk_slices)

    for i in tqdm(range(0, n, chunk_slices), total=num_steps, desc="First pass..", leave=False):
        chunk_np = data_mmap[i : i + chunk_slices]  # numpy memmap slice
        chunk = jnp.asarray(chunk_np, dtype=dtype)
        chunk = jnp.abs(chunk)

        if eps > 0.0:
            # Replace <= eps with 0 so they don't affect max
            chunk = jnp.where(chunk > eps, chunk, 0.0)

        max_abs = jnp.maximum(max_abs, jnp.max(chunk))

    max_abs_val = float(max_abs)
    if max_abs_val <= eps:
        return 0.0

    # Pass 2: histogram accumulation
    hist = jnp.zeros((bins,), dtype=jnp.int64)
    total = jnp.array(0, dtype=jnp.int64)

    for i in tqdm(range(0, n, chunk_slices), total=num_steps, desc="Second pass", leave=False):
        chunk_np = data_mmap[i : i + chunk_slices]
        chunk = jnp.asarray(chunk_np, dtype=dtype)
        chunk = jnp.abs(chunk).reshape(-1)

        if eps > 0.0:
            chunk = chunk[chunk > eps]

        if chunk.size == 0:
            continue

        h, _ = jnp.histogram(chunk, bins=bins, range=(eps, max_abs))
        hist = hist + h.astype(jnp.int64)
        total = total + jnp.sum(h).astype(jnp.int64)

    total_val = int(total)
    if total_val == 0:
        return 0.0

    # CDF -> percentile bin
    target = (q / 100.0) * total_val
    cdf = jnp.cumsum(hist)
    idx = int(jnp.searchsorted(cdf, target, side="left"))

    # Map bin index to value (bin center)
    bin_edges = jnp.linspace(eps, max_abs, bins + 1, dtype=dtype)
    val = 0.5 * (bin_edges[idx] + bin_edges[idx + 1])
    return float(val)


import matplotlib.pyplot as plt


def save_array_plot_simple(array2d, out_path, title=None):
    """
    array2d : (H, W) numpy array
    out_path: Path or str
    """

    a = array2d

    fig, ax = plt.subplots(figsize=(6, 4))

    im = ax.imshow(
        a,
        origin="lower",
        cmap="viridis",
    )

    fig.colorbar(im, ax=ax)

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if title is not None:
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize slices of a .npy tensor.")
    parser.add_argument("--input-file", type=Path, help="Path to the input .npy file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="visualizations",
        help="Directory to save the output images.",
    )
    parser.add_argument(
        "--transform",
        type=str,
        default="log10",
        help="Transform to apply to the images.",
    )
    parser.add_argument("--scale", type=float, default=None, help="Scale for the transform.")
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        data = np.load(args.input_file, mmap_mode="r")

    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_file}")
        return

    if data.ndim != 3 or data.shape[1:] != (256, 256):
        print(f"Error: Input tensor must have shape [N, 256, 256], but got {data.shape}")
        return

    if args.scale is None:
        scale = percentile_abs_histogram_jnp(data, q=95, bins=65536, chunk_slices=4, eps=0.0)
    else:
        scale = args.scale

    transform, _ = make_transform(args.transform, scale=scale)

    for i, slice_2d in tqdm(enumerate(data), desc="Processing Images...", total=args.n):
        if i > args.n:
            break

        # Apply the transform
        transformed_slice = transform(slice_2d)

        # # Normalize to 0-255 for image saving
        # img_array = np.array(transformed_slice)
        # # img_array -= img_array.min()
        # img_array /= img_array.max()
        # img_array *= 255
        # img_array = img_array.astype(np.uint8)
        # or
        # Save the image
        # img = Image.fromarray(img_array)
        output_path = args.output_dir / f"{args.input_file.stem}_slice_{i:04d}.png"

        save_array_plot_simple(transformed_slice, output_path, title=f"Slice {i}")

    print(f"Saved {args.n} images to {args.output_dir}")


if __name__ == "__main__":
    main()
