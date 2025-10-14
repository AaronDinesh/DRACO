import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.utils import make_transform


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
        default="signed_log1p",
        help="Transform to apply to the images.",
    )
    parser.add_argument("--scale", type=float, default=1.0, help="Scale for the transform.")
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

    transform, _ = make_transform(args.transform, scale=args.scale)

    for i, slice_2d in tqdm(enumerate(data), desc="Processing Images...", total=data.shape[0]):
        # Apply the transform
        transformed_slice = transform(slice_2d)
        # Normalize to 0-255 for image saving
        img_array = np.array(transformed_slice)
        # img_array -= img_array.min()
        img_array /= img_array.max()
        img_array *= 255
        img_array = img_array.astype(np.uint8)

        # Save the image
        img = Image.fromarray(img_array)
        output_path = args.output_dir / f"{args.input_file.stem}_slice_{i:04d}.png"
        img.save(output_path)

    print(f"Saved {len(data)} images to {args.output_dir}")


if __name__ == "__main__":
    main()
