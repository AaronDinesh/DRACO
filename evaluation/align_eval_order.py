import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


def _hash_row(params: np.ndarray, pk: np.ndarray, decimals: int) -> str:
    """Stable-ish hash for matching rows across runs."""
    return hashlib.sha1(
        np.round(params, decimals=decimals).tobytes() + np.round(pk, decimals=decimals).tobytes()
    ).hexdigest()


def _hash_params_only(params: np.ndarray, decimals: int) -> str:
    return hashlib.sha1(np.round(params, decimals=decimals).tobytes()).hexdigest()


def _hash_image(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _memmap_without_header(path: Path, dtype: np.dtype, cols: int) -> np.memmap:
    bytes_per_elem = np.dtype(dtype).itemsize
    total_bytes = path.stat().st_size
    if cols <= 0:
        raise ValueError(f"Invalid column count ({cols}) for {path}")
    rows = total_bytes // (bytes_per_elem * cols)
    if rows * bytes_per_elem * cols != total_bytes:
        raise ValueError(
            f"Size mismatch when memmapping {path}: total bytes {total_bytes} not divisible by {cols}*{bytes_per_elem}"
        )
    return np.memmap(path, mode="r", dtype=dtype, shape=(rows, cols))


def _load_arrays(run_dir: Path, allow_pickle: bool):
    k_vals = np.load(run_dir / "k_vals.npy", allow_pickle=allow_pickle)
    k_len = int(k_vals.shape[0])

    target_pk = _memmap_without_header(run_dir / "target_pk.npy", np.float32, k_len)

    cosmos_path = run_dir / "cosmos_params.npy"
    cosmos: np.ndarray
    try:
        cosmos = np.load(cosmos_path, allow_pickle=allow_pickle)
        if cosmos.dtype == object:
            raise ValueError("cosmos_params.npy stored as object dtype; fall back to raw memmap")
    except Exception:
        # Infer cosmos_params columns from metadata if available; otherwise from file size and row count.
        meta_path = run_dir / "power_spectra_metadata.csv"
        cosmos_cols: int | None = None
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                header = f.readline().strip().split(",")
                if len(header) >= 3:
                    cosmos_cols = len(header) - 2
        if cosmos_cols is None:
            rows = target_pk.shape[0]
            bytes_per_elem = np.dtype(np.float32).itemsize
            total_bytes = cosmos_path.stat().st_size
            cosmos_cols = total_bytes // (bytes_per_elem * rows)
        cosmos = _memmap_without_header(cosmos_path, np.float32, cosmos_cols)

    return k_vals, cosmos, target_pk


def align_runs(
    ref_dir: Path,
    other_dir: Path,
    out_dir: Path,
    pred_filename: str,
    hash_decimals: int,
    copy_images: bool,
    allow_pickle: bool,
    match_on: str,
) -> None:
    ref_k, ref_params, ref_target_pk = _load_arrays(ref_dir, allow_pickle=allow_pickle)
    other_k, other_params, other_target_pk = _load_arrays(other_dir, allow_pickle=allow_pickle)
    if ref_k.shape[0] != other_k.shape[0]:
        raise ValueError("k_vals lengths differ between runs; cannot align.")
    k_len = int(ref_k.shape[0])
    other_pred_pk = _memmap_without_header(other_dir / pred_filename, np.float32, k_len)

    if len(ref_params) != len(other_params):
        raise ValueError("Runs have different dataset lengths; cannot align.")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images" / "target").mkdir(parents=True, exist_ok=True)
    (out_dir / "images" / "pred").mkdir(parents=True, exist_ok=True)

    mapping: list[int] = []

    if match_on == "target-image":
        src_t_dir = other_dir / "images" / "target"
        ref_t_dir = ref_dir / "images" / "target"
        if not src_t_dir.exists() or not ref_t_dir.exists():
            raise ValueError(
                "target-image matching requires target images in both runs (images/target/*.png)."
            )
        lookup: dict[str, list[int]] = {}
        for idx in tqdm(range(len(other_params)), desc="Indexing other run images", unit="img"):
            img_path = src_t_dir / f"sample_{idx:05d}.png"
            if not img_path.exists():
                continue
            lookup.setdefault(_hash_image(img_path), []).append(idx)
        for i in tqdm(range(len(ref_params)), desc="Matching images to ref", unit="img"):
            img_path = ref_t_dir / f"sample_{i:05d}.png"
            if not img_path.exists():
                raise ValueError(f"Ref image missing: {img_path}")
            h = _hash_image(img_path)
            hits = lookup.get(h, [])
            if not hits:
                raise ValueError(f"No matching target image hash for ref index {i}")
            mapping.append(hits.pop())
    else:
        # Build lookup from other-run (params +/- target_pk) -> indices.
        lookup: dict[str, list[int]] = {}
        for idx, (p, pk) in enumerate(
            tqdm(
                zip(other_params, other_target_pk),
                total=len(other_params),
                desc="Indexing other run",
                unit="row",
            )
        ):
            if match_on == "params":
                h = _hash_params_only(p, decimals=hash_decimals)
            else:
                h = _hash_row(p, pk, decimals=hash_decimals)
            lookup.setdefault(h, []).append(idx)

        for i, (p, pk) in enumerate(
            tqdm(
                zip(ref_params, ref_target_pk),
                total=len(ref_params),
                desc="Matching to ref",
                unit="row",
            )
        ):
            if match_on == "params":
                h = _hash_params_only(p, decimals=hash_decimals)
            else:
                h = _hash_row(p, pk, decimals=hash_decimals)
            hits = lookup.get(h, [])
            if not hits:
                raise ValueError(
                    f"No matching row in other run for ref index {i} using match_on='{match_on}'. "
                    "Try --match-on params or lower --hash-decimals, or use --match-on target-image."
                )
            mapping.append(hits.pop())

    # Allocate aligned arrays.
    aligned_params = np.memmap(
        out_dir / "cosmos_params.npy",
        mode="w+",
        dtype=np.float32,
        shape=ref_params.shape,
    )
    aligned_target_pk = np.memmap(
        out_dir / "target_pk.npy",
        mode="w+",
        dtype=np.float32,
        shape=ref_target_pk.shape,
    )
    aligned_pred_pk = np.memmap(
        out_dir / pred_filename,
        mode="w+",
        dtype=np.float32,
        shape=other_pred_pk.shape,
    )

    # Copy/symlink images if present.
    src_t_dir = other_dir / "images" / "target"
    src_p_dir = other_dir / "images" / "pred"
    dst_t_dir = out_dir / "images" / "target"
    dst_p_dir = out_dir / "images" / "pred"

    for ref_idx, other_idx in enumerate(
        tqdm(mapping, total=len(mapping), desc="Realigning data", unit="row")
    ):
        aligned_params[ref_idx] = other_params[other_idx]
        aligned_target_pk[ref_idx] = other_target_pk[other_idx]
        aligned_pred_pk[ref_idx] = other_pred_pk[other_idx]

        if copy_images:
            for src_dir, dst_dir in (
                (src_t_dir, dst_t_dir),
                (src_p_dir, dst_p_dir),
            ):
                src = src_dir / f"sample_{other_idx:05d}.png"
                if src.exists():
                    dst = dst_dir / f"sample_{ref_idx:05d}.png"
                    if dst.exists():
                        dst.unlink()
                    dst.write_bytes(src.read_bytes())

    aligned_params.flush()
    aligned_target_pk.flush()
    aligned_pred_pk.flush()

    # Rewrite metadata CSV to reflect new ordering.
    meta_path = out_dir / "power_spectra_metadata.csv"
    with meta_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["sample_idx", "spectra_row"] + [
            f"cosmo_param_{i}" for i in range(ref_params.shape[1])
        ]
        writer.writerow(header)
        for i, p in enumerate(ref_params):
            writer.writerow([i, i, *p.tolist()])

    # Save mapping and provenance info.
    (out_dir / "mapping.json").write_text(
        json.dumps(
            {
                "ref_dir": str(ref_dir),
                "other_dir": str(other_dir),
                "pred_filename": pred_filename,
                "hash_decimals": hash_decimals,
                "other_index_for_ref": mapping,
            },
            indent=2,
        )
    )

    # Copy k_vals.npy from the reference run (ordering anchor).
    k_vals_path = ref_dir / "k_vals.npy"
    if k_vals_path.exists():
        np.save(out_dir / "k_vals.npy", np.load(k_vals_path, allow_pickle=allow_pickle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align one evaluation run's ordering to another without re-running models."
    )
    parser.add_argument(
        "--ref-dir", required=True, type=Path, help="Canonical run directory (order to follow)."
    )
    parser.add_argument("--other-dir", required=True, type=Path, help="Run directory to reorder.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for aligned files (default: other-dir/aligned_to_ref).",
    )
    parser.add_argument(
        "--pred-filename",
        type=str,
        default="gan_pk.npy",
        help="Prediction spectra filename to realign (e.g., gan_pk.npy or si_pk.npy).",
    )
    parser.add_argument(
        "--hash-decimals",
        type=int,
        default=6,
        help="Round floats to this many decimals before hashing to match rows.",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip re-linking images; only realign arrays.",
    )
    parser.add_argument(
        "--allow-pickle",
        action="store_true",
        help="Load .npy files with allow_pickle=True (use only if you trust the data).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing out-dir (otherwise we abort if it exists).",
    )
    parser.add_argument(
        "--match-on",
        choices=["params+pk", "params", "target-image"],
        default="params+pk",
        help="Alignment key: params+pk (default), params only, or target-image hashes.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir or (args.other_dir / "aligned_to_ref")
    if out_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output directory {out_dir} already exists. "
            "Use --overwrite to reuse it or choose a different --out-dir."
        )
    align_runs(
        ref_dir=args.ref_dir,
        other_dir=args.other_dir,
        out_dir=out_dir,
        pred_filename=args.pred_filename,
        hash_decimals=args.hash_decimals,
        copy_images=not args.no_images,
        allow_pickle=args.allow_pickle,
        match_on=args.match_on,
    )


if __name__ == "__main__":
    main()
