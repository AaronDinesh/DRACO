#!/usr/bin/env python
import argparse
import csv
from pathlib import Path

import jax.numpy as jnp
import jax.random as random
import numpy as np
import pandas as pd


def build_split(
    seed: int,
    test_ratio: float,
    dataset_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_indices, test_indices, perm) matching utils.make_train_test_loaders/eval scripts."""
    master_key = random.key(seed)
    _k1, _k2, _k3, train_test_key = random.split(master_key, 4)
    perm = np.asarray(random.permutation(train_test_key, dataset_len))
    n_test = max(1, int(round(test_ratio * dataset_len)))
    test_indices = perm[:n_test]
    train_indices = perm[n_test:]
    return train_indices, test_indices, perm


def main():
    parser = argparse.ArgumentParser(
        "Create CSV mapping local train/test indices to global indices and cosmos params."
    )
    parser.add_argument("--input-maps", required=True, help="Path to input .npy array (N,H,W[,C])")
    parser.add_argument(
        "--cosmos-params", required=True, help="Path to cosmos params txt/csv (as in training/eval)"
    )
    parser.add_argument("--output-csv", required=True, help="Where to write the mapping CSV")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=0, help="Seed used for the permutation")
    args = parser.parse_args()

    input_maps = np.load(args.input_maps, mmap_mode="r")
    cosmos_params_df = pd.read_csv(args.cosmos_params, header=None, sep=" ")

    repeat_factor: int = input_maps.shape[0] // len(cosmos_params_df)
    cosmos_params_df = cosmos_params_df.loc[
        cosmos_params_df.index.repeat(repeat_factor)
    ].reset_index(drop=True)
    cosmos_params = jnp.asarray(cosmos_params_df.to_numpy(), dtype=jnp.float32)

    if len(input_maps) != len(cosmos_params):
        raise ValueError(
            f"Input maps ({len(input_maps)}) and cosmos params ({len(cosmos_params)}) differ in length"
        )

    train_idx, test_idx, perm = build_split(
        seed=args.seed, test_ratio=args.test_ratio, dataset_len=len(input_maps)
    )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = ["split", "local_idx", "global_idx"] + [
        f"cosmos_param_{i}" for i in range(cosmos_params.shape[1])
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for local, gidx in enumerate(test_idx):
            params = np.asarray(cosmos_params[gidx])
            writer.writerow(["test", local, int(gidx), *params])
        for local, gidx in enumerate(train_idx):
            params = np.asarray(cosmos_params[gidx])
            writer.writerow(["train", local, int(gidx), *params])

    print(f"Wrote mapping to {output_path}")


if __name__ == "__main__":
    main()
