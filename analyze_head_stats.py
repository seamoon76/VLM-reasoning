import argparse
from pathlib import Path

import numpy as np


def summarize_counts(arr: np.ndarray) -> dict:
    total = int(arr.sum())
    top_idx = int(arr.argmax())
    top_val = int(arr[top_idx])
    top3_idx = list(np.argsort(arr)[-3:][::-1])
    top3_vals = [int(arr[i]) for i in top3_idx]
    return {
        "total": total,
        "top_idx": top_idx,
        "top_val": top_val,
        "top3": list(zip(top3_idx, top3_vals)),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize head top3 counts from .npy files.")
    parser.add_argument("paths", nargs="+", help="Paths to .npy files or folders containing them.")
    parser.add_argument("--file", default="layer_30_top3_counts.npy")
    args = parser.parse_args()

    for raw in args.paths:
        path = Path(raw)
        if path.is_dir():
            path = path / args.file
        if not path.exists():
            print(f"{raw}: missing {path}")
            continue
        arr = np.load(path)
        stats = summarize_counts(arr)
        print(
            f"{path}: total={stats['total']} top={stats['top_idx']} ({stats['top_val']}) "
            f"top3={stats['top3']} mean={stats['mean']:.2f} std={stats['std']:.2f}"
        )


if __name__ == "__main__":
    main()
