import argparse
from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt


LAYER_RE = re.compile(r"layer_(\d+)_diff_abs\.npy$")


def summarize_single(diff_abs: np.ndarray) -> dict:
    top1_idx = diff_abs.argmax(axis=1)
    counts = np.bincount(top1_idx, minlength=diff_abs.shape[1])
    sorted_vals = np.sort(diff_abs, axis=1)
    top1 = sorted_vals[:, -1]
    top2 = sorted_vals[:, -2]
    median = np.median(sorted_vals, axis=1)
    return {
        "counts": counts,
        "top1_mean": float(top1.mean()),
        "top2_mean": float(top2.mean()),
        "median_mean": float(median.mean()),
        "top1_minus_median_mean": float((top1 - median).mean()),
    }


def find_layer_files(folder: Path) -> list[tuple[int, Path]]:
    items = []
    for path in folder.glob("layer_*_diff_abs.npy"):
        match = LAYER_RE.search(path.name)
        if not match:
            continue
        layer_id = int(match.group(1))
        items.append((layer_id, path))
    items.sort(key=lambda x: x[0])
    return items


def plot_heatmap(matrix: np.ndarray, layers: list[int], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, max(4, len(layers) * 0.35)), dpi=150)
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest")
    ax.set_xlabel("Head index")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    yticks = list(range(len(layers)))
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(l) for l in layers])
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_peak_curve(
    peaks: np.ndarray,
    layers: list[int],
    top_heads: list[int],
    out_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 3), dpi=150)
    ax.plot(layers, peaks, marker="o", linewidth=1.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Top1 count")
    ax.set_title(title)
    for layer, peak, head in zip(layers, peaks, top_heads):
        ax.annotate(
            str(head),
            (layer, peak),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze diff_abs arrays (single layer or all layers)."
    )
    parser.add_argument("paths", nargs="+", help="Folders or .npy files to analyze.")
    parser.add_argument("--file", default="layer_30_diff_abs.npy")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--all-layers", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    for raw in args.paths:
        path = Path(raw)
        if path.is_dir() and args.all_layers:
            layer_files = find_layer_files(path)
            if not layer_files:
                print(f"{raw}: no layer_*_diff_abs.npy found")
                continue
            layers = [layer for layer, _ in layer_files]
            matrices = []
            peaks = []
            top_heads = []
            for layer_id, file_path in layer_files:
                diff_abs = np.load(file_path)
                stats = summarize_single(diff_abs)
                counts = stats["counts"].astype(float)
                if args.normalize:
                    counts = counts / diff_abs.shape[0]
                matrices.append(counts)
                peaks.append(counts.max())
                top_heads.append(int(counts.argmax()))
            matrix = np.stack(matrices, axis=0)
            peaks = np.array(peaks)

            out_dir = Path(args.out_dir) if args.out_dir else path
            out_dir.mkdir(parents=True, exist_ok=True)
            tag = "ratio" if args.normalize else "count"
            corner_label = path.name
            plot_heatmap(
                matrix,
                layers,
                out_dir / f"diff_abs_top1_{tag}_heatmap.png",
                f"Top1 head {tag} per layer ({corner_label})",
            )
            plot_peak_curve(
                peaks,
                layers,
                top_heads,
                out_dir / f"diff_abs_top1_{tag}_peak.png",
                f"Top1 {tag} peak per layer ({corner_label})",
            )
            print(f"{path}: saved heatmap and peak plots to {out_dir}")
            continue

        if path.is_dir():
            path = path / args.file
        if not path.exists():
            print(f"{raw}: missing {path}")
            continue
        diff_abs = np.load(path)
        if diff_abs.ndim != 2:
            print(f"{path}: expected 2D array, got {diff_abs.shape}")
            continue
        stats = summarize_single(diff_abs)
        counts = stats["counts"]
        top_heads = np.argsort(counts)[-args.topk:][::-1]

        print(f"{path}: samples={diff_abs.shape[0]}")
        print(
            " top1_mean={:.6f} top2_mean={:.6f} median_mean={:.6f} top1_minus_median_mean={:.6f}".format(
                stats["top1_mean"],
                stats["top2_mean"],
                stats["median_mean"],
                stats["top1_minus_median_mean"],
            )
        )
        print(
            " top1_head_freq:",
            " ".join(f"{h}:{int(counts[h])}" for h in top_heads),
        )


if __name__ == "__main__":
    main()
