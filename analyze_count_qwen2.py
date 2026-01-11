import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_heatmap(
    matrix: np.ndarray,
    out_path: Path,
    title: str,
    top_pixels: list[tuple[int, int, float]] | None = None,
    topk_label: str = "Top",
) -> None:
    """
    matrix: [L,H]
    top_pixels: list of (l, h, score) with 0-based indices.
    """
    L = matrix.shape[0]
    fig, ax = plt.subplots(figsize=(12, max(4, L * 0.35)), dpi=150)
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest")

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title if title else "")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    # ---- overlay top-k markers ----
    if top_pixels:
        for rank, (l0, h0, _score) in enumerate(top_pixels, start=1):
            # arrow from left to the center of the cell
            ax.annotate(
                "",                      # no text, just arrow
                xy=(h0 - 0.5, l0),              # arrow head (target cell center)
                xytext=(h0 - 1.5, l0),    # arrow tail (from left)
                arrowprops=dict(
                    arrowstyle="->",
                    color="#FF0000",
                    linewidth=5.0,
                    shrinkA=0,
                    shrinkB=0,
                ),
                zorder=6,
            )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_head_curve(values_L: np.ndarray, out_path: Path, title: str) -> None:
    L = values_L.shape[0]
    fig, ax = plt.subplots(figsize=(10, 3), dpi=150)
    ax.plot(list(range(1, L + 1)), values_L, marker="o", linewidth=1.5)
    ax.set_xlabel("Layer (1-based)")
    ax.set_ylabel("Value")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def reduce_over_level(x_LH: np.ndarray, reduce_level: str) -> np.ndarray:
    if reduce_level == "max":
        return x_LH.max(axis=0)
    if reduce_level == "mean":
        return x_LH.mean(axis=0)
    raise ValueError("reduce_level must be 'max' or 'mean'")


def topk_pixels_2d(
    x_LH: np.ndarray,
    k: int,
    mode: str = "raw",
) -> list[tuple[int, int, float]]:
    """
    Return top-k (l, h, value_for_sorting) for a [L,H] matrix, sorted by value desc.
    l,h are 0-based indices.
    mode:
      - "raw": sort by x_LH
      - "abs": sort by abs(x_LH)
    """
    if x_LH.ndim != 2:
        raise ValueError(f"topk_pixels_2d expects [L,H], got {x_LH.shape}")
    L, H = x_LH.shape
    flat = x_LH.reshape(-1)

    if mode == "raw":
        score_flat = flat
    elif mode == "abs":
        score_flat = np.abs(flat)
    else:
        raise ValueError("mode must be 'raw' or 'abs'")

    k = int(min(k, score_flat.size))
    if k <= 0:
        return []

    idx = np.argpartition(score_flat, -k)[-k:]
    idx = idx[np.argsort(score_flat[idx])[::-1]]

    out: list[tuple[int, int, float]] = []
    for t in idx:
        l = int(t // H)
        h = int(t % H)
        out.append((l, h, float(score_flat[t])))
    return out


def write_top_pixels_report(
    out_path: Path,
    title: str,
    x_LH: np.ndarray,
    top_pixels: list[tuple[int, int, float]],
    names: list[str],
    per_corner_maps: list[np.ndarray],
    scale: float,
    suffix: str,
    diff_brightness: str,
) -> None:
    """
    x_LH: the map we are reporting on (already scaled or not doesn't matter; we use scale for printing raw values)
    per_corner_maps: list of [L,H] maps for each corner (usually M[i] scaled or residual[i] scaled) to print at same (l,h)
    """
    with open(out_path, "w") as f:
        f.write(f"{title}\n")
        f.write(f"shape L,H = {x_LH.shape[0]},{x_LH.shape[1]}\n")
        f.write(f"scale_suffix={suffix}, scale={scale}\n")
        f.write(f"diff_brightness={diff_brightness}\n")
        f.write("\n")
        for rank, (l0, h0, v_sort) in enumerate(top_pixels, start=1):
            v_main = float(x_LH[l0, h0])
            per_corner = [float(m[l0, h0]) for m in per_corner_maps]
            f.write(
                f"#{rank}: layer={l0+1} head={h0}  "
                f"main_value_{suffix}={v_main:.6f}  "
                f"sort_score_{suffix}={v_sort:.6f}  "
                f"corners={dict(zip(names, [round(x, 6) for x in per_corner]))}\n"
            )


def main():
    parser = argparse.ArgumentParser(description="Shared/unique head analysis on COUNT cube (count_LHN).")
    parser.add_argument("folders", nargs=4, help="4 corner folders (topleft/topright/bottomleft/bottomright)")
    parser.add_argument("--out-dir", default="count_shared_unique")
    parser.add_argument("--reduce-level", choices=["max", "mean"], default="max",
                        help="How to reduce over layer dimension when choosing best head.")
    parser.add_argument("--use-scale", choices=["freq", "count"], default="count",
                        help="Whether to visualize values as frequency (0..1) or count scale (0..N).")

    # existing (mean map top-k)
    parser.add_argument("--topk-pixels", type=int, default=3,
                        help="Top-k brightest pixels (layer, head) from the shared mean map.")

    # diff map top-k + brightness definition
    parser.add_argument("--topk-diff-pixels", type=int, default=3,
                        help="Top-k brightest pixels (layer, head) for EACH difference map (residual).")
    parser.add_argument("--diff-brightness", choices=["abs", "raw"], default="abs",
                        help="How to define 'brightest' on residual maps: "
                             "'abs' uses |residual|, 'raw' uses residual itself (max).")

    args = parser.parse_args()

    folders = [Path(x) for x in args.folders]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cubes = []
    names = []
    for f in folders:
        p = f / "count_LHN.npy"
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. You must run stats with --save-diffs.")
        c = np.load(p)  # [L,H,N] uint8 0/1
        if c.ndim != 3:
            raise ValueError(f"{p}: expected [L,H,N], got {c.shape}")
        cubes.append(c.astype(np.float32))
        names.append(f.name)

    L, H, N = cubes[0].shape
    for i, c in enumerate(cubes):
        if c.shape != (L, H, N):
            raise ValueError(f"Shape mismatch: {folders[i]} has {c.shape}, expected {(L,H,N)}")

    # mean over count dim -> [L,H] (frequency 0..1)
    M = [c.mean(axis=2) for c in cubes]          # list of [L,H] freq
    M_stack = np.stack(M, axis=0)                # [4,L,H]
    M_shared = M_stack.mean(axis=0)              # [L,H] freq

    # shared head (head-wise score by reducing over layers)
    shared_score = reduce_over_level(M_shared, args.reduce_level)  # [H] freq
    shared_best = int(shared_score.argmax())

    # unique heads: residual in freq space
    residuals = []
    unique_best = []
    unique_score = []
    for i in range(4):
        R = M[i] - M_shared        # [L,H] freq residual
        residuals.append(R)
        s = reduce_over_level(R, args.reduce_level)
        unique_score.append(s)
        unique_best.append(int(s.argmax()))

    # scaling for visualization/printing
    if args.use_scale == "count":
        scale = float(N)
        suffix = "count"
    else:
        scale = 1.0
        suffix = "freq"

    # scaled maps used for plotting and reporting
    M_scaled = [m * scale for m in M]                     # per-corner mean maps
    M_shared_scaled = M_shared * scale                    # shared mean map
    residuals_scaled = [r * scale for r in residuals]     # per-corner residual maps

    # ---------- mean map: top-k brightest ----------
    topk_mean = int(args.topk_pixels)
    top_pixels_mean = topk_pixels_2d(M_shared_scaled, topk_mean, mode="raw")

    # ---------- shared mean heatmap (with Top-K overlay) ----------
    plot_heatmap(
        M_shared_scaled,
        out_dir / f"mean_shared_M_heatmap_{suffix}.png",
        title="",
        top_pixels=top_pixels_mean,
        topk_label="MeanTop",
    )

    # report for mean map
    write_top_pixels_report(
        out_dir / "top_pixels_shared.txt",
        title="Top pixels in shared mean map (M_shared)",
        x_LH=M_shared_scaled,
        top_pixels=top_pixels_mean,
        names=names,
        per_corner_maps=M_scaled,   # show per-corner mean at same (l,h)
        scale=scale,
        suffix=suffix,
        diff_brightness="N/A",
    )

    print(f"[saved] {out_dir / f'mean_shared_M_heatmap_{suffix}.png'}")
    print(f"[saved] {out_dir / 'top_pixels_shared.txt'}")
    print(f"\nTop-{topk_mean} brightest pixels in shared mean map (M_shared):")
    for rank, (l0, h0, _v_sort) in enumerate(top_pixels_mean, start=1):
        per_corner = [float(M_scaled[i][l0, h0]) for i in range(4)]
        print(
            f"  #{rank}: layer={l0+1} head={h0}  shared={M_shared_scaled[l0, h0]:.6f}  "
            + "  ".join([f"{names[i]}={per_corner[i]:.6f}" for i in range(4)])
        )

    # ---------- diff maps (residuals) top-k brightest + reports ----------
    topk_diff = int(args.topk_diff_pixels)
    if topk_diff > 0:
        for i in range(4):
            R_scaled = residuals_scaled[i]  # [L,H]
            top_pixels_diff = topk_pixels_2d(R_scaled, topk_diff, mode=args.diff_brightness)

            report_path = out_dir / f"top_pixels_diff_{names[i]}.txt"
            write_top_pixels_report(
                report_path,
                title=f"Top pixels in difference map (residual): {names[i]} - shared",
                x_LH=R_scaled,
                top_pixels=top_pixels_diff,
                names=names,
                per_corner_maps=residuals_scaled,  # show all corners' residual at same (l,h)
                scale=scale,
                suffix=suffix,
                diff_brightness=args.diff_brightness,
            )

            print(f"[saved] {report_path}")
            print(f"\nTop-{topk_diff} brightest pixels in diff map ({names[i]} - shared), mode={args.diff_brightness}:")
            for rank, (l0, h0, v_sort) in enumerate(top_pixels_diff, start=1):
                vals_all = [float(residuals_scaled[j][l0, h0]) for j in range(4)]
                print(
                    f"  #{rank}: layer={l0+1} head={h0}  "
                    f"{names[i]}_residual={R_scaled[l0, h0]:.6f}  sort_score={v_sort:.6f}  "
                    + "  ".join([f"{names[j]}={vals_all[j]:.6f}" for j in range(4)])
                )

    # ---------- Existing outputs (shared/unique heads) ----------
    with open(out_dir / "shared_head.txt", "w") as f:
        f.write(f"L,H,N={L},{H},{N}\n")
        f.write(f"reduce_level={args.reduce_level}\n")
        f.write(f"shared_best_head={shared_best}\n")
        f.write(f"shared_best_score_{suffix}={shared_score[shared_best]*scale:.6f}\n")

    with open(out_dir / "unique_heads.txt", "w") as f:
        f.write(f"L,H,N={L},{H},{N}\n")
        f.write(f"reduce_level={args.reduce_level}\n")
        for i in range(4):
            f.write(
                f"{names[i]} unique_best_head={unique_best[i]} "
                f"score_{suffix}={unique_score[i][unique_best[i]]*scale:.6f}\n"
            )

    # ---------- residual plots (with Top-K overlay) ----------
    for i in range(4):
        top_pixels_diff = topk_pixels_2d(
            residuals_scaled[i],
            int(args.topk_diff_pixels),
            mode=args.diff_brightness
        )

        plot_heatmap(
            residuals_scaled[i],
            out_dir / f"unique_residual_{names[i]}_{suffix}.png",
            title="",
            top_pixels=top_pixels_diff,
            topk_label=f"DiffTop({names[i]})",
        )

        plot_head_curve(
            residuals_scaled[i][:, unique_best[i]],
            out_dir / f"unique_head_curve_{names[i]}_{suffix}.png",
            f"Unique best head={unique_best[i]} residual curve ({suffix}): {names[i]}"
        )

    print(f"\nShared best head = {shared_best} (reduce_level={args.reduce_level})")
    for i in range(4):
        print(f"{names[i]} unique best head = {unique_best[i]}")


if __name__ == "__main__":
    main()
