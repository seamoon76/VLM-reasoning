import os
import re
import math
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# -----------------------------
# Config
# -----------------------------
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
REL_WORDS = ["left", "right", "above", "below"]

SPECIAL_PAT = re.compile(r"^<\|.*\|>$")
PUNCT_PAT = re.compile(r"^[\.\,\!\?\:\;\(\)\[\]\{\}\"\'\-]+$")

def clean_piece(tok: str) -> str:
    return tok.replace("Ġ", "").replace("Ċ", "\\n")

def is_bad(tok_raw: str) -> bool:
    tc = clean_piece(tok_raw)
    if tc == "\\n":
        return True
    if SPECIAL_PAT.match(tok_raw) is not None:
        return True
    if "image_pad" in tok_raw or "vision_" in tok_raw:
        return True
    if PUNCT_PAT.match(tc) is not None:
        return True
    return False

def sanitize_fname(s: str, max_len=40):
    s = s.replace("/", "_")
    s = re.sub(r"[^a-zA-Z0-9_\-\.\u4e00-\u9fff]+", "", s)
    return (s[:max_len] if s else "EMPTY")

def guess_grid_from_n(n: int):
    s = int(round(math.sqrt(n)))
    if s * s == n:
        return (s, s)
    best = None
    for a in range(1, int(math.sqrt(n)) + 1):
        if n % a == 0:
            b = n // a
            if best is None or abs(a - b) < abs(best[0] - best[1]):
                best = (a, b)
    return best if best is not None else (1, n)

def find_caption_span(tokens_raw):
    ve = None
    for i, t in enumerate(tokens_raw):
        if t == "<|vision_end|>" or "vision_end" in t:
            ve = i
            break
    if ve is None:
        raise RuntimeError("Cannot find <|vision_end|>")

    im_end = None
    for i in range(ve + 1, len(tokens_raw)):
        if tokens_raw[i] == "<|im_end|>":
            im_end = i
            break
    if im_end is None:
        raise RuntimeError("Cannot find <|im_end|> after vision_end")

    return ve + 1, im_end - 1  # inclusive

def group_subwords_with_G(tokens_raw, attn_TN, start, end, merge="sum"):
    groups = []
    cur_ids, cur_vecs, cur_word = [], [], ""

    def flush():
        nonlocal cur_ids, cur_vecs, cur_word
        if not cur_ids:
            return
        if merge == "sum":
            agg = np.sum(cur_vecs, axis=0)
        elif merge == "mean":
            agg = np.mean(cur_vecs, axis=0)
        elif merge == "max":
            agg = np.max(cur_vecs, axis=0)
        else:
            raise ValueError("merge must be sum/mean/max")
        groups.append({"word": cur_word, "tok_ids": cur_ids, "attn": agg})
        cur_ids, cur_vecs, cur_word = [], [], ""

    for i in range(start, end + 1):
        tr = tokens_raw[i]
        if is_bad(tr):
            flush()
            continue

        piece = clean_piece(tr)

        if tr.startswith("Ġ"):
            flush()
            cur_word = piece
            cur_ids = [i]
            cur_vecs = [attn_TN[i]]
        else:
            if not cur_ids:
                cur_word = piece
                cur_ids = [i]
                cur_vecs = [attn_TN[i]]
            else:
                cur_word += piece
                cur_ids.append(i)
                cur_vecs.append(attn_TN[i])

    flush()
    return groups

def build_raw_token_groups(tokens_raw, attn_TN, start, end):
    """No merging: each (subword) token becomes one group."""
    groups = []
    for i in range(start, end + 1):
        tr = tokens_raw[i]
        if is_bad(tr):
            continue
        groups.append({
            "word": clean_piece(tr),
            "tok_ids": [i],
            "attn": attn_TN[i],
        })
    return groups

def upsample_grid_to_image(grid: np.ndarray, out_h: int, out_w: int):
    grid = grid.astype(np.float32)
    grid = grid - grid.min()
    if grid.max() > 1e-8:
        grid = grid / grid.max()
    g = (grid * 255).astype(np.uint8)
    g_img = Image.fromarray(g, mode="L")
    g_up = g_img.resize((out_w, out_h), resample=Image.BILINEAR)
    return np.asarray(g_up).astype(np.float32) / 255.0

def overlay(img_pil: Image.Image, heatmap: np.ndarray, alpha: float = 0.45, cmap_name="viridis"):
    img = np.asarray(img_pil).astype(np.float32) / 255.0
    cmap = plt.get_cmap(cmap_name)
    hm_rgb = cmap(np.clip(heatmap, 0, 1))[..., :3]
    out = (1 - alpha) * img + alpha * hm_rgb
    return np.clip(out, 0, 1)

def get_layer_indices(L: int, mode: str = "mid_third", last_k: int = 4):
    if mode == "all":
        return list(range(L))

    if mode in ["first_third", "mid_third", "last_third"]:
        t = L // 3
        if t == 0:
            return list(range(L))
        if mode == "first_third":
            return list(range(0, t))
        if mode == "mid_third":
            return list(range(t, 2 * t))
        if mode == "last_third":
            return list(range(2 * t, L))

    if mode == "last_k":
        k = min(last_k, L)
        return list(range(L - k, L))

    raise ValueError("layer_mode must be one of: all / first_third / mid_third / last_third / last_k")

def load_captions_and_agreements(caption_path, agreement_path):
    with open(caption_path, "r", encoding="utf-8") as f:
        captions = [line.strip() for line in f if line.strip()]
    with open(agreement_path, "r", encoding="utf-8") as f:
        agreements = [float(line.strip()) for line in f if line.strip()]
    assert len(captions) == len(agreements)
    return captions, agreements


# -----------------------------
# FIXED: render all tokens in one figure with a REAL colorbar mappable
# -----------------------------
def render_all_tokens_one_figure(
    img_pil: Image.Image,
    groups,
    gh: int,
    gw: int,
    out_path: str,
    alpha: float = 0.45,
    ncols: int = 6,
    show_overlay: bool = True,
    dpi: int = 200,
    cmap_name: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
):
    """
    One big figure:
      - token maps in a grid (titles on top)
      - shared colorbar with colors (ScalarMappable)
      - original image at bottom spanning all columns
    """
    W, H = img_pil.size
    ntok = len(groups)
    if ntok == 0:
        return

    ncols = max(1, min(ncols, ntok))
    nrows_tok = int(math.ceil(ntok / ncols))
    nrows_total = nrows_tok + 1

    fig_w = 3.2 * ncols
    fig_h = 2.8 * nrows_tok + 3.5

    # Use constrained_layout to avoid tight_layout warning with gridspec+colorbar
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)

    gs = fig.add_gridspec(
        nrows_total, ncols,
        height_ratios=[1.0] * nrows_tok + [1.2],
        hspace=0.35, wspace=0.15
    )

    cmap = plt.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # IMPORTANT: Use a ScalarMappable for colorbar (always has colors)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # precompute heats
    heats = []
    for g in groups:
        vec = g["attn"]
        grid = vec.reshape(gh, gw)
        heat = upsample_grid_to_image(grid, H, W)  # [0,1]
        heats.append(heat)

    token_axes = []
    for i, (g, heat) in enumerate(zip(groups, heats)):
        r = i // ncols
        c = i % ncols
        ax = fig.add_subplot(gs[r, c])
        token_axes.append(ax)

        if show_overlay:
            over = overlay(img_pil, heat, alpha=alpha, cmap_name=cmap_name)
            ax.imshow(over)
        else:
            ax.imshow(heat, cmap=cmap, norm=norm)

        ax.axis("off")
        word = g["word"]
        ax.set_title(f"{i:02d}: {word}", fontsize=10, pad=6)

    # shared colorbar for all token axes
    cbar = fig.colorbar(sm, ax=token_axes, fraction=0.02, pad=0.01)
    cbar.set_label("normalized attention", rotation=90)

    # bottom original image
    ax_img = fig.add_subplot(gs[-1, :])
    ax_img.imshow(np.asarray(img_pil))
    ax_img.axis("off")
    ax_img.set_title("Original image", fontsize=12, pad=8)

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


# -----------------------------
# Core: one sample end-to-end
# -----------------------------
def run_one(
    idx, image_path, caption, model, processor,
    out_root,
    layer_mode="mid_third", last_k=4, merge="sum", alpha=0.45,
    all_tokens_mode="grouped",   # "grouped" or "raw"
    all_tokens_cols=6,
    all_tokens_show_overlay=True
):
    if not os.path.exists(image_path):
        print(f"[SKIP] missing image: {image_path}")
        return

    img = Image.open(image_path).convert("RGB")
    img = img.resize((256, 256), resample=Image.BILINEAR)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": caption},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[text],
        images=[img],
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, return_dict=True)

    input_ids = inputs["input_ids"][0].detach().cpu()
    tokens_raw = processor.tokenizer.convert_ids_to_tokens(input_ids.tolist())

    try:
        vs = tokens_raw.index("<|vision_start|>")
        ve = tokens_raw.index("<|vision_end|>")
    except ValueError:
        print(f"[SKIP] cannot find vision markers for idx={idx}")
        return

    vision_idx = torch.arange(vs + 1, ve, dtype=torch.long)
    Nvision = int(vision_idx.numel())
    if Nvision <= 0:
        print(f"[SKIP] idx={idx}: Nvision<=0")
        return

    gh, gw = guess_grid_from_n(Nvision)
    if gh * gw != Nvision:
        print(f"[SKIP] idx={idx}: cannot factor Nvision={Nvision}")
        return

    attns = outputs.attentions  # tuple(L) of (B,H,T,T)
    L = len(attns)
    layers = get_layer_indices(L, mode=layer_mode, last_k=last_k)

    q2v_layers = []
    for la in layers:
        layer_attn = attns[la][0].detach().cpu()   # (H,T,T)
        sub = layer_attn[:, :, vision_idx]         # (H,T,Nvision)
        q2v_layers.append(sub.float())
    q2v = torch.stack(q2v_layers, dim=0)           # (Lsel,H,T,Nvision)
    attn_mean = q2v.mean(dim=1).mean(dim=0).numpy()  # (T,Nvision)

    cap_s, cap_e = find_caption_span(tokens_raw)

    # groups for ALL_TOKENS
    if all_tokens_mode == "raw":
        groups_for_all = build_raw_token_groups(tokens_raw, attn_mean, cap_s, cap_e)
    else:
        groups_for_all = group_subwords_with_G(tokens_raw, attn_mean, cap_s, cap_e, merge=merge)

    # groups for per-word single images (keep grouped)
    groups_for_single = group_subwords_with_G(tokens_raw, attn_mean, cap_s, cap_e, merge=merge)

    out_dir = os.path.join(out_root, f"idx_{idx:06d}")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "meta.txt"), "w", encoding="utf-8") as f:
        f.write(f"idx={idx}\n")
        f.write(f"image_path={image_path}\n")
        f.write(f"caption={caption}\n")
        f.write(f"Nvision={Nvision}, grid={gh}x{gw}\n")
        f.write(f"layer_mode={layer_mode}, layers={layers}\n")
        f.write(f"merge={merge}\n")
        f.write(f"all_tokens_mode={all_tokens_mode}\n")
        f.write(f"all_tokens_cols={all_tokens_cols}\n")
        f.write(f"all_tokens_show_overlay={all_tokens_show_overlay}\n")

    with open(os.path.join(out_dir, "groups.txt"), "w", encoding="utf-8") as f:
        for gi, g in enumerate(groups_for_single):
            f.write(f"{gi}\t{g['word']}\t{g['tok_ids']}\n")

    # per-word maps (same as before)
    W, H = img.size
    for gi, g in enumerate(groups_for_single):
        vec = g["attn"]
        grid = vec.reshape(gh, gw)
        heat = upsample_grid_to_image(grid, H, W)
        over = overlay(img, heat, alpha=alpha, cmap_name="viridis")

        word = g["word"]
        fname = f"{gi:03d}_{sanitize_fname(word)}.png"
        out_path = os.path.join(out_dir, fname)

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(over)
        plt.axis("off")
        plt.title(f"{word} | tok_ids={g['tok_ids']} | grid {gh}x{gw}")

        ax2 = plt.subplot(1, 2, 2)
        im2 = ax2.imshow(heat, vmin=0.0, vmax=1.0, cmap="viridis")
        ax2.axis("off")
        ax2.set_title(f"heat ({layer_mode}, merge={merge})")

        cbar = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label("normalized attention")

        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

    # NEW: ALL TOKENS in ONE figure
    all_path = os.path.join(out_dir, "ALL_TOKENS.png")
    render_all_tokens_one_figure(
        img_pil=img,
        groups=groups_for_all,
        gh=gh,
        gw=gw,
        out_path=all_path,
        alpha=alpha,
        ncols=all_tokens_cols,
        show_overlay=all_tokens_show_overlay,
        dpi=200,
        cmap_name="viridis",
        vmin=0.0,
        vmax=1.0
    )

    print(f"[OK] idx={idx}: saved {len(groups_for_single)} word maps + ALL_TOKENS -> {out_dir}")


# -----------------------------
# Batch main
# -----------------------------
def main():
    base_path = "/home/ubuntu/spatial_twoshapes/agreement/relational/test/shard0"
    caption_path = os.path.join(base_path, "caption.txt")
    agreement_path = os.path.join(base_path, "agreement.txt")
    out_root = os.path.join(base_path, "viz_from_chat_words_mid_third")

    layer_mode = "mid_third"   # all / first_third / mid_third / last_third / last_k
    last_k = 4
    merge = "sum"
    alpha = 0.45

    # Big figure options
    all_tokens_mode = "grouped"      # "grouped" (word-level) or "raw" (subword-level)
    all_tokens_cols = 6              # grid columns for ALL_TOKENS
    all_tokens_show_overlay = True   # True: overlay; False: heat only

    max_matched = None               # debug: 5

    os.makedirs(out_root, exist_ok=True)
    captions, agreements = load_captions_and_agreements(caption_path, agreement_path)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    count = 0
    for idx, (cap, agr) in enumerate(zip(captions, agreements)):
        if agr != 1.0:
            continue

        image_path = os.path.join(base_path, f"world-{idx}.png")
        run_one(
            idx=idx,
            image_path=image_path,
            caption=cap,
            model=model,
            processor=processor,
            out_root=out_root,
            layer_mode=layer_mode,
            last_k=last_k,
            merge=merge,
            alpha=alpha,
            all_tokens_mode=all_tokens_mode,
            all_tokens_cols=all_tokens_cols,
            all_tokens_show_overlay=all_tokens_show_overlay
        )
        count += 1
        if max_matched is not None and count >= max_matched:
            break

    print(f"[DONE] processed {count} matched samples.")

if __name__ == "__main__":
    main()
