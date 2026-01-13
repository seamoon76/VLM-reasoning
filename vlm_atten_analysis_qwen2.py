import os
import json
import re
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# =====================
# Qwen2-VL setup
# =====================
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ======= DATA: all shards =======
NUM_SHARDS = 5
BASE_ROOT = "data/spatial_twoshapes/agreement/relational"  
SPLIT = ""  # e.g. "test" / "train" / "agreement"

FEED_SIZE = 672
LAYER_MODE = "mid_third"   # all / first_third / mid_third / last_third / last_k
LAST_K = 4

# ======= OUTPUT =======
OUT_ROOT = os.path.join(BASE_ROOT, SPLIT, f"qwen2_metrics_allshards_feed{FEED_SIZE}_{LAYER_MODE}")
os.makedirs(OUT_ROOT, exist_ok=True)

# =====================
# DEBUG switches
# =====================
DEBUG_ATTN = True              
DEBUG_SHOW_TOKENS = False     
DEBUG_LAYERWISE = True         
AUTO_SOFTMAX_IF_NEEDED = True 
# =====================
# Baseline: shuffled-image (PATCH shuffle, same spirit as your LLaVA baseline)
# =====================
ENABLE_SHUFFLED_IMAGE_BASELINE = True
SHUFFLE_SEED = 123  
SHUFFLE_GRID_SIZE = 24  # baseline 24x24 patch shuffle

def shuffle_image_patches(image_pil: Image.Image, grid_size: int = 24, seed: int = 123) -> Image.Image:
    rng = np.random.RandomState(seed)
    img = np.array(image_pil.convert("RGB"))
    H, W, C = img.shape

    patch_h = H // grid_size
    patch_w = W // grid_size
    H2 = patch_h * grid_size
    W2 = patch_w * grid_size
    img = img[:H2, :W2, :]

    patches = []
    for y in range(grid_size):
        for x in range(grid_size):
            patches.append(img[y*patch_h:(y+1)*patch_h, x*patch_w:(x+1)*patch_w].copy())

    perm = rng.permutation(len(patches))
    shuffled = [patches[i] for i in perm]

    new_img = np.zeros_like(img)
    idx = 0
    for y in range(grid_size):
        for x in range(grid_size):
            new_img[y*patch_h:(y+1)*patch_h, x*patch_w:(x+1)*patch_w] = shuffled[idx]
            idx += 1

    return Image.fromarray(new_img)

# =====================
# Token cleaning / filtering
# =====================
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
    raise ValueError("layer_mode must be all/first_third/mid_third/last_third/last_k")

def find_vision_span(tokens_raw):
    vs = tokens_raw.index("<|vision_start|>")
    ve = tokens_raw.index("<|vision_end|>")
    return vs, ve

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

def group_words_in_span(tokens_raw, start, end):
    groups = []
    cur_word = ""
    cur_ids = []

    def flush():
        nonlocal cur_word, cur_ids
        if cur_ids:
            groups.append({"word": cur_word, "tok_ids": cur_ids})
        cur_word, cur_ids = "", []

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
        else:
            if not cur_ids:
                cur_word = piece
                cur_ids = [i]
            else:
                cur_word += piece
                cur_ids.append(i)
    flush()
    return groups

def find_word_tokids(groups, keyword):
    kw = keyword.lower()
    for g in groups:
        if g["word"].lower() == kw:
            return g["tok_ids"]
    for g in groups:
        if kw in g["word"].lower():
            return g["tok_ids"]
    return []

# =====================
# Visualization
# =====================
def upsample_grid_to_image(grid, image_size):
    grid_np = grid.detach().float().cpu().numpy()
    return cv2.resize(grid_np, (image_size, image_size), interpolation=cv2.INTER_NEAREST)

def visualize(image, attn_map, save_path, caption):
    image_np = np.array(image)
    H = image_np.shape[0]
    attn_up = upsample_grid_to_image(attn_map, H)

    if DEBUG_ATTN:
        amin = float(attn_map.min().item())
        amean = float(attn_map.mean().item())
        amax = float(attn_map.max().item())
        print(f"[VIS] {caption}  attn_map(min/mean/max)={amin:.6g}/{amean:.6g}/{amax:.6g}")

    attn_up = attn_up / (attn_up.max() + 1e-6)
    plt.figure(figsize=(5, 5))
    plt.imshow(image_np)
    plt.imshow(attn_up, cmap="jet", alpha=0.6)
    plt.colorbar()
    plt.title(caption)             
    plt.axis("off")
    plt.savefig(save_path + ".png") 
    plt.close()


def visualize_patch_self_attn(image, attn_map, save_path, caption):
    image_np = np.array(image)
    H = image_np.shape[0]
    attn_up = upsample_grid_to_image(attn_map, H)

    if DEBUG_ATTN:
        amin = float(attn_map.min().item())
        amean = float(attn_map.mean().item())
        amax = float(attn_map.max().item())
        print(f"[VIS-SELF] {caption}  attn_map(min/mean/max)={amin:.6g}/{amean:.6g}/{amax:.6g}")

    attn_up = attn_up / (attn_up.max() + 1e-6)
    plt.figure(figsize=(5, 5))
    plt.imshow(image_np)
    plt.imshow(attn_up, cmap="jet", alpha=0.6)
    plt.colorbar()
    plt.title(caption)
    plt.axis("off")
    plt.savefig(save_path + ".png")
    plt.close()


def visualize_mask(image, mask, title, alpha=0.6, cmap="gray"):
    image_np = np.array(image)
    H = image_np.shape[0]
    mask_up = upsample_grid_to_image(mask, H)
    mask_up = (mask_up > 0).astype(np.float32)
    plt.figure(figsize=(5, 5))
    plt.imshow(image_np)
    plt.imshow(mask_up, cmap=cmap, alpha=alpha)
    plt.title(title)
    plt.axis("off")
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()

# =====================
# Seeds / self-attn map
# =====================
def get_entity_center_seed(gt_mask):
    ys, xs = torch.where(gt_mask > 0)
    y_center = ys.float().mean().round().long()
    x_center = xs.float().mean().round().long()
    return y_center.item() * gt_mask.shape[1] + x_center.item()

def get_background_seed(mask_entity1, mask_entity2):
    bg_mask = 1 - torch.clamp(mask_entity1 + mask_entity2, 0, 1)
    ys, xs = torch.where(bg_mask > 0)
    if len(xs) == 0:
        return None
    idx = np.random.randint(len(xs))
    y_bg = ys[idx].item()
    x_bg = xs[idx].item()
    return y_bg * bg_mask.shape[1] + x_bg

def get_patch_self_attention_map(attn, seed_idx, grid_h, grid_w, remove_self=True):
    vec = attn[seed_idx].clone()
    if remove_self:
        vec[seed_idx] = 0.0
    vec = vec / (vec.sum() + 1e-6)
    return vec.reshape(grid_h, grid_w)

# =====================
# Metrics
# =====================
def _normalize_attn(attn_map: torch.Tensor) -> torch.Tensor:
    return attn_map / (attn_map.sum() + 1e-6)

def _exclusive_masks(m1: torch.Tensor, m2: torch.Tensor):
    m1 = (m1 > 0).float()
    m2 = (m2 > 0).float()
    overlap = (m1 * m2).float()
    m1_only = (m1 * (1 - m2)).float()
    m2_only = (m2 * (1 - m1)).float()
    bg = (1 - torch.clamp(m1_only + m2_only + overlap, 0, 1)).float()
    return m1_only, m2_only, overlap, bg

def compute_center_of_mass_distance(attn_map, gt_mask):
    attn_map = _normalize_attn(attn_map)
    H, W = attn_map.shape
    xs = torch.arange(W).view(1, -1)
    ys = torch.arange(H).view(-1, 1)
    attn_com = torch.tensor([(attn_map * ys).sum(), (attn_map * xs).sum()])

    gt_mask = gt_mask.float()
    gt_mask = gt_mask / (gt_mask.sum() + 1e-6)
    gt_com = torch.tensor([(gt_mask * ys).sum(), (gt_mask * xs).sum()])
    return torch.norm(attn_com - gt_com).item()

def compute_iou(attn_map, gt_mask):
    k = int(gt_mask.sum().item())
    if k <= 0:
        return 0.0
    flat = attn_map.flatten()
    thresh = torch.topk(flat, k).values.min()
    attn_binary = (flat >= thresh).float().reshape_as(gt_mask)
    intersection = (attn_binary * gt_mask).sum().item()
    union = ((attn_binary + gt_mask) > 0).sum().item()
    return intersection / (union + 1e-6)

def compute_soft_iou(attn_map, gt_mask):
    attn = _normalize_attn(attn_map)
    gt   = gt_mask.float()
    intersection = (attn * gt).sum()
    union = attn.sum() + gt.sum() - intersection
    return (intersection / (union + 1e-6)).item()

def compute_x_wasserstein(attn_map, gt_mask):
    attn = _normalize_attn(attn_map)
    p = attn.sum(dim=0)
    q = gt_mask.float().sum(dim=0)
    p = p / (p.sum() + 1e-6)
    q = q / (q.sum() + 1e-6)
    cdf_p = torch.cumsum(p, dim=0)
    cdf_q = torch.cumsum(q, dim=0)
    return torch.sum(torch.abs(cdf_p - cdf_q)).item()

def compute_disperson(attn_map):
    attn = _normalize_attn(attn_map)
    H, W = attn.shape
    ys = torch.arange(H).view(-1, 1)
    xs = torch.arange(W).view(1, -1)
    y_mean = (attn * ys).sum()
    x_mean = (attn * xs).sum()
    var = ((ys - y_mean)**2 + (xs - x_mean)**2) * attn
    return var.sum().item()

def compute_entity_background_ratio(attn_map, mask_entity, mask_entity_other, debug_tag=None):
    attn = _normalize_attn(attn_map)
    m1_only, m2_only, overlap, bg = _exclusive_masks(mask_entity, mask_entity_other)

    inside = (attn * m1_only).sum()
    other  = (attn * m2_only).sum()
    ovl    = (attn * overlap).sum()
    bgmass = (attn * bg).sum()

    ratio = (inside / (bgmass + 1e-6)).item()

    if debug_tag is not None:
        total = (inside + other + ovl + bgmass).item()
        print(f"[{debug_tag}] inside={inside.item():.6f}, other={other.item():.6f}, "
              f"overlap={ovl.item():.6f}, bg={bgmass.item():.6f}, total={total:.6f}, ratio={ratio:.6f}")

    return ratio

def build_relation_gt_x(mask_entity1, mask_entity2, relation):
    p = mask_entity1.sum(dim=0).float()
    q = mask_entity2.sum(dim=0).float()
    if relation == "left":
        p, q = q, p
    p = p / (p.sum() + 1e-6)
    q = q / (q.sum() + 1e-6)
    cdf_p = torch.cumsum(p, dim=0)
    cdf_q = torch.cumsum(q, dim=0)
    gt_relation_x = torch.relu(cdf_p - cdf_q)
    gt_relation_x = gt_relation_x / (gt_relation_x.sum() + 1e-6)
    return gt_relation_x

def get_attn_x_distribution(attn_map):
    attn = _normalize_attn(attn_map)
    attn_x = attn.sum(dim=0)
    attn_x = attn_x / (attn_x.sum() + 1e-6)
    return attn_x

def compute_x_wasserstein_from_distributions(p, q):
    cdf_p = torch.cumsum(p, dim=0)
    cdf_q = torch.cumsum(q, dim=0)
    return torch.sum(torch.abs(cdf_p - cdf_q)).item()

def compute_entity_metrics(attn_map, gt_mask, other_gt_mask=None, debug_tag=None):
    return {
        "com": compute_center_of_mass_distance(attn_map, gt_mask),
        "iou": compute_iou(attn_map, gt_mask),
        "soft_iou": compute_soft_iou(attn_map, gt_mask),
        "wasserstein_x": compute_x_wasserstein(attn_map, gt_mask),
        "entity_bg_ratio": compute_entity_background_ratio(attn_map, gt_mask, other_gt_mask, debug_tag=debug_tag),
        "disperson": compute_disperson(attn_map),
    }

# =====================
# GT bbox -> patch mask (normalized bbox to patch grid)
# =====================
def gt_bbox_to_patch_mask_norm(entity, grid_h, grid_w):
    xmin = entity["bounding_box"]["topleft"]["x"]
    ymin = entity["bounding_box"]["topleft"]["y"]
    xmax = entity["bounding_box"]["bottomright"]["x"]
    ymax = entity["bounding_box"]["bottomright"]["y"]

    px_min = int(np.floor(xmin * grid_w))
    py_min = int(np.floor(ymin * grid_h))
    px_max = int(np.floor(xmax * grid_w))
    py_max = int(np.floor(ymax * grid_h))

    px_min = max(0, min(grid_w - 1, px_min))
    py_min = max(0, min(grid_h - 1, py_min))
    px_max = max(0, min(grid_w - 1, px_max))
    py_max = max(0, min(grid_h - 1, py_max))

    mask = torch.zeros(grid_h, grid_w)
    mask[py_min:py_max + 1, px_min:px_max + 1] = 1
    return mask

# =====================
# Attention debug helpers
# =====================
def _attn_row_sum_stats(attn_probs_2d: torch.Tensor):
    rs = attn_probs_2d.sum(dim=-1)
    return float(rs.min().item()), float(rs.max().item()), float(rs.mean().item())

def _decide_need_softmax(a_head_TT: torch.Tensor):
    if a_head_TT.dim() == 3:
        m = a_head_TT.mean(dim=0)  # (T,T)
    else:
        m = a_head_TT              # (T,T)

    amin = float(m.min().item())
    row_sums = m.sum(dim=-1)
    row_sum_min = float(row_sums.min().item())
    row_sum_mean = float(row_sums.mean().item())

    need = (amin < -1e-6) or (abs(row_sum_mean - 1.0) > 0.05) or (abs(row_sum_min - 1.0) > 0.05)
    return need, amin, row_sum_min, row_sum_mean

# =====================
# Core: Qwen2 attention extraction
# =====================
def extract_prompt_level_cross_attention_qwen2(image_pil, prompt_text, model, processor, debug_prefix=""):
    """
    The input construction and prompt formatting (including removal of the system prompt) follow the implementation in the official LLaVA demo notebook.
    https://github.com/zjysteven/VLM-Visualizer/blob/main/llava_example.ipynb
    image_pil: PIL.Image (RGB)
    Return:
        cross_attn: (Tprompt_filtered, Nvision)   # text->vision
        prompt_words: grouped words with token ids
        image_feed: PIL resized
        vis_self_attn: (Nvision, Nvision)
        gh, gw: grid of Nvision
        token_to_promptidx: mapping raw token id -> row index in cross_attn
    """
    img0 = image_pil.convert("RGB")
    image_feed = img0.resize((FEED_SIZE, FEED_SIZE), resample=Image.BILINEAR)

    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": prompt_text}],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[text],
        images=[image_feed],
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    input_ids = inputs["input_ids"][0].detach().cpu()
    tokens_raw = processor.tokenizer.convert_ids_to_tokens(input_ids.tolist())
    T = len(tokens_raw)

    if DEBUG_SHOW_TOKENS:
        print("=== tokens_raw ===")
        for i, t in enumerate(tokens_raw):
            print(i, t)

    vs, ve = find_vision_span(tokens_raw)
    vision_token_ids = list(range(vs + 1, ve))
    Nvision = len(vision_token_ids)
    gh, gw = guess_grid_from_n(Nvision)

    cap_s, cap_e = find_caption_span(tokens_raw)
    prompt_token_ids = [tid for tid in range(cap_s, cap_e + 1) if not is_bad(tokens_raw[tid])]
    token_to_promptidx = {tid: j for j, tid in enumerate(prompt_token_ids)}
    prompt_words = group_words_in_span(tokens_raw, cap_s, cap_e)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, return_dict=True)

    attns = outputs.attentions
    L = len(attns)
    layers = get_layer_indices(L, mode=LAYER_MODE, last_k=LAST_K)

    mats = []
    need_softmax_votes = 0

    for la in layers:
        a = attns[la][0].detach().float().cpu()   # (H,T,T)

        if DEBUG_LAYERWISE:
            m = a.mean(dim=0)  # (T,T)
            amin = float(m.min().item())
            amax = float(m.max().item())
            rmin, rmax, rmean = _attn_row_sum_stats(m)
            print(f"{debug_prefix}[L{la}] mean-attn min/max={amin:.6g}/{amax:.6g} | row-sum(min/max/mean)={rmin:.6g}/{rmax:.6g}/{rmean:.6g}")

        need_sf, _, _, _ = _decide_need_softmax(a)
        if need_sf:
            need_softmax_votes += 1

        if AUTO_SOFTMAX_IF_NEEDED and need_sf:
            a = torch.softmax(a, dim=-1)

        mats.append(a.mean(dim=0))  # (T,T)

    attn_full = torch.stack(mats, dim=0).mean(dim=0)  # (T,T)

    if DEBUG_ATTN:
        amin = float(attn_full.min().item())
        amax = float(attn_full.max().item())
        rmin, rmax, rmean = _attn_row_sum_stats(attn_full)
        print(f"{debug_prefix}[FULL] min/max={amin:.6g}/{amax:.6g} | row-sum(min/max/mean)={rmin:.6g}/{rmax:.6g}/{rmean:.6g} | softmax_votes={need_softmax_votes}/{len(layers)}")
        if amin < -1e-6:
            print(f"{debug_prefix}[WARN] attn_full has negative values -> not a probability attention (or numerical issue).")
        if abs(rmean - 1.0) > 0.05:
            print(f"{debug_prefix}[WARN] attn_full row-sum mean not ~1 -> likely not softmax probs.")

    # cross: prompt tokens -> vision tokens
    cross_attn = attn_full[prompt_token_ids][:, vision_token_ids].clone()  # (Tprompt_filtered, Nvision)
    cross_attn = cross_attn / (cross_attn.sum(dim=1, keepdim=True) + 1e-6)

    # vision self-attn
    vis_self_attn = attn_full[vision_token_ids][:, vision_token_ids].clone()
    vis_self_attn = vis_self_attn / (vis_self_attn.sum(dim=1, keepdim=True) + 1e-6)

    if DEBUG_ATTN:
        cmin = float(cross_attn.min().item()); cmax = float(cross_attn.max().item())
        vmin = float(vis_self_attn.min().item()); vmax = float(vis_self_attn.max().item())
        crow = cross_attn.sum(dim=1)
        vrow = vis_self_attn.sum(dim=1)
        print(f"{debug_prefix}[CROSS] min/max={cmin:.6g}/{cmax:.6g} row-sum(min/max/mean)={float(crow.min()):.6g}/{float(crow.max()):.6g}/{float(crow.mean()):.6g}")
        print(f"{debug_prefix}[VSELF] min/max={vmin:.6g}/{vmax:.6g} row-sum(min/max/mean)={float(vrow.min()):.6g}/{float(vrow.max()):.6g}/{float(vrow.mean()):.6g}")

    print("\n====================")
    print(debug_prefix + "Feed size:", image_feed.size)
    print(debug_prefix + "T total tokens:", T)
    print(debug_prefix + "vision span:", vs, ve, "Nvision:", Nvision, "grid:", f"{gh}x{gw}")
    print(debug_prefix + "caption span:", cap_s, cap_e, "| prompt tokens(filtered):", len(prompt_token_ids))
    print(debug_prefix + "Selected layers:", layers, f"(mode={LAYER_MODE}, last_k={LAST_K})")
    print(debug_prefix + "Caption words (grouped):")
    for g in prompt_words:
        print(f"  {g['word']}\t{g['tok_ids']}")
    print("====================\n")

    print(debug_prefix + "Text->Vision attn shape:", tuple(cross_attn.shape), " (Tprompt, Nvision)")
    print(debug_prefix + "Vision self-attn shape:", tuple(vis_self_attn.shape), " (Nvision, Nvision)")
    print(debug_prefix + "grid:", gh, gw)

    return cross_attn, prompt_words, image_feed, vis_self_attn, gh, gw, token_to_promptidx

def pool_attention_qwen2(cross_attn, tok_ids, token_to_promptidx, mode="mean"):
    rows = []
    for tid in tok_ids:
        if tid in token_to_promptidx:
            rows.append(token_to_promptidx[tid])
    if len(rows) == 0:
        return None
    sub = cross_attn[rows, :]   # (n_tok, Nvision)
    if mode == "mean":
        return sub.mean(dim=0)  # (Nvision,)
    elif mode == "sum":
        return sub.sum(dim=0)
    else:
        raise ValueError

# =====================
# Mean helpers
# =====================
def safe_mean(x):
    return float(np.mean(x)) if len(x) > 0 else float("nan")

# =====================
# Load model once
# =====================
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model.eval()

# =====================
# Global accumulators (ALL SHARDS)
# =====================

# --- real metrics ---
entity1_com_dists, entity2_com_dists = [], []
entity1_ious, entity2_ious = [], []
entity1_soft_ious, entity2_soft_ious = [], []
entity1_wassersteins, entity2_wassersteins = [], []
relation_x_wassersteins = []

# --- shuffled baseline metrics ---
entity1_com_dists_shuf, entity2_com_dists_shuf = [], []
entity1_ious_shuf, entity2_ious_shuf = [], []
entity1_soft_ious_shuf, entity2_soft_ious_shuf = [], []
entity1_wassersteins_shuf, entity2_wassersteins_shuf = [], []
relation_x_wassersteins_shuf = []

# --- vision self-attn metrics (real) ---
entity1_self_attn_metrics_all = {k: [] for k in ["com", "iou", "soft_iou", "wasserstein_x", "entity_bg_ratio", "disperson"]}
entity2_self_attn_metrics_all = {k: [] for k in ["com", "iou", "soft_iou", "wasserstein_x", "entity_bg_ratio", "disperson"]}
bg_dispersons = []

# --- vision self-attn metrics (baseline) ---
entity1_self_attn_metrics_all_shuf = {k: [] for k in ["com", "iou", "soft_iou", "wasserstein_x", "entity_bg_ratio", "disperson"]}
entity2_self_attn_metrics_all_shuf = {k: [] for k in ["com", "iou", "soft_iou", "wasserstein_x", "entity_bg_ratio", "disperson"]}
bg_dispersons_shuf = []

# =====================
# Main: iterate all shards, all samples
# =====================
for shard_id in range(NUM_SHARDS):
    BASE = os.path.join(BASE_ROOT, SPLIT, f"shard{shard_id}")
    json_path = os.path.join(BASE, "world_model.json")
    caption_path = os.path.join(BASE, "caption.txt")

    if not os.path.exists(json_path):
        print(f"[WARN] missing json: {json_path}, skip shard{shard_id}")
        continue
    if not os.path.exists(caption_path):
        print(f"[WARN] missing caption: {caption_path}, skip shard{shard_id}")
        continue

    world = json.load(open(json_path))
    captions = [x.strip() for x in open(caption_path).readlines()]
    filtered_ids = list(range(len(captions)))

    shard_save_dir = os.path.join(OUT_ROOT, f"shard{shard_id}")
    os.makedirs(shard_save_dir, exist_ok=True)

    print(f"\n========== Processing shard{shard_id}: {BASE} | #samples={len(filtered_ids)} ==========\n")

    for sample_idx in filtered_ids:
        prompt = captions[sample_idx]
        if ("right" not in prompt) and ("left" not in prompt):
            continue

        image_path = os.path.join(BASE, f"world-{sample_idx}.png")
        if not os.path.exists(image_path):
            print(f"[WARN] missing image: {image_path}, skip")
            continue

        print(f"Processing shard{shard_id} sample idx: {sample_idx}")

        entity1 = world[sample_idx]["entities"][0]
        entity2 = world[sample_idx]["entities"][1]
        entity1_name = entity1["shape"]["name"].lower()
        entity2_name = entity2["shape"]["name"].lower()
        relation = "right" if "right" in prompt else "left"

        img_orig = Image.open(image_path).convert("RGB")

        # =====================
        # REAL
        # =====================
        cross_attn, prompt_words, image_feed, vis_self_attn, gh, gw, token_to_promptidx = \
            extract_prompt_level_cross_attention_qwen2(img_orig, prompt, model, processor, debug_prefix=f"[REAL s{shard_id} i{sample_idx}] ")

        cross_attn = torch.nan_to_num(cross_attn, nan=0.0)
        vis_self_attn = torch.nan_to_num(vis_self_attn, nan=0.0)

        # masks on gh x gw
        mask_entity1 = gt_bbox_to_patch_mask_norm(entity1, gh, gw)
        mask_entity2 = gt_bbox_to_patch_mask_norm(entity2, gh, gw)

        # visualize masks (optional but same as your current script)
        m1_only, m2_only, overlap, bg_mask = _exclusive_masks(mask_entity1, mask_entity2)
        visualize_mask(image_feed, mask_entity1, f"{shard_save_dir}/{sample_idx}_{entity1_name}_GT_mask")
        visualize_mask(image_feed, mask_entity2, f"{shard_save_dir}/{sample_idx}_{entity2_name}_GT_mask")
        visualize_mask(image_feed, bg_mask,      f"{shard_save_dir}/{sample_idx}_background_GT_mask")
        visualize_mask(image_feed, overlap,      f"{shard_save_dir}/{sample_idx}_overlap_GT_mask", cmap="Reds", alpha=0.7)

        # token ids
        entity1_tokids = find_word_tokids(prompt_words, entity1_name)
        relation_tokids = find_word_tokids(prompt_words, relation)
        entity2_tokids = find_word_tokids(prompt_words, entity2_name)

        print(f"{entity1_name} tok_ids:", entity1_tokids)
        print(f"{relation} tok_ids:", relation_tokids)
        print(f"{entity2_name} tok_ids:", entity2_tokids)

        # pooled attentions: [Nvision]
        entity1_attn = pool_attention_qwen2(cross_attn, entity1_tokids, token_to_promptidx, mode="mean")
        entity2_attn = pool_attention_qwen2(cross_attn, entity2_tokids, token_to_promptidx, mode="mean")
        relation_attn = pool_attention_qwen2(cross_attn, relation_tokids, token_to_promptidx, mode="mean")
        if relation_attn is None:
            print("[WARN] relation token not found; skip sample.")
            continue

        entity1_map = entity1_attn.reshape(gh, gw) if entity1_attn is not None else None
        entity2_map = entity2_attn.reshape(gh, gw) if entity2_attn is not None else None
        relation_map = relation_attn.reshape(gh, gw)

        # visualize attn maps (REAL)
        if entity1_map is not None:
            save_path = f"{shard_save_dir}/{sample_idx}_{entity1_name}_attention_REAL"
            caption   = f"\"{entity1_name}\" cross attention"
            visualize(image_feed, entity1_map, save_path, caption)

        save_path = f"{shard_save_dir}/{sample_idx}_{relation}_attention_REAL"
        caption   = f"\"{relation}\" cross attention"
        visualize(image_feed, relation_map, save_path, caption)
        if entity2_map is not None:
            save_path = f"{shard_save_dir}/{sample_idx}_{entity2_name}_attention_REAL"
            caption   = f"\"{entity2_name}\" cross attention"
            visualize(image_feed, entity2_map, save_path, caption)
        metric_path = os.path.join(shard_save_dir, f"metrics_sample_{sample_idx}.txt")
        with open(metric_path, "w") as f:
            f.write(f"shard idx: {shard_id}\n")
            f.write(f"sample idx: {sample_idx}\n")
            f.write(f"grid: {gh}x{gw}\n")
            f.write(f"prompt: {prompt}\n")
            f.write(f"baseline: patch-shuffle enabled={ENABLE_SHUFFLED_IMAGE_BASELINE}, seed={SHUFFLE_SEED}, grid={SHUFFLE_GRID_SIZE}\n\n")

        # entity metrics (REAL)
        if entity1_map is not None:
            e1_com = compute_center_of_mass_distance(entity1_map, mask_entity1)
            e1_iou = compute_iou(entity1_map, mask_entity1)
            e1_sio = compute_soft_iou(entity1_map, mask_entity1)
            e1_wx  = compute_x_wasserstein(entity1_map, mask_entity1)

            entity1_com_dists.append(e1_com)
            entity1_ious.append(e1_iou)
            entity1_soft_ious.append(e1_sio)
            entity1_wassersteins.append(e1_wx)

            with open(metric_path, "a") as f:
                f.write(f"[REAL] Entity1({entity1_name}) - COM: {e1_com:.4f}, IoU: {e1_iou:.4f}, SoftIoU: {e1_sio:.4f}, WassersteinX: {e1_wx:.4f}\n")

        if entity2_map is not None:
            e2_com = compute_center_of_mass_distance(entity2_map, mask_entity2)
            e2_iou = compute_iou(entity2_map, mask_entity2)
            e2_sio = compute_soft_iou(entity2_map, mask_entity2)
            e2_wx  = compute_x_wasserstein(entity2_map, mask_entity2)

            entity2_com_dists.append(e2_com)
            entity2_ious.append(e2_iou)
            entity2_soft_ious.append(e2_sio)
            entity2_wassersteins.append(e2_wx)

            with open(metric_path, "a") as f:
                f.write(f"[REAL] Entity2({entity2_name}) - COM: {e2_com:.4f}, IoU: {e2_iou:.4f}, SoftIoU: {e2_sio:.4f}, WassersteinX: {e2_wx:.4f}\n")

        # relation metric (REAL)
        gt_relation_x = build_relation_gt_x(mask_entity1, mask_entity2, relation)
        attn_x = get_attn_x_distribution(relation_map)
        rel_w = compute_x_wasserstein_from_distributions(attn_x, gt_relation_x)
        relation_x_wassersteins.append(rel_w)
        with open(metric_path, "a") as f:
            f.write(f"[REAL] Relation({relation}) x-axis Wasserstein: {rel_w:.4f}\n")

        # vision self-attn (REAL)
        seed1 = get_entity_center_seed(mask_entity1)
        seed2 = get_entity_center_seed(mask_entity2)
        bg_seed = get_background_seed(mask_entity1, mask_entity2)

        e1_self = get_patch_self_attention_map(vis_self_attn, seed1, gh, gw)
        e2_self = get_patch_self_attention_map(vis_self_attn, seed2, gh, gw)

        if bg_seed is not None:
            bg_self = get_patch_self_attention_map(vis_self_attn, bg_seed, gh, gw)
            bg_disp = compute_disperson(bg_self)
            bg_dispersons.append(bg_disp)
            with open(metric_path, "a") as f:
                f.write(f"[REAL] Background disperson: {bg_disp:.4f}\n")

        save_path = f"{shard_save_dir}/{sample_idx}_{entity1_name}_self_attention_REAL"
        caption   = f"\"{entity1_name}\" self attention"
        visualize_patch_self_attn(image_feed, e1_self, save_path, caption)

        save_path = f"{shard_save_dir}/{sample_idx}_{entity2_name}_self_attention_REAL"
        caption   = f"\"{entity2_name}\" self attention"
        visualize_patch_self_attn(image_feed, e2_self, save_path, caption)

        m1 = compute_entity_metrics(e1_self, mask_entity1, other_gt_mask=mask_entity2, debug_tag="entity1_self_REAL")
        m2 = compute_entity_metrics(e2_self, mask_entity2, other_gt_mask=mask_entity1, debug_tag="entity2_self_REAL")

        for k in entity1_self_attn_metrics_all:
            entity1_self_attn_metrics_all[k].append(m1[k])
            entity2_self_attn_metrics_all[k].append(m2[k])

        with open(metric_path, "a") as f:
            f.write("\n[REAL] Vision Self-Attention (LM vision sub-block)\n")
            for k, v in m1.items():
                f.write(f"[REAL] Entity1 self {k}: {v:.4f}\n")
            for k, v in m2.items():
                f.write(f"[REAL] Entity2 self {k}: {v:.4f}\n")

        # =====================
        # Baseline: PATCH-SHUFFLED IMAGE (same masks, same prompt)
        # =====================
        if ENABLE_SHUFFLED_IMAGE_BASELINE:
            # baseline should operate on the FEED_SIZE image (like your original baseline did)
            img_for_shuffle = img_orig.resize((FEED_SIZE, FEED_SIZE), resample=Image.BILINEAR)
            img_shuf = shuffle_image_patches(
                img_for_shuffle,
                grid_size=SHUFFLE_GRID_SIZE,
                seed=SHUFFLE_SEED + shard_id * 100000 + sample_idx
            )

            cross_b, prompt_words_b, image_feed_b, vis_self_b, gh_b, gw_b, token_to_promptidx_b = \
                extract_prompt_level_cross_attention_qwen2(img_shuf, prompt, model, processor, debug_prefix=f"[SHUF s{shard_id} i{sample_idx}] ")

            cross_b = torch.nan_to_num(cross_b, nan=0.0)
            vis_self_b = torch.nan_to_num(vis_self_b, nan=0.0)

            if (gh_b != gh) or (gw_b != gw):
                print(f"[WARN] baseline grid mismatch: real={gh}x{gw}, shuf={gh_b}x{gw_b}. Skip baseline metrics for this sample.")
            else:
                # token ids baseline
                e1_tok_b = find_word_tokids(prompt_words_b, entity1_name)
                rel_tok_b = find_word_tokids(prompt_words_b, relation)
                e2_tok_b = find_word_tokids(prompt_words_b, entity2_name)

                e1_att_b = pool_attention_qwen2(cross_b, e1_tok_b, token_to_promptidx_b, mode="mean")
                e2_att_b = pool_attention_qwen2(cross_b, e2_tok_b, token_to_promptidx_b, mode="mean")
                rel_att_b = pool_attention_qwen2(cross_b, rel_tok_b, token_to_promptidx_b, mode="mean")
                if rel_att_b is None:
                    print("[WARN] relation token not found in baseline; skip baseline.")
                else:
                    e1_map_b = e1_att_b.reshape(gh, gw) if e1_att_b is not None else None
                    e2_map_b = e2_att_b.reshape(gh, gw) if e2_att_b is not None else None
                    rel_map_b = rel_att_b.reshape(gh, gw)

                    # visualize baseline
                    if e1_map_b is not None:
                        save_path = f"{shard_save_dir}/{sample_idx}_{entity1_name}_attention_SHUF"
                        caption   = f"\"{entity1_name}\" cross attention"
                        visualize(image_feed_b, e1_map_b, save_path, caption)
                    save_path = f"{shard_save_dir}/{sample_idx}_{relation}_attention_SHUF"
                    caption   = f"\"{relation}\" cross attention"
                    visualize(image_feed_b, rel_map_b, save_path, caption)
                    if e2_map_b is not None:
                        save_path = f"{shard_save_dir}/{sample_idx}_{entity2_name}_attention_SHUF"
                        caption   = f"\"{entity2_name}\" cross attention"
                        visualize(image_feed_b, e2_map_b, save_path, caption)

                    with open(metric_path, "a") as f:
                        f.write("\n[SHUF] ===== Patch-Shuffle Baseline =====\n")

                    # entity metrics baseline
                    if e1_map_b is not None:
                        e1_com_b = compute_center_of_mass_distance(e1_map_b, mask_entity1)
                        e1_iou_b = compute_iou(e1_map_b, mask_entity1)
                        e1_sio_b = compute_soft_iou(e1_map_b, mask_entity1)
                        e1_wx_b  = compute_x_wasserstein(e1_map_b, mask_entity1)

                        entity1_com_dists_shuf.append(e1_com_b)
                        entity1_ious_shuf.append(e1_iou_b)
                        entity1_soft_ious_shuf.append(e1_sio_b)
                        entity1_wassersteins_shuf.append(e1_wx_b)

                        with open(metric_path, "a") as f:
                            f.write(f"[SHUF] Entity1({entity1_name}) - COM: {e1_com_b:.4f}, IoU: {e1_iou_b:.4f}, SoftIoU: {e1_sio_b:.4f}, WassersteinX: {e1_wx_b:.4f}\n")

                    if e2_map_b is not None:
                        e2_com_b = compute_center_of_mass_distance(e2_map_b, mask_entity2)
                        e2_iou_b = compute_iou(e2_map_b, mask_entity2)
                        e2_sio_b = compute_soft_iou(e2_map_b, mask_entity2)
                        e2_wx_b  = compute_x_wasserstein(e2_map_b, mask_entity2)

                        entity2_com_dists_shuf.append(e2_com_b)
                        entity2_ious_shuf.append(e2_iou_b)
                        entity2_soft_ious_shuf.append(e2_sio_b)
                        entity2_wassersteins_shuf.append(e2_wx_b)

                        with open(metric_path, "a") as f:
                            f.write(f"[SHUF] Entity2({entity2_name}) - COM: {e2_com_b:.4f}, IoU: {e2_iou_b:.4f}, SoftIoU: {e2_sio_b:.4f}, WassersteinX: {e2_wx_b:.4f}\n")

                    # relation baseline
                    attn_x_b = get_attn_x_distribution(rel_map_b)
                    rel_w_b = compute_x_wasserstein_from_distributions(attn_x_b, gt_relation_x)
                    relation_x_wassersteins_shuf.append(rel_w_b)
                    with open(metric_path, "a") as f:
                        f.write(f"[SHUF] Relation({relation}) x-axis Wasserstein: {rel_w_b:.4f}\n")

                    # self-attn baseline
                    seed1_b = get_entity_center_seed(mask_entity1)
                    seed2_b = get_entity_center_seed(mask_entity2)
                    bg_seed_b = get_background_seed(mask_entity1, mask_entity2)
                    e1_self_b = get_patch_self_attention_map(vis_self_b, seed1_b, gh, gw)
                    e2_self_b = get_patch_self_attention_map(vis_self_b, seed2_b, gh, gw)

                    if bg_seed_b is not None:
                        bg_self_b = get_patch_self_attention_map(vis_self_b, bg_seed_b, gh, gw)
                        bg_disp_b = compute_disperson(bg_self_b)
                        bg_dispersons_shuf.append(bg_disp_b)
                        with open(metric_path, "a") as f:
                            f.write(f"[SHUF] Background disperson: {bg_disp_b:.4f}\n")

                    save_path = f"{shard_save_dir}/{sample_idx}_{entity1_name}_self_attention_SHUF"
                    caption   = f"\"{entity1_name}\" self attention"
                    visualize_patch_self_attn(image_feed_b, e1_self_b, save_path, caption)
                    save_path = f"{shard_save_dir}/{sample_idx}_{entity2_name}_self_attention_SHUF"
                    caption   = f"\"{entity2_name}\" self attention"
                    visualize_patch_self_attn(image_feed_b, e2_self_b, save_path, caption)

                    m1b = compute_entity_metrics(e1_self_b, mask_entity1, other_gt_mask=mask_entity2, debug_tag="entity1_self_SHUF")
                    m2b = compute_entity_metrics(e2_self_b, mask_entity2, other_gt_mask=mask_entity1, debug_tag="entity2_self_SHUF")

                    for k in entity1_self_attn_metrics_all_shuf:
                        entity1_self_attn_metrics_all_shuf[k].append(m1b[k])
                        entity2_self_attn_metrics_all_shuf[k].append(m2b[k])

                    with open(metric_path, "a") as f:
                        f.write("\n[SHUF] Vision Self-Attention (LM vision sub-block)\n")
                        for k, v in m1b.items():
                            f.write(f"[SHUF] Entity1 self {k}: {v:.4f}\n")
                        for k, v in m2b.items():
                            f.write(f"[SHUF] Entity2 self {k}: {v:.4f}\n")

        print(f"[OK] shard{shard_id} sample saved under: {shard_save_dir}")

# =====================
# Global mean analysis (ALL SHARDS)
# =====================
print("\n=== Mean Analysis over ALL shards ===")
print(f"[REAL] Entity1 - COM: {safe_mean(entity1_com_dists):.4f}, IoU: {safe_mean(entity1_ious):.4f}, SoftIoU: {safe_mean(entity1_soft_ious):.4f}, WassersteinX: {safe_mean(entity1_wassersteins):.4f}")
print(f"[REAL] Entity2 - COM: {safe_mean(entity2_com_dists):.4f}, IoU: {safe_mean(entity2_ious):.4f}, SoftIoU: {safe_mean(entity2_soft_ious):.4f}, WassersteinX: {safe_mean(entity2_wassersteins):.4f}")
print(f"[REAL] Relation x-axis Wasserstein: {safe_mean(relation_x_wassersteins):.4f}")

if ENABLE_SHUFFLED_IMAGE_BASELINE:
    print(f"[SHUF] Entity1 - COM: {safe_mean(entity1_com_dists_shuf):.4f}, IoU: {safe_mean(entity1_ious_shuf):.4f}, SoftIoU: {safe_mean(entity1_soft_ious_shuf):.4f}, WassersteinX: {safe_mean(entity1_wassersteins_shuf):.4f}")
    print(f"[SHUF] Entity2 - COM: {safe_mean(entity2_com_dists_shuf):.4f}, IoU: {safe_mean(entity2_ious_shuf):.4f}, SoftIoU: {safe_mean(entity2_soft_ious_shuf):.4f}, WassersteinX: {safe_mean(entity2_wassersteins_shuf):.4f}")
    print(f"[SHUF] Relation x-axis Wasserstein: {safe_mean(relation_x_wassersteins_shuf):.4f}")

mean_path = os.path.join(OUT_ROOT, "mean_analysis_allshards.txt")
with open(mean_path, "w") as f:
    f.write("=== Mean Analysis over ALL shards ===\n")
    f.write(f"[REAL] Entity1 - COM: {safe_mean(entity1_com_dists):.4f}, IoU: {safe_mean(entity1_ious):.4f}, SoftIoU: {safe_mean(entity1_soft_ious):.4f}, WassersteinX: {safe_mean(entity1_wassersteins):.4f}\n")
    f.write(f"[REAL] Entity2 - COM: {safe_mean(entity2_com_dists):.4f}, IoU: {safe_mean(entity2_ious):.4f}, SoftIoU: {safe_mean(entity2_soft_ious):.4f}, WassersteinX: {safe_mean(entity2_wassersteins):.4f}\n")
    f.write(f"[REAL] Relation x-axis Wasserstein: {safe_mean(relation_x_wassersteins):.4f}\n")

    if ENABLE_SHUFFLED_IMAGE_BASELINE:
        f.write("\n=== Patch-Shuffle Baseline ===\n")
        f.write(f"[SHUF] Entity1 - COM: {safe_mean(entity1_com_dists_shuf):.4f}, IoU: {safe_mean(entity1_ious_shuf):.4f}, SoftIoU: {safe_mean(entity1_soft_ious_shuf):.4f}, WassersteinX: {safe_mean(entity1_wassersteins_shuf):.4f}\n")
        f.write(f"[SHUF] Entity2 - COM: {safe_mean(entity2_com_dists_shuf):.4f}, IoU: {safe_mean(entity2_ious_shuf):.4f}, SoftIoU: {safe_mean(entity2_soft_ious_shuf):.4f}, WassersteinX: {safe_mean(entity2_wassersteins_shuf):.4f}\n")
        f.write(f"[SHUF] Relation x-axis Wasserstein: {safe_mean(relation_x_wassersteins_shuf):.4f}\n")

print("\n=== Mean Vision Self-Attention Analysis (ALL SHARDS) ===")
with open(mean_path, "a") as f:
    f.write("\n=== Mean Vision Self-Attention Analysis (ALL SHARDS) ===\n")

    # REAL
    f.write("\n[REAL] Vision Self-Attention\n")
    for k in entity1_self_attn_metrics_all:
        e1_mean = safe_mean(entity1_self_attn_metrics_all[k])
        e2_mean = safe_mean(entity2_self_attn_metrics_all[k])
        print(f"[REAL] {k}: Entity1={e1_mean:.4f}, Entity2={e2_mean:.4f}")
        f.write(f"[REAL] {k}: Entity1={e1_mean:.4f}, Entity2={e2_mean:.4f}\n")
    if len(bg_dispersons) > 0:
        bg_mean = safe_mean(bg_dispersons)
        print(f"[REAL] Background disperson mean: {bg_mean:.4f}")
        f.write(f"[REAL] Background disperson mean: {bg_mean:.4f}\n")

    # SHUF
    if ENABLE_SHUFFLED_IMAGE_BASELINE:
        f.write("\n[SHUF] Vision Self-Attention\n")
        for k in entity1_self_attn_metrics_all_shuf:
            e1_mean_b = safe_mean(entity1_self_attn_metrics_all_shuf[k])
            e2_mean_b = safe_mean(entity2_self_attn_metrics_all_shuf[k])
            print(f"[SHUF] {k}: Entity1={e1_mean_b:.4f}, Entity2={e2_mean_b:.4f}")
            f.write(f"[SHUF] {k}: Entity1={e1_mean_b:.4f}, Entity2={e2_mean_b:.4f}\n")
        if len(bg_dispersons_shuf) > 0:
            bg_mean_b = safe_mean(bg_dispersons_shuf)
            print(f"[SHUF] Background disperson mean: {bg_mean_b:.4f}")
            f.write(f"[SHUF] Background disperson mean: {bg_mean_b:.4f}\n")

print(f"\n[OK] mean analysis saved to: {mean_path}")
