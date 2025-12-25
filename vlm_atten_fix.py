import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

sys.path.append("./models")

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.utils import disable_torch_init

from utils import load_image, aggregate_vit_attention

# =====================
# Model loading
# =====================
model_path = "liuhaotian/llava-v1.5-7b"
device = "cuda" if torch.cuda.is_available() else "cpu"

disable_torch_init()

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path,
    None,
    model_path,
    load_8bit=False,
    load_4bit=False,
    device=device,
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
)

model.eval()

# =====================
# Utils
# =====================

def upsample_grid_to_image(grid, image_size):
    grid_np = grid.detach().float().cpu().numpy()
    return cv2.resize(grid_np, (image_size, image_size), interpolation=cv2.INTER_NEAREST)


def visualize(image, attn_map, title):
    image_np = np.array(image)
    H = image_np.shape[0]
    attn_up = upsample_grid_to_image(attn_map, H)
    attn_up = attn_up / (attn_up.max() + 1e-6)
    plt.figure(figsize=(5, 5))
    plt.imshow(image_np)
    plt.imshow(attn_up, cmap="jet", alpha=0.6)
    plt.colorbar()
    plt.title(title)
    plt.axis("off")
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()
    plt.close()

def visualize_patch_self_attn(image, attn_map, title):
    print("Visualizing patch self-attention:", title)
    image_np = np.array(image)
    H = image_np.shape[0]
    attn_up = upsample_grid_to_image(attn_map, H)
    attn_up = attn_up / (attn_up.max() + 1e-6)

    plt.figure(figsize=(5, 5))
    plt.imshow(image_np)
    plt.imshow(attn_up, cmap="jet", alpha=0.6)
    plt.colorbar()
    plt.title(title)
    plt.axis("off")
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()
    plt.close()


def get_entity_center_seed(gt_mask):
    """
    gt_mask: [H, W]
    return: int (patch index)
    """
    ys, xs = torch.where(gt_mask > 0)
    y_center = ys.float().mean().round().long()
    x_center = xs.float().mean().round().long()
    return y_center.item() * gt_mask.shape[1] + x_center.item()

def get_background_seed(mask_entity1, mask_entity2):
    """
    mask_entity1, mask_entity2: [H, W]
    return: int (patch index)
    """
    bg_mask = 1 - torch.clamp(mask_entity1 + mask_entity2, 0, 1)
    ys, xs = torch.where(bg_mask > 0)

    if len(xs) == 0:
        return None  # 极端情况

    # choose random background pixel
    idx = np.random.randint(len(xs))
    y_bg = ys[idx].item()
    x_bg = xs[idx].item()
    return y_bg * bg_mask.shape[1] + x_bg


def get_patch_self_attention_map(attn, seed_idx, grid_size):
    """
    attn: [N, N] vision self-attention
    """
    vec = attn[seed_idx]          # seed → all
    vec = vec / (vec.sum() + 1e-6)
    return vec.reshape(grid_size, grid_size)

def compute_entity_metrics(attn_map, gt_mask, grid_size, other_gt_mask=None):
    return {
        "com": compute_center_of_mass_distance(attn_map, gt_mask, grid_size),
        "iou": compute_iou(attn_map, gt_mask, grid_size),
        "soft_iou": compute_soft_iou(attn_map, gt_mask),
        "wasserstein_x": compute_x_wasserstein(attn_map, gt_mask),
        "entity_bg_ratio": compute_entity_background_ratio(attn_map, gt_mask, other_gt_mask),
        "disperson": compute_disperson(attn_map),
    }



# =====================
# Core: prompt-level cross attention
# =====================

def extract_prompt_level_cross_attention(image_path, prompt_text):
    """
    Return:
        cross_attn: [num_patches, num_prompt_tokens]
        input_tokens: list[str]
    """

    # ---- image ----
    image = load_image(image_path)
    image_tensor, _ = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(device, dtype=torch.float16)

    # ---- prompt ----
    if model.config.mm_use_im_start_end:
        inp = (
            DEFAULT_IM_START_TOKEN
            + DEFAULT_IMAGE_TOKEN
            + DEFAULT_IM_END_TOKEN
            + "\n"
            + prompt_text
        )
    else:
        inp = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text

    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    prompt = prompt.replace(
        "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. ",
        ""
    )
    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(device)

    # ---- token list (for indexing sanity) ----
    input_tokens = []
    for tid in input_ids[0]:
        if tid.item() == IMAGE_TOKEN_INDEX:
            input_tokens.append("<image>")
        else:
            input_tokens.append(tokenizer.convert_ids_to_tokens(tid.item()))

    print("=== Input tokens ===")
    for i, t in enumerate(input_tokens):
        print(f"{i:02d}: {t}")

    # ---- forward pass (NO generate) ----
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            images=image_tensor,
            image_sizes=[image.size],
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

    # ---- average attention over layers & heads ----
    # outputs.attentions: list[num_layers] of [1, num_heads, seq, seq]
    # global mean
    attns = torch.stack(
        [layer.squeeze(0).mean(0) for layer in outputs.attentions]
    ).mean(0)   # [seq, seq]

    # or mid-layer mean
    # attns_by_layer = [layer.squeeze(0).mean(0) for layer in outputs.attentions]
    # mid_layers = attns_by_layer[len(attns_by_layer)//3: 2*len(attns_by_layer)//3]
    # attns = torch.stack(mid_layers).mean(0) # [600,600], 600=24+576

    # ---- locate vision tokens ----
    num_patches = model.get_vision_tower().num_patches
    vision_start = attns.shape[0] - num_patches
    vision_end = attns.shape[0]

    # ---- cross attention: vision queries -> prompt tokens ----
    cross_attn = attns[vision_start:vision_end, :vision_start]
    cross_attn[:, :1] = 0      # 屏蔽特殊 token（保留）

    row_sum = cross_attn.sum(dim=1, keepdim=True)
    cross_attn = cross_attn / (row_sum + 1e-6)

    # ===========================================================
    # vision encoder attention extraction (for analysis)
    vis_attn_matrix = aggregate_vit_attention(
        model.get_vision_tower().image_attentions,
        select_layer=model.get_vision_tower().select_layer,
        all_prev_layers=True
    )
    print("Vision encoder attention matrix shape:", vis_attn_matrix.shape)  # [N, N]
    return cross_attn.cpu(), input_tokens, image, vis_attn_matrix.cpu()

# =====================
# Token selection helpers
# =====================

def find_token_positions(input_tokens, keywords):
    """
    keywords: list[str], e.g. ["rectangle"]
    return: list[int] of token indices
    """
    positions = []
    for i, tok in enumerate(input_tokens):
        for kw in keywords:
            if tok == kw:
                positions.append(i)
    return positions


def pool_attention(cross_attn, token_positions, mode="mean"):
    """
    cross_attn: [576, T]
    """
    selected = cross_attn[:, token_positions]
    if mode == "mean":
        selected = selected.mean(dim=1)
    elif mode == "sum":
        selected =selected.sum(dim=1)
    else:
        raise ValueError
    return selected

def gt_bbox_to_patch_mask(entity1, grid_size, image_size=64):
    """
    Convert GT bounding box (normalized 0-1) into a binary mask
    on a patch grid (grid_size x grid_size).
    """
    # normalized → pixel
    xmin = entity1["bounding_box"]["topleft"]["x"] * image_size
    ymin = entity1["bounding_box"]["topleft"]["y"] * image_size
    xmax = entity1["bounding_box"]["bottomright"]["x"] * image_size
    ymax = entity1["bounding_box"]["bottomright"]["y"] * image_size

    patch_size = image_size / grid_size

    # convert pixel bbox → patch indices
    px_min = int(xmin / patch_size)
    py_min = int(ymin / patch_size)
    px_max = int(xmax / patch_size)
    py_max = int(ymax / patch_size)

    mask = torch.zeros(grid_size, grid_size)
    mask[py_min:py_max+1, px_min:px_max+1] = 1

    return mask

def compute_center_of_mass_distance(attn_map, gt_mask, grid_size):
    xs = torch.arange(grid_size).view(1, -1)
    ys = torch.arange(grid_size).view(-1, 1)
    attn_map = attn_map / (attn_map.sum() + 1e-6)
    attn_com = torch.tensor([(attn_map * ys).sum(),
                            (attn_map * xs).sum()])

    gt_mask = gt_mask / (gt_mask.sum() + 1e-6)
    gt_com = torch.tensor([(gt_mask * ys).sum(),
                        (gt_mask * xs).sum()])

    return torch.norm(attn_com - gt_com).item()




def compute_iou(attn_map, gt_mask, grid_size):
    k = int(gt_mask.sum().item())
    flat = attn_map.flatten()
    thresh = torch.topk(flat, k).values.min()

    attn_binary = (flat >= thresh).float().reshape(grid_size, grid_size)

    intersection = (attn_binary * gt_mask).sum().item()
    union = ((attn_binary + gt_mask) > 0).sum().item()
    return intersection / union


def compute_soft_iou(attn_map, gt_mask):
    attn = attn_map / (attn_map.sum() + 1e-6)
    gt   = gt_mask.float()
    intersection = (attn * gt).sum()
    union = attn.sum() + gt.sum() - intersection

    return (intersection / (union + 1e-6)).item()

def compute_x_wasserstein(attn_map, gt_mask):
    # project to x-axis
    p = attn_map.sum(dim=0)
    q = gt_mask.sum(dim=0)

    p = p / (p.sum() + 1e-6)
    q = q / (q.sum() + 1e-6)

    cdf_p = torch.cumsum(p, dim=0)
    cdf_q = torch.cumsum(q, dim=0)

    # W1 distance
    return torch.sum(torch.abs(cdf_p - cdf_q)).item()

def build_relation_gt_x(mask_entity1, mask_entity2, relation):
    """
    Build ground-truth relation distribution on x-axis.
    relation: str, "right" or "left"
    """
    # x-axis projection
    p = mask_entity1.sum(dim=0).float()   # [W]
    q = mask_entity2.sum(dim=0).float()  # [W]
    if relation == "left":
        p, q = q, p
    # normalize to distributions
    p = p / (p.sum() + 1e-6)
    q = q / (q.sum() + 1e-6)

    # CDF difference encodes "right-of"
    cdf_p = torch.cumsum(p, dim=0)
    cdf_q = torch.cumsum(q, dim=0)

    # only keep positive evidence of "p is to the right of q"
    gt_relation_x = torch.relu(cdf_p - cdf_q)

    # normalize again
    gt_relation_x = gt_relation_x / (gt_relation_x.sum() + 1e-6)

    return gt_relation_x

def get_attn_x_distribution(attn_map):
    """
    attn_map: [H, W], attention map for 'right' token
    return:
        attn_x: [W], 概率分布
    """
    attn_x = attn_map.sum(dim=0)
    attn_x = attn_x / (attn_x.sum() + 1e-6)
    return attn_x

def compute_x_wasserstein_from_distributions(p, q):
    """
    p, q: [W], probability distributions
    return:
        W1 distance (scalar)
    """
    cdf_p = torch.cumsum(p, dim=0)
    cdf_q = torch.cumsum(q, dim=0)

    return torch.sum(torch.abs(cdf_p - cdf_q)).item()

def compute_disperson(attn_map):
    H, W = attn_map.shape
    ys = torch.arange(H).view(-1, 1)
    xs = torch.arange(W).view(1, -1)
    attn = attn_map / (attn_map.sum() + 1e-6)
    y_mean = (attn * ys).sum()
    x_mean = (attn * xs).sum()
    var = ((ys - y_mean)**2 + (xs - x_mean)**2) * attn
    dispersion = var.sum().item()
    return dispersion

def compute_in_out_ratio(attn_map, gt_mask):
    inside = (attn_map * gt_mask).sum()
    outside = (attn_map * (1 - gt_mask)).sum()
    return (inside / (outside + 1e-6)).item()

def compute_entity_background_ratio(attn_map, mask_entity, mask_entity_other):
    """
    attn_map: [H, W]
    mask_entity: 当前 entity 的 mask
    mask_entity_other: 另一个 entity 的 mask
    """
    bg_mask = 1 - torch.clamp(mask_entity + mask_entity_other, 0, 1)

    inside = (attn_map * mask_entity).sum()
    background = (attn_map * bg_mask).sum()

    return (inside / (background + 1e-6)).item()


import json
BASE = "/home/maqima/VLM-Visualizer/data/spatial_twoshapes/agreement/relational/test/shard0"
json_path = f"{BASE}/world_model.json"
# save metrics for mean analysis
entity1_com_dists = []
entity2_com_dists = []
entity1_ious = []
entity2_ious = []
entity1_soft_ious = []
entity2_soft_ious = []
entity1_wassersteins = []
entity2_wassersteins = []
relation_x_wassersteins = []


entity1_self_attn_metrics_all = {k: [] for k in ["com", "iou", "soft_iou", "wasserstein_x", "entity_bg_ratio", "disperson"]}
entity2_self_attn_metrics_all = {k: [] for k in ["com", "iou", "soft_iou", "wasserstein_x", "entity_bg_ratio", "disperson"]}
bg_dispersons = []

for sample_idx in [1,3,4,9,10,11,25,26,29,49,50,52,91,99]:
    image_path = f"{BASE}/world-{sample_idx}.png"
    if sample_idx == 1:
        prompt = "a pentagon is to the right of an ellipse ."
    elif sample_idx == 3:
        prompt = "a red triangle is to the left of a blue semicircle ."
    elif sample_idx == 4:
        prompt = "a triangle is to the right of an ellipse ."
    # elif sample_idx == 5:
    #     prompt = "a yellow rectangle is above a circle ."
    elif sample_idx == 9:
        prompt = "a cross is to the right of a semicircle ."
    elif sample_idx == 10:
        prompt = "a cross is to the right of a rectangle ."
    elif sample_idx == 11:
        prompt = "a blue circle is to the right of a semicircle ."
    elif sample_idx == 25:
        prompt = "a blue semicircle is to the right of a yellow cross ."
    elif sample_idx == 26:
        prompt = "a gray cross is to the right of a green semicircle ."
    elif sample_idx == 29:
        prompt = "a blue ellipse is to the right of a yellow square ."
    elif sample_idx == 49:
        prompt = "a magenta pentagon is to the left of a magenta circle ."
    elif sample_idx == 50:
        prompt = "a pentagon is to the right of a magenta circle ."
    elif sample_idx == 52:
        prompt = "a green cross is to the left of a magenta circle ."
    elif sample_idx == 91:
        prompt = "a yellow square is to the right of a blue ellipse ."
    elif sample_idx == 99:
        prompt = "a magenta square is to the left of a magenta circle ."
    else:
        raise ValueError("Only sample_idx 1,3,4 are supported in this script.")
    
    world = json.load(open(json_path))
    entity1 = world[sample_idx]["entities"][0]
    entity1_name = entity1["shape"]["name"]
    entity2 = world[sample_idx]["entities"][1]
    entity2_name = entity2["shape"]["name"]
    mask_entity1 = gt_bbox_to_patch_mask(entity1, grid_size=24)
    mask_entity2 = gt_bbox_to_patch_mask(entity2, grid_size=24)
    cross_attn, input_tokens, image, attn_patch = extract_prompt_level_cross_attention(
        image_path, prompt
    )
    # replace nan in cross_attn as 0
    cross_attn = torch.nan_to_num(cross_attn, nan=0.0)
    print("number of input tokens:", len(input_tokens))
    print("Cross-attention shape:", cross_attn.shape)  # [num_patches, num_prompt_tokens]
    grid_size = 24
    relation = "right" if "right" in prompt else "left"
    # ---- entity / relation token indices ----
    if sample_idx == 1:
        # pentagon, right, ellipse
        entity1_pos = find_token_positions(input_tokens, ["▁pent", "agon"])
        relation_pos = find_token_positions(input_tokens, ["▁right"])
        entity2_pos = find_token_positions(input_tokens, ["▁el", "lipse"])
    elif sample_idx == 3:
        # triangle, left, semicircle
        entity1_pos = find_token_positions(input_tokens, ["▁triangle"])
        relation_pos = find_token_positions(input_tokens, ["▁left"])
        entity2_pos = find_token_positions(input_tokens, ["▁sem", "ic", "irc", "le"])
    elif sample_idx == 4:
        # triangle, right, ellipse
        entity1_pos = find_token_positions(input_tokens, ["▁triangle"])
        relation_pos = find_token_positions(input_tokens, ["▁right"])
        entity2_pos = find_token_positions(input_tokens, ["▁el", "lipse"])
    # elif sample_idx == 5:
    #     # rectangle, above, circle
    #     entity1_pos = find_token_positions(input_tokens, ["▁yellow", "▁rectangle"])
    #     relation_pos = find_token_positions(input_tokens, ["▁above"])
    #     entity2_pos = find_token_positions(input_tokens, ["▁circle"])
    elif sample_idx == 9:
        # cross, right, semicircle
        entity1_pos = find_token_positions(input_tokens, ["▁cross"])
        relation_pos = find_token_positions(input_tokens, ["▁right"])
        entity2_pos = find_token_positions(input_tokens, ["▁sem", "ic", "irc", "le"])
    elif sample_idx == 10:
        # cross, right, rectangle
        entity1_pos = find_token_positions(input_tokens, ["▁cross"])
        relation_pos = find_token_positions(input_tokens, ["▁right"])
        entity2_pos = find_token_positions(input_tokens, ["▁rectangle"])
    elif sample_idx == 11:
        # circle, right, semicircle
        entity1_pos = find_token_positions(input_tokens, ["▁blue", "▁circle"])
        relation_pos = find_token_positions(input_tokens, ["▁right"])
        entity2_pos = find_token_positions(input_tokens, ["▁sem", "ic", "irc", "le"])
    elif sample_idx == 25:
        # semicircle, right, cross
        entity1_pos = find_token_positions(input_tokens, ["▁blue", "▁sem", "ic", "irc", "le"])
        relation_pos = find_token_positions(input_tokens, ["▁right"])
        entity2_pos = find_token_positions(input_tokens, ["▁yellow", "▁cross"])
    elif sample_idx == 26:
        # cross, right, semicircle
        entity1_pos = find_token_positions(input_tokens, ["▁gray", "▁cross"])
        relation_pos = find_token_positions(input_tokens, ["▁right"])
        entity2_pos = find_token_positions(input_tokens, ["▁green", "▁sem", "ic", "irc", "le"])
    elif sample_idx == 29:
        # ellipse, right, square
        entity1_pos = find_token_positions(input_tokens, ["▁blue", "▁el", "lipse"])
        relation_pos = find_token_positions(input_tokens, ["▁right"])
        entity2_pos = find_token_positions(input_tokens, ["▁yellow", "▁square"])
    elif sample_idx == 49:
        # pentagon, left, circle
        entity1_pos = find_token_positions(input_tokens, ["▁magenta", "▁pent", "agon"])
        relation_pos = find_token_positions(input_tokens, ["▁left"])
        entity2_pos = find_token_positions(input_tokens, ["▁magenta", "▁circle"])
    elif sample_idx == 50:
        # pentagon, right, circle
        entity1_pos = find_token_positions(input_tokens, ["▁pent", "agon"])
        relation_pos = find_token_positions(input_tokens, ["▁right"])
        entity2_pos = find_token_positions(input_tokens, ["▁magenta", "▁circle"])
    elif sample_idx == 52:
        # cross, left, circle
        entity1_pos = find_token_positions(input_tokens, ["▁green", "▁cross"])
        relation_pos = find_token_positions(input_tokens, ["▁left"])
        entity2_pos = find_token_positions(input_tokens, ["▁magenta", "▁circle"])
    elif sample_idx == 91:
        # square, right, ellipse
        entity1_pos = find_token_positions(input_tokens, ["▁yellow", "▁square"])
        relation_pos = find_token_positions(input_tokens, ["▁right"])
        entity2_pos = find_token_positions(input_tokens, ["▁blue", "▁el", "lipse"])
    elif sample_idx == 99:
        # square, left, circle
        entity1_pos = find_token_positions(input_tokens, ["▁magenta", "▁square"])
        relation_pos = find_token_positions(input_tokens, ["▁left"])
        entity2_pos = find_token_positions(input_tokens, ["▁magenta", "▁circle"])

    else:
        raise ValueError("Only sample_idx 1,3,4 are supported in this script.")
    print(f"{entity1_name} tokens:", entity1_pos)
    print(f"{relation} tokens:", relation_pos)
    print(f"{entity2_name} tokens:", entity2_pos)

    # ---- pooled attentions ----
    entity1_attn = pool_attention(cross_attn, entity1_pos)
    entity2_attn = pool_attention(cross_attn, entity2_pos)
    relation_attn = pool_attention(cross_attn, relation_pos)


    # ---- reshape ----
    entity1_map = entity1_attn.reshape(grid_size, grid_size)
    entity2_map = entity2_attn.reshape(grid_size, grid_size)
    relation_map = relation_attn.reshape(grid_size, grid_size)

    save_dir = "left_right_analysis_outputs"
    os.makedirs(save_dir, exist_ok=True)
    # ---- visualize ----
    visualize(image, entity1_map, f"{save_dir}/{sample_idx} {entity1_name} attention")
    visualize(image, relation_map, f"{save_dir}/{sample_idx} {relation} attention")
    visualize(image, entity2_map, f"{save_dir}/{sample_idx} {entity2_name} attention")
    # ---- compute metrics ----
    entity2_com_dist = compute_center_of_mass_distance(entity2_map, mask_entity2, grid_size)
    entity2_iou = compute_iou(entity2_map, mask_entity2, grid_size)
    entity1_com_dist = compute_center_of_mass_distance(entity1_map, mask_entity1, grid_size)
    entity1_iou = compute_iou(entity1_map, mask_entity1, grid_size)
    entity2_soft_iou = compute_soft_iou(entity2_map, mask_entity2)
    entity1_soft_iou = compute_soft_iou(entity1_map, mask_entity1)
    entity2_wasserstein = compute_x_wasserstein(entity2_map, mask_entity2)
    entity1_wasserstein = compute_x_wasserstein(entity1_map, mask_entity1)
    print("sample idx:", sample_idx)
    print(f"Entity1 - COM distance: {entity1_com_dist:.2f}, IoU: {entity1_iou:.4f}, Soft IoU: {entity1_soft_iou:.4f}, Wasserstein: {entity1_wasserstein:.4f}")
    print(f"Entity2 - COM distance: {entity2_com_dist:.2f}, IoU: {entity2_iou:.4f}, Soft IoU: {entity2_soft_iou:.4f}, Wasserstein: {entity2_wasserstein:.4f}")
    # write into a txt file
    with open(f"{save_dir}/metrics_sample_{sample_idx}.txt", "w") as f:
        f.write(f"sample idx: {sample_idx}\n")
        f.write(f"Entity1 - COM distance: {entity1_com_dist:.2f}, IoU: {entity1_iou:.4f}, Soft IoU: {entity1_soft_iou:.4f}, Wasserstein: {entity1_wasserstein:.4f}\n")
        f.write(f"Entity2 - COM distance: {entity2_com_dist:.2f}, IoU: {entity2_iou:.4f}, Soft IoU: {entity2_soft_iou:.4f}, Wasserstein: {entity2_wasserstein:.4f}\n")
    if sample_idx == 5:
        continue    # skip relation analysis for "above" case
    # 1. build GT relation distribution on x-axis
    gt_relation_x = build_relation_gt_x(
        mask_entity1, mask_entity2, relation
    )

    # 2. get attention x distribution for "right" token
    attn_x = get_attn_x_distribution(relation_map)

    # 3. compute Wasserstein distance
    relation_x_wasserstein = compute_x_wasserstein_from_distributions(
        attn_x, gt_relation_x
    )

    print(f"Right token x-axis Wasserstein: {relation_x_wasserstein:.4f}")
    with open(f"{save_dir}/metrics_sample_{sample_idx}.txt", "a") as f:
        f.write(f"Right token x-axis Wasserstein: {relation_x_wasserstein:.4f}\n")

    # vision tower self-attention analysis

    # ---- entity seeds ----
    seed1 = get_entity_center_seed(mask_entity1)
    seed2 = get_entity_center_seed(mask_entity2)
    bg_seed = get_background_seed(mask_entity1, mask_entity2)

    # ---- self-attention maps ----
    entity1_self_map = get_patch_self_attention_map(attn_patch, seed1, 24)
    entity2_self_map = get_patch_self_attention_map(attn_patch, seed2, 24)
    print(f"[Sample {sample_idx}] Self-Attention Metrics")
    if bg_seed is not None:
        bg_self_map = get_patch_self_attention_map(attn_patch, bg_seed, 24)
        bg_disperson = compute_disperson(bg_self_map)
        print(f"Background disperson: {bg_disperson:.4f}")
        with open(f"{save_dir}/metrics_sample_{sample_idx}.txt", "a") as f:
            f.write(f"Background disperson: {bg_disperson:.4f}\n")
        bg_dispersons.append(bg_disperson)

    # ---- visualization ----
    visualize_patch_self_attn(
        image, entity1_self_map,
        f"{save_dir}/{sample_idx}_entity1_self_attention"
    )
    visualize_patch_self_attn(
        image, entity2_self_map,
        f"{save_dir}/{sample_idx}_entity2_self_attention"
    )

    # ---- metrics ----
    m1 = compute_entity_metrics(entity1_self_map, mask_entity1, 24, other_gt_mask=mask_entity2)
    m2 = compute_entity_metrics(entity2_self_map, mask_entity2, 24, other_gt_mask=mask_entity1)

    # ---- print ----
    
    print(f"Entity1: {m1}")
    print(f"Entity2: {m2}")

    # ---- save per-sample ----
    with open(f"{save_dir}/metrics_sample_{sample_idx}.txt", "a") as f:
        f.write(f"Sample {sample_idx} - Vision Self-Attention\n")
        for k, v in m1.items():
            f.write(f"Entity1 {k}: {v:.4f}\n")
        for k, v in m2.items():
            f.write(f"Entity2 {k}: {v:.4f}\n")

    # ---- collect ----
    for k in entity1_self_attn_metrics_all:
        entity1_self_attn_metrics_all[k].append(m1[k])
        entity2_self_attn_metrics_all[k].append(m2[k])

    # ---- collect for mean analysis ----
    entity1_com_dists.append(entity1_com_dist)
    entity2_com_dists.append(entity2_com_dist)
    entity1_ious.append(entity1_iou)
    entity2_ious.append(entity2_iou)
    entity1_soft_ious.append(entity1_soft_iou)
    entity2_soft_ious.append(entity2_soft_iou)
    entity1_wassersteins.append(entity1_wasserstein)
    entity2_wassersteins.append(entity2_wasserstein)
    relation_x_wassersteins.append(relation_x_wasserstein)
# ---- mean analysis ----
num_samples = len(entity1_com_dists)
print("=== Mean Analysis over all samples ===")
print(f"Entity1 - COM distance: {np.mean(entity1_com_dists):.2f}, IoU: {np.mean(entity1_ious):.4f}, Soft IoU: {np.mean(entity1_soft_ious):.4f}, Wasserstein: {np.mean(entity1_wassersteins):.4f}")
print(f"Entity2 - COM distance: {np.mean(entity2_com_dists):.2f}, IoU: {np.mean(entity2_ious):.4f}, Soft IoU: {np.mean(entity2_soft_ious):.4f}, Wasserstein: {np.mean(entity2_wassersteins):.4f}")
print(f"Relation token x-axis Wasserstein: {np.mean(relation_x_wassersteins):.4f}")
with open(f"{save_dir}/mean_analysis.txt", "w") as f:
    f.write("=== Mean Analysis over all samples ===\n")
    f.write(f"Entity1 - COM distance: {np.mean(entity1_com_dists):.2f}, IoU: {np.mean(entity1_ious):.4f}, Soft IoU: {np.mean(entity1_soft_ious):.4f}, Wasserstein: {np.mean(entity1_wassersteins):.4f}\n")
    f.write(f"Entity2 - COM distance: {np.mean(entity2_com_dists):.2f}, IoU: {np.mean(entity2_ious):.4f}, Soft IoU: {np.mean(entity2_soft_ious):.4f}, Wasserstein: {np.mean(entity2_wassersteins):.4f}\n")
    f.write(f"Relation token x-axis Wasserstein: {np.mean(relation_x_wassersteins):.4f}\n")


print("=== Mean Vision Self-Attention Analysis ===")

with open(f"{save_dir}/mean_analysis.txt", "a") as f:
    f.write("=== Mean Vision Self-Attention Analysis ===\n")

    for k in entity1_self_attn_metrics_all:
        e1_mean = np.mean(entity1_self_attn_metrics_all[k])
        e2_mean = np.mean(entity2_self_attn_metrics_all[k])

        print(f"{k}: Entity1={e1_mean:.4f}, Entity2={e2_mean:.4f}")

        f.write(f"{k}: Entity1={e1_mean:.4f}, Entity2={e2_mean:.4f}\n")
    if len(bg_dispersons) > 0:
        bg_disperson_mean = np.mean(bg_dispersons)
        print(f"Background disperson mean: {bg_disperson_mean:.4f}")
        f.write(f"Background disperson mean: {bg_disperson_mean:.4f}\n")
