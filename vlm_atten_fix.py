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

from utils import load_image   # 你原来就有

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
    grid_np = grid.cpu().numpy()
    return cv2.resize(grid_np, (image_size, image_size), interpolation=cv2.INTER_NEAREST)


def visualize(image, attn_map, title):
    image_np = np.array(image)
    H = image_np.shape[0]
    print("attn_map shape:", attn_map.shape)
    attn_up = upsample_grid_to_image(attn_map, H)
    print("upsampled attn_map shape:", attn_up.shape)
    plt.figure(figsize=(5, 5))
    plt.imshow(image_np)
    plt.imshow(attn_up, cmap="jet", alpha=0.6)
    plt.colorbar()
    plt.title(title)
    plt.axis("off")
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()
    plt.close()


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
    # attns = torch.stack(
    #     [layer.squeeze(0).mean(0) for layer in outputs.attentions]
    # ).mean(0)   # [seq, seq]

    # or mid-layer mean
    attns_by_layer = [layer.squeeze(0).mean(0) for layer in outputs.attentions]
    mid_layers = attns_by_layer[len(attns_by_layer)//3: 2*len(attns_by_layer)//3]
    attns = torch.stack(mid_layers).mean(0) # [600,600], 600=24+576

    # ---- locate vision tokens ----
    num_patches = model.get_vision_tower().num_patches
    vision_start = attns.shape[0] - num_patches
    vision_end = attns.shape[0]

    # ---- cross attention: vision queries -> prompt tokens ----
    cross_attn = attns[vision_start:vision_end, :vision_start]
    cross_attn[:, :1] = 0      # 屏蔽特殊 token（保留）

    row_sum = cross_attn.sum(dim=1, keepdim=True)
    cross_attn = cross_attn / (row_sum + 1e-6)
    return cross_attn.cpu(), input_tokens, image


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
        return selected.mean(dim=1)
    elif mode == "sum":
        return selected.sum(dim=1)
    else:
        raise ValueError

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

BASE = "/home/maqima/VLM-Visualizer/data/spatial_twoshapes/agreement/relational/test/shard0"
json_path = f"{BASE}/world_model.json"
sample_idx = 1
image_path = f"{BASE}/world-{sample_idx}.png"
prompt = "a pentagon is to the right of an ellipse ."
import json
world = json.load(open(json_path))
entity1 = world[sample_idx]["entities"][0]
entity2 = world[sample_idx]["entities"][1]
mask_pentagon = gt_bbox_to_patch_mask(entity1, grid_size=24)
mask_ellipse = gt_bbox_to_patch_mask(entity2, grid_size=24)
cross_attn, input_tokens, image = extract_prompt_level_cross_attention(
    image_path, prompt
)
# replace nan in cross_attn as 0
cross_attn = torch.nan_to_num(cross_attn, nan=0.0)
print("number of input tokens:", len(input_tokens))
print("Cross-attention shape:", cross_attn.shape)  # [num_patches, num_prompt_tokens]
grid_size = 24

# ---- entity / relation token indices ----
pentagon_pos = find_token_positions(input_tokens, ["▁pent", "agon"])
right_pos = find_token_positions(input_tokens, ["▁right"])
ellipse_pos = find_token_positions(input_tokens, ["▁el", "lipse"])

print("pentagon tokens:", pentagon_pos)
print("right tokens:", right_pos)
print("ellipse tokens:", ellipse_pos)

# ---- pooled attentions ----
ellipse_attn = pool_attention(cross_attn, ellipse_pos)
right_attn = pool_attention(cross_attn, right_pos)
pentagon_attn = pool_attention(cross_attn, pentagon_pos)

# ---- reshape ----
ellipse_map = ellipse_attn.reshape(grid_size, grid_size)
right_map = right_attn.reshape(grid_size, grid_size)
pentagon_map = pentagon_attn.reshape(grid_size, grid_size)

# ---- visualize ----
visualize(image, ellipse_map, "ellipse attention")
visualize(image, right_map, "right (relation) attention")
visualize(image, pentagon_map, "pentagon attention")

# ---- compute metrics ----
ellipse_com_dist = compute_center_of_mass_distance(ellipse_map, mask_ellipse, grid_size)
ellipse_iou = compute_iou(ellipse_map, mask_ellipse, grid_size)
pentagon_com_dist = compute_center_of_mass_distance(pentagon_map, mask_pentagon, grid_size)
pentagon_iou = compute_iou(pentagon_map, mask_pentagon, grid_size)
ellipse_soft_iou = compute_soft_iou(ellipse_map, mask_ellipse)
pentagon_soft_iou = compute_soft_iou(pentagon_map, mask_pentagon)
ellipse_wasserstein = compute_x_wasserstein(ellipse_map, mask_ellipse)
pentagon_wasserstein = compute_x_wasserstein(pentagon_map, mask_pentagon)

print(f"Ellipse - COM distance: {ellipse_com_dist:.2f}, IoU: {ellipse_iou:.4f}, Soft IoU: {ellipse_soft_iou:.4f}, Wasserstein: {ellipse_wasserstein:.4f}")
print(f"Pentagon - COM distance: {pentagon_com_dist:.2f}, IoU: {pentagon_iou:.4f}, Soft IoU: {pentagon_soft_iou:.4f}, Wasserstein: {pentagon_wasserstein:.4f}")