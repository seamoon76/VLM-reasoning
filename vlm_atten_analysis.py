import os
import numpy as np
import torch
import torch.nn.functional as F
import sys
sys.path.append("./models")

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from utils import (
    load_image,
    aggregate_llm_attention, aggregate_vit_attention,
    heterogenous_stack,
    show_mask_on_image
)
model_path = "liuhaotian/llava-v1.5-7b"

load_8bit = False
load_4bit = False
device = "cuda" if torch.cuda.is_available() else "cpu"
disable_torch_init()

model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path,
    None,
    model_name,
    load_8bit,
    load_4bit,
    device=device,
    attn_implementation="eager",
    torch_dtype=torch.bfloat16
)

def renormalize_cross_attention_with_pool(cross_attn_maps):
    """
    cross_attn_maps: [num_visual_patches, num_text_tokens]
    return: vector [num_visual_patches] normalized to sum=1
    """
    # max-pooling across text tokens
    # top 3 averaging
    topk = 3
    topk_vals, _ = torch.topk(cross_attn_maps, topk, dim=1)
    pooled = topk_vals.mean(dim=1)
    #pooled = cross_attn_maps.max(dim=1).values   # [num_visual_patches]
    pooled = torch.clamp(pooled, min=0)
    pooled = pooled / (pooled.sum() + 1e-6)
    return pooled

def renormalize_cross_attention_with_pool_sharpen(cross_attn_maps):
    #pooled = cross_attn_maps.max(dim=1).values
    
    # top 3 averaging
    topk = 3
    topk_vals, _ = torch.topk(cross_attn_maps, topk, dim=1)
    pooled = topk_vals.mean(dim=1)
    max_idx = pooled.argmax()
    mask = torch.zeros_like(pooled)
    mask[max_idx] = 1
    return mask

def threshold_topk_by_gt_area(cross_attn_maps, gt_mask):
    """
    cross_attn_maps: [num_visual_patches, num_text_tokens]
    gt_mask: [grid_size, grid_size]
    """
    # max-pooling across text tokens
    #pooled = cross_attn_maps.max(dim=1).values
    # top 3 averaging
    topk = 3
    topk_vals, _ = torch.topk(cross_attn_maps, topk, dim=1)
    pooled = topk_vals.mean(dim=1)

    pooled = torch.clamp(pooled, min=0)

    # number of GT patches
    k = int(gt_mask.sum().item())
    k = max(k, 1)

    # pick top-k patches
    topk_vals, topk_idx = torch.topk(pooled, k)

    binary = torch.zeros_like(pooled)
    binary[topk_idx] = 1
    return binary

def extract_input_cross_attention_maps(image_path_or_url, prompt_text):
    """
    Extract cross-attention maps between visual patches and text tokens.

    Args:
        image_path_or_url (str): Path to the input image.
        prompt_text (str): Input text prompt.

    Returns:
        torch.Tensor: Cross-attention maps of shape [num_visual_patches, num_text_tokens].
    """
    # Load and preprocess the image
    image = load_image(image_path_or_url)
    image_tensor, images = process_images([image], image_processor, model.config)
    image_size = image.size

    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    # Prepare the input prompt
    if model.config.mm_use_im_start_end:
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt_text
    else:
        inp = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text

    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).to(model.device)

    # Generate outputs and extract attention
    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            do_sample=False,
            max_new_tokens=512,
            use_cache=True,
            return_dict_in_generate=True,
            output_attentions=True,
        )
        print(model.get_vision_tower().num_patches)
        

    # Extract LLM attention matrix
    aggregated_prompt_attention = []
    for layer in outputs["attentions"][0]:
        layer_attns = layer.squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        cur = attns_per_head[:-1].cpu().clone()
        cur[1:, 0] = 0.  # Zero out attention to the first <bos> token
        cur[1:] = cur[1:] / cur[1:].sum(-1, keepdim=True)
        aggregated_prompt_attention.append(cur)

    aggregated_prompt_attention = torch.stack(aggregated_prompt_attention).mean(dim=0)
    llm_attn_matrix = heterogenous_stack(
        [torch.tensor([1])]
        + list(aggregated_prompt_attention)
        + list(map(aggregate_llm_attention, outputs["attentions"]))
    )

    # Extract cross-attention maps
    vision_token_start = len(tokenizer(prompt.split("<image>")[0], return_tensors='pt')["input_ids"][0])
    vision_token_end = vision_token_start + model.get_vision_tower().num_patches

    cross_attention_maps = llm_attn_matrix[vision_token_start:vision_token_end, :vision_token_start]

    return cross_attention_maps

def gt_bbox_to_patch_mask(entity1, entity2, grid_size, image_size=64):
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

    xmin = entity1["bounding_box"]["topleft"]["x"] * image_size
    ymin = entity1["bounding_box"]["topleft"]["y"] * image_size
    xmax = entity1["bounding_box"]["bottomright"]["x"] * image_size
    ymax = entity1["bounding_box"]["bottomright"]["y"] * image_size

    px_min = int(xmin / patch_size)
    py_min = int(ymin / patch_size)
    px_max = int(xmax / patch_size)
    py_max = int(ymax / patch_size)
    mask[py_min:py_max+1, px_min:px_max+1] = 1

    return mask

def reshape_attention_to_grid(attn_vec, grid_size):
    """
    attn_vec: [num_visual_patches] after re-normalization
    return: [grid_size, grid_size]
    """
    num_patches = grid_size * grid_size
    attn_vec = attn_vec[:num_patches]
    attn_map = attn_vec.reshape(grid_size, grid_size)
    return attn_map



def compute_center_of_mass_distance(attn_map, gt_mask, grid_size):
    xs = torch.arange(grid_size).view(1, -1)
    ys = torch.arange(grid_size).view(-1, 1)

    attn_com = torch.tensor([(attn_map * ys).sum(),
                             (attn_map * xs).sum()])

    gt_mask = gt_mask / (gt_mask.sum() + 1e-6)
    gt_com = torch.tensor([(gt_mask * ys).sum(),
                           (gt_mask * xs).sum()])

    return torch.norm(attn_com - gt_com).item()




def compute_iou(attn_map, gt_mask, grid_size, topk_ratio=0.1):
    # threshold at top-k patches
    k = int(grid_size * grid_size * topk_ratio)
    flat = attn_map.flatten()
    thresh = torch.topk(flat, k).values.min()

    attn_binary = (flat >= thresh).float().reshape(grid_size, grid_size)

    intersection = (attn_binary * gt_mask).sum().item()
    union = ((attn_binary + gt_mask) > 0).sum().item()
    return intersection / union



import json

import json
import numpy as np

BASE = "/home/maqima/VLM-Visualizer/data/spatial_twoshapes/agreement/relational/test/shard0"
json_path = f"{BASE}/world_model.json"
agreement_path = f"{BASE}/agreement.txt"

world = json.load(open(json_path))
agreement = [float(x.strip()) for x in open(agreement_path).readlines()]  # 长度 100

grid_size = 24  # depends on llava vision tower

all_com = []
all_iou = []

for i, flag in enumerate(agreement):

    if flag != 1.0:
        continue 
    print(f"Processing sample {i}...")

    img_path = f"{BASE}/world-{i}.png"

    entity1 = world[i]["entities"][0]
    entity2 = world[i]["entities"][1]

    gt_mask = gt_bbox_to_patch_mask(entity1, entity2, grid_size)

    attn = extract_input_cross_attention_maps(img_path, "the circle is left of the square")
    # print("attn shape:", attn.shape)
    # print("before renorm:",attn.max(), attn.mean(), attn.min())
    #normed = threshold_topk_by_gt_area(attn, gt_mask)
    normed = renormalize_cross_attention_with_pool_sharpen(attn)
    # print("atten shape:", normed.shape)
    # print("after renorm:", normed.max(), normed.mean(), normed.min())
    attn_map = reshape_attention_to_grid(normed, grid_size)

    com = compute_center_of_mass_distance(attn_map, gt_mask, grid_size)
    iou = compute_iou(attn_map, gt_mask, grid_size)


    all_com.append(com)
    all_iou.append(iou)

avg_com = np.mean(all_com)
avg_iou = np.mean(all_iou)

print("=================================")
print(f"Processed samples: {len(all_com)}")
print(f"Average CoM: {avg_com:.4f}")
print(f"Average IoU: {avg_iou:.4f}")
print("=================================")

