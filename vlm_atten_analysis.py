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

def gt_bbox_to_patch_mask(entity, grid_size, image_size=64):
    """
    Convert GT bounding box (normalized 0-1) into a binary mask
    on a patch grid (grid_size x grid_size).
    """
    # normalized → pixel
    xmin = entity["bounding_box"]["topleft"]["x"] * image_size
    ymin = entity["bounding_box"]["topleft"]["y"] * image_size
    xmax = entity["bounding_box"]["bottomright"]["x"] * image_size
    ymax = entity["bounding_box"]["bottomright"]["y"] * image_size

    patch_size = image_size / grid_size

    # convert pixel bbox → patch indices
    px_min = int(xmin / patch_size)
    py_min = int(ymin / patch_size)
    px_max = int(xmax / patch_size)
    py_max = int(ymax / patch_size)

    mask = torch.zeros(grid_size, grid_size)
    mask[py_min:py_max+1, px_min:px_max+1] = 1
    return mask

def reshape_attention_to_grid(attn, grid_size):
    """
    attn: [num_text_tokens, num_visual_patches]
    returns: [grid_size, grid_size] averaged over text tokens
    """
    num_patches = grid_size * grid_size
    attn = attn[:, :num_patches]  # ensure alignment
    attn_map = attn.mean(dim=1).reshape(grid_size, grid_size)
    attn_map = attn_map / attn_map.sum()
    return attn_map


def compute_center_of_mass_distance(cross_attention_maps, gt_mask, grid_size):
    """
    cross_attention_maps: [num_text_tokens, num_visual_patches]
    gt_mask: [grid_size, grid_size]
    """
    attn_map = reshape_attention_to_grid(cross_attention_maps, grid_size)

    # coordinates
    xs = torch.arange(grid_size).view(1, -1)  # row vector
    ys = torch.arange(grid_size).view(-1, 1)  # column vector

    attn_com = torch.tensor([
        (attn_map * ys).sum(),
        (attn_map * xs).sum()
    ])

    gt_mask = gt_mask / (gt_mask.sum() + 1e-6)
    gt_com = torch.tensor([
        (gt_mask * ys).sum(),
        (gt_mask * xs).sum()
    ])

    return torch.norm(attn_com - gt_com).item()



def compute_iou(cross_attention_maps, gt_mask, grid_size, topk_ratio=0.1):
    attn_map = reshape_attention_to_grid(cross_attention_maps, grid_size)

    # threshold by top-k rather than fixed threshold
    k = int(grid_size * grid_size * topk_ratio)
    flat = attn_map.flatten()
    threshold = torch.topk(flat, k).values.min()

    attn_binary = (attn_map >= threshold).float()

    intersection = (attn_binary * gt_mask).sum().item()
    union = ((attn_binary + gt_mask) > 0).sum().item()

    return intersection / union



import json

i = 4    # example index
json_path = f"/home/maqima/VLM-Visualizer/data/spatial_twoshapes/agreement/relational/test/shard0/world_model.json"
img_path = f"/home/maqima/VLM-Visualizer/data/spatial_twoshapes/agreement/relational/test/shard0/world-{i}.png"

world = json.load(open(json_path))
entity = world[i]["entities"][0]   # choose entity 0 for example

grid_size = 24   # depends on llava vision tower
gt_mask = gt_bbox_to_patch_mask(entity, grid_size)

attn = extract_input_cross_attention_maps(img_path, "the circle is left of the square")
print(attn.shape)
com = compute_center_of_mass_distance(attn, gt_mask, grid_size)
iou = compute_iou(attn, gt_mask, grid_size)

print("COM:", com)
print("IoU:", iou)
