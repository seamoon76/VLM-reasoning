import os
import numpy as np
import torch
import torch.nn.functional as F
from utils import (
    load_image,
    aggregate_llm_attention,
    aggregate_vit_attention,
    heterogenous_stack,
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

def compute_center_of_mass_distance(cross_attention_maps, ground_truth):
    """
    Compute the center-of-mass distance between cross-attention maps and ground-truth annotations.

    Args:
        cross_attention_maps (torch.Tensor): Cross-attention maps of shape [num_text_tokens, num_visual_patches].
        ground_truth (torch.Tensor): Ground-truth binary mask of shape [grid_size, grid_size].

    Returns:
        float: Center-of-mass distance.
    """
    # Compute the center of mass for the attention map
    grid_size = int(ground_truth.shape[0] ** 0.5)
    attn_map = cross_attention_maps.mean(dim=0).reshape(grid_size, grid_size)
    attn_map = attn_map / attn_map.sum()

    attn_com = torch.tensor(
        [torch.sum(attn_map * torch.arange(grid_size).view(-1, 1)),
         torch.sum(attn_map * torch.arange(grid_size).view(1, -1))]
    )

    # Compute the center of mass for the ground-truth mask
    gt_com = torch.tensor(
        [torch.sum(ground_truth * torch.arange(grid_size).view(-1, 1)),
         torch.sum(ground_truth * torch.arange(grid_size).view(1, -1))]
    )

    # Compute the Euclidean distance between the two centers of mass
    return torch.norm(attn_com - gt_com).item()


def compute_iou(cross_attention_maps, ground_truth, threshold=0.5):
    """
    Compute the Intersection over Union (IoU) between cross-attention maps and ground-truth annotations.

    Args:
        cross_attention_maps (torch.Tensor): Cross-attention maps of shape [num_text_tokens, num_visual_patches].
        ground_truth (torch.Tensor): Ground-truth binary mask of shape [grid_size, grid_size].
        threshold (float): Threshold to binarize the attention map.

    Returns:
        float: IoU score.
    """
    grid_size = int(ground_truth.shape[0] ** 0.5)
    attn_map = cross_attention_maps.mean(dim=0).reshape(grid_size, grid_size)
    attn_map = attn_map / attn_map.max()

    # Binarize the attention map
    attn_binary = (attn_map > threshold).float()

    # Compute IoU
    intersection = (attn_binary * ground_truth).sum().item()
    union = (attn_binary + ground_truth).clamp(0, 1).sum().item()

    return intersection / union


# Example usage
if __name__ == "__main__":
    image_path = "/home/maqima/VLM-Visualizer/data/spatial_twoshapes/agreement/relational/test/shard0/world-1.png"
    prompt_text = "a pentagon is to the right of an ellipse."

    # Extract cross-attention maps
    cross_attention_maps = extract_cross_attention_maps(image_path, prompt_text)

    # print shape
    print("Cross-Attention Maps Shape:", cross_attention_maps.shape)

    # Example ground-truth mask (binary mask of shape [grid_size, grid_size])
    grid_size = 14  # Example grid size
    ground_truth = torch.zeros(grid_size, grid_size)
    ground_truth[3:5, 6:8] = 1  # Example ground-truth region

    # Compute metrics
    com_distance = compute_center_of_mass_distance(cross_attention_maps, ground_truth)
    iou_score = compute_iou(cross_attention_maps, ground_truth)

    print(f"Center-of-Mass Distance: {com_distance}")
    print(f"IoU Score: {iou_score}")