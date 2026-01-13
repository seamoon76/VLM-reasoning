"""
Spatial Relation Complexity Analysis: Entropy and Dispersion Comparison

This script performs a controlled experiment comparing attention patterns
between complex spatial relations (left/right/above/below) and simple relations (adjacent).
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm

sys.path.append("./models")
sys.path.append(".")

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
from dataloader import load_all_shards
from vlm_atten_analysis_llava import (
    extract_prompt_level_cross_attention,
    find_word_token_positions,
    pool_attention,
    gt_bbox_to_patch_mask,
    get_entity_center_seed,
    get_patch_self_attention_map,
    compute_entropy,
    compute_disperson,
)

# =====================
# Model loading
# =====================
model_path = "liuhaotian/llava-v1.5-7b"
device = "cuda" if torch.cuda.is_available() else "cpu"

disable_torch_init()

print("Loading model...")
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
print("Model loaded successfully.")

# =====================
# Caption replacement function
# =====================

COMPLEX_RELATIONS = ["left", "right", "above", "below"]

# def parse_relation_sentence(sent):
#     """
#     Parse sentences like:
#     'there are two shapes a green rectangle and a green triangle . 
#     what is the spatial relationship from the first shape to the second shape ? 
#     the spatial relation is above '
    
#     Returns:
#         (entity1, rel, entity2) or None
#     """
#     sent = sent.strip().lower()
#     if "adjacent to" in sent:
#         pattern = " is adjacent to "
#         left, right = sent.split(pattern)
#         entity1 = left.strip()
#         entity2 = right.replace(".", "").strip()
#         return entity1, "adjacent", entity2
#     for rel in COMPLEX_RELATIONS:
#         pattern = f" is to the {rel} of "
#         if pattern in sent:
#             left, right = sent.split(pattern)
#             entity1 = left.strip()
#             entity2 = right.replace(".", "").strip()
#             return entity1, rel, entity2
#     return None

def replace_complex_relation_with_adjacent(caption):
    """
    Replace sentence-final spatial relation (above/below/left/right)
    with 'adjacent to' in the new QA-style template.
    """
    caption = caption.strip()

    COMPLEX_RELATIONS = ["left", "right", "above", "below"]

    for rel in COMPLEX_RELATIONS:
        # match: "the spatial relation is above"
        pattern = f"the spatial relation is {rel}"
        if pattern in caption.lower():
            # preserve original casing except replaced phrase
            return caption[:-len(rel)] + "adjacent to"

    return None


# =====================
# Token finding helpers
# =====================

def find_entity_tokens(input_tokens, entity_text, world_entity=None):
    """
    Find token positions for an entity from caption text.
    
    Args:
        input_tokens: List of tokenized tokens
        entity_text: Entity text from caption (e.g., "a gray ellipse")
        world_entity: Optional world entity dict with shape info
    
    Returns:
        List of token indices
    """
    positions = []
    entity_text_lower = entity_text.lower()
    
    # Extract shape name if available
    shape_name = None
    if world_entity:
        shape_name = world_entity.get("shape", {}).get("name", "").lower()
    
    # Common shape names and their possible tokenizations
    shape_tokens_map = {
        "circle": ["▁circle", "circle"],
        "ellipse": ["▁el", "lipse", "▁ellipse"],
        "triangle": ["▁triangle", "triangle"],
        "square": ["▁square", "square"],
        "rectangle": ["▁rectangle", "rectangle"],
        "pentagon": ["▁pent", "agon", "▁pentagon"],
        "semicircle": ["▁sem", "ic", "irc", "le", "▁semicircle"],
        "cross": ["▁cross", "cross"],
    }
    
    # Try to find shape tokens
    if shape_name and shape_name in shape_tokens_map:
        positions = find_token_positions(input_tokens, shape_tokens_map[shape_name])
    
    # If not found, try color + shape combinations
    if len(positions) == 0:
        # Extract color if present
        colors = ["red", "blue", "green", "yellow", "gray", "magenta"]
        color_token = None
        for color in colors:
            if color in entity_text_lower:
                color_token = f"▁{color}"
                break
        
        if color_token and shape_name and shape_name in shape_tokens_map:
            # Look for color token followed by shape tokens
            for i in range(len(input_tokens) - len(shape_tokens_map[shape_name])):
                if input_tokens[i] == color_token:
                    # Check if following tokens match shape
                    match = True
                    for j, shape_tok in enumerate(shape_tokens_map[shape_name]):
                        if i + j + 1 >= len(input_tokens):
                            match = False
                            break
                        if shape_tok not in input_tokens[i + j + 1]:
                            match = False
                            break
                    if match:
                        positions = [i] + list(range(i + 1, i + 1 + len(shape_tokens_map[shape_name])))
                        break
    
    # Fallback: search for any token containing shape name
    if len(positions) == 0 and shape_name:
        for i, tok in enumerate(input_tokens):
            clean_tok = tok.replace("▁", "").lower()
            if shape_name in clean_tok or clean_tok in shape_name:
                positions.append(i)
    
    # Remove duplicates and sort
    positions = sorted(list(set(positions)))
    
    return positions

def find_relation_token(input_tokens, relation):
    """
    Find token position for relation word.
    
    Args:
        input_tokens: List of tokenized tokens
        relation: Relation word ("left", "right", "above", "below", or "adjacent")
    
    Returns:
        List of token indices (usually single element)
    """
    # Try exact match first
    positions = find_token_positions(input_tokens, [f"▁{relation}", relation])
    if len(positions) == 0:
        # Try without underscore prefix
        for i, tok in enumerate(input_tokens):
            if relation in tok.replace("▁", "").lower():
                positions.append(i)
                break
    return positions

# =====================
# Analysis functions
# =====================

def analyze_attention_for_caption(image_path, caption, world_data, sample_idx, model, image_processor, tokenizer, device, grid_size=24):
    """
    Extract and analyze attention maps for a given caption.
    
    Args:
        image_path: Path to image file
        caption: Caption text
        world_data: World model JSON data for this sample
        sample_idx: Sample index
        grid_size: Grid size for attention maps (default 24)
    
    Returns:
        Dictionary containing all attention metrics
    """
    # Extract cross-attention and vision self-attention
    # This is during forward pass (not generation stage)
    cross_attn, input_tokens, image, vis_attn_matrix = extract_prompt_level_cross_attention(
        image_path, caption, model, image_processor, tokenizer, device
    )
    
    # Replace NaN values
    cross_attn = torch.nan_to_num(cross_attn, nan=0.0)
    
    # Get entity masks from world data
    entity1 = world_data["entities"][0]
    entity1_name = entity1["shape"]["name"]
    entity2 = world_data["entities"][1]
    entity2_name = entity2["shape"]["name"]
    mask_entity1 = gt_bbox_to_patch_mask(entity1, grid_size=grid_size)
    mask_entity2 = gt_bbox_to_patch_mask(entity2, grid_size=grid_size)
    relation = caption.strip().lower().split()[-1]  # last word
    entity1_pos = find_word_token_positions(tokenizer, input_tokens, entity1_name)
    entity2_pos = find_word_token_positions(tokenizer, input_tokens, entity2_name)
    relation_pos = find_word_token_positions(tokenizer, input_tokens, relation)
    
    if len(entity1_pos) == 0 or len(entity2_pos) == 0 or len(relation_pos) == 0:
        print(f"Warning: Could not find all tokens for sample {sample_idx}")
        return None
    
    results = {
        "sample_idx": sample_idx,
        "caption": caption,
        "relation": relation,
    }
    
    # ===========================================================
    # 1. Relation token cross-attention analysis
    # ===========================================================
    # Cross-attention map: vision tokens (queries) attending to relation token (key)
    # Shape: [num_vision_patches], extracted during forward pass
    # This represents how vision patches attend to the spatial relation word
    relation_attn_flat = pool_attention(cross_attn, relation_pos, mode="mean")
    relation_attn_map = relation_attn_flat.reshape(grid_size, grid_size)
    
    results["relation_token_entropy"] = compute_entropy(relation_attn_map)
    results["relation_token_dispersion"] = compute_disperson(relation_attn_map)
    
    # ===========================================================
    # 2. Entity1 token cross-attention analysis
    # ===========================================================
    # Cross-attention map: vision tokens (queries) attending to entity1 tokens (keys)
    # Shape: [num_vision_patches], extracted during forward pass
    # This represents how vision patches attend to the first entity
    entity1_attn_flat = pool_attention(cross_attn, entity1_pos, mode="mean")
    entity1_attn_map = entity1_attn_flat.reshape(grid_size, grid_size)
    
    results["entity1_token_entropy"] = compute_entropy(entity1_attn_map)
    results["entity1_token_dispersion"] = compute_disperson(entity1_attn_map)
    
    # ===========================================================
    # 3. Entity2 token cross-attention analysis
    # ===========================================================
    # Cross-attention map: vision tokens (queries) attending to entity2 tokens (keys)
    # Shape: [num_vision_patches], extracted during forward pass
    # This represents how vision patches attend to the second entity
    entity2_attn_flat = pool_attention(cross_attn, entity2_pos, mode="mean")
    entity2_attn_map = entity2_attn_flat.reshape(grid_size, grid_size)
    
    results["entity2_token_entropy"] = compute_entropy(entity2_attn_map)
    results["entity2_token_dispersion"] = compute_disperson(entity2_attn_map)
    
    # ===========================================================
    # 4. Vision self-attention analysis
    # ===========================================================
    # Self-attention map: vision token (query) attending to vision tokens (keys)
    # Shape: [grid_size, grid_size], extracted during forward pass
    # This represents how a vision patch attends to other vision patches
    # We use entity center seeds as starting points
    seed1 = get_entity_center_seed(mask_entity1)
    seed2 = get_entity_center_seed(mask_entity2)
    
    # Entity1 self-attention: from entity1 center to all vision patches
    entity1_self_map = get_patch_self_attention_map(vis_attn_matrix, seed1, grid_size)
    results["entity1_self_attn_entropy"] = compute_entropy(entity1_self_map)
    results["entity1_self_attn_dispersion"] = compute_disperson(entity1_self_map)
    
    # Entity2 self-attention: from entity2 center to all vision patches
    entity2_self_map = get_patch_self_attention_map(vis_attn_matrix, seed2, grid_size)
    results["entity2_self_attn_entropy"] = compute_entropy(entity2_self_map)
    results["entity2_self_attn_dispersion"] = compute_disperson(entity2_self_map)
    
    return results

def batch_analyze(data_dir, max_samples=None):
    """
    Batch process all samples in the dataset.
    
    Args:
        data_dir: Base directory containing shard directories
        max_samples: Maximum number of samples to process (None for all)
    
    Returns:
        DataFrame with all results
    """
    # Load data from all shards
    print(f"Loading data from {data_dir}...")
    
    # Load data from each shard separately to get correct world_data
    complex_samples = []
    shard_count = 5
    
    for shard_num in range(shard_count):
        shard_dir = os.path.join(data_dir, f"shard{shard_num}")
        if not os.path.exists(shard_dir):
            continue
        
        # Load data from this shard
        from dataloader import load_data_from_shard
        shard_image_paths, shard_captions = load_data_from_shard(shard_dir)
        
        # Load world_model.json for this shard
        json_path = os.path.join(shard_dir, "world_model.json")
        if not os.path.exists(json_path):
            print(f"Warning: world_model.json not found in {shard_dir}")
            continue
        
        with open(json_path, "r") as f:
            world_data_all = json.load(f)
        
        # world_data_all is a list, convert to dict if needed
        if isinstance(world_data_all, list):
            world_data_dict = {i: world_data_all[i] for i in range(len(world_data_all))}
        else:
            world_data_dict = world_data_all
        
        # Filter for complex relations in this shard
        for i, (img_path, caption) in enumerate(zip(shard_image_paths, shard_captions)):
            if any(rel in caption.lower() for rel in COMPLEX_RELATIONS):
                # Extract sample index from image path (e.g., "world-123.png" -> 123)
                try:
                    filename = os.path.basename(img_path)
                    if "world-" in filename:
                        sample_idx = int(filename.replace("world-", "").replace(".png", ""))
                    else:
                        sample_idx = i
                except:
                    sample_idx = i
                
                # Store shard info and world_data_dict for this sample
                complex_samples.append((img_path, caption, sample_idx, world_data_dict))
    
    print(f"Found {len(complex_samples)} samples with complex relations")
    
    if max_samples:
        complex_samples = complex_samples[:max_samples]
        print(f"Processing first {max_samples} samples")
    
    results_original = []
    results_modified = []
    
    # Process each sample
    for img_path, original_caption, sample_idx, world_data_dict in tqdm(complex_samples, desc="Processing samples"):
        # Get modified caption
        modified_caption = replace_complex_relation_with_adjacent(original_caption)
        if modified_caption is None:
            continue
        
        # Get world data for this sample
        if sample_idx not in world_data_dict:
            print(f"Warning: Sample {sample_idx} not found in world_data")
            continue
        
        world_data = world_data_dict[sample_idx]
        
        
        # Analyze with original caption
        result_orig = analyze_attention_for_caption(
            img_path, original_caption, world_data, sample_idx, model, image_processor, tokenizer, device
        )
        if result_orig:
            result_orig["caption_type"] = "complex"
            results_original.append(result_orig)
        
        # Analyze with modified caption
        result_mod = analyze_attention_for_caption(
            img_path, modified_caption, world_data, sample_idx, model, image_processor, tokenizer, device
        )
        if result_mod:
            result_mod["caption_type"] = "adjacent"
            result_mod["original_caption"] = original_caption
            results_modified.append(result_mod)
        
        # except Exception as e:
        #     print(f"Error processing sample {sample_idx}: {e}")
        #     continue
    
    # Combine results
    all_results = results_original + results_modified
    df = pd.DataFrame(all_results)
    
    return df

# =====================
# Statistical comparison
# =====================

def compute_comparison_stats(df):
    """
    Compute statistical comparison between complex and adjacent groups.
    
    Args:
        df: DataFrame with results
    
    Returns:
        Dictionary with statistics
    """
    metrics = [
        "relation_token_entropy",
        "relation_token_dispersion",
        "entity1_token_entropy",
        "entity1_token_dispersion",
        "entity2_token_entropy",
        "entity2_token_dispersion",
        "entity1_self_attn_entropy",
        "entity1_self_attn_dispersion",
        "entity2_self_attn_entropy",
        "entity2_self_attn_dispersion",
    ]
    
    stats_dict = {}
    
    complex_df = df[df["caption_type"] == "complex"]
    adjacent_df = df[df["caption_type"] == "adjacent"]
    
    for metric in metrics:
        if metric not in df.columns:
            continue
        
        complex_vals = complex_df[metric].dropna()
        adjacent_vals = adjacent_df[metric].dropna()
        
        if len(complex_vals) == 0 or len(adjacent_vals) == 0:
            continue
        
        # Compute means and stds
        stats_dict[f"{metric}_complex_mean"] = complex_vals.mean()
        stats_dict[f"{metric}_complex_std"] = complex_vals.std()
        stats_dict[f"{metric}_adjacent_mean"] = adjacent_vals.mean()
        stats_dict[f"{metric}_adjacent_std"] = adjacent_vals.std()
        
        # T-test for significance
        t_stat, p_value = stats.ttest_ind(complex_vals, adjacent_vals)
        stats_dict[f"{metric}_t_stat"] = t_stat
        stats_dict[f"{metric}_p_value"] = p_value
    
    return stats_dict

# =====================
# Visualization
# =====================

def visualize_comparison(df, output_dir="results"):
    """
    Create visualizations comparing complex vs adjacent relations.
    
    Args:
        df: DataFrame with results
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = [
        ("relation_token_entropy", "Relation Token Entropy"),
        ("relation_token_dispersion", "Relation Token Dispersion"),
        ("entity1_token_entropy", "Entity1 Token Entropy"),
        ("entity1_token_dispersion", "Entity1 Token Dispersion"),
        ("entity2_token_entropy", "Entity2 Token Entropy"),
        ("entity2_token_dispersion", "Entity2 Token Dispersion"),
        # ("entity1_self_attn_entropy", "Entity1 Self-Attention Entropy"),
        # ("entity1_self_attn_dispersion", "Entity1 Self-Attention Dispersion"),
        # ("entity2_self_attn_entropy", "Entity2 Self-Attention Entropy"),
        # ("entity2_self_attn_dispersion", "Entity2 Self-Attention Dispersion"),
    ]
    
    # Filter available metrics
    available_metrics = [(m, label) for m, label in metrics if m in df.columns]
    
    # 1. Bar plots comparing means
    fig, axes = plt.subplots(2, 3, figsize=(20, 8))
    axes = axes.flatten()
    
    for idx, (metric, label) in enumerate(available_metrics[:6]):
        ax = axes[idx]
        complex_vals = df[df["caption_type"] == "complex"][metric].dropna()
        adjacent_vals = df[df["caption_type"] == "adjacent"][metric].dropna()
        
        means = [complex_vals.mean(), adjacent_vals.mean()]
        stds = [complex_vals.std(), adjacent_vals.std()]
        
        x = np.arange(2)
        ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color=["#1f77b4", "#ff7f0e"])
        ax.set_xticks(x)
        ax.set_xticklabels(["Complex", "Adjacent"])
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_barplots.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # 2. Box plots
    fig, axes = plt.subplots(2, 3, figsize=(20, 8))
    axes = axes.flatten()
    
    for idx, (metric, label) in enumerate(available_metrics[:6]):
        ax = axes[idx]
        complex_vals = df[df["caption_type"] == "complex"][metric].dropna()
        adjacent_vals = df[df["caption_type"] == "adjacent"][metric].dropna()
        
        data_to_plot = [complex_vals.values, adjacent_vals.values]
        bp = ax.boxplot(data_to_plot, labels=["Complex", "Adjacent"], patch_artist=True)
        bp["boxes"][0].set_facecolor("#1f77b4")
        bp["boxes"][1].set_facecolor("#ff7f0e")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_boxplots.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # 3. Scatter plots: entropy vs dispersion
    fig, axes = plt.subplots(1, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    scatter_configs = [
        ("relation_token_entropy", "relation_token_dispersion", "Relation Token"),
        ("entity1_token_entropy", "entity1_token_dispersion", "Entity1 Token"),
        ("entity2_token_entropy", "entity2_token_dispersion", "Entity2 Token"),
        # ("entity1_self_attn_entropy", "entity1_self_attn_dispersion", "Entity1 Self-Attn"),
        # ("entity2_self_attn_entropy", "entity2_self_attn_dispersion", "Entity2 Self-Attn"),
    ]
    
    for idx, (entropy_col, dispersion_col, title) in enumerate(scatter_configs):
        if entropy_col not in df.columns or dispersion_col not in df.columns:
            continue
        
        ax = axes[idx]
        
        complex_df_sub = df[df["caption_type"] == "complex"]
        adjacent_df_sub = df[df["caption_type"] == "adjacent"]
        ax.scatter(
            complex_df_sub[entropy_col],
            complex_df_sub[dispersion_col],
            alpha=0.6,
            label="Complex",
            color="#1f77b4",
        )
        ax.scatter(
            adjacent_df_sub[entropy_col],
            adjacent_df_sub[dispersion_col],
            alpha=0.6,
            label="Adjacent",
            color="#ff7f0e",
        )
        
        ax.set_xlabel("Entropy")
        ax.set_ylabel("Dispersion")
        ax.set_title(f"{title}: Entropy vs Dispersion")
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "entropy_vs_dispersion_scatter.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Visualizations saved to {output_dir}/")

# =====================
# Main execution
# =====================

if __name__ == "__main__":
    # Configuration
    data_dir = "data/spatial_twoshapes/agreement/relational"
    output_dir = "results"
    max_samples = None  # Set to a number to limit samples for testing
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Run batch analysis
    print("Starting batch analysis...")
    df = batch_analyze(data_dir, max_samples=max_samples)
    
    if df is None or len(df) == 0:
        print("No results to analyze.")
        sys.exit(1)
    
    # Save results to CSV
    csv_path = os.path.join(output_dir, "attention_analysis_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Compute statistics
    print("\nComputing comparison statistics...")
    stats_dict = compute_comparison_stats(df)
    
    # Save statistics
    stats_df = pd.DataFrame([stats_dict])
    stats_path = os.path.join(output_dir, "comparison_statistics.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"Statistics saved to {stats_path}")
    
    # Print summary
    print("\n=== Summary Statistics ===")
    for key, value in sorted(stats_dict.items()):
        if "p_value" in key:
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value:.4f}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    visualize_comparison(df, output_dir)
    
    print("\nAnalysis complete!")

