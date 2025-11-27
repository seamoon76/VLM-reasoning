import os
import sys
sys.path.append("./models")
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

# Load the saved attention data
print("Loading attention data...")
data_with = torch.load('attention_data/attention_with_triangle.pt')
data_without = torch.load('attention_data/attention_without_triangle.pt')

print(f"\nWith triangle: {data_with['response_text']}")
print(f"Without triangle: {data_without['response_text']}")

# Extract key information
grid_size = data_with['grid_size']
vision_token_start = data_with['vision_token_start']
vision_token_end = data_with['vision_token_end']

print(f"\nGrid size: {grid_size} x {grid_size}")
print(f"Number of vision tokens: {vision_token_end - vision_token_start}")

# ==================== ANALYSIS 1: Per-Head Attention Comparison ====================
print("\n" + "="*80)
print("ANALYSIS 1: Per-Head Attention Comparison")
print("="*80)

def get_top_left_quadrant_indices(grid_size):
    """Get vision token indices corresponding to top-left quadrant"""
    indices = []
    mid_row = grid_size // 2
    mid_col = grid_size // 2

    for i in range(mid_row):
        for j in range(mid_col):
            idx = i * grid_size + j
            indices.append(idx)
    return indices

top_left_indices = get_top_left_quadrant_indices(grid_size)
print(f"Top-left quadrant has {len(top_left_indices)} vision tokens")

def analyze_per_head_attention(data, top_left_indices, vision_token_start, vision_token_end, label):
    """
    Analyze attention per head for the first token generation step
    Returns: list of (layer_idx, head_idx, top_left_attention_score)
    """
    # Get the first generation step's attention (for the first output token)
    first_step_attns = data['llm_attentions'][0]  # List of layer attentions

    num_layers = len(first_step_attns)
    results = []

    for layer_idx, layer_attn in enumerate(first_step_attns):
        # layer_attn shape: [batch=1, num_heads, seq_len, seq_len]
        layer_attn = layer_attn.squeeze(0)  # [num_heads, seq_len, seq_len]
        num_heads = layer_attn.shape[0]

        for head_idx in range(num_heads):
            # Get attention from the last position (the position being generated)
            # to all vision tokens
            head_attn = layer_attn[head_idx, -1, vision_token_start:vision_token_end]  # [num_vision_tokens]

            # Normalize to make it a probability distribution
            head_attn = head_attn / (head_attn.sum() + 1e-10)

            # Calculate attention to top-left quadrant
            top_left_attn = head_attn[top_left_indices].sum().item()

            results.append({
                'layer': layer_idx,
                'head': head_idx,
                'top_left_attn': top_left_attn,
                'full_attn': head_attn.cpu()
            })

    return results

print("\nAnalyzing per-head attention for WITH triangle...")
results_with = analyze_per_head_attention(data_with, top_left_indices, vision_token_start, vision_token_end, "with")

print("Analyzing per-head attention for WITHOUT triangle...")
results_without = analyze_per_head_attention(data_without, top_left_indices, vision_token_start, vision_token_end, "without")

# Calculate differences
print("\nCalculating differences between the two conditions...")
head_differences = []
for res_with, res_without in zip(results_with, results_without):
    diff = res_with['top_left_attn'] - res_without['top_left_attn']
    head_differences.append({
        'layer': res_with['layer'],
        'head': res_with['head'],
        'diff': diff,
        'with_attn': res_with['top_left_attn'],
        'without_attn': res_without['top_left_attn'],
        'attn_with': res_with['full_attn'],
        'attn_without': res_without['full_attn']
    })

# Sort by absolute difference
head_differences_sorted = sorted(head_differences, key=lambda x: abs(x['diff']), reverse=True)

# Print top 20 heads with largest differences
print("\n" + "-"*80)
print("Top 20 attention heads with largest differences in top-left attention:")
print("-"*80)
print(f"{'Rank':<6} {'Layer':<8} {'Head':<8} {'Diff':<12} {'With':<12} {'Without':<12}")
print("-"*80)

for rank, hd in enumerate(head_differences_sorted[:20], 1):
    print(f"{rank:<6} {hd['layer']:<8} {hd['head']:<8} {hd['diff']:>11.4f} {hd['with_attn']:>11.4f} {hd['without_attn']:>11.4f}")

# Save results
print("\nSaving per-head analysis results...")
torch.save({
    'head_differences': head_differences_sorted,
    'top_left_indices': top_left_indices,
}, 'attention_data/per_head_analysis.pt')

# Visualize top different heads
print("\nGenerating visualization of top different heads...")
fig, axes = plt.subplots(5, 4, figsize=(16, 20))
axes = axes.flatten()

for idx in range(20):
    hd = head_differences_sorted[idx]
    ax = axes[idx]

    # Reshape attention to grid
    attn_with = hd['attn_with'].reshape(grid_size, grid_size).numpy()
    attn_without = hd['attn_without'].reshape(grid_size, grid_size).numpy()

    # Create difference map
    diff_map = attn_with - attn_without

    im = ax.imshow(diff_map, cmap='RdBu_r', vmin=-diff_map.max(), vmax=diff_map.max())
    ax.set_title(f"L{hd['layer']} H{hd['head']}\nΔ={hd['diff']:.4f}", fontsize=9)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.savefig('attention_data/top_different_heads.png', dpi=150, bbox_inches='tight')
print("Saved to attention_data/top_different_heads.png")

print("\n" + "="*80)
print("Per-Head Attention Analysis Complete!")
print("="*80)
