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

grid_size = data_with['grid_size']
vision_token_start = data_with['vision_token_start']
vision_token_end = data_with['vision_token_end']

# ==================== ANALYSIS 2: Spatial Attention Distribution ====================
print("\n" + "="*80)
print("ANALYSIS 2: Spatial Attention Distribution (4 Quadrants)")
print("="*80)

def get_quadrant_indices(grid_size):
    """Get vision token indices for all 4 quadrants"""
    mid_row = grid_size // 2
    mid_col = grid_size // 2

    quadrants = {
        'top_left': [],
        'top_right': [],
        'bottom_left': [],
        'bottom_right': []
    }

    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if i < mid_row and j < mid_col:
                quadrants['top_left'].append(idx)
            elif i < mid_row and j >= mid_col:
                quadrants['top_right'].append(idx)
            elif i >= mid_row and j < mid_col:
                quadrants['bottom_left'].append(idx)
            else:
                quadrants['bottom_right'].append(idx)

    return quadrants

quadrants = get_quadrant_indices(grid_size)
print(f"Quadrant sizes: {[(k, len(v)) for k, v in quadrants.items()]}")

def analyze_spatial_distribution(data, quadrants, vision_token_start, vision_token_end, label):
    """Analyze attention distribution across 4 quadrants"""
    # Use the aggregated LLM attention matrix
    llm_attn = data['llm_attn_matrix']

    # Get attention for output tokens to vision tokens
    output_start = data['output_token_start']
    output_end = data['output_token_end']

    # Average attention across all output tokens
    avg_attn = llm_attn[output_start:output_end, vision_token_start:vision_token_end].mean(dim=0)
    avg_attn = avg_attn / (avg_attn.sum() + 1e-10)

    # Calculate attention per quadrant
    quad_attn = {}
    for quad_name, indices in quadrants.items():
        quad_attn[quad_name] = avg_attn[indices].sum().item()

    return quad_attn, avg_attn

print("\nAnalyzing spatial distribution...")
quad_with, attn_with = analyze_spatial_distribution(data_with, quadrants, vision_token_start, vision_token_end, "with")
quad_without, attn_without = analyze_spatial_distribution(data_without, quadrants, vision_token_start, vision_token_end, "without")

print("\n" + "-"*80)
print("Spatial Attention Distribution:")
print("-"*80)
print(f"{'Quadrant':<20} {'With Triangle':<20} {'Without Triangle':<20} {'Difference':<15}")
print("-"*80)

for quad_name in ['top_left', 'top_right', 'bottom_left', 'bottom_right']:
    diff = quad_with[quad_name] - quad_without[quad_name]
    print(f"{quad_name:<20} {quad_with[quad_name]:>19.4f} {quad_without[quad_name]:>19.4f} {diff:>14.4f}")

# Visualize spatial distribution
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# With triangle
ax = axes[0]
attn_map = attn_with.reshape(grid_size, grid_size).numpy()
im = ax.imshow(attn_map, cmap='hot')
ax.set_title('Spatial Attention\n(With Triangle)', fontsize=14)
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046)

# Without triangle
ax = axes[1]
attn_map = attn_without.reshape(grid_size, grid_size).numpy()
im = ax.imshow(attn_map, cmap='hot')
ax.set_title('Spatial Attention\n(Without Triangle)', fontsize=14)
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046)

# Difference
ax = axes[2]
diff_map = (attn_with - attn_without).reshape(grid_size, grid_size).numpy()
im = ax.imshow(diff_map, cmap='RdBu_r', vmin=-diff_map.max(), vmax=diff_map.max())
ax.set_title('Attention Difference\n(With - Without)', fontsize=14)
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.savefig('attention_data/spatial_distribution.png', dpi=150, bbox_inches='tight')
print("\nSaved spatial distribution visualization to attention_data/spatial_distribution.png")

# ==================== ANALYSIS 3: Vision Encoder Layer-wise Analysis ====================
print("\n" + "="*80)
print("ANALYSIS 3: Vision Encoder Layer-wise Analysis")
print("="*80)

def analyze_vision_encoder_layers(data, quadrants, label):
    """Analyze each layer of vision encoder's attention to quadrants"""
    vision_attns = data['vision_attentions']
    num_layers = len(vision_attns)

    layer_results = []

    for layer_idx, layer_attn in enumerate(vision_attns):
        # layer_attn shape: [batch=1, num_heads, num_patches+1, num_patches+1]
        # Remove CLS token (first token)
        layer_attn = layer_attn.squeeze(0)  # [num_heads, num_patches+1, num_patches+1]
        layer_attn = layer_attn.mean(dim=0)  # Average across heads [num_patches+1, num_patches+1]
        layer_attn = layer_attn[1:, 1:]  # Remove CLS token [num_patches, num_patches]

        # Average attention from all patches
        avg_attn = layer_attn.mean(dim=0)  # [num_patches]
        avg_attn = avg_attn / (avg_attn.sum() + 1e-10)

        # Calculate attention per quadrant
        quad_attn = {}
        for quad_name, indices in quadrants.items():
            quad_attn[quad_name] = avg_attn[indices].sum().item()

        layer_results.append({
            'layer': layer_idx,
            'quadrants': quad_attn,
            'full_attn': avg_attn.cpu()
        })

    return layer_results

print("\nAnalyzing vision encoder layers...")
vision_layers_with = analyze_vision_encoder_layers(data_with, quadrants, "with")
vision_layers_without = analyze_vision_encoder_layers(data_without, quadrants, "without")

print("\n" + "-"*80)
print("Vision Encoder Attention to Top-Left Quadrant by Layer:")
print("-"*80)
print(f"{'Layer':<8} {'With Triangle':<20} {'Without Triangle':<20} {'Difference':<15}")
print("-"*80)

for res_with, res_without in zip(vision_layers_with, vision_layers_without):
    layer = res_with['layer']
    with_tl = res_with['quadrants']['top_left']
    without_tl = res_without['quadrants']['top_left']
    diff = with_tl - without_tl
    print(f"{layer:<8} {with_tl:>19.4f} {without_tl:>19.4f} {diff:>14.4f}")

# Visualize layer-wise trend
fig, ax = plt.subplots(figsize=(12, 6))

layers = [res['layer'] for res in vision_layers_with]
with_tl = [res['quadrants']['top_left'] for res in vision_layers_with]
without_tl = [res['quadrants']['top_left'] for res in vision_layers_without]

ax.plot(layers, with_tl, marker='o', label='With Triangle', linewidth=2)
ax.plot(layers, without_tl, marker='s', label='Without Triangle', linewidth=2)
ax.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='Random (25%)')
ax.set_xlabel('Vision Encoder Layer', fontsize=12)
ax.set_ylabel('Attention to Top-Left Quadrant', fontsize=12)
ax.set_title('Vision Encoder Layer-wise Attention to Top-Left Quadrant', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('attention_data/vision_encoder_layers.png', dpi=150, bbox_inches='tight')
print("Saved vision encoder layer analysis to attention_data/vision_encoder_layers.png")

# ==================== ANALYSIS 4: Attention Divergence ====================
print("\n" + "="*80)
print("ANALYSIS 4: Attention Divergence (KL & JS Divergence)")
print("="*80)

def compute_divergences(attn1, attn2):
    """Compute KL and JS divergence between two attention distributions"""
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    attn1 = attn1.float().numpy() + eps
    attn2 = attn2.float().numpy() + eps

    # Normalize
    attn1 = attn1 / attn1.sum()
    attn2 = attn2 / attn2.sum()

    # KL divergence (asymmetric)
    kl_div = entropy(attn1, attn2)

    # JS divergence (symmetric)
    js_div = jensenshannon(attn1, attn2)

    return kl_div, js_div

# Overall spatial attention divergence
kl, js = compute_divergences(attn_with, attn_without)
print(f"\nOverall Spatial Attention:")
print(f"  KL Divergence: {kl:.6f}")
print(f"  JS Divergence: {js:.6f}")

# Per-layer vision encoder divergence
print("\nVision Encoder Layer-wise Divergence:")
print("-"*60)
print(f"{'Layer':<8} {'KL Divergence':<20} {'JS Divergence':<20}")
print("-"*60)

layer_divergences = []
for res_with, res_without in zip(vision_layers_with, vision_layers_without):
    kl, js = compute_divergences(res_with['full_attn'], res_without['full_attn'])
    layer_divergences.append({'layer': res_with['layer'], 'kl': kl, 'js': js})
    print(f"{res_with['layer']:<8} {kl:>19.6f} {js:>19.6f}")

# Visualize divergence across layers
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

layers = [ld['layer'] for ld in layer_divergences]
kl_divs = [ld['kl'] for ld in layer_divergences]
js_divs = [ld['js'] for ld in layer_divergences]

axes[0].plot(layers, kl_divs, marker='o', linewidth=2, color='steelblue')
axes[0].set_xlabel('Vision Encoder Layer')
axes[0].set_ylabel('KL Divergence')
axes[0].set_title('KL Divergence by Layer')
axes[0].grid(True, alpha=0.3)

axes[1].plot(layers, js_divs, marker='s', linewidth=2, color='coral')
axes[1].set_xlabel('Vision Encoder Layer')
axes[1].set_ylabel('JS Divergence')
axes[1].set_title('JS Divergence by Layer')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('attention_data/divergence_analysis.png', dpi=150, bbox_inches='tight')
print("\nSaved divergence analysis to attention_data/divergence_analysis.png")

# Save all results
torch.save({
    'spatial_distribution': {
        'with': quad_with,
        'without': quad_without,
    },
    'vision_encoder_layers': {
        'with': vision_layers_with,
        'without': vision_layers_without,
    },
    'divergences': {
        'overall': {'kl': kl, 'js': js},
        'per_layer': layer_divergences,
    }
}, 'attention_data/spatial_vision_analysis.pt')

print("\n" + "="*80)
print("Spatial and Vision Encoder Analysis Complete!")
print("="*80)
