import os
import sys
sys.path.append("./models")
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy.ndimage import label as scipy_label
from scipy.ndimage import maximum_filter
from sklearn.feature_selection import mutual_info_regression

# Load the saved attention data
print("Loading attention data...")
data_with = torch.load('attention_data/attention_with_triangle.pt')
data_without = torch.load('attention_data/attention_without_triangle.pt')

grid_size = data_with['grid_size']
vision_token_start = data_with['vision_token_start']
vision_token_end = data_with['vision_token_end']

from llava.mm_utils import get_model_name_from_path
from transformers import AutoTokenizer

# Load tokenizer to decode tokens
model_path = "liuhaotian/llava-v1.5-7b"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

# ==================== METRIC 5: Attention Dynamics ====================
print("\n" + "="*80)
print("METRIC 5: Attention Dynamics (How attention changes during generation)")
print("="*80)

def analyze_attention_dynamics(data, vision_token_start, vision_token_end, tokenizer, label):
    """Analyze how attention changes as tokens are generated"""
    llm_attn_matrix = data['llm_attn_matrix']
    sequences = data['sequences'][0]
    output_token_start = data['output_token_start']
    output_token_end = data['output_token_end']

    # Get attention for each output token
    dynamics = []

    for token_idx in range(output_token_start, output_token_end):
        attn_to_vision = llm_attn_matrix[token_idx, vision_token_start:vision_token_end]
        attn_to_vision = attn_to_vision / (attn_to_vision.sum() + 1e-10)

        token_id = sequences[token_idx - output_token_start].item()
        token_str = tokenizer.decode([token_id], skip_special_tokens=False).strip()

        # Compute metrics for this token's attention
        entropy_val = -torch.sum(attn_to_vision * torch.log(attn_to_vision + 1e-10)).item()

        # Top-left quadrant attention
        top_left_indices = []
        mid_row = grid_size // 2
        mid_col = grid_size // 2
        for i in range(mid_row):
            for j in range(mid_col):
                top_left_indices.append(i * grid_size + j)

        tl_attn = attn_to_vision[top_left_indices].sum().item()

        # Peak attention value
        peak_attn = attn_to_vision.max().item()

        dynamics.append({
            'token_idx': token_idx - output_token_start,
            'token': token_str,
            'entropy': entropy_val,
            'top_left_attn': tl_attn,
            'peak_attn': peak_attn,
            'attn_dist': attn_to_vision.cpu()
        })

    return dynamics

print("\nAnalyzing attention dynamics for WITH triangle...")
dynamics_with = analyze_attention_dynamics(data_with, vision_token_start, vision_token_end, tokenizer, "with")

print("Analyzing attention dynamics for WITHOUT triangle...")
dynamics_without = analyze_attention_dynamics(data_without, vision_token_start, vision_token_end, tokenizer, "without")

# Display token-by-token dynamics
print("\n" + "-"*80)
print("Token-by-Token Attention Dynamics (WITH triangle):")
print("-"*80)
print(f"{'Idx':<5} {'Token':<15} {'Entropy':<12} {'TL Attn':<12} {'Peak':<12}")
print("-"*80)
for d in dynamics_with[:20]:  # Show first 20 tokens
    print(f"{d['token_idx']:<5} {d['token']:<15} {d['entropy']:>11.4f} {d['top_left_attn']:>11.4f} {d['peak_attn']:>11.4f}")

print("\n" + "-"*80)
print("Token-by-Token Attention Dynamics (WITHOUT triangle):")
print("-"*80)
print(f"{'Idx':<5} {'Token':<15} {'Entropy':<12} {'TL Attn':<12} {'Peak':<12}")
print("-"*80)
for d in dynamics_without[:20]:
    print(f"{d['token_idx']:<5} {d['token']:<15} {d['entropy']:>11.4f} {d['top_left_attn']:>11.4f} {d['peak_attn']:>11.4f}")

# Visualize dynamics
fig, axes = plt.subplots(3, 1, figsize=(16, 10))

# Entropy over time
ax = axes[0]
with_entropy = [d['entropy'] for d in dynamics_with]
without_entropy = [d['entropy'] for d in dynamics_without]
with_tokens = [d['token'] for d in dynamics_with]
without_tokens = [d['token'] for d in dynamics_without]

ax.plot(with_entropy, marker='o', label='With Triangle', linewidth=2)
ax.plot(without_entropy, marker='s', label='Without Triangle', linewidth=2)
ax.set_ylabel('Entropy', fontsize=11)
ax.set_title('Attention Entropy Dynamics', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# Top-left attention over time
ax = axes[1]
with_tl = [d['top_left_attn'] for d in dynamics_with]
without_tl = [d['top_left_attn'] for d in dynamics_without]

ax.plot(with_tl, marker='o', label='With Triangle', linewidth=2)
ax.plot(without_tl, marker='s', label='Without Triangle', linewidth=2)
ax.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='Random (25%)')
ax.set_ylabel('Top-Left Attention', fontsize=11)
ax.set_title('Top-Left Quadrant Attention Dynamics', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# Peak attention over time
ax = axes[2]
with_peak = [d['peak_attn'] for d in dynamics_with]
without_peak = [d['peak_attn'] for d in dynamics_without]

ax.plot(with_peak, marker='o', label='With Triangle', linewidth=2)
ax.plot(without_peak, marker='s', label='Without Triangle', linewidth=2)
ax.set_xlabel('Token Index', fontsize=11)
ax.set_ylabel('Peak Attention', fontsize=11)
ax.set_title('Peak Attention Value Dynamics', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('attention_data/attention_dynamics.png', dpi=150, bbox_inches='tight')
print("\nSaved attention dynamics visualization to attention_data/attention_dynamics.png")

# ==================== METRIC 6: Hotspot Detection ====================
print("\n" + "="*80)
print("METRIC 6: Hotspot Detection (Finding attention peaks)")
print("="*80)

def detect_hotspots(attn_dist, grid_size, threshold_percentile=90):
    """Detect attention hotspots using local maxima"""
    attn_map = attn_dist.reshape(grid_size, grid_size).float().cpu().numpy()

    # Find local maxima
    local_max = maximum_filter(attn_map, size=3)
    hotspots = (attn_map == local_max)

    # Apply threshold: only consider strong attention values
    threshold = np.percentile(attn_map, threshold_percentile)
    hotspots = hotspots & (attn_map >= threshold)

    # Label connected components
    labeled_array, num_hotspots = scipy_label(hotspots)

    # Get hotspot locations and strengths
    hotspot_info = []
    for hotspot_id in range(1, num_hotspots + 1):
        mask = labeled_array == hotspot_id
        y_coords, x_coords = np.where(mask)

        # Center of this hotspot
        y_center = y_coords.mean() / grid_size
        x_center = x_coords.mean() / grid_size

        # Strength (sum of attention in this hotspot)
        strength = attn_map[mask].sum()

        hotspot_info.append({
            'id': hotspot_id,
            'x_center': x_center,
            'y_center': y_center,
            'strength': strength,
            'size': mask.sum()
        })

    # Sort by strength
    hotspot_info = sorted(hotspot_info, key=lambda x: x['strength'], reverse=True)

    return hotspot_info, hotspots, attn_map

# Get aggregated attention
llm_attn_with = data_with['llm_attn_matrix']
llm_attn_without = data_without['llm_attn_matrix']

output_start_with = data_with['output_token_start']
output_end_with = data_with['output_token_end']
output_start_without = data_without['output_token_start']
output_end_without = data_without['output_token_end']

avg_attn_with = llm_attn_with[output_start_with:output_end_with, vision_token_start:vision_token_end].mean(dim=0)
avg_attn_without = llm_attn_without[output_start_without:output_end_without, vision_token_start:vision_token_end].mean(dim=0)

print("\nDetecting hotspots...")
hotspots_with, hotspot_mask_with, attn_map_with = detect_hotspots(avg_attn_with, grid_size)
hotspots_without, hotspot_mask_without, attn_map_without = detect_hotspots(avg_attn_without, grid_size)

print("\n" + "-"*80)
print(f"Detected {len(hotspots_with)} hotspots in WITH triangle condition")
print(f"Detected {len(hotspots_without)} hotspots in WITHOUT triangle condition")
print("-"*80)

print("\nTop 5 Hotspots (WITH triangle):")
print(f"{'ID':<5} {'Position (x,y)':<20} {'Quadrant':<15} {'Strength':<12} {'Size':<8}")
print("-"*80)

def get_quadrant(x, y):
    if x < 0.5 and y < 0.5:
        return "Top-Left"
    elif x >= 0.5 and y < 0.5:
        return "Top-Right"
    elif x < 0.5 and y >= 0.5:
        return "Bottom-Left"
    else:
        return "Bottom-Right"

for h in hotspots_with[:5]:
    quadrant = get_quadrant(h['x_center'], h['y_center'])
    print(f"{h['id']:<5} ({h['x_center']:.3f}, {h['y_center']:.3f}){'':<6} {quadrant:<15} {h['strength']:>11.4f} {h['size']:>7}")

print("\nTop 5 Hotspots (WITHOUT triangle):")
print(f"{'ID':<5} {'Position (x,y)':<20} {'Quadrant':<15} {'Strength':<12} {'Size':<8}")
print("-"*80)
for h in hotspots_without[:5]:
    quadrant = get_quadrant(h['x_center'], h['y_center'])
    print(f"{h['id']:<5} ({h['x_center']:.3f}, {h['y_center']:.3f}){'':<6} {quadrant:<15} {h['strength']:>11.4f} {h['size']:>7}")

# Visualize hotspots
fig, axes = plt.subplots(2, 2, figsize=(14, 14))

for row_idx, (attn_map, hotspot_mask, hotspots, label) in enumerate([
    (attn_map_with, hotspot_mask_with, hotspots_with, 'With Triangle'),
    (attn_map_without, hotspot_mask_without, hotspots_without, 'Without Triangle')
]):
    # Attention heatmap with hotspot markers
    ax = axes[row_idx, 0]
    im = ax.imshow(attn_map, cmap='hot', extent=[0, 1, 1, 0])
    ax.contour(hotspot_mask, levels=[0.5], colors='cyan', linewidths=2, extent=[0, 1, 1, 0])

    # Mark top 3 hotspots
    for i, h in enumerate(hotspots[:3]):
        ax.scatter(h['x_center'], h['y_center'], c='cyan', s=200, marker='x', linewidths=3)
        ax.text(h['x_center'] + 0.05, h['y_center'], f"#{i+1}", color='cyan', fontsize=10, fontweight='bold')

    ax.set_title(f'{label}\nAttention Hotspots', fontsize=12)
    ax.set_xlabel('X (0=left, 1=right)')
    ax.set_ylabel('Y (0=top, 1=bottom)')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Hotspot mask only
    ax = axes[row_idx, 1]
    im = ax.imshow(hotspot_mask, cmap='binary', extent=[0, 1, 1, 0])
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
    ax.set_title(f'{label}\nHotspot Regions', fontsize=12)
    ax.set_xlabel('X (0=left, 1=right)')
    ax.set_ylabel('Y (0=top, 1=bottom)')

plt.tight_layout()
plt.savefig('attention_data/hotspot_detection.png', dpi=150, bbox_inches='tight')
print("\nSaved hotspot detection visualization to attention_data/hotspot_detection.png")

# ==================== METRIC 7: Mutual Information ====================
print("\n" + "="*80)
print("METRIC 7: Mutual Information (Vision-Output dependency)")
print("="*80)

def compute_mutual_information(data, vision_token_start, vision_token_end):
    """Compute mutual information between vision token attention and output tokens"""
    llm_attn_matrix = data['llm_attn_matrix']
    output_token_start = data['output_token_start']
    output_token_end = data['output_token_end']

    # Get attention from all output tokens to all vision tokens
    # Shape: [num_output_tokens, num_vision_tokens]
    attn_matrix = llm_attn_matrix[output_token_start:output_token_end, vision_token_start:vision_token_end]
    attn_matrix = attn_matrix.float().cpu().numpy()

    # Compute MI for each vision token
    # MI measures how much knowing attention to a vision token reduces uncertainty about output tokens
    num_vision = attn_matrix.shape[1]
    mi_scores = []

    # Create output token indices as target variable
    y = np.arange(attn_matrix.shape[0])

    for vis_idx in range(num_vision):
        X = attn_matrix[:, vis_idx].reshape(-1, 1)
        # Compute MI between this vision token's attention and the output token sequence
        mi = mutual_info_regression(X, y, random_state=42)[0]
        mi_scores.append(mi)

    mi_scores = np.array(mi_scores)
    return mi_scores

print("\nComputing mutual information...")
mi_with = compute_mutual_information(data_with, vision_token_start, vision_token_end)
mi_without = compute_mutual_information(data_without, vision_token_start, vision_token_end)

# Analyze MI in top-left quadrant
top_left_indices = []
mid_row = grid_size // 2
mid_col = grid_size // 2
for i in range(mid_row):
    for j in range(mid_col):
        top_left_indices.append(i * grid_size + j)

mi_tl_with = mi_with[top_left_indices].mean()
mi_tl_without = mi_without[top_left_indices].mean()

mi_overall_with = mi_with.mean()
mi_overall_without = mi_without.mean()

print("\n" + "-"*80)
print("Mutual Information Analysis:")
print("-"*80)
print(f"{'Region':<30} {'With Triangle':<20} {'Without Triangle':<20} {'Difference':<15}")
print("-"*80)
print(f"{'Overall (mean)':<30} {mi_overall_with:>19.6f} {mi_overall_without:>19.6f} {mi_overall_with - mi_overall_without:>14.6f}")
print(f"{'Top-Left Quadrant':<30} {mi_tl_with:>19.6f} {mi_tl_without:>19.6f} {mi_tl_with - mi_tl_without:>14.6f}")

print("\nInterpretation:")
if mi_tl_with > mi_tl_without:
    print("  ✓ WITH triangle has HIGHER MI in top-left")
    print("    → Top-left vision tokens contain MORE information about output when triangle is present")
else:
    print("  ✗ WITHOUT triangle has higher MI in top-left")

# Visualize MI
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# With triangle
ax = axes[0]
mi_map = mi_with.reshape(grid_size, grid_size)
im = ax.imshow(mi_map, cmap='viridis')
ax.set_title('Mutual Information\n(With Triangle)', fontsize=12)
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046)

# Without triangle
ax = axes[1]
mi_map = mi_without.reshape(grid_size, grid_size)
im = ax.imshow(mi_map, cmap='viridis')
ax.set_title('Mutual Information\n(Without Triangle)', fontsize=12)
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046)

# Difference
ax = axes[2]
mi_diff = (mi_with - mi_without).reshape(grid_size, grid_size)
im = ax.imshow(mi_diff, cmap='RdBu_r', vmin=-np.abs(mi_diff).max(), vmax=np.abs(mi_diff).max())
ax.set_title('MI Difference\n(With - Without)', fontsize=12)
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.savefig('attention_data/mutual_information.png', dpi=150, bbox_inches='tight')
print("\nSaved mutual information visualization to attention_data/mutual_information.png")

# Save results
print("\nSaving extended metrics results...")
torch.save({
    'dynamics': {
        'with': dynamics_with,
        'without': dynamics_without,
    },
    'hotspots': {
        'with': hotspots_with,
        'without': hotspots_without,
    },
    'mutual_information': {
        'with': mi_with,
        'without': mi_without,
        'overall': {'with': mi_overall_with, 'without': mi_overall_without},
        'top_left': {'with': mi_tl_with, 'without': mi_tl_without},
    }
}, 'attention_data/extended_metrics.pt')

print("\n" + "="*80)
print("Extended Metrics Analysis Complete!")
print("="*80)
print("\nGenerated files:")
print("  - attention_data/attention_dynamics.png")
print("  - attention_data/hotspot_detection.png")
print("  - attention_data/mutual_information.png")
print("  - attention_data/extended_metrics.pt")
