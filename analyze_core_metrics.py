import os
import sys
sys.path.append("./models")
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon, cosine
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the saved attention data
print("Loading attention data...")
data_with = torch.load('attention_data/attention_with_triangle.pt')
data_without = torch.load('attention_data/attention_without_triangle.pt')

grid_size = data_with['grid_size']
vision_token_start = data_with['vision_token_start']
vision_token_end = data_with['vision_token_end']

print(f"\nWith triangle: {data_with['response_text']}")
print(f"Without triangle: {data_without['response_text']}")

# ==================== METRIC 1: Attention Entropy ====================
print("\n" + "="*80)
print("METRIC 1: Attention Entropy & Concentration")
print("="*80)

def compute_attention_entropy(attn_dist):
    """Compute Shannon entropy of attention distribution"""
    # Ensure it's a probability distribution
    attn_dist = attn_dist / (attn_dist.sum() + 1e-10)
    attn_np = attn_dist.float().cpu().numpy()

    # Shannon entropy
    ent = entropy(attn_np + 1e-10)

    # Normalized entropy (0-1)
    max_ent = np.log(len(attn_np))
    normalized_ent = ent / max_ent if max_ent > 0 else 0

    return ent, normalized_ent

def compute_gini_coefficient(attn_dist):
    """Compute Gini coefficient - measures inequality in attention distribution"""
    attn_np = attn_dist.float().cpu().numpy()
    attn_sorted = np.sort(attn_np)
    n = len(attn_sorted)
    cumsum = np.cumsum(attn_sorted)
    gini = (2 * np.sum((np.arange(1, n+1)) * attn_sorted)) / (n * cumsum[-1]) - (n + 1) / n
    return gini

def compute_effective_span(attn_dist, threshold=0.9):
    """Number of tokens needed to cover threshold% of total attention"""
    attn_sorted = torch.sort(attn_dist, descending=True)[0]
    cumsum = torch.cumsum(attn_sorted, dim=0)
    span = (cumsum < threshold).sum().item() + 1
    return span

def compute_peak_ratio(attn_dist):
    """Ratio of max attention to mean attention"""
    max_attn = attn_dist.max().item()
    mean_attn = attn_dist.mean().item()
    return max_attn / (mean_attn + 1e-10)

# Analyze both conditions
print("\nComputing entropy metrics for both conditions...")

# Get aggregated attention over vision tokens
llm_attn_with = data_with['llm_attn_matrix']
llm_attn_without = data_without['llm_attn_matrix']

output_start_with = data_with['output_token_start']
output_end_with = data_with['output_token_end']
output_start_without = data_without['output_token_start']
output_end_without = data_without['output_token_end']

# Average attention across all output tokens to vision tokens
avg_attn_with = llm_attn_with[output_start_with:output_end_with, vision_token_start:vision_token_end].mean(dim=0)
avg_attn_without = llm_attn_without[output_start_without:output_end_without, vision_token_start:vision_token_end].mean(dim=0)

# Compute metrics
ent_with, norm_ent_with = compute_attention_entropy(avg_attn_with)
ent_without, norm_ent_without = compute_attention_entropy(avg_attn_without)

gini_with = compute_gini_coefficient(avg_attn_with)
gini_without = compute_gini_coefficient(avg_attn_without)

span_with = compute_effective_span(avg_attn_with)
span_without = compute_effective_span(avg_attn_without)

peak_with = compute_peak_ratio(avg_attn_with)
peak_without = compute_peak_ratio(avg_attn_without)

print("\n" + "-"*80)
print("Entropy & Concentration Metrics:")
print("-"*80)
print(f"{'Metric':<30} {'With Triangle':<20} {'Without Triangle':<20} {'Difference':<15}")
print("-"*80)
print(f"{'Shannon Entropy':<30} {ent_with:>19.4f} {ent_without:>19.4f} {ent_with - ent_without:>14.4f}")
print(f"{'Normalized Entropy (0-1)':<30} {norm_ent_with:>19.4f} {norm_ent_without:>19.4f} {norm_ent_with - norm_ent_without:>14.4f}")
print(f"{'Gini Coefficient':<30} {gini_with:>19.4f} {gini_without:>19.4f} {gini_with - gini_without:>14.4f}")
print(f"{'Effective Span (90%)':<30} {span_with:>19} {span_without:>19} {span_with - span_without:>14}")
print(f"{'Peak/Mean Ratio':<30} {peak_with:>19.4f} {peak_without:>19.4f} {peak_with - peak_without:>14.4f}")

print("\nInterpretation:")
if norm_ent_with < norm_ent_without:
    print("  ✓ WITH triangle has LOWER entropy → attention is more CONCENTRATED")
    print("    (Model focuses on specific regions when object is present)")
else:
    print("  ✗ WITH triangle has HIGHER entropy → attention is more DISPERSED")

if gini_with > gini_without:
    print("  ✓ WITH triangle has HIGHER Gini → attention is more UNEQUAL/FOCUSED")
else:
    print("  ✗ WITH triangle has LOWER Gini → attention is more EQUAL/DISPERSED")

# ==================== METRIC 2: Center of Mass ====================
print("\n" + "="*80)
print("METRIC 2: Attention Center of Mass & Spatial Moments")
print("="*80)

def compute_center_of_mass(attn_dist, grid_size):
    """Compute the center of mass of attention distribution"""
    attn_map = attn_dist.reshape(grid_size, grid_size).float().cpu().numpy()

    # Normalize to probability distribution
    attn_map = attn_map / (attn_map.sum() + 1e-10)

    # Create coordinate grids (normalized to [0, 1])
    y_coords, x_coords = np.meshgrid(
        np.linspace(0, 1, grid_size),
        np.linspace(0, 1, grid_size),
        indexing='ij'
    )

    # Compute center of mass
    x_center = np.sum(x_coords * attn_map)
    y_center = np.sum(y_coords * attn_map)

    # Compute second moments (spread)
    x_variance = np.sum((x_coords - x_center)**2 * attn_map)
    y_variance = np.sum((y_coords - y_center)**2 * attn_map)

    return x_center, y_center, x_variance, y_variance

x_com_with, y_com_with, x_var_with, y_var_with = compute_center_of_mass(avg_attn_with, grid_size)
x_com_without, y_com_without, x_var_without, y_var_without = compute_center_of_mass(avg_attn_without, grid_size)

print("\n" + "-"*80)
print("Center of Mass (normalized coordinates, 0=top/left, 1=bottom/right):")
print("-"*80)
print(f"{'Metric':<30} {'With Triangle':<20} {'Without Triangle':<20} {'Difference':<15}")
print("-"*80)
print(f"{'X Center (horizontal)':<30} {x_com_with:>19.4f} {x_com_without:>19.4f} {x_com_with - x_com_without:>14.4f}")
print(f"{'Y Center (vertical)':<30} {y_com_with:>19.4f} {y_com_without:>19.4f} {y_com_with - y_com_without:>14.4f}")
print(f"{'X Variance (spread)':<30} {x_var_with:>19.4f} {x_var_without:>19.4f} {x_var_with - x_var_without:>14.4f}")
print(f"{'Y Variance (spread)':<30} {y_var_with:>19.4f} {y_var_without:>19.4f} {y_var_with - y_var_without:>14.4f}")

print("\nInterpretation:")
print(f"  With triangle: attention center at ({x_com_with:.2f}, {y_com_with:.2f})")
print(f"  Without triangle: attention center at ({x_com_without:.2f}, {y_com_without:.2f})")

if x_com_with < x_com_without:
    print(f"  ✓ WITH triangle shifts attention LEFT by {(x_com_without - x_com_with):.4f}")
if y_com_with < y_com_without:
    print(f"  ✓ WITH triangle shifts attention UP by {(y_com_without - y_com_with):.4f}")

# Visualize center of mass
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, (attn, x_com, y_com, title) in enumerate([
    (avg_attn_with, x_com_with, y_com_with, 'With Triangle'),
    (avg_attn_without, x_com_without, y_com_without, 'Without Triangle')
]):
    ax = axes[idx]
    attn_map = attn.reshape(grid_size, grid_size).float().cpu().numpy()

    im = ax.imshow(attn_map, cmap='hot', extent=[0, 1, 1, 0])
    ax.scatter(x_com, y_com, c='cyan', s=200, marker='x', linewidths=3, label='Center of Mass')

    # Mark quadrants
    ax.axhline(y=0.5, color='white', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=0.5, color='white', linestyle='--', alpha=0.5, linewidth=1)

    ax.set_title(f'{title}\nCoM: ({x_com:.3f}, {y_com:.3f})', fontsize=12)
    ax.set_xlabel('X (0=left, 1=right)')
    ax.set_ylabel('Y (0=top, 1=bottom)')
    ax.legend(loc='upper right')
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.savefig('attention_data/center_of_mass.png', dpi=150, bbox_inches='tight')
print("\nSaved center of mass visualization to attention_data/center_of_mass.png")

# ==================== METRIC 3: Attention Rollout ====================
print("\n" + "="*80)
print("METRIC 3: Attention Rollout (Complete Information Flow)")
print("="*80)

def compute_attention_rollout(llm_attentions, vision_attentions, vision_token_start, vision_token_end):
    """
    Compute attention rollout from output tokens back to image patches
    Combines LLM attention and vision encoder attention
    """
    # Step 1: Get LLM attention from first output token to vision tokens
    first_step_attns = llm_attentions[0]  # First generation step

    # Average across all layers and heads for the last position (being generated)
    llm_to_vision = []
    for layer_attn in first_step_attns:
        layer_attn = layer_attn.squeeze(0)  # [num_heads, seq_len, seq_len]
        # Get attention from last position to vision tokens
        head_avg = layer_attn[:, -1, vision_token_start:vision_token_end].mean(dim=0)  # [num_vision_tokens]
        llm_to_vision.append(head_avg)

    llm_to_vision = torch.stack(llm_to_vision).mean(dim=0)  # Average across layers
    llm_to_vision = llm_to_vision / (llm_to_vision.sum() + 1e-10)

    # Step 2: Get vision encoder attention (already averaged in vis_attn_matrix)
    # This is attention from each patch to all other patches
    # We'll use the average attention to each patch

    # For simplicity, use the diagonal (self-attention) as a proxy for importance
    # Or compute the mean attention received by each patch
    vis_attn = vision_attentions[0].squeeze(0)  # First layer, [num_heads, patches+1, patches+1]
    vis_attn = vis_attn.mean(dim=0)[1:, 1:]  # Remove CLS, average heads, [patches, patches]

    # Average attention received by each patch
    patch_importance = vis_attn.mean(dim=0)  # [patches]
    patch_importance = patch_importance / (patch_importance.sum() + 1e-10)

    # Step 3: Combine LLM->vision and vision internal attention
    # Rollout: LLM attention to vision tokens × importance of those tokens in vision encoder
    rollout = llm_to_vision.cpu() * patch_importance.cpu()
    rollout = rollout / (rollout.sum() + 1e-10)

    return rollout, llm_to_vision, patch_importance

print("\nComputing attention rollout...")
rollout_with, llm_vis_with, patch_imp_with = compute_attention_rollout(
    data_with['llm_attentions'],
    data_with['vision_attentions'],
    vision_token_start,
    vision_token_end
)

rollout_without, llm_vis_without, patch_imp_without = compute_attention_rollout(
    data_without['llm_attentions'],
    data_without['vision_attentions'],
    vision_token_start,
    vision_token_end
)

# Analyze rollout to top-left quadrant
def get_top_left_quadrant_indices(grid_size):
    indices = []
    mid_row = grid_size // 2
    mid_col = grid_size // 2
    for i in range(mid_row):
        for j in range(mid_col):
            indices.append(i * grid_size + j)
    return indices

top_left_indices = get_top_left_quadrant_indices(grid_size)

rollout_tl_with = rollout_with[top_left_indices].sum().item()
rollout_tl_without = rollout_without[top_left_indices].sum().item()

print("\n" + "-"*80)
print("Attention Rollout to Top-Left Quadrant:")
print("-"*80)
print(f"  With Triangle:    {rollout_tl_with:.4f}")
print(f"  Without Triangle: {rollout_tl_without:.4f}")
print(f"  Difference:       {rollout_tl_with - rollout_tl_without:.4f}")

# Visualize rollout
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for row_idx, (rollout, llm_vis, patch_imp, label) in enumerate([
    (rollout_with, llm_vis_with, patch_imp_with, 'With Triangle'),
    (rollout_without, llm_vis_without, patch_imp_without, 'Without Triangle')
]):
    # LLM to Vision
    ax = axes[row_idx, 0]
    attn_map = llm_vis.reshape(grid_size, grid_size).float().cpu().numpy()
    im = ax.imshow(attn_map, cmap='hot')
    ax.set_title(f'{label}\nLLM→Vision Attention', fontsize=11)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Vision Encoder Internal
    ax = axes[row_idx, 1]
    attn_map = patch_imp.reshape(grid_size, grid_size).float().cpu().numpy()
    im = ax.imshow(attn_map, cmap='hot')
    ax.set_title(f'{label}\nVision Encoder Importance', fontsize=11)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Rollout (combined)
    ax = axes[row_idx, 2]
    attn_map = rollout.reshape(grid_size, grid_size).float().cpu().numpy()
    im = ax.imshow(attn_map, cmap='hot')
    ax.set_title(f'{label}\nAttention Rollout', fontsize=11)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.savefig('attention_data/attention_rollout.png', dpi=150, bbox_inches='tight')
print("Saved attention rollout visualization to attention_data/attention_rollout.png")

# ==================== METRIC 4: Head Similarity & Clustering ====================
print("\n" + "="*80)
print("METRIC 4: Head Similarity & Functional Clustering")
print("="*80)

def extract_all_head_attentions(data, vision_token_start, vision_token_end):
    """Extract attention to vision tokens for all heads in all layers"""
    first_step_attns = data['llm_attentions'][0]

    all_heads = []
    head_labels = []

    for layer_idx, layer_attn in enumerate(first_step_attns):
        layer_attn = layer_attn.squeeze(0)  # [num_heads, seq_len, seq_len]
        num_heads = layer_attn.shape[0]

        for head_idx in range(num_heads):
            # Get attention from last position to vision tokens
            head_attn = layer_attn[head_idx, -1, vision_token_start:vision_token_end]
            head_attn = head_attn / (head_attn.sum() + 1e-10)

            all_heads.append(head_attn.cpu().float().numpy())
            head_labels.append(f"L{layer_idx}H{head_idx}")

    return np.array(all_heads), head_labels

print("\nExtracting attention patterns from all heads...")
heads_with, labels_with = extract_all_head_attentions(data_with, vision_token_start, vision_token_end)
heads_without, labels_without = extract_all_head_attentions(data_without, vision_token_start, vision_token_end)

print(f"Total number of heads: {len(heads_with)}")

# Compute similarity matrix (using cosine similarity)
print("\nComputing head similarity matrix...")
num_heads = len(heads_with)
similarity_matrix_with = np.zeros((num_heads, num_heads))
similarity_matrix_without = np.zeros((num_heads, num_heads))

for i in range(num_heads):
    for j in range(i, num_heads):
        sim_with = 1 - cosine(heads_with[i], heads_with[j])
        sim_without = 1 - cosine(heads_without[i], heads_without[j])

        similarity_matrix_with[i, j] = sim_with
        similarity_matrix_with[j, i] = sim_with

        similarity_matrix_without[i, j] = sim_without
        similarity_matrix_without[j, i] = sim_without

# Clustering heads by their attention patterns
print("\nClustering heads by attention patterns...")
n_clusters = 5  # Try 5 functional groups

kmeans_with = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters_with = kmeans_with.fit_predict(heads_with)

kmeans_without = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters_without = kmeans_without.fit_predict(heads_without)

# Silhouette score (measure of clustering quality)
silhouette_with = silhouette_score(heads_with, clusters_with)
silhouette_without = silhouette_score(heads_without, clusters_without)

print(f"\nClustering quality (Silhouette Score, higher is better):")
print(f"  With Triangle:    {silhouette_with:.4f}")
print(f"  Without Triangle: {silhouette_without:.4f}")

# Analyze cluster composition
print("\n" + "-"*80)
print("Cluster Composition (WITH triangle):")
print("-"*80)
for cluster_id in range(n_clusters):
    cluster_heads = [labels_with[i] for i in range(num_heads) if clusters_with[i] == cluster_id]
    print(f"Cluster {cluster_id}: {len(cluster_heads)} heads")
    print(f"  Sample members: {', '.join(cluster_heads[:5])}")

    # Show cluster centroid's top-left attention
    cluster_indices = np.where(clusters_with == cluster_id)[0]
    cluster_center = heads_with[cluster_indices].mean(axis=0)
    tl_attn = cluster_center[top_left_indices].sum()
    print(f"  Avg attention to top-left: {tl_attn:.4f}")

print("\n" + "-"*80)
print("Cluster Composition (WITHOUT triangle):")
print("-"*80)
for cluster_id in range(n_clusters):
    cluster_heads = [labels_without[i] for i in range(num_heads) if clusters_without[i] == cluster_id]
    print(f"Cluster {cluster_id}: {len(cluster_heads)} heads")
    print(f"  Sample members: {', '.join(cluster_heads[:5])}")

    cluster_indices = np.where(clusters_without == cluster_id)[0]
    cluster_center = heads_without[cluster_indices].mean(axis=0)
    tl_attn = cluster_center[top_left_indices].sum()
    print(f"  Avg attention to top-left: {tl_attn:.4f}")

# Visualize similarity matrix (subsample for readability)
print("\nVisualizing head similarity (subsampled)...")
subsample_step = 20  # Show every 20th head to keep visualization readable
subsample_indices = list(range(0, num_heads, subsample_step))

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for idx, (sim_matrix, title) in enumerate([
    (similarity_matrix_with, 'With Triangle'),
    (similarity_matrix_without, 'Without Triangle')
]):
    ax = axes[idx]
    subsampled = sim_matrix[np.ix_(subsample_indices, subsample_indices)]

    im = ax.imshow(subsampled, cmap='viridis', aspect='auto')
    ax.set_title(f'Head Similarity Matrix\n{title}', fontsize=12)
    ax.set_xlabel('Head Index (subsampled)')
    ax.set_ylabel('Head Index (subsampled)')

    # Set tick labels
    tick_labels = [labels_with[i] for i in subsample_indices]
    ax.set_xticks(range(len(subsample_indices)))
    ax.set_yticks(range(len(subsample_indices)))
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
    ax.set_yticklabels(tick_labels, fontsize=6)

    plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.savefig('attention_data/head_similarity.png', dpi=150, bbox_inches='tight')
print("Saved head similarity visualization to attention_data/head_similarity.png")

# Save all metrics
print("\nSaving core metrics results...")
torch.save({
    'entropy': {
        'with': {'shannon': ent_with, 'normalized': norm_ent_with, 'gini': gini_with,
                 'effective_span': span_with, 'peak_ratio': peak_with},
        'without': {'shannon': ent_without, 'normalized': norm_ent_without, 'gini': gini_without,
                    'effective_span': span_without, 'peak_ratio': peak_without},
    },
    'center_of_mass': {
        'with': {'x': x_com_with, 'y': y_com_with, 'x_var': x_var_with, 'y_var': y_var_with},
        'without': {'x': x_com_without, 'y': y_com_without, 'x_var': x_var_without, 'y_var': y_var_without},
    },
    'rollout': {
        'with': {'rollout': rollout_with, 'top_left': rollout_tl_with},
        'without': {'rollout': rollout_without, 'top_left': rollout_tl_without},
    },
    'clustering': {
        'with': {'clusters': clusters_with, 'silhouette': silhouette_with},
        'without': {'clusters': clusters_without, 'silhouette': silhouette_without},
        'labels': labels_with,
    }
}, 'attention_data/core_metrics.pt')

print("\n" + "="*80)
print("Core Metrics Analysis Complete!")
print("="*80)
print("\nGenerated files:")
print("  - attention_data/center_of_mass.png")
print("  - attention_data/attention_rollout.png")
print("  - attention_data/head_similarity.png")
print("  - attention_data/core_metrics.pt")
