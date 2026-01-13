import os
import sys
import re
from typing import List, Optional

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# =====================
# Qwen2-VL model loading
# =====================
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto" if DEVICE.startswith("cuda") else None,
    attn_implementation="eager",
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
tokenizer = processor.tokenizer
model.eval()

# =====================
# Data: 5 shards
# =====================
BASE_DIR = "data/spatial_twoshapes/agreement/relational"
SHARD_IDS = list(range(5))  # 0..4

H_REL = ["left", "right"]
V_REL = ["above", "below"]

VALID_SHAPES = {
    "circle", "ellipse", "triangle", "square",
    "rectangle", "pentagon", "semicircle", "cross"
}

REL_FLIP = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}

# =====================
# Utilities
# =====================
def extract_mean_attention(outputs, softmax_if_needed: bool = True) -> torch.Tensor:
    """
    outputs.attentions: tuple/list[num_layers] of Tensor[bs, heads, T, T]
    Return mean over layers & heads -> [T, T]
    """
    mats = []
    for layer_attn in outputs.attentions:
        # typical: [bs, heads, T, T]
        a = layer_attn[0].detach().float()  # [heads, T, T]

        if softmax_if_needed:
            m = a.mean(dim=0)  # [T, T]
            row_mean = m.sum(dim=-1).mean().item()
            if (m.min().item() < -1e-6) or (abs(row_mean - 1.0) > 0.05):
                a = torch.softmax(a, dim=-1)

        mats.append(a.mean(dim=0))  # [T, T]

    return torch.stack(mats, dim=0).mean(dim=0)  # [T, T]


def is_special(tok: str) -> bool:
    return tok.startswith("<|") and tok.endswith("|>")


def clean_piece(tok: str) -> str:
    return tok.replace("Ġ", "").replace("▁", "")


def find_caption_span(tokens_raw: List[str]) -> tuple[int, int]:
    """
    For text-only Qwen2 chat template:
    We take everything before the last "<|assistant|>" as the "user content span".
    """
    last_assistant = None
    for i, t in enumerate(tokens_raw):
        if t == "<|assistant|>":
            last_assistant = i
    if last_assistant is None:
        return 0, len(tokens_raw) - 1
    return 0, last_assistant - 1


def extract_entities(tokens_raw: List[str]) -> List[tuple[int, str]]:
    entities = []
    for i, t in enumerate(tokens_raw):
        if is_special(t):
            continue
        clean = clean_piece(t).lower()
        if clean in VALID_SHAPES:
            entities.append((i, clean))
    return entities


def find_relation(tokens_raw: List[str]) -> tuple[Optional[str], List[int]]:
    """
    Find relation after 'the spatial relation is' trigger.
    """
    trigger = ["spatial", "relation", "is"]
    cleaned = [clean_piece(t).lower() for t in tokens_raw]

    for i in range(len(cleaned) - len(trigger)):
        window = cleaned[i:i + len(trigger)]
        if window == trigger:
            rel = cleaned[i + len(trigger)]
            if rel in H_REL + V_REL:
                return rel, [i + len(trigger)]
    return None, []


def pool_attention(attn_TT: torch.Tensor, query_pos: List[int], key_pos: List[int]) -> float:
    return attn_TT[query_pos][:, key_pos].mean().item()


def make_counterfactual_swap_entities_only(sent: str) -> Optional[str]:
    """
    Swap the two entities in the first sentence, keep middle question,
    and flip final relation token.
    """
    sent = sent.strip().lower()

    # "there are two shapes X and Y ."
    match = re.match(r"(there are two shapes )(.+) and (.+?)( \.)", sent)
    if not match:
        return None

    prefix, ent1, ent2, suffix = match.groups()
    swapped_entity_sentence = f"{prefix}{ent2} and {ent1}{suffix}"

    # final: "the spatial relation is <rel> ."
    rel_pattern = re.search(r"(the spatial relation is )(\w+)( ?\.)$", sent)
    if not rel_pattern:
        return None

    rel_prefix, rel_token, rel_suffix = rel_pattern.groups()
    rel_flipped = REL_FLIP.get(rel_token, rel_token)
    swapped_relation_sentence = f"{rel_prefix}{rel_flipped}{rel_suffix}"

    middle_start = match.end()
    middle_end = rel_pattern.start()
    middle_question = sent[middle_start:middle_end].strip()

    return f"{swapped_entity_sentence} {middle_question} {swapped_relation_sentence}"


def relation_token_to_entities(sent: str) -> Optional[dict]:
    """
    Text-only Qwen2:
      - build chat template with user text
      - forward with output_attentions=True
      - mean attention [T,T]
      - locate first two entities + relation token (after trigger)
      - compute r_to_e1, r_to_e2, asym
    """
    messages = [{"role": "user", "content": [{"type": "text", "text": sent}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[text],
        images=None,
        padding=True,
        return_tensors="pt",
    )

    for k, v in list(inputs.items()):
        if torch.is_tensor(v):
            inputs[k] = v.to(model.device)

    input_ids = inputs["input_ids"][0].detach().cpu()
    tokens_raw = tokenizer.convert_ids_to_tokens(input_ids.tolist())

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, use_cache=False, return_dict=True)

    attn = extract_mean_attention(outputs).detach().cpu()  # [T,T]

    cap_s, cap_e = find_caption_span(tokens_raw)
    tokens_span = tokens_raw[cap_s:cap_e + 1]
    offset = cap_s

    entities = extract_entities(tokens_span)
    if len(entities) < 2:
        return None

    (e1_pos, e1_name), (e2_pos, e2_name) = entities[:2]
    e1_pos += offset
    e2_pos += offset

    rel, rel_pos = find_relation(tokens_span)
    if not rel_pos:
        return None
    rel_pos = [p + offset for p in rel_pos]

    r_to_e1 = pool_attention(attn, rel_pos, [e1_pos])
    r_to_e2 = pool_attention(attn, rel_pos, [e2_pos])

    return {
        "sent": sent,
        "tokens": tokens_raw,
        "e1": e1_name,
        "e2": e2_name,
        "relation": rel,
        "r_to_e1": r_to_e1,
        "r_to_e2": r_to_e2,
        "asym": r_to_e1 - r_to_e2
    }


# =====================
# Main: all shards
# =====================
paired_captions_all = []

total_lines = 0
total_pairs = 0
total_cf_fail = 0

for shard_id in SHARD_IDS:
    cap_path = os.path.join(BASE_DIR, f"shard{shard_id}", "caption.txt")
    if not os.path.exists(cap_path):
        print(f"[WARN] caption not found: {cap_path} (skip)")
        continue

    with open(cap_path, "r") as f:
        for line in f:
            sent = line.strip()
            if not sent:
                continue
            total_lines += 1
            cf = make_counterfactual_swap_entities_only(sent)
            if cf is None:
                total_cf_fail += 1
                continue
            paired_captions_all.append((sent, cf))
            total_pairs += 1

print(f"[INFO] Read captions from shards {SHARD_IDS}")
print(f"[INFO] total_lines={total_lines}, total_pairs={total_pairs}, cf_fail={total_cf_fail}")

pair_results = []
skipped = 0

for idx, (sent1, sent2) in enumerate(paired_captions_all):
    r1 = relation_token_to_entities(sent1)
    r2 = relation_token_to_entities(sent2)
    if r1 is None or r2 is None:
        skipped += 1
        continue

    asym_s1 = r1["asym"]
    asym_s2 = r2["asym"]
    symmetry_score = asym_s1 + asym_s2  # ideal ~ 0

    pair_results.append({
        "sent1": sent1,
        "sent2": sent2,
        "asym_s1": asym_s1,
        "asym_s2": asym_s2,
        "symmetry_score": symmetry_score
    })

    if (idx + 1) % 200 == 0:
        print(f"[INFO] processed {idx+1}/{len(paired_captions_all)} pairs, kept={len(pair_results)}, skipped={skipped}")

df_pairs = pd.DataFrame(pair_results)

print("\n=== Pair-level symmetry statistics (ALL SHARDS) ===")
if len(df_pairs) == 0:
    print("No valid pairs found (entity/relation token matching failed).")
    sys.exit(0)

mean_stats = df_pairs[["asym_s1", "asym_s2", "symmetry_score"]].mean()
abs_sym_mean = df_pairs["symmetry_score"].abs().mean()

print(mean_stats)
print("\nMean |symmetry_score|:", abs_sym_mean)
print(f"\n[INFO] kept_pairs={len(df_pairs)}, skipped_pairs={skipped}")

# =====================
# Save ALL-only outputs
# =====================
out_png_scatter = "pair_level_symmetry_qwen2_allshards.png"
out_png_bars = "example_attention_bars_qwen2_allshards.png"
out_txt = "pair_level_symmetry_stats_qwen2_allshards.txt"
out_csv = "pair_level_symmetry_pairs_qwen2_allshards.csv"

# scatter
plt.figure(figsize=(5, 5))
plt.scatter(df_pairs["asym_s1"], df_pairs["asym_s2"], alpha=0.35, s=8)
plt.axhline(0)
plt.axvline(0)
plt.plot([-0.05, 0.05], [0.05, -0.05], linestyle="--", color="gray", label="Ideal symmetry")
plt.xlabel("Asymmetry S1")
plt.ylabel("Asymmetry S2")
plt.title("Pair-level Attention Asymmetry (Qwen2-VL)")
plt.legend()
plt.tight_layout()
plt.savefig(out_png_scatter, dpi=200)
plt.show()
plt.close()

# save txt stats
with open(out_txt, "w") as f:
    f.write("=== Pair-level symmetry statistics (Qwen2, ALL SHARDS) ===\n")
    f.write(f"shards={SHARD_IDS}\n")
    f.write(f"total_lines={total_lines}\n")
    f.write(f"total_pairs_after_cf={total_pairs}\n")
    f.write(f"cf_fail={total_cf_fail}\n")
    f.write(f"kept_pairs={len(df_pairs)}\n")
    f.write(f"skipped_pairs={skipped}\n\n")
    for k, v in mean_stats.items():
        f.write(f"{k}_mean={float(v):.6f}\n")
    f.write(f"abs_symmetry_score_mean={float(abs_sym_mean):.6f}\n")

# optional: save csv of all pairs for later analysis
df_pairs.to_csv(out_csv, index=False)

# example bars (first row)
example = df_pairs.iloc[0]
print("\nExample sentences:")
print("S1:", example["sent1"])
print("S2:", example["sent2"])

r1 = relation_token_to_entities(example["sent1"])
r2 = relation_token_to_entities(example["sent2"])

labels = ["Entity 1", "Entity 2"]
vals1 = [r1["r_to_e1"], r1["r_to_e2"]]
vals2 = [r2["r_to_e1"], r2["r_to_e2"]]

plt.figure(figsize=(8, 3))

plt.subplot(1, 2, 1)
plt.bar(labels, vals1)
plt.title(f"{r1['e1']} → {r1['e2']} : {r1['relation']}")

plt.subplot(1, 2, 2)
plt.bar(labels, vals2)
plt.title(f"{r2['e1']} → {r2['e2']} : {r2['relation']}")

plt.tight_layout()
plt.savefig(out_png_bars, dpi=200)
plt.show()
plt.close()

print(f"\n[OK] Saved ALL-only outputs:\n  {out_txt}\n  {out_png_scatter}\n  {out_png_bars}\n  {out_csv}")
