import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# =====================
# Config
# =====================
MODEL_NAME = "lmsys/vicuna-7b-v1.5"
CAPTION_PATH = "/home/maqima/VLM-Visualizer/data/spatial_twoshapes/agreement/relational/test/shard0/caption.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

H_REL = ["left", "right"]
V_REL = ["above", "below"]

VALID_SHAPES = {
    "circle", "ellipse", "triangle", "square",
    "rectangle", "pentagon", "semicircle", "cross"
}

# =====================
# Load model
# =====================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    output_attentions=True
)
model.eval()

# =====================
# Utilities
# =====================
def extract_mean_attention(outputs):
    layers = []
    for layer in outputs.attentions:
        layers.append(layer.squeeze(0).mean(0))
    return torch.stack(layers).mean(0)  # [L, L]


def extract_entities(tokens):
    entities = []
    for i, t in enumerate(tokens):
        clean = t.replace("▁", "")
        if clean in VALID_SHAPES:
            entities.append((i, clean))
    return entities


def find_relation(tokens):
    for i, t in enumerate(tokens):
        clean = t.replace("▁", "")
        if clean in H_REL + V_REL:
            return clean, [i]
    return None, []


def pool_attention(attn, query_pos, key_pos):
    return attn[query_pos][:, key_pos].mean().item()

# =====================
# Main loop
# =====================
results = []

with open(CAPTION_PATH) as f:
    captions = [l.strip() for l in f.readlines() if l.strip()]

valid_sample_count = 0
for idx, sent in enumerate(captions):
    inputs = tokenizer(sent, return_tensors="pt").to(DEVICE)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        outputs = model(**inputs)

    attn = extract_mean_attention(outputs)

    # ---- entities ----
    entities = extract_entities(tokens)
    if len(entities) != 2:
        continue
    (e1_pos, e1_name), (e2_pos, e2_name) = entities

    # ---- relation ----
    rel, rel_pos = find_relation(tokens)
    if not rel_pos:
        continue

    # ---- attentions ----
    r_to_e1 = pool_attention(attn, rel_pos, [e1_pos])
    r_to_e2 = pool_attention(attn, rel_pos, [e2_pos])

    e1_to_e2 = pool_attention(attn, [e1_pos], [e2_pos])
    e2_to_e1 = pool_attention(attn, [e2_pos], [e1_pos])

    asym_r = r_to_e1 - r_to_e2
    asym_e = e1_to_e2 - e2_to_e1

    expected_sign = +1 if rel in ["right", "above"] else -1
    direction_correct = (np.sign(asym_r) == expected_sign)

    results.append({
        "sentence": sent,
        "relation": rel,
        "rel_to_entity1": r_to_e1,
        "rel_to_entity2": r_to_e2,
        "entity1_to_entity2": e1_to_e2,
        "entity2_to_entity1": e2_to_e1,
        "asymmetry_r": asym_r,
        "asymmetry_e": asym_e,
        "direction_correct": direction_correct
    })
    valid_sample_count += 1
    if idx % 10 == 0:
        print(f"[{idx}] {sent}")
        print(f"  rel→E1={r_to_e1:.4f}, rel→E2={r_to_e2:.4f}, correct={direction_correct}")
print(f"Processed {valid_sample_count} valid samples out of {len(captions)}")
# =====================
# Aggregate analysis
# =====================
df = pd.DataFrame(results)

print("\n=== Mean statistics by relation ===")
print(df.groupby("relation")[[
    "rel_to_entity1",
    "rel_to_entity2",
    "asymmetry_r",
    "direction_correct"
]].mean())

df.to_csv("text_only_relation_attention.csv", index=False)
print("Saved to text_only_relation_attention.csv")
