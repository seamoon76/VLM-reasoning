import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# =====================
# Config
# =====================
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
CAPTION_PATH = "/home/maqima/VLM-Visualizer/data/spatial_twoshapes/agreement/relational/test/shard0/caption.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

def pool_from_last(attn, key_pos):
    """
    attn: [L, L]
    key_pos: list[int]
    return: scalar
    """
    last_pos = attn.shape[0] - 1
    return attn[last_pos, key_pos].mean().item()

def parse_relation_sentence(sent):
    """
    Parse sentences like:
    'a gray ellipse is to the left of a blue circle .'

    return:
        (entity1, rel, entity2)
    where entity = 'a gray ellipse'
    """
    sent = sent.strip().lower()
    for rel in REL_FLIP.keys():
        pattern = f" is to the {rel} of "
        if pattern in sent:
            left, right = sent.split(pattern)
            entity1 = left.strip()
            entity2 = right.replace(".", "").strip()
            return entity1, rel, entity2
    return None

def make_counterfactual(sent):
    parsed = parse_relation_sentence(sent)
    if parsed is None:
        return None

    entity1, rel, entity2 = parsed
    flipped_rel = REL_FLIP[rel]

    # Swap entities + flip relation
    new_sent = f"{entity2} is to the {flipped_rel} of {entity1} ."
    return new_sent

# =====================
# Main loop
# =====================
results = []

paired_captions = []

with open(CAPTION_PATH) as f:
    for line in f:
        sent = line.strip()
        if not sent:
            continue

        cf = make_counterfactual(sent)
        if cf is None:
            continue

        paired_captions.append((sent, cf))

def relation_token_to_entities(sent, tokenizer, model, device):
    inputs = tokenizer(sent, return_tensors="pt").to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        outputs = model(**inputs)

    attn = extract_mean_attention(outputs)  # [L, L]

    entities = extract_entities(tokens)
    if len(entities) != 2:
        return None

    (e1_pos, e1_name), (e2_pos, e2_name) = entities
    rel_tokens = ["left", "right", "above", "below"]
    rel_pos = find_relation(tokens)[1]
    if not rel_pos:
        return None
    # pool attention from relation token instead of last token
    r_to_e1 = pool_attention(attn, rel_pos, [e1_pos])
    r_to_e2 = pool_attention(attn, rel_pos, [e2_pos])
    asym_r = r_to_e1 - r_to_e2

    return {
        "sent": sent,
        "tokens": tokens,
        "e1_name": e1_name,
        "e2_name": e2_name,
        "rel_pos": rel_pos,
        "r_to_e1": r_to_e1,
        "r_to_e2": r_to_e2,
        "asym": asym_r
    }

pair_results = []

for sent1, sent2 in paired_captions:
    r1 = relation_token_to_entities(sent1, tokenizer, model, DEVICE)
    r2 = relation_token_to_entities(sent2, tokenizer, model, DEVICE)

    if r1 is None or r2 is None:
        continue

    # 注意：S2 中 entity 顺序是反的
    # S1: (A, B)
    # S2: (B, A)
    asym_s1 = r1["asym"]
    asym_s2 = r2["asym"]

    # 如果关系被显式编码，这两个应该近似互为相反数
    symmetry_score = asym_s1 + asym_s2  # 理想情况 ≈ 0

    pair_results.append({
        "sent1": sent1,
        "sent2": sent2,
        "asym_s1": asym_s1,
        "asym_s2": asym_s2,
        "symmetry_score": symmetry_score
    })


import pandas as pd

df_pairs = pd.DataFrame(pair_results)

print("=== Pair-level symmetry statistics ===")
print(df_pairs[["asym_s1", "asym_s2", "symmetry_score"]].mean())
print()
print("Mean |symmetry_score|:",
      df_pairs["symmetry_score"].abs().mean())

import matplotlib.pyplot as plt

plt.figure(figsize=(5, 5))
plt.scatter(df_pairs["asym_s1"], df_pairs["asym_s2"], alpha=0.5)
plt.axhline(0)
plt.axvline(0)
plt.plot(
    [-0.05, 0.05], [0.05, -0.05],
    linestyle="--", color="gray", label="Ideal symmetry"
)

plt.xlabel("Asymmetry (A left of B)")
plt.ylabel("Asymmetry (B right of A)")
plt.title("Pair-level Attention Asymmetry")
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig("pair_level_symmetry.png")
plt.close()

example = df_pairs.iloc[0]

r1 = relation_token_to_entities(example["sent1"], tokenizer, model, DEVICE)
r2 = relation_token_to_entities(example["sent2"], tokenizer, model, DEVICE)

labels = ["Entity 1", "Entity 2"]
vals1 = [r1["r_to_e1"], r1["r_to_e2"]]
vals2 = [r2["r_to_e1"], r2["r_to_e2"]]

plt.figure(figsize=(8, 3))

plt.subplot(1, 2, 1)
plt.bar(labels, vals1)
plt.title("A is left of B")

plt.subplot(1, 2, 2)
plt.bar(labels, vals2)
plt.title("B is right of A")

plt.tight_layout()
plt.show()
plt.savefig("example_attention_bars.png")
plt.close()
