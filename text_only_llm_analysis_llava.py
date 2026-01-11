import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os
sys.path.append("./models")

from llava.constants import (
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

# =====================
# Model loading
# =====================
model_path = "liuhaotian/llava-v1.5-7b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

disable_torch_init()

tokenizer, model, _, context_len = load_pretrained_model(
    model_path,
    None,
    model_path,
    load_8bit=False,
    load_4bit=False,
    device=DEVICE,
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
)

model.eval()
CAPTION_BASE = "/home/maqima/VLM-Visualizer/data/spatial_twoshapes/agreement/relational/"


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
    """
    Only find relation after 'the spatial relation is'
    """
    trigger = ["spatial", "relation", "is"]

    for i in range(len(tokens) - len(trigger)):
        window = [t.replace("▁", "") for t in tokens[i:i+len(trigger)]]
        if window == trigger:
            # relation should be right after
            rel_token = tokens[i + len(trigger)].replace("▁", "")
            if rel_token in H_REL + V_REL:
                return rel_token, [i + len(trigger)]

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




import re

def make_counterfactual_swap_entities_only(sent):
    """
    Swap the two entities in the description sentence,
    keep the middle question unchanged,
    and flip the final relation token.
    """

    sent = sent.strip().lower()

    # ---- swap entities ----
    # match pattern: "there are two shapes X and Y ."
    match = re.match(r"(there are two shapes )(.+) and (.+?)( \.)", sent)
    if not match:
        return None

    prefix, ent1, ent2, suffix = match.groups()
    swapped_entity_sentence = f"{prefix}{ent2} and {ent1}{suffix}"

    # ---- flip relation token ----
    # match final 'the spatial relation is <rel>'
    rel_pattern = re.search(r"(the spatial relation is )(\w+)( ?\.)$", sent)
    if not rel_pattern:
        return None

    rel_prefix, rel_token, rel_suffix = rel_pattern.groups()
    rel_flipped = REL_FLIP.get(rel_token, rel_token)
    swapped_relation_sentence = f"{rel_prefix}{rel_flipped}{rel_suffix}"

    # ---- extract middle question (everything between entity sentence and relation sentence) ----
    middle_start = match.end()
    middle_end = rel_pattern.start()
    middle_question = sent[middle_start:middle_end].strip()

    # ---- recombine ----
    sent_cf = f"{swapped_entity_sentence} {middle_question} {swapped_relation_sentence}"
    return sent_cf



# =====================
# Main loop
# =====================
results = []

paired_captions = []

for shard_id in range(5):
    caption_path = os.path.join(
        CAPTION_BASE, f"shard{shard_id}", "caption.txt"
    )
    with open(caption_path) as f:
        for line in f:
            sent = line.strip()
            if not sent:
                continue
            cf = make_counterfactual_swap_entities_only(sent)
            if cf is None:
                continue

            paired_captions.append((sent, cf))


def relation_token_to_entities(sent, tokenizer, model, device):
    inputs = tokenizer(sent, return_tensors="pt").to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            use_cache=False
        )

    attn = extract_mean_attention(outputs)  # [L, L]

    # ---- entities ----
    entities = extract_entities(tokens)
    if len(entities) < 2:
        return None

    (e1_pos, e1_name), (e2_pos, e2_name) = entities[:2]

    # ---- relation (only from answer sentence) ----
    rel, rel_pos = find_relation(tokens)
    if not rel_pos:
        return None

    r_to_e1 = pool_attention(attn, rel_pos, [e1_pos])
    r_to_e2 = pool_attention(attn, rel_pos, [e2_pos])

    return {
        "sent": sent,
        "tokens": tokens,
        "e1": e1_name,
        "e2": e2_name,
        "relation": rel,
        "r_to_e1": r_to_e1,
        "r_to_e2": r_to_e2,
        "asym": r_to_e1 - r_to_e2
    }


pair_results = []

for sent1, sent2 in paired_captions:
    r1 = relation_token_to_entities(sent1, tokenizer, model, DEVICE)
    r2 = relation_token_to_entities(sent2, tokenizer, model, DEVICE)

    if r1 is None or r2 is None:
        continue

    # S1: (A, B)
    # S2: (B, A)
    asym_s1 = r1["asym"]
    asym_s2 = r2["asym"]

    # if the relation is explicitly encoded, these two should be approximately negations
    symmetry_score = asym_s1 + asym_s2  # ideal: 0

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

plt.xlabel("Asymmetry S1")
plt.ylabel("Asymmetry S2")
plt.title("Pair-level Attention Asymmetry (LLaVA)")
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig("pair_level_symmetry (LLaVA).png")
plt.close()

example = df_pairs.iloc[0]
print("Example sentences:")
print("S1:", example["sent1"])
print("S2:", example["sent2"])
r1 = relation_token_to_entities(example["sent1"], tokenizer, model, DEVICE)
r2 = relation_token_to_entities(example["sent2"], tokenizer, model, DEVICE)

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
plt.show()
plt.savefig("example_attention_bars.png")
plt.close()
