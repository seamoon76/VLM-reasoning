import argparse
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from dataset_utils import load_dataset


MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"


def parse_layers(layers_arg: str, num_layers: int) -> List[int]:
    if layers_arg.strip().lower() == "all":
        return list(range(num_layers))
    layers: List[int] = []
    for part in layers_arg.split(","):
        part = part.strip()
        if not part:
            continue
        layer_one_based = int(part)
        if layer_one_based < 1 or layer_one_based > num_layers:
            raise ValueError(f"Layer index {layer_one_based} out of range 1..{num_layers}")
        layers.append(layer_one_based - 1)
    if not layers:
        raise ValueError("No layers parsed from --layers")
    return layers


def get_num_layers(model) -> int:
    if hasattr(model.config, "num_hidden_layers"):
        return int(model.config.num_hidden_layers)
    if hasattr(model.config, "n_layer"):
        return int(model.config.n_layer)
    raise ValueError("Cannot infer number of layers from model config")


def find_vision_span(tokens_raw: List[str]) -> Tuple[int, int]:
    vs = tokens_raw.index("<|vision_start|>")
    ve = tokens_raw.index("<|vision_end|>")
    if ve <= vs + 1:
        raise RuntimeError("Vision span is empty")
    return vs, ve


def guess_grid_from_n(n: int) -> Tuple[int, int]:
    s = int(round(np.sqrt(n)))
    if s * s == n:
        return (s, s)
    best = None
    for a in range(1, int(np.sqrt(n)) + 1):
        if n % a == 0:
            b = n // a
            if best is None or abs(a - b) < abs(best[0] - best[1]):
                best = (a, b)
    return best if best is not None else (1, n)


def build_patch_indices_for_corner(
    vision_token_start: int,
    gh: int,
    gw: int,
    corner: str,
    patch_size: int,
) -> torch.Tensor:
    if patch_size <= 0:
        raise ValueError("--patch-size must be > 0 when --corner is set")
    if patch_size > gh or patch_size > gw:
        raise ValueError("patch_size exceeds grid size")

    if corner == "topleft":
        rows = range(0, patch_size)
        cols = range(0, patch_size)
    elif corner == "topright":
        rows = range(0, patch_size)
        cols = range(gw - patch_size, gw)
    elif corner == "bottomleft":
        rows = range(gh - patch_size, gh)
        cols = range(0, patch_size)
    elif corner == "bottomright":
        rows = range(gh - patch_size, gh)
        cols = range(gw - patch_size, gw)
    else:
        raise ValueError(f"Unknown corner: {corner}")

    patch_indices = [vision_token_start + r * gw + c for r in rows for c in cols]
    return torch.tensor(patch_indices, dtype=torch.long)


def load_qwen2_model(device: str, load_4bit: bool, load_8bit: bool):
    if load_4bit and load_8bit:
        raise ValueError("Choose at most one of --load-4bit or --load-8bit")

    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="auto" if device.startswith("cuda") else None,
    )
    if load_8bit:
        model_kwargs["load_in_8bit"] = True
    if load_4bit:
        model_kwargs["load_in_4bit"] = True

    model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, **model_kwargs)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()
    return model, processor


def extract_head_scores_for_image_qwen2(
    image,
    prompt_text: str,
    model,
    processor,
    layer_indices: Iterable[int],
    max_new_tokens: int,
    corner: Optional[str] = None,
    patch_size: Optional[int] = None,
) -> Dict[int, torch.Tensor]:
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": prompt_text}],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    for k, v in list(inputs.items()):
        if torch.is_tensor(v):
            inputs[k] = v.to(model.device)

    input_ids = inputs["input_ids"][0].detach().cpu()
    tokens_raw = processor.tokenizer.convert_ids_to_tokens(input_ids.tolist())

    vs, ve = find_vision_span(tokens_raw)
    vision_token_start = vs + 1
    vision_token_end = ve
    nvision = vision_token_end - vision_token_start
    gh, gw = guess_grid_from_n(nvision)

    if corner is not None:
        if patch_size is None:
            raise ValueError("--patch-size must be set when --corner is set")
        patch_indices = build_patch_indices_for_corner(
            vision_token_start=vision_token_start,
            gh=gh,
            gw=gw,
            corner=corner,
            patch_size=patch_size,
        )
    else:
        patch_indices = None

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
            output_attentions=True,
        )

    step_attns = outputs.attentions
    num_steps = len(step_attns) if step_attns is not None else 0
    if num_steps == 0:
        raise ValueError("No attentions returned from generate")

    first_layer = next(iter(layer_indices))
    t = step_attns[0][first_layer]
    if t.dim() == 4:
        num_heads = t.shape[1]
    elif t.dim() == 3:
        num_heads = t.shape[0]
    else:
        raise RuntimeError(f"Unexpected attention tensor shape: {t.shape}")

    scores: Dict[int, torch.Tensor] = {layer_idx: torch.zeros(num_heads) for layer_idx in layer_indices}

    for step in range(num_steps):
        per_layer = step_attns[step]
        for layer_idx in layer_indices:
            layer_attn = per_layer[layer_idx]
            if layer_attn.dim() == 4:
                a = layer_attn[0]  # [heads, seq, seq]
            elif layer_attn.dim() == 3:
                a = layer_attn
            else:
                raise RuntimeError(f"Unexpected layer_attn dim={layer_attn.dim()} shape={layer_attn.shape}")

            if patch_indices is None:
                step_scores = a[:, -1, vision_token_start:vision_token_end].sum(dim=-1)
            else:
                step_scores = a[:, -1, patch_indices].sum(dim=-1)

            scores[layer_idx] += step_scores.detach().to("cpu")

    for layer_idx in layer_indices:
        scores[layer_idx] = scores[layer_idx] / float(num_steps)

    return scores


def run_stats(
    dataset: str,
    prompt: str,
    layers: str,
    max_pairs: Optional[int],
    max_new_tokens: int,
    out_dir: str,
    save_diffs: bool = True,   # 这里保留参数名，但我们用于“保存 count cube”
    topk: int = 3,             # NEW: topk for counting
    model_bundle=None,
    load_8bit: bool = False,
    load_4bit: bool = False,
    corner: Optional[str] = None,
    patch_size: Optional[int] = None,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_bundle is None:
        model, processor = load_qwen2_model(device=device, load_4bit=load_4bit, load_8bit=load_8bit)
    else:
        model, processor = model_bundle

    num_layers = get_num_layers(model)
    layer_indices = parse_layers(layers, num_layers)
    L = len(layer_indices)

    entries = load_dataset(dataset)
    if max_pairs is not None:
        entries = entries[:max_pairs]
    if not entries:
        raise ValueError("Dataset is empty")
    N = len(entries)

    os.makedirs(out_dir, exist_ok=True)

    # legacy: per-layer total counts [H]
    counts_by_layer: Dict[int, np.ndarray] = {layer_idx: None for layer_idx in layer_indices}

    # NEW: count cube [L,H,N] with 0/1 membership for each sample
    count_LHN: Optional[np.ndarray] = None
    num_heads: Optional[int] = None

    for i, record in enumerate(entries):
        full_image = record["full_image"]
        ctrl_image = record["control_image"]

        full_scores = extract_head_scores_for_image_qwen2(
            full_image, prompt, model, processor, layer_indices, max_new_tokens,
            corner=corner, patch_size=patch_size,
        )
        ctrl_scores = extract_head_scores_for_image_qwen2(
            ctrl_image, prompt, model, processor, layer_indices, max_new_tokens,
            corner=corner, patch_size=patch_size,
        )

        if num_heads is None:
            num_heads = len(next(iter(full_scores.values())))
            for layer_idx in layer_indices:
                counts_by_layer[layer_idx] = np.zeros(num_heads, dtype=int)
            if save_diffs:
                count_LHN = np.zeros((L, num_heads, N), dtype=np.uint8)

        # compute topk membership per layer for THIS sample
        for li, layer_idx in enumerate(layer_indices):
            diff = (full_scores[layer_idx] - ctrl_scores[layer_idx]).numpy()
            diff_abs = np.abs(diff)

            topk_idx = np.argsort(diff_abs)[-topk:]  # indices of topk heads

            # legacy total counts
            counts_by_layer[layer_idx][topk_idx] += 1

            # NEW per-sample membership
            if save_diffs:
                count_LHN[li, topk_idx, i] = 1

        print(f"Processed {i + 1}/{len(entries)}: {record.get('id', i)}")

    # Save legacy per-layer barplots + npy (unchanged)
    for layer_idx in layer_indices:
        layer_counts = counts_by_layer[layer_idx]
        layer_id = layer_idx + 1
        fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
        ax.bar(np.arange(num_heads), layer_counts)
        ax.set_title(f"Layer {layer_id} head top{topk} counts")
        ax.set_xlabel("Head index")
        ax.set_ylabel(f"Top{topk} count")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"layer_{layer_id}_top{topk}_counts.png"))
        plt.close(fig)

        np.save(os.path.join(out_dir, f"layer_{layer_id}_top{topk}_counts.npy"), layer_counts)

    # Save NEW cube + aggregated LH matrix
    if save_diffs:
        assert count_LHN is not None
        np.save(os.path.join(out_dir, "count_LHN.npy"), count_LHN)  # [L,H,N] 0/1

        count_LH = count_LHN.sum(axis=2).astype(np.int32)  # [L,H] total counts
        np.save(os.path.join(out_dir, "count_LH.npy"), count_LH)

        with open(os.path.join(out_dir, "count_shape.txt"), "w") as f:
            f.write(f"count_LHN shape [L,H,N] = {count_LHN.shape}\n")
            f.write(f"L={L}, H={num_heads}, N={N}\n")
            f.write(f"topk={topk}\n")
            f.write(f"layer_indices (0-based) = {layer_indices}\n")

        print(f"[saved] {os.path.join(out_dir, 'count_LHN.npy')} shape={count_LHN.shape}")
        print(f"[saved] {os.path.join(out_dir, 'count_LH.npy')} shape={count_LH.shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="./dataset_sample")
    parser.add_argument("--prompt", default="Is there anything in the top left corner?")
    parser.add_argument("--layers", default="30")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=3)
    parser.add_argument("--out-dir", default="head_diff_stats_qwen2")
    parser.add_argument("--save-diffs", action="store_true", help="Save count_LHN.npy and count_LH.npy")
    parser.add_argument("--topk", type=int, default=3, help="Count heads that fall into top-k per sample (default=3)")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument(
        "--corner",
        choices=["topleft", "topright", "bottomleft", "bottomright"],
        default=None,
    )
    parser.add_argument("--patch-size", type=int, default=None)
    args = parser.parse_args()

    run_stats(
        dataset=args.dataset,
        prompt=args.prompt,
        layers=args.layers,
        max_pairs=args.max_pairs,
        max_new_tokens=args.max_new_tokens,
        out_dir=args.out_dir,
        save_diffs=args.save_diffs,
        topk=args.topk,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        corner=args.corner,
        patch_size=args.patch_size,
    )


if __name__ == "__main__":
    main()
