import argparse
import os
import sys
from typing import Dict, Iterable, List

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append("./models")

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from llava.utils import disable_torch_init

from dataset_utils import load_dataset


MODEL_PATH = "liuhaotian/llava-v1.5-7b"


def get_conv_mode(model_name: str) -> str:
    if "llama-2" in model_name.lower():
        return "llava_llama_2"
    if "mistral" in model_name.lower():
        return "mistral_instruct"
    if "v1.6-34b" in model_name.lower():
        return "chatml_direct"
    if "v1" in model_name.lower():
        return "llava_v1"
    if "mpt" in model_name.lower():
        return "mpt"
    return "llava_v0"


def build_prompt(prompt_text: str, model, model_name: str) -> str:
    conv_mode = get_conv_mode(model_name)
    conv = conv_templates[conv_mode].copy()
    if model.config.mm_use_im_start_end:
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt_text
    else:
        inp = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    prompt = prompt.replace(
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions. ",
        ""
    )
    return prompt


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


def load_llava_model(device: str, load_8bit: bool, load_4bit: bool):
    model_name = get_model_name_from_path(MODEL_PATH)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        MODEL_PATH,
        None,
        model_name,
        load_8bit=load_8bit,
        load_4bit=load_4bit,
        device=device,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    return tokenizer, model, image_processor, model_name


def extract_head_scores_for_image(
    image,
    prompt_text: str,
    model,
    tokenizer,
    image_processor,
    model_name: str,
    layer_indices: Iterable[int],
    max_new_tokens: int,
    corner: str | None = None,
    patch_size: int | None = None,
) -> Dict[int, torch.Tensor]:
    prompt = build_prompt(prompt_text, model, model_name)
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(model.device)

    image_tensor, images = process_images([image], image_processor, model.config)
    image = images[0]
    image_size = image.size

    if isinstance(image_tensor, list):
        image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    vision_token_start = len(
        tokenizer(prompt.split("<image>")[0], return_tensors="pt")["input_ids"][0]
    )
    num_patches = model.get_vision_tower().num_patches
    vision_token_end = vision_token_start + num_patches
    grid_size = model.get_vision_tower().num_patches_per_side
    if corner is not None:
        if patch_size is None or patch_size <= 0:
            raise ValueError("--patch-size must be > 0 when --corner is set")
        if patch_size > grid_size:
            raise ValueError("patch_size exceeds grid size")
        if corner == "topleft":
            rows = range(0, patch_size)
            cols = range(0, patch_size)
        elif corner == "topright":
            rows = range(0, patch_size)
            cols = range(grid_size - patch_size, grid_size)
        elif corner == "bottomleft":
            rows = range(grid_size - patch_size, grid_size)
            cols = range(0, patch_size)
        elif corner == "bottomright":
            rows = range(grid_size - patch_size, grid_size)
            cols = range(grid_size - patch_size, grid_size)
        else:
            raise ValueError(f"Unknown corner: {corner}")
        patch_indices = [
            vision_token_start + r * grid_size + c for r in rows for c in cols
        ]
        patch_indices = torch.tensor(patch_indices, dtype=torch.long)
    else:
        patch_indices = None

    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
            output_attentions=True,
        )

    num_steps = len(outputs["attentions"])
    if num_steps == 0:
        raise ValueError("No attentions returned from generate")

    num_heads = outputs["attentions"][0][0].shape[1]
    scores: Dict[int, torch.Tensor] = {
        layer_idx: torch.zeros(num_heads) for layer_idx in layer_indices
    }

    for step_attn in outputs["attentions"]:
        for layer_idx in layer_indices:
            layer_attn = step_attn[layer_idx][0]
            if patch_indices is None:
                step_scores = layer_attn[:, -1, vision_token_start:vision_token_end].sum(dim=-1)
            else:
                step_scores = layer_attn[:, -1, patch_indices].sum(dim=-1)
            scores[layer_idx] += step_scores.to("cpu")

    for layer_idx in layer_indices:
        scores[layer_idx] = scores[layer_idx] / num_steps

    return scores


def summarize_diff_gaps(diff_abs: np.ndarray) -> dict:
    # diff_abs shape: [num_pairs, num_heads]
    sorted_vals = np.sort(diff_abs, axis=1)
    top1 = sorted_vals[:, -1]
    top2 = sorted_vals[:, -2]
    top3 = sorted_vals[:, -3]
    top4 = sorted_vals[:, -4] if diff_abs.shape[1] >= 4 else sorted_vals[:, 0]
    median = np.median(sorted_vals, axis=1)
    return {
        "top1_mean": float(top1.mean()),
        "top2_mean": float(top2.mean()),
        "top3_mean": float(top3.mean()),
        "top4_mean": float(top4.mean()),
        "median_mean": float(median.mean()),
        "top1_minus_median_mean": float((top1 - median).mean()),
        "top1_minus_top4_mean": float((top1 - top4).mean()),
        "top1_minus_top2_mean": float((top1 - top2).mean()),
    }


def run_stats(
    dataset: str,
    prompt: str,
    layers: str,
    max_pairs: int | None,
    max_new_tokens: int,
    out_dir: str,
    save_diffs: bool = True,
    model_bundle=None,
    load_8bit: bool = False,
    load_4bit: bool = False,
    corner: str | None = None,
    patch_size: int | None = None,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    disable_torch_init()

    if model_bundle is None:
        tokenizer, model, image_processor, model_name = load_llava_model(
            device=device, load_8bit=load_8bit, load_4bit=load_4bit
        )
    else:
        tokenizer, model, image_processor, model_name = model_bundle

    num_layers = get_num_layers(model)
    layer_indices = parse_layers(layers, num_layers)

    entries = load_dataset(dataset)
    if max_pairs is not None:
        entries = entries[: max_pairs]

    if not entries:
        raise ValueError("Dataset is empty")

    os.makedirs(out_dir, exist_ok=True)

    counts: Dict[int, np.ndarray] = {}
    diff_abs_by_layer: Dict[int, List[np.ndarray]] = {}
    num_heads = None
    for layer_idx in layer_indices:
        counts[layer_idx] = None
        diff_abs_by_layer[layer_idx] = []

    for i, record in enumerate(entries):
        full_image = record["full_image"]
        ctrl_image = record["control_image"]

        full_scores = extract_head_scores_for_image(
            full_image,
            prompt,
            model,
            tokenizer,
            image_processor,
            model_name,
            layer_indices,
            max_new_tokens,
            corner=corner,
            patch_size=patch_size,
        )
        ctrl_scores = extract_head_scores_for_image(
            ctrl_image,
            prompt,
            model,
            tokenizer,
            image_processor,
            model_name,
            layer_indices,
            max_new_tokens,
            corner=corner,
            patch_size=patch_size,
        )

        if num_heads is None:
            num_heads = len(next(iter(full_scores.values())))
            for layer_idx in layer_indices:
                counts[layer_idx] = np.zeros(num_heads, dtype=int)

        for layer_idx in layer_indices:
            diff = (full_scores[layer_idx] - ctrl_scores[layer_idx]).numpy()
            diff_abs = np.abs(diff)
            top3 = np.argsort(diff_abs)[-3:]
            counts[layer_idx][top3] += 1
            if save_diffs:
                diff_abs_by_layer[layer_idx].append(diff_abs)

        print(f"Processed {i + 1}/{len(entries)}: {record['id']}")

    for layer_idx in layer_indices:
        layer_counts = counts[layer_idx]
        layer_id = layer_idx + 1
        fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
        ax.bar(np.arange(num_heads), layer_counts)
        ax.set_title(f"Layer {layer_id} head top3 counts")
        ax.set_xlabel("Head index")
        ax.set_ylabel("Top3 count")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"layer_{layer_id}_top3_counts.png"))
        plt.close(fig)

        np.save(os.path.join(out_dir, f"layer_{layer_id}_top3_counts.npy"), layer_counts)
        if save_diffs:
            diff_abs = np.stack(diff_abs_by_layer[layer_idx], axis=0)
            np.save(os.path.join(out_dir, f"layer_{layer_id}_diff_abs.npy"), diff_abs)
            stats = summarize_diff_gaps(diff_abs)
            with open(os.path.join(out_dir, f"layer_{layer_id}_diff_gap_stats.txt"), "w") as f:
                for key, value in stats.items():
                    f.write(f"{key}={value:.6f}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="./dataset_sample")
    parser.add_argument("--prompt", default="Is there anything in the top left corner?")
    parser.add_argument("--layers", default="30")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=3)
    parser.add_argument("--out-dir", default="head_diff_stats")
    parser.add_argument("--save-diffs", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument(
        "--corner",
        choices=["topleft", "topright", "bottomleft", "bottomright"],
        default=None,
        help="Restrict attention to a corner patch grid.",
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
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        corner=args.corner,
        patch_size=args.patch_size,
    )


if __name__ == "__main__":
    main()
