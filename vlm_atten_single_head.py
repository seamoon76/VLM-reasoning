import os
import torch
import matplotlib.pyplot as plt
import sys
sys.path.append("./models")
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.utils import disable_torch_init
import numpy as np
from utils import load_image

# =========================
# 1. 模型加载
# =========================
model_path = "liuhaotian/llava-v1.5-7b"
device = "cuda" if torch.cuda.is_available() else "cpu"

disable_torch_init()

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path,
    None,
    model_path,
    load_8bit=False,
    load_4bit=False,
    device=device,
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
)

model.eval()

# =========================
# 2. sample & prompt
# =========================
BASE = "/home/maqima/VLM-Visualizer/data/spatial_twoshapes/agreement/relational/test/shard0"
sample_idx = 2
caption_path = f"{BASE}/caption.txt"
captions = [x.strip() for x in open(caption_path).readlines()]
caption = captions[sample_idx]
img_path = f"{BASE}/world-{sample_idx}.png"
prompt_text = caption

# =========================
# 3. 构造输入
# =========================
image = load_image(img_path)
image_tensor, _ = process_images([image], image_processor, model.config)
image_size = image.size

image_tensor = image_tensor.to(model.device, dtype=torch.float16)

inp = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text
conv = conv_templates["llava_v1"].copy()
conv.append_message(conv.roles[0], inp)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
prompt = prompt.replace(
    "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. ",
    ""
)
input_ids = tokenizer_image_token(
    prompt,
    tokenizer,
    IMAGE_TOKEN_INDEX,
    return_tensors="pt",
).unsqueeze(0).to(model.device)

input_token_len = input_ids.shape[1]
print("Input token length:", input_token_len)
# =========================
# 4. forward（只生成 1 token）
# =========================
with torch.inference_mode():
    outputs = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=[image_size],
        do_sample=False,
        max_new_tokens=1,
        return_dict_in_generate=True,
        output_attentions=True,
    )

# =========================
# 5. token & vision token 区间（只在 input 内）
# =========================
input_token_ids = input_ids[0].tolist()

input_tokens = []
for tid in input_token_ids:
    if tid == IMAGE_TOKEN_INDEX:
        input_tokens.append("<image>")
    else:
        input_tokens.append(tokenizer.convert_ids_to_tokens(tid))

image_token_idx = input_tokens.index("<image>")
assistant_start = None
for i in range(len(input_tokens) - 4):
    if input_tokens[i:i+5] == ['▁A', 'SS', 'IST', 'ANT', ':']:
        assistant_start = i
        break
query_token_start = image_token_idx + 2
query_token_end   = assistant_start
query_token_indices = list(range(query_token_start, query_token_end))
query_tokens = input_tokens[query_token_start:query_token_end]
print("Query tokens:", query_tokens)

vision_token_start = len(
    tokenizer(prompt.split("<image>")[0], return_tensors="pt")["input_ids"][0]
)
vision_token_end = vision_token_start + model.get_vision_tower().num_patches

num_layers = len(outputs["attentions"][0])
num_heads = outputs["attentions"][0][0].shape[1]

#print(f"Input tokens: {len(input_tokens)}")
print(f"Vision tokens: [{vision_token_start}, {vision_token_end})")
print(f"Layers={num_layers}, Heads={num_heads}")

# =========================
# 6. 保存目录
# =========================
save_dir = str(sample_idx) + "_headwise_input_token_to_vision_curves"
os.makedirs(save_dir, exist_ok=True)

last_k_layers = 4

# =========================
# 7. 核心：只遍历 input tokens
# =========================
for layer_idx in range(num_layers - last_k_layers, num_layers):
    # shape: [heads, seq_len, seq_len]
    layer_attn = outputs["attentions"][0][layer_idx][0]

    for head_idx in range(num_heads):
        attn = layer_attn[head_idx]

        query_to_vision = []
        for tok_i in query_token_indices:
            score = attn[tok_i, vision_token_start:vision_token_end].sum().item()
            query_to_vision.append(score)

        query_to_vision = np.array(query_to_vision)
        query_to_vision_norm = query_to_vision / (query_to_vision.sum() + 1e-6)

        # =========================
        # 绘图
        # =========================
        plt.figure(figsize=(20, 4))
        plt.plot(query_to_vision_norm, linewidth=2)
        plt.xticks(
            range(len(query_tokens)),
            [t.replace("▁", "") for t in query_tokens],
            rotation=70,
            fontsize=10,
        )

        plt.ylabel("Attention sum to vision tokens")
        plt.xlabel("Input query token")
        plt.title(f"Layer {layer_idx} | Head {head_idx}")

        plt.tight_layout()

        fname = f"layer{layer_idx}_head{head_idx}.png"
        plt.savefig(os.path.join(save_dir, fname), dpi=150)
        plt.close()

print("Done. Plots saved to:", save_dir)
