# Interpreting Attention Mechanisms in Vision-Language Models for Spatial Reasoning

This repository provides code and instructions for analyzing and visualizing attention mechanisms in Vision-Language Models (VLMs) with a focus on spatial relational reasoning. We conduct experiments on both multimodal (vision + language) and text-only settings to study cross-attention, self-attention, and head/layer specialization.

---

## Installation

We evaluate two models:

* **LLaVA-v1.5-7B** ([Hugging Face](https://huggingface.co/liuhaotian/llava-v1.5-7b))
* **Qwen2-VL-7B-Instruct** ([Hugging Face](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct))

Due to differences in dependencies, we recommend setting up **separate virtual environments** for the two models.

### Installation for LLaVA

1. Download the modified LLaVA code from [this repository](https://github.com/zjysteven/VLM-Visualizer/tree/main/models/llava) and place it under `models/llava/`.

   > **Note:** This version is adapted from LLaVA v1.5 to support exporting attention maps from the CLIP vision encoder.

2. Modify the following line in `models/llava/model/multimodal_encoder/clip_encoder.py` (around line 30):

```diff
- self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
+ self.vision_tower = CLIPVisionModel.from_pretrained(
+     self.vision_tower_name,
+     device_map=device_map,
+     attn_implementation="eager",
+     torch_dtype=torch.bfloat16
+ )
```

3. Follow the instructions in `env_setup_llava.bash` to create and activate the LLaVA virtual environment.

### Installation for Qwen2-VL

Here we describe how to set up a minimal Python environment for running **Qwen2-VL** attention analysis using the 🤗 Transformers library.

#### Requirements

- Python **3.9+**
- CUDA-enabled GPU (recommended)
- PyTorch compatible with your CUDA version

#### Install Dependencies

Qwen2-VL requires a recent version of `transformers`.

```bash
pip install "transformers>=4.57.0"
```

Install additional commonly used dependencies:
```bash
pip install torch numpy pillow opencv-python matplotlib
```

⚠️ Make sure your installed PyTorch version matches your CUDA setup.
Refer to: https://pytorch.org/get-started/locally/


---

## Data Preparation

Download the following two datasets:

* **ShapeWorld-based Dataset**: [https://polybox.ethz.ch/index.php/s/6gN7q5LqbpczGdJ](https://polybox.ethz.ch/index.php/s/6gN7q5LqbpczGdJ), visit https://github.com/YimingZhao-art/ShapeWorld to reproduce the dataset.
* **Fixed-position Counterfactual Dataset**: [https://polybox.ethz.ch/index.php/s/pjmxGEPZnEXQTjR](https://polybox.ethz.ch/index.php/s/pjmxGEPZnEXQTjR), or run `bash generate_corner_datasets.sh` to reproduce the dataset.

After downloading, place them under the `data/` directory. The directory structure should look like:

```
data/spatial_twoshapes/agreement/relational/shard0/world-0.png
data/dataset_topleft/images/pair_00000_control.png
```

---

## Running Experiments

### In the LLaVA Environment

1. **Cross-attention and self-attention analysis**:

Set line 581 in `vlm_atten_analysis_llava.py` to your data storage path, such as `BASE = f"/home/maqima/VLM-Visualizer/data/spatial_twoshapes/agreement/relational/shard{shard_id}"`.
```bash
python vlm_atten_analysis_llava.py
```

2. **Text-only attention analysis**:

Set line 36 in `text_only_llm_analysis_llava.py` to your data storage path, such as `CAPTION_BASE = "/home/maqima/VLM-Visualizer/data/spatial_twoshapes/agreement/relational/"`.
```bash
python text_only_llm_analysis_llava.py
```

### In the Qwen2-VL Environment

1. **Cross-attention and self-attention analysis**:

```bash
python vlm_atten_analysis_qwen2.py
```

2. **Text-only attention analysis**:

```bash
python text_only_llm_analysis_qwen2.py
```

3. **Head and layer specialization analysis**:

```bash
bash run_corner_experiments_qwen2.sh
```

---

## Acknowledgements

* **VLM-Visualizer** ([GitHub](https://github.com/zjysteven/VLM-Visualizer))
  A visualization toolkit for inspecting attention maps in LLaVA.

* **LLaVA** ([GitHub](https://github.com/haotian-liu/LLaVA))
  The official implementation of the LLaVA model.

* **attention** ([GitHub](https://github.com/mattneary/attention))
  Code for attention aggregation in large language models, which this project heavily builds upon.
