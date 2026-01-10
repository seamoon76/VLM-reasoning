
# Interpreting Attention Mechanisms in Vision-Language Models for Spatial Reasoning

## Installation
We have two models: [LLaVA-v1.5-7B](https://huggingface.co/liuhaotian/llava-v1.5-7b) and [QWen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct). We need to prepare two environments for them.

### Installation for LLaVA
1. Download LLaVA model code from [this link](https://github.com/zjysteven/VLM-Visualizer/tree/main/models/llava) as `models/llava/`. This is a modified version of LLaVA v1.5 that can output the attention of CLIP vision encoder. Make a modification as below in `models/llava/model/multimodal_encoder/clip_encoder.py`, line 30:
```
- self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
+ self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map,attn_implementation="eager",torch_dtype=torch.bfloat16)
```
2. Follow the `env_setup_llava.bash` to prepare a virtual environment for LLaVA.

### Installation for QWen

## Data preparition
Download two datasets from below links: [ShapeWorld-based Dataset](https://polybox.ethz.ch/index.php/s/6gN7q5LqbpczGdJ) and [Fixed-position Counterfactual Dataset](https://polybox.ethz.ch/index.php/s/KpLcHJJSyexQqcQ), put them under `data` directory. It should looks like: `data/spatial_twoshapes/agreement/relational/shard0/world-0.png`
and 
`data/dataset_sample/images/pair_00000_control.png`.

## Run Commands
### In LLaVA Env
1. cross- and self- attention analysis:
```
python vlm_atten_analysis_llava.py
```
2. text-only attention analysis
```
python text_only_llm_analysis_llava.py
```
### In Qwen Env
1. cross- and self- attention analysis:
```
python vlm_atten_analysis_qwen2.py
```
2. text-only attention analysis
```
python text_only_llm_analysis_qwen2.py
```
3. head/layer specialization structure analysis
```
bash run_corner_experiments.sh
```

## Acknowledgements
[VLM-VISULALIZER](https://github.com/zjysteven/VLM-Visualizer): A tool to visualize the attention of LLaVA.

[LLaVA](https://github.com/haotian-liu/LLaVA): The official implementation of the LLaVA model.

[attention](https://github.com/mattneary/attention): Attention aggregation for LLMs is heavily borrowed from this repo.
