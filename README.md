# RubricRM: Generative Reward Modeling via Dynamic Rubrics for Image Generation and Editing

Official repository for the paper *RubricRM: Generative Reward Modeling via Dynamic Rubrics for Image Generation and Editing*. 

<p align="center">
  <img src="assets/main_figure.png" width="100%">
</p>

## Quick Start

### Installation

**Inference** (lightweight, vLLM-based):

```bash
pip install vllm transformers pillow numpy qwen-vl-utils
```

**Training** (verl framework with FSDP / Megatron-LM):

```bash
cd verl
pip install -e .
pip install -r requirements.txt
```
### Inference

```bash
cd inference
bash run_inference.sh
```

### Grading

```bash
# A/B preference accuracy
python inference/grade_single.py output/result.jsonl

# EditScore-Bench (multi-dimension)
python inference/grade_editscore_bench.py output/editscore_result.jsonl

# EditReward-Bench (2/3/4-pair)
python inference/grade_reward_bench.py output/editreward_result.jsonl
```

## Training

```bash
cd verl
bash train.sh
```
