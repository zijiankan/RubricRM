# RubricRM: Generative Reward Modeling via Dynamic Rubrics for Image Generation and Editing

Official repository for the paper [*RubricRM: Generative Reward Modeling via Dynamic Rubrics for Image Generation and Editing*](https://arxiv.org/abs/2608.26956).

[![arXiv](https://img.shields.io/badge/arXiv-2608.26956-b31b1b.svg)](https://arxiv.org/abs/2608.26956)

**Accepted to EMNLP 2026 Main Conference.**

## Models and Data

RubricRM provides four checkpoints for evaluating text-to-image generation and image editing at two model scales.

| Task | 4B | 9B |
| --- | --- | --- |
| Text-to-image generation | [![RubricRM-Gen-4B](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-RubricRM--Gen--4B-FFD21E)](https://huggingface.co/skylenage-ai/SkyJM-Gen-4B) | [![RubricRM-Gen-9B](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-RubricRM--Gen--9B-FFD21E)](https://huggingface.co/skylenage-ai/SkyJM-Gen-9B) |
| Image editing | [![RubricRM-Edit-4B](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-RubricRM--Edit--4B-FFD21E)](https://huggingface.co/skylenage-ai/SkyJM-Edit-4B) | [![RubricRM-Edit-9B](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-RubricRM--Edit--9B-FFD21E)](https://huggingface.co/skylenage-ai/SkyJM-Edit-9B) |

The training data used by RubricRM are available at [![RubricRM-Data](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-RubricRM--Data-FFD21E)](https://huggingface.co/datasets/skylenage-ai/RubricRM-Data)

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

## Citation

If you find RubricRM useful in your research, please cite our paper:

```bibtex
@article{kan2026rubricrm,
  title   = {RubricRM: Generative Reward Modeling via Dynamic Rubrics for Image Generation and Editing},
  author  = {Kan, Zijian and Wang, Wei and Luo, Long and Zhao, Bing and Ren, Xuan and Qiao, Weixu and Li, Wenbo and Wei, Hu and Qu, Lin},
  journal = {arXiv preprint arXiv:2608.26956},
  year    = {2026},
  doi     = {10.48550/arXiv.2608.26956},
  url     = {https://arxiv.org/abs/2608.26956}
}
```
