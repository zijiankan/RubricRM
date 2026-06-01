#!/usr/bin/env python3
"""Merge RL-trained language model weights with original visual encoder weights.

The verl model_merger only exports language model weights (FSDP-wrapped parts).
This script restores the visual encoder from the original pretrained model.

Usage:
    python scripts/merge_with_visual.py \
        --original rl_model \
        --merged /path/to/merged_model \
        --output /path/to/final_model
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser(description="Merge RL weights with visual encoder from original model")
    parser.add_argument("--original", required=True, help="Path to original pretrained model (with visual encoder)")
    parser.add_argument("--merged", required=True, help="Path to model_merger output (language model only)")
    parser.add_argument("--output", required=True, help="Path to save the complete model")
    args = parser.parse_args()

    original_dir = Path(args.original)
    merged_dir = Path(args.merged)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load visual weights from original model
    print("Loading visual weights from original model...")
    visual_state_dict = {}
    original_index_file = original_dir / "model.safetensors.index.json"

    if original_index_file.exists():
        # Sharded model - find which files contain visual weights
        with open(original_index_file) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        visual_files = set()
        for key, filename in weight_map.items():
            if key.startswith("visual."):
                visual_files.add(filename)

        for filename in sorted(visual_files):
            filepath = original_dir / filename
            print(f"  Loading {filename}...")
            shard = load_file(str(filepath))
            for key, tensor in shard.items():
                if key.startswith("visual."):
                    visual_state_dict[key] = tensor
    else:
        # Single file model
        shard = load_file(str(original_dir / "model.safetensors"))
        for key, tensor in shard.items():
            if key.startswith("visual."):
                visual_state_dict[key] = tensor

    print(f"  Found {len(visual_state_dict)} visual parameters")

    # Step 2: Load merged (RL-trained) weights
    print("Loading merged RL-trained weights...")
    merged_state_dict = {}
    merged_index_file = merged_dir / "model.safetensors.index.json"

    if merged_index_file.exists():
        with open(merged_index_file) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        loaded_files = set()
        for key, filename in weight_map.items():
            if filename not in loaded_files:
                loaded_files.add(filename)
                shard = load_file(str(merged_dir / filename))
                merged_state_dict.update(shard)
    else:
        merged_state_dict = load_file(str(merged_dir / "model.safetensors"))

    print(f"  Found {len(merged_state_dict)} merged parameters")

    # Check overlap
    overlap = set(visual_state_dict.keys()) & set(merged_state_dict.keys())
    if overlap:
        print(f"  WARNING: {len(overlap)} overlapping visual keys in merged model (will use original)")

    # Step 3: Combine - merged weights take priority, visual from original fills gaps
    print("Combining weights...")
    combined = {}
    combined.update(visual_state_dict)  # visual first
    combined.update(merged_state_dict)  # RL-trained overwrites any overlap

    print(f"  Total parameters: {len(combined)}")
    visual_count = sum(1 for k in combined if k.startswith("visual."))
    lang_count = len(combined) - visual_count
    print(f"  Visual: {visual_count}, Language: {lang_count}")

    # Step 4: Save as sharded safetensors
    print(f"Saving to {output_dir}...")

    # Save in shards (~4GB each)
    max_shard_size = 4 * 1024 * 1024 * 1024  # 4GB
    shards = []
    current_shard = {}
    current_size = 0

    for key in sorted(combined.keys()):
        tensor = combined[key]
        tensor_size = tensor.nelement() * tensor.element_size()
        if current_size + tensor_size > max_shard_size and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
        current_shard[key] = tensor
        current_size += tensor_size

    if current_shard:
        shards.append(current_shard)

    weight_map = {}
    total_size = 0

    if len(shards) == 1:
        # Single file
        filename = "model.safetensors"
        save_file(shards[0], str(output_dir / filename))
        for key in shards[0]:
            weight_map[key] = filename
            total_size += shards[0][key].nelement() * shards[0][key].element_size()
    else:
        # Multiple shards
        for i, shard in enumerate(shards):
            filename = f"model-{i+1:05d}-of-{len(shards):05d}.safetensors"
            save_file(shard, str(output_dir / filename))
            for key in shard:
                weight_map[key] = filename
                total_size += shard[key].nelement() * shard[key].element_size()

        # Save index
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map,
        }
        with open(output_dir / "model.safetensors.index.json", "w") as f:
            json.dump(index, f, indent=2)

    # Step 5: Copy config files from merged model (has correct tokenizer, config, etc.)
    print("Copying config files...")
    config_files = [
        "config.json", "generation_config.json", "tokenizer_config.json",
        "tokenizer.json", "vocab.json", "merges.txt", "special_tokens_map.json",
        "preprocessor_config.json", "chat_template.jinja",
    ]
    # Copy from merged first, then fill missing from original
    for fname in config_files:
        src_merged = merged_dir / fname
        src_original = original_dir / fname
        dst = output_dir / fname
        if not dst.exists():
            if src_merged.exists():
                shutil.copy2(str(src_merged), str(dst))
            elif src_original.exists():
                shutil.copy2(str(src_original), str(dst))

    print("Done!")
    print(f"Output model: {output_dir}")
    print(f"  {len(combined)} parameters, {len(shards)} shard(s)")


if __name__ == "__main__":
    main()
