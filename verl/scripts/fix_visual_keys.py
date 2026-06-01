#!/usr/bin/env python3
"""Fix visual encoder key prefix in model_merger output.

model_merger incorrectly saves visual weights as:
    model.language_model.visual.* 
but HuggingFace expects:
    model.visual.*

This script renames the keys in-place (or to a new directory).

Usage:
    python scripts/fix_visual_keys.py /path/to/merged_model
    python scripts/fix_visual_keys.py /path/to/merged_model --output /path/to/fixed_model
"""

import argparse
import json
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file


def fix_key(key: str) -> str:
    """Rename model.language_model.visual.* -> model.visual.*"""
    if key.startswith("model.language_model.visual."):
        return "model.visual." + key[len("model.language_model.visual."):]
    return key


def main():
    parser = argparse.ArgumentParser(description="Fix visual encoder key prefix in model_merger output")
    parser.add_argument("model_dir", help="Path to model_merger output directory")
    parser.add_argument("--output", default=None, help="Output directory (default: overwrite in-place)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output) if args.output else model_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all safetensors files
    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        print(f"Error: no .safetensors files found in {model_dir}")
        return

    renamed_count = 0
    total_keys = 0

    for st_file in st_files:
        print(f"Processing {st_file.name}...")
        sd = load_file(str(st_file))

        new_sd = {}
        file_renamed = 0
        for key, tensor in sd.items():
            new_key = fix_key(key)
            if new_key != key:
                file_renamed += 1
            new_sd[new_key] = tensor

        total_keys += len(new_sd)
        renamed_count += file_renamed

        out_path = output_dir / st_file.name
        save_file(new_sd, str(out_path))
        print(f"  {len(new_sd)} keys, {file_renamed} renamed")

    # Fix index.json if it exists
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as f:
            index = json.load(f)
        new_weight_map = {}
        for key, filename in index["weight_map"].items():
            new_weight_map[fix_key(key)] = filename
        index["weight_map"] = new_weight_map
        with open(output_dir / "model.safetensors.index.json", "w") as f:
            json.dump(index, f, indent=2)
        print("Fixed model.safetensors.index.json")

    # Copy other config files if output is different dir
    if output_dir != model_dir:
        for f in model_dir.iterdir():
            if f.suffix != ".safetensors" and f.name != "model.safetensors.index.json":
                dst = output_dir / f.name
                if not dst.exists():
                    shutil.copy2(str(f), str(dst))

    print(f"\nDone! Renamed {renamed_count}/{total_keys} keys")
    print(f"  model.language_model.visual.* -> model.visual.*")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
