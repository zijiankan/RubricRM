#!/usr/bin/env python3
"""Inspect safetensors weight keys and shapes.

Usage:
    python scripts/inspect_weights.py /path/to/model_dir
    python scripts/inspect_weights.py /path/to/model.safetensors
"""

import sys
import collections
from pathlib import Path

from safetensors.torch import load_file


def inspect(path: str):
    p = Path(path)
    files = []
    if p.is_dir():
        files = sorted(p.glob("*.safetensors"))
    elif p.suffix == ".safetensors":
        files = [p]
    else:
        print(f"Error: {path} is not a .safetensors file or directory")
        return

    all_keys = {}
    for f in files:
        print(f"Loading {f.name} ({f.stat().st_size / 1e9:.2f} GB)...")
        sd = load_file(str(f))
        for k, v in sd.items():
            all_keys[k] = (v.shape, v.dtype, f.name)

    print(f"\nTotal keys: {len(all_keys)}")

    # Prefix distribution
    prefixes = collections.Counter(k.split(".")[0] for k in all_keys)
    print(f"\nPrefix distribution:")
    for prefix, count in sorted(prefixes.items(), key=lambda x: -x[1]):
        print(f"  {prefix}: {count}")

    # Visual keys
    visual_keys = [k for k in all_keys if "visual" in k.lower()]
    print(f"\nVisual-related keys: {len(visual_keys)}")
    for k in visual_keys[:20]:
        shape, dtype, fname = all_keys[k]
        print(f"  {k}: {list(shape)} ({dtype})")

    # All keys summary
    print(f"\nAll keys:")
    for k in sorted(all_keys):
        shape, dtype, fname = all_keys[k]
        print(f"  {k}: {list(shape)} ({dtype})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_weights.py /path/to/model_or_safetensors")
        sys.exit(1)
    inspect(sys.argv[1])
