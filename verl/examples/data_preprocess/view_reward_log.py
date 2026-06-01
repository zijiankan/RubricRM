#!/usr/bin/env python3
"""
Viewer for T2I reward JSONL log files (compact format).

Usage:
    python view_reward_log.py logs/reward/*.jsonl               # summary
    python view_reward_log.py logs/reward/*.jsonl --head 5       # first 5 records
    python view_reward_log.py logs/reward/*.jsonl --index 42     # specific sample
    python view_reward_log.py logs/reward/*.jsonl --filter wrong # reward==0 only
    python view_reward_log.py logs/reward/*.jsonl --filter v2    # v2 mode only
    python view_reward_log.py logs/reward/*.jsonl --filter low --threshold 0.5
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def load_records(files: list[str]) -> list[dict]:
    records = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            for ln, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"  [WARN] {f}:{ln} invalid JSON", file=sys.stderr)
    records.sort(key=lambda r: r.get("_seq", 0))
    return records


# ── Summary ──────────────────────────────────────────────────────────
def print_summary(records: list[dict]):
    print(f"\n{'='*60}")
    print(f"  Total: {len(records)} records")
    print(f"{'='*60}")

    modes = Counter(r.get("mode", "?") for r in records)
    print(f"\n  Modes: {dict(modes)}")

    for mode in sorted(modes):
        subset = [r for r in records if r.get("mode") == mode]
        rewards = [r["reward"] for r in subset if "reward" in r]
        if not rewards:
            continue

        avg_r = sum(rewards) / len(rewards)
        pos = sum(1 for r in rewards if r > 0)

        print(f"\n  [{mode.upper()}] {len(subset)} samples")
        print(f"    Avg reward : {avg_r:.4f}")
        print(f"    Positive   : {pos}/{len(rewards)} ({pos/len(rewards)*100:.1f}%)")
        print(f"    Range      : [{min(rewards):.3f}, {max(rewards):.3f}]")

        if mode == "v2":
            dir_ok = [r.get("n_dir_ok", 0) for r in subset]
            n_dims = [len(r.get("dims", [])) for r in subset]
            pairs = [(d, t) for d, t in zip(dir_ok, n_dims) if t > 0]
            if pairs:
                dir_acc = sum(d / t for d, t in pairs) / len(pairs)
                print(f"    Dir match  : {dir_acc*100:.1f}%")

    # Label / prediction distribution
    labels = Counter(r.get("label", "?") for r in records)
    preds = Counter(r.get("prediction") or "<none>" for r in records)
    print(f"\n  Labels : {dict(labels)}")
    print(f"  Preds  : {dict(preds)}")
    print(f"{'='*60}\n")


# ── Detail ───────────────────────────────────────────────────────────
def print_record(r: dict, show_prompt: bool = False):
    seq = r.get("_seq", "?")
    mode = r.get("mode", "?")
    idx = r.get("index", "?")
    pred = r.get("prediction") or "<none>"
    label = r.get("label", "?")
    reward = r.get("reward", "?")

    print(f"\n--- #{seq} | idx={idx} | {mode} | pred={pred} | label={label} | reward={reward} ---")

    # Prompt (compact, image pads already stripped)
    if show_prompt:
        prompt = r.get("prompt", "")
        if prompt:
            # Truncate very long prompts
            if len(prompt) > 2000:
                prompt = prompt[:1000] + f"\n... ({len(prompt)} chars total) ...\n" + prompt[-500:]
            print(f"  [Prompt]\n{prompt}\n")

    # V2: per-dimension rewards
    dims = r.get("dims", [])
    if dims:
        n_dir = r.get("n_dir_ok", 0)
        print(f"  [Dimensions] {n_dir}/{len(dims)} direction correct")
        print(f"  {'Name':<25s} {'W':>4s}  {'GT(A,B)':>8s}  {'Pred(A,B)':>10s}  {'Base':>5s} {'Pen':>4s} {'Reward':>6s}")
        print(f"  {'-'*72}")
        for d in dims:
            name = d["name"][:24]
            w = f"{d['w']*100:.0f}%"
            gt = f"({d['gt'][0]},{d['gt'][1]})"
            if d.get("pred"):
                pr = f"({d['pred'][0]},{d['pred'][1]})"
            else:
                pr = "  N/A"
            base = f"{d['base']:.2f}"
            pen = f"{d['penalty']:.1f}"
            dr = f"{d['reward']:.3f}"
            print(f"  {name:<25s} {w:>4s}  {gt:>8s}  {pr:>10s}  {base:>5s} {pen:>4s} {dr:>6s}")


# ── Main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="View T2I reward logs.")
    parser.add_argument("files", nargs="+", help="JSONL file(s)")
    parser.add_argument("--head", type=int, help="Show first N records")
    parser.add_argument("--tail", type=int, help="Show last N records")
    parser.add_argument("--index", type=int, help="Show record by sample index")
    parser.add_argument("--filter", choices=["wrong", "correct", "v1", "v2", "low", "no_parse"])
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for 'low'")
    parser.add_argument("--prompt", action="store_true", help="Show prompt text")
    parser.add_argument("--export", type=str, help="Export to JSON file")

    args = parser.parse_args()
    records = load_records(args.files)
    if not records:
        print("No records found.")
        return

    filtered = records
    if args.filter == "wrong":
        filtered = [r for r in filtered if r.get("reward", 1) == 0]
    elif args.filter == "correct":
        filtered = [r for r in filtered if r.get("reward", 0) > 0]
    elif args.filter == "v1":
        filtered = [r for r in filtered if r.get("mode") == "v1"]
    elif args.filter == "v2":
        filtered = [r for r in filtered if r.get("mode") == "v2"]
    elif args.filter == "v3":
        filtered = [r for r in filtered if r.get("mode") == "v3"]
    elif args.filter == "low":
        filtered = [r for r in filtered if r.get("reward", 1) < args.threshold]
    elif args.filter == "no_parse":
        filtered = [r for r in filtered if r.get("prediction") is None]

    if args.filter:
        print(f"Filter '{args.filter}': {len(filtered)}/{len(records)} records")

    if args.export:
        with open(args.export, "w", encoding="utf-8") as f:
            json.dump(filtered, f, ensure_ascii=False, indent=2)
        print(f"Exported {len(filtered)} records to {args.export}")
        return

    print_summary(filtered)

    if args.index is not None:
        found = [r for r in filtered if r.get("index") == args.index]
        for r in found:
            print_record(r, show_prompt=args.prompt)
        if not found:
            print(f"No record with index={args.index}")
        return

    show = []
    if args.head is not None:
        show = filtered[:args.head]
    elif args.tail is not None:
        show = filtered[-args.tail:]

    for r in show:
        print_record(r, show_prompt=args.prompt)


if __name__ == "__main__":
    main()
