"""
Preprocess T2I validation data to parquet format.

Validation always uses V1 mode (full generation) regardless of training mode.

Input JSON format (list of dicts):
  - id: str, unique identifier
  - user_prompt: str, the full evaluation prompt with <image> placeholders
  - images: list[str], absolute image file paths
  - chosen: str, ground truth label ("A" or "B")
  - metadata: dict, optional metadata (source, category, class, model_a, model_b)

Usage:
    python t2i_val_data.py --input_file val_data/mmrb2_gen.json --local_save_dir /dev/shm/t2i/val
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import datasets
from PIL import Image
from tqdm.auto import tqdm

VALID_LABELS = {"A", "B"}

# Formats that can be read as raw bytes without PIL re-encoding
_RAW_READABLE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def load_and_embed_image(path: str) -> dict:
    """Load image and return as {"bytes": ...} for parquet embedding."""
    ext = os.path.splitext(path)[1].lower()
    if ext in _RAW_READABLE_EXTS:
        with open(path, "rb") as f:
            return {"bytes": f.read()}
    else:
        img = Image.open(path).convert("RGB")
        buf = BytesIO()
        img.save(buf, format="PNG")
        return {"bytes": buf.getvalue()}


def make_map_fn(data_source: str):
    """Create map function for V1 mode validation data."""

    def process_fn(example, idx):
        user_prompt = example["user_prompt"]
        chosen = example["chosen"]
        sample_id = example.get("id", str(idx))
        metadata = example.get("metadata") or {}

        # V1 mode: single user message
        prompt_messages = [{"role": "user", "content": user_prompt}]

        extra_info = {
            "index": idx,
            "id": sample_id,
            "label": chosen,
            "source": metadata.get("source", ""),
            "category": metadata.get("category", ""),
            "model_a": metadata.get("model_a", ""),
            "model_b": metadata.get("model_b", ""),
            "mode": "v1",
            "split": "val",
        }

        data = {
            "data_source": data_source,
            "prompt": prompt_messages,
            "images": example["images"],
            "ability": "t2i_eval",
            "reward_model": {"style": "rule", "ground_truth": chosen},
            "extra_info": extra_info,
        }
        return data

    return process_fn


def debug_print_sample(sample, idx):
    """Print key fields for verification."""
    print(f"\n{'=' * 60}")
    print(f"[DEBUG] Val Sample #{idx}")
    print(f"{'=' * 60}")

    prompt = sample["prompt"]
    print(f"\n[Prompt] {len(prompt)} message(s):")
    for i, msg in enumerate(prompt):
        content = msg["content"]
        preview = content[:200] + "..." if len(content) > 200 else content
        print(f"  [{i}] role={msg['role']}, content_len={len(content)}")
        print(f"      preview: {preview}")

    images = sample["images"]
    print(f"\n[Images] {len(images)} image(s):")
    for i, img in enumerate(images):
        if isinstance(img, dict) and "bytes" in img:
            print(f"  [{i}] embedded, size={len(img['bytes'])} bytes")
        else:
            print(f"  [{i}] type={type(img).__name__}, value={str(img)[:100]}")

    rm = sample.get("reward_model", {})
    print(f"\n[RewardModel] ground_truth={rm.get('ground_truth')}")

    extra = sample.get("extra_info", {})
    print(f"[ExtraInfo] id={extra.get('id')}, label={extra.get('label')}, "
          f"source={extra.get('source')}, category={extra.get('category')}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess T2I validation data to parquet.")
    parser.add_argument("--input_file", required=True, help="Path to input JSON file.")
    parser.add_argument("--local_save_dir", default="~/data/t2i_val", help="Output directory.")
    parser.add_argument("--data_source", default="t2i_rm", help="Data source identifier.")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of parallel workers for dataset.map.")
    parser.add_argument("--num_threads", type=int, default=32, help="Number of threads for image loading.")
    parser.add_argument("--output_name", default="val_gen.parquet", help="Output parquet filename.")

    args = parser.parse_args()

    # Load raw data
    with open(args.input_file, "r") as f:
        raw_data = json.load(f)
    print(f"Loaded {len(raw_data)} samples from {args.input_file}")

    # Validate labels
    invalid = []
    for i, item in enumerate(raw_data):
        chosen = item.get("chosen", "").strip()
        if chosen not in VALID_LABELS:
            invalid.append((i, item.get("id", i), chosen))
        else:
            item["chosen"] = chosen
    if invalid:
        print(f"WARNING: {len(invalid)} samples with invalid chosen label:")
        for idx, sid, label in invalid[:10]:
            print(f"  [{idx}] id={sid}: chosen='{label}'")
        raw_data = [item for item in raw_data if item.get("chosen", "").strip() in VALID_LABELS]
        print(f"Filtered to {len(raw_data)} valid samples.")

    # Validate <image> count matches images list
    img_mismatch = []
    for i, item in enumerate(raw_data):
        n_ph = item["user_prompt"].count("<image>")
        n_img = len(item["images"])
        if n_ph != n_img:
            img_mismatch.append((i, item.get("id", i), n_ph, n_img))
    if img_mismatch:
        print(f"\nERROR: {len(img_mismatch)} samples have <image> count mismatch:")
        for idx, sid, n_ph, n_img in img_mismatch[:10]:
            print(f"  [{idx}] id={sid}: {n_ph} placeholders but {n_img} images")
        raise ValueError("Fix image count mismatches before proceeding.")

    # Parallel image embedding
    print("Embedding images into data ...")
    _PLACEHOLDER_BUF = BytesIO()
    Image.new("RGB", (1, 1), (255, 0, 0)).save(_PLACEHOLDER_BUF, format="PNG")
    _PLACEHOLDER_BYTES = {"bytes": _PLACEHOLDER_BUF.getvalue()}

    load_tasks = []  # (sample_idx, img_idx, path)
    for i, item in enumerate(raw_data):
        for j, img_path in enumerate(item["images"]):
            if os.path.exists(img_path):
                load_tasks.append((i, j, img_path))
            else:
                print(f"  [WARN] Sample {i} (id={item.get('id')}): image not found: {img_path}")
                load_tasks.append((i, j, None))

    skipped_img = 0
    results = [None] * len(load_tasks)
    num_threads = min(args.num_threads, max(1, len(load_tasks)))

    def _load_one(task_idx):
        _, _, path = load_tasks[task_idx]
        if path is None:
            return _PLACEHOLDER_BYTES
        return load_and_embed_image(path)

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = {pool.submit(_load_one, idx): idx for idx in range(len(load_tasks))}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Loading images"):
            task_idx = futures[fut]
            try:
                results[task_idx] = fut.result()
            except Exception as e:
                si, ji, path = load_tasks[task_idx]
                print(f"  [ERROR] Sample {si} img {ji} ({path}): {e}")
                results[task_idx] = _PLACEHOLDER_BYTES
                skipped_img += 1

    # Assign back
    for task_idx, (si, ji, path) in enumerate(load_tasks):
        if path is None:
            skipped_img += 1
        raw_data[si]["images"][ji] = results[task_idx]

    if skipped_img > 0:
        print(f"WARNING: {skipped_img} images could not be loaded (used placeholder).")
    print(f"Image embedding complete: {len(raw_data)} samples, {len(load_tasks)} images, {num_threads} threads.")

    # Process with datasets.map
    dataset = datasets.Dataset.from_list(raw_data)
    dataset = dataset.map(
        function=make_map_fn(args.data_source),
        with_indices=True,
        num_proc=args.num_proc,
        remove_columns=dataset.column_names,
    )

    # Debug: print first 2 samples
    print(f"\n>>> Debug output (first 2 samples):")
    for i in range(min(2, len(dataset))):
        debug_print_sample(dataset[i], i)

    # Save
    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, args.output_name)
    dataset.to_parquet(filepath)
    print(f"Saved {len(dataset)} samples -> {filepath}")
    print("Done!")
