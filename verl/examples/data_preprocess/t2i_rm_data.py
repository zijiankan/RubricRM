# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess multimodal T2I reward model training data to parquet format.

Two prompt modes:
  v1 (full generation):
      prompt = [{role: user, content: user_prompt}]
      Model generates the complete evaluation from scratch.

  v2 (rubric-guided):
      prompt = [{role: user, content: user_prompt},
                {role: assistant, content: rubric}]
      Model continues from the rubric, only generating the scoring part.

Images are embedded directly in parquet as {"bytes": <png_bytes>}.
Image path mapping is provided via --image_mapping to resolve original paths
to local paths that actually exist on disk.

Input JSON format (list of dicts):
  - user_prompt: str, the full prompt with <image> placeholders
  - images: list[str], original image file paths
  - answer: str, the expected full response (must contain \\boxed{A} or \\boxed{B})
  - label: str, the ground truth label ("A" or "B")
  - rubric: str, the thinking process / evaluation rubric (used in v2 mode)
  - score: dict, structured dimension scores (used by v2 reward function)
  - source, classification_cn, classification_en: optional metadata

Usage:
    # V1 mode (full generation)
    python t2i_rm_data.py --input_file data.json --image_mapping mapping.json --mode v1

    # V2 mode (rubric-guided, only train scoring)
    python t2i_rm_data.py --input_file data.json --image_mapping mapping.json --mode v2

    # Both modes at once
    python t2i_rm_data.py --input_file data.json --image_mapping mapping.json --mode both
"""

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import datasets
from PIL import Image
from tqdm.auto import tqdm

try:
    from verl.utils.hdfs_io import copy, makedirs
except ImportError:
    copy, makedirs = None, None

BOXED_PATTERN = re.compile(r"\\boxed\{([^}]*)\}")
VALID_LABELS = {"A", "B"}


def parse_label_from_answer(answer: str) -> str | None:
    """Parse the last \\boxed{...} from the answer string. Returns 'A'/'B' or None."""
    matches = BOXED_PATTERN.findall(answer)
    if not matches:
        return None
    last_match = matches[-1].strip()
    return last_match if last_match in VALID_LABELS else None


def validate_and_filter_data(raw_data: list[dict]) -> list[dict]:
    """Validate labels and filter out problematic samples."""
    valid_data = []
    discarded = []

    for i, item in enumerate(raw_data):
        index = item.get("index", i)
        answer = item.get("answer") or ""
        raw_label = item.get("label")

        parsed_label = parse_label_from_answer(answer)

        if parsed_label is None:
            reason = "Cannot parse \\boxed{A/B} from answer"
            discarded.append((i, index, reason))
            continue

        if raw_label and raw_label.strip() in VALID_LABELS:
            if raw_label.strip() != parsed_label:
                reason = (
                    f"Label mismatch: label='{raw_label.strip()}' "
                    f"vs answer \\boxed{{{parsed_label}}}"
                )
                discarded.append((i, index, reason))
                continue
            final_label = parsed_label
        else:
            if raw_label is not None and raw_label.strip() not in VALID_LABELS:
                print(f"  [INFO] Sample {i}: invalid label '{raw_label}', using parsed \\boxed{{{parsed_label}}}.")
            final_label = parsed_label

        item["label"] = final_label
        valid_data.append(item)

    if discarded:
        print(f"\n{'='*60}")
        print(f"DISCARDED {len(discarded)} samples:")
        for idx_in_list, index, reason in discarded:
            print(f"  [{idx_in_list}] index={index}: {reason}")
        print(f"{'='*60}\n")
    else:
        print("All samples passed label validation.")

    return valid_data


# Formats that can be read as raw bytes without PIL re-encoding
_RAW_READABLE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def load_and_embed_image(path: str) -> dict:
    """Load image from local path and return as {"bytes": ...} dict for parquet embedding.

    For common formats (JPEG/PNG/WebP), reads raw bytes directly without PIL decode+re-encode.
    Falls back to PIL for other formats.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in _RAW_READABLE_EXTS:
        # Fast path: read raw bytes, skip PIL decode/re-encode
        with open(path, "rb") as f:
            return {"bytes": f.read()}
    else:
        # Slow path: PIL decode + re-encode for unusual formats
        img = Image.open(path).convert("RGB")
        buf = BytesIO()
        img.save(buf, format="PNG")
        return {"bytes": buf.getvalue()}


def make_map_fn(data_source: str, mode: str):
    """Create map function for the specified mode (v1 or v2)."""
    def process_fn(example, idx):
        user_prompt = example["user_prompt"]
        label = example["label"]
        answer = example.get("answer") or ""
        source = example.get("source") or ""
        index = example.get("index", idx)
        prompt_text = example.get("prompt") or ""
        score = example.get("score")
        rubric = example.get("rubric") or ""

        # Build prompt messages based on mode
        if mode == "v1":
            prompt_messages = [{"role": "user", "content": user_prompt}]
        else:  # v2
            prompt_messages = [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": rubric},
            ]

        # Build extra_info - unified format for both modes
        extra_info = {
            "index": index,
            "answer": answer,
            "label": label,
            "source": source,
            "prompt": prompt_text,
            "rubric": rubric,
            "mode": mode,
        }
        # Store score dict as JSON string for parquet compatibility
        if score is not None:
            extra_info["score"] = json.dumps(score, ensure_ascii=False) if isinstance(score, dict) else str(score)

        data = {
            "data_source": data_source,
            "prompt": prompt_messages,
            "images": example["images"],
            "ability": "t2i_eval",
            "reward_model": {"style": "rule", "ground_truth": label},
            "extra_info": extra_info,
        }
        return data

    return process_fn


def debug_print_sample(sample, mode, sample_idx):
    """Print key fields of a processed sample for verification."""
    print(f"\n{'='*60}")
    print(f"[DEBUG] Mode={mode}, Sample #{sample_idx}")
    print(f"{'='*60}")

    # 1. Prompt structure
    prompt = sample["prompt"]
    print(f"\n[Prompt] {len(prompt)} message(s):")
    for i, msg in enumerate(prompt):
        role = msg["role"]
        content = msg["content"]
        preview = content[:200] + "..." if len(content) > 200 else content
        print(f"  [{i}] role={role}, content_len={len(content)}")
        print(f"      preview: {preview}")

    # 2. Images
    images = sample["images"]
    print(f"\n[Images] {len(images)} image(s):")
    for i, img in enumerate(images):
        if isinstance(img, dict) and "bytes" in img:
            print(f"  [{i}] embedded, size={len(img['bytes'])} bytes")
        else:
            print(f"  [{i}] type={type(img).__name__}, value={str(img)[:100]}")

    # 3. Reward model info
    rm = sample.get("reward_model", {})
    print(f"\n[RewardModel] ground_truth={rm.get('ground_truth')}")

    # 4. Extra info (especially mode and score)
    extra = sample.get("extra_info", {})
    print(f"\n[ExtraInfo] mode={extra.get('mode')}, label={extra.get('label')}")
    score_str = extra.get("score")
    if score_str:
        print(f"  score preview: {str(score_str)[:300]}")
    print(f"{'='*60}\n")


def process_and_save(raw_data, data_source, mode, save_dir, num_proc):
    """Process data for one mode and save as parquet."""
    dataset = datasets.Dataset.from_list(raw_data)
    dataset = dataset.map(
        function=make_map_fn(data_source, mode),
        with_indices=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
    )

    # Debug: print first 2 samples
    print(f"\n>>> Debug output for mode={mode} (first 2 samples):")
    for i in range(min(2, len(dataset))):
        debug_print_sample(dataset[i], mode, i)

    filename = f"train_{mode}.parquet"
    filepath = os.path.join(save_dir, filename)
    dataset.to_parquet(filepath)
    print(f"[{mode}] Saved {len(dataset)} samples -> {filepath}")
    return filepath


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess T2I RM data to parquet (v1/v2 modes).")
    parser.add_argument("--input_file", required=True, help="Path to input JSON file.")
    parser.add_argument("--image_mapping", required=True, help="Path to image mapping JSON (old_path -> new_path).")
    parser.add_argument("--mode", default="both", choices=["v1", "v2", "both"],
                        help="Prompt mode: v1 (full gen), v2 (rubric-guided), both.")
    parser.add_argument("--local_save_dir", default="~/data/t2i_rm",
                        help="Output directory for parquet files.")
    parser.add_argument("--data_source", default="t2i_rm", help="Data source identifier.")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of parallel workers.")
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS directory to copy results.")

    args = parser.parse_args()

    # Load image mapping
    print(f"Loading image mapping from {args.image_mapping} ...")
    with open(args.image_mapping, "r") as f:
        image_mapping = json.load(f)
    print(f"  {len(image_mapping)} path mappings loaded.")

    # Load raw data
    with open(args.input_file, "r") as f:
        raw_data = json.load(f)
    print(f"Loaded {len(raw_data)} samples from {args.input_file}")

    # Validate <image> count matches images list
    img_mismatch = []
    for i, item in enumerate(raw_data):
        n_ph = item["user_prompt"].count("<image>")
        n_img = len(item["images"])
        if n_ph != n_img:
            img_mismatch.append((i, item.get("index", i), n_ph, n_img))
    if img_mismatch:
        print(f"\nERROR: {len(img_mismatch)} samples have <image> count mismatch:")
        for idx, index, n_ph, n_img in img_mismatch:
            print(f"  [{idx}] index={index}: {n_ph} placeholders but {n_img} images")
        raise ValueError("Fix image count mismatches before proceeding.")

    # Validate and filter labels
    raw_data = validate_and_filter_data(raw_data)
    print(f"{len(raw_data)} samples remaining after validation.")
    if len(raw_data) == 0:
        raise ValueError("No valid samples remaining.")

    # V2 mode requires rubric field
    if args.mode in ("v2", "both"):
        missing_rubric = sum(1 for item in raw_data if not item.get("rubric"))
        if missing_rubric > 0:
            print(f"WARNING: {missing_rubric} samples missing 'rubric' field (will be empty string in v2).")

    # Embed images: resolve paths via mapping, load as bytes (parallel I/O)
    print("Embedding images into data ...")
    skipped_img = 0

    # Build flat list of (sample_idx, img_idx, local_path_or_None) for parallel loading
    _PLACEHOLDER_BUF = BytesIO()
    Image.new("RGB", (1, 1), (255, 0, 0)).save(_PLACEHOLDER_BUF, format="PNG")
    _PLACEHOLDER_BYTES = {"bytes": _PLACEHOLDER_BUF.getvalue()}

    load_tasks = []  # list of (sample_idx, img_idx_in_sample, local_path | None)
    for i, item in enumerate(raw_data):
        for j, orig_path in enumerate(item["images"]):
            local_path = image_mapping.get(orig_path)
            if local_path is None:
                print(f"  [WARN] Sample {i}: no mapping for {orig_path}")
                load_tasks.append((i, j, None))
            elif not os.path.exists(local_path):
                print(f"  [WARN] Sample {i}: mapped path not found: {local_path}")
                load_tasks.append((i, j, None))
            else:
                load_tasks.append((i, j, local_path))

    # Parallel image loading with thread pool (I/O bound)
    num_threads = min(32, max(1, args.num_proc * 4))
    results = [None] * len(load_tasks)

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

    # Assign loaded images back to samples
    for task_idx, (si, ji, path) in enumerate(load_tasks):
        if path is None:
            skipped_img += 1
        # Ensure images list is the right type
        if not isinstance(raw_data[si]["images"][ji], dict):
            raw_data[si]["images"][ji] = results[task_idx]
        else:
            raw_data[si]["images"][ji] = results[task_idx]

    if skipped_img > 0:
        print(f"WARNING: {skipped_img} images could not be loaded (used placeholder).")
    print(f"Image embedding complete for {len(raw_data)} samples ({len(load_tasks)} images, {num_threads} threads).")

    # Save
    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)

    modes = ["v1", "v2"] if args.mode == "both" else [args.mode]
    for m in modes:
        process_and_save(raw_data, args.data_source, m, save_dir, args.num_proc)

    if args.hdfs_dir:
        if copy is None or makedirs is None:
            raise ImportError("verl is required for HDFS copy.")
        makedirs(args.hdfs_dir)
        copy(src=save_dir, dst=args.hdfs_dir)
        print(f"Copied to HDFS: {args.hdfs_dir}")

    print("Done!")
