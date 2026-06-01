import argparse
import json
import os
import re
import time
from io import BytesIO
from pathlib import Path

CACHE_DIR =""

os.environ["HF_HOME"] = f"{CACHE_DIR}/huggingface"
os.environ["TMPDIR"] = f"{CACHE_DIR}/tmp"
os.environ["VLLM_CACHE_ROOT"] = f"{CACHE_DIR}/vllm"

os.environ["VLLM_BATCH_INVARIANT"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"

import numpy as np
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

try:
    from qwen_vl_utils import process_vision_info as _qwen_process_vision_info
    HAS_QWEN_VL_UTILS = True
except ImportError:
    HAS_QWEN_VL_UTILS = False
    print("[WARN] qwen_vl_utils not found. Images will NOT be pre-resized.")
    print("       Install: pip install qwen-vl-utils")

def qwen2_5_vl_dedup_image_tokens(prompt_ids: list[int], processor) -> list[int]:
    """Deduplicate consecutive image/video pad tokens for Qwen2.5-VL."""
    if (
        processor is not None
        and hasattr(processor, "image_processor")
        and "Qwen2VLImageProcessor" in processor.image_processor.__class__.__name__
    ):
        ids = np.array(prompt_ids)
        mask = np.ones(len(ids), dtype=bool)
        is_pad = (ids == processor.image_token_id) | (ids == processor.video_token_id)
        mask[1:] &= ~(is_pad[1:] & is_pad[:-1])
        return ids[mask].tolist()
    return prompt_ids


def load_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_done_indices(output_path: Path) -> set[int]:
    if not output_path.exists():
        return set()
    done = set()
    for line in output_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            if "index" in rec:
                done.add(rec["index"])
        except json.JSONDecodeError:
            pass
    return done


def build_messages(user_prompt: str, images: list) -> tuple[list[dict], list[Image.Image]]:
    content_list = []
    pil_images = []

    segments = re.split(r"(<image>)", user_prompt)
    segments = [s for s in segments if s != ""]

    image_offset = 0
    for segment in segments:
        if segment == "<image>":
            assert image_offset < len(images), (
                f"image_offset {image_offset} >= len(images) {len(images)}"
            )
            img_source = images[image_offset]
            if isinstance(img_source, str):
                pil_img = Image.open(img_source).convert("RGB")
            elif isinstance(img_source, dict) and "bytes" in img_source:
                pil_img = Image.open(BytesIO(img_source["bytes"])).convert("RGB")
            elif isinstance(img_source, Image.Image):
                pil_img = img_source.convert("RGB")
            else:
                raise TypeError(f"Unsupported image type: {type(img_source)}")

            content_list.append({"type": "image", "image": pil_img})
            pil_images.append(pil_img)
            image_offset += 1
        else:
            content_list.append({"type": "text", "text": segment})

    messages = [{"role": "user", "content": content_list}]
    return messages, pil_images


def extract_and_resize_images(
    messages: list[dict],
    processor,
) -> list[Image.Image]:
    if HAS_QWEN_VL_UTILS and hasattr(processor, "image_processor"):
        patch_size = getattr(processor.image_processor, "patch_size", None)
        try:
            result = _qwen_process_vision_info(
                messages,
                image_patch_size=patch_size,
            )
        except TypeError:
            result = _qwen_process_vision_info(messages)
        images = result[0]
        return images if images else []
    else:
        pil_images = []
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "image" and "image" in item:
                        pil_images.append(item["image"])
        return pil_images


def safe_apply_chat_template(processor, messages, **kwargs):
    try:
        return processor.apply_chat_template(messages, **kwargs)
    except Exception:
        dummy_user = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
        dummy_prefix = processor.apply_chat_template(
            dummy_user,
            tokenize=kwargs.get("tokenize", True),
            add_generation_prompt=False,
        )
        full_output = processor.apply_chat_template(
            dummy_user + messages, **kwargs
        )
        if not kwargs.get("tokenize", True):
            return full_output[len(dummy_prefix):]
        return full_output[len(dummy_prefix):]


def tokenize(
    processor,
    messages: list[dict],
    enable_thinking: bool = False,
) -> tuple[list[int], list[Image.Image]]:
    processed_images = extract_and_resize_images(messages, processor)

    template_kwargs = dict(
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    raw_prompt = safe_apply_chat_template(processor, messages, **template_kwargs)

    model_inputs = processor(
        text=[raw_prompt],
        images=processed_images if processed_images else None,
        return_tensors="pt",
    )
    prompt_ids = model_inputs["input_ids"][0].tolist()

    prompt_ids = qwen2_5_vl_dedup_image_tokens(prompt_ids, processor)

    return prompt_ids, processed_images


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline evaluation for RubricRM")
    p.add_argument("--model", "-m", required=True, help="Model path")
    p.add_argument("--input", "-i", required=True, help="Input JSON/JSONL file")
    p.add_argument("--output", "-o", required=True, help="Output JSONL file")
    p.add_argument("--batch-size", "-b", type=int, default=256, help="Batch size")
    p.add_argument("--max-new-tokens", type=int, default=8192, help="Max new tokens to generate")
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0=greedy)")
    p.add_argument("--top-p", type=float, default=1.0, help="top_p")
    p.add_argument("--top-k", type=int, default=-1, help="top_k (-1=disabled)")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--max-model-len", type=int, default=32768, help="Max context length")
    p.add_argument("--tensor-parallel", type=int, default=1, help="TP size")
    p.add_argument("--gpu-memory-util", type=float, default=0.7, help="GPU memory utilization")
    p.add_argument("--limit-mm-per-prompt", type=int, default=8, help="Max images per prompt")
    p.add_argument("--enforce-eager", action="store_true", default=False)
    p.add_argument("--enable-chunked-prefill", action="store_true", default=True,
                   help="Enable chunked prefill")
    p.add_argument("--no-chunked-prefill", dest="enable_chunked_prefill", action="store_false")
    p.add_argument("--max-num-batched-tokens", type=int, default=131072,
                   help="Max number of batched tokens")
    p.add_argument("--max-num-seqs", type=int, default=1024,
                   help="Max number of sequences")
    p.add_argument("--trust-remote-code", action="store_true", default=True)
    p.add_argument("--enable-thinking", action="store_true", default=False,
                   help="Qwen3.5: False=inject empty <think></think> block")
    p.add_argument("--shard-id", type=int, default=0, help="Current shard ID")
    p.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    return p.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"[ERROR] Input file does not exist: {input_path}")
        return

    records = load_records(input_path)
    for i, rec in enumerate(records):
        if "index" not in rec:
            rec["index"] = i
    total = len(records)

    done_indices = load_done_indices(output_path)
    if done_indices:
        print(f"[Resume] Detected existing output, {len(done_indices)} done, skipping.")

    pending = [r for r in records if r["index"] not in done_indices]

    if args.num_shards > 1:
        pending = [r for i, r in enumerate(pending) if i % args.num_shards == args.shard_id]
        print(f"[Shard] shard {args.shard_id}/{args.num_shards}, pending in this shard: {len(pending)}")

    if not pending:
        print("All records already processed.")
        return

    print(f"Total {total} records, pending {len(pending)} | batch_size={args.batch_size}")

    print("\n[1/2] Loading processor...")
    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )

    print("[2/2] Loading vLLM model...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel,
        gpu_memory_utilization=args.gpu_memory_util,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        limit_mm_per_prompt={"image": args.limit_mm_per_prompt},
        trust_remote_code=args.trust_remote_code,
        seed=args.seed,
        enforce_eager=args.enforce_eager,
        enable_chunked_prefill=args.enable_chunked_prefill,
        attention_backend="FLASH_ATTN",
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
        repetition_penalty=1.0,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    n_batches = (len(pending) + args.batch_size - 1) // args.batch_size

    with open(output_path, "a", encoding="utf-8") as out_f:
        for batch_idx in range(n_batches):
            s = batch_idx * args.batch_size
            e = min(s + args.batch_size, len(pending))
            batch = pending[s:e]

            print(f"\n--- Batch {batch_idx + 1}/{n_batches}  [{s + 1}~{e}/{len(pending)}] ---")

            vllm_inputs = []
            for rec in batch:
                try:
                    user_prompt = rec.get("user_prompt", "")
                    image_paths = rec.get("images", [])[:args.limit_mm_per_prompt]

                    messages, pil_images = build_messages(user_prompt, image_paths)

                    prompt_ids, processed_images = tokenize(
                        processor, messages,
                        enable_thinking=args.enable_thinking,
                    )

                    multi_modal_data = {}
                    if processed_images:
                        multi_modal_data["image"] = processed_images
                    vllm_inputs.append(
                        TokensPrompt(
                            prompt_token_ids=prompt_ids,
                            multi_modal_data=multi_modal_data if multi_modal_data else None,
                        )
                    )
                except Exception as ex:
                    print(f"  [WARN] Failed to build input (index={rec.get('index')}): {ex}")
                    vllm_inputs.append(TokensPrompt(prompt_token_ids=[]))

            outputs = llm.generate(vllm_inputs, sampling_params)

            for rec, out in zip(batch, outputs):
                try:
                    predict = out.outputs[0].text
                except Exception as ex:
                    predict = f"[ERROR] {ex}"

                result = {**rec, "predict": predict}
                if "images" in result:
                    result["images"] = [
                        p if isinstance(p, str) else "<embedded>"
                        for p in result.get("images", [])
                    ]
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

            out_f.flush()
            elapsed = time.time() - start

            print(f"  Done {e}/{len(pending)} | elapsed {elapsed:.1f}s")

    print(f"\n{'=' * 60}")
    print(f"All done! Processed {len(pending)} records | total time {time.time() - start:.1f}s")
    print(f"Output: {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
