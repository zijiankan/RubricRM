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
Reward functions for T2I evaluation reward model training.

Two reward modes (auto-detected via extra_info["mode"]):

  V1 (full generation, mode="v1") / V3 (validation, mode="v3"):
    - Binary accuracy: 1.0 if last \boxed{A/B} matches ground_truth, else 0.0

  V2 (rubric-guided, mode="v2"):
    - Signed-difference matching with three-state direction penalty.
    - For each dimension:
        base  = 1.0 - |pred_diff - gt_diff| / 8.0
        where diff = score_a - score_b
    - Direction penalty (three states: A>B, A=B, A<B):
        same state  -> x1.0
        off-by-one  -> x0.6  (e.g. A>B predicted as A=B)
        reversal    -> x0.1  (e.g. A>B predicted as A<B)
    - dim_reward = base * penalty
    - Final score = weighted sum of dim_reward across dimensions.
    - Falls back to V1 accuracy if parsing fails.

Logging:
    Set env T2I_REWARD_LOG_DIR to a directory path to enable JSON log output.
    Set env T2I_REWARD_LOG_SAMPLE_RATE to a float in (0,1] to log only a fraction
    of records (default 1.0 = log all). E.g. 0.1 logs ~10%.

    Compact JSONL format per record:
      mode, index, prompt (image pads stripped), prediction, label,
      reward, dims (v2 per-dimension details), n_dir_ok (v2).
"""

import json
import os
import random
import re
import threading
import time
from datetime import datetime
from typing import Any, Optional

# === Shared patterns ===
BOXED_PATTERN = re.compile(r"\\boxed\{([^}]*)\}")
FORMAT_PATTERN = re.compile(r"<think>.*</think>.*\\boxed\{.*\}", re.DOTALL)
VALID_LABELS = {"A", "B"}

# Pattern to match dimension scores: "Score: X/4", "**Score**: X/4", "Score: **X/4**"
_SCORE_RE = re.compile(
    r"\*{0,2}(?:Score|score)\*{0,2}\s*[:：]\s*\*{0,2}(\d+(?:\.\d+)?)\s*/\s*4\*{0,2}",
)
# Fallback: "Image A: X/4" or "Image B: X/4" (used when Score: pattern absent)
_SCORE_AB_RE = re.compile(
    r"\*{0,2}Image\s+[AB]\*{0,2}\s*[:：]\s*(\d+(?:\.\d+)?)\s*/\s*4",
)
# Pattern to match weight percentages in parentheses: "(30%)"
_WEIGHT_RE = re.compile(r"\((\d+(?:\.\d+)?)%\)")
# Section delimiters (any markdown decoration: ##, **, [], etc.)
_EVAL_START_RE = re.compile(r"Detailed\s+Evaluation", re.IGNORECASE)
_EVAL_END_RE = re.compile(r"Final\s+Conclusion", re.IGNORECASE)

# Pattern to strip image pad tokens for compact logging
_IMAGE_PAD_RE = re.compile(r"(<\|image_pad\|>)+")


# =====================================================================
# JSON File Logger
# =====================================================================
class RewardLogger:
    """Thread-safe JSONL file logger for reward computation results."""

    def __init__(self):
        self._file = None
        self._lock = threading.Lock()
        self._call_count = 0
        self._initialized = False
        self._sample_rate = 1.0

    def _lazy_init(self):
        if self._initialized:
            return
        self._initialized = True
        log_dir = os.environ.get("T2I_REWARD_LOG_DIR")
        if not log_dir:
            return
        self._sample_rate = float(os.environ.get("T2I_REWARD_LOG_SAMPLE_RATE", "1.0"))
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        pid = os.getpid()
        log_path = os.path.join(log_dir, f"reward_log_{ts}_pid{pid}.jsonl")
        self._file = open(log_path, "a", encoding="utf-8")
        print(f"[RewardLogger] Logging to {log_path} (sample_rate={self._sample_rate})")

    @property
    def enabled(self) -> bool:
        if not self._initialized:
            self._lazy_init()
        return self._file is not None

    def log(self, record: dict):
        if not self.enabled:
            return
        with self._lock:
            self._call_count += 1
            # Probabilistic sampling: only log a fraction of records
            if self._sample_rate < 1.0 and random.random() > self._sample_rate:
                return
            record["_seq"] = self._call_count
            record["_ts"] = time.time()
            self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
            # Flush every 10 records
            if self._call_count % 10 == 0:
                self._file.flush()

    def close(self):
        if self._file:
            self._file.flush()
            self._file.close()


_logger = RewardLogger()
_fail_logger = RewardLogger()  # Separate logger for V2 parse failures


def _init_fail_logger():
    """Lazily initialize the V2 parse failure logger."""
    if _fail_logger._initialized:
        return
    log_dir = os.environ.get("T2I_REWARD_LOG_DIR")
    if not log_dir:
        _fail_logger._initialized = True
        return
    fail_dir = os.path.join(log_dir, "v2_parse_failures")
    os.makedirs(fail_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()
    log_path = os.path.join(fail_dir, f"fail_{ts}_pid{pid}.jsonl")
    _fail_logger._initialized = True
    _fail_logger._sample_rate = 1.0  # Log ALL failures, no sampling
    _fail_logger._file = open(log_path, "a", encoding="utf-8")
    print(f"[RewardLogger] V2 parse failure log: {log_path}")


def _log_v2_parse_failure(
    reason: str,
    extra_info: Optional[dict],
    solution_str: str,
    ground_truth: str,
):
    """Log a V2 parse failure with full input/output to the failure log."""
    _init_fail_logger()
    if _fail_logger._file is None:
        return
    prompt_text = ""
    if extra_info and isinstance(extra_info, dict):
        prompt_text = extra_info.get("_prompt_text", "")
    record = {
        "reason": reason,
        "index": extra_info.get("index") if extra_info else None,
        "prompt": _IMAGE_PAD_RE.sub("<image>", prompt_text) if prompt_text else "",
        "output": solution_str,
        "ground_truth": ground_truth,
        "score_raw": extra_info.get("score") if extra_info else None,
    }
    _fail_logger.log(record)


# =====================================================================
# V1: Binary accuracy
# =====================================================================
def v1_acc_reward(predict_str: str, ground_truth: str) -> float:
    """1.0 if last \\boxed{A/B} matches ground_truth, else 0.0."""
    matches = BOXED_PATTERN.findall(predict_str)
    if not matches:
        return 0.0
    predicted = matches[-1].strip()
    if predicted not in VALID_LABELS:
        return 0.0
    return 1.0 if predicted == ground_truth.strip() else 0.0


# =====================================================================
# V2: Dimension-wise scoring
# =====================================================================
def parse_dimension_scores(text: str) -> list[dict]:
    """Robustly parse dimension scores from model output.

    Strategy (tolerant of markdown formatting variations):
      1. Locate "Detailed Evaluation" and "Final Conclusion" anchors.
      2. Extract the section between them.
      3. Sequentially collect all (N%) weight tokens and Score: X/4 tokens.
      4. Pair by position: each weight gets the next 2 scores (A, B).

    Returns list of dicts: [{"weight": float, "score_a": float, "score_b": float}]
    """
    # Step 1: delimit the evaluation section
    start = _EVAL_START_RE.search(text)
    if not start:
        return []
    remainder = text[start.end():]
    end = _EVAL_END_RE.search(remainder)
    section = remainder[:end.start()] if end else remainder

    # Step 2: extract weights and scores in document order
    weights = [float(m) / 100.0 for m in _WEIGHT_RE.findall(section)]
    scores = [float(m) for m in _SCORE_RE.findall(section)]

    # Fallback: if no "Score: X/4" found, try "Image A/B: X/4" pattern
    if not scores:
        scores = [float(m) for m in _SCORE_AB_RE.findall(section)]

    # Step 3: pair positionally – each weight maps to 2 consecutive scores
    n_dims = min(len(weights), len(scores) // 2)
    return [
        {"weight": weights[i], "score_a": scores[2 * i], "score_b": scores[2 * i + 1]}
        for i in range(n_dims)
    ]


# Direction penalty constants
_PENALTY_SAME = 1.0       # Same state (e.g. A>B -> A>B)
_PENALTY_OFF_BY_ONE = 0.6  # Off by one level (e.g. A>B -> A=B)
_PENALTY_REVERSAL = 0.1    # Complete reversal (e.g. A>B -> A<B)
_MAX_DIFF_RANGE = 8.0      # Score range [-4, +4], max |pred_diff - gt_diff| = 8


def _sign(x: float) -> int:
    """Return +1, 0, or -1."""
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0


def _direction_penalty(gt_diff: float, pred_diff: float) -> float:
    """Compute direction penalty based on three-state comparison.

    States: A>B (diff > 0), A=B (diff == 0), A<B (diff < 0)
    - Same state:  1.0
    - Off-by-one:  0.6  (one side is equal)
    - Reversal:    0.1  (opposite directions)
    """
    gt_dir = _sign(gt_diff)
    pred_dir = _sign(pred_diff)
    if gt_dir == pred_dir:
        return _PENALTY_SAME
    elif gt_dir == 0 or pred_dir == 0:
        return _PENALTY_OFF_BY_ONE
    else:
        return _PENALTY_REVERSAL


def v2_dimension_reward(predict_str: str, score_info: dict) -> dict:
    """Signed-difference matching with three-state direction penalty.

    GT dimensions and parsed dimensions are matched **by position** (document
    order), not by name.  This avoids false negatives caused by minor
    dimension-name formatting differences.

    For each dimension:
        gt_diff   = gt_a - gt_b
        pred_diff = pred_a - pred_b
        base      = 1.0 - |pred_diff - gt_diff| / 8.0
        penalty   = direction_penalty(gt_diff, pred_diff)
        dim_reward = base * penalty

    Final score = weighted sum of dim_reward across dimensions.
    """
    gt_dims = score_info.get("dimensions", [])
    if not gt_dims:
        return {"score": 0.0, "dimension_details": [], "num_direction_correct": 0, "num_total": 0}

    pred_dims = parse_dimension_scores(predict_str)

    total_weight = 0.0
    weighted_score = 0.0
    details = []
    num_direction_correct = 0

    for i, gt_dim in enumerate(gt_dims):
        gt_name = gt_dim["name"]
        gt_weight = gt_dim["weight"]
        gt_sa = gt_dim["score_a"]
        gt_sb = gt_dim["score_b"]
        gt_diff = gt_sa - gt_sb

        # Positional matching
        if i < len(pred_dims):
            pred_sa = pred_dims[i]["score_a"]
            pred_sb = pred_dims[i]["score_b"]
            pred_diff = pred_sa - pred_sb

            base = 1.0 - abs(pred_diff - gt_diff) / _MAX_DIFF_RANGE
            penalty = _direction_penalty(gt_diff, pred_diff)
            dim_reward = base * penalty

            direction_match = _sign(gt_diff) == _sign(pred_diff)
            if direction_match:
                num_direction_correct += 1

            details.append({
                "name": gt_name,
                "weight": gt_weight,
                "gt_a": gt_sa, "gt_b": gt_sb,
                "pred_a": pred_sa, "pred_b": pred_sb,
                "base": round(base, 4),
                "penalty": penalty,
                "dim_reward": round(dim_reward, 4),
                "direction_match": direction_match,
            })
        else:
            # Dimension not parsed -> zero reward for this dimension
            details.append({
                "name": gt_name,
                "weight": gt_weight,
                "gt_a": gt_sa, "gt_b": gt_sb,
                "pred_a": None, "pred_b": None,
                "base": 0.0,
                "penalty": 0.0,
                "dim_reward": 0.0,
                "direction_match": False,
            })

        weighted_score += gt_weight * details[-1]["dim_reward"]
        total_weight += gt_weight

    if total_weight > 0 and abs(total_weight - 1.0) > 0.01:
        weighted_score = weighted_score / total_weight

    return {
        "score": weighted_score,
        "dimension_details": details,
        "num_direction_correct": num_direction_correct,
        "num_total": len(gt_dims),
        "num_parsed": len(pred_dims),
    }


# =====================================================================
# Main entry point
# =====================================================================
def compute_score(
    data_source: Optional[str],
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict[str, Any]] = None,
    **kwargs,
) -> dict[str, float]:
    """Compute reward score, auto-routing by mode in extra_info."""

    mode = "v1"
    if extra_info and isinstance(extra_info, dict):
        mode = extra_info.get("mode", "v1")

    # --- Build compact log record ---
    log_record = {"mode": mode}
    if extra_info and isinstance(extra_info, dict):
        log_record["index"] = extra_info.get("index")
        # Compact prompt: strip image pad tokens, keep structure
        prompt_text = extra_info.get("_prompt_text", "")
        if prompt_text:
            log_record["prompt"] = _IMAGE_PAD_RE.sub("<image>", prompt_text)

    # --- V2: Dimension-wise scoring ---
    if mode == "v2" and extra_info and extra_info.get("score"):
        score_str = extra_info["score"]
        try:
            score_info = json.loads(score_str) if isinstance(score_str, str) else score_str
        except (json.JSONDecodeError, TypeError):
            score_info = None

        if score_info and isinstance(score_info, dict) and score_info.get("dimensions"):
            dim_result = v2_dimension_reward(solution_str, score_info)
            n_gt = dim_result["num_total"]
            n_parsed = dim_result["num_parsed"]
            v2_parse_failed = 0.0 if n_parsed == n_gt else 1.0

            # Dimension count mismatch -> zero reward (prevents degenerate outputs)
            if n_parsed != n_gt:
                score = 0.0
            else:
                score = dim_result["score"]

            boxed = BOXED_PATTERN.findall(solution_str)
            log_record["prediction"] = boxed[-1].strip() if boxed else None
            log_record["label"] = ground_truth
            log_record["n_parsed"] = n_parsed
            log_record["n_gt"] = n_gt
            log_record["dims"] = [
                {
                    "name": d["name"],
                    "w": d["weight"],
                    "gt": [d["gt_a"], d["gt_b"]],
                    "pred": [d["pred_a"], d["pred_b"]] if d["pred_a"] is not None else None,
                    "base": d["base"],
                    "penalty": d["penalty"],
                    "reward": d["dim_reward"],
                }
                for d in dim_result["dimension_details"]
            ]
            log_record["n_dir_ok"] = dim_result["num_direction_correct"]
            log_record["reward"] = round(score, 4)

            _logger.log(log_record)

            # Log parse failures (dimension count mismatch) to separate file
            if v2_parse_failed > 0:
                _log_v2_parse_failure(
                    reason=f"dim_mismatch(parsed={n_parsed},gt={n_gt})",
                    extra_info=extra_info,
                    solution_str=solution_str,
                    ground_truth=ground_truth,
                )

            return {
                "score": score,
                "dimension_reward": score,
                "num_direction_correct": dim_result["num_direction_correct"],
                "num_dim_total": dim_result["num_total"],
                "v2_parse_failed": v2_parse_failed,
                "mode": "v2",
            }

    # --- V1 / V3: Binary accuracy ---
    # V3 = validation mode (same logic as V1, separate label for logging)
    boxed_matches = BOXED_PATTERN.findall(solution_str)
    predicted = boxed_matches[-1].strip() if boxed_matches else None
    score = v1_acc_reward(solution_str, ground_truth)

    log_record["prediction"] = predicted
    log_record["label"] = ground_truth
    log_record["reward"] = score

    _logger.log(log_record)

    result = {
        "score": score,
        "accuracy_reward": score,
        "mode": mode,  # v1, v2, or v3
    }
    # V2 mode sample that fell back to V1 → count as parse failure
    if mode == "v2":
        result["v2_parse_failed"] = 1.0
        _log_v2_parse_failure(
            reason="fallback_to_v1",
            extra_info=extra_info,
            solution_str=solution_str,
            ground_truth=ground_truth,
        )
    return result
