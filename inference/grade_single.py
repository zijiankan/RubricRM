import json
import re
import sys
from pathlib import Path

BOXED_PATTERN = re.compile(r"\\boxed\{([^}]*)\}")
VALID_LABELS = {"A", "B"}


def v1_acc_reward(predict_str: str, ground_truth: str) -> float:
    """1.0 if last \\boxed{A/B} matches ground_truth, else 0.0."""
    matches = BOXED_PATTERN.findall(predict_str)
    if not matches:
        return 0.0
    predicted = matches[-1].strip()
    if predicted not in VALID_LABELS:
        return 0.0
    return 1.0 if predicted == ground_truth.strip() else 0.0


def grade_file(file_path: Path, verbose: bool = True) -> dict:
    total = 0
    correct = 0
    format_error = 0
    invalid_label = 0
    format_error_samples = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            chosen = rec.get("chosen", "").strip()
            if chosen not in VALID_LABELS:
                continue

            predict = str(rec.get("predict", ""))
            total += 1

            if predict.strip() in VALID_LABELS:
                if predict.strip() == chosen:
                    correct += 1
                continue

            matches = BOXED_PATTERN.findall(predict)
            if not matches:
                format_error += 1
                format_error_samples.append({
                    "line": line_no,
                    "index": rec.get("index", "?"),
                    "chosen": chosen,
                    "predict": predict,
                })
                continue

            predicted = matches[-1].strip()
            if predicted not in VALID_LABELS:
                invalid_label += 1
                format_error_samples.append({
                    "line": line_no,
                    "index": rec.get("index", "?"),
                    "chosen": chosen,
                    "extracted": predicted,
                    "predict": predict,
                })
                continue

            if predicted == chosen:
                correct += 1

    accuracy = correct / total if total > 0 else 0.0

    print(f"\n{'=' * 60}")
    print(f"File: {file_path.name}")
    print(f"{'=' * 60}")
    print(f"  Total samples:    {total}")
    print(f"  Correct:          {correct}")
    print(f"  Accuracy:         {accuracy:.2%}")
    print(f"  Format error:     {format_error} (no \\boxed{{...}})")
    print(f"  Invalid label:    {invalid_label} (\\boxed{{X}} but X not in A/B)")

    if verbose and format_error_samples:
        print(f"\n  --- Format error details (total {len(format_error_samples)}) ---")
        for i, sample in enumerate(format_error_samples):
            print(f"\n  [{i+1}] line={sample['line']}, index={sample['index']}, chosen={sample['chosen']}")
            if "extracted" in sample:
                print(f"       extracted: \\boxed{{{sample['extracted']}}} (not A/B)")
            print(f"       predict:")
            print(f"{sample['predict']}")
            print(f"       {'─' * 40}")

    return {
        "file": file_path.name,
        "total": total,
        "correct": correct,
        "format_error": format_error,
        "invalid_label": invalid_label,
        "accuracy": accuracy,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python grade_single.py <file_or_dir> [--verbose]")
        print("  file_or_dir: JSONL file or directory containing .jsonl")
        print("  --verbose: print full predict of format-error samples (default off)")
        return

    target = Path(sys.argv[1])
    verbose = "--verbose" in sys.argv

    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = sorted(target.glob("*.jsonl"))
        if not files:
            print(f"[ERROR] No .jsonl files in directory {target}")
            return
    else:
        print(f"[ERROR] Path does not exist: {target}")
        return

    results = []
    for f in files:
        result = grade_file(f, verbose=verbose)
        results.append(result)

    if len(results) > 1:
        print(f"\n\n{'=' * 60}")
        print("Summary Table")
        print(f"{'=' * 60}")
        print(f"{'File':<45} {'Accuracy':>8} {'Correct/Total':>12} {'FmtErr':>8}")
        print(f"{'-' * 45} {'-' * 8} {'-' * 12} {'-' * 8}")
        for r in results:
            print(f"{r['file']:<45} {r['accuracy']:>7.2%} "
                  f"{r['correct']:>4}/{r['total']:<6} {r['format_error']:>6}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
