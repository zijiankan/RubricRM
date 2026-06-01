import json
import os
import re
import sys
from collections import defaultdict


def extract_preference(predict_text):

    if not predict_text:
        return None
    patterns = [
        r"\\boxed\{([AB])\}",
        r"\\boxed\{\{([AB])\}\}",
        r"boxed\{([AB])\}",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, predict_text)
        if matches:
            return matches[-1]
    stripped = predict_text.strip()
    if stripped in ("A", "B"):
        return stripped
    if stripped.lower() == "tie":
        return "tie"

    return None


def get_base_id(item_id):
    parts = item_id.rsplit("_", 1)
    if len(parts) == 2 and re.match(r"^[A-D]v[A-D]$", parts[1]):
        return parts[0]
    return item_id


def get_pair_labels(comparison_type):

    ct = comparison_type.replace("vs", "v")
    match = re.match(r"([A-D])v([A-D])", ct)
    if match:
        return match.group(1), match.group(2)
    return None, None


def get_gt_preference(ranking, label1, label2):
    match = re.match(r"([A-D])([>=])([A-D])", ranking)
    if not match:
        return None
    winner, op, loser = match.group(1), match.group(2), match.group(3)

    if op == "=":
        return "tie"
    if winner == label1:
        return "A" 
    elif winner == label2:
        return "B" 
    return None


def evaluate_2pair(items):
    correct = 0
    total = 0
    failed_extract = 0
    failed_items = []

    for item in items:
        predict = item.get("predict", "")
        pred_pref = extract_preference(predict)
        if pred_pref is None:
            failed_extract += 1
            total += 1
            failed_items.append(item)
            continue

        ranking = item["ranking"]
        label1, label2 = get_pair_labels(item["comparison_type"])
        gt_pref = get_gt_preference(ranking, label1, label2)

        total += 1
        if gt_pref == "tie":
            correct += 1
        elif pred_pref == "tie":
            pass
        elif pred_pref == gt_pref:
            correct += 1

    return correct, total, failed_extract, failed_items


def evaluate_multi_pair(groups, num_candidates):
    correct = 0
    total = 0
    failed_extract = 0
    failed_items = []

    for base_id, items in groups.items():
        total += 1

        all_correct = True
        group_failed = False
        for item in items:
            predict = item.get("predict", "")
            pred_pref = extract_preference(predict)
            if pred_pref is None:
                group_failed = True
                failed_extract += 1
                failed_items.append(item)
                break

            ranking = item["ranking"]
            label1, label2 = get_pair_labels(item["comparison_type"])
            gt_pref = get_gt_preference(ranking, label1, label2)

            if gt_pref == "tie":
                continue
            elif pred_pref != gt_pref:
                all_correct = False

        if group_failed:
            continue

        if all_correct:
            correct += 1

    return correct, total, failed_extract, failed_items


def load_data(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if content.startswith("["):
        return json.loads(content)

    data = []
    for i, line in enumerate(content.splitlines(), 1):
        line = line.strip()
        if line:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"⚠️ Skipped line {i}, reason: {e}")
                continue
    return data



def print_failed_items(failed_items, dataset_name):
    if not failed_items:
        return
    print(f"\n{'=' * 60}")
    print(f"[{dataset_name}] Failed extraction samples (total {len(failed_items)})")
    print("=" * 60)
    for i, item in enumerate(failed_items, 1):
        item_id = item.get("id", "N/A")
        predict = item.get("predict", "")
        print(f"\n--- Failed sample {i} | id: {item_id} ---")
        print(f"Full model output:\n{predict}")


def grade_single_file(input_file):
    data = load_data(input_file)

    pair2_items = []
    pair3_groups = defaultdict(list)
    pair4_groups = defaultdict(list)

    for item in data:
        dataset = item.get("dataset", "")
        if dataset == "2pair":
            pair2_items.append(item)
        elif dataset == "3pair":
            base_id = get_base_id(item["id"])
            pair3_groups[base_id].append(item)
        elif dataset == "4pair":
            base_id = get_base_id(item["id"])
            pair4_groups[base_id].append(item)

    results = {}

    c2, t2, f2, failed2 = evaluate_2pair(pair2_items)
    results["2pair"] = {"correct": c2, "total": t2, "failed_extract": f2}

    c3, t3, f3, failed3 = evaluate_multi_pair(pair3_groups, 3)
    results["3pair"] = {"correct": c3, "total": t3, "failed_extract": f3}

    c4, t4, f4, failed4 = evaluate_multi_pair(pair4_groups, 4)
    results["4pair"] = {"correct": c4, "total": t4, "failed_extract": f4}

    total_correct = c2 + c3 + c4
    total_all = t2 + t3 + t4
    total_failed = f2 + f3 + f4
    overall_acc = total_correct / total_all * 100 if total_all > 0 else 0

    return results, overall_acc, total_correct, total_all, total_failed


def print_single_result(input_file, results, overall_acc, total_correct, total_all, total_failed):
    print(f"\n{'=' * 60}")
    print(f"File: {os.path.basename(input_file)}")
    print("=" * 60)

    for name, r in results.items():
        acc = r["correct"] / r["total"] * 100 if r["total"] > 0 else 0
        print(f"  [{name}] correct/total: {r['correct']}/{r['total']}  accuracy: {acc:.2f}%  failed_extract: {r['failed_extract']}")

    print(f"  [Overall] correct/total: {total_correct}/{total_all}  accuracy: {overall_acc:.2f}%  failed_extract: {total_failed}")


def print_summary_table(all_results):
    if len(all_results) <= 1:
        return

    print(f"\n\n{'=' * 80}")
    print("EditReward-Bench Summary Table")
    print("=" * 80)

    header = f"{'Filename':<45} {'2pair':>8} {'3pair':>8} {'4pair':>8} {'Overall':>8}"
    print(header)
    print("-" * 80)

    for filename, (results, overall_acc, _, _, _) in all_results.items():
        accs = []
        for name in ["2pair", "3pair", "4pair"]:
            r = results[name]
            acc = r["correct"] / r["total"] * 100 if r["total"] > 0 else 0
            accs.append(f"{acc:.2f}%")
        accs.append(f"{overall_acc:.2f}%")
        print(f"{filename:<45} {accs[0]:>8} {accs[1]:>8} {accs[2]:>8} {accs[3]:>8}")

    print("=" * 80)


def main():
    if len(sys.argv) < 2:
        print("Usage: python grade_reward_bench.py <file_or_dir>")
        sys.exit(1)
    input_path = sys.argv[1]

    if os.path.isdir(input_path):
        files = sorted([
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.startswith("EditReward-Bench") and (f.endswith(".jsonl") or f.endswith(".json"))
        ])
        if not files:
            print(f"No files starting with EditReward-Bench found in {input_path}")
            sys.exit(1)

        print(f"Found {len(files)} files in {input_path}")
        all_results = {}
        for f in files:
            results, overall_acc, total_correct, total_all, total_failed = grade_single_file(f)
            print_single_result(f, results, overall_acc, total_correct, total_all, total_failed)
            all_results[os.path.basename(f)] = (results, overall_acc, total_correct, total_all, total_failed)

        print_summary_table(all_results)
    else:
        results, overall_acc, total_correct, total_all, total_failed = grade_single_file(input_path)
        print_single_result(input_path, results, overall_acc, total_correct, total_all, total_failed)


if __name__ == "__main__":
    main()
