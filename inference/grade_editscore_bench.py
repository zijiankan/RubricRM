import json
import re
import os
import glob
import argparse
import numpy as np
from collections import defaultdict


TASK_GROUPS = {
    "Subject": ["subject-add", "subject-remove", "subject-replace"],
    "Appear.": ["color_alter", "material_alter", "style_change", "tone_transfer"],
    "Scene": ["background_change", "extract"],
    "Advanced": ["ps_human", "text_change", "motion_change", "compose"],
}

DIMENSION_MAP = {
    "prompt_following": "PF",
    "consistency": "C",
    "overall": "O",
}


def parse_boxed(text):
    if not text:
        return None
    patterns = [
        r"\\boxed\{([AB])\}",
        r"\\boxed\{\{([AB])\}\}",
        r"boxed\{([AB])\}",
        r"boxed\{\{([AB])\}\}",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1]

    stripped = text.strip()
    if stripped in ("A", "B"):
        return stripped

    return None


def load_data(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if content.startswith("["):
        return json.loads(content)
    else:
        return [json.loads(line) for line in content.splitlines() if line.strip()]


def evaluate(input_file):
    data = load_data(input_file)

    if not data or "predict" not in data[0]:
        print(f"Skipped (no predict field)")
        return None

    print(f"Total {len(data)} records")

    stats = defaultdict(lambda: defaultdict(list))
    parse_fail_count = 0

    for item in data:
        task_type = item["task_type"]
        dimension = item["dimension"]
        chosen = item["chosen"]
        predict_text = item.get("predict", "")

        prediction = parse_boxed(predict_text)
        if prediction is None:
            parse_fail_count += 1
            stats[task_type][dimension].append(0)
        else:
            stats[task_type][dimension].append(1 if prediction == chosen else 0)

    if parse_fail_count > 0:
        print(f"Warning: {parse_fail_count} records failed to parse \\boxed{{}}, treated as incorrect")
    print()

    group_results = {}

    for group_name, task_types in TASK_GROUPS.items():
        group_results[group_name] = {}
        for dim_full, dim_short in DIMENSION_MAP.items():
            task_accs = []
            for tt in task_types:
                if tt in stats and dim_full in stats[tt]:
                    acc = np.mean(stats[tt][dim_full])
                    task_accs.append(acc)
            if task_accs:
                group_results[group_name][dim_short] = np.mean(task_accs)
            else:
                group_results[group_name][dim_short] = None

    group_results["Overall"] = {}
    for dim_full, dim_short in DIMENSION_MAP.items():
        all_flags = [v for tt in stats for v in stats[tt].get(dim_full, [])]
        if all_flags:
            group_results["Overall"][dim_short] = sum(all_flags) / len(all_flags)
        else:
            group_results["Overall"][dim_short] = None

    display_order = ["Overall", "Subject", "Appear.", "Scene", "Advanced"]
    print("\n===== Summary =====\n")
    header = f"{'':12s} {'PF':>8s} {'C':>8s} {'O':>8s}"
    print(header)
    print("-" * 40)
    for group_name in display_order:
        if group_name not in group_results:
            continue
        pf_val = group_results[group_name].get("PF")
        c_val = group_results[group_name].get("C")
        o_val = group_results[group_name].get("O")
        pf_str = f"{pf_val:.4f}" if pf_val is not None else "N/A"
        c_str = f"{c_val:.4f}" if c_val is not None else "N/A"
        o_str = f"{o_val:.4f}" if o_val is not None else "N/A"
        print(f"{group_name:12s} {pf_str:>8s} {c_str:>8s} {o_str:>8s}")
    print("-" * 40)

    total_correct = sum(1 for tt in stats for dim in stats[tt] for v in stats[tt][dim] if v == 1)
    total_count = sum(len(stats[tt][dim]) for tt in stats for dim in stats[tt])
    sample_acc = total_correct / total_count if total_count > 0 else 0
    print(f"\nOverall accuracy: {total_correct}/{total_count} = {sample_acc:.4f}")

    return {
        "group_results": group_results,
        "sample_acc": sample_acc,
        "total_correct": total_correct,
        "total_count": total_count,
    }


def main():
    parser = argparse.ArgumentParser(description="EditReward-Bench evaluation script")
    parser.add_argument("input_path", type=str, help="Path to inference result file or folder")
    args = parser.parse_args()

    input_path = args.input_path

    if os.path.isdir(input_path):
        patterns = [
            os.path.join(input_path, "EditScore-Bench*"),
        ]
        files = []
        for p in patterns:
            files.extend(glob.glob(p))
        files = sorted([f for f in files if os.path.isfile(f)])
        if not files:
            print(f"No files starting with EditScore-Bench found in {input_path}")
            return
        print(f"Found {len(files)} files:\n")
        for f in files:
            print(f"  {os.path.basename(f)}")
        print()
        all_results = {}
        for f in files:
            print("=" * 60)
            print(f"▶ {os.path.basename(f)}")
            print("=" * 60)
            res = evaluate(f)
            if res is not None:
                all_results[os.path.basename(f)] = res
            print()

        if len(all_results) >= 1:
            print_cross_file_summary(all_results)
    else:
        evaluate(input_path)


def print_cross_file_summary(all_results):
    display_order = ["Overall", "Subject", "Appear.", "Scene", "Advanced"]

    print("\n" + "=" * 80)
    print("Cross-file summary: PF / C / O results per file")
    print("=" * 80)

    col_width = 7
    header1 = f"{'Filename':<42s}"
    header2 = f"{'':<42s}"
    for g in display_order:
        header1 += f" | {g:^{col_width*3+2}s}"
        header2 += f" | {'PF':>{col_width}s} {'C':>{col_width}s} {'O':>{col_width}s}"
    print(header1)
    print(header2)
    print("-" * len(header2))

    for fname, res in all_results.items():
        gr = res["group_results"]
        row = f"{fname:<42s}"
        for g in display_order:
            if g in gr:
                for k in ["PF", "C", "O"]:
                    v = gr[g].get(k)
                    s = f"{v:.4f}" if v is not None else "N/A"
                    row += f" {s:>{col_width}s}" if k == "PF" else f" {s:>{col_width}s}"
                row = row
            else:
                row += f" {'N/A':>{col_width}s} {'N/A':>{col_width}s} {'N/A':>{col_width}s}"
        print(row)

    print("=" * 80)

    print("\nSimplified (Overall only):")
    print(f"{'Filename':<42s} {'PF':>7s} {'C':>7s} {'O':>7s} {'SampleAcc':>9s}")
    print("-" * 75)
    for fname, res in all_results.items():
        gr = res["group_results"].get("Overall", {})
        pf = gr.get("PF"); c = gr.get("C"); o = gr.get("O")
        pf_s = f"{pf:.4f}" if pf is not None else "N/A"
        c_s = f"{c:.4f}" if c is not None else "N/A"
        o_s = f"{o:.4f}" if o is not None else "N/A"
        sa = f"{res['sample_acc']:.4f}"
        print(f"{fname:<42s} {pf_s:>7s} {c_s:>7s} {o_s:>7s} {sa:>9s}")
    print("-" * 75)


if __name__ == "__main__":
    main()
