#!/usr/bin/env python3
"""run_all_eval.sh が作ったログを集計し、Markdown表とCSVにまとめる。

使い方: python eval_tools/summarize_eval.py <ログのディレクトリ>
"""
import csv
import math
import re
import sys
from pathlib import Path

NAME_RE = re.compile(r"^(floating_arti|floating_grasp|fixed|compose)_(.+)_(test|train)$")
SCALAR_RE = re.compile(
    r"^\s*([A-Za-z][A-Za-z _]*?)\s*:?\s+([-+]?(?:nan|inf|\d+\.?\d*(?:[eE][-+]?\d+)?))\s*$", re.I)

# (表の列名, ログに出力される指標名)。指標名は評価スクリプトの表記(綴り含む)に合わせる。
COLUMNS = {
    "floating_arti": [
        ("success", "success rate"),
        ("success_once", "succeed once rate"),
        ("angle_err(succ)", "average final angle differnence"),
        ("angle_err(all)", "average final angle differnence all"),
        ("sim_dist(succ)", "average simulated distance"),
    ],
    "floating_grasp": [
        ("success", "success rate"),
        ("pos_err(succ)", "average final pos error"),
        ("pos_err(all)", "average final pos error all"),
        ("angle_err(succ)", "average final angle error"),
        ("angle_err(all)", "average final angle error all"),
    ],
    "fixed": [
        ("success", "success rate"),
        ("success_once", "succeed once rate"),
        ("angle_err(succ)", "final average angle differnence succeed"),
        ("angle_err(all)", "final average angle differnence all"),
    ],
    "compose": [
        ("success", "average success rate"),
        ("arrive", "average success arrive rate"),
        ("open", "average success open rate"),
        ("ori", "average success ori rate"),
        ("pos_err", "average pos error"),
        ("angle_err", "average angle error"),
        ("arti_angle_err", "average arti angle error"),
        ("succ|arrive", "ave success rate on successful arrival"),
        ("succ|grasp", "ave success rate on successful grasp"),
    ],
}
# floating/fixed は最後の "!!!" 行以降にエピソード平均のスカラー値がまとまって出力される
USES_MARKER = {"floating_arti", "floating_grasp", "fixed"}


def norm(key):
    return " ".join(key.lower().replace("_", " ").split())


def parse_scalars(lines, task):
    if task in USES_MARKER:
        marks = [i for i, line in enumerate(lines) if line.startswith("!!!")]
        if not marks:
            return {}
        lines = lines[marks[-1] + 1:]
    out = {}
    for line in lines:
        m = SCALAR_RE.match(line)
        if m:
            out[norm(m.group(1))] = float(m.group(2))
    return out


def last_line(lines):
    for line in reversed(lines):
        if line.strip():
            return line.strip()
    return ""


def fmt(v):
    return "-" if v is None or math.isnan(v) or math.isinf(v) else f"{v:.3f}"


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    out_dir = Path(sys.argv[1])
    runs = {}
    failures = []
    for log in sorted(out_dir.glob("*.log")):
        m = NAME_RE.match(log.stem)
        if not m:
            continue
        task, obj, split = m.groups()
        lines = log.read_text(errors="replace").splitlines()
        status_file = log.with_suffix(".status")
        code = status_file.read_text().strip() if status_file.exists() else "?"
        scalars = parse_scalars(lines, task)
        values = {label: scalars.get(norm(key)) for label, key in COLUMNS[task]}
        if code != "0" or all(v is None for v in values.values()):
            failures.append((log.stem, code, last_line(lines)))
        runs[(task, obj, split)] = values

    md, rows = [], []
    for task, cols in COLUMNS.items():
        task_runs = {k: v for k, v in runs.items() if k[0] == task}
        if not task_runs:
            continue
        labels = [c[0] for c in cols]
        md += [f"### {task}", "", "| object | split | " + " | ".join(labels) + " |",
               "|" + "---|" * (len(labels) + 2)]
        for (_, obj, split), values in sorted(task_runs.items(), key=lambda kv: (kv[0][2], kv[0][1])):
            md.append(f"| {obj} | {split} | " + " | ".join(fmt(values[l]) for l in labels) + " |")
            for label in labels:
                rows.append([task, obj, split, label, values[label]])
        for split in ("test", "train"):
            group = [v for (t, _, s), v in task_runs.items() if s == split]
            if not group:
                continue
            means = []
            for label in labels:
                vals = [v[label] for v in group
                        if v[label] is not None and math.isfinite(v[label])]
                means.append(fmt(sum(vals) / len(vals)) if vals else "-")
            md.append(f"| **mean** | {split} | " + " | ".join(means) + " |")
        md.append("")

    if failures:
        md += ["### failed / no metrics", "", "| run | exit code | last log line |", "|---|---|---|"]
        md += [f"| {n} | {c} | {msg.replace('|', '/')[:120]} |" for n, c, msg in failures]
        md.append("")
    md.append("mean は物体ごとの値の単純平均(非有限値は除外)。'-' は値なし。")

    (out_dir / "summary.md").write_text("\n".join(md) + "\n")
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "object", "split", "metric", "value"])
        w.writerows(rows)
    print("\n".join(md))
    print(f"\nsaved: {out_dir / 'summary.md'}, {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
