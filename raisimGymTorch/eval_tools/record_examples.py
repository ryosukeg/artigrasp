#!/usr/bin/env python3
"""Record success/failure example videos per task and object via RaisimUnity.

Episodes are chosen from the logs of run_all_eval.sh, re-run with the selected trial
placed in env 0 (the only visualized env), and the videos are sorted by the outcome
of the re-run.

Usage (RaisimUnity must be running and connected, i.e. Auto-connect checked):
    python eval_tools/record_examples.py --logs eval_results/paper8_test \
        --out /home/cvrt/goto/artigrasp/Result_mp4 [--tasks grasp ...] [--objs box ...]
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
UNITY_VIDEO_DIR = ROOT.parent / "raisimUnity" / "linux" / "Screenshot"

TASKS = {
    # folder name: (log prefix, runner script, extra args, success key in log, envs per label)
    "grasp": ("floating_grasp", "floating_evaluation/runner_eval.py", ["-grasp"], "success", 30),
    "articulation_free_base": ("floating_arti", "floating_evaluation/runner_eval.py", [], "succeed", 5),
    "articulation_fixed_base": ("fixed", "fixed_arti_evaluation/runner_eval.py", [], "succeed", 5),
}
OBJECTS = ["box", "espressomachine", "ketchup", "laptop", "microwave", "mixer", "notebook", "waffleiron"]


def parse_outcomes(log_path, key, n_envs):
    """Return a (labels, envs) 0/1 array of per-trial success from an evaluation log."""
    text = log_path.read_text(errors="replace")
    rows = []
    for arr in re.findall(rf"^{key}:\s+(\[\[.*?\]\])", text, re.M | re.S):
        vals = [float(v) for v in arr.strip("[]").split()]
        if len(vals) != n_envs:
            raise ValueError(f"{log_path}: expected {n_envs} values per label, got {len(vals)}")
        rows.append(vals)
    return np.array(rows)


def pick(labels_envs, n):
    """Pick up to n trials, spreading over distinct labels first, preferring env 0."""
    by_label = {}
    for label, env in labels_envs:
        by_label.setdefault(label, []).append(env)
    labels = sorted(by_label)
    chosen = []
    for idx in np.linspace(0, len(labels) - 1, min(n, len(labels))).round().astype(int):
        label = labels[idx]
        env = 0 if 0 in by_label[label] else by_label[label][0]
        chosen.append((label, env))
    # not enough distinct labels: fill with other envs of the same labels
    for label in labels:
        for env in by_label[label]:
            if len(chosen) >= n:
                return chosen
            if (label, env) not in chosen:
                chosen.append((label, env))
    return chosen


def target_tag(task, env):
    if task == "grasp":
        return f"target{env:02d}"
    return f"angle{0.5 + env * 0.25:.2f}rad"


def wait_for_file(path, timeout=60):
    """Wait until RaisimUnity has written the file and its size stops changing."""
    deadline = time.time() + timeout
    last = -1
    while time.time() < deadline:
        if path.exists():
            size = path.stat().st_size
            if size > 0 and size == last:
                return True
            last = size
        time.sleep(1)
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", default="eval_results/paper8_test")
    ap.add_argument("--out", default=str(ROOT.parent / "Result_mp4"))
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--objs", nargs="+", default=OBJECTS)
    ap.add_argument("--n", type=int, default=3, help="examples per outcome")
    args = ap.parse_args()

    os.chdir(ROOT)
    try:
        # importing two env modules in one process clashes in pybind11, so check just one
        from raisimGymTorch.env.bin import fixed_arti_evaluation  # noqa: F401
    except ImportError:
        sys.exit(f"Cannot import raisimGymTorch envs with {sys.executable}. Run 'conda activate artigrasp' first.")
    logs, out = Path(args.logs), Path(args.out)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    report = []

    for task in args.tasks:
        prefix, script, extra, key, n_envs = TASKS[task]
        for obj in args.objs:
            log_path = logs / f"{prefix}_{obj}_test.log"
            if not log_path.exists():
                print(f"[skip] {task}/{obj}: no log ({log_path})")
                continue
            outcomes = parse_outcomes(log_path, key, n_envs)
            if outcomes.size == 0:
                print(f"[skip] {task}/{obj}: no labels")
                continue
            # re-runs occasionally flip near-threshold trials, so keep backup candidates
            pools = {"success": pick([tuple(x) for x in np.argwhere(outcomes == 1)], outcomes.size),
                     "failure": pick([tuple(x) for x in np.argwhere(outcomes == 0)], outcomes.size)}
            saved = {"success": 0, "failure": 0}
            for rnd in range(4):
                selection = []
                for outcome in pools:
                    need = args.n - saved[outcome]
                    selection += pools[outcome][:need]
                    pools[outcome] = pools[outcome][need:]
                if not selection:
                    break
                sel_arg = ",".join(f"{l}:{e}" for l, e in selection)
                json_path = Path(f"/tmp/record_{task}_{obj}.json")
                run_log = Path(f"/tmp/record_{task}_{obj}.log")
                print(f"[run {rnd}] {task}/{obj}: {sel_arg}")

                cmd = [sys.executable, f"raisimGymTorch/env/envs/{script}", "-obj", obj, "-test",
                       "-select", sel_arg, "-video", "-select_out", str(json_path)] + extra
                json_path.unlink(missing_ok=True)
                with open(run_log, "w") as f:
                    code = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
                if code != 0 or not json_path.exists():
                    print(f"  [error] exit={code}, see {run_log}")
                    report.append((task, obj, "run failed", run_log))
                    break
                if "is established" not in run_log.read_text(errors="replace"):
                    sys.exit("RaisimUnity did not connect. Check 'Auto-connect' in RaisimUnity and re-run.")

                for ep in json.loads(json_path.read_text()):
                    outcome = "success" if ep["success"] == 1 else "failure"
                    src = UNITY_VIDEO_DIR / ep["video"]
                    if not wait_for_file(src):
                        print(f"  [missing video] {src}")
                        report.append((task, obj, "missing video", src))
                        continue
                    if saved[outcome] >= args.n:
                        src.unlink()
                        continue
                    name = f"label{ep['label']:02d}_{ep['seq_name']}_{target_tag(task, ep['env'])}.mp4"
                    dst = out / task / outcome / obj / name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), dst)
                    saved[outcome] += 1
                    print(f"  {outcome}: {dst.relative_to(out)}")
            for outcome, count in saved.items():
                if count < args.n:
                    report.append((task, obj, f"only {count} {outcome} example(s)", ""))

    if report:
        print("\nIssues:")
        for item in report:
            print("  ", *item)


if __name__ == "__main__":
    main()
