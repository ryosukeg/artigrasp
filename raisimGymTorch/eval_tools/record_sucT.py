#!/usr/bin/env python3
"""Record strict Suc. T success/failure example videos per object via RaisimUnity.

Trials come from eval_results/sucT_strict/<obj>.json (written by compose_eval_strict.py /
ketchup_eval_strict.py). Each chosen trial is re-run alone (targets are seeded, so it gets the
same target) with the trial shown as env 0. A single-trial run does not always reproduce the
full-batch outcome, so each video is filed by the outcome of the run it shows, and manifest.csv
keeps both the batch and the single-run outcome.

Usage (RaisimUnity must be running with Auto-connect checked):
    conda activate artigrasp
    python eval_tools/record_sucT.py --out /home/cvrt/goto/artigrasp/Result_mp4/sucT [--objs box ...]
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UNITY_VIDEO_DIR = ROOT.parent / "raisimUnity" / "linux" / "Screenshot"
OBJECTS = ["box", "espressomachine", "ketchup", "laptop", "microwave", "mixer", "notebook", "waffleiron"]
KETCHUP_TYPE = {"ketchup", "mixer"}


def spread(items, k):
    """k items evenly spread over a list."""
    if not items:
        return []
    if len(items) <= k:
        return list(items)
    return [items[round(i * (len(items) - 1) / (k - 1))] for i in range(k)] if k > 1 else [items[len(items) // 2]]


def ordered_candidates(trials, success, k):
    """Deterministic candidate order: a diverse first pick of k, then the remaining trials as backups."""
    pool = [t for t in trials if t["success"] == success]
    pool.sort(key=lambda t: (t["arti_label"], t["grasp_label"], t["target_angle"]))
    if success:
        first = spread(pool, k)
    else:
        # cover different failure modes (which of PE/AE/AAE failed) before repeating one
        groups = defaultdict(list)
        for t in pool:
            groups[(t["pe_ok"], t["ae_ok"], t["aae_ok"])].append(t)
        order = sorted(groups.values(), key=len, reverse=True)
        first = []
        while len(first) < min(k, len(pool)):
            for g in order:
                pick = spread([t for t in g if t not in first], 1)
                if pick and len(first) < k:
                    first.append(pick[0])
    rest = [t for t in spread(pool, len(pool)) if t not in first]
    return first + rest


def failure_tag(t):
    parts = [n for n, ok in (("PE", t["pe_ok"]), ("AE", t["ae_ok"]), ("AAE", t["aae_ok"])) if not ok]
    return "fail-" + "-".join(parts) if parts else "success"


def wait_for_file(path, timeout=90):
    """Wait until RaisimUnity has written the file and its size stops changing."""
    deadline, last = time.time() + timeout, -1
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
    ap.add_argument("--results", default="eval_results/sucT_strict")
    ap.add_argument("--out", default=str(ROOT.parent / "Result_mp4" / "sucT"))
    ap.add_argument("--objs", nargs="+", default=OBJECTS)
    ap.add_argument("--n", type=int, default=3, help="examples per outcome")
    ap.add_argument("--np_seed", type=int, default=0)
    ap.add_argument("--max_attempts", type=int, default=12, help="single-trial runs per object")
    args = ap.parse_args()

    os.chdir(ROOT)
    try:
        from raisimGymTorch.env.bin import compose_eval  # noqa: F401
    except ImportError:
        sys.exit(f"Cannot import raisimGymTorch envs with {sys.executable}. Run 'conda activate artigrasp' first.")
    results, out = Path(args.results), Path(args.out)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    issues = []
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.csv"
    new_manifest = not manifest_path.exists()
    manifest = open(manifest_path, "a", newline="")
    writer = csv.writer(manifest)
    if new_manifest:
        writer.writerow(["object", "trial", "file", "batch_outcome", "single_run_outcome", "PE", "AE", "AAE"])

    for obj in args.objs:
        data = json.loads((results / f"{obj}.json").read_text())
        if data["np_seed"] != args.np_seed:
            sys.exit(f"{obj}.json was made with np_seed={data['np_seed']}, not {args.np_seed}")
        script = "ketchup_eval_strict.py" if obj in KETCHUP_TYPE else "compose_eval_strict.py"
        # slots already filled by an earlier run are kept
        saved = {w: len(list((out / w / obj).glob("*.mp4"))) for w in ("success", "failure")}
        succ_c = ordered_candidates(data["trials"], True, args.n)
        fail_c = ordered_candidates(data["trials"], False, args.n)
        queue = [t for pair in zip(succ_c, fail_c) for t in pair]
        queue += succ_c[len(fail_c):] + fail_c[len(succ_c):]
        attempts = 0
        for t in queue:
            if all(saved[w] >= args.n for w in saved) or attempts >= args.max_attempts:
                break
            tid = t["trial"]
            batch_outcome = "success" if t["success"] else "failure"
            attempts += 1
            json_path, run_log = Path(f"/tmp/sucT_{obj}_{tid}.json"), Path(f"/tmp/sucT_{obj}_{tid}.log")
            json_path.unlink(missing_ok=True)
            cmd = [sys.executable, f"raisimGymTorch/env/envs/compose_eval/{script}", "-obj", obj, "-test",
                   "-select", str(tid), "-video", "-np_seed", str(args.np_seed), "-out", str(json_path)]
            print(f"[run] {obj} trial {tid} (batch: {batch_outcome})")
            with open(run_log, "w") as f:
                code = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
            if code != 0 or not json_path.exists():
                print(f"  [error] exit={code}, see {run_log}")
                issues.append((obj, tid, "run failed", str(run_log)))
                continue
            if "is established" not in run_log.read_text(errors="replace"):
                sys.exit("RaisimUnity did not connect. Check 'Auto-connect' in RaisimUnity and re-run.")
            rerun = json.loads(json_path.read_text())["trials"][0]
            src = UNITY_VIDEO_DIR / rerun["video"]
            if not wait_for_file(src):
                print(f"  [missing video] {src}")
                issues.append((obj, tid, "missing video", str(src)))
                continue
            got = "success" if rerun["success"] else "failure"
            if saved[got] >= args.n:
                print(f"  single run: {got} (already have {args.n}); discarded")
                src.unlink()
                continue
            name = (f"trial{tid:04d}_{rerun['grasp_seq']}__{rerun['arti_seq']}"
                    f"_angle{rerun['target_angle']:.2f}rad_{failure_tag(rerun)}.mp4")
            dst = out / got / obj / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), dst)
            saved[got] += 1
            writer.writerow([obj, tid, str(dst.relative_to(out)), batch_outcome, got,
                             f"{rerun['pe']:.3f}", f"{rerun['ae']:.3f}", f"{rerun['aae']:.3f}"])
            manifest.flush()
            note = "" if got == batch_outcome else f"  (batch: {batch_outcome})"
            print(f"  {got}: {dst.relative_to(out)}  PE={rerun['pe']:.3f} AE={rerun['ae']:.3f} AAE={rerun['aae']:.3f}{note}")
        for want, count in saved.items():
            if count < args.n:
                issues.append((obj, "-", f"only {count} {want} example(s)", ""))
    manifest.close()

    if issues:
        print("\nIssues:")
        for item in issues:
            print("  ", *item)


if __name__ == "__main__":
    main()
