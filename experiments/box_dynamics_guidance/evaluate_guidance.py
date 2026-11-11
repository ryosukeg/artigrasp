#!/usr/bin/env python3
"""Paired box evaluation: released PPO versus the same PPO plus frozen guidance."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np

from collect import (HERE, ROOT, START, STEPS, HORIZON, RecordingController,
                     simulate_recorded, window_validity, atomic_npz, provenance,
                     prepare_data, load_environment_config, make_environment,
                     Policies, timing_for, lift_metrics, _paths)
from guidance import SETTINGS, Selector, ScheduledPolicy
from ood_suite.io import file_sha256, read_json, write_json
from ood_suite.targets import load_config, build_plan
from ood_suite.metrics import evaluate_episode, evaluate_goal_diagnostics, quaternion_geodesic_rad


def fingerprint(plan):
    value = provenance(plan)
    value["evaluation_sources"] = {name: file_sha256(HERE / name) for name in (
        "evaluate_guidance.py", "guidance.py", "dataset.py", "train_model.py", "train_smoke.py")}
    value["predictor_sha256"] = file_sha256(plan["predictor"]["path"])
    value["label_set_sha256"] = file_sha256(ROOT / (
        "rsc/mano_double/grasp_label/final_labels/{}_set/box.npy".format(plan["split"])))
    return value


def check_plan(output):
    plan = read_json(output / "plan.json")
    saved = read_json(output / "provenance.json")
    if saved["plan_sha256"] != file_sha256(output / "plan.json"):
        raise ValueError("Plan changed; use a new output directory")
    current = fingerprint(plan)
    for key, value in current.items():
        if saved[key] != value:
            raise ValueError("Evaluation provenance changed: {}".format(key))
    return plan


def tasks(plan):
    size = plan["config"]["runtime"]["batch_size"]
    result = []
    for profile in plan["profiles"]:
        group = [row for row in plan["episodes"] if row["profile"] == profile]
        for offset in range(0, len(group), size):
            for method in ("baseline", "guided"):
                result.append(("{}__{}__{:04d}".format(profile, method, offset // size),
                               method, group[offset:offset + size]))
    return result


def create_plan(args):
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh output directory, or resume with run")
    config = load_config(args.config)
    config["runtime"].update(batch_size=args.batch_size, num_threads=args.num_threads, device=args.device)
    plan = build_plan(ROOT, config, samples=args.samples, seed=args.seed,
                      split="train" if args.smoke else "test",
                      profiles=["normal"] if args.smoke else None)
    if plan["objects"] != ["box"]:
        raise ValueError("This predictor supports box only")
    checkpoint = args.checkpoint.resolve()
    training = read_json(checkpoint.parent / "report.json")
    if training["artifact_sha256"]["best.pt"] != file_sha256(checkpoint):
        raise ValueError("Predictor does not match its training report")
    plan["predictor"] = {"path": str(checkpoint), "sha256": file_sha256(checkpoint),
                         "best_epoch": training["best_epoch"],
                         "training_report_sha256": file_sha256(checkpoint.parent / "report.json"),
                         "training_episode_counts": training["dataset"]["episode_counts"],
                         "training_window_counts": training["dataset"]["window_counts"],
                         "val_model": training["val_model"], "val_persistence": training["val_persistence"]}
    plan["guidance"] = SETTINGS
    plan["methods"] = ["baseline", "guided"]
    plan["purpose"] = "technical_smoke" if args.smoke else "exploratory_single_seed_paired_comparison"
    plan["primary_metrics"] = ["ood_goal_success", "paper_success"]
    _paths(ROOT)
    Selector(checkpoint, args.device)  # Fail before creating artifacts on incompatible checkpoints.
    source = fingerprint(plan)
    old = read_json(Path(training["dataset"]["dataset_path"]) / "provenance.json")
    if old["runtime"]["extension_sha256"] != source["runtime"]["extension_sha256"]:
        raise ValueError("Simulator binary differs from response-model collection")
    shared = set(old["input_sha256"]) & set(source["input_sha256"])
    if any(old["input_sha256"][name] != source["input_sha256"][name] for name in shared):
        raise ValueError("Simulation assets/PPO differ from response-model collection")
    plan["shared_training_assets_verified"] = len(shared)
    write_json(output / "plan.json", plan)
    source["plan_sha256"] = file_sha256(output / "plan.json")
    source["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(output / "provenance.json", source)
    print("Frozen plan: {} episodes x 2 methods; {}".format(len(plan["episodes"]), output), flush=True)


def check_batch(output, task):
    name, method, specs = task
    value = read_json(output / "batches" / (name + ".json"))
    if not value.get("complete") or value["plan_sha256"] != file_sha256(output / "plan.json"):
        raise ValueError("Incomplete or changed batch")
    if [row["episode_id"] for row in value["episodes"]] != [row["episode_id"] for row in specs]:
        raise ValueError("Batch IDs do not match plan")
    for row in value["episodes"]:
        if row["method"] != method or file_sha256(output / row["trajectory_path"]) != row["trajectory_sha256"]:
            raise ValueError("Changed trajectory or method")
    return value


def worker(output, index):
    _paths(ROOT)
    plan = check_plan(output)
    task = tasks(plan)[index]
    name, method, specs = task
    if (output / "batches" / (name + ".json")).exists():
        check_batch(output, task)
        return
    wall_start = time.perf_counter()
    runtime = plan["config"]["runtime"]
    data = prepare_data(ROOT, specs)
    config = load_environment_config(ROOT, len(specs), False, runtime["num_threads"])
    if not config["environment"]["root_guided"] or config["environment"]["control_dt"] != .01:
        raise ValueError("Expected root-guided 0.01-second control")
    env = make_environment(ROOT, data, config, "box")
    try:
        env.seed(plan["seed"])
        base = Policies(ROOT, plan["config"]["checkpoints"], config, env, runtime["device"], plan["seed"])
        controller = RecordingController(env, data, specs, timing_for("box"), base)
        goals = np.concatenate((controller.goal[:, :7], controller.angle), axis=1).astype(np.float32)
        schedules = np.zeros((len(specs), 15, 3), np.float32)
        selector = Selector(plan["predictor"]["path"], runtime["device"]) if method == "guided" else None
        policy = ScheduledPolicy(base, env, goals, schedules, selector)
        before = time.perf_counter()
        arrays, goals, initial, terminal, invalid, first = simulate_recorded(env, controller, policy, schedules)
        rollout_seconds = time.perf_counter() - before
        rows = []
        for i, spec in enumerate(specs):
            saved = {key: value[:, i] for key, value in arrays.items()}
            for key in ("grasp", "articulation"):
                saved.update({"{}_{}".format(key, j): value[i] for j, value in enumerate(data[key])})
            for key in ("left", "right", "arti_left", "arti_right"):
                saved[key] = data[key][i]
            saved.update(goal=goals[i], window_z=schedules[i],
                         window_start_step=np.arange(START, STEPS, HORIZON, dtype=np.int32),
                         window_end_step=np.arange(START + HORIZON, STEPS + 1, HORIZON, dtype=np.int32),
                         window_valid=window_validity(arrays, i, initial[i], terminal[i]),
                         initial_invalid=initial[i], terminal_bad_observation=terminal[i],
                         control_dt=np.asarray(.01), horizon=np.asarray(HORIZON), schema_version=np.asarray(1),
                         guidance_selected=policy.selected[:, i],
                         guidance_fallback=policy.fallback[:, i])
            if selector is not None:
                saved.update(guidance_predictions=policy.predictions[:, i], guidance_scores=policy.scores[:, i])
            metric_spec = controller.metric_spec(i)
            row = dict(metric_spec, **evaluate_episode(saved["states"], metric_spec, controller.timing))
            valid = row["valid"] and not bool(invalid[i])
            if not valid:
                row.update({key: False for key in row if key.endswith("success")})
                row.update({key: None for key in row if "_error_" in key})
            row.update(lift_metrics(saved["states"], metric_spec, controller.timing, valid))
            row.update(evaluate_goal_diagnostics(saved["states"], metric_spec, controller.timing, valid))
            entry = saved["final_states"][0]
            entry_good = np.isfinite(entry).all() and np.linalg.norm(entry[125:129]) > 1e-8
            prefix_valid = not bool(initial[i]) and entry_good and not any(
                arrays[key][:START, i].any() for key in
                ("bad_observation", "bad_action", "bad_state", "auto_reset"))
            prior_lift = lift_metrics(saved["states"], metric_spec, controller.timing, prefix_valid)
            row["opening_entry"] = {
                "position_error_m": float(np.linalg.norm(entry[122:125] - goals[i, :3])) if entry_good else None,
                "orientation_error_rad": quaternion_geodesic_rad(entry[125:129], goals[i, 3:7]) if entry_good else None,
                "articulation_rad": float(entry[129]) if entry_good else None,
                "prior_rollout_valid": bool(prefix_valid),
                "prior_lift_success": prior_lift["lift_target_success"]}
            path = Path("trajectories") / method / (spec["episode_id"] + ".npz")
            atomic_npz(output / path, saved)
            row.update(method=method, valid=valid, first_invalid_step=int(first[i]),
                       guidance_candidate_counts=np.bincount(policy.selected[:, i], minlength=7).tolist(),
                       guidance_fallback_count=int(policy.fallback[:, i].sum()),
                       trajectory_path=str(path), trajectory_sha256=file_sha256(output / path),
                       environment_config=config, timing=controller.timing, policy_device=str(base.device))
            rows.append(row)
        record = {"complete": True, "plan_sha256": file_sha256(output / "plan.json"), "episodes": rows,
                  "performance": {"wall_seconds": time.perf_counter() - wall_start,
                                  "rollout_seconds": rollout_seconds,
                                  "guidance_seconds": selector.seconds if selector else 0.,
                                  "environments": len(specs), "policy_forward_devices": base.forward_devices}}
        write_json(output / "batches" / (name + ".json"), record)
        print("Committed {}: {} trials, {:.1f}s".format(name, len(rows), record["performance"]["wall_seconds"]), flush=True)
    finally:
        env.close()


def run(output):
    _paths(ROOT)
    plan = check_plan(output)
    with (output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = []
        planned = tasks(plan)
        for index, task in enumerate(planned):
            name = task[0]
            if not (output / "batches" / (name + ".json")).exists():
                log = output / "logs" / (name + ".log")
                log.parent.mkdir(parents=True, exist_ok=True)
                print("Batch {}/{}: {}".format(index + 1, len(planned), name), flush=True)
                with log.open("w") as handle:
                    process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                                                "worker", "--output", str(output), "--batch-index", str(index)],
                                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                               universal_newlines=True, bufsize=1, start_new_session=True)
                    try:
                        for line in process.stdout:
                            print(line, end="", flush=True)
                            handle.write(line)
                            handle.flush()
                        if process.wait() != 0:
                            raise RuntimeError("Evaluation worker failed: {}".format(log))
                    except BaseException:
                        import signal
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGTERM)
                            process.wait()
                        raise
            records.append(check_batch(output, task))
        write_json(output / "results.json", {"complete": True,
                   "plan_sha256": file_sha256(output / "plan.json"),
                   "episodes": [row for record in records for row in record["episodes"]],
                   "batch_performance": [record["performance"] for record in records]})
        from evaluate_report import write_report
        write_report(output)
        print("Comparison complete: {}".format(output / "COMPARISON_REVIEW.md"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "worker", "report"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=HERE / "configs/evaluation.json")
    parser.add_argument("--checkpoint", type=Path, default=HERE / "checkpoints/box_response_1000_seed301_full/best.pt")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=202)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--num-threads", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if min(args.samples, args.batch_size, args.num_threads) < 1:
        parser.error("samples, batch size and thread count must be positive")
    if args.command == "plan":
        create_plan(args)
    elif args.command == "worker":
        worker(args.output, args.batch_index)
    elif args.command == "run":
        run(args.output)
    else:
        from evaluate_report import write_report
        write_report(args.output)


if __name__ == "__main__":
    main()
