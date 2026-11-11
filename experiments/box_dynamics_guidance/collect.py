#!/usr/bin/env python3
"""Collect fixed-PPO, 20-step wrist-residual responses for box only.

Run `plan`, then `run --output DIR`; interrupted runs reuse committed batches.
The released evaluator, controller, environment and checkpoints are unchanged.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from ood_suite.io import code_fingerprint, file_sha256, read_json, write_json
from ood_suite.runtime import (Controller, Policies, prepare_data, make_environment,
                               load_environment_config, timing_for, total_steps,
                               lift_metrics, _paths, _float)

START, STEPS, HORIZON = 1260, 1560, 20
SCHEMA_VERSION = 1


class RecordingController(Controller):
    def switch(self, left, right):
        super().switch(left, right)
        self.modes_l = np.asarray(left, dtype=np.int32).copy()
        self.modes_r = np.asarray(right, dtype=np.int32).copy()


def provenance(plan):
    from ood_eval import _asset_hashes
    from raisimGymTorch.env.bin import compose_eval
    import torch
    result = code_fingerprint(ROOT)
    # Readers, reports and smoke fitting do not change simulated transitions.
    # Their own outputs record their source hashes separately.
    result["collector_source_sha256"] = {
        str((HERE / name).relative_to(ROOT)): file_sha256(HERE / name)
        for name in ("collect.py", "planning.py")}
    result["input_sha256"] = _asset_hashes(plan)
    # URDF hashes alone do not detect changed collision geometry.
    for relative in list(result["input_sha256"]):
        if not relative.endswith(".urdf"):
            continue
        urdf = ROOT / relative
        for mesh in ET.parse(urdf).getroot().iter("mesh"):
            path = (urdf.parent / mesh.attrib["filename"]).resolve()
            result["input_sha256"][str(path.relative_to(ROOT))] = file_sha256(path)
    labels = "rsc/mano_double/grasp_label/final_labels/train_set/box.npy"
    result["input_sha256"][labels] = file_sha256(ROOT / labels)
    result["runtime"] = {
        "python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
        "extension_path": str(Path(compose_eval.__file__).resolve()),
        "extension_sha256": file_sha256(compose_eval.__file__)}
    return result


def check_provenance(output):
    plan = read_json(output / "plan.json")
    saved = read_json(output / "provenance.json")
    if saved["plan_sha256"] != file_sha256(output / "plan.json"):
        raise ValueError("Plan changed; use a new output directory")
    current = provenance(plan)
    for key in current:
        if current[key] != saved[key]:
            raise ValueError("Provenance changed ({}); use a new output directory".format(key))
    return plan


def make_plan(args):
    from planning import build_collection_plan
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output is not empty; resume with run, or use a new directory")
    plan = build_collection_plan(
        ROOT, args.kind, args.baseline.resolve(), seed=args.seed, episodes=args.episodes,
        amplitude=args.amplitude, batch_size=args.batch_size,
        num_threads=args.num_threads, device=args.device)
    _paths(ROOT)
    fingerprint = provenance(plan)
    fingerprint["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(output / "plan.json", plan)
    fingerprint["plan_sha256"] = file_sha256(output / "plan.json")
    write_json(output / "provenance.json", fingerprint)
    print("Planned {} {} episodes: {}".format(len(plan["episodes"]), args.kind, output), flush=True)


def check_observation_layout(obs_r, states, goal, valid):
    """Refuse silently mismatched compiled environments / index conventions."""
    if not np.any(valid):
        return
    np.testing.assert_allclose(obs_r[valid, 273], states[valid, 129], atol=2e-5, rtol=1e-5)
    np.testing.assert_allclose(obs_r[valid, 273] + obs_r[valid, 275],
                               goal[valid, 7], atol=2e-5, rtol=1e-5)
    np.testing.assert_allclose(states[valid, 130:138], goal[valid], atol=2e-5, rtol=1e-5)


def simulate_recorded(env, controller, policy, schedules):
    """Record unnormalised inputs and actual executed actions, without extra steps."""
    count = controller.count
    if (env.num_obs_r, env.num_obs_l, env.num_acts, env.num_gs) != (276, 276, 51, 138):
        raise ValueError("Unexpected compiled environment dimensions")
    if total_steps(controller.timing) != STEPS or schedules.shape != (count, 15, 3):
        raise ValueError("Only box H=20/final 300-step schedules are supported")
    if not np.isfinite(schedules).all():
        raise ValueError("Nonfinite residual schedule")
    arrays = {"states": np.empty((STEPS + 1, count, 138), np.float32)}
    for key in ("actions_r", "actions_l", "ppo_actions_r", "ppo_actions_l"):
        arrays[key] = np.empty((STEPS, count, 51), np.float32)
    arrays["residuals"] = np.zeros((STEPS, count, 3), np.float32)
    for key in ("modes_r", "modes_l"):
        arrays[key] = np.empty((STEPS, count), np.int32)
    for key in ("bad_observation", "bad_action", "bad_state", "auto_reset"):
        arrays[key] = np.zeros((STEPS, count), bool)
    for side in ("r", "l"):
        arrays["final_obs_" + side] = np.empty((301, count, 276), np.float32)
    arrays["final_states"] = np.empty((301, count, 138), np.float32)
    goal = np.concatenate((controller.goal[:, :7], controller.angle), axis=1).astype(np.float32)
    controller.reset()
    arrays["states"][0] = env.get_global_state()
    initial_invalid = ~np.isfinite(arrays["states"][0]).all(axis=1)
    invalid = initial_invalid.copy()
    first_invalid = np.where(invalid, 0, -1).astype(np.int32)
    wall_start = time.perf_counter()
    for step in range(STEPS):
        controller.phase(step)
        obs_r, obs_l = env.observe(update_mean=False)
        bad = ~np.isfinite(obs_r).all(axis=1) | ~np.isfinite(obs_l).all(axis=1)
        arrays["bad_observation"][step] = bad
        invalid |= bad
        arrays["modes_r"][step] = controller.modes_r
        arrays["modes_l"][step] = controller.modes_l
        if step >= START:
            offset = step - START
            if not np.all(controller.modes_r == 8) or not np.all(controller.modes_l == 7):
                raise ValueError("Residual outside expected right mode 8 / left mode 7")
            if policy.articulation:
                raise ValueError("Box must use cooperative_right throughout")
            arrays["final_obs_r"][offset] = obs_r
            arrays["final_obs_l"][offset] = obs_l
            # phase(1260) resets the hands. states[1260] deliberately retains
            # the old evaluator semantics, while this is AFTER the phase reset.
            state = env.get_global_state()
            arrays["final_states"][offset] = state
            arrays["bad_state"][step] |= ~np.isfinite(state).all(axis=1)
            invalid |= arrays["bad_state"][step]
            check_observation_layout(obs_r, state, goal, ~invalid)
        right, left = policy.actions(
            np.nan_to_num(obs_r, nan=0., posinf=0., neginf=0.),
            np.nan_to_num(obs_l, nan=0., posinf=0., neginf=0.))
        arrays["ppo_actions_r"][step], arrays["ppo_actions_l"][step] = right, left
        right, left = right.copy(), left.copy()
        if step >= START:
            z = schedules[:, (step - START) // HORIZON]
            right[:, :3] += z
            arrays["residuals"][step] = z
        bad = ~np.isfinite(right).all(axis=1) | ~np.isfinite(left).all(axis=1)
        arrays["bad_action"][step] = bad
        invalid |= bad
        right = _float(np.nan_to_num(right, nan=0., posinf=0., neginf=0.))
        left = _float(np.nan_to_num(left, nan=0., posinf=0., neginf=0.))
        arrays["actions_r"][step], arrays["actions_l"][step] = right, left
        _, _, done = env.step(right, left)
        arrays["states"][step + 1] = env.get_global_state()
        arrays["auto_reset"][step] = np.asarray(done, bool)
        arrays["bad_state"][step] |= ~np.isfinite(arrays["states"][step + 1]).all(axis=1)
        invalid |= arrays["auto_reset"][step] | arrays["bad_state"][step]
        first_invalid[(first_invalid < 0) & invalid] = step
        if step and step % 500 == 0:
            print("  {}/{} steps | {:.1f} s".format(step, STEPS, time.perf_counter() - wall_start), flush=True)
    obs_r, obs_l = env.observe(update_mean=False)
    arrays["final_obs_r"][-1], arrays["final_obs_l"][-1] = obs_r, obs_l
    arrays["final_states"][-1] = env.get_global_state()
    terminal_bad = (~np.isfinite(obs_r).all(axis=1) | ~np.isfinite(obs_l).all(axis=1) |
                    ~np.isfinite(arrays["final_states"][-1]).all(axis=1))
    invalid |= terminal_bad
    first_invalid[(first_invalid < 0) & invalid] = STEPS
    check_observation_layout(obs_r, arrays["final_states"][-1], goal, ~invalid)
    return arrays, goal, initial_invalid, terminal_bad, invalid, first_invalid


def window_validity(arrays, index, initial_invalid, terminal_bad):
    event = np.zeros(STEPS, bool)
    for key in ("bad_observation", "bad_action", "bad_state", "auto_reset"):
        event |= arrays[key][:, index]
    result = []
    for start in range(0, 300, HORIZON):
        end = start + HORIZON
        # Any earlier reset invalidates all later windows in this episode too.
        good = not initial_invalid and not event[:START + end].any()
        for key in ("final_obs_r", "final_obs_l", "final_states"):
            good &= np.isfinite(arrays[key][start:end + 1, index]).all()
        if end == 300:
            good &= not terminal_bad
        result.append(good)
    return np.asarray(result, bool)


def atomic_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def batch_specs(plan):
    size = plan["config"]["runtime"]["batch_size"]
    return [plan["episodes"][i:i + size] for i in range(0, len(plan["episodes"]), size)]


def batch_path(output, index):
    return output / "batches" / "batch_{:04d}.json".format(index)


def check_batch(output, index, specs):
    record = read_json(batch_path(output, index))
    if not record.get("complete") or record["plan_sha256"] != file_sha256(output / "plan.json"):
        raise ValueError("Incomplete or mismatched batch manifest")
    if [row["episode_id"] for row in record["episodes"]] != [s["episode_id"] for s in specs]:
        raise ValueError("Batch episode IDs disagree with plan")
    for row in record["episodes"]:
        if file_sha256(output / row["trajectory_path"]) != row["trajectory_sha256"]:
            raise ValueError("Trajectory changed: " + row["episode_id"])
    return record


def worker(output, batch_index):
    from ood_suite.metrics import evaluate_episode, quaternion_geodesic_rad
    _paths(ROOT)
    plan = check_provenance(output)
    specs = batch_specs(plan)[batch_index]
    if batch_path(output, batch_index).exists():
        check_batch(output, batch_index, specs)
        print("Batch already committed; no simulation needed", flush=True)
        return
    start_time = time.perf_counter()
    data = prepare_data(ROOT, specs)
    runtime = plan["config"]["runtime"]
    config = load_environment_config(ROOT, len(specs), False, runtime["num_threads"])
    if not config["environment"]["root_guided"] or config["environment"]["control_dt"] != .01:
        raise ValueError("Collector expects root_guided=True and control_dt=.01")
    env = make_environment(ROOT, data, config, "box")
    try:
        env.seed(plan["config"]["seed"])
        policy = Policies(ROOT, plan["config"]["checkpoints"], config, env,
                          runtime["device"], plan["config"]["seed"])
        controller = RecordingController(env, data, specs, timing_for("box"), policy)
        schedules = np.asarray([spec["residuals"] for spec in specs], np.float32)
        before = time.perf_counter()
        arrays, goal, initial, terminal, invalid, first = simulate_recorded(env, controller, policy, schedules)
        rollout_seconds = time.perf_counter() - before
        rows = []
        for i, spec in enumerate(specs):
            saved = {name: value[:, i] for name, value in arrays.items()}
            for key in ("grasp", "articulation"):
                saved.update({"{}_{}".format(key, j): value[i] for j, value in enumerate(data[key])})
            for key in ("left", "right", "arti_left", "arti_right"):
                saved[key] = data[key][i]
            valid_windows = window_validity(arrays, i, initial[i], terminal[i])
            saved.update(schema_version=np.asarray(SCHEMA_VERSION),
                         window_start_step=np.arange(START, STEPS, HORIZON, dtype=np.int32),
                         window_end_step=np.arange(START + HORIZON, STEPS + 1, HORIZON, dtype=np.int32),
                         window_z=schedules[i], window_valid=valid_windows, goal=goal[i],
                         initial_invalid=initial[i], terminal_bad_observation=terminal[i],
                         action_frame_quaternion_wxyz=saved["final_states"][0, 9:13],
                         control_dt=np.asarray(.01), horizon=np.asarray(HORIZON))
            metric_spec = controller.metric_spec(i)
            metrics = evaluate_episode(saved["states"], metric_spec, controller.timing)
            if invalid[i]:
                metrics.update(valid=False, paper_success=False, public_success=False)
            lift = lift_metrics(saved["states"], metric_spec, controller.timing, not invalid[i])
            entry = saved["final_states"][0]
            entry_finite = np.isfinite(entry).all()
            entry_info = {"position_error_m": float(np.linalg.norm(entry[122:125] - goal[i, :3])) if entry_finite else None,
                          "orientation_error_rad": quaternion_geodesic_rad(entry[125:129], goal[i, 3:7]) if entry_finite else None,
                          "articulation_rad": float(entry[129]) if entry_finite else None,
                          "prior_lift_success": lift["lift_target_success"]}
            path = Path("trajectories") / (spec["episode_id"] + ".npz")
            atomic_npz(output / path, saved)
            row = dict(spec, **metrics)
            row.update(lift)
            row.update(trajectory_path=str(path), trajectory_sha256=file_sha256(output / path),
                       valid=not bool(invalid[i]), first_invalid_step=int(first[i]),
                       valid_windows=int(valid_windows.sum()), invalid_windows=int((~valid_windows).sum()),
                       opening_entry=entry_info, environment_config=config, timing=controller.timing,
                       policy_device=str(policy.device))
            row["ood_goal_success"] = bool(row["paper_success"] and lift["lift_target_success"])
            rows.append(row)
        record = {"schema_version": SCHEMA_VERSION, "complete": True,
                  "plan_sha256": file_sha256(output / "plan.json"), "episodes": rows,
                  "performance": {"wall_seconds": time.perf_counter() - start_time,
                                  "rollout_seconds": rollout_seconds,
                                  "environments": len(specs), "policy_device": str(policy.device),
                                  "policy_forward_devices": policy.forward_devices}}
        write_json(batch_path(output, batch_index), record)
        print("Committed batch {}: {} episodes, {} valid windows, {:.1f}s".format(
            batch_index, len(rows), sum(r["valid_windows"] for r in rows),
            record["performance"]["wall_seconds"]), flush=True)
    finally:
        env.close()


def run(output):
    _paths(ROOT)
    plan = check_provenance(output)
    batches = batch_specs(plan)
    records = []
    for index, specs in enumerate(batches):
        path = batch_path(output, index)
        if not path.exists():
            log = output / "logs" / "batch_{:04d}.log".format(index)
            log.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(Path(__file__).resolve()), "worker", "--output", str(output),
                       "--batch-index", str(index)]
            print("Running batch {}/{} ({} episodes)".format(index + 1, len(batches), len(specs)), flush=True)
            with log.open("w") as handle:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           universal_newlines=True, bufsize=1)
                for line in process.stdout:
                    print(line, end="", flush=True)
                    handle.write(line)
                    handle.flush()
                if process.wait() != 0:
                    raise RuntimeError("Worker failed; see {}. Completed batches are retained.".format(log))
        else:
            print("Checking committed batch {}".format(index), flush=True)
        records.append(check_batch(output, index, specs))
    episodes = [row for record in records for row in record["episodes"]]
    write_json(output / "dataset.json", {
        "schema_version": SCHEMA_VERSION, "complete": True,
        "plan_sha256": file_sha256(output / "plan.json"), "episodes": episodes,
        "valid_windows": sum(row["valid_windows"] for row in episodes),
        "invalid_windows": sum(row["invalid_windows"] for row in episodes),
        "batch_wall_seconds": sum(r["performance"]["wall_seconds"] for r in records),
        "note": "Complete collection; task failures are retained. Not an OOD improvement evaluation."})
    print("Dataset complete: {} episodes, {} valid windows".format(
        len(episodes), sum(row["valid_windows"] for row in episodes)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--kind", choices=("zero", "calibration", "random"), required=True)
    plan.add_argument("--baseline", type=Path, default=HERE / "runs/baseline_smoke_seed101")
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--seed", type=int, default=301)
    plan.add_argument("--episodes", type=int, default=40)
    plan.add_argument("--amplitude", type=float, default=1.)
    plan.add_argument("--batch-size", type=int, default=8)
    plan.add_argument("--num-threads", type=int, default=4)
    plan.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    for name in ("run", "worker"):
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        if name == "worker":
            command.add_argument("--batch-index", type=int, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        make_plan(args)
    elif args.command == "run":
        run(args.output.resolve())
    else:
        worker(args.output.resolve(), args.batch_index)


if __name__ == "__main__":
    main()
