#!/usr/bin/env python3
"""Audit adaptive rollouts against frozen three-way trials, without simulation.

The rolling stop is reconstructed independently; adaptive_stop and either
production collector/evaluator are deliberately not imported. Physical metrics,
hash checks and candidate-score arithmetic reuse the separate fixed-run auditor.
No predictor reinference is performed: saved predictions are compared exactly
through 3 s, and their costs/argmin are independently checked for every window.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

import audit_threeway as fixed

HERE = Path(__file__).resolve().parent
START, FIXED_END, HORIZON = 1260, 1560, 20
METHODS = ("baseline", "guided_absolute", "guided_delta")
EVENTS = ("bad_observation", "bad_action", "bad_state", "auto_reset")
ACTION_ARRAYS = ("actions_r", "actions_l", "ppo_actions_r", "ppo_actions_l")
STEP_ARRAYS = ACTION_ARRAYS + ("residuals", "modes_r", "modes_l") + EVENTS
FINAL_ARRAYS = ("final_states", "final_obs_r", "final_obs_l")
WINDOW_ARRAYS = ("window_z", "guidance_selected", "guidance_fallback",
                 "guidance_predictions", "guidance_scores")
EXPECTED_SETTINGS = {
    "control_dt": .01, "opening_start_step": 1260,
    "min_opening_steps": 300, "max_opening_steps": 1000,
    "success_hold_steps": 50, "success_position_threshold_m": .05,
    "success_articulation_threshold_rad": .5,
    "success_orientation_threshold_rad": .2, "stall_window_steps": 200,
    "stall_articulation_range_rad": .02, "stall_position_displacement_m": .002,
    "stall_orientation_displacement_rad": .02,
    "stall_requires_current_failure": True, "quaternion_norm_epsilon": 1e-12,
    "priority": ["invalid", "success_hold", "stalled", "timeout"],
}


def require(condition, label):
    fixed.require(condition, label)


def exact(actual, expected, label):
    """Include dtype and all bytes, including NaN payloads, in prefix equality."""
    require(fixed.array_digest(actual) == fixed.array_digest(expected),
            label + ": dtype/shape/bytes differ")


def scalar(value):
    return np.asarray(value).item()


def load_archive(root, row):
    path = fixed.safe_path(root, row["trajectory_path"])
    require(fixed.sha(path) == row["trajectory_sha256"], "Changed trajectory: " + str(path))
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def metric_series(states, goal, epsilon):
    """Independent float64 physical metrics and valid unit quaternions."""
    states, goal = np.asarray(states, float), np.asarray(goal, float)
    quaternions = states[:, 125:129]
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        norms = np.linalg.norm(quaternions, axis=1)
        goal_norm = np.linalg.norm(goal[3:7])
        unit = quaternions / norms[:, None]
        target = goal[3:7] / goal_norm
        metrics = np.column_stack((
            np.linalg.norm(states[:, 122:125] - goal[:3], axis=1),
            np.abs(states[:, 129] - goal[7]),
            2 * np.arccos(np.clip(np.abs(unit @ target), 0., 1.))))
    valid = (np.isfinite(states).all(axis=1) & np.isfinite(goal).all() &
             (norms > epsilon) & (goal_norm > epsilon) & np.isfinite(metrics).all(axis=1))
    metrics[~valid] = np.nan
    return metrics, unit, valid


def independent_stop(data, settings=EXPECTED_SETTINGS):
    """Find the first causal stop using the saved pre-action physical states.

    Each action event t is visible from state t+1 onward. Current observation
    and post-reset state validity are visible at the current state itself.
    Return None when the supplied history ends before a stopping event.
    """
    states = data["final_states"]
    metrics, unit, physical_valid = metric_series(states, data["goal"],
                                                  settings["quaternion_norm_epsilon"])
    observed_valid = (np.isfinite(data["final_obs_r"]).all(axis=1) &
                      np.isfinite(data["final_obs_l"]).all(axis=1))
    events = np.logical_or.reduce([data[name] for name in EVENTS])
    cumulative = np.concatenate(([False], np.logical_or.accumulate(events)))
    thresholds = np.asarray([settings["success_position_threshold_m"],
                             settings["success_articulation_threshold_rad"],
                             settings["success_orientation_threshold_rad"]])
    good = np.all(metrics < thresholds, axis=1)
    first_success = -1
    success_start = None
    for opening in range(len(states)):
        global_step = START + opening
        bad = (bool(data["initial_invalid"]) or bool(cumulative[global_step]) or
               not physical_valid[opening] or not observed_valid[opening])
        reason = None
        ranges = None
        if bad:
            reason = "invalid"
        else:
            if good[opening]:
                if first_success < 0:
                    first_success = opening
                if success_start is None:
                    success_start = opening
            else:
                success_start = None
            if opening >= settings["min_opening_steps"]:
                if success_start is not None and opening - success_start >= settings["success_hold_steps"]:
                    reason = "success_hold"
                elif not good[opening]:
                    begin = opening - settings["stall_window_steps"]
                    window = states[begin:opening + 1].astype(float)
                    angle_range = float(window[:, 129].max() - window[:, 129].min())
                    displacement = float(np.linalg.norm(window[:, 122:125] - window[0, 122:125], axis=1).max())
                    dots = np.abs(unit[begin:opening + 1] @ unit[begin])
                    rotation = float((2 * np.arccos(np.clip(dots, 0., 1.))).max())
                    ranges = dict(angle_range_rad=angle_range,
                                  position_excursion_m=displacement,
                                  orientation_excursion_rad=rotation)
                    if (angle_range < settings["stall_articulation_range_rad"] and
                            displacement < settings["stall_position_displacement_m"] and
                            rotation < settings["stall_orientation_displacement_rad"]):
                        reason = "stalled"
                if reason is None and opening == settings["max_opening_steps"]:
                    reason = "timeout"
        if reason is not None:
            return dict(reason=reason, opening_step=opening,
                        first_success_step=first_success,
                        success_hold_intervals=(0 if success_start is None else opening - success_start),
                        terminal_metrics=metrics[opening].tolist(), stall_ranges=ranges)
    return None


def validate_provenance(output, plan, source, source_plan):
    verified = fixed.validate_provenance(output, plan)
    old_verified = fixed.validate_provenance(source, source_plan)
    saved = fixed.read_json(output / "provenance.json")
    old = fixed.read_json(source / "provenance.json")
    allowed_changes = {"purpose", "primary_metrics"}
    for key, value in source_plan.items():
        if key not in allowed_changes:
            require(plan.get(key) == value, "Copied source plan differs: " + key)
    require(saved["source_plan_sha256"] == fixed.sha(source / "plan.json"), "Source plan hash differs")
    require(saved["source_results_sha256"] == fixed.sha(source / "results.json"), "Source result hash differs")
    require(set(saved["adaptive_sources"]) == {"evaluate_adaptive.py", "collect_adaptive.py", "adaptive_stop.py"},
            "Unexpected adaptive source manifest")
    for name, digest in saved["adaptive_sources"].items():
        require(fixed.sha(HERE / name) == digest, "Changed adaptive source: " + name)
        verified[str(HERE / name)] = digest
    for group in ("source_sha256", "collector_source_sha256", "input_sha256", "evaluation_sources",
                  "runtime", "predictor_sha256", "training_report_sha256", "label_set_sha256"):
        require(saved[group] == old[group], "Fixed simulation/model provenance differs: " + group)
    require(all(verified.get(path) == digest for path, digest in old_verified.items()),
            "Source assets/model provenance not preserved")
    return verified


def validate_lengths(data, row, method):
    label = method + "/" + row["episode_id"]
    opening = int(row["termination"]["opening_step"])
    require(0 <= opening <= EXPECTED_SETTINGS["max_opening_steps"], label + " stop bounds")
    end, windows = START + opening, (opening + HORIZON - 1) // HORIZON
    shapes = {"states": (end + 1, 138), "goal": (8,), "final_states": (opening + 1, 138),
              "final_obs_r": (opening + 1, 276), "final_obs_l": (opening + 1, 276),
              "residuals": (end, 3), "window_z": (windows, 3),
              "window_start_step": (windows,), "window_end_step": (windows,),
              "window_valid": (windows,), "guidance_selected": (windows,), "guidance_fallback": (windows,)}
    shapes.update({name: (end, 51) for name in ACTION_ARRAYS})
    shapes.update({name: (end,) for name in ("modes_r", "modes_l") + EVENTS})
    if method != "baseline":
        shapes.update(guidance_predictions=(windows, 7, 9), guidance_scores=(windows, 7))
    for name, shape in shapes.items():
        require(data[name].shape == shape, label + " length/shape: " + name)
    for name, value in (("schema_version", 2), ("horizon", 20), ("control_dt", .01),
                        ("stop_opening_step", opening), ("stop_reason", row["termination"]["reason"]),
                        ("first_success_step", row["termination"]["first_success_step"])):
        require(scalar(data[name]) == value, label + " metadata: " + name)
    for name in EVENTS + ("window_valid", "guidance_fallback"):
        require(data[name].dtype.kind == "b", label + " non-Boolean " + name)
    timing = row["timing"]
    require(timing["protocol"] == "standard" and timing["grasp_steps"] == 260 and
            timing["transit_steps"] == 300 and timing["arti_steps"] == opening + 100,
            label + " timing")
    fixed.same(row["termination"]["opening_seconds"], opening * .01, label + " opening seconds")
    fixed.same(row["termination"]["total_seconds"], end * .01, label + " total seconds")
    starts = np.arange(START, end, HORIZON, dtype=np.int32)
    exact(data["window_start_step"], starts, label + " window starts")
    exact(data["window_end_step"], np.minimum(starts + HORIZON, end), label + " clipped window ends")
    exact(data["final_states"][1:], data["states"][START + 1:], label + " recorded physical states")
    exact(data["action_frame_quaternion_wxyz"], data["final_states"][0, 9:13], label + " post-reset frame")
    target = np.asarray(row["target_position_m"] + row["target_quaternion_wxyz"] +
                        [row["target_articulation_rad"]], np.float32)
    exact(data["goal"], target, label + " saved goal")
    finite = np.isfinite(data["final_states"]).all(axis=1)
    if finite.any():
        fixed.close(data["final_obs_r"][finite, 273], data["final_states"][finite, 129],
                    "metrics", label + " observed physical articulation")
    return opening, end, windows


def audit_episode(data, row, old, old_row, method, settings):
    label = method + "/" + row["episode_id"]
    opening, end, windows = validate_lengths(data, row, method)
    common_end = min(end, FIXED_END)
    exact(data["states"][:common_end + 1], old["states"][:common_end + 1], label + " fixed prefix states")
    for name in STEP_ARRAYS:
        exact(data[name][:common_end], old[name][:common_end], label + " fixed prefix " + name)
    final_count = min(opening, 300) + 1
    for name in FINAL_ARRAYS:
        exact(data[name][:final_count], old[name][:final_count], label + " fixed prefix " + name)
    for name in WINDOW_ARRAYS:
        if name in old:
            exact(data[name][:15], old[name][:min(windows, 15)], label + " fixed prefix " + name)
    for name in old:
        if name == "goal" or name.startswith(("grasp_", "articulation_")) or name in ("left", "right", "arti_left", "arti_right"):
            exact(data[name], old[name], label + " unchanged reference " + name)
    require(row["opening_entry"] == old_row["opening_entry"], label + " reset entry metrics")
    require(row["environment_config"] == old_row["environment_config"] and
            row["policy_device"] == old_row["policy_device"], label + " runtime/config")
    fixed.arrays_equal(data["residuals"][:START], 0., label + " zero prefix residual")
    held = np.repeat(data["window_z"], HORIZON, axis=0)[:opening]
    exact(data["residuals"][START:], held, label + " held residual including last partial window")
    right = data["ppo_actions_r"].copy()
    right[:, :3] += data["residuals"]
    left = data["ppo_actions_l"]
    fixed.arrays_equal(data["bad_action"], ~np.isfinite(right).all(axis=1) | ~np.isfinite(left).all(axis=1), label + " bad actions")
    exact(data["actions_r"], np.nan_to_num(right, nan=0., posinf=0., neginf=0.), label + " actual right actions")
    exact(data["actions_l"], np.nan_to_num(left, nan=0., posinf=0., neginf=0.), label + " actual left actions")
    fixed.arrays_equal(data["modes_r"][START:], 8, label + " right mode")
    fixed.arrays_equal(data["modes_l"][START:], 7, label + " left mode")
    current_bad = (~np.isfinite(data["final_obs_r"]).all(axis=1) |
                   ~np.isfinite(data["final_obs_l"]).all(axis=1))
    fixed.arrays_equal(data["bad_observation"][START:], current_bad[:-1], label + " observation events")
    bad_state = ~np.isfinite(data["states"][1:]).all(axis=1)
    bad_state[START:] |= ~np.isfinite(data["final_states"][:-1]).all(axis=1)
    fixed.arrays_equal(data["bad_state"], bad_state, label + " physical state events")
    fixed.same(bool(data["initial_invalid"]), not np.isfinite(data["states"][0]).all(), label + " initial invalid")
    terminal_bad = bool(current_bad[-1] or not np.isfinite(data["final_states"][-1]).all())
    fixed.same(bool(data["terminal_bad_observation"]), terminal_bad, label + " terminal invalid")
    events = np.logical_or.reduce([data[name] for name in EVENTS])
    expected = independent_stop(data, settings)
    require(expected is not None, label + " stopped before any causal event")
    require(expected["opening_step"] == opening, label + " not stopped at first causal event")
    for key in ("reason", "first_success_step"):
        require(row["termination"][key] == expected[key], label + " independent termination " + key)
    invalid = expected["reason"] == "invalid"
    if invalid:
        first = 0 if bool(data["initial_invalid"]) else int(np.flatnonzero(events)[0]) if events.any() else end
    else:
        first = -1
    fixed.same(row["first_invalid_step"], first, label + " first invalid step")
    window_valid = []
    for begin, finish in zip(data["window_start_step"], data["window_end_step"]):
        local_a, local_b = int(begin) - START, int(finish) - START
        good = (not bool(data["initial_invalid"]) and not events[:int(finish)].any() and
                int(finish - begin) == HORIZON and
                all(np.isfinite(data[name][local_a:local_b + 1]).all() for name in FINAL_ARRAYS))
        window_valid.append(bool(good and (finish < end or (not terminal_bad and not invalid))))
    fixed.arrays_equal(data["window_valid"], np.asarray(window_valid, bool), label + " window validity")
    metrics = fixed.physical_metrics(data, row, invalid)
    for name, value in metrics.items():
        if not metrics["valid"] and name in ("paper_position_success", "paper_orientation_success", "paper_articulation_success") and name not in row:
            continue
        fixed.same(row[name], value, label + " independent metric " + name)
    if opening >= 300:
        require(isinstance(row["fixed_3s_metrics"], dict), label + " missing fixed-3s metrics")
        for name, value in row["fixed_3s_metrics"].items():
            fixed.same(value, old_row[name], label + " source fixed-3s metric " + name)
    else:
        require(invalid and row["fixed_3s_metrics"] is None, label + " valid early stop or fabricated fixed metrics")
    selected = data["guidance_selected"]
    require(selected.dtype.kind in "iu" and np.all((selected >= 0) & (selected < 7)), label + " candidate indices")
    exact(data["window_z"], fixed.CANDIDATES[selected], label + " selected residuals")
    fixed.arrays_equal(row["guidance_candidate_counts"], np.bincount(selected, minlength=7), label + " counts")
    fixed.same(row["guidance_fallback_count"], int(data["guidance_fallback"].sum()), label + " fallbacks")
    if method == "baseline":
        fixed.arrays_equal(selected, 0, label + " baseline selection")
        fixed.arrays_equal(data["guidance_fallback"], False, label + " baseline fallback")
        require("guidance_predictions" not in data and "guidance_scores" not in data, label + " baseline predictions")
    elif windows:
        offsets = np.arange(windows) * HORIZON
        right = np.nan_to_num(data["final_obs_r"][offsets], nan=0., posinf=0., neginf=0.)
        left = np.nan_to_num(data["final_obs_l"][offsets], nan=0., posinf=0., neginf=0.)
        goals = np.repeat(data["goal"][None], windows, axis=0)
        features = fixed.make_features(right, left, data["final_states"][offsets], goals)
        scores = fixed.score_candidates(data["guidance_predictions"], goals, np.isfinite(features).all(axis=(1, 2)))
        fixed.close(data["guidance_scores"], scores, "recomputed_scores", label + " independent candidate scores")
        choices, fallback = fixed.choice(scores)
        fixed.arrays_equal(selected, choices, label + " argmin")
        fixed.arrays_equal(data["guidance_fallback"], fallback, label + " fallback")
    prefix = {"states": fixed.array_digest(data["states"][:START + 1]), "goal": fixed.array_digest(data["goal"])}
    prefix.update({name: fixed.array_digest(data[name][:START]) for name in STEP_ARRAYS})
    prefix.update({name: fixed.array_digest(data[name][0]) for name in FINAL_ARRAYS})
    return expected, prefix, dict(action_steps=end, windows=windows,
                                 partial_windows=int(opening % HORIZON != 0),
                                 fixed_3s_verified=int(opening >= 300))


def audit(output):
    output = Path(output).resolve()
    require(output.is_dir(), "Missing adaptive output directory")
    started = time.monotonic()
    report = dict(schema_version=1, status="running", output=str(output),
                  audit_script_sha256=fixed.sha(__file__), helper_source_sha256=fixed.sha(HERE / "audit_threeway.py"),
                  simulation_performed=False, predictor_reinference_performed=False,
                  episodes_checked=0, batches={}, stop_counts={},
                  action_steps_checked=0, hold_windows_checked=0, partial_windows_checked=0,
                  fixed_3s_prefixes_verified=0)
    try:
        plan = fixed.read_json(output / "plan.json")
        require(plan["termination"] == EXPECTED_SETTINGS, "Stopping thresholds differ from reviewed settings")
        require(plan["methods"] == list(METHODS), "Wrong method list")
        source = Path(plan["source_run"]).resolve()
        source_plan = fixed.read_json(source / "plan.json")
        source_results = fixed.read_json(source / "results.json")
        require(source_results["complete"] and source_results["plan_sha256"] == fixed.sha(source / "plan.json"),
                "Incomplete/changed frozen source results")
        report.update(source_run=str(source), plan_sha256=fixed.sha(output / "plan.json"),
                      source_plan_sha256=fixed.sha(source / "plan.json"),
                      source_results_sha256=fixed.sha(source / "results.json"),
                      provenance_sha256=fixed.sha(output / "provenance.json"))
        report["verified_source_and_input_sha256"] = validate_provenance(output, plan, source, source_plan)
        original = {(row["method"], row["episode_id"]): row for row in source_results["episodes"]}
        require(len(original) == len(source_results["episodes"]), "Duplicate source results")
        results = fixed.read_json(output / "results.json")
        require(results["complete"] and results["plan_sha256"] == report["plan_sha256"], "Incomplete adaptive results")
        report["results_sha256"] = fixed.sha(output / "results.json")
        batch_size = plan["config"]["runtime"]["batch_size"]
        tasks = []
        for profile in plan["profiles"]:
            specs = [item for item in plan["episodes"] if item["profile"] == profile]
            for offset in range(0, len(specs), batch_size):
                for method in METHODS:
                    tasks.append(("{}__{}__{:04d}".format(profile, method, offset // batch_size),
                                  method, specs[offset:offset + batch_size]))
        require({path.stem for path in (output / "batches").glob("*.json")} == {x[0] for x in tasks},
                "Unexpected/missing batch files")
        observed, prefixes, stops = {}, {}, Counter()
        for name, method, specs in tasks:
            report["current_batch"] = name
            path = output / "batches" / (name + ".json")
            batch, old_batch = fixed.read_json(path), fixed.read_json(source / "batches" / (name + ".json"))
            require(batch["complete"] and batch["plan_sha256"] == report["plan_sha256"], "Incomplete batch " + name)
            require([row["episode_id"] for row in batch["episodes"]] == [spec["episode_id"] for spec in specs],
                    "Batch order differs: " + name)
            require(old_batch["episodes"] == [original[(method, spec["episode_id"])] for spec in specs],
                    "Source batch/result order differs: " + name)
            require(batch["performance"]["environments"] == len(specs) == old_batch["performance"]["environments"],
                    "Changed inference batch size")
            per_batch, max_stop = Counter(), 0
            for index, (row, spec) in enumerate(zip(batch["episodes"], specs)):
                key = method, row["episode_id"]
                require(key not in observed and row["method"] == method, "Duplicate/wrong-method adaptive row")
                require(all(row.get(k) == v for k, v in spec.items()), "Row differs from planned episode")
                previous = original[key]
                saved, old = load_archive(output, row), load_archive(source, previous)
                stop, prefix, counts = audit_episode(saved, row, old, previous, method, plan["termination"])
                observed[key], prefixes[key] = row, prefix
                stops[stop["reason"]] += 1
                per_batch[stop["reason"]] += 1
                max_stop = max(max_stop, START + stop["opening_step"])
                report["episodes_checked"] += 1
                report["action_steps_checked"] += counts["action_steps"]
                report["hold_windows_checked"] += counts["windows"]
                report["partial_windows_checked"] += counts["partial_windows"]
                report["fixed_3s_prefixes_verified"] += counts["fixed_3s_verified"]
                if (index + 1) % 25 == 0:
                    print("Audit {}: {}/{} episodes".format(name, index + 1, len(specs)), flush=True)
            require(batch["performance"]["batch_physics_steps"] == max_stop, "Batch did not end at last latched stop")
            report["batches"][name] = dict(sha256=fixed.sha(path), source_sha256=fixed.sha(source / "batches" / (name + ".json")),
                                                   episodes=len(specs), stops=dict(per_batch), batch_physics_steps=max_stop)
            print("Audited {}: {} episodes".format(name, len(specs)), flush=True)
        final_rows = {(row["method"], row["episode_id"]): row for row in results["episodes"]}
        require(len(final_rows) == len(results["episodes"]) and final_rows == observed, "Results/batch rows differ")
        require(set(observed) == set(original), "Adaptive/frozen episode identities differ")
        for spec in plan["episodes"]:
            base = prefixes[("baseline", spec["episode_id"])]
            for method in METHODS[1:]:
                require(prefixes[(method, spec["episode_id"])] == base, "Cross-method common prefix differs")
        report["common_prefix_pairs_verified"] = 2 * len(plan["episodes"])
        report["stop_counts"] = dict(stops)
        report["source_hashes_rechecked_after_audit"] = validate_provenance(output, plan, source, source_plan) == report["verified_source_and_input_sha256"]
        require(report["source_hashes_rechecked_after_audit"], "Sources changed during audit")
        require(fixed.sha(output / "results.json") == report["results_sha256"] and
                fixed.sha(source / "results.json") == report["source_results_sha256"], "Results changed during audit")
        report.update(status="completed", checks=[
            "Unchanged fixed simulation/model sources, input/checkpoint hashes and copied plan",
            "Exact dtype/shape/byte equality to fixed run through the first 3 s of opening",
            "Independent causal first-stop reconstruction, 51/201-state inclusive windows",
            "Current success suppresses stall; invalid/success/stall/timeout priority",
            "Independent terminal physical metrics; fixed-3s metrics equal original rows",
            "Variable lengths, post-reset entry, partial residual window bounds and validity",
            "Every actual action equals sanitized PPO plus held translation residual",
            "All saved candidate costs/argmin/fallback independently recomputed",
            "Identical batch shape/order; finished slots excluded by saved-array truncation",
        ])
        report.pop("current_batch", None)
    except BaseException as error:
        report.update(status="failed", error="{}: {}".format(type(error).__name__, error))
        raise
    finally:
        report["audited_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["elapsed_seconds"] = time.monotonic() - started
        fixed.write_report(output / "independent_audit.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit(args.output)
    except (Exception, KeyboardInterrupt) as error:
        print("Audit failed: {}".format(error), file=sys.stderr, flush=True)
        return 1
    print(json.dumps({name: result[name] for name in ("status", "episodes_checked", "stop_counts",
                                                    "fixed_3s_prefixes_verified", "partial_windows_checked")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
