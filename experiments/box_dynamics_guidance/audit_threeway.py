#!/usr/bin/env python3
"""Independently audit saved three-way rollouts; never initialize a simulator.

Reconstructs metrics, features, candidate costs and checkpoint predictions. The
only output is independent_audit.json inside the requested evaluation directory.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
METHODS = ("baseline", "guided_absolute", "guided_delta")
START, STEPS, HORIZON = 1260, 1560, 20
REPLAY_WINDOWS = (0, 7, 14)
CANDIDATES = np.asarray([[0., 0., 0.], [1., 0., 0.], [-1., 0., 0.],
                         [0., 1., 0.], [0., -1., 0.], [0., 0., 1.], [0., 0., -1.]], np.float32)
TOLERANCES = {
    "metrics": {"rtol": 1e-10, "atol": 1e-10},
    "recomputed_scores": {"rtol": 1e-12, "atol": 1e-12},
    "replayed_predictions": {"rtol": 2e-6, "atol": 2e-6},
    "replayed_scores": {"rtol": 5e-6, "atol": 5e-5},
    "actions_and_selection": "exact array equality",
    "common_prefix": "exact dtype/shape/array-byte SHA256 equality",
    "previous_run": "exact dtype/shape/array-byte SHA256 equality; differences are reported, not hidden",
}
SUCCESS_KEYS = ("paper_success", "lift_target_success", "lift_pose_success",
                "ood_goal_success", "strict_goal_success", "articulation_strict_success")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def safe_path(root, relative):
    path = (root / relative).resolve()
    require(root.resolve() in path.parents, "Path escapes evaluation directory: {}".format(relative))
    return path


def same(actual, expected, label):
    if isinstance(expected, (bool, str)) or expected is None:
        require(actual == expected, "{}: {} != {}".format(label, actual, expected))
    else:
        require(actual is not None and math.isclose(float(actual), float(expected),
                rel_tol=TOLERANCES["metrics"]["rtol"], abs_tol=TOLERANCES["metrics"]["atol"]),
                "{}: {} != {}".format(label, actual, expected))


def arrays_equal(actual, expected, label):
    try:
        np.testing.assert_array_equal(actual, expected)
    except AssertionError as error:
        raise AssertionError("{}: {}".format(label, error)) from error


def close(actual, expected, tolerance, label):
    try:
        np.testing.assert_allclose(actual, expected, equal_nan=True, **TOLERANCES[tolerance])
    except AssertionError as error:
        raise AssertionError("{}: {}".format(label, error)) from error


def array_digest(array):
    array = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def qangle(a, b):
    # Deliberately separate from the production metric/scoring helpers.
    a, b = [float(x) for x in a], [float(x) for x in b]
    na, nb = math.sqrt(sum(x*x for x in a)), math.sqrt(sum(x*x for x in b))
    require(na >= 1e-12 and nb >= 1e-12, "Zero quaternion in metric")
    dot = abs(sum((x / na) * (y / nb) for x, y in zip(a, b)))
    return 2.0 * math.acos(min(1.0, max(0.0, dot)))


def distance(a, b):
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def physical_metrics(data, row, invalid):
    states, goal = data["states"], data["goal"]
    valid = not invalid and bool(np.isfinite(states).all())
    result = {"valid": valid, "diagnostic_valid": valid}
    error_keys = ("paper_position_error_m", "paper_orientation_error_rad", "paper_articulation_error_rad",
                  "lift_max_height_m", "lift_final_height_m", "lift_target_error_m",
                  "lift_position_error_min_m", "lift_orientation_error_min_rad",
                  "lift_position_error_final_m", "lift_orientation_error_final_rad")
    for key in SUCCESS_KEYS + ("paper_position_success", "paper_orientation_success", "paper_articulation_success"):
        result[key] = False
    result.update({key: None for key in error_keys})
    if not valid:
        return result
    end = states[-1]
    pe, ae, aae = distance(end[122:125], goal[:3]), qangle(end[125:129], goal[3:7]), abs(float(end[129]) - float(goal[7]))
    position_ok, orientation_ok, angle_ok = pe < .05, ae < .2, aae < .5
    result.update(paper_position_error_m=pe, paper_orientation_error_rad=ae,
                  paper_articulation_error_rad=aae, paper_position_success=position_ok,
                  paper_orientation_success=orientation_ok, paper_articulation_success=angle_ok,
                  paper_success=position_ok and orientation_ok and angle_ok,
                  articulation_strict_success=aae < .1)
    g, t = row["timing"]["grasp_steps"], row["timing"]["transit_steps"]
    lift_states = states[g + 1:g + t + 1]
    height = float(row["target"]["lift_m"])
    heights = lift_states[:, 124] - states[0, 124]
    lift_error = float(np.min(np.abs(heights - height)))
    lift_position = [distance(state[122:125], row["lift_target_position_m"]) for state in lift_states]
    lift_orientation = [qangle(state[125:129], row["lift_target_quaternion_wxyz"]) for state in lift_states]
    lift_pose = any(p < .05 and a < .2 for p, a in zip(lift_position, lift_orientation))
    # Diagnostics uses Python floats; lift_metrics uses float32 height arrays.
    height_reached = any(abs(float(state[124]) - float(states[0, 124]) - height) < .05 for state in lift_states)
    intermediate = height_reached and (lift_pose if row.get("require_lift_pose_success") else True)
    result.update(lift_max_height_m=float(np.max(heights)), lift_final_height_m=float(heights[-1]),
                  lift_target_error_m=lift_error, lift_target_success=lift_error < .05,
                  lift_pose_success=lift_pose, lift_position_error_min_m=min(lift_position),
                  lift_orientation_error_min_rad=min(lift_orientation),
                  lift_position_error_final_m=lift_position[-1], lift_orientation_error_final_rad=lift_orientation[-1],
                  ood_goal_success=position_ok and orientation_ok and angle_ok and intermediate,
                  strict_goal_success=position_ok and orientation_ok and aae < .1 and intermediate)
    return result


def make_features(right, left, states, goals):
    objects = np.asarray(states[:, 122:130], np.float32)
    goals = np.asarray(goals, np.float32).copy()
    goals[np.sum(goals[:, 3:7] * objects[:, 3:7], axis=1) < 0, 3:7] *= -1
    common = np.concatenate((right, left, objects, goals), axis=1)
    return np.concatenate((np.repeat(common[:, None, :], 7, axis=1),
                           np.broadcast_to(CANDIDATES, (len(common), 7, 3))), axis=2).astype(np.float32)


def score_candidates(predictions, goals, finite_inputs):
    p, g = np.asarray(predictions, np.float64), np.asarray(goals, np.float64)
    norm, goal_norm = np.linalg.norm(p[:, :, 5:9], axis=-1), np.linalg.norm(g[:, 3:7], axis=-1)
    good = (np.isfinite(p).all(axis=-1) & (norm > 1e-8) &
            np.isfinite(g).all(axis=-1)[:, None] & (goal_norm > 1e-8)[:, None])
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        dot = np.sum(p[:, :, 5:9] * g[:, None, 3:7], axis=-1)
        angle = 2 * np.arccos(np.clip(np.abs(dot / (norm * goal_norm[:, None])), 0, 1))
        score = ((p[:, :, 0] - g[:, None, 7]) / .5) ** 2
        score += np.sum((p[:, :, 2:5] - g[:, None, :3]) ** 2, axis=-1) / .05 ** 2
        score += (angle / .2) ** 2
        score += .01 * np.sum(CANDIDATES ** 2, axis=-1)[None, :]
    score = np.where(good & np.isfinite(score), score, np.inf)
    score[~finite_inputs] = np.inf
    return score


def choice(scores):
    selected = np.argmin(scores, axis=1)
    fallback = ~np.isfinite(scores).any(axis=1)
    selected[fallback] = 0
    return selected, fallback


class ReplayPredictor(nn.Module):
    """Independent network/feature/decoder path; no production Selector import."""
    def __init__(self, spec, method, device):
        super().__init__()
        checkpoint = torch.load(spec["path"], map_location="cpu", weights_only=True)
        expected = "box_response_best_predictor" if method == "guided_absolute" else "box_response_delta_best_predictor"
        require(checkpoint["artifact_type"] == expected, "Wrong predictor artifact type")
        require(checkpoint["best_epoch"] == spec["best_epoch"], "Predictor best epoch mismatch")
        config = checkpoint["model_config"]
        width, modules = 571, []
        for _ in range(config["layers"]):
            modules.extend((nn.Linear(width, config["hidden_dim"]), nn.ReLU()))
            width = config["hidden_dim"]
        modules.append(nn.Linear(width, 9))
        self.network = nn.Sequential(*modules)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.device, self.method = torch.device(device), method
        self.eval().to(self.device)
        self.normalization = {key: value.to(self.device) for key, value in checkpoint["normalization"].items()}
        require(all(torch.isfinite(value).all().item() for value in self.state_dict().values()), "Nonfinite predictor")

    def predict(self, right, left, states, goals):
        x = make_features(right, left, states, goals)
        finite = np.isfinite(x).all(axis=(1, 2))
        x = np.nan_to_num(x, nan=0., posinf=0., neginf=0.)
        n = self.normalization
        with torch.no_grad():
            tensor = torch.from_numpy(x.reshape(-1, 571)).to(self.device)
            output = self.network((tensor - n["x_mean"]) / n["x_std"]) * n["y_std"] + n["y_mean"]
            prediction = output.cpu().numpy().reshape(len(x), 7, 9)
        if self.method == "guided_delta":
            prediction[:, :, 0] += np.asarray(right[:, 273], np.float32)[:, None]
            prediction[:, :, 2:5] += np.asarray(states[:, 122:125], np.float32)[:, None, :]
        scores = score_candidates(prediction, goals, finite)
        selected, fallback = choice(scores)
        return selected, prediction, scores, fallback


def validate_provenance(output, plan):
    saved = read_json(output / "provenance.json")
    require(saved["plan_sha256"] == sha(output / "plan.json"), "Provenance plan hash mismatch")
    verified = {}
    for group in ("source_sha256", "collector_source_sha256", "input_sha256", "evaluation_sources"):
        require(group in saved, "Missing provenance group: " + group)
        for relative, digest in saved[group].items():
            path = (HERE if group == "evaluation_sources" else ROOT) / relative
            require(sha(path) == digest, "Changed source/input: {}".format(path))
            verified[str(path.resolve())] = digest
    extension = saved["runtime"]
    require(sha(extension["extension_path"]) == extension["extension_sha256"], "Simulator binary hash mismatch")
    verified[extension["extension_path"]] = extension["extension_sha256"]
    label = ROOT / "rsc/mano_double/grasp_label/final_labels" / (plan["split"] + "_set") / "box.npy"
    require(sha(label) == saved["label_set_sha256"], "Evaluation label hash mismatch")
    verified[str(label)] = saved["label_set_sha256"]
    for method, spec in plan["predictors"].items():
        require(sha(spec["path"]) == spec["sha256"] == saved["predictor_sha256"][method], "Predictor hash mismatch")
        report_path = Path(spec["path"]).parent / "report.json"
        report_hash = sha(report_path)
        require(report_hash == spec["training_report_sha256"] == saved["training_report_sha256"][method], "Training report hash mismatch")
        training = read_json(report_path)
        require(training["artifact_sha256"]["best.pt"] == spec["sha256"], "Training report/checkpoint mismatch")
        verified[str(Path(spec["path"]).resolve())], verified[str(report_path.resolve())] = spec["sha256"], report_hash
    return verified


def audit_episode(data, row, method):
    label = method + "/" + row["episode_id"]
    shapes = {"states": (1561, 138), "final_states": (301, 138),
              "final_obs_r": (301, 276), "final_obs_l": (301, 276),
              "residuals": (1560, 3), "window_z": (15, 3), "goal": (8,),
              "guidance_selected": (15,), "guidance_fallback": (15,), "window_valid": (15,)}
    shapes.update({key: (1560, 51) for key in ("actions_r", "actions_l", "ppo_actions_r", "ppo_actions_l")})
    shapes.update({key: (1560,) for key in ("modes_r", "modes_l", "bad_observation", "bad_action", "bad_state", "auto_reset")})
    if method != "baseline":
        shapes.update(guidance_predictions=(15, 7, 9), guidance_scores=(15, 7))
    for key, shape in shapes.items():
        require(data[key].shape == shape, label + " bad shape: " + key)
    require(data["schema_version"].item() == 1 and data["horizon"].item() == HORIZON
            and data["control_dt"].item() == .01, label + " bad schema/horizon/control dt")
    timing = row["timing"]
    require(timing["protocol"] == "standard" and timing["grasp_steps"] + 3*timing["transit_steps"]
            + timing["arti_steps"] == STEPS, label + " bad timing")
    arrays_equal(data["window_start_step"], np.arange(1260, 1560, 20), label + " window starts")
    arrays_equal(data["window_end_step"], np.arange(1280, 1561, 20), label + " window ends")
    arrays_equal(data["residuals"][:START], 0., label + " prefix residual")
    arrays_equal(data["residuals"][START:], np.repeat(data["window_z"], 20, axis=0), label + " held residual")
    expected_r = data["ppo_actions_r"].copy()
    expected_r[:, :3] += data["residuals"]
    expected_l = data["ppo_actions_l"]
    arrays_equal(data["bad_action"], ~np.isfinite(expected_r).all(axis=1) | ~np.isfinite(expected_l).all(axis=1), label + " bad action flags")
    arrays_equal(data["actions_r"], np.nan_to_num(expected_r, nan=0., posinf=0., neginf=0.), label + " right action")
    arrays_equal(data["actions_l"], np.nan_to_num(expected_l, nan=0., posinf=0., neginf=0.), label + " left action")
    arrays_equal(data["modes_r"][START:], 8, label + " right mode")
    arrays_equal(data["modes_l"][START:], 7, label + " left mode")
    arrays_equal(data["final_states"][1:], data["states"][START + 1:], label + " physical state history")
    target = np.asarray(row["target_position_m"] + row["target_quaternion_wxyz"] + [row["target_articulation_rad"]], np.float32)
    arrays_equal(data["goal"], target, label + " saved goal")
    if np.isfinite(data["states"][-1]).all() and not data["auto_reset"].any():
        arrays_equal(data["goal"], data["states"][-1, 130:138], label + " simulator goal")
    arrays_equal(data["bad_observation"][START:],
                 ~np.isfinite(data["final_obs_r"][:-1]).all(axis=1) | ~np.isfinite(data["final_obs_l"][:-1]).all(axis=1), label + " bad observation flags")
    bad_state = ~np.isfinite(data["states"][1:]).all(axis=1)
    bad_state[START:] |= ~np.isfinite(data["final_states"][:-1]).all(axis=1)
    arrays_equal(data["bad_state"], bad_state, label + " bad state flags")
    terminal = any(not np.isfinite(data[key][-1]).all() for key in ("final_obs_r", "final_obs_l", "final_states"))
    same(bool(data["initial_invalid"]), not np.isfinite(data["states"][0]).all(), label + " initial invalid")
    same(bool(data["terminal_bad_observation"]), terminal, label + " terminal invalid")
    event = np.logical_or.reduce([data[key] for key in ("bad_observation", "bad_action", "bad_state", "auto_reset")])
    invalid = bool(data["initial_invalid"] or terminal or event.any())
    first = 0 if data["initial_invalid"] else int(np.flatnonzero(event)[0]) if event.any() else STEPS if terminal else -1
    same(row["first_invalid_step"], first, label + " first invalid")
    valid_windows = []
    for window in range(15):
        start, end = window * 20, (window + 1) * 20
        valid = not data["initial_invalid"] and not event[:START + end].any()
        valid &= all(np.isfinite(data[key][start:end + 1]).all() for key in ("final_obs_r", "final_obs_l", "final_states"))
        valid_windows.append(bool(valid and (end != 300 or not terminal)))
    arrays_equal(data["window_valid"], valid_windows, label + " valid windows")
    metrics = physical_metrics(data, row, invalid)
    for key, expected in metrics.items():
        if not metrics["valid"] and key in ("paper_position_success", "paper_orientation_success", "paper_articulation_success") and key not in row:
            # Production's invalid-rollout schema omits these submetrics.
            continue
        same(row[key], expected, label + "/" + key)
    selected = data["guidance_selected"]
    require(np.issubdtype(selected.dtype, np.integer) and np.all((selected >= 0) & (selected < 7)), label + " candidate indices")
    arrays_equal(data["window_z"], CANDIDATES[selected], label + " candidate residual")
    arrays_equal(row["guidance_candidate_counts"], np.bincount(selected, minlength=7), label + " candidate counts")
    same(row["guidance_fallback_count"], int(data["guidance_fallback"].sum()), label + " fallback count")
    if method == "baseline":
        arrays_equal(selected, 0, label + " baseline selection")
        arrays_equal(data["guidance_fallback"], False, label + " baseline fallback")
    else:
        offsets = np.arange(0, 300, 20)
        right = np.nan_to_num(data["final_obs_r"][offsets], nan=0., posinf=0., neginf=0.)
        left = np.nan_to_num(data["final_obs_l"][offsets], nan=0., posinf=0., neginf=0.)
        goals = np.repeat(data["goal"][None], 15, axis=0)
        x = make_features(right, left, data["final_states"][offsets], goals)
        scores = score_candidates(data["guidance_predictions"], goals, np.isfinite(x).all(axis=(1, 2)))
        close(scores, data["guidance_scores"], "recomputed_scores", label + " candidate costs")
        expected_selected, fallback = choice(scores)
        arrays_equal(selected, expected_selected, label + " candidate argmin")
        arrays_equal(data["guidance_fallback"], fallback, label + " fallback")
        # Cross-check representative inputs against the unchanged training loader.
        from dataset import _window
        for window in REPLAY_WINDOWS:
            training, _ = _window(data, window, row["episode_id"])
            if training is not None:
                arrays_equal(training[0], x[window, selected[window]], label + " training feature order")
    prefix = {"states": array_digest(data["states"][:START + 1]), "goal": array_digest(data["goal"])}
    for key in ("actions_r", "actions_l", "ppo_actions_r", "ppo_actions_l", "residuals", "modes_r", "modes_l",
                "bad_observation", "bad_action", "bad_state", "auto_reset"):
        prefix[key] = array_digest(data[key][:START])
    for key in ("final_obs_r", "final_obs_l", "final_states"):
        prefix[key + "_entry"] = array_digest(data[key][0])
    return metrics, prefix


def max_finite_error(actual, expected):
    finite = np.isfinite(actual) & np.isfinite(expected)
    return float(np.max(np.abs(actual[finite] - expected[finite]))) if finite.any() else 0.


def compare_previous(output, plan, digests):
    previous = HERE / "runs/comparison_seed202_v1"
    if plan["seed"] != 202 or output == previous or not (previous / "results.json").exists():
        return {"status": "not_applicable", "reason": "No matching seed-202 prior evaluation"}
    old_plan, old_results = read_json(previous / "plan.json"), read_json(previous / "results.json")
    if old_plan["episodes"] != plan["episodes"]:
        return {"status": "not_comparable", "reason": "Planned episode specifications differ"}
    require(old_results["complete"] and old_results["plan_sha256"] == sha(previous / "plan.json"), "Prior result manifest mismatch")
    mapping = {"baseline": "baseline", "guided": "guided_absolute"}
    counts = {method: {"episodes_compared": 0, "identical_arrays": 0, "differing_episodes": []} for method in mapping.values()}
    for row in old_results["episodes"]:
        if row["method"] not in mapping:
            continue
        method = mapping[row["method"]]
        key = (method, row["episode_id"])
        if key not in digests:
            continue
        archive = safe_path(previous, row["trajectory_path"])
        require(sha(archive) == row["trajectory_sha256"], "Changed prior trajectory: " + str(archive))
        with np.load(archive, allow_pickle=False) as saved:
            old = {name: array_digest(saved[name]) for name in saved.files}
        differences = sorted(name for name in set(old) | set(digests[key]) if old.get(name) != digests[key].get(name))
        counts[method]["episodes_compared"] += 1
        if differences:
            counts[method]["differing_episodes"].append({"episode_id": row["episode_id"], "arrays": differences})
        else:
            counts[method]["identical_arrays"] += 1
    return {"status": "compared", "previous_output": str(previous), "plan_sha256": sha(previous / "plan.json"),
            "results_sha256": sha(previous / "results.json"), "methods": counts,
            "comparison": "All saved arrays, independent of NPZ ZIP metadata; differences do not invalidate the new matched comparison"}


def write_report(path, report):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def audit(output):
    output = Path(output).expanduser().resolve()
    require(output.is_dir(), "Evaluation output directory does not exist")
    started = time.monotonic()
    report = {"schema_version": 1, "status": "running", "output": str(output),
              "audit_script_sha256": sha(__file__), "tolerances": TOLERANCES,
              "replay_windows": list(REPLAY_WINDOWS), "episodes_checked": 0,
              "prediction_windows_replayed": 0, "batches": {}, "simulation_performed": False}
    try:
        plan = read_json(output / "plan.json")
        require(plan["methods"] == list(METHODS), "Expected exactly three methods")
        require(set(plan["predictors"]) == set(METHODS[1:]), "Expected both guided predictors")
        settings = plan["guidance"]
        for key, expected in (("horizon_steps", 20), ("reselect_every_steps", 20), ("phase_start_step", 1260),
                              ("angle_scale_rad", .5), ("position_scale_m", .05), ("orientation_scale_rad", .2),
                              ("residual_squared_weight", .01)):
            require(settings[key] == expected, "Unexpected guidance setting: " + key)
        arrays_equal(settings["candidates"], CANDIDATES, "frozen candidates")
        report["plan_sha256"] = sha(output / "plan.json")
        report["verified_source_and_input_sha256"] = validate_provenance(output, plan)
        report["provenance_sha256"] = sha(output / "provenance.json")
        result_file = output / "results.json"
        results = read_json(result_file)
        require(results["complete"] and results["plan_sha256"] == report["plan_sha256"], "Incomplete final results")
        report["results_sha256"] = sha(result_file)
        torch.set_num_threads(plan["config"]["runtime"]["num_threads"])
        runtime = plan["config"]["runtime"]
        report["reinference_device"] = runtime["device"]
        predictors = {method: ReplayPredictor(plan["predictors"][method], method, runtime["device"]) for method in METHODS[1:]}
        batches, all_rows, prefixes, digests = [], {}, {}, {}
        for profile in plan["profiles"]:
            specs = [spec for spec in plan["episodes"] if spec["profile"] == profile]
            for offset in range(0, len(specs), runtime["batch_size"]):
                for method in METHODS:
                    batches.append(("{}__{}__{:04d}".format(profile, method, offset // runtime["batch_size"]),
                                    method, specs[offset:offset + runtime["batch_size"]]))
        require({path.stem for path in (output / "batches").glob("*.json")} == {item[0] for item in batches}, "Batch files do not match plan")
        for name, method, specs in batches:
            report["current_batch"] = name
            batch_path = output / "batches" / (name + ".json")
            record = read_json(batch_path)
            require(record["complete"] and record["plan_sha256"] == report["plan_sha256"], "Incomplete batch: " + name)
            rows = record["episodes"]
            require([row["episode_id"] for row in rows] == [spec["episode_id"] for spec in specs], "Batch episode order mismatch")
            summary = {"sha256": sha(batch_path), "method": method, "episodes": len(rows), "invalid": 0,
                       "prediction_windows_replayed": 0, "max_prediction_abs_error": 0., "max_score_abs_error": 0.,
                       "candidate_counts": [0]*7, "fallback_windows": 0,
                       "success_counts": {key: 0 for key in SUCCESS_KEYS}}
            replay = {key: [] for key in ("right", "left", "states", "goals", "predictions", "scores", "selected", "fallback")}
            for index, (row, spec) in enumerate(zip(rows, specs)):
                key = (method, row["episode_id"])
                require(key not in all_rows and row["method"] == method, "Duplicate/wrong-method episode")
                require(all(row.get(k) == v for k, v in spec.items()), "Episode differs from frozen plan")
                archive = safe_path(output, row["trajectory_path"])
                require(sha(archive) == row["trajectory_sha256"], "Trajectory hash mismatch: " + str(archive))
                with np.load(archive, allow_pickle=False) as saved:
                    data = {key: saved[key] for key in saved.files}
                metrics, prefix = audit_episode(data, row, method)
                prefixes[key], digests[key], all_rows[key] = prefix, {k: array_digest(v) for k, v in data.items()}, row
                summary["invalid"] += int(not metrics["valid"])
                summary["fallback_windows"] += row["guidance_fallback_count"]
                summary["candidate_counts"] = (np.asarray(summary["candidate_counts"]) + row["guidance_candidate_counts"]).tolist()
                for metric in SUCCESS_KEYS:
                    summary["success_counts"][metric] += int(metrics[metric])
                report["episodes_checked"] += 1
                if method != "baseline":
                    offsets = np.asarray(REPLAY_WINDOWS) * 20
                    for field, value in (("right", data["final_obs_r"][offsets]), ("left", data["final_obs_l"][offsets]),
                                         ("states", data["final_states"][offsets]), ("goals", data["goal"]),
                                         ("predictions", data["guidance_predictions"][list(REPLAY_WINDOWS)]),
                                         ("scores", data["guidance_scores"][list(REPLAY_WINDOWS)]),
                                         ("selected", data["guidance_selected"][list(REPLAY_WINDOWS)]),
                                         ("fallback", data["guidance_fallback"][list(REPLAY_WINDOWS)])):
                        replay[field].append(value)
                if (index + 1) % 25 == 0:
                    print("Audit {}: {}/{} trajectories".format(name, index + 1, len(rows)), flush=True)
            if method != "baseline":
                replay = {key: np.asarray(value) for key, value in replay.items()}
                for position, window in enumerate(REPLAY_WINDOWS):
                    selected, prediction, scores, fallback = predictors[method].predict(
                        np.nan_to_num(replay["right"][:, position], nan=0., posinf=0., neginf=0.),
                        np.nan_to_num(replay["left"][:, position], nan=0., posinf=0., neginf=0.),
                        replay["states"][:, position], replay["goals"])
                    close(prediction, replay["predictions"][:, position], "replayed_predictions", name + " prediction window " + str(window))
                    close(scores, replay["scores"][:, position], "replayed_scores", name + " scores window " + str(window))
                    arrays_equal(selected, replay["selected"][:, position], name + " reinferred argmin")
                    arrays_equal(fallback, replay["fallback"][:, position], name + " reinferred fallback")
                    summary["max_prediction_abs_error"] = max(summary["max_prediction_abs_error"], max_finite_error(prediction, replay["predictions"][:, position]))
                    summary["max_score_abs_error"] = max(summary["max_score_abs_error"], max_finite_error(scores, replay["scores"][:, position]))
                    summary["prediction_windows_replayed"] += len(rows)
                report["prediction_windows_replayed"] += summary["prediction_windows_replayed"]
            report["batches"][name] = summary
            print("Audited {}: {} episodes; {} replay windows".format(name, len(rows), summary["prediction_windows_replayed"]), flush=True)
        result_rows = {(row["method"], row["episode_id"]): row for row in results["episodes"]}
        require(len(result_rows) == len(results["episodes"]) and result_rows == all_rows, "results.json differs from audited batch rows")
        for spec in plan["episodes"]:
            reference = prefixes[("baseline", spec["episode_id"])]
            for method in METHODS[1:]:
                other = prefixes[(method, spec["episode_id"])]
                changed = [key for key in reference if reference[key] != other[key]]
                require(not changed, "Common prefix differs for {}/{}: {}".format(method, spec["episode_id"], changed))
        report["common_prefix_pairs_verified"] = len(plan["episodes"]) * 2
        report["previous_comparison"] = compare_previous(output, plan, digests)
        report.update(status="completed", action_steps_checked=report["episodes_checked"]*STEPS,
                      hold_windows_checked=report["episodes_checked"]*15,
                      invalid_episodes=sum(value["invalid"] for value in report["batches"].values()),
                      source_hashes_rechecked_after_audit=validate_provenance(output, plan) == report["verified_source_and_input_sha256"])
        report.pop("current_batch", None)
        report["checks"] = ["All provenance sources, inputs, checkpoints and NPZ hashes",
                            "Physical endpoint PE/AE/AAE, paper, lift, OOD and strict metrics independently recomputed",
                            "Every executed action equals sanitized PPO plus held right-translation residual",
                            "Zero residual before step 1260; all fifteen 20-step windows",
                            "Independent feature and candidate-score construction; exact argmin/fallback",
                            "Representative features cross-checked against unchanged training loader",
                            "Independent checkpoint reinference in original environment batch order/size",
                            "Exact common prefix and opening entry across all three methods",
                            "Results rows match batch rows; numerical validity and first-invalid step"]
    except BaseException as error:
        report.update(status="failed", error="{}: {}".format(type(error).__name__, error))
        raise
    finally:
        report["audited_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["elapsed_seconds"] = time.monotonic() - started
        write_report(output / "independent_audit.json", report)
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
    print(json.dumps({key: result[key] for key in ("status", "episodes_checked", "prediction_windows_replayed", "common_prefix_pairs_verified")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
