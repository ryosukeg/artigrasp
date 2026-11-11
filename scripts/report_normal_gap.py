#!/usr/bin/env python3
"""Build an offline report after both 800-trial public diagnostics finish.

Uses only the Python standard library and saved metrics. Refuses missing,
incomplete, duplicate, or inconsistent inputs before writing any report file.
This script does not rescore trajectories or change experiment artifacts.
"""

import argparse
import ast
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
OBJECTS = ("box", "laptop", "notebook", "waffleiron", "mixer", "ketchup",
           "espressomachine", "microwave")
SUCCESS_KEYS = ("public_success", "paper_success", "paper_position_success",
                "paper_orientation_success", "paper_articulation_success")
ERROR_KEYS = ("paper_position_error_m", "paper_orientation_error_rad",
              "paper_articulation_error_rad", "public_position_error_clipped_m",
              "public_orientation_error_clipped_rad", "public_articulation_error_clipped_rad")
IDENTITY_KEYS = ("object", "sample_index", "seed", "split", "grasp_label_index",
                 "arti_label_index", "grasp_label", "arti_label")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_run(directory, split, profile, diagnostic):
    directory = Path(directory).resolve()
    for name in ("plan.json", "episodes.jsonl", "summary.json", "provenance.json", "execution.json"):
        require((directory / name).is_file(), "Missing run input: " + str(directory / name))
    plan = read_json(directory / "plan.json")
    saved_summary = read_json(directory / "summary.json")
    provenance = read_json(directory / "provenance.json")
    execution = read_json(directory / "execution.json")
    rows = [json.loads(line) for line in (directory / "episodes.jsonl").read_text().splitlines() if line.strip()]
    require(plan["split"] == split and plan["profiles"] == [profile], "Unexpected split/profile: " + str(directory))
    require(plan["samples_per_object"] == 100, "Expected exactly 100 planned trials per object")
    require(len(plan["objects"]) == len(set(plan["objects"])), "Duplicate plan objects")
    require(set(OBJECTS).issubset(plan["objects"]), "Paper objects missing from plan")
    expected_count = 800 if diagnostic else 1000
    if diagnostic:
        require(set(plan["objects"]) == set(OBJECTS), "Diagnostics require exactly the eight paper objects")
        require(plan["config"]["evaluation_protocol"] == "literal_public_diagnostic_v1", "Unexpected diagnostic protocol")
    else:
        require(set(plan["objects"]) == set(OBJECTS) | {"phone", "capsulemachine"}, "Unexpected baseline objects")
        require(plan["config"]["evaluation_protocol"] == "five_v1", "Unexpected baseline protocol")
    require(len(plan["episodes"]) == expected_count, "Incomplete planned grid: " + str(directory))
    require(len(rows) == expected_count, "Incomplete results: {} ({}/{})".format(directory, len(rows), expected_count))
    specs = {spec["episode_id"]: spec for spec in plan["episodes"]}
    require(len(specs) == expected_count, "Duplicate planned episode IDs")
    require(len({r["episode_id"] for r in rows}) == expected_count, "Duplicate result episode IDs")
    require({r["episode_id"] for r in rows} == set(specs), "Results differ from planned IDs")
    seen = set()
    for row in rows:
        spec = specs[row["episode_id"]]
        require(all(row.get(key) == spec.get(key) for key in IDENTITY_KEYS + ("profile",)), "Result differs from plan: " + row["episode_id"])
        # Runtime resolves articulation targets to float32 and may add resolved
        # fields; reference identity must match exactly, resolved floats need not.
        for key in ("lift_m", "translation_m", "orientation_delta_rad", "articulation_rule"):
            require(row["target"].get(key) == spec["target"].get(key), "Target convention differs from plan: " + key)
        require(row["profile"] == profile and row["split"] == split, "Wrong row profile/split")
        require(type(row["sample_index"]) is int and 0 <= row["sample_index"] < 100, "Invalid sample index")
        identity = (row["object"], row["sample_index"])
        require(identity not in seen, "Duplicate object/sample identity")
        seen.add(identity)
        for key in ("valid",) + SUCCESS_KEYS:
            require(type(row.get(key)) is bool, "Missing Boolean {} in {}".format(key, row["episode_id"]))
        if row["valid"]:
            for key in ERROR_KEYS + ("initial_articulation_rad",):
                value = row.get(key)
                require(type(value) in (int, float) and math.isfinite(value), "Invalid numerical metric {} in {}".format(key, row["episode_id"]))
                if key in ERROR_KEYS:
                    require(value >= 0, "Negative error metric")
        else:
            require(not row["public_success"] and not row["paper_success"], "Invalid trial counted as successful")
        require(row.get("paper_object") == (row["object"] in OBJECTS), "Incorrect paper membership")
    for obj in plan["objects"]:
        require({r["sample_index"] for r in rows if r["object"] == obj} == set(range(100)), "Incomplete object: " + obj)
    if diagnostic:
        require(saved_summary["overall"]["trials"] == expected_count, "Run summary is incomplete")
        require(set(saved_summary["by_object"]) == set(OBJECTS), "Run summary has incomplete objects")
        require({b["object"] for b in execution["batches"]} == set(OBJECTS), "Execution metadata is incomplete")
        require(len(execution["batches"]) == len(OBJECTS), "Duplicate execution objects")
        for key in ("public_success", "paper_success"):
            count = sum(r["valid"] and r[key] for r in rows)
            require(saved_summary["overall"][key]["successes"] == count, "Saved aggregate differs from rows")
        require(provenance.get("original_ketchup_relative_angle_observation_preserved") is True, "Missing literal ketchup diagnostic provenance")
    else:
        require(saved_summary.get("complete") is True and saved_summary["completed"] == expected_count, "Baseline incomplete")
    return {"path": str(directory), "plan": plan, "rows": rows, "provenance": provenance,
            "execution": execution, "input_sha256": {name: sha256(directory / name) for name in
              ("plan.json", "episodes.jsonl", "summary.json", "provenance.json", "execution.json")}}


def metrics(rows):
    require(rows, "Cannot summarize empty rows")
    valid = [r for r in rows if r["valid"]]
    result = {"trials": len(rows), "valid": len(valid), "invalid": len(rows) - len(valid)}
    for key in SUCCESS_KEYS + ("public_arrived", "public_opened", "public_orientation_reached", "public_terminal_guard"):
        require(all(type(row.get(key)) is bool for row in rows), "Missing saved Boolean: " + key)
        count = sum(r[key] for r in valid)
        result[key] = {"successes": count, "rate": count / len(rows)}
    arrivals = result["public_arrived"]["successes"]
    result["public_success_conditional_on_arrival"] = {
        "successes": result["public_success"]["successes"], "arrivals": arrivals,
        "rate": result["public_success"]["successes"] / arrivals if arrivals else None,
        "is_overall_task_success": False,
    }
    if all("ood_goal_success" in r for r in rows):
        count = sum(r["ood_goal_success"] for r in valid)
        result["ood_goal_success"] = {"successes": count, "rate": count / len(rows)}
    count = sum(r["paper_position_success"] and r["paper_orientation_success"] for r in valid)
    result["physical_base_pose_success"] = {"successes": count, "rate": count / len(rows)}
    for key in ERROR_KEYS:
        result[key] = math.fsum(r[key] for r in valid) / len(valid) if valid else None
    result["public_orientation_error_clipped_rad_max"] = max((r["public_orientation_error_clipped_rad"] for r in valid), default=None)
    result["error_measurements"] = len(valid)
    result["public_vs_final"] = {
        "both_success": sum(r["public_success"] and r["paper_success"] for r in valid),
        "public_only": sum(r["public_success"] and not r["paper_success"] for r in valid),
        "final_only": sum(r["paper_success"] and not r["public_success"] for r in valid),
    }
    return result


def summarize(run):
    rows = [r for r in run["rows"] if r["object"] in OBJECTS]
    return {"overall": metrics(rows), "by_object": {obj: metrics([r for r in rows if r["object"] == obj]) for obj in OBJECTS}}


def load_weights(path):
    rows = read_csv(path)
    require(len(rows) == 16, "Expected sixteen split/object label-count rows")
    weights = {}
    for split in ("train", "test"):
        records = [row for row in rows if row["split"] == split]
        require({r["object"] for r in records} == set(OBJECTS) and len(records) == 8, "Incomplete label-count weights")
        weights[split] = {
            "raw_all_grasp_x_articulation": {r["object"]: int(r["raw_all_grasp_x_articulation"]) for r in records},
            "compose_helper_pair_count": {r["object"]: int(r["compose_helper_pair_count"]) for r in records},
        }
    require(sum(weights["train"]["raw_all_grasp_x_articulation"].values()) == 7517, "Unexpected raw train pair count")
    require(sum(weights["test"]["raw_all_grasp_x_articulation"].values()) == 2091, "Unexpected raw test pair count")
    return weights


def reweight(summary, weights, condition, split, mode):
    total = sum(weights.values())
    require(total > 0, "Zero label-pair weights")
    result = {"condition": condition, "weight_split": split, "weight_mode": mode,
              "weight_pair_count": total, "weights": weights,
              "is_exhaustive_evaluation": False,
              "estimator": "Sum of measured per-object success rates times label-pair counts, divided by total pair count; no new trials and no exhaustive claim."}
    for key in ("public_success", "paper_success"):
        result[key] = math.fsum(summary["by_object"][obj][key]["rate"] * weights[obj] for obj in OBJECTS) / total
    return result


def paired_comparison(baseline, diagnostic):
    old = {(r["object"], r["sample_index"]): r for r in baseline["rows"] if r["object"] in OBJECTS}
    pairs, rejected = [], []
    for new in diagnostic["rows"]:
        previous = old[(new["object"], new["sample_index"])]
        identity_equal = all(previous.get(key) == new.get(key) for key in IDENTITY_KEYS)
        initial_equal = (previous.get("initial_articulation_rad") is not None and new.get("initial_articulation_rad") is not None
                         and abs(previous["initial_articulation_rad"] - new["initial_articulation_rad"]) <= 1e-6)
        if identity_equal and initial_equal:
            pairs.append((previous, new))
        else:
            rejected.append({"object": new["object"], "sample_index": new["sample_index"],
                             "same_reference_identity": identity_equal, "same_initial_articulation": initial_equal})
    result = {"definition": "Same object/sample/episode seed/split/full grasp+articulation label dictionaries and label indices, with initial object articulation agreeing within 1e-6 rad. Changed goal/controller bundle is intentional; no single-axis causal attribution.",
              "matched": len(pairs), "not_matched": len(rejected), "not_matched_details": rejected, "groups": []}
    for obj in ("paper8",) + OBJECTS:
        selected = [p for p in pairs if obj == "paper8" or p[0]["object"] == obj]
        if not selected:
            continue
        group = {"object": obj, "pairs": len(selected),
                 "both_valid": sum(a["valid"] and b["valid"] for a, b in selected)}
        for key in ("public_success", "paper_success"):
            before = sum(a["valid"] and a[key] for a, _ in selected)
            after = sum(b["valid"] and b[key] for _, b in selected)
            group[key] = {"before": before, "after": after, "before_rate": before / len(selected),
                          "after_rate": after / len(selected), "delta_pp": 100 * (after - before) / len(selected),
                          "both_success": sum(a["valid"] and a[key] and b["valid"] and b[key] for a, b in selected),
                          "before_only": sum(a["valid"] and a[key] and not (b["valid"] and b[key]) for a, b in selected),
                          "after_only": sum(b["valid"] and b[key] and not (a["valid"] and a[key]) for a, b in selected)}
        result["groups"].append(group)
    return result


def normalization_evidence(repo):
    vec_path = repo / "raisimGymTorch/raisimGymTorch/env/RaisimGymVecEnv.py"
    original_path = repo / "raisimGymTorch/raisimGymTorch/env/envs/compose_eval/compose_eval.py"
    runtime_path = repo / "tools/ood_suite/runtime.py"
    tree = ast.parse(vec_path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RaisimGymVecEnv")
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    defaults = dict(zip([arg.arg for arg in init.args.args][-len(init.args.defaults):], init.args.defaults))
    default_false = isinstance(defaults.get("normalize_ob"), ast.Constant) and defaults["normalize_ob"].value is False
    original_calls = [node for node in ast.walk(ast.parse(original_path.read_text())) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "VecEnv"]
    runtime_calls = [node for node in ast.walk(ast.parse(runtime_path.read_text())) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("VecEnv", "RaisimGymVecEnv")]
    original_uses_default = len(original_calls) == 1 and len(original_calls[0].args) <= 2 and not any(kw.arg == "normalize_ob" for kw in original_calls[0].keywords)
    runtime_explicit_false = any(any(kw.arg == "normalize_ob" and isinstance(kw.value, ast.Constant) and kw.value.value is False for kw in call.keywords) for call in runtime_calls)
    return {"method": "Static AST inspection of constructor default and VecEnv call sites; no physics claim.",
            "constructor_normalize_ob_default_false": default_false,
            "original_composed_runner_uses_constructor_default": original_uses_default,
            "runtime_explicit_normalize_ob_false": runtime_explicit_false,
            "both_observation_normalization_disabled": default_false and original_uses_default and runtime_explicit_false,
            "sources": [str(p.relative_to(repo)) for p in (vec_path, original_path, runtime_path)],
            "source_sha256": {str(p.relative_to(repo)): sha256(p) for p in (vec_path, original_path, runtime_path)}}


def global_state_layout_evidence(repo):
    cpp = repo / "raisimGymTorch/raisimGymTorch/env/envs/compose_eval/Environment.hpp"
    runner = repo / "raisimGymTorch/raisimGymTorch/env/envs/compose_eval/compose_eval.py"
    source = cpp.read_text()
    runner_source = runner.read_text()
    pattern = r"global_state_\s*<<.*?obj_state,\s*obj_goal_pos_true;"
    require(re.search(pattern, source, re.DOTALL), "Released global-state suffix layout changed")
    require("current_obj_quat = global_state[:, -5:-1]" in runner_source, "Released public orientation selector changed")
    return {
        "method": "Static source layout: six scalar features, right wrist 3+4 and hand 51, left wrist 3+4 and hand 51, object state 8, object goal 8.",
        "global_state_dimension": 138,
        "physical_object_state_indices": [122, 130],
        "physical_object_quaternion_indices": [125, 129],
        "goal_object_state_indices": [130, 138],
        "goal_object_quaternion_indices": [133, 137],
        "public_python_quaternion_slice": "-5:-1",
        "public_python_slice_resolves_to": [133, 137],
        "public_orientation_compares_target_against_target": True,
        "limitations": "Source/binary-layout agreement uses the environment's reported 138-field dimension checked by runtime, not a reconstruction of the original paper's binary.",
        "sources": {str(path.relative_to(repo)): sha256(path) for path in (cpp, runner)},
    }


def build_summary(baseline_dir, diagnosis_dir):
    diagnosis_dir = Path(diagnosis_dir).resolve()
    baseline = load_run(baseline_dir, "test", "normal", False)
    public_test = load_run(diagnosis_dir / "public_test", "test", "public_normal", True)
    public_train = load_run(diagnosis_dir / "public_train", "train", "public_normal", True)
    for name in ("trajectory_audit.json", "upstream_audit.json", "upstream_label_counts.csv", "trajectory_episode_rescore.csv", "controller_call_trace.json", "trajectory_conditional_metrics.json"):
        require((diagnosis_dir / name).is_file(), "Missing audit: " + str(diagnosis_dir / name))
    trajectory = read_json(diagnosis_dir / "trajectory_audit.json")
    upstream = read_json(diagnosis_dir / "upstream_audit.json")
    controller = read_json(diagnosis_dir / "controller_call_trace.json")
    conditional = read_json(diagnosis_dir / "trajectory_conditional_metrics.json")
    require(trajectory.get("mismatch_count") == 0, "Independent trajectory audit reports mismatches")
    paper8 = next(row for row in trajectory["summary"] if row["group"] == "paper8")
    require(paper8["n"] == 800, "Trajectory audit incomplete")
    require(upstream.get("all_counted_labels_identical_to_audited_upstream") is True, "Label provenance differs from upstream")
    calls = controller["reference_vs_public_diagnostic_controller"]
    require(calls and all(not row["differences"] and row["expected_calls"] == row["actual_calls"] for row in calls), "Public diagnostic controller call trace differs from release")
    for key in ("checkpoint_sha256", "diagnostic_script_sha256"):
        require(public_test["provenance"].get(key) == public_train["provenance"].get(key), "Diagnostic provenance mismatch: " + key)
    for key, digest in public_test["provenance"]["checkpoint_sha256"].items():
        path = public_test["plan"]["config"]["checkpoints"][key]
        require(baseline["plan"]["config"]["checkpoints"][key] == path and baseline["provenance"]["input_sha256"].get(path) == digest, "Baseline checkpoint differs: " + key)
    require(public_test["provenance"]["source_sha256"] == public_train["provenance"]["source_sha256"], "Diagnostic source snapshots differ")
    binary_hashes = {batch["compose_eval_sha256"] for run in (public_test, public_train) for batch in run["execution"]["batches"]}
    require(len(binary_hashes) == 1, "Different diagnostic simulator binaries")
    require(baseline["execution"]["runtime"]["compose_eval_sha256"] in binary_hashes, "Baseline simulator binary differs")
    bs, ts, rs = (summarize(run) for run in (baseline, public_test, public_train))
    for key, field in (("public_success", "public_success"), ("paper_success", "paper_success"), ("ood_goal_success", "ood_goal_success")):
        require(bs["overall"][key]["successes"] == paper8[field], "Baseline differs from independent trajectory audit: " + key)
    conditional_paper8 = conditional["results"]["paper8"]["valid_filter"]
    require(conditional_paper8["successes"] == bs["overall"]["public_success"]["successes"] and conditional_paper8["arrivals"] == bs["overall"]["public_arrived"]["successes"], "Conditional denominator audit differs from saved baseline")
    weights = load_weights(diagnosis_dir / "upstream_label_counts.csv")
    estimates = []
    for condition, summary, splits in (("normal_baseline", bs, ("test", "train")), ("public_test", ts, ("test",)), ("public_train", rs, ("train",))):
        for split in splits:
            for mode, object_weights in weights[split].items():
                estimates.append(reweight(summary, object_weights, condition, split, mode))
    rescore = read_csv(diagnosis_dir / "trajectory_episode_rescore.csv")
    require(len(rescore) == 1000 and len({r["episode_id"] for r in rescore}) == 1000, "Independent per-episode rescore incomplete")
    normalization = normalization_evidence(ROOT)
    summary = {
        "schema_version": 1,
        "complete": True,
        "all_numbers_from_saved_complete_runs": True,
        "reported_paper": upstream["paper"]["reported"],
        "paper_objects": list(OBJECTS),
        "conditions": {"normal_baseline": bs, "public_test": ts, "public_train": rs},
        "reweighting_estimates": estimates,
        "weights": weights,
        "paired_baseline_public_test": paired_comparison(baseline, public_test),
        "baseline_phase_and_failure_audit": paper8,
        "controller_call_audit": controller,
        "normalization_source_audit": normalization,
        "global_state_layout_source_audit": global_state_layout_evidence(ROOT),
        "baseline_conditional_denominator_audit": conditional["results"]["paper8"],
        "upstream_audit": upstream,
        "run_inputs": {name: {"path": run["path"], "sha256": run["input_sha256"], "planned": len(run["plan"]["episodes"]),
                               "completed": len(run["rows"]), "split": run["plan"]["split"], "samples_per_object": 100,
                               "provenance": run["provenance"], "execution": run["execution"]}
                       for name, run in (("normal_baseline", baseline), ("public_test", public_test), ("public_train", public_train))},
        "audit_input_sha256": {name: sha256(diagnosis_dir / name) for name in
                               ("trajectory_audit.json", "upstream_audit.json", "upstream_label_counts.csv", "trajectory_episode_rescore.csv", "controller_call_trace.json", "trajectory_conditional_metrics.json")},
        "source_urls": upstream["source_urls"],
        "limitations": [
            "Both diagnostic splits sample 100 reference pairs per object, with replacement, rather than the original exhaustive Cartesian product.",
            "The mixer selector was made usable by the common sampler; the literal generic upstream mixer command produces zero environments.",
            "Original saved evaluation bias draws are unavailable. Diagnostic target draws are newly sampled and are not numerically identical to the paper's draws.",
            "Baseline-to-public changes combine Euler orientation, articulation-target conventions, placement correction removal, random draws, and ketchup goal forwarding; aggregate changes do not isolate one cause.",
            "Reweighting uses measured per-object rates as estimates; changing weights does not create unmeasured trial outcomes or an exhaustive evaluation.",
            "The 7517 train-pair match to approximately 7500 paper trials is a split clue, not proof of the paper's evaluation split.",
            "Checkpoint identity for paper Table 3 and exact original trial selection remain unresolved.",
        ],
    }
    motion_path = diagnosis_dir / "trajectory_articulation_motion.json"
    if motion_path.is_file():
        summary["baseline_articulation_motion_audit"] = read_json(motion_path)
        summary["audit_input_sha256"][motion_path.name] = sha256(motion_path)
    hand_path = diagnosis_dir / "controller_hand_models.json"
    if hand_path.is_file():
        summary["hand_model_source_audit"] = read_json(hand_path)
        summary["audit_input_sha256"][hand_path.name] = sha256(hand_path)
    verified_path = diagnosis_dir / "trajectory_controlled_verification.json"
    if verified_path.is_file():
        verification = read_json(verified_path)
        for name in ("public_test", "public_train"):
            checked = verification["runs"][name]
            require(checked["episodes"] == 800 and checked["checked_trajectory_hashes"] == 800
                    and checked["independently_rescored_trajectories"] == 800
                    and checked["mismatch_count"] == 0, "Controlled trajectory rescore failed: " + name)
        summary["controlled_trajectory_verification"] = verification
        summary["audit_input_sha256"][verified_path.name] = sha256(verified_path)
    return summary


def number(value, digits=3):
    return "—" if value is None else ("{:0." + str(digits) + "f}").format(value)


def percent(value):
    return "—" if value is None else "{:.3f}%".format(100 * value)


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |"] + ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]


def render_report(summary):
    conditions = summary["conditions"]
    baseline, test, train = (conditions[key]["overall"] for key in ("normal_baseline", "public_test", "public_train"))
    sources = summary["source_urls"]
    phase = summary["baseline_phase_and_failure_audit"]
    paired = summary["paired_baseline_public_test"]
    paper = summary["reported_paper"]
    labels = {"normal_baseline": "前回normal（論文目標の解釈）", "public_test": "公開目標・test", "public_train": "公開目標・train"}
    lines = ["# 論文 Suc.T=0.50 との乖離の調査", "",
             "**同じ公開条件・同じ重みでも、公開判定はtest {}、train {}となり、参照splitの選択で{:.3f}ポイントの差が出ました。一方、論文記述の最終3条件ではtrainでも{}です。公開判定で50%前後が出ることと、論文定義の50%を再現することは別の結果です。**".format(percent(test["public_success"]["rate"]), percent(train["public_success"]["rate"]), 100 * (train["public_success"]["rate"] - test["public_success"]["rate"]), percent(train["paper_success"]["rate"])), "",
             "前回normalの論文対象8物体は、公開コード判定 {}、最終3条件 {}、高さ条件追加 {} です。同じ軌跡でも判定が違うため、前回の3.0%と論文50%は直接比較できません。".format(percent(baseline["public_success"]["rate"]), percent(baseline["paper_success"]["rate"]), percent(baseline["ood_goal_success"]["rate"])), "",
             "今回、公開実装の目標・制御条件で test {}試行、train {}試行を追加実行しました。公開判定は test {}／train {}、論文定義の最終3条件は test {}／train {} です。これらは各物体100試行の実測値です。論文 Table 3 の厳密再現が完了したという意味ではありません。".format(test["trials"], train["trials"], percent(test["public_success"]["rate"]), percent(train["public_success"]["rate"]), percent(test["paper_success"]["rate"]), percent(train["paper_success"]["rate"])), "",
             "## 比較する成功判定", "",
             "論文の Suc.T は最終 PE<0.05m、SO(3) AE<0.2rad、AAE<0.5rad の3条件です。中間の持ち上げ条件は含みません。Table 3 は50%、PE=0.03m、AE=0.10rad、AAE=0.41radを報告しています。[論文 §6.1・Table 3](" + sources["paper_v2"] + ")", "",
             "公開コード判定は、配置期間中に一度でも位置・Euler角成分の条件を達成し、開放期間中に一度でも角度条件を達成した各フラグを保持します。別々の時刻の達成を組み合わせ、最後に緩い非落下条件を掛けます。PE/AEは開放前に採取し、PE/AE/AAEは0.1m／0.3rad／1radでクリップします。論文式の最終同時達成やSO(3)誤差とは異なります。[公開評価コード](" + sources["compose_runner"] + ")", ""]
    lines += table(["同じ前回normal・8物体800試行", "成功数", "成功率"], [
        ["高さ条件を加えた従来OOD指標", baseline["ood_goal_success"]["successes"], percent(baseline["ood_goal_success"]["rate"])],
        ["論文記述の最終3条件", baseline["paper_success"]["successes"], percent(baseline["paper_success"]["rate"])],
        ["公開コードの時系列判定", baseline["public_success"]["successes"], percent(baseline["public_success"]["rate"])],
    ])
    if "controlled_trajectory_verification" in summary:
        lines += ["", "追加test/trainの全1,600軌跡も、保存ハッシュ・公開式・最終3条件を独立に再検証し、不一致は0件でした。追加testと前回normalの初期物理状態は800／800件で完全一致しています。[追加軌跡の独立検証](trajectory_controlled_verification.json)"]
    lines += ["", "保存された全1,000軌跡から、元の条件式と元の回転変換を用いて独立に再計算し、既存保存値との不一致は0件でした。以下の比較対象は論文対象8物体に限定します。[独立再計算](trajectory_audit.json)", "",
              "### 向き誤差の参照先が公開コード内で食い違う", "",
              "138次元のglobal_stateでは、実物体のquaternionは125:129、目標のquaternionは133:137です。元Pythonの `global_state[:, -5:-1]` は後者の目標quaternionを読みます。そのため公開AEは物体の実測姿勢を評価していません。前回normalの有効793試行の公開AE平均は {:.3g}rad、最大 {:.3g}radでほぼゼロでした。論文Table 3のAE=0.10radとは計測経路が一致しません。これは公開コード内の参照不整合で、論文50%の算出過程が同じであることを意味しません。[C++のstate格納順](".format(baseline["public_orientation_error_clipped_rad"], baseline["public_orientation_error_clipped_rad_max"]) +
              sources["compose_runner"].replace("compose_eval.py#L108", "Environment.hpp#L1661") + ")、[Pythonの向き参照](" + sources["compose_runner"] + ")", "",
              "Euler角の表現が折り返すケースでは、目標Euler角と目標quaternionから変換した角度に差が出るため、追加実験の公開AEが常に0になるわけではありません。この値にも実物体の回転は使われていません。", "",
              "### 全試行の成功率と到達済み試行だけの成功率", "",
              "元runnerはsuccess/num_envsに続いてsuccess/success_arriveも表示します。同じ8物体の保存軌跡で、有効性フィルタを含めると全試行では {}／800={}、到達済みだけなら {}／{}={} です。後者は位置到達に失敗した試行を分母から外す条件付き成功率です。論文は全体のタスク成功率を説明しており、この条件付き分母をTable 3に使った証拠はありません。[分母の独立確認](trajectory_conditional_metrics.json)、[元runnerの出力](".format(
                  baseline["public_success"]["successes"], percent(baseline["public_success"]["rate"]), baseline["public_success"]["successes"], baseline["public_arrived"]["successes"], percent(baseline["public_success_conditional_on_arrival"]["rate"])) + sources["compose_runner"] + ")", "",
              "## 公開実装の目標条件を使った追加実験", "",
              "公開実装はEuler-Y成分を変更し、初期関節角が1rad超なら初期角から0.5〜0.6radを引く目標を使います。ketchup は角度0.5〜1.0rad・向き±0.3rad・固有の制御を使います。前回normalは世界Z軸yaw・絶対角0.5〜0.6rad・支持面補正を使いました。追加実験はこれらの差とketchupの目標転送を公開コードに合わせ、目標乱数も再生成しています。差は複数条件の変更によるもので、Euler軸だけの効果とは判断できません。[通常runner](" + sources["compose_runner"] + ")、[ketchup runner](" + sources["ketchup_runner"] + ")", "",
              "各実験は8物体×100試行、seed=1、同じ公開重み・同じバイナリで計測しています。異常試行は成功率の分母に含めて失敗扱いとし、誤差平均から除きます。公開クリップ誤差と最終未クリップ誤差を分けて示します。", ""]
    lines += table(["条件", "試行", "公開判定", "最終3条件", "異常", "公開PE", "公開AE", "公開AAE"], [
        [labels[key], conditions[key]["overall"]["trials"], percent(conditions[key]["overall"]["public_success"]["rate"]),
         percent(conditions[key]["overall"]["paper_success"]["rate"]), conditions[key]["overall"]["invalid"]] +
        [number(conditions[key]["overall"][field]) for field in ERROR_KEYS[3:]]
        for key in ("normal_baseline", "public_test", "public_train")])
    lines += ["", "trainで公開判定が成功した{}試行のうち、{}試行（{:.1f}%）は論文記述の最終3条件を満たしません。したがって53.5%を論文Suc.T=0.50の再現値として扱うことはできません。".format(train["public_success"]["successes"], train["public_vs_final"]["public_only"], 100 * train["public_vs_final"]["public_only"] / train["public_success"]["successes"]), ""]
    lines += table(["公開条件", "全試行を分母", "到達済みだけを分母（条件付き）"], [
        [labels[key], percent(conditions[key]["overall"]["public_success"]["rate"]), percent(conditions[key]["overall"]["public_success_conditional_on_arrival"]["rate"])]
        for key in ("public_test", "public_train")])
    lines += ["", "公開PEはm、公開AE/AAEはrad。公開誤差はそれぞれクリップ済み・公開コードの採取時刻です。最終未クリップ誤差は次のとおりです。", ""]
    lines += table(["条件", "有効測定数", "最終PE (m)", "最終SO(3) AE (rad)", "最終AAE (rad)"], [
        [labels[key], conditions[key]["overall"]["error_measurements"]] + [number(conditions[key]["overall"][field]) for field in ERROR_KEYS[:3]]
        for key in ("normal_baseline", "public_test", "public_train")])
    lines += ["", "### 物体別の実測成功率", ""]
    lines += table(["物体", "前回公開", "追加test公開", "追加train公開", "前回最終3条件", "追加test最終3条件", "追加train最終3条件"], [
        [obj] + [percent(conditions[key]["by_object"][obj]["public_success"]["rate"]) for key in ("normal_baseline", "public_test", "public_train")] +
        [percent(conditions[key]["by_object"][obj]["paper_success"]["rate"]) for key in ("normal_baseline", "public_test", "public_train")]
        for obj in OBJECTS])
    lines += ["", "### 参照ラベルと初期角が一致するtest試行", "",
              "前回normalと追加testをobject/sample/seed/split/参照ラベル全体・ラベルindex・初期物体関節角で照合しました。該当 {}／800 試行で比較します。目標条件は変えているため、完全に同一の物理入力を再実行した比較ではありません。".format(paired["matched"]), ""]
    lines += table(["対象", "対応数", "公開判定: 前回→追加test", "Δ公開 (pp)", "最終3条件: 前回→追加test", "Δ最終3条件 (pp)"], [
        [row["object"], row["pairs"], "{} → {}".format(percent(row["public_success"]["before_rate"]), percent(row["public_success"]["after_rate"])),
         number(row["public_success"]["delta_pp"]), "{} → {}".format(percent(row["paper_success"]["before_rate"]), percent(row["paper_success"]["after_rate"])), number(row["paper_success"]["delta_pp"])]
        for row in paired["groups"]])
    lines += ["", "## 物体の均等平均と参照組合せの重み", "",
              "今回の各物体100試行は8物体を均等に平均します。論文は物体ごとの全grasp×articulation組合せを約7,500試行と記載しており、実測参照数ではtrain 7,517組、test 2,091組です。train参照488／test参照257も論文と一致します。この試行数の一致はsplit確認の強い手掛かりですが、論文がtrainを評価した証明ではありません。[論文 §6.2・B.1](" + sources["paper_v2"] + ")、[公開参照の計数](upstream_label_counts.csv)", "",
              "次は実測した物体別成功率に公開参照の組合せ数を掛けた**重み変更の推定値**です。全組合せを実行した結果ではなく、未測定の組合せの成否を補完したものでもありません。rawは全grasp種類、composeは公開helperがmixer/ketchupで左手graspだけを選ぶ数です。", ""]
    lines += table(["実測の条件", "重みのsplit", "重み", "組合せ数", "公開判定の推定", "最終3条件の推定"], [
        [labels[row["condition"]], row["weight_split"], "raw全種類" if row["weight_mode"] == "raw_all_grasp_x_articulation" else "compose選択後",
         row["weight_pair_count"], percent(row["public_success"]), percent(row["paper_success"])]
        for row in summary["reweighting_estimates"]])
    lines += ["", "## どの段階で失敗しているか", "",
              "前回normalの保存軌跡では、配置終了時に基部の位置・SO(3)向きが両方許容内だった {} 試行が、最終時点で {} 試行になっています。配置後に条件を失ったのは {}、後で条件に入ったのは {} です。この基部pose判定にはAAEを含みません。配置できても、手の退避・再接近・開放中に基部姿勢を保てない失敗が残っています。[段階別軌跡](trajectory_phase_metrics.csv)".format(phase["physical_pose_at_placement_end"], phase["physical_pose_final"], phase["pose_lost_after_placement"], phase["pose_gained_after_placement"]), ""]
    failures = phase["failure_stages"]
    lines += table(["公開判定の排他的内訳（前回normal）", "試行数", "全800中"], [
        [label, failures.get(key, 0), percent(failures.get(key, 0) / phase["n"])]
        for key, label in (("success", "成功"), ("arrival", "位置到達の未達成"), ("articulation", "開放条件の未達成"), ("terminal_guard", "最後の非落下条件"), ("invalid", "数値異常・自動reset"))])
    lines += ["", "内訳は無効→位置到達→向き→開放→最後の条件の順で最初の不合格を数えた階層集計です。失敗原因の独立した因果分解ではありません。数値異常は {}／800 で、これだけで50%との差を説明することはできません。[試行別判定](trajectory_episode_rescore.csv)".format(phase["n"] - phase["valid"]), "",
              "## 実装監査で確認できたことと残る不確定点", "",
              "元runnerの段階制御をASTから抽出し、同じ配列を入力した環境API呼出しを追加診断controllerと照合しました。検査した6物体・全呼出しが一致しました。前回controllerはketchupのstep1300だけに2個の追加goal転送があり、今回の診断では元コードに合わせています。この検査は制御呼出しの一致を確認するもので、全物理軌跡の一致を保証するものではありません。[呼出し監査](controller_call_trace.json)", ""]
    norm = summary["normalization_source_audit"]
    if norm["both_observation_normalization_disabled"]:
        lines += ["観測正規化はVecEnvの既定値がfalseで、元runnerは既定値を使用し、今回runtimeもfalseを明示します。平均・分散CSVの有無だけから正規化の欠落を原因とは判断できません。[VecEnv](../../raisimGymTorch/raisimGymTorch/env/RaisimGymVecEnv.py)、[runtime](../../tools/ood_suite/runtime.py)", ""]
    else:
        lines += ["観測正規化の静的検査は一致を確認できませんでした。詳細はsummary.jsonのnormalization_source_auditを参照してください。", ""]
    if "hand_model_source_audit" in summary:
        lines += ["checkpointに付属する学習設定と公開評価のdemo用手モデルも照合しました。質量・慣性・衝突形状・指関節設定は一致し、違いは両手各3か所の手首並進制限が±0.8mから±2.8mになる点だけです。demo用モデルの選択は初公開時から存在します。この制限差が成績に及ぼす因果効果は今回の追加実験では測定していません。[手モデルの独立監査](controller_hand_models.json)", ""]
    lines += ["公開READMEはketchup以外を汎用runnerと案内しますが、そのrunnerが使うfloating selectorはmixerのarticulation参照を除外して0環境になります。今回の共通samplerではmixerを評価可能にしています。ketchupの元専用runnerは20 train／9 test参照だけを使うため、全組合せとも異なります。公開コード自体に論文との評価集合の対応が明記されていない部分があります。[公開helper](" + sources["label_helper"] + ")、[ketchup runner](" + sources["ketchup_runner"] + ")", "",
              "著者は、比較用に保存したpos_bias.txt／angle_bias.txtはランダム目標で、再生成可能・必要なら共有可能と回答しています。元の乱数列は今回使用できません。また、公開READMEの事前学習モデルがTable 3を生成した重みそのものか、論文の最終集計にどのsplit・trial重みを使ったかは確認できていません。[著者の回答](" + sources["author_missing_targets"] + ")、[公開README](" + sources["readme"] + ")", "",
              "2025年のMANO指先collision修正は生成スクリプト等の変更で、配布済みruntimeのURDF/STLやcheckpointを更新する変更ではありません。今回の原因とする証拠はありません。[該当コミット](" + sources["later_collision_generation_fix"] + ")", "",
              "今回確認できたのは、判定式・採取時刻・目標の解釈・平均の重み・試行集合の対応が異なることと、公開条件をそろえた追加試行の実測値です。これらをそろえても残る差は、参照集合・全組合せ・元目標列・Table 3のcheckpointをそろえるまで確定的には帰属できません。", "",
              "## 保存した検証資料", "",
              "[全数値・入力ハッシュ](summary.json)、[追加test](public_test/summary.json)、[追加train](public_train/summary.json)、[公開実装・論文監査](upstream_audit.json)、[参照数の再現スクリプト](upstream_audit.py)、[保存軌跡の独立再計算](trajectory_audit.py)、[本レポートの再生成](../../scripts/report_normal_gap.py)", "",
              "再生成: `python3 scripts/report_normal_gap.py`。追加test/trainが各800試行、各物体100試行そろわない場合は出力せず停止します。"]
    return "\n".join(lines) + "\n"


def markdown_html(markdown):
    """Render this report's limited Markdown without optional dependencies."""
    def inline(value):
        escaped = html.escape(value)
        escaped = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', escaped)
        escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
        return re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    out, in_table = [], False
    for line in markdown.splitlines():
        if line.startswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                continue
            if not in_table:
                out.append('<div class="table"><table>')
                in_table = True
            out.append("<tr>" + "".join("<td>" + inline(cell) + "</td>" for cell in cells) + "</tr>")
            continue
        if in_table:
            out.append("</table></div>")
            in_table = False
        if not line:
            continue
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            out.append("<h{0}>{1}</h{0}>".format(level, inline(line[level:].strip())))
        else:
            out.append("<p>" + inline(line) + "</p>")
    if in_table:
        out.append("</table></div>")
    return ('<!doctype html><html lang="ja"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>論文 Suc.T=0.50 との乖離の調査</title><style>'
            'body{font:16px system-ui,sans-serif;background:#f5f7fa;color:#213047;margin:0}'
            'main{max-width:1320px;margin:auto;padding:32px}p{line-height:1.85}'
            'h2{margin-top:40px}table{border-collapse:collapse;width:100%;background:white}'
            'td{padding:10px;border-bottom:1px solid #dde3ea;text-align:right;white-space:nowrap}'
            'td:first-child{text-align:left}tr:first-child{background:#e8eef6;font-weight:bold}'
            '.table{overflow:auto;margin:16px 0}a{color:#175aa9}code{overflow-wrap:anywhere}'
            '</style><main>' + "\n".join(out) + '</main></html>\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "results/normal_seed1")
    parser.add_argument("--diagnosis", type=Path, default=ROOT / "results/normal_gap_diagnosis")
    parser.add_argument("--output", type=Path, help="Output directory; defaults to --diagnosis")
    args = parser.parse_args()
    output = (args.output or args.diagnosis).resolve()
    for input_run in (args.baseline.resolve(), (args.diagnosis / "public_test").resolve(), (args.diagnosis / "public_train").resolve()):
        require(output != input_run and input_run not in output.parents, "Report output must be separate from each immutable run")
    summary = build_summary(args.baseline, args.diagnosis)
    markdown = render_report(summary)
    page = markdown_html(markdown)
    serialized = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    # All completeness/identity/audit checks and rendering precede the first write.
    output.mkdir(parents=True, exist_ok=True)
    for name, content in (("report.md", markdown), ("report.html", page), ("summary.json", serialized)):
        (output / name).write_text(content)
    print(json.dumps({"report": str(output / "report.html"), "conditions": {name: data["overall"] for name, data in summary["conditions"].items()}}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, FileNotFoundError, StopIteration) as error:
        print("Refusing to generate incomplete or inconsistent gap report: " + str(error), file=sys.stderr)
        sys.exit(2)
