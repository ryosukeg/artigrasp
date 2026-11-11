#!/usr/bin/env python3
"""Report saved adaptive rollouts, separating point success from held success.

Only reads JSON/NPZ artifacts. No simulator, predictor, network or video renderer
is imported. Missing fixed-3s measurements remain in success denominators.
"""
import argparse
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path

import numpy as np


METHODS = ("baseline", "guided_absolute", "guided_delta")
PROFILES = ("normal", "ood_articulation")
LABELS = {"baseline": "通常のArtiGrasp", "guided_absolute": "絶対状態モデル", "guided_delta": "差分モデル"}
COLORS = {"baseline": "#475569", "guided_absolute": "#c66a23", "guided_delta": "#2563eb"}
REASONS = ("success_hold", "stalled", "timeout", "invalid")
REASON_LABELS = {"success_hold": "成功維持", "stalled": "観測上の停滞", "timeout": "時間上限", "invalid": "無効終了"}
ERRORS = {"PE": "paper_position_error_m", "AAE": "paper_articulation_error_rad", "AE": "paper_orientation_error_rad"}
SUCCESSES = ("paper_success", "ood_goal_success", "strict_goal_success")


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _read(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _require(value, message):
    if not value:
        raise ValueError(message)


def _finite(value):
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_)) and math.isfinite(float(value))


def _distribution(values, total):
    values = [float(value) for value in values if _finite(value)]
    return {"n": len(values), "excluded_or_missing": total - len(values),
            "mean": float(np.mean(values)) if values else None,
            "median": float(np.median(values)) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def _rate(values):
    values = list(values)
    count = sum(bool(value) for value in values)
    return {"count": count, "n": len(values), "rate": count / len(values) if values else None}


def _stage(rows, fixed=False):
    metrics = [(row.get("fixed_3s_metrics") if fixed else row) for row in rows]
    valid = [isinstance(value, dict) and value.get("valid") is True for value in metrics]
    result = {"n": len(rows), "valid": sum(valid), "invalid_or_missing": len(rows) - sum(valid),
              "missing_measurement": sum(value is None for value in metrics)}
    result["success"] = {key: _rate(good and value.get(key) is True for good, value in zip(valid, metrics)) for key in SUCCESSES}
    result["errors"] = {name: _distribution((value.get(key) if good else None for good, value in zip(valid, metrics)), len(rows))
                        for name, key in ERRORS.items()}
    return result


def _changes(rows):
    result = {}
    for key in SUCCESSES:
        pairs = [(row.get("fixed_3s_metrics"), row) for row in rows]
        measured = [(a, b) for a, b in pairs if isinstance(a, dict) and a.get("valid") is True and b.get("valid") is True]
        result[key] = {"paired_valid_n": len(measured), "excluded_or_missing_pairs": len(rows)-len(measured),
                       "gain": sum(not a.get(key, False) and b.get(key, False) for a, b in measured),
                       "loss": sum(a.get(key, False) and not b.get(key, False) for a, b in measured)}
    result["errors_adaptive_minus_fixed"] = {
        name: _distribution((row[key] - row["fixed_3s_metrics"][key]
                             if isinstance(row.get("fixed_3s_metrics"), dict)
                             and row["fixed_3s_metrics"].get("valid") is True and row.get("valid") is True
                             and _finite(row.get(key)) and _finite(row["fixed_3s_metrics"].get(key)) else None for row in rows), len(rows))
        for name, key in ERRORS.items()}
    return result


def _groups(rows):
    result = {}
    for profile in PROFILES:
        result[profile] = {}
        for method in METHODS:
            subset = [row for row in rows if row["profile"] == profile and row["method"] == method]
            result[profile][method] = {
                "n": len(subset), "fixed_3s": _stage(subset, True), "adaptive": _stage(subset),
                "termination": {reason: _rate(row["termination"]["reason"] == reason for row in subset) for reason in REASONS},
                "opening_seconds": _distribution((row["termination"]["opening_seconds"] for row in subset), len(subset)),
                "total_seconds": _distribution((row["termination"]["total_seconds"] for row in subset), len(subset)),
                "first_point_success_seconds": _distribution((row["termination"]["first_success_step"] * .01
                    if row["termination"]["first_success_step"] >= 0 else None for row in subset), len(subset)),
                "changes_from_fixed_3s": _changes(subset),
            }
    return result


def _same_value(a, b):
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return type(a) is type(b) and a == b
    return _finite(a) and _finite(b) and math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-8)


def _qangle(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    an, bn = np.linalg.norm(a), np.linalg.norm(b)
    _require(an > 1e-12 and bn > 1e-12, "Zero quaternion in a valid saved endpoint")
    return float(2*np.arccos(np.clip(abs(np.dot(a/an, b/bn)), 0., 1.)))


def _check_endpoint(state, goal, metric, label):
    if metric.get("valid") is not True:
        _require(all(metric.get(key) is False for key in SUCCESSES), label + ": invalid result marked successful")
        _require(all(metric.get(key) is None for key in ERRORS.values()), label + ": invalid result has numeric errors")
        return
    _require(np.isfinite(state).all(), label + ": nonfinite state in valid measurement")
    measured = {"paper_position_error_m": float(np.linalg.norm(state[122:125].astype(np.float64) - goal[:3])),
                "paper_articulation_error_rad": abs(float(state[129]) - float(goal[7])),
                "paper_orientation_error_rad": _qangle(state[125:129], goal[3:7])}
    for key, value in measured.items():
        _require(_same_value(metric.get(key), value), label + ": endpoint metric mismatch: " + key)
    success = measured[ERRORS["PE"]] < .05 and measured[ERRORS["AAE"]] < .5 and measured[ERRORS["AE"]] < .2
    _require(metric.get("paper_success") is bool(success), label + ": simultaneous endpoint success mismatch")


def _source_comparison(output, plan, rows):
    source = Path(plan["source_run"]).expanduser()
    if not source.is_absolute():
        source = output / source
    source = source.resolve()
    original_plan = _read(source / "plan.json")
    original = _read(source / "results.json")
    _require(original_plan["episodes"] == plan["episodes"], "Source run episode specifications differ")
    _require(original.get("complete") is True and original["plan_sha256"] == _sha(source / "plan.json"), "Source results are incomplete or changed")
    lookup = {(row["profile"], row["episode_id"], row["method"]): row for row in original["episodes"]}
    _require(len(lookup) == len(original["episodes"]), "Duplicate source results")
    expected = {(row["profile"], row["episode_id"], row["method"]) for row in rows}
    _require(set(lookup) == expected, "Source/adaptive result slots differ")
    compared, unavailable, differences = 0, [], []
    keys = ("valid",) + SUCCESSES + tuple(ERRORS.values())
    for row in rows:
        fixed = row.get("fixed_3s_metrics")
        if fixed is None:
            unavailable.append({"episode_id": row["episode_id"], "method": row["method"]})
            continue
        previous = lookup[(row["profile"], row["episode_id"], row["method"])]
        changed = [key for key in keys if not _same_value(fixed.get(key), previous.get(key))]
        if changed:
            differences.append({"episode_id": row["episode_id"], "method": row["method"], "fields": changed})
        compared += 1
    qualitative = source / "qualitative/index.html"
    return {"source_run": str(source), "source_plan_sha256": _sha(source / "plan.json"),
            "source_results_sha256": _sha(source / "results.json"), "compared_trials": compared,
            "unavailable_fixed_measurements": unavailable, "differences": differences,
            "all_available_metrics_match": not differences,
            "numeric_tolerance": {"rtol": 1e-6, "atol": 1e-8},
            "scope": "Saved fixed-3s metrics only; full action/state prefix equivalence requires the independent audit",
            "qualitative_href": os.path.relpath(str(qualitative), str(output)) if qualitative.is_file() else None}


def _load_rows(output, plan, results):
    _require(plan.get("methods") == list(METHODS), "Expected the three frozen methods")
    _require(results.get("complete") is True and results.get("plan_sha256") == _sha(output / "plan.json"), "Incomplete results or plan hash mismatch")
    settings = plan["termination"]
    _require(settings["control_dt"] == .01 and settings["opening_start_step"] == 1260, "Unexpected time base")
    specifications = {(row["profile"], row["episode_id"]): row for row in plan["episodes"]}
    _require(len(specifications) == len(plan["episodes"]), "Duplicate planned episodes")
    expected = {(profile, episode, method) for profile, episode in specifications for method in METHODS}
    lookup = {(row["profile"], row["episode_id"], row["method"]): row for row in results["episodes"]}
    _require(len(lookup) == len(results["episodes"]) and set(lookup) == expected, "Missing, duplicate or unplanned result slots")
    rows, series, hashes = [], {}, {}
    for index, key in enumerate(sorted(expected)):
        row = lookup[key]
        _require(row["profile"] in PROFILES, "Unknown profile")
        _require(all(row.get(name) == value for name, value in specifications[key[:2]].items()), "Result differs from planned episode")
        termination = row["termination"]
        step = termination["opening_step"]
        reason = termination["reason"]
        _require(type(step) is int and 0 <= step <= settings["max_opening_steps"] and reason in REASONS, "Invalid termination record")
        _require(reason == "invalid" or step >= settings["min_opening_steps"], "Valid termination before minimum duration")
        _require(reason != "timeout" or step == settings["max_opening_steps"], "Timeout below maximum duration")
        _require(_same_value(termination["opening_seconds"], step*.01)
                 and _same_value(termination["total_seconds"], (1260+step)*.01), "Termination time/step mismatch")
        first = termination["first_success_step"]
        _require(type(first) is int and -1 <= first <= step, "Invalid first-success step")
        _require(type(row.get("valid")) is bool and (reason == "invalid") == (not row["valid"]), "Termination reason/validity mismatch")
        _require(reason != "success_hold" or row.get("paper_success") is True, "Held success without terminal success")
        path = (output / row["trajectory_path"]).resolve()
        _require(output in path.parents, "Escaping trajectory path")
        relative_path = str(path.relative_to(output))
        _require(relative_path not in hashes, "Reused trajectory path")
        digest = _sha(path)
        _require(digest == row["trajectory_sha256"], "Trajectory SHA256 mismatch: " + str(path))
        hashes[relative_path] = digest
        with np.load(path, allow_pickle=False) as saved:
            states, goal = saved["final_states"], saved["goal"]
            _require(states.shape == (step+1, 138) and goal.shape == (8,), "Adaptive trajectory shape mismatch")
            _require(saved["control_dt"].item() == .01, "Trajectory control_dt mismatch")
        target = np.asarray(row["target_position_m"] + row["target_quaternion_wxyz"] + [row["target_articulation_rad"]], np.float64)
        _require(np.isfinite(goal).all() and np.allclose(goal, target, rtol=0, atol=1e-6), "Trajectory goal mismatch")
        _check_endpoint(states[-1], target, row, str(key) + " adaptive")
        fixed = row.get("fixed_3s_metrics")
        if step < 300:
            _require(fixed is None, "Fixed-3s snapshot exists before 3 seconds")
        else:
            _require(isinstance(fixed, dict), "Missing available fixed-3s snapshot")
            _check_endpoint(states[300], target, fixed, str(key) + " fixed-3s")
        # A 20 Hz plot keeps every endpoint and the exact fixed-3s comparison.
        indices = sorted(set(range(0, len(states), 5)) | {len(states)-1} | ({300} if step >= 300 else set()))
        series[key] = {"time_s": [round(i*.01, 6) for i in indices],
                       "angle_rad": [float(states[i, 129]) if np.isfinite(states[i, 129]) else None for i in indices]}
        rows.append(row)
        if (index+1) % 100 == 0:
            print("Adaptive report: verified {}/{} saved trajectories".format(index+1, len(expected)), flush=True)
    return rows, series, hashes


def _cases(plan, rows, series):
    lookup = {(row["profile"], row["episode_id"], row["method"]): row for row in rows}
    result = []
    for spec in sorted(plan["episodes"], key=lambda value: (PROFILES.index(value["profile"]), value["episode_id"])):
        key = (spec["profile"], spec["episode_id"])
        first = lookup[key + (METHODS[0],)]
        item = {"episode_id": spec["episode_id"], "profile": spec["profile"],
                "target_angle_rad": first["target_articulation_rad"], "methods": {}}
        for method in METHODS:
            row = lookup[key + (method,)]
            fixed = row.get("fixed_3s_metrics")
            item["methods"][method] = {"termination": row["termination"], "valid": row["valid"],
                "adaptive": {name: row.get(field) for name, field in ERRORS.items()},
                "fixed_3s": {name: fixed.get(field) for name, field in ERRORS.items()} if isinstance(fixed, dict) else None,
                "paper_success": row.get("paper_success", False), "ood_goal_success": row.get("ood_goal_success", False),
                "series": series[key + (method,)]}
        result.append(item)
    return result


def _load_video_manifest(output, rows):
    """Validate optional rendered media without changing evaluation metrics."""
    folder = output / "qualitative"
    path = folder / "manifest.json"
    if not path.is_file():
        return None
    manifest = _read(path)
    _require(manifest.get("schema_version") == 1 and manifest.get("status") in ("partial", "complete"),
             "Unsupported qualitative manifest")
    for field, source in (("source_plan_sha256", "plan.json"), ("source_results_sha256", "results.json"),
                          ("source_audit_sha256", "independent_audit.json")):
        _require(manifest.get(field) == _sha(output / source), "Qualitative source hash mismatch: " + source)
    audit = _read(output / "independent_audit.json")
    _require(audit.get("status") == "completed"
             and audit.get("plan_sha256") == manifest["source_plan_sha256"]
             and audit.get("results_sha256") == manifest["source_results_sha256"],
             "Qualitative media requires the completed audit of these results")
    lookup = {(row["profile"], row["episode_id"], row["method"]): row for row in rows}
    _require(isinstance(manifest.get("cases"), list), "Qualitative cases must be a list")
    videos, seen, media_paths = [], set(), set()
    for case in manifest["cases"]:
        key = (case["profile"], case["episode_id"])
        _require(key not in seen and all(key + (method,) in lookup for method in METHODS),
                 "Duplicate or unknown qualitative case")
        seen.add(key)
        _require(set(case["terminations"]) == set(METHODS), "Qualitative termination methods mismatch")
        maximum_step = 0
        for method in METHODS:
            expected = lookup[key + (method,)]["termination"]
            _require(case["terminations"][method] == expected, "Qualitative stop record differs from evaluation")
            maximum_step = max(maximum_step, 1260 + expected["opening_step"])
        _require(_same_value(case["opening_start"], 12.6) and _same_value(case["duration"], maximum_step*.01),
                 "Qualitative physical duration mismatch")
        fps, frames = case["fps"], case["frame_state_indices"]
        _require(_finite(fps) and fps > 0 and isinstance(frames, list) and len(frames) >= 2
                 and all(type(value) is int for value in frames) and frames[0] == 0 and frames[-1] == maximum_step
                 and all(a < b for a, b in zip(frames, frames[1:])), "Invalid qualitative frame mapping")
        _require(_same_value(case["encoded"]["duration"], len(frames)/float(fps)),
                 "Encoded duration/frame count mismatch")
        mappings = case["method_frame_state_indices"]
        _require(set(mappings) == set(METHODS), "Qualitative method frame mapping mismatch")
        for method in METHODS:
            end = 1260 + case["terminations"][method]["opening_step"]
            _require(mappings[method] == [min(step, end) for step in frames],
                     "Qualitative stopped method must freeze at its actual endpoint")
        record = {name: case[name] for name in ("episode_id", "profile", "fps", "duration", "opening_start", "frame_state_indices", "terminations")}
        record["encoded_duration"] = case["encoded"]["duration"]
        record["selection_reason"] = case.get("selection_reason", "旧固定時間版と同じ試行を採用。新結果で再選抜していません。")
        for field in ("video", "poster", "terminal_image"):
            relative = case[field]
            _require(isinstance(relative, str) and not Path(relative).is_absolute(), "Media path must be relative")
            asset = (folder / relative).resolve()
            _require(folder.resolve() in asset.parents and asset not in media_paths, "Escaping or reused media path")
            _require(asset.is_file() and _sha(asset) == case[field + "_sha256"], "Qualitative asset hash mismatch: " + relative)
            media_paths.add(asset)
            record[field] = asset.relative_to(output).as_posix()
        videos.append(record)
    return {"status": manifest["status"], "manifest_sha256": _sha(path), "cases": videos}


def _fmt(value, digits=4):
    return ("{:." + str(digits) + "f}").format(value) if _finite(value) else "—"


def _cell(rate):
    return "{} / {} ({}%)".format(rate["count"], rate["n"], _fmt(100*rate["rate"] if rate["rate"] is not None else None, 1))


def _markdown(summary):
    stop, validation = summary["termination_settings"], summary["validation"]
    dt = stop["control_dt"]
    minimum, maximum, hold = stop["min_opening_steps"]*dt, stop["max_opening_steps"]*dt, stop["success_hold_steps"]*dt
    lines = ["# 開閉時間を適応的に延長した3方式比較", "",
             "seed {}、{} 目標 × 3 方式 = {} 試行の探索的比較です。参照 split は {}。".format(summary["evaluation_seed"], summary["planned_episodes"], summary["planned_trials"], summary["split"]),
             "同じ試行の固定3秒時点と適応終了時点を並べています。公開 PPO と予測モデルの重みは変更していません。", "",
             "## 終了条件", "",
             "開閉を最低 {} 秒、最大 {} 秒実行します。終了の優先順は無効→成功維持→観測上の停滞→時間上限です。".format(_fmt(minimum,1), _fmt(maximum,1)),
             "- 成功維持: PE < {} m、AAE < {} rad、AE < {} rad の3条件が {} 秒（{} 状態）連続で成立。従来の paper_success は終点1時点での3条件達成であり、別の指標です。".format(stop["success_position_threshold_m"],stop["success_articulation_threshold_rad"],stop["success_orientation_threshold_rad"],_fmt(hold,1),stop["success_hold_steps"]+1),
             "- 観測上の停滞: 直近 {} 秒の蓋角度レンジ < {} rad、基部位置の先頭状態からの最大変位 < {} m、姿勢変位 < {} rad。現在の3条件成功が未達のときだけ適用します。これは到達不可能の証明ではありません。".format(_fmt(stop["stall_window_steps"]*dt,1),stop["stall_articulation_range_rad"],stop["stall_position_displacement_m"],stop["stall_orientation_displacement_rad"]),
             "- 無効試行は成功率の全数分母に含めます。固定3秒に到達しない試行も固定時点の成功数は0、誤差は欠測です。平均誤差は各時点の有効測定だけから算出します。", "",
             "## 固定3秒と適応終端", "",
             "終点3条件成功 = paper_success、lift込み = ood_goal_success、strict = lift込みで角度閾値0.1 rad。誤差は平均値で、括弧内は測定数です。", "",
             "| 条件 | 方式 | 時点 | 全数 | 無効・欠測 | PE [m] | AAE [rad] | AE [rad] | 終点3条件 | lift込み | strict |",
             "|---|---|---|---:|---:|---:|---:|---:|---|---|---|"]
    for profile in PROFILES:
        for method in METHODS:
            for stage, label in (("fixed_3s", "固定3秒"), ("adaptive", "適応終端")):
                row = summary["groups"][profile][method][stage]
                errors = ["{} (n={})".format(_fmt(row["errors"][name]["mean"], 6 if name=="AAE" else 4), row["errors"][name]["n"]) for name in ERRORS]
                lines.append("| {} | {} | {} | {} | {} | {} | {} |".format("normal" if profile=="normal" else "OOD", LABELS[method],label,row["n"],row["invalid_or_missing"]," | ".join(errors)," | ".join(_cell(row["success"][key]) for key in SUCCESSES)))
    lines += ["", "## 終了理由と開閉時間", "", "時間は各試行の実際の終了時刻を使います。開閉開始より前の12.6秒は含みません。無効終了も時間の集計に含めます。", "",
              "| 条件 | 方式 | 成功維持 | 観測上の停滞 | 上限終了 | 無効終了 | 開閉時間 平均 / 中央値 [s] |",
              "|---|---|---|---|---|---|---|"]
    for profile in PROFILES:
        for method in METHODS:
            row=summary["groups"][profile][method]
            lines.append("| {} | {} | {} | {} / {} |".format("normal" if profile=="normal" else "OOD",LABELS[method]," | ".join(_cell(row["termination"][key]) for key in REASONS),_fmt(row["opening_seconds"]["mean"],2),_fmt(row["opening_seconds"]["median"],2)))
    lines += ["", "## 旧固定時間評価との照合", "",
              "元 run: `{}`。固定3秒測定 {} 試行を照合、差異 {}、測定未到達 {}。数値許容差 rtol=1e-6、atol=1e-8。".format(summary["source_comparison"]["source_run"], summary["source_comparison"]["compared_trials"],len(summary["source_comparison"]["differences"]),len(summary["source_comparison"]["unavailable_fixed_measurements"])),
              "このレポートは trajectory SHA256、保存終点のPE/AAE/AE、停止step/秒数、固定3秒の値を検査しています。全 action/state prefix の一致を確認する独立監査とは別です。",
              "独立監査の状態: `{}`。".format(validation["independent_audit"]["status"])]
    if summary["source_comparison"]["differences"]:
        lines += ["**旧固定時間評価との差異があります。差異の詳細は summary.json を確認し、対応比較の解釈を保留してください。**"]
    if summary["source_comparison"]["qualitative_href"]:
        lines += ["", "[旧3秒固定評価の動作比較動画]({})".format(summary["source_comparison"]["qualitative_href"])]
    lines += ["", "停止時刻は方式によって異なります。適応終端同士の比較には、制御の効果と観察時間の違いの両方が含まれます。成功維持による終了は途中liftを条件に含めず、lift込み成功は別に報告します。",
              "予測モデルの訓練データは開閉開始から3秒までの収集に基づきます。3秒を超えた延長区間では状態分布が変わり得るため、同じ予測精度は保証されません。単一 seed の探索的比較であり、seed 間の変動は未評価です。",
              "normal と OOD の成功率をそのまま難易度比較に使わないでください。小角OODの目標0.2–0.4 radは初期角度が0付近でもAAE < 0.5 radを満たし得るため、同じ条件内で方式を比較する必要があります。",
              "角度図は保存した final_states から0.05秒間隔で抽出し、各方式の実際の終了点を必ず含めます。終了後に曲線を引き延ばす補間は行いません。", ""]
    return "\n".join(lines)


def _csv(summary):
    fields = ["profile","method","stage","n","valid","invalid_or_missing","missing_measurement"]
    fields += [key+suffix for key in SUCCESSES for suffix in ("_count","_rate")]
    fields += [name+suffix for name in ERRORS for suffix in ("_mean","_median","_n")]
    fields += ["termination_"+reason+"_count" for reason in REASONS]
    fields += ["opening_seconds_mean","opening_seconds_median"]
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
    for profile in PROFILES:
        for method in METHODS:
            group=summary["groups"][profile][method]
            for stage in ("fixed_3s","adaptive"):
                stats=group[stage]
                row={"profile":profile,"method":method,"stage":stage,**{key:stats[key] for key in ("n","valid","invalid_or_missing","missing_measurement")}}
                for key in SUCCESSES:
                    for suffix in ("count","rate"):row[key+"_"+suffix]=stats["success"][key][suffix]
                for name in ERRORS:
                    for suffix in ("mean","median","n"):row[name+"_"+suffix]=stats["errors"][name][suffix]
                if stage=="adaptive":
                    row.update({"termination_"+reason+"_count":group["termination"][reason]["count"] for reason in REASONS})
                    row.update(opening_seconds_mean=group["opening_seconds"]["mean"],opening_seconds_median=group["opening_seconds"]["median"])
                writer.writerow(row)
    return stream.getvalue()


def _atomic(path, text):
    temporary=path.with_name(path.name+".tmp")
    with temporary.open("w",encoding="utf-8",newline="") as stream:
        stream.write(text);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def write_report(output):
    output=Path(output).expanduser().resolve()
    plan,results=_read(output/"plan.json"),_read(output/"results.json")
    rows,series,hashes=_load_rows(output,plan,results)
    source=_source_comparison(output,plan,rows)
    audit_path=output/"independent_audit.json"
    audit={"status":"not_available", "note":"独立監査はまだ確認できません。"}
    if audit_path.is_file():
        value=_read(audit_path)
        plan_matches=value.get("plan_sha256")==_sha(output/"plan.json")
        results_matches=value.get("results_sha256")==_sha(output/"results.json")
        status=value.get("status","unknown") if plan_matches and results_matches else "stale_plan" if not plan_matches else "stale_results"
        audit={"status":status,"recorded_status":value.get("status","unknown"),
               "sha256":_sha(audit_path),"plan_matches":plan_matches,"results_matches":results_matches}
    summary={"schema_version":1,"report_type":"adaptive_opening_comparison","generated_at_utc":datetime.now(timezone.utc).isoformat(),
             "evaluation_seed":plan["seed"],"split":plan["split"],"purpose":plan.get("purpose"),
             "planned_episodes":len(plan["episodes"]),"planned_trials":len(rows),"termination_settings":plan["termination"],
             "methods":[{"id":method,"label":LABELS[method],"color":COLORS[method]} for method in METHODS],
             "groups":_groups(rows),"source_comparison":source,
             "validation":{"plan_sha256":_sha(output/"plan.json"),"results_sha256":_sha(output/"results.json"),
                           "report_source_sha256":_sha(__file__),"trajectory_sha256":hashes,"verified_trajectories":len(hashes),
                           "endpoint_metric_tolerance":{"rtol":1e-6,"atol":1e-8},"independent_audit":audit},
             "metric_definitions":{"paper_success":"simultaneous terminal PE<0.05m, AAE<0.5rad, AE<0.2rad",
                                   "success_hold":"termination.reason == success_hold; sustained success, distinct from one-time terminal success",
                                   "stalled":"observed motion below fixed thresholds, not a proof of impossibility",
                                   "denominators":"all planned trials; invalid/missing count as failure, errors use valid measurements only"}}
    presentation={"summary":summary,"cases":_cases(plan,rows,series),"reason_labels":REASON_LABELS,
                  "videos":_load_video_manifest(output,rows)}
    payload=json.dumps(presentation,ensure_ascii=False,allow_nan=False)
    for literal,escaped in (("&","\\u0026"),("<","\\u003c"),(">","\\u003e"),("\u2028","\\u2028"),("\u2029","\\u2029")):
        payload=payload.replace(literal,escaped)
    _atomic(output/"summary.json",json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    _atomic(output/"metricstable.csv",_csv(summary))
    _atomic(output/"ADAPTIVE_REVIEW.md",_markdown(summary))
    _atomic(output/"index.html",_HTML.replace("__REPORT_DATA__",payload))
    return summary


_HTML=r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ArtiGrasp 適応終了の比較</title><style>
:root{font-family:system-ui,-apple-system,"Segoe UI","Noto Sans JP",sans-serif;color:#203044;background:#f3f5f8}*{box-sizing:border-box}body{margin:0}main{max-width:1420px;padding:28px 22px;margin:auto}h1{font-size:29px;margin:0 0 10px}h2{font-size:20px;margin:0 0 12px}p{line-height:1.75;font-size:13px;color:#56677a}.card{background:#fff;border:1px solid #dce3eb;border-radius:10px;padding:21px;margin:20px 0}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums;white-space:nowrap}th,td{padding:10px 9px;border-bottom:1px solid #e7ecf1;text-align:right}th{background:#f5f7fa;color:#5c6d80;font-weight:600;font-size:11px}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}.group-start td{border-top:2px solid #d3dce7}.muted{color:#728296;font-size:11px}.notice{padding:12px 15px;background:#edf3fa;border-radius:7px;font-size:13px;line-height:1.8}.warning{background:#fff0d7;color:#855b17}.method{border-left:4px solid var(--color);padding-left:7px}.controls{display:flex;align-items:center;gap:9px;flex-wrap:wrap}button,select{font:inherit;background:white;border:1px solid #d3dde8;border-radius:7px;padding:9px 11px;color:inherit}button{cursor:pointer}button:disabled{opacity:.45;cursor:default}select{max-width:100%;min-width:220px}button:focus-visible,select:focus-visible,a:focus-visible{outline:3px solid #7ea6db;outline-offset:3px}.case-title{display:flex;justify-content:space-between;align-items:center;gap:15px;margin:20px 0 10px}.target{color:#13834c;font-weight:650;font-size:16px}#angleChart{width:100%;display:block;height:auto;cursor:crosshair}.legend{display:flex;gap:17px;flex-wrap:wrap;margin:8px 5px;font-size:12px}.stop-bars{display:grid;gap:9px;margin:15px 0 20px}.bar-row{display:grid;grid-template-columns:150px 1fr 190px;align-items:center;gap:12px;font-size:12px}.track{height:9px;background:#eaf0f5;border-radius:6px;overflow:hidden}.bar{height:100%;border-radius:6px}.bar-label{font-variant-numeric:tabular-nums;color:#56677a}.links{display:flex;gap:17px;flex-wrap:wrap;font-size:13px}a{color:#2b609c}.good{color:#147642}.bad{color:#728195}#cursorText{font-size:12px;font-variant-numeric:tabular-nums;color:#4a5e75;min-height:20px}.note{font-size:12px;color:#64758a;line-height:1.8}details{font-size:12px;line-height:1.8;margin-top:13px}summary{cursor:pointer;font-weight:600}ul{padding-left:22px}#caseTitle{font-size:15px;word-break:break-all}#dataNotice{margin-top:12px}@media(max-width:700px){main{padding:20px 12px}h1{font-size:24px}.card{padding:15px 10px}.bar-row{grid-template-columns:115px 1fr;gap:6px}.bar-label{grid-column:2}.case-title{align-items:flex-start}.target{font-size:13px}.controls select{min-width:0;max-width:100%}.legend{gap:10px;font-size:10px}.case-title{flex-direction:column;gap:4px}}
[hidden]{display:none!important}.video-filter{display:flex;align-items:center;gap:5px;font-size:12px}.video-filter input{accent-color:#2563eb}#mediaNotice{margin:12px 0 5px}#mediaEmpty{padding:14px;border:1px dashed #cfd9e5;border-radius:7px}#mediaPanel{margin:15px 0 23px}.media-frame{background:#142030;border-radius:8px;overflow:hidden;aspect-ratio:24/11}.media-frame video,.media-frame img{display:block;width:100%;height:100%;object-fit:contain}#mediaControls{margin-top:11px}#mediaControls select{min-width:84px}.media-status{font-size:12px;font-variant-numeric:tabular-nums;line-height:1.7;color:#50647c}#mediaMessage{color:#99611a;min-height:0}.media-reason{font-size:12px;line-height:1.7;margin:8px 0;color:#60738a}.playback-clock{font-size:12px;color:#566b82;margin:7px 0 0}
</style></head><body><main>
<h1>開閉時間を適応的に延長した3方式比較</h1><p id="intro"></p>
<div id="protocol" class="notice"></div><div id="dataNotice" class="notice"></div>
<section class="card"><h2>固定3秒時点と適応終了時点</h2><p>成功率は全試行を分母とし、無効・欠測は失敗に含めます。誤差は有効測定の平均値、括弧内は測定数です。終点3条件成功と、一定時間成功を維持した終了は別の指標です。</p><div class="scroll"><table><thead><tr><th>条件</th><th>方式</th><th>時点</th><th>全数</th><th>無効・欠測</th><th>PE [m]</th><th>AAE [rad]</th><th>AE [rad]</th><th>終点3条件</th><th>lift込み</th><th>strict</th></tr></thead><tbody id="metricRows"></tbody></table></div></section>
<section class="card"><h2>終了理由と開閉時間</h2><p>成功維持は連続成立を条件とし、途中の持ち上げは条件に含みません。「観測上の停滞」は到達不可能を意味しません。時間は開閉開始からの経過秒数で、前段の12.6秒を除いています。</p><div class="scroll"><table><thead><tr><th>条件</th><th>方式</th><th>成功維持</th><th>観測上の停滞</th><th>上限終了</th><th>無効終了</th><th>開閉時間 平均 [s]</th><th>中央値 [s]</th></tr></thead><tbody id="terminationRows"></tbody></table></div></section>
<section class="card"><h2>各試行の動画・角度と終了時刻</h2><div class="controls"><label>条件 <select id="profile"><option value="all">全部</option><option value="normal">normal</option><option value="ood_articulation">OOD</option></select></label><select id="caseSelect" aria-label="比較する試行"></select><label class="video-filter"><input id="videoOnly" type="checkbox">動画ありのみ</label><button id="previous" type="button">前へ</button><button id="next" type="button">次へ</button><span id="caseCount" class="muted"></span></div>
<p id="mediaNotice" class="note"></p><div class="case-title"><strong id="caseTitle"></strong><span id="target" class="target"></span></div><p id="mediaEmpty" class="note"></p>
<div id="mediaPanel" hidden><p id="mediaReason" class="media-reason"></p><div class="media-frame"><video id="comparisonVideo" controls playsinline preload="metadata" aria-label="同じ試行の3方式を並べた適応終了動画"></video><img id="terminalImage" alt="3方式それぞれの実際の終点状態" hidden></div><div id="mediaControls" class="controls"><button id="playToggle" type="button">再生</button><button id="openingJump" type="button">開閉開始 12.6秒から</button><button id="terminalToggle" type="button">終点画像</button><label>速度 <select id="playbackRate"><option value="1">1×</option><option value="0.5">0.5×</option><option value="0.25">0.25×</option></select></label><label class="video-filter"><input id="openingLoop" type="checkbox">開閉区間をループ</label></div><p id="mediaClock" class="playback-clock"></p><div id="mediaStops" class="media-status"></div><p id="mediaMessage" class="media-status" role="status"></p><p class="note">保存状態とURDFから描画した映像です。再シミュレーションは行っていません。動画は最も遅い方式の終了まで続き、先に終了した列は実際の終点で静止します。STOPPED以降の静止表示は、終了後の計測結果や動作の継続を示すものではありません。上段の物体拡大は中心を追うため、基部の移動は下段の共通固定視点とPEで確認してください。動画の代表例は全600試行の集計を置き換えるものではありません。</p></div>
<div id="legend" class="legend"></div>
<svg id="angleChart" viewBox="0 0 1100 340" role="img" aria-label="3方式の蓋角度、目標角度と実際の終了点"></svg><div id="cursorText"></div><div id="stopBars" class="stop-bars"></div>
<div class="scroll"><table><thead><tr><th>方式</th><th>終了理由</th><th>開閉 / 全体 [s]</th><th>固定3秒 AAE [rad]</th><th>終点 PE [m]</th><th>終点 AAE [rad]</th><th>終点 AE [rad]</th><th>終点3条件</th></tr></thead><tbody id="caseRows"></tbody></table></div><p class="note">曲線は保存された物理状態から0.05秒刻みで抽出し、各方式の実際の終了点を丸印で示しています。終了後に状態を引き延ばして表示しません。破線は固定3秒時点、緑は目標角度です。図の上にポインタを置くとその時刻の角度を確認でき、動画がある場合はクリックで対応する時刻へ移動できます。</p></section>
<section class="card"><h2>比較の範囲と照合</h2><p id="sourceCheck"></p><p id="auditStatus"></p><p>方式ごとに終了時刻が異なるため、適応終端の差には制御と観察時間の違いの両方が含まれます。固定3秒時点を併せて確認してください。単一 seed の探索的結果であり、seed 間の変動は評価していません。</p><p>予測モデルの訓練データは開閉開始から3秒までの収集に基づきます。3秒を超えた延長区間では状態分布が変わり得るため、同じ予測精度は保証されません。</p><p>normal と OOD の成功率をそのまま難易度比較に使わないでください。小角OODの目標0.2–0.4 radは初期角度が0付近でもAAE &lt; 0.5 radを満たし得るため、同じ条件内で方式を比較する必要があります。</p>
<div class="links"><a href="ADAPTIVE_REVIEW.md">詳細説明</a><a href="metricstable.csv" download>集計 CSV</a><a href="summary.json">集計 JSON</a><a id="qualitative" hidden>旧3秒固定評価の動作比較動画</a></div><details><summary>指標と停止条件</summary><ul id="definitions"></ul></details></section>
</main><script id="reportData" type="application/json">__REPORT_DATA__</script><script>
"use strict";(()=>{
const data=JSON.parse(document.getElementById("reportData").textContent),s=data.summary,p=s.termination_settings,methods=s.methods,profiles=["normal","ood_articulation"],reasonIds=["success_hold","stalled","timeout","invalid"];
const $=id=>document.getElementById(id),el=(tag,text,cls)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e},svg=(tag,attrs,text)=>{const e=document.createElementNS("http://www.w3.org/2000/svg",tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,String(v));if(text!==undefined)e.textContent=text;return e};
const fmt=(v,n=4)=>Number.isFinite(v)?v.toLocaleString("ja-JP",{minimumFractionDigits:n,maximumFractionDigits:n}):"—",rate=v=>v.count+" / "+v.n+" ("+fmt(v.rate===null?null:100*v.rate,1)+"%)",profileName=v=>v==="normal"?"normal":"OOD",td=v=>el("td",v),methodTd=m=>{const c=td();const b=el("span",m.label,"method");b.style.setProperty("--color",m.color);c.append(b);return c};
const minimum=p.min_opening_steps*p.control_dt,maximum=p.max_opening_steps*p.control_dt,hold=p.success_hold_steps*p.control_dt;
const videoManifest=data.videos,videoMap=new Map((videoManifest?videoManifest.cases:[]).map(item=>[item.episode_id,item]));
$("videoOnly").disabled=videoMap.size===0;$("videoOnly").checked=videoMap.size>0;
$("mediaNotice").textContent=videoManifest?(videoManifest.status==="partial"?"動画生成中：現時点 ":"比較動画 ")+videoMap.size+" 本。動画フィルターはケース選択だけに適用され、上の全600試行の集計は変わりません。":"比較動画はまだ追加されていません。全試行の保存角度と終了時刻を確認できます。";
$("intro").textContent="seed "+s.evaluation_seed+" · "+s.planned_episodes+" 目標 × 3 方式 = "+s.planned_trials+" 試行 · 参照 "+s.split+"。同じ試行の開閉3秒時点と、その後の適応終了を比較します。";
$("protocol").textContent="開閉は最低 "+fmt(minimum,1)+" 秒、最大 "+fmt(maximum,1)+" 秒。3条件成功を "+fmt(hold,1)+" 秒維持、観測上の停滞、または時間上限で終了します。数値異常は即時終了します。";
$("dataNotice").textContent="保存軌道 "+s.validation.verified_trajectories+" 件のハッシュ・終点指標・時刻を確認。全 action/state の対応は下記の独立監査状態をご確認ください。";
for(const profile of profiles)for(const method of methods){const g=s.groups[profile][method.id];for(const[stage,name]of[["fixed_3s","固定3秒"],["adaptive","適応終端"]]){const r=g[stage],tr=el("tr",undefined,stage==="fixed_3s"?"group-start":"");tr.append(td(profileName(profile)),methodTd(method),td(name),td(String(r.n)),td(String(r.invalid_or_missing)));for(const name of["PE","AAE","AE"]){const v=r.errors[name];tr.append(td(fmt(v.mean,name==="AAE"?6:4)+" (n="+v.n+")"));}for(const key of["paper_success","ood_goal_success","strict_goal_success"])tr.append(td(rate(r.success[key])));$("metricRows").append(tr);}const tr=el("tr");tr.append(td(profileName(profile)),methodTd(method));for(const reason of reasonIds)tr.append(td(rate(g.termination[reason])));tr.append(td(fmt(g.opening_seconds.mean,2)),td(fmt(g.opening_seconds.median,2)));$("terminationRows").append(tr);}
const source=s.source_comparison;$("sourceCheck").textContent="旧固定時間評価との固定3秒指標照合："+source.compared_trials+" 件、差異 "+source.differences.length+" 件、3秒測定未到達 "+source.unavailable_fixed_measurements.length+" 件。これは全軌跡のprefix照合とは別です。";if(source.differences.length){$("dataNotice").classList.add("warning");$("dataNotice").textContent+=" 旧固定時間評価との差異があるため、対応比較の解釈は保留してください。";}
const audit=s.validation.independent_audit;$("auditStatus").textContent="独立監査："+(audit.status==="completed"?"完了（現在の plan.json・results.json のハッシュに一致）":audit.status==="not_available"?"未確認。独立監査結果はまだ保存されていません。":audit.status==="stale_plan"||audit.status==="stale_results"?"現在の計画または結果と監査時のハッシュが異なるため、再監査が必要です。":audit.status);if(source.qualitative_href){$("qualitative").href=source.qualitative_href;$("qualitative").hidden=false;}
for(const text of["終点3条件成功：PE < 0.05 m、AAE < 0.5 rad、AE < 0.2 rad。","成功維持：同じ3条件が連続 "+fmt(hold,1)+" 秒（"+(p.success_hold_steps+1)+" 状態）成立。終点1時点の成功とは別です。","lift込み成功は途中の持ち上げ条件も要求。strictはlift込みで角度誤差 < 0.1 rad。","観測上の停滞：直近 "+fmt(p.stall_window_steps*p.control_dt,1)+" 秒の角度レンジ < "+p.stall_articulation_range_rad+" rad、基部変位 < "+p.stall_position_displacement_m+" m、姿勢変位 < "+p.stall_orientation_displacement_rad+" rad。現在の3条件成功が未達の場合のみ適用。","成功率の分母は全数。無効・固定時点未到達も失敗として数え、誤差の平均は有効測定のみ。", "終了理由の優先順：無効、成功維持、観測上の停滞、時間上限。停滞は到達不可能の証明ではありません。"])$("definitions").append(el("li",text));
let visible=[],current=null,plot=null;const chart=$("angleChart"),video=$("comparisonVideo");
let activeMedia=null,terminalView=false,mediaGeneration=0,pendingSeek=null,animation=null,graphHover=false;
function stopAnimation(){if(animation!==null)cancelAnimationFrame(animation);animation=null;}
function currentPhysicalTime(){if(!activeMedia)return null;if(terminalView)return activeMedia.duration;const indices=activeMedia.frame_state_indices,index=Math.max(0,Math.min(indices.length-1,Math.floor(video.currentTime*activeMedia.fps+1e-6)));return indices[index]*p.control_dt;}
function showChartCursor(time,prefix=""){if(!plot||!current)return;if(time<0){plot.cursor.setAttribute("visibility","hidden");$("cursorText").textContent=prefix+"開閉開始前";return;}const t=Math.max(0,Math.min(maximum,time));plot.cursor.setAttribute("visibility","visible");plot.cursor.setAttribute("x1",plot.x(t));plot.cursor.setAttribute("x2",plot.x(t));$("cursorText").textContent=prefix+fmt(t,2)+" s · "+methods.map(m=>{const v=valueAt(current.methods[m.id].series,t);return m.label+" "+(v===null?"終了後 / 未測定":fmt(v,4)+" rad");}).join(" / ");}
function updatePlayback(){if(!activeMedia)return;const time=currentPhysicalTime(),opening=time-activeMedia.opening_start;$("mediaClock").textContent=(terminalView?"各方式の終点画像 · ":"映像の物理時刻 ")+fmt(time,2)+" s / "+fmt(activeMedia.duration,2)+" s · "+(opening<0?"開閉開始前":"開閉開始から "+fmt(opening,2)+" s");$("mediaStops").textContent=methods.map(m=>{const stop=activeMedia.terminations[m.id],stopped=time+1e-8>=stop.total_seconds;return m.label+"："+(stopped?"停止（"+data.reason_labels[stop.reason]+"） 開閉 "+fmt(stop.opening_seconds,2)+" s / 全体 "+fmt(stop.total_seconds,2)+" s":opening<0?"開閉前":"実行中");}).join("　｜　");$("playToggle").textContent=video.paused||terminalView?"再生":"一時停止";if(!graphHover)showChartCursor(opening,"映像対応：");}
function tickPlayback(){animation=null;updatePlayback();if(activeMedia&&!video.paused&&!video.ended&&!terminalView)animation=requestAnimationFrame(tickPlayback);}
function showVideo(){terminalView=false;$("terminalImage").hidden=true;video.hidden=false;$("terminalToggle").textContent="終点画像";updatePlayback();}
function safePlay(){if(!activeMedia)return;showVideo();const generation=mediaGeneration;$("mediaMessage").textContent="";try{const promise=video.play();if(promise&&typeof promise.catch==="function")promise.catch(()=>{if(generation===mediaGeneration)$("mediaMessage").textContent="再生を開始できませんでした。動画内の再生ボタンを押してください。";});}catch(error){$("mediaMessage").textContent="このブラウザでは再生できませんでした。動画の再生ボタンをお試しください。";}}
function seekPhysical(time,playAfter=false){if(!activeMedia)return;showVideo();const frames=activeMedia.frame_state_indices,target=Math.max(0,Math.min(activeMedia.duration,time))/p.control_dt;let lo=0,hi=frames.length-1;while(lo<hi){const mid=Math.floor((lo+hi)/2);if(frames[mid]<target)lo=mid+1;else hi=mid;}if(lo>0&&Math.abs(frames[lo-1]-target)<Math.abs(frames[lo]-target))lo--;const request={time:lo/activeMedia.fps,play:playAfter,generation:mediaGeneration};if(video.readyState<1){pendingSeek=request;return;}try{video.currentTime=request.time;pendingSeek=null;updatePlayback();if(playAfter)safePlay();}catch(error){pendingSeek=request;$("mediaMessage").textContent="動画の読込後に指定時刻へ移動します。";}}
function setMedia(item){mediaGeneration++;activeMedia=null;pendingSeek=null;terminalView=false;graphHover=false;stopAnimation();video.pause();video.removeAttribute("src");video.removeAttribute("poster");video.load();$("terminalImage").removeAttribute("src");$("terminalImage").hidden=true;video.hidden=false;$("terminalToggle").textContent="終点画像";$("mediaMessage").textContent="";$("mediaClock").textContent="";$("mediaStops").textContent="";activeMedia=item?videoMap.get(item.episode_id)||null:null;$("mediaPanel").hidden=!activeMedia;$("mediaEmpty").hidden=!!activeMedia;$("mediaEmpty").textContent=item?"この試行の動画は未作成です。保存角度と終点指標は下で確認できます。":"該当する動画・試行はありません。";if(!activeMedia)return;$("mediaReason").textContent=activeMedia.selection_reason;video.poster=activeMedia.poster;video.src=activeMedia.video;video.playbackRate=Number($("playbackRate").value);$("terminalImage").src=activeMedia.terminal_image;video.load();updatePlayback();}
video.addEventListener("loadedmetadata",()=>{if(!activeMedia)return;video.playbackRate=Number($("playbackRate").value);if(pendingSeek&&pendingSeek.generation===mediaGeneration){const request=pendingSeek;pendingSeek=null;video.currentTime=request.time;updatePlayback();if(request.play)safePlay();}else updatePlayback();});
for(const event of["timeupdate","seeked","durationchange"])video.addEventListener(event,updatePlayback);
video.addEventListener("play",()=>{stopAnimation();tickPlayback();});video.addEventListener("pause",()=>{stopAnimation();updatePlayback();});
video.addEventListener("ended",()=>{stopAnimation();updatePlayback();if(activeMedia&&$("openingLoop").checked&&!terminalView)seekPhysical(activeMedia.opening_start,true);});
video.addEventListener("error",()=>{if(activeMedia)$("mediaMessage").textContent="動画を読み込めませんでした。MP4ファイルがHTMLと同じフォルダー構成で保存されているか確認してください。";});
$("terminalImage").addEventListener("error",()=>{if(activeMedia)$("mediaMessage").textContent="終点画像を読み込めませんでした。画像ファイルの保存場所を確認してください。";});
$("playToggle").addEventListener("click",()=>{if(!activeMedia)return;if(!video.paused&&!terminalView){video.pause();return;}if(video.ended||($("openingLoop").checked&&currentPhysicalTime()<activeMedia.opening_start))seekPhysical($("openingLoop").checked?activeMedia.opening_start:0,true);else safePlay();});
$("openingJump").addEventListener("click",()=>{if(activeMedia)seekPhysical(activeMedia.opening_start,true);});
$("terminalToggle").addEventListener("click",()=>{if(!activeMedia)return;if(terminalView){showVideo();return;}video.pause();stopAnimation();terminalView=true;video.hidden=true;$("terminalImage").hidden=false;$("terminalToggle").textContent="動画へ戻る";updatePlayback();});
$("playbackRate").addEventListener("change",()=>{video.playbackRate=Number($("playbackRate").value);});
$("openingLoop").addEventListener("change",()=>{if(activeMedia&&!terminalView&&$("openingLoop").checked&&currentPhysicalTime()<activeMedia.opening_start)seekPhysical(activeMedia.opening_start,!video.paused);});
function valueAt(series,time){const t=series.time_s,v=series.angle_rad;if(time<t[0]||time>t[t.length-1]+1e-8)return null;let lo=0,hi=t.length-1;while(lo<hi){const mid=Math.floor((lo+hi+1)/2);if(t[mid]<=time)lo=mid;else hi=mid-1;}const next=Math.min(lo+1,t.length-1);if(!Number.isFinite(v[lo]))return null;if(!Number.isFinite(v[next])||next===lo||t[next]===t[lo])return v[lo];return v[lo]+(v[next]-v[lo])*(time-t[lo])/(t[next]-t[lo]);}
function renderCase(item){current=item;$("caseTitle").textContent=profileName(item.profile)+" · "+item.episode_id;$("target").textContent="目標 "+fmt(item.target_angle_rad,3)+" rad ("+fmt(item.target_angle_rad*180/Math.PI,1)+"°)";$("legend").replaceChildren();$("stopBars").replaceChildren();$("caseRows").replaceChildren();chart.replaceChildren();$("cursorText").textContent="";
const values=[0,item.target_angle_rad];for(const m of methods)for(const a of item.methods[m.id].series.angle_rad)if(Number.isFinite(a))values.push(a);const low=Math.min(...values)-.08,high=Math.max(.2,...values)+.15,w=1100,h=340,l=58,r=22,top=28,bottom=35,x=t=>l+(w-l-r)*t/maximum,y=a=>h-bottom-(h-top-bottom)*(a-low)/(high-low);
for(let i=0;i<=4;i++){const a=low+(high-low)*i/4,yy=y(a);chart.append(svg("line",{x1:l,x2:w-r,y1:yy,y2:yy,stroke:"#e5ebf2"}),svg("text",{x:l-9,y:yy+4,"text-anchor":"end",fill:"#708298","font-size":11},fmt(a,2)));}for(let i=0;i<=5;i++){const t=maximum*i/5;chart.append(svg("text",{x:x(t),y:h-11,"text-anchor":"middle",fill:"#708298","font-size":11},fmt(t,1)+" s"));}chart.append(svg("text",{x:9,y:15,fill:"#607388","font-size":11},"rad"),svg("line",{x1:x(3),x2:x(3),y1:top,y2:h-bottom,stroke:"#9aa9ba","stroke-dasharray":"5 5"}),svg("text",{x:x(3)+6,y:18,fill:"#708298","font-size":11},"固定3秒"),svg("line",{x1:l,x2:w-r,y1:y(item.target_angle_rad),y2:y(item.target_angle_rad),stroke:"#16834b","stroke-width":2,"stroke-dasharray":"8 5"}),svg("text",{x:l+5,y:y(item.target_angle_rad)-6,fill:"#16834b","font-size":11},"目標"));
for(const method of methods){const record=item.methods[method.id],series=record.series,end=record.termination.opening_seconds;let path="",connected=false;series.time_s.forEach((t,i)=>{const a=series.angle_rad[i];if(!Number.isFinite(a)){connected=false;return;}path+=(connected?"L":"M")+x(t).toFixed(2)+","+y(a).toFixed(2)+" ";connected=true;});chart.append(svg("path",{d:path,fill:"none",stroke:method.color,"stroke-width":2.2,"stroke-linejoin":"round"}));const angle=series.angle_rad[series.angle_rad.length-1];if(Number.isFinite(angle)){const point=svg("circle",{cx:x(end),cy:y(angle),r:5,fill:method.color,stroke:"white","stroke-width":1.5});point.append(svg("title",{},method.label+" / "+data.reason_labels[record.termination.reason]+" / "+fmt(end,2)+"秒"));chart.append(point);}const legend=el("span",method.label+" · "+fmt(end,2)+"秒","method");legend.style.setProperty("--color",method.color);$("legend").append(legend);
const barRow=el("div",undefined,"bar-row"),track=el("div",undefined,"track"),bar=el("div",undefined,"bar");bar.style.width=(100*end/maximum)+"%";bar.style.background=method.color;track.append(bar);barRow.append(el("span",method.label),track,el("span",fmt(end,2)+" s · "+data.reason_labels[record.termination.reason],"bar-label"));$("stopBars").append(barRow);
const tr=el("tr");tr.append(methodTd(method),td(data.reason_labels[record.termination.reason]),td(fmt(end,2)+" / "+fmt(record.termination.total_seconds,2)),td(fmt(record.fixed_3s?record.fixed_3s.AAE:null,6)),td(fmt(record.adaptive.PE)),td(fmt(record.adaptive.AAE,6)),td(fmt(record.adaptive.AE)),td(record.paper_success?"成功":"未達"));$("caseRows").append(tr);}
const cursor=svg("line",{x1:l,x2:l,y1:top,y2:h-bottom,stroke:"#243951","stroke-width":1,"stroke-dasharray":"3 3",visibility:"hidden"});chart.append(cursor);plot={width:w,left:l,right:w-r,x,cursor};const index=visible.indexOf(item);$("previous").disabled=index<=0;$("next").disabled=index>=visible.length-1;$("caseSelect").value=item.episode_id;setMedia(item);}
function updateChoices(){const profile=$("profile").value,onlyVideo=$("videoOnly").checked;visible=data.cases.filter(item=>(profile==="all"||item.profile===profile)&&(!onlyVideo||videoMap.has(item.episode_id)));$("caseSelect").replaceChildren();for(const item of visible){const option=el("option",(videoMap.has(item.episode_id)?"[動画] ":"")+item.episode_id+" · 目標 "+fmt(item.target_angle_rad,3)+" rad");option.value=item.episode_id;$("caseSelect").append(option);}$("caseCount").textContent=visible.length+" / "+data.cases.length+" 目標"+(onlyVideo?"（動画あり）":"");if(visible.length)renderCase(visible.includes(current)?current:visible[0]);else{current=null;plot=null;chart.replaceChildren();$("caseTitle").textContent="該当する試行はありません";$("target").textContent="";$("cursorText").textContent="";for(const id of["legend","stopBars","caseRows"])$(id).replaceChildren();$("previous").disabled=true;$("next").disabled=true;setMedia(null);}}
$("profile").addEventListener("change",updateChoices);$("videoOnly").addEventListener("change",updateChoices);$("caseSelect").addEventListener("change",()=>{const item=visible.find(value=>value.episode_id===$("caseSelect").value);if(item)renderCase(item);});$("previous").addEventListener("click",()=>{const i=visible.indexOf(current);if(i>0)renderCase(visible[i-1]);});$("next").addEventListener("click",()=>{const i=visible.indexOf(current);if(i+1<visible.length)renderCase(visible[i+1]);});
function chartTime(event){const bounds=chart.getBoundingClientRect(),px=(event.clientX-bounds.left)*plot.width/bounds.width;return Math.max(0,Math.min(maximum,(px-plot.left)/(plot.right-plot.left)*maximum));}
chart.addEventListener("pointermove",event=>{if(!current||!plot)return;graphHover=true;showChartCursor(chartTime(event));});chart.addEventListener("pointerleave",()=>{graphHover=false;if(activeMedia)updatePlayback();else if(plot)plot.cursor.setAttribute("visibility","hidden");});
chart.addEventListener("click",event=>{if(!current||!plot||!activeMedia)return;const time=chartTime(event);graphHover=false;seekPhysical(activeMedia.opening_start+time,false);});
updateChoices();
})();</script></body></html>'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    summary=write_report(args.output)
    print("Adaptive report: {} trials; {}".format(summary["planned_trials"],args.output/"index.html"),flush=True)


if __name__=="__main__":
    main()
