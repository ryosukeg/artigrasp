#!/usr/bin/env python3
"""Audit saved three-way rollouts before reporting paired normal/OOD results."""
import argparse
import copy
import csv
import io
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools"))
try:
    from .evaluate_report import (GROUPS, ERRORS, CANDIDATES, _sha256, _finite, _group,
                                  _fmt, _success_cell, summarize_rows, paired_summary)
except ImportError:
    from evaluate_report import (GROUPS, ERRORS, CANDIDATES, _sha256, _finite, _group,
                                 _fmt, _success_cell, summarize_rows, paired_summary)
from ood_suite.metrics import quaternion_geodesic_rad, evaluate_goal_diagnostics

METHODS = ("baseline", "guided_absolute", "guided_delta")
METHOD_LABELS = {"baseline": "PPO", "guided_absolute": "PPO + 元 absolute", "guided_delta": "PPO + delta"}
GROUP_LABELS = {"normal": "normal", "ood_articulation": "OOD 全体", "ood_small": "OOD 小角側", "ood_large": "OOD 大角側"}
TIMING = {"protocol": "standard", "grasp_steps": 260, "transit_steps": 300, "arti_steps": 400}


def _assert(condition, message):
    if not condition:
        raise ValueError(message)


def _invalid_slot(spec, method, reason):
    row = dict(spec, method=method, valid=False, measurement_unverified=True,
               measurement_issue=reason, target_articulation_rad=spec.get("target", {}).get("articulation_rad"))
    row.update({key: None for key in ERRORS})
    row.update(paper_success=False, ood_goal_success=False, strict_goal_success=False)
    return row


def _load_checked(output, row, spec):
    path = (output / row["trajectory_path"]).resolve()
    _assert(output in path.parents, "Trajectory is outside evaluation directory")
    _assert(_sha256(path) == row["trajectory_sha256"], "Trajectory SHA256 mismatch")
    shapes = {"states": (1561, 138), "final_states": (301, 138), "goal": (8,),
              "actions_r": (1560, 51), "actions_l": (1560, 51),
              "ppo_actions_r": (1560, 51), "ppo_actions_l": (1560, 51),
              "initial_invalid": (), "terminal_bad_observation": ()}
    shapes.update({key: (1560,) for key in ("bad_observation", "bad_action", "bad_state", "auto_reset")})
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in shapes}
    for key, shape in shapes.items():
        _assert(arrays[key].shape == shape and arrays[key].dtype.kind in "biuf", "Invalid array layout: " + key)
    for key in ("initial_invalid", "terminal_bad_observation", "bad_observation", "bad_action", "bad_state", "auto_reset"):
        _assert(arrays[key].dtype.kind == "b", "Non-Boolean invalid flag: " + key)
    for key, value in spec.items():
        _assert(row.get(key) == value, "Result disagrees with planned metadata: " + key)
    goal = arrays["goal"]
    _assert(np.isfinite(goal).all() and np.linalg.norm(goal[3:7]) > 1e-8, "Invalid saved goal")
    expected_goal = np.asarray(row["target_position_m"] + row["target_quaternion_wxyz"] + [row["target_articulation_rad"]])
    _assert(expected_goal.shape == (8,) and np.allclose(goal, expected_goal, atol=1e-6, rtol=0), "Saved goal disagrees with result target")
    _assert(np.isclose(goal[7], spec["target"]["articulation_rad"], atol=1e-6, rtol=0), "Saved angle disagrees with planned target")
    _assert(type(row.get("valid")) is bool, "Missing Boolean trial validity")
    has_event = any(bool(arrays[key].any()) for key in
                    ("initial_invalid", "terminal_bad_observation", "bad_observation", "bad_action", "bad_state", "auto_reset"))
    if row["valid"]:
        _assert(not has_event and all(np.isfinite(value).all() for value in arrays.values()), "Valid trial contains invalid event/state/action")
        final = arrays["states"][-1]
        measured = {
            "paper_position_error_m": float(np.linalg.norm(final[122:125].astype(float) - expected_goal[:3])),
            "paper_orientation_error_rad": quaternion_geodesic_rad(final[125:129], expected_goal[3:7]),
            "paper_articulation_error_rad": abs(float(final[129]) - float(expected_goal[7])),
        }
        for key, value in measured.items():
            _assert(_finite(row.get(key)) and np.isclose(value, row[key], atol=1e-8, rtol=1e-6), "Final metric disagrees with saved trajectory: " + key)
        paper = measured[ERRORS[0]] < .5 and measured[ERRORS[1]] < .05 and measured[ERRORS[2]] < .2
        _assert(row.get("paper_success") is bool(paper), "paper_success disagrees with simultaneous final thresholds")
        diagnostics = evaluate_goal_diagnostics(arrays["states"], row, row.get("timing", TIMING), valid=True)
        _assert(diagnostics["diagnostic_valid"], "Cannot recompute lift/strict diagnostic")
        for key in ("ood_goal_success", "strict_goal_success"):
            _assert(row.get(key) is diagnostics[key], "Success disagrees with saved lift/final trajectory: " + key)
    else:
        _assert(all(row.get(key) is False for key in ("paper_success", "ood_goal_success", "strict_goal_success")),
                "Invalid trial marked successful")
        _assert(all(row.get(key) is None for key in ERRORS), "Invalid trial exposes finite final-error metrics")
    # Keep only the common prefix in memory; full post-intervention paths differ.
    return {"states": arrays["states"][:1261], "final_states": arrays["final_states"][0], "goal": goal,
            **{key: arrays[key][:1260] for key in ("actions_r", "actions_l", "ppo_actions_r", "ppo_actions_l")}}


def _artifact_audit(output, plan):
    issues, checked, warnings = [], {}, []
    for method in METHODS[1:]:
        try:
            predictor = plan["predictors"][method]
            path = Path(predictor["path"]).resolve()
            _assert(_sha256(path) == predictor["sha256"], "Predictor SHA256 mismatch: " + method)
            report_path = path.parent / "report.json"
            _assert(_sha256(report_path) == predictor["training_report_sha256"], "Training report SHA256 mismatch: " + method)
            training = json.loads(report_path.read_text())
            _assert(training["artifact_sha256"]["best.pt"] == predictor["sha256"], "Predictor differs from training artifact")
            checked[str(path)] = predictor["sha256"]
            checked[str(report_path)] = predictor["training_report_sha256"]
        except (KeyError, ValueError, OSError, TypeError) as exc:
            issues.append(str(exc))
    provenance_path = output / "provenance.json"
    if provenance_path.exists():
        try:
            source = json.loads(provenance_path.read_text())
            _assert(source["plan_sha256"] == _sha256(output / "plan.json"), "Provenance plan SHA256 mismatch")
            for key, base in (("source_sha256", ROOT), ("collector_source_sha256", ROOT), ("evaluation_sources", HERE)):
                for name, expected in source.get(key, {}).items():
                    path = (base / name).resolve()
                    _assert(base.resolve() in path.parents, "Source hash path is outside repository")
                    _assert(_sha256(path) == expected, "Source SHA256 mismatch: " + name)
                    checked[str(path)] = expected
            checked[str(provenance_path)] = _sha256(provenance_path)
        except (KeyError, ValueError, OSError, TypeError) as exc:
            issues.append(str(exc))
    else:
        warnings.append("provenance.json absent; runtime source provenance was not checked")
    return {"passed": not issues, "verified_sha256": checked, "issues": issues, "warnings": warnings}


def audit_threeway(output, plan, manifest):
    """Use one slot per planned episode/method; never inflate/drop denominators."""
    issues, planned, indexed, prefixes, rows, checks = [], {}, {}, {}, [], []
    if plan.get("methods") != list(METHODS):
        issues.append("Plan methods must be baseline/guided_absolute/guided_delta")
    if manifest.get("complete") is not True:
        issues.append("results.json is not marked complete")
    if manifest.get("plan_sha256") != _sha256(output / "plan.json"):
        issues.append("results.json plan SHA256 mismatch")
    settings = plan.get("guidance", {})
    expected_settings = {"horizon_steps": 20, "reselect_every_steps": 20, "phase_start_step": 1260,
                         "candidates": CANDIDATES, "angle_scale_rad": .5, "position_scale_m": .05,
                         "orientation_scale_rad": .2, "residual_squared_weight": .01}
    if any(settings.get(key) != value for key, value in expected_settings.items()):
        issues.append("Guidance differs from the fixed seven-candidate/20-step score settings")
    for spec in plan.get("episodes", []):
        key = (spec.get("profile"), spec.get("episode_id"))
        if key[0] not in ("normal", "ood_articulation") or not isinstance(key[1], str) or not key[1]:
            issues.append("Unsupported planned episode: " + str(key))
            continue
        if key in planned:
            issues.append("Duplicate episode in plan: " + str(key))
        planned[key] = spec
    if len({key[1] for key in planned}) != len(planned):
        issues.append("Episode IDs are reused across profiles")
    if not planned:
        issues.append("Plan contains no supported episodes")
    for row in manifest.get("episodes", []):
        key = (row.get("profile"), row.get("episode_id"), row.get("method"))
        indexed.setdefault(key, []).append(row)
    expected = {(profile, episode, method) for profile, episode in planned for method in METHODS}
    for key in sorted(set(indexed) - expected, key=str):
        issues.append("Unplanned result: " + str(key))
    path_keys = {}
    for key in sorted(expected):
        spec = planned[key[:2]]
        matches = indexed.get(key, [])
        if len(matches) != 1:
            message = ("Missing result: " if not matches else "Duplicate result: ") + str(key)
            issues.append(message)
            rows.append(_invalid_slot(spec, key[2], message))
            continue
        row = copy.deepcopy(matches[0])
        try:
            if row["profile"] == "normal":
                _assert(.5 - 1e-6 <= row["target_articulation_rad"] <= .6 + 1e-6, "normal target outside [0.5,0.6]")
            else:
                _assert(_group(row, "ood_small") or _group(row, "ood_large"), "OOD target outside configured angle bands")
            path = str((output / row["trajectory_path"]).resolve())
            _assert(path not in path_keys, "Trajectory path reused by multiple results")
            path_keys[path] = key
            prefixes[key] = _load_checked(output, row, spec)
            row["measurement_unverified"] = False
        except (KeyError, ValueError, OSError, TypeError, IndexError) as exc:
            message = "{}: {}".format(key, exc)
            issues.append(message)
            row = _invalid_slot(spec, key[2], message)
        rows.append(row)
    for profile, episode in sorted(planned):
        for method in METHODS[1:]:
            a = prefixes.get((profile, episode, "baseline"))
            b = prefixes.get((profile, episode, method))
            record = {"profile": profile, "episode_id": episode, "method": method,
                      "arrays_equal": {}, "issues": []}
            if a is None or b is None:
                record["issues"].append("Verified trajectory unavailable for paired prefix")
            else:
                for key in a:
                    equal = a[key].dtype == b[key].dtype and np.array_equal(a[key], b[key], equal_nan=True)
                    record["arrays_equal"][key] = bool(equal)
                    if not equal:
                        record["issues"].append("Shared prefix differs: " + key)
            record["passed"] = not record["issues"]
            checks.append(record)
            issues.extend("{} / {} / {}: {}".format(profile, episode, method, item) for item in record["issues"])
    source = _artifact_audit(output, plan)
    issues.extend(source["issues"])
    return {"passed": not issues, "planned_episodes": len(planned), "planned_trials": len(expected),
            "observed_rows": len(manifest.get("episodes", [])), "planned_pairs": 2 * len(planned),
            "checked_pairs": len(checks), "matching_pairs": sum(row["passed"] for row in checks),
            "unverified_slots": sum(row["measurement_unverified"] for row in rows),
            "checks": checks, "issues": issues, "artifact_audit": source}, rows


def _performance(output, plan, rows):
    result = {"groups": {}, "warnings": [], "batches": []}
    indexed = {(row["profile"], row["episode_id"], row["method"]): row for row in rows}
    seen = set()
    for path in sorted((output / "batches").glob("*.json")):
        try:
            batch = json.loads(path.read_text())
            entries = batch["episodes"]
            keys = {(row["profile"], row["episode_id"], row["method"]) for row in entries}
            _assert(batch.get("complete") is True and batch["plan_sha256"] == _sha256(output / "plan.json"), "Batch incomplete or plan hash mismatch")
            _assert(bool(keys) and len(keys) == len(entries) and keys <= set(indexed) and not seen.intersection(keys), "Batch result IDs overlap or disagree")
            profiles, methods = {key[0] for key in keys}, {key[2] for key in keys}
            _assert(len(profiles) == len(methods) == 1, "Batch mixes profiles or methods")
            perf = batch["performance"]
            _assert(perf["environments"] == len(keys), "Batch environment count mismatch")
            for key in ("wall_seconds", "rollout_seconds", "guidance_seconds"):
                _assert(_finite(perf[key]) and perf[key] >= 0, "Invalid batch timing")
            for row in entries:
                original = indexed[(row["profile"], row["episode_id"], row["method"])]
                _assert(row["trajectory_sha256"] == original.get("trajectory_sha256"), "Batch trajectory hash differs from results")
            profile, method = next(iter(profiles)), next(iter(methods))
            values = {key: float(perf[key]) for key in ("wall_seconds", "rollout_seconds", "guidance_seconds")}
            values.update(episodes=len(keys), profile=profile, method=method, path=str(path.relative_to(output)), sha256=_sha256(path))
            result["batches"].append(values)
            seen.update(keys)
        except (KeyError, ValueError, OSError, TypeError) as exc:
            result["warnings"].append("{}: {}".format(path.name, exc))
    for profile in ("normal", "ood_articulation"):
        result["groups"][profile] = {}
        for method in METHODS:
            subset = [row for row in result["batches"] if row["profile"] == profile and row["method"] == method]
            if subset:
                totals = {key: sum(row[key] for row in subset) for key in ("episodes", "wall_seconds", "rollout_seconds", "guidance_seconds")}
                totals["batches"] = len(subset)
                totals["guidance_seconds_per_episode_amortized"] = totals["guidance_seconds"] / totals["episodes"]
                result["groups"][profile][method] = totals
    result["missing_timing_trials"] = len(set(indexed) - seen)
    return result


def _markdown(summary):
    smoke = summary["purpose"] == "technical_smoke"
    audit = summary["pairing_audit"]
    lines = ["# PPO・元 absolute・delta の3方式比較", "",
             "公式 train 参照を使った技術確認です。OOD 改善に関する結論は出しません。" if smoke else
             "同じ参照・目標を用いた、探索的な単一 seed の追試です。3方式ともこの計画で新規実行した結果を集計し、以前の PPO・absolute の結果は流用しません。",
             "評価 seed: `{}`。参照 split: `{}`。{} 目標 × 3 方式 = {} 試行。".format(summary["evaluation_seed"], summary["split"], audit["planned_episodes"], audit["planned_trials"]),
             "公開 PPO は固定し、最終開閉の右手首並進だけに補正します。元 absolute は終点の絶対状態を予測し、delta は角度・基部位置の変化を予測して絶対状態に戻してから採点します。", "",
             "ゼロと各軸 ±1 action の7候補、20 step（0.2秒）ごとの選択、step 1,260〜1,559 の適用を3方式で固定しました。PPO 単体の補正は常にゼロです。",
             "score = (角度誤差/0.5 rad)² + (位置誤差/0.05 m)² + (SO(3)姿勢誤差/0.2 rad)² + 0.01‖補正‖²。設定は plan.json に保存し、この比較結果では調整しません。",
             "normal は目標角 0.5〜0.6 rad、OOD は [0.2,0.4] ∪ [1.6,3.1] rad。OOD の小角側・大角側は区間長に比例した抽出であり、同数ではありません。", "",
             "## 対応条件と分母", "",
             "PPO と各補正方式の {} 対中 {} 対で、trajectory hash、states[:1261]、左右の実行 action と PPO action の [:1260]、phase reset 後の final_states[0]、goal を照合しました。".format(audit["planned_pairs"], audit["matching_pairs"])]
    if not summary["comparison_eligible"]:
        lines += ["**監査に失敗したため、性能差に関する結論を保留します。対応差は表示しません。以下は参考集計です。**"]
        lines += ["- " + issue for issue in audit["issues"]]
    else:
        lines.append("共通の補正前軌跡と目標が一致しています。")
    lines += ["計画した全試行を成功率の分母に含め、無効試行は失敗として数えます。欠損・重複・破損などで検証できない枠も除外せず、未検証無効として計数します。",
              "無効数 {}、うち未検証 {}。最終誤差の平均・中央値は有効測定のみを使い、測定数 n を別記します。".format(sum(summary["groups"][g][m]["invalid"] for g in ("normal", "ood_articulation") for m in METHODS), audit["unverified_slots"]), "",
              "## 成功率", "",
              "paper_success は最終 PE < 0.05 m・AAE < 0.5 rad・AE < 0.2 rad の同時達成です。ood_goal_success はさらに途中の lift 達成を要求します。strict_goal_success は lift 込み条件で角度閾値を 0.1 rad にした別指標です。角度単独 < 0.1 rad は基部安定や lift を要求しません。", "",
              "| 条件 | 方式 | 無効 / 全数 | paper | lift 込み | strict（lift 込み） | 角度単独 < 0.1 rad |",
              "|---|---|---:|---|---|---|---|"]
    for group in GROUPS:
        for method in METHODS:
            row = summary["groups"][group][method]
            cells = [_success_cell(row["success"][key]) for key in ("paper_success", "ood_goal_success", "strict_goal_success", "angle_lt_0p1")]
            lines.append("| {} | {} | {} / {} | {} |".format(GROUP_LABELS[group], METHOD_LABELS[method], row["invalid"], row["n"], " | ".join(cells)))
    lines += ["", "## 最終誤差", "", "PE は最終物体基部位置のユークリッド距離、AAE は関節角の絶対誤差、AE は quaternion の SO(3) 回転角距離です。全て小さいほど良く、表は平均 / 中央値（測定 n / 全試行数）です。", "",
              "| 条件 | 方式 | PE [m] | AAE [rad] | AE [rad] |", "|---|---|---|---|---|"]
    for group in GROUPS:
        for method in METHODS:
            row = summary["groups"][group][method]
            cells = ["{} / {} (n={}/{})".format(_fmt(row["errors"][key]["mean"]), _fmt(row["errors"][key]["median"]), row["errors"][key]["n"], row["n"]) for key in (ERRORS[1], ERRORS[0], ERRORS[2])]
            lines.append("| {} | {} | {} |".format(GROUP_LABELS[group], METHOD_LABELS[method], " | ".join(cells)))
    if summary["comparison_eligible"]:
        lines += ["", "## PPO に対する対応差", "", "gain は PPO 失敗→補正成功、loss は PPO 成功→補正失敗です。角度差は補正−PPO、負値ほど補正側の誤差が小さいことを表します。", "",
                  "| 条件 | 方式 | 対数 | gain | loss | paper 差 [pt] | AAE差平均 [rad] | bootstrap 95% CI |",
                  "|---|---|---:|---:|---:|---:|---:|---|"]
        for group in GROUPS:
            for method in METHODS[1:]:
                pair = summary["groups"][group]["paired_vs_baseline"][method]
                success, error = pair["paper_success"], pair["angle_difference_guided_minus_baseline_rad"]
                difference = success["rate_difference_guided_minus_baseline"]
                ci = error["bootstrap_95_percent_ci"]
                lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(GROUP_LABELS[group], METHOD_LABELS[method], pair["n"], success["gains"], success["losses"], _fmt(100 * difference if difference is not None else None, 1), _fmt(error["mean"]), "[{}, {}]".format(*map(_fmt, ci)) if ci else "—"))
    lines += ["", "## 補助情報と解釈", "",
              "小角 OOD は初期状態ですでに角度条件を満たす場合があります。初期・補正直前の角度条件達成数、補正開始時状態、候補選択頻度は summary.json に保存しています。",
              "対応 bootstrap は試行の組を10,000回復元抽出した記述的区間です。参照元の共有による相関や seed 間変動は扱わず、多重比較の補正もしていません。少数の小角側の結果は特に不安定です。",
              "両予測モデルは同じ追加収集データに由来します。PPO 単体と追加学習・推論コストを揃えた比較ではありません。予測誤差の改善からタスク成功の改善を推測しません。",
              "実行時間は summary.json の performance に記録しています。worker 時間は初期化・保存込み、guidance 時間は候補の特徴量作成・転送・推論・採点込みです。1試行当たりはバッチの按分であり単独遅延ではありません。",
              "方式間差の一般化には複数 seed の確認が必要です。技術 smoke は性能結論に使用しません。", ""]
    return "\n".join(lines)


def _csv(summary):
    fields = ["group", "method", "n", "invalid", "unverified", "comparison_eligible"]
    rates = ("paper_success", "ood_goal_success", "strict_goal_success", "angle_lt_0p1", "base_stable")
    fields += [key + suffix for key in rates for suffix in ("_count", "_rate")]
    for metric in ("PE_m", "AAE_rad", "AE_rad"):
        fields += [metric + suffix for suffix in ("_mean", "_median", "_n", "_excluded")]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for group in GROUPS:
        for method in METHODS:
            source = summary["groups"][group][method]
            row = {"group": group, "method": method, "n": source["n"], "invalid": source["invalid"],
                   "unverified": source["unverified"], "comparison_eligible": summary["comparison_eligible"]}
            for key in rates:
                row.update({key + "_count": source["success"][key]["count"], key + "_rate": source["success"][key]["rate"]})
            for metric, key in zip(("PE_m", "AAE_rad", "AE_rad"), (ERRORS[1], ERRORS[0], ERRORS[2])):
                for suffix, stat in (("_mean", "mean"), ("_median", "median"), ("_n", "n"), ("_excluded", "excluded_or_missing")):
                    row[metric + suffix] = source["errors"][key][stat]
            writer.writerow(row)
    return stream.getvalue()


def write_report(output):
    output = Path(output).resolve()
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "results.json").read_text(encoding="utf-8"))
    audit, rows = audit_threeway(output, plan, manifest)
    seed = plan.get("seed", plan.get("config", {}).get("seed", 202))
    summary = {"schema_version": 1, "methods": list(METHODS), "evaluation_seed": seed,
               "split": plan.get("split"), "purpose": plan.get("purpose", "unspecified"),
               "predictors": plan.get("predictors", {}), "guidance": plan.get("guidance", {}),
               "comparison_eligible": audit["passed"],
               "comparison_conclusion_allowed": audit["passed"] and plan.get("purpose") != "technical_smoke",
               "pairing_audit": audit, "performance": _performance(output, plan, rows), "groups": {},
               "metric_definitions": {"PE": "final base Euclidean position error [m]", "AAE": "final absolute hinge angle error [rad]", "AE": "final SO(3) quaternion geodesic error [rad]"},
               "success_thresholds": {"PE_m": .05, "AAE_rad": .5, "AE_rad": .2, "strict_AAE_rad": .1, "comparison": "strictly less than"},
               "source_sha256": {name: _sha256(output / name) for name in ("plan.json", "results.json")},
               "report_source_sha256": {"report_threeway.py": _sha256(__file__), "evaluate_report.py": _sha256(HERE / "evaluate_report.py")},
               "interpretation": "Exploratory single-seed three-way rerun; no claims beyond the saved normal/OOD trials."}
    index = {(row["profile"], row["episode_id"], row["method"]): row for row in rows}
    for group in GROUPS:
        values = {}
        for method in METHODS:
            selected = [row for row in rows if row["method"] == method and _group(row, group)]
            values[method] = summarize_rows(selected)
            values[method]["unverified"] = sum(row["measurement_unverified"] for row in selected)
        values["paired_vs_baseline"] = None
        values["delta_vs_absolute"] = None
        if audit["passed"]:
            baseline = [row for row in rows if row["method"] == "baseline" and _group(row, group)]
            values["paired_vs_baseline"] = {method: paired_summary(
                [(row, index[(row["profile"], row["episode_id"], method)]) for row in baseline], seed=seed)
                for method in METHODS[1:]}
            values["delta_vs_absolute"] = paired_summary(
                [(index[(row["profile"], row["episode_id"], "guided_absolute")], index[(row["profile"], row["episode_id"], "guided_delta")]) for row in baseline], seed=seed)
        summary["groups"][group] = values
    for name, text in (("summary.json", json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n"),
                       ("THREEWAY_REVIEW.md", _markdown(summary)), ("metricstable.csv", _csv(summary))):
        path = output / name
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = write_report(args.output)
    print("Three-way report saved; comparison eligible: {}".format(result["comparison_eligible"]))
