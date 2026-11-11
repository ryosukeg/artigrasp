#!/usr/bin/env python3
"""Audit and summarize paired fixed-PPO / learned-response guidance rollouts.

This module only reads saved trajectories. It never imports the simulator or
reruns an episode, and it withholds comparisons when the shared prefix differs.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


METHODS = ("baseline", "guided")
GROUPS = ("normal", "ood_articulation", "ood_small", "ood_large")
ERRORS = ("paper_articulation_error_rad", "paper_position_error_m",
          "paper_orientation_error_rad")
CANDIDATES = [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0],
              [0, -1, 0], [0, 0, 1], [0, 0, -1]]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value):
    try:
        return value is not None and bool(np.isfinite(float(value)))
    except (ValueError, TypeError):
        return False


def _valid(row):
    return row.get("valid") is True


def _error(row, name):
    value = row.get(name)
    return float(value) if _valid(row) and _finite(value) else None


def _success(row, name):
    if not _valid(row):
        return False
    angle, position, orientation = (_error(row, key) for key in ERRORS)
    if name == "angle_lt_0p5":
        return angle is not None and angle < .5
    if name == "angle_lt_0p1":
        return angle is not None and angle < .1
    if name == "base_stable":
        return (position is not None and orientation is not None and
                position < .05 and orientation < .2)
    return row.get(name) is True


def _rate(values):
    values = list(values)
    count = sum(bool(value) for value in values)
    return {"count": count, "n": len(values),
            "rate": count / len(values) if values else None}


def _distribution(values, total):
    values = [float(value) for value in values if _finite(value)]
    return {"n": len(values), "excluded_or_missing": total - len(values),
            "mean": float(np.mean(values)) if values else None,
            "median": float(np.median(values)) if values else None}


def _group(row, group):
    if group in ("normal", "ood_articulation"):
        return row.get("profile") == group
    if row.get("profile") != "ood_articulation":
        return False
    angle = row.get("target_articulation_rad")
    if not _finite(angle):
        return False
    low, high = (.2, .4) if group == "ood_small" else (1.6, 3.1)
    return low - 1e-6 <= float(angle) <= high + 1e-6


def summarize_rows(rows):
    """Use all trials in success denominators; exclude invalid errors explicitly."""
    result = {"n": len(rows), "invalid": sum(not _valid(row) for row in rows)}
    result["success"] = {
        key: _rate(_success(row, key) for row in rows)
        for key in ("paper_success", "angle_lt_0p5", "angle_lt_0p1",
                    "base_stable", "ood_goal_success", "strict_goal_success")}
    result["errors"] = {key: _distribution((_error(row, key) for row in rows), len(rows))
                        for key in ERRORS}
    for key in ERRORS:
        result["errors"][key]["missing_or_nonfinite_raw"] = sum(
            not _finite(row.get(key)) for row in rows)
    entry = [row.get("opening_entry", {}) for row in rows]
    result["opening_entry"] = {
        key: _distribution((value.get(key) for value in entry), len(entry))
        for key in ("position_error_m", "orientation_error_rad", "articulation_rad")}
    result["opening_entry"]["prior_lift_success"] = _rate(
        value.get("prior_lift_success") is True for value in entry)
    result["opening_entry"]["prior_lift_validity_scope"] = (
        "pre_guidance_prefix" if entry and all("prior_rollout_valid" in value for value in entry)
        else "entire_episode_legacy")
    tolerance_counts = {}
    for label, key in (("initial", "initial_articulation_error_rad"),
                       ("opening_before_phase_reset", "articulation_start_error_rad"),
                       ("opening_entry", None)):
        errors = []
        for row in rows:
            if key is not None:
                value = row.get(key)
            else:
                angle = row.get("opening_entry", {}).get("articulation_rad")
                target = row.get("target_articulation_rad")
                value = abs(float(angle) - float(target)) if _finite(angle) and _finite(target) else None
            errors.append(float(value) if _finite(value) else None)
        tolerance_counts[label] = {
            "measured_n": sum(value is not None for value in errors),
            "missing": sum(value is None for value in errors),
            "angle_lt_0p5": _rate(value is not None and value < .5 for value in errors),
            "angle_lt_0p1": _rate(value is not None and value < .1 for value in errors)}
    result["angle_tolerance_before_final"] = tolerance_counts
    counts = np.zeros(7, dtype=np.int64)
    missing = 0
    for row in rows:
        value = row.get("guidance_candidate_counts")
        if (not isinstance(value, list) or len(value) != 7 or
                any(type(count) is not int or count < 0 for count in value)):
            missing += 1
        else:
            counts += np.asarray(value, dtype=np.int64)
    result["guidance_candidates"] = {
        "order_action_units": CANDIDATES, "counts": counts.tolist(),
        "total_decisions": int(counts.sum()), "missing_episodes": missing,
        "frequency": (counts / counts.sum()).tolist() if counts.sum() else [None] * 7}
    return result


def paired_summary(pairs, seed=202, resamples=10000):
    """Bootstrap complete paired errors, with a fixed seed for reproducibility."""
    gains = losses = both = neither = 0
    differences = []
    for baseline, guided in pairs:
        a, b = _success(baseline, "paper_success"), _success(guided, "paper_success")
        gains += int(not a and b)
        losses += int(a and not b)
        both += int(a and b)
        neither += int(not a and not b)
        angle_a, angle_b = (_error(row, ERRORS[0]) for row in (baseline, guided))
        if angle_a is not None and angle_b is not None:
            differences.append(angle_b - angle_a)
    n = len(differences)
    interval = None
    if n >= 2:
        values = np.asarray(differences, dtype=float)
        rng = np.random.default_rng(seed)
        bootstrap_means = values[rng.integers(0, n, size=(resamples, n))].mean(axis=1)
        interval = np.quantile(bootstrap_means, [.025, .975]).tolist()
    return {"n": len(pairs), "paper_success": {
                "gains": gains, "losses": losses, "both_success": both,
                "both_failure": neither,
                "rate_difference_guided_minus_baseline": (gains - losses) / len(pairs) if pairs else None},
            "angle_difference_guided_minus_baseline_rad": {
                "n": n, "excluded_or_missing_pairs": len(pairs) - n,
                "mean": float(np.mean(differences)) if differences else None,
                "median": float(np.median(differences)) if differences else None,
                "bootstrap_95_percent_ci": interval,
                "bootstrap_resamples": resamples, "bootstrap_seed": seed,
                "small_sample_warning": n < 20,
                "note": "Episode-pair percentile bootstrap; shared source references and single-seed uncertainty are not modeled."}}


def _load_prefix(output, row):
    path = (output / row["trajectory_path"]).resolve()
    if output.resolve() not in path.parents:
        raise ValueError("Trajectory is outside the evaluation directory")
    if _sha256(path) != row["trajectory_sha256"]:
        raise ValueError("Trajectory SHA256 mismatch")
    required = {"states": (1561, 138), "actions_r": (1560, 51),
                "actions_l": (1560, 51), "final_states": (301, 138), "goal": (8,)}
    prefix = {}
    with np.load(path, allow_pickle=False) as data:
        for key, shape in required.items():
            value = data[key]
            if value.shape != shape:
                raise ValueError("{} shape {}, expected {}".format(key, value.shape, shape))
            if key == "states":
                value = value[:1261]
            elif key in ("actions_r", "actions_l"):
                value = value[:1260]
            elif key == "final_states":
                value = value[0]
            prefix[key] = value
    return prefix


def audit_pairs(output, plan, manifest):
    """Record every mismatch, so the report cannot silently claim a paired effect."""
    issues, indexed = [], {}
    if manifest.get("complete") is not True:
        issues.append("results.json is not marked complete")
    if manifest.get("plan_sha256") != _sha256(output / "plan.json"):
        issues.append("results.json plan SHA256 mismatch")
    for row in manifest.get("episodes", []):
        key = (row.get("profile"), row.get("episode_id"), row.get("method"))
        if key[0] not in ("normal", "ood_articulation") or key[2] not in METHODS:
            issues.append("Unsupported profile/method: {}".format(key))
        if key in indexed:
            issues.append("Duplicate result: {}".format(key))
        indexed[key] = row
    planned = {(row["profile"], row["episode_id"]) for row in plan.get("episodes", [])}
    if len(planned) != len(plan.get("episodes", [])):
        issues.append("Duplicate episode in plan")
    if not planned:
        issues.append("Plan contains no episodes")
    expected = {(profile, episode, method) for profile, episode in planned for method in METHODS}
    for key in sorted(expected - set(indexed)):
        issues.append("Missing result: {}".format(key))
    for key in sorted(set(indexed) - expected, key=str):
        issues.append("Unplanned result: {}".format(key))
    pairs, checks = [], []
    for profile, episode in sorted(planned):
        if any((profile, episode, method) not in indexed for method in METHODS):
            continue
        baseline, guided = (indexed[(profile, episode, method)] for method in METHODS)
        pairs.append((baseline, guided))
        record = {"profile": profile, "episode_id": episode, "passed": True,
                  "arrays_equal": {}, "issues": []}
        for field in ("sample_index", "target_articulation_rad"):
            if baseline.get(field) != guided.get(field):
                record["issues"].append("Metadata differs: " + field)
        for row in (baseline, guided):
            if not _finite(row.get("target_articulation_rad")):
                record["issues"].append("Missing finite target_articulation_rad")
            elif profile == "ood_articulation" and not (_group(row, "ood_small") or _group(row, "ood_large")):
                record["issues"].append("OOD goal is outside the two configured angle bands")
        try:
            a, b = (_load_prefix(output, row) for row in (baseline, guided))
            for key in a:
                equal = a[key].dtype == b[key].dtype and np.array_equal(a[key], b[key], equal_nan=True)
                record["arrays_equal"][key] = bool(equal)
                if not equal:
                    record["issues"].append("Shared prefix differs: " + key)
            for row, arrays in ((baseline, a), (guided, b)):
                if (_finite(row.get("target_articulation_rad")) and
                        not np.isclose(arrays["goal"][7], row["target_articulation_rad"], atol=1e-6, rtol=0)):
                    record["issues"].append("Saved goal disagrees with target_articulation_rad")
        except (KeyError, ValueError, OSError) as error:
            record["issues"].append(str(error))
        record["passed"] = not record["issues"]
        checks.append(record)
        issues.extend("{} / {}: {}".format(profile, episode, issue) for issue in record["issues"])
    return {"passed": not issues, "planned_pairs": len(planned), "checked_pairs": len(checks),
            "matching_pairs": sum(row["passed"] for row in checks),
            "checks": checks, "issues": issues}, pairs


def _fmt(value, digits=4):
    return "—" if value is None else ("{: ." + str(digits) + "f}").format(value).strip()


def _success_cell(value):
    return "{}/{} ({:.1f}%)".format(value["count"], value["n"], 100 * value["rate"]) if value["n"] else "0/0 (—)"


def summarize_performance(output, plan, rows):
    """Read committed batch timings; amortized time is not isolated latency."""
    settings = plan.get("guidance", {})
    decisions = len(range(settings.get("phase_start_step", 1260), 1560,
                          settings.get("reselect_every_steps", 20)))
    result = {"groups": {}, "batches": [], "warnings": [],
              "selections_per_episode": decisions,
              "definitions": {
                  "wall_seconds": "Worker elapsed time including initialization, rollout and artifact saving; subprocess startup and report generation excluded.",
                  "rollout_seconds": "Elapsed time for the complete 1560-step batched rollout, including guidance when enabled.",
                  "guidance_seconds": "Elapsed selector time summed across selection calls; includes feature preparation, model execution, CPU/GPU transfers and scoring for all environments in each batch.",
                  "guidance_seconds_per_selection_batch": "Total guidance time / number of batched selection calls, each evaluating seven candidates for all environments.",
                  "guidance_seconds_per_episode_all_selections_amortized": "Total guidance time / episodes, covering 15 selections per episode; an amortized allocation, not single-environment latency.",
                  "comparison_caution": "Baseline/guided wall-time differences also reflect system load and are not a causal measurement of pure guidance overhead."}}
    expected = {(row.get("profile"), row.get("episode_id"), row.get("method")) for row in rows}
    seen = set()
    for path in sorted((output / "batches").glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            episodes = value.get("episodes", [])
            profiles = {row.get("profile") for row in episodes}
            methods = {row.get("method") for row in episodes}
            keys = {(row.get("profile"), row.get("episode_id"), row.get("method")) for row in episodes}
            if (value.get("complete") is not True or not episodes or len(profiles) != 1 or
                    len(methods) != 1 or len(keys) != len(episodes) or
                    not keys.issubset(expected) or keys.intersection(seen) or
                    value.get("plan_sha256") != _sha256(output / "plan.json")):
                raise ValueError("Incomplete, mixed or mismatched batch metadata")
            profile, method = next(iter(profiles)), next(iter(methods))
            if profile not in ("normal", "ood_articulation") or method not in METHODS:
                raise ValueError("Unsupported profile or method")
            performance = value["performance"]
            if performance.get("environments") != len(episodes):
                raise ValueError("Environment count does not match batch episodes")
            times = {key: performance.get(key) for key in ("wall_seconds", "rollout_seconds", "guidance_seconds")}
            if any(not _finite(seconds) or float(seconds) < 0 for seconds in times.values()):
                raise ValueError("Missing or invalid elapsed time")
            record = {"path": str(path.relative_to(output)), "sha256": _sha256(path),
                      "profile": profile, "method": method, "episodes": len(episodes),
                      "selection_batch_calls": decisions if method == "guided" else 0}
            record.update({key: float(seconds) for key, seconds in times.items()})
            result["batches"].append(record)
            seen.update(keys)
        except (OSError, KeyError, ValueError, TypeError) as error:
            result["warnings"].append("{}: {}".format(path.name, error))
    for profile in ("normal", "ood_articulation", "all"):
        for method in METHODS:
            batches = [row for row in result["batches"] if row["method"] == method and
                       (profile == "all" or row["profile"] == profile)]
            if not batches:
                continue
            aggregate = {"batch_count": len(batches), "episodes": sum(row["episodes"] for row in batches),
                         "selection_batch_calls": sum(row["selection_batch_calls"] for row in batches)}
            for key in ("wall_seconds", "rollout_seconds", "guidance_seconds"):
                aggregate[key] = sum(row[key] for row in batches)
            calls = aggregate["selection_batch_calls"]
            aggregate["guidance_seconds_per_selection_batch"] = aggregate["guidance_seconds"] / calls if calls else None
            aggregate["guidance_seconds_per_episode_all_selections_amortized"] = (
                aggregate["guidance_seconds"] / aggregate["episodes"] if calls else None)
            result["groups"].setdefault(profile, {})[method] = aggregate
    result["timed_episodes"] = len(seen)
    result["episodes_without_batch_timing"] = len(expected - seen)
    return result


def _markdown(summary):
    audit = summary["pairing_audit"]
    smoke = summary["purpose"] == "technical_smoke"
    training = summary["predictor"].get("training_episode_counts", {})
    budget = ("予測器には追加の sim {} 試行（train {} / val {}）を使用しています。".format(
        sum(training.values()), training.get("train", 0), training.get("val", 0))
        if training else "予測器の追加学習データの構成は、学習時のレポートを参照してください。")
    lines = ["# 学習した応答モデルによる補正と ArtiGrasp PPO の比較", "",
             ("これは公式 train 参照を使った少数の技術確認です。OOD 性能評価の結果として解釈しません。" if smoke else
              "これは予測精度・補正選択の有効性が未確認だった予測器による、探索的な単一 seed の比較です。"),
             "ArtiGrasp の公開 PPO を固定した対照と、同じ PPO に学習済み応答モデルで選んだ右手首並進補正を加える方式を比較します。",
             budget + "同じ学習予算による手法間比較ではありません。", "",
             "最終開閉の step 1,260〜1,559 で、ゼロと各軸 ±1 action 単位の 7 候補から選び、各補正を 20 step（0.2 秒）保持します。",
             "normal は目標角 0.5〜0.6 rad、OOD は [0.2, 0.4] ∪ [1.6, 3.1] rad。小角側と大角側は区間長に比例して抽出されます。",
             "評価 seed: `{}`。参照 split: `{}`。目的: `{}`。".format(
                 summary["evaluation_seed"], summary["split"], summary["purpose"]), "",
             "候補のスコアは `(予測角度誤差 / 0.5 rad)^2 + (予測位置誤差 / 0.05 m)^2 + (予測姿勢誤差 / 0.2 rad)^2 + 0.01 × ||補正||²` です。姿勢誤差は quaternion の回転角距離で計算します。",
             "最小スコアの候補を選び、ゼロ補正が最小値で同点なら優先し、全候補が非有限ならゼロに戻します。各尺度と重みはタスク閾値を基に test 評価前に固定し、この評価結果で調整していません。設定の原本は `plan.json` の `guidance` です。", "",
             "## 対応条件の検証", "",
             "全 {} 対のうち {} 対で、保存軌跡の hash、`states[:1261]`、左右の `actions[:1260]`、phase reset 後の `final_states[0]` と `goal` の完全一致を確認しました。".format(audit["planned_pairs"], audit["matching_pairs"])]
    if audit["passed"]:
        lines += ["補正開始までの物理状態・指令・目標が一致しており、この設定内での対応比較が可能です。"]
    else:
        lines += ["**検証に失敗したため、補正による改善・悪化の結論を保留します。以下の集計は参考値です。**", ""]
        lines.extend("- " + value for value in audit["issues"])
    lines += ["", "## 成功率", "",
              "全試行を分母とし、無効試行は失敗として数えます。paper は最終位置誤差 < 0.05 m・姿勢誤差 < 0.2 rad・角度誤差 < 0.5 rad の同時達成、lift 込みはこれに途中の lift 達成を加えた値です。strict は lift 込みの角度閾値を 0.1 rad にした指標です。角度のみの列は基部の安定を要求しません。", "",
              "| 条件 | 方式 | 無効 | paper | 角度 < 0.5 rad | 角度 < 0.1 rad | 基部安定 | lift 込み | strict |",
              "|---|---|---:|---|---|---|---|---|---|"]
    labels = {"normal": "通常", "ood_articulation": "OOD 全体", "ood_small": "OOD 小角側", "ood_large": "OOD 大角側"}
    method_labels = {"baseline": "PPO", "guided": "PPO + 補正"}
    for group in GROUPS:
        for method in METHODS:
            row = summary["groups"][group][method]
            values = [_success_cell(row["success"][key]) for key in
                      ("paper_success", "angle_lt_0p5", "angle_lt_0p1", "base_stable", "ood_goal_success", "strict_goal_success")]
            lines.append("| {} | {} | {} | {} |".format(labels[group], method_labels[method], row["invalid"], " | ".join(values)))
    lines += ["", "## 最終誤差", "",
              "値は平均 / 中央値です。無効試行と欠損・非有限値を除外し、使用数 n / 全試行数を併記します。失敗率の分母とは異なるため、誤差だけで性能を判断しません。", "",
              "| 条件 | 方式 | 角度 [rad] | 位置 [m] | 姿勢 [rad] |", "|---|---|---|---|---|"]
    for group in GROUPS:
        for method in METHODS:
            row = summary["groups"][group][method]
            cells = ["{} / {} (n={}/{})".format(_fmt(row["errors"][key]["mean"]),
                     _fmt(row["errors"][key]["median"]), row["errors"][key]["n"], row["n"]) for key in ERRORS]
            lines.append("| {} | {} | {} |".format(labels[group], method_labels[method], " | ".join(cells)))
    lines += ["", "## 対応する試行の差", "",
              "gain は PPO 失敗 → 補正成功、loss は PPO 成功 → 補正失敗です。角度差は「補正 − PPO」で、負値ほど補正側の誤差が小さいことを表します。", "",
              "| 条件 | 対数 | gain | loss | paper の差 [pt] | 角度差平均 [rad] | 対応 bootstrap 95% CI | 有効角度対数 |",
              "|---|---:|---:|---:|---:|---:|---|---:|"]
    for group in GROUPS:
        row = summary["groups"][group]["paired"]
        success, angle = row["paper_success"], row["angle_difference_guided_minus_baseline_rad"]
        interval = angle["bootstrap_95_percent_ci"]
        ci = "[{}, {}]".format(*(_fmt(value) for value in interval)) if interval else "—"
        difference = success["rate_difference_guided_minus_baseline"]
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {}{} |".format(
            labels[group], row["n"], success["gains"], success["losses"],
            _fmt(100 * difference if difference is not None else None, 1), _fmt(angle["mean"]), ci,
            angle["n"], "（少数）" if angle["small_sample_warning"] else ""))
    lines += ["", "CI は試行の組を 10,000 回復元抽出した percentile 区間です。小標本、とくに小角側の値は不安定です。共通の参照元による相関や seed 間の変動を扱っておらず、一般的な優劣や統計的確証を示すものではありません。", "",
              "## 補正の選択頻度", "", "| 条件 | 0 | +x | -x | +y | -y | +z | -z | 記録なし試行 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for group in GROUPS:
        choices = summary["groups"][group]["guided"]["guidance_candidates"]
        cells = ["{} ({})".format(count, "{:.1f}%".format(100 * frequency) if frequency is not None else "—")
                 for count, frequency in zip(choices["counts"], choices["frequency"])]
        lines.append("| {} | {} | {} |".format(labels[group], " | ".join(cells), choices["missing_episodes"]))
    lines += ["", "## 最終評価より前に角度条件を満たしていた試行", "",
              "各欄は角度誤差 < 0.5 rad / < 0.1 rad の件数です。初期状態と、補正開始 step 1,260 の phase reset 前・後を区別します。これは過去時点の測定であり、最終タスク成功とは異なります。欠損を到達と扱わず、測定数を併記します。", "",
              "| 条件 | 方式 | 初期 | 補正開始 reset 前 | 補正開始 reset 後 |", "|---|---|---|---|---|"]
    for group in GROUPS:
        for method in METHODS:
            row = summary["groups"][group][method]
            counts = row["angle_tolerance_before_final"]
            cells = ["{} / {} (n={}/{})".format(counts[key]["angle_lt_0p5"]["count"],
                     counts[key]["angle_lt_0p1"]["count"], counts[key]["measured_n"], row["n"])
                     for key in ("initial", "opening_before_phase_reset", "opening_entry")]
            lines.append("| {} | {} | {} |".format(labels[group], method_labels[method], " | ".join(cells)))
    scopes = {summary["groups"][group][method]["opening_entry"]["prior_lift_validity_scope"]
              for group in GROUPS for method in METHODS if summary["groups"][group][method]["n"]}
    note = ("前段の lift 指標は補正開始までの軌跡の有効性で判定し、その後の数値異常の影響を受けません。"
            if scopes == {"pre_guidance_prefix"} else
            "旧形式の保存済み lift 指標は試行全体が無効な場合も失敗となるため、false が前段での失敗だけを示すとは限りません。")
    lines += ["", "補正開始時点の基部位置・姿勢・角度と、前段の lift 成功率は `summary.json` の `opening_entry` に保存しています。" + note + "前段の失敗を最終開閉の補正だけで解消できるとは限りません。", "",
              "## 実行時間", "",
              "worker 全体は初期化・軌跡保存込み、rollout は全 1,560 step の実行時間、補正選択は予測モデルの特徴量作成・CPU↔GPU 転送・推論・スコア計算を含む実測 wall 時間です。rollout と worker 全体には補正選択時間が含まれます。", "",
              "| 条件 | 方式 | 試行数 / batch数 | worker 全体 [s] | rollout [s] | 補正選択合計 [s] | 選択1batch平均 [s] | 1試行15選択への按分 [s] |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    performance = summary["performance"]
    for group in ("normal", "ood_articulation"):
        for method in METHODS:
            value = performance["groups"].get(group, {}).get(method)
            if value is None:
                continue
            cells = [_fmt(value[key]) for key in ("wall_seconds", "rollout_seconds", "guidance_seconds",
                     "guidance_seconds_per_selection_batch", "guidance_seconds_per_episode_all_selections_amortized")]
            lines.append("| {} | {} | {} / {} | {} |".format(labels[group], method_labels[method],
                         value["episodes"], value["batch_count"], " | ".join(cells)))
    if not performance["batches"]:
        lines.append("| 計測ファイルなし | — | — | — | — | — | — | — |")
    lines += ["", "1回の選択 batch で、並列の全試行について各7候補を評価します。100並列なら1回で100試行分を処理し、これを各試行15回行います。「選択1batch平均」は補正選択合計を15回×batch数で割った値です。「1試行15選択への按分」は合計を試行数で割った値で、1試行だけを単独実行した遅延ではありません。",
              "方式間の時間差にはシステム負荷や実行順の差も含まれるため、純粋な補正機構の追加コストを因果的に測った値ではありません。小角側・大角側は同じ batch で実行するため、個別の実測時間へ分解していません。"]
    if performance["episodes_without_batch_timing"]:
        lines.append("batch 時間のない試行: {} 件。".format(performance["episodes_without_batch_timing"]))
    if performance["warnings"]:
        lines += [""] + ["- " + warning for warning in performance["warnings"]]
    lines += ["", "## 解釈", ""]
    if smoke:
        lines.append("この実行は技術確認のため、OOD 改善に関する結論は出しません。")
    elif audit["passed"]:
        row = summary["groups"]["ood_articulation"]
        lines.append("この seed の OOD paper 成功率は PPO {}、補正 {} でした。".format(
            _success_cell(row["baseline"]["success"]["paper_success"]),
            _success_cell(row["guided"]["success"]["paper_success"])))
        lines.append("これは、この学習済みモデル・補正候補・評価条件で観測した結果です。追加の sim データ・学習・推論コストを含むため、ArtiGrasp と同じ予算での優位性を意味しません。")
    else:
        lines.append("対応条件または成果物の整合性を確認できていないため、性能差を補正の効果と解釈できません。")
    val = summary["predictor"].get("val_model", {}).get("angle_rad_rmse")
    persistence = summary["predictor"].get("val_persistence", {}).get("angle_rad_rmse")
    if val is not None and persistence is not None:
        lines.append("予測器の検証時角度 RMSE は {} rad、状態維持予測は {} rad でした。今回の評価で直接確認するのはタスク成功率であり、予測誤差の改善を前提にしません。".format(_fmt(val), _fmt(persistence)))
    lines += ["再現情報と全検証結果は `plan.json`、`results.json`、`summary.json`、各 `trajectories/*.npz` を参照してください。", ""]
    return "\n".join(lines)


def write_report(output):
    output = Path(output).resolve()
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "results.json").read_text(encoding="utf-8"))
    audit, pairs = audit_pairs(output, plan, manifest)
    rows = manifest.get("episodes", [])
    seed = plan.get("seed", plan.get("config", {}).get("seed"))
    summary = {"schema_version": 1, "evaluation_seed": seed,
               "split": plan.get("split", plan.get("config", {}).get("split")),
               "purpose": plan.get("purpose", "unspecified"),
               "predictor": plan.get("predictor", {}), "guidance": plan.get("guidance", {}),
               "performance": summarize_performance(output, plan, rows),
               "comparison_conclusion_allowed": audit["passed"] and plan.get("purpose") != "technical_smoke",
               "pairing_audit": audit, "groups": {},
               "source_sha256": {name: _sha256(output / name) for name in ("plan.json", "results.json")},
               "report_source_sha256": _sha256(__file__),
               "interpretation": "Exploratory single-seed paired comparison using an unvalidated guidance predictor; additional simulation data and compute are not matched to the baseline."}
    for group in GROUPS:
        summary["groups"][group] = {
            method: summarize_rows([row for row in rows if row.get("method") == method and _group(row, group)])
            for method in METHODS}
        summary["groups"][group]["paired"] = paired_summary([pair for pair in pairs if _group(pair[0], group)])
    for name, content in (("summary.json", json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n"),
                          ("COMPARISON_REVIEW.md", _markdown(summary))):
        path = output / name
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = write_report(args.output)
    print("Report saved: {} (paired audit: {})".format(
        args.output / "COMPARISON_REVIEW.md", "passed" if result["pairing_audit"]["passed"] else "FAILED"))
