"""Regenerate auditable Markdown and JSON summaries from saved OOD trials."""

import argparse
import collections
import itertools
import json
import math
from pathlib import Path

from .metrics import wilson_interval


PAPER_OBJECTS = ("box", "espressomachine", "ketchup", "laptop", "microwave", "mixer", "notebook", "waffleiron")
ERROR_FIELDS = (
    "paper_position_error_m", "paper_orientation_error_rad", "paper_articulation_error_rad",
    "public_position_error_m", "public_orientation_error_rad", "public_articulation_error_rad",
    "public_position_error_clipped_m", "public_orientation_error_clipped_rad", "public_articulation_error_clipped_rad",
    "lift_max_height_m", "lift_final_height_m", "lift_target_error_m",
    "lift_position_error_min_m", "lift_orientation_error_min_rad",
    "lift_position_error_final_m", "lift_orientation_error_final_rad",
    "initial_articulation_error_rad", "articulation_motion_rad",
    "articulation_start_angle_rad", "articulation_start_error_rad", "articulation_phase_motion_rad",
    "requested_final_rotation_rad", "requested_lift_rotation_rad",
)
SUCCESS_METRICS = ("paper_success", "public_success", "ood_goal_success", "strict_goal_success",
                   "lift_target_success", "lift_pose_success", "articulation_strict_success",
                   "paper_position_success", "paper_orientation_success", "paper_articulation_success")


def _load_rows(path):
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError("Invalid JSON at %s:%d: %s" % (path, number, error))
            if not isinstance(row, dict):
                raise ValueError("Each episodes.jsonl row must be an object")
            rows.append(row)
    return rows


def _is_success(row, metric):
    return bool(row.get("valid", True) and row.get(metric, False))


def _reachable(row):
    for key in ("target_reachable", "joint_target_reachable", "articulation_reachable", "target_within_joint_limits", "joint_limit_feasible"):
        if key in row:
            return bool(row[key])
    return None


def _aggregate(planned, completed):
    results = [completed[item["episode_id"]] for item in planned if item["episode_id"] in completed]
    valid = [row for row in results if row.get("valid", True)]
    count = len(planned)
    summary = {
        "planned": count,
        "completed": len(results),
        "missing": count - len(results),
        "invalid": len(results) - len(valid),
        "valid": len(valid),
        "unreachable_targets": sum(_reachable(completed.get(item["episode_id"], item)) is False for item in planned),
        "reachability_unknown": sum(_reachable(completed.get(item["episode_id"], item)) is None for item in planned),
        "provisional": len(results) < count,
    }
    for metric in SUCCESS_METRICS:
        successes = sum(_is_success(row, metric) for row in results)
        summary[metric + "_count"] = successes
        measured = any(isinstance(row.get(metric), bool) for row in results)
        summary[metric + "_rate"] = successes / count if count and measured else None
        # Unobserved trials are a provisional lower bound, not observed failures
        # from which a binomial confidence interval can legitimately be computed.
        summary[metric + "_wilson95"] = (wilson_interval(successes, count)
                                         if len(results) == count and measured else None)
    eligible = [item for item in planned if _reachable(completed.get(item["episode_id"], item)) is True
                and completed.get(item["episode_id"], item).get("articulation_randomized_target_ood") is True]
    summary["joint_ood_planned"] = len(eligible)
    summary["joint_ood_completed"] = sum(item["episode_id"] in completed for item in eligible)
    for metric in ("paper_success", "public_success", "strict_goal_success"):
        successes = sum(_is_success(completed.get(item["episode_id"], {}), metric) for item in eligible)
        summary["joint_ood_" + metric + "_count"] = successes
        measured = any(isinstance(completed.get(item["episode_id"], {}).get(metric), bool) for item in eligible)
        summary["joint_ood_" + metric + "_rate"] = successes / len(eligible) if eligible and measured else None
    summary["initial_articulation_within_paper_tolerance"] = sum(
        row.get("articulation_initial_within_paper_tolerance") is True for row in valid)
    summary["initial_articulation_measured"] = sum(
        isinstance(row.get("articulation_initial_within_paper_tolerance"), bool) for row in valid)
    summary["articulation_start_within_paper_tolerance"] = sum(
        row.get("articulation_start_within_paper_tolerance") is True for row in valid)
    summary["articulation_start_measured"] = sum(
        isinstance(row.get("articulation_start_within_paper_tolerance"), bool) for row in valid)
    for field in ERROR_FIELDS:
        values = [float(row[field]) for row in valid
                  if row.get(field) is not None and math.isfinite(float(row[field]))]
        summary[field + "_mean"] = sum(values) / len(values) if values else None
        summary[field + "_n"] = len(values)
    return summary


def _fmt(value, digits=4):
    return "—" if value is None else ("%.*f" % (digits, value))


def _profile_description(profile, field, default):
    value = profile.get(field)
    return default if value is None else json.dumps(value, ensure_ascii=False)


def _success_cell(summary, prefix):
    rate = summary[prefix + "_rate"]
    if rate is None:
        return "—"
    interval = summary[prefix + "_wilson95"]
    if interval is None:
        return "%d/%d (%.1f%%; 暫定下限・CI未算出)" % (
            summary[prefix + "_count"], summary["planned"], 100 * rate)
    low, high = interval
    return "%d/%d (%.1f%%; %.1f–%.1f%%)" % (
        summary[prefix + "_count"], summary["planned"], 100 * rate, 100 * low, 100 * high)


def _mean_cells(summary, public=False, clipped=False):
    prefix = "public" if public else "paper"
    suffix = "_clipped" if clipped else ""
    return " | ".join(_fmt(summary["%s_%s_error%s_%s_mean" % (prefix, name, suffix, unit)])
                      for name, unit in (("position", "m"), ("orientation", "rad"), ("articulation", "rad")))


def _paper_baselines(plan):
    explicit = plan.get("paper_baselines")
    if isinstance(explicit, dict) and "dynamic_task_table3" in explicit:
        return explicit
    path = Path(__file__).resolve().parents[2] / "configs" / "ood_paper_baselines.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _paired_comparisons(planned, completed, profiles, id_profile, comparisons=None):
    by_profile = collections.defaultdict(dict)
    for item in planned:
        key = (item["object"], item.get("sample_index", item.get("sample", 0)))
        by_profile[item["profile"]][key] = item
    pairs = []
    for profile in profiles:
        reference = (comparisons or {}).get(profile, id_profile)
        if reference is None or profile == reference:
            continue
        metrics = ["paper_success", "public_success"] + [name for name in ("ood_goal_success", "strict_goal_success")
                  if any(name in row for row in completed.values())]
        for metric in metrics:
            counts = {"both_success": 0, "id_only_success": 0, "ood_only_success": 0, "both_failure": 0, "missing_pairs": 0}
            keys = set(by_profile[reference]) | set(by_profile[profile])
            for key in sorted(keys):
                left = by_profile[reference].get(key)
                right = by_profile[profile].get(key)
                if not left or not right or left["episode_id"] not in completed or right["episode_id"] not in completed:
                    counts["missing_pairs"] += 1
                    continue
                if any(left.get(field) != right.get(field) for field in ("grasp_label", "arti_label", "seed")):
                    raise ValueError("Paired references differ for {} and {}".format(reference, profile))
                left_success = _is_success(completed[left["episode_id"]], metric)
                right_success = _is_success(completed[right["episode_id"]], metric)
                name = "both_success" if left_success and right_success else "id_only_success" if left_success else "ood_only_success" if right_success else "both_failure"
                counts[name] += 1
            n = sum(counts[key] for key in ("both_success", "id_only_success", "ood_only_success", "both_failure"))
            pairs.append(dict(profile=profile, reference_profile=reference, metric=metric, paired_trials=n,
                              success_difference_pp=100.0 * (counts["ood_only_success"] - counts["id_only_success"]) / n if n else None,
                              reference_success_count=counts["both_success"] + counts["id_only_success"],
                              target_success_given_reference_success=(
                                  counts["both_success"] / (counts["both_success"] + counts["id_only_success"])
                                  if counts["both_success"] + counts["id_only_success"] else None),
                              **counts))
    return pairs


def _interaction_summaries(planned, completed, definitions):
    """Paired 2x2 contrast; descriptive percentage points, not a significance test."""
    by_profile = collections.defaultdict(dict)
    for item in planned:
        by_profile[item["profile"]][(item["object"], item.get("sample_index", 0))] = item
    result = []
    for definition in definitions:
        names = [definition[key] for key in ("control", "a", "b", "combined")]
        if names[-1] not in by_profile:
            continue
        keys = set().union(*(set(by_profile[name]) for name in names))
        for metric in ("paper_success", "public_success", "ood_goal_success", "strict_goal_success"):
            contrasts, missing = [], 0
            for key in keys:
                items = [by_profile[name].get(key) for name in names]
                if any(item is None or item["episode_id"] not in completed for item in items):
                    missing += 1
                    continue
                if any(any(item.get(field) != items[0].get(field) for field in
                           ("grasp_label", "arti_label", "seed")) for item in items[1:]):
                    raise ValueError("Interaction requires matching initial references")
                if not all(isinstance(completed[item["episode_id"]].get(metric), bool) for item in items):
                    missing += 1
                    continue
                control, a, b, combined = [int(_is_success(completed[item["episode_id"]], metric)) for item in items]
                contrasts.append(combined - a - b + control)
            result.append(dict(definition, metric=metric, complete_quadruples=len(contrasts),
                               missing_quadruples=missing,
                               interaction_pp=100 * sum(contrasts) / len(contrasts) if contrasts else None))
    return result


FACTOR_NAMES = ("horizontal", "lift", "articulation")


def _factorial_summaries(planned, completed, definitions):
    """Matched 2^3 differences, averaged over the other factors and trials.

    Each contrast uses complete octuples with identical reset references.
    Invalid completed episodes are failures; missing measurements stay missing.
    """
    by_mask = collections.defaultdict(dict)
    for item in planned:
        flags = definitions[item["profile"]]["factor_flags"]
        mask = tuple(flags[name] for name in FACTOR_NAMES)
        key = (item["object"], item.get("sample_index", 0))
        if key in by_mask[mask]:
            raise ValueError("Duplicate factorial combination for {}".format(key))
        by_mask[mask][key] = item
    masks = list(itertools.product((False, True), repeat=3))
    keys = set().union(*(set(items) for items in by_mask.values()))
    metrics = ["paper_success", "public_success"] + [name for name in
               ("ood_goal_success", "strict_goal_success") if any(name in row for row in completed.values())]
    result = []
    for metric in metrics:
        octuples, missing = [], 0
        for key in sorted(keys):
            items = [by_mask[mask].get(key) for mask in masks]
            if any(item is None or item["episode_id"] not in completed for item in items):
                missing += 1
                continue
            if any(any(item.get(field) != items[0].get(field) for field in
                       ("grasp_label", "arti_label", "seed")) for item in items[1:]):
                raise ValueError("Factorial comparisons require matching initial references")
            rows = [completed[item["episode_id"]] for item in items]
            if any(row.get("valid", True) and not isinstance(row.get(metric), bool) for row in rows):
                missing += 1
                continue
            octuples.append([int(_is_success(row, metric)) for row in rows])
        for order in range(1, 4):
            for factors in itertools.combinations(range(3), order):
                values = [sum((-1) ** (order - sum(mask[i] for i in factors)) * value
                              for mask, value in zip(masks, outcomes)) / (2 ** (3 - order))
                          for outcomes in octuples]
                result.append(dict(factors=[FACTOR_NAMES[i] for i in factors], order=order, metric=metric,
                                   complete_octuples=len(values), missing_octuples=missing,
                                   difference_pp=100 * sum(values) / len(values) if values else None))
    return result


def _factorial_tables(plan, groups, aggregates, effects):
    if plan.get("config", {}).get("evaluation_protocol") != "factorial_v1":
        return []
    definitions = plan["config"]["profiles"]
    lines = ["", "## 3因子・8通りの組合せ", "",
             "水平移動・持ち上げ・開閉の3因子をON/OFFします。OFFはnormalと同じ目標、ONは分布外の範囲から試行ごとにランダム抽出した目標です。同じ物体・sample_indexでは初期状態、通常の向き、各ON因子の抽出値を8通りで共有します。", "",
             "| 条件 | 水平移動OOD | 持ち上げOOD | 開閉OOD | n | 論文Suc.T | 公開実装互換 | OOD目標達成 | 厳密目標達成 |",
             "| --- | --- | --- | --- | ---: | --- | --- | --- | --- |"]
    for group in aggregates:
        if group["scope"] != "all_objects":
            continue
        flags = definitions[group["profile"]]["factor_flags"]
        cells = ["ON" if flags[name] else "OFF" for name in FACTOR_NAMES]
        successes = [_success_cell(group, metric) for metric in
                     ("paper_success", "public_success", "ood_goal_success", "strict_goal_success")]
        lines.append("| %s | %s | %d | %s |" % (group["profile"], " | ".join(cells),
                                                     group["planned"], " | ".join(successes)))
    lines.extend(["", "### 開閉OODの確認", "",
                  "開閉角は学習時のランダム目標範囲[0.5,1.5] radの外側とURDF可動域の共通部分から一様抽出します。小角度目標は開閉開始時点で既に論文の許容誤差0.5 rad内の場合があるため、厳密指標（0.1 rad）と開始時の達成件数を併記します。", "",
                  "| 条件 | 物体 | 関節範囲外 | 範囲内の関節OOD n | うち完了 | 条件付き論文Suc.T | 条件付き厳密目標達成 | 開閉開始時に論文閾値内/測定数 |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"])
    for group in groups:
        if not definitions[group["profile"]]["factor_flags"]["articulation"]:
            continue
        lines.append("| %s | %s | %d | %d | %d | %s | %s | %d/%d |" % (
            group["profile"], group["object"], group["unreachable_targets"], group["joint_ood_planned"],
            group["joint_ood_completed"], _fmt(group["joint_ood_paper_success_rate"]),
            _fmt(group["joint_ood_strict_goal_success_rate"]),
            group["articulation_start_within_paper_tolerance"], group["articulation_start_measured"]))
    lines.extend(["", "### 因子の効果と組合せによる変化", "",
                  "全物体の同じ初期状態8通りが完了した組だけを使う記述統計です。単因子は他の2因子について平均したON−OFF差、2因子は残る因子について平均した差の差、3因子は差の差の差です。100倍したppで示し、有意差検定とは区別します。未実行や指標未測定の組数も表示します。", "",
                  "| 因子 | 次数 | 指標 | 完了8条件組 | 未完/未測定8条件組 | 差 (pp) |",
                  "| --- | ---: | --- | ---: | ---: | ---: |"])
    for effect in effects:
        lines.append("| %s | %d | %s | %d | %d | %s |" % (
            " × ".join(effect["factors"]), effect["order"], effect["metric"],
            effect["complete_octuples"], effect["missing_octuples"], _fmt(effect["difference_pp"], 1)))
    return lines


def _stratified_summaries(planned, completed, profiles):
    summaries = []
    for profile in profiles:
        items = [item for item in planned if item["profile"] == profile]
        factors = sorted({name for item in items for name in item.get("target", {}).get("strata", {})})
        for factor in factors:
            levels = sorted({str(item.get("target", {}).get("strata", {}).get(factor)) for item in items})
            if len(levels) < 2 or len(levels) > 12:
                continue
            for level in levels:
                subset = [item for item in items if str(item.get("target", {}).get("strata", {}).get(factor)) == level]
                summaries.append(dict(profile=profile, factor=factor, level=level,
                                      **_aggregate(subset, completed)))
    return summaries


def _multiaxis_tables(plan, groups, aggregates, strata, interactions):
    if plan.get("config", {}).get("evaluation_protocol") != "multiaxis_v1":
        return []
    definitions = plan["config"]["profiles"]
    lines = ["", "## 因子・強度別の比較", "",
             "同じsample_indexでは初期ラベル・乱数を共有します。空中のX/Y/3D回転は高さ0.6 mのair_controlが対照です。机上yawは世界Z軸回転です。強度は各因子内の段階で、異なる因子間の物理量を同一尺度とはみなしません。", "",
             "| 条件 | 因子 | 強度 | 対照 | n | 論文Suc.T | OOD目標達成 | 厳密目標達成 |",
             "| --- | --- | ---: | --- | ---: | --- | --- | --- |"]
    for summary in aggregates:
        if summary["scope"] != "all_objects":
            continue
        definition = definitions.get(summary["profile"], {})
        lines.append("| %s | %s | %s | %s | %d | %s | %s | %s |" % (
            summary["profile"], definition.get("family", "—"), definition.get("severity", "—"),
            definition.get("compare_to", "—"), summary["planned"], _success_cell(summary, "paper_success"),
            _success_cell(summary, "ood_goal_success"), _success_cell(summary, "strict_goal_success")))
    lines.extend(["", "## どの目標で失敗したか", "",
                  "各列は別々の条件達成率で、排他的な失敗原因の分類ではありません。空中姿勢は位置と姿勢を同一時刻に満たす必要があります。", "",
                  "| 条件 | 物体 | 最終位置 | 最終姿勢 | 最終関節角（論文） | 空中姿勢 | OOD目標達成 | 厳密目標達成 | 空中終端PE (m) | 空中終端AE (rad) |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: |"])
    for summary in groups:
        cells = [_success_cell(summary, metric) for metric in (
            "paper_position_success", "paper_orientation_success", "paper_articulation_success",
            "lift_pose_success", "ood_goal_success", "strict_goal_success")]
        lines.append("| %s | %s | %s | %s | %s |" % (
            summary["profile"], summary["object"], " | ".join(cells),
            _fmt(summary["lift_position_error_final_m_mean"]),
            _fmt(summary["lift_orientation_error_final_rad_mean"])))
    lines.extend(["", "## 関節目標の適用範囲と初期状態", "",
                  "関節角のOODは公開学習コードのランダム目標範囲[0.5,1.5] radを基準とします。可動域上限に対する比率が大きくても、espressomachineなどではこの範囲内に留まります。下表の条件付き成功率は『可動域内かつこの関節目標範囲外』の計画試行だけを分母にします。主表の全物体成功率からは除外しません。", "",
                  "小角度目標や移送中に関節が動いた場合は、開閉開始時点で既に論文の0.5 rad閾値内であることがあります。0.1 radの厳密指標と、開閉開始前から許容内だった数を併記します。リセット直後の角度と開閉開始時の角度・位相内の変化量も保存します。小角度目標を、必ず開状態から閉じる動作と解釈しないでください。", "",
                  "| 条件 | 物体 | 関節範囲外 | 範囲内の関節OOD n | うち完了 | 条件付き論文Suc.T | 条件付き厳密目標達成 | 開閉開始時に論文閾値内/測定数 |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"])
    for summary in groups:
        definition = definitions.get(summary["profile"], {})
        if not ("articulation_deg" in definition or "articulation_limit_fraction" in definition):
            continue
        lines.append("| %s | %s | %d | %d | %d | %s | %s | %d/%d |" % (
            summary["profile"], summary["object"], summary["unreachable_targets"], summary["joint_ood_planned"],
            summary["joint_ood_completed"], _fmt(summary["joint_ood_paper_success_rate"]),
            _fmt(summary["joint_ood_strict_goal_success_rate"]),
            summary["articulation_start_within_paper_tolerance"], summary["articulation_start_measured"]))
    if strata:
        lines.extend(["", "## 方向・回転符号の内訳", "",
                      "全物体を集約した内訳です。方向4群は各物体100例なら25例ずつ、回転符号は50例ずつです。方向条件は対照の移動距離を保ちます。", "",
                      "| 条件 | 分類 | 値 | n | 論文Suc.T | OOD目標達成 | 厳密目標達成 |",
                      "| --- | --- | --- | ---: | --- | --- | --- |"])
        for summary in strata:
            lines.append("| %s | %s | %s | %d | %s | %s | %s |" % (
                summary["profile"], summary["factor"], summary["level"], summary["planned"],
                _success_cell(summary, "paper_success"), _success_cell(summary, "ood_goal_success"),
                _success_cell(summary, "strict_goal_success")))
    if interactions:
        lines.extend(["", "## 距離と向きの交互作用", "",
                      "同じ初期状態の4条件を対応させた記述統計です。差は100×(組合せ成功−距離単独成功−向き単独成功＋対照成功)の平均（pp）です。負値は単独要因の低下の和を超える低下を示します。有意差の検定とは区別してください。", "",
                      "| 組合せ | 指標 | 完了4条件組 | 未完4条件組 | 交互作用 (pp) |",
                      "| --- | --- | ---: | ---: | ---: |"])
        for value in interactions:
            lines.append("| %s | %s | %d | %d | %s |" % (
                value["name"], value["metric"], value["complete_quadruples"], value["missing_quadruples"],
                _fmt(value["interaction_pp"], 1)))
    return lines


def generate_report(results_dir):
    if json.loads((Path(results_dir) / "plan.json").read_text(encoding="utf-8")).get("config", {}).get("evaluation_protocol") == "five_v1":
        from .five_report import generate_report as five_report
        return five_report(results_dir)
    """Read plan.json + episodes.jsonl; write report.md and summary.json.

    Planned trials always determine success denominators. Invalid trials count
    as failures, while uncompleted plans are explicit provisional non-successes.
    Error means use valid finite measurements only and retain their sample counts.
    No actual simulation is performed by this function.
    """
    directory = Path(results_dir).resolve()
    plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    planned = plan.get("episodes", plan.get("trials", []))
    if not planned:
        raise ValueError("plan.json must contain a non-empty episodes list")
    for item in planned:
        for field in ("episode_id", "object", "profile"):
            if field not in item:
                raise ValueError("Each planned episode requires %s" % field)
    ids = [item["episode_id"] for item in planned]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate episode_id in plan.json")
    planned_by_id = {item["episode_id"]: item for item in planned}
    rows = _load_rows(directory / "episodes.jsonl")
    completed = {}
    for row in rows:
        identifier = row.get("episode_id")
        if identifier not in planned_by_id:
            raise ValueError("Result episode_id is absent from plan: %r" % identifier)
        if identifier in completed:
            raise ValueError("Duplicate episode result: %s" % identifier)
        expected = planned_by_id[identifier]
        if any(row.get(key) != expected[key] for key in ("object", "profile")):
            raise ValueError("Result object/profile differs from plan for %s" % identifier)
        completed[identifier] = row
    baseline = _paper_baselines(plan)
    paper_objects = tuple(baseline.get("paper_objects", PAPER_OBJECTS))
    objects = list(dict.fromkeys(item["object"] for item in planned))
    profiles = list(dict.fromkeys(item["profile"] for item in planned))
    definitions = plan.get("config", {}).get("profiles", {})
    groups = []
    aggregates = []
    for profile in profiles:
        for obj in objects:
            subset = [item for item in planned if item["object"] == obj and item["profile"] == profile]
            if subset:
                definition = definitions.get(profile, {})
                groups.append(dict(object=obj, profile=profile, family=definition.get("family", "legacy"),
                                   severity=definition.get("severity"), compare_to=definition.get("compare_to"),
                                   **_aggregate(subset, completed)))
        for scope, subset in (
            ("all_objects", [item for item in planned if item["profile"] == profile]),
            ("paper_objects", [item for item in planned if item["profile"] == profile and item["object"] in paper_objects]),
        ):
            if subset:
                summary = _aggregate(subset, completed)
                per_object = [group for group in groups if group["profile"] == profile and (scope == "all_objects" or group["object"] in paper_objects)]
                summary["object_count"] = len(per_object)
                for metric in ("paper_success", "public_success"):
                    values = [group[metric + "_rate"] for group in per_object]
                    summary[metric + "_macro_rate"] = (sum(values) / len(values)
                                                       if all(value is not None for value in values) else None)
                aggregates.append(dict(profile=profile, scope=scope, **summary))

    id_profile = next((name for name in ("normal", "id", "in_distribution", "id_control", "published") if name in profiles), None)
    comparisons = {name: definition["compare_to"] for name, definition in definitions.items()
                   if definition.get("compare_to")}
    paired = _paired_comparisons(planned, completed, profiles, id_profile, comparisons)
    interactions = _interaction_summaries(planned, completed, plan.get("config", {}).get("interactions", []))
    factorial = (_factorial_summaries(planned, completed, definitions)
                 if plan.get("config", {}).get("evaluation_protocol") == "factorial_v1" else [])
    strata = _stratified_summaries(planned, completed, profiles)
    controller_changes = sorted({change for row in rows for change in row.get("controller_changes", [])})
    summary = {
        "schema_version": 1, "planned": len(planned), "completed": len(completed),
        "missing": len(planned) - len(completed), "objects": objects, "profiles": profiles,
        "groups": groups, "aggregates": aggregates, "paired_comparisons": paired,
        "interaction_comparisons": interactions, "stratified_summaries": strata,
        "factorial_comparisons": factorial,
        "controller_changes": controller_changes,
        "baseline_source": baseline["source"], "published_baselines": baseline["dynamic_task_table3"],
    }
    range_note = ("8条件版は関節の可動域内でのみ抽出します。空間目標の到達可能性は保証せず、失敗試行も分母に含めます。"
                  if plan.get("config", {}).get("evaluation_protocol") == "factorial_v1" else
                  "180°など関節範囲外の要求目標も勝手に丸めず評価します。範囲外件数を併記し、到達可能な目標への汎化とは区別してください。")
    lines = [
        "# ArtiGrasp OOD 目標状態の評価結果", "",
        "進捗: **%d / %d 試行完了**。" % (len(completed), len(planned)), "",
        "成功率の分母は全計画試行です。実行異常は失敗として含め、未実行は未達として明示します。未実行が残る表は暫定の下限値で、信頼区間は未算出です。全試行完了後にWilson法の95%区間を算出します。", "",
        "## 評価条件", "",
        "分割: `%s`、seed: `%s`。条件間で同じ参照・初期状態を対応させ、目標を変えて評価します。持ち上げ量は移送途中の目標で、最終配置位置のz成分は机上です。" % (plan.get("split", plan.get("config", {}).get("split", "不明")), plan.get("seed", plan.get("config", {}).get("seed", "不明"))), "",
        "| 条件 | 因子 / 強度 | 対照 | 持ち上げ量 (m) | 水平移動距離 (m) / 方向 | 回転軸・角度・段階 | 関節角度 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in profiles:
        configuration = plan.get("config", {}).get("profiles", {}).get(name, {})
        normal_goal = plan.get("config", {}).get("normal_goal", {})
        default_angle = (json.dumps(normal_goal["articulation_rad"]) + " rad（絶対角）"
                         if normal_goal else "公開設定（初期角度を参照）")
        angle_description = _profile_description(configuration, "articulation_deg", default_angle)
        if "articulation_deg" in configuration:
            angle_description += " deg"
        elif "articulation_rad" in configuration:
            angle_description = _profile_description(configuration, "articulation_rad", "") + " rad"
        elif "articulation_limit_fraction" in configuration:
            angle_description = "可動域の " + json.dumps(configuration["articulation_limit_fraction"])
        elif "articulation_ood" in configuration:
            angle_definition = configuration["articulation_ood"]
            angle_description = "%s rad ∩ 可動域内（端から%s rad除外）" % (
                json.dumps(angle_definition["intervals_rad"]), angle_definition.get("joint_margin_rad", 0.05))
        direction = configuration.get("translation_directions_deg", configuration.get("translation_direction", "公開設定"))
        default_orientation = ({"axis": "world z", "yaw_rad": normal_goal["yaw_rad"]}
                               if normal_goal else "公開設定")
        orientation = configuration.get("orientation", configuration.get("orientation_delta_deg", default_orientation))
        lines.append("| %s | %s / %s | %s | %s | %s / %s | %s | %s |" % (name,
            configuration.get("family", "legacy"), configuration.get("severity", "—"),
            configuration.get("compare_to", "id_control"),
            _profile_description(configuration, "lift_m", "公開設定: 通常0.15 / ketchup 0.30"),
            _profile_description(configuration, "translation_distance_m", "公開設定: x,y各[-0.10,-0.05]"),
            json.dumps(direction, ensure_ascii=False), json.dumps(orientation, ensure_ascii=False), angle_description))
    lines.extend(["",
        "## 指標の区別", "",
        "- **論文 Suc.T**: 最後の `env.step` 後の物理状態で PE < 0.05 m、SO(3)測地距離 AE < 0.2 rad、AAE < 0.5 rad を同時に満たす割合。誤差は無加工です。",
        "- **公開実装の判定互換**: `92fb307` の compose/ketchup の成功・誤差の計算式を同じロールアウトへ適用します。到達・姿勢・開閉を別々の時刻に達成しても成功を保持し、最後に公開コードのguardを適用します。元コードの1ステップ遅れと、物理姿勢の代わりに目標quaternionを読む添字も保持します。制御側の補正がある場合、元スクリプトと同一の軌跡を再現したという意味ではありません。",
        "- **OOD目標達成**: 論文Suc.Tに加え、持ち上げ目標を満たす本実験の補助指標です。従来版と8条件版factorial_v1は上昇量誤差 < 0.05 m、旧多軸版multiaxis_v1は空中目標の位置誤差 < 0.05 mかつ姿勢誤差 < 0.2 radを同じ時刻に達成することも要求します。",
        "- **厳密目標達成**: OOD目標達成の関節角閾値を0.5 radから0.1 radへ狭めた補助指標です。論文Suc.Tは変更していません。小角度目標では初期角が既に論文閾値内にある場合も別に数えます。",
        "- 公開実装のPE/AEは移送終了付近、AAEは最後のステップの直前の状態です。公開実装の表示に合わせたクリップ値（PE 0.1 m、AE 0.3 rad、AAE 1.0 rad上限）を無加工値と分けます。",
        "- 誤差平均は正常かつ有限の測定値だけを使います。測定数は別表に表示します。異常試行を誤差0として埋めません。", "",
        range_note, "",
        "## 物体別の成功率", "",
        "| 条件 | 物体 | 完了/計画 | 異常 | 未実行 | 関節範囲外 | 論文 Suc.T (95% CI) | 公開実装互換 (95% CI) |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ])
    if "ketchup_forward_requested_articulation" in controller_changes:
        location = lines.index(range_note)
        lines[location:location] = [
            "**ketchupの目標角度入力の補正**: 公開スクリプトは`set_obj_goal(angle)`で表示・位置用の目標を変えても、ポリシーの関節目標`final_obj_angle_`を更新せず0 radを維持します。本実験では開閉準備への遷移時に要求角度を`set_goals`にも渡します。元の段階・制御切替は保持しますが、要求目標を実際にポリシーへ渡すための補正により軌跡は公開スクリプトと異なり得ます。公開互換の数値は、この補正後の同じ軌跡に元の判定式を適用した結果です。", "",
        ]
    for group in groups:
        lines.append("| %s | %s | %d/%d | %d | %d | %d | %s | %s |" % (
            group["profile"], group["object"], group["completed"], group["planned"], group["invalid"], group["missing"], group["unreachable_targets"],
            _success_cell(group, "paper_success"), _success_cell(group, "public_success")))

    lines.extend(_multiaxis_tables(plan, groups, aggregates, strata, interactions))
    lines.extend(_factorial_tables(plan, groups, aggregates, factorial))

    lines.extend(["", "## 論文 Table 3 との比較", "",
        "[ArtiGrasp 論文・Table 3 / §6.1 / Supplement B.4](%s)。公開値は8物体・約7500試行の集約で、物体別のTable 3値は掲載されていません。物体・条件ごとに同数のサンプルを評価する本実験とは重みや目標分布が異なります。差は参考比較であり、再現誤差とは解釈しません。" % baseline["source"]["url"], "",
        "| 結果 | 対象 | n | 論文 Suc.T | PE (m) | AE (rad) | AAE (rad) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in baseline["dynamic_task_table3"]:
        lines.append("| 論文 %s | 論文8物体 | 約7500 | %.2f | %.2f | %.2f | %.2f |" % (
            row["model"], row["task_success"], row["position_error_m"], row["orientation_error_rad"], row["articulation_error_rad"]))
    for aggregate in aggregates:
        scope = "全%d物体" % aggregate["object_count"] if aggregate["scope"] == "all_objects" else "論文対象の%d物体" % aggregate["object_count"]
        lines.append("| 実験 %s | %s | %d | %s | %s |" % (aggregate["profile"], scope, aggregate["planned"], _fmt(aggregate["paper_success_rate"]), _mean_cells(aggregate)))
    lines.extend(["", "### 集約方法", "",
        "| 条件 | 対象 | 論文・試行平均 (95% CI) | 論文・物体平均 | 公開互換・試行平均 (95% CI) | 公開互換・物体平均 |",
        "| --- | --- | --- | ---: | --- | ---: |"])
    for group in aggregates:
        lines.append("| %s | %s | %s | %s | %s | %s |" % (group["profile"], group["scope"], _success_cell(group, "paper_success"),
            _fmt(group["paper_success_macro_rate"]), _success_cell(group, "public_success"), _fmt(group["public_success_macro_rate"])))
    lines.extend(["", "## 持ち上げ目標と複合目標の補助評価", "",
        "持ち上げ達成は区間 `[grasp_steps, grasp_steps + transit_steps]` の物理高さから初期高さを引いた上昇量で判定します。目標誤差は区間内での最小絶対誤差です。これらはTable 3の論文指標と区別して解釈してください。", "",
        "| 条件 | 物体 | 持ち上げ目標達成 (95% CI) | OOD目標達成 (95% CI) | 最大上昇量 (m) | 持ち上げ区間終了の上昇量 (m) | 目標との最小誤差 (m) |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |"])
    for group in groups:
        lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (group["profile"], group["object"],
            _success_cell(group, "lift_target_success"), _success_cell(group, "ood_goal_success"), _fmt(group["lift_max_height_m_mean"]),
            _fmt(group["lift_final_height_m_mean"]), _fmt(group["lift_target_error_m_mean"])))
    lines.extend(["", "## 誤差の物体別集計", "",
        "nはPE/AE/AAEの各有限測定数です。論文の誤差は同一の最終時刻、公開互換は元実装の各測定時刻です。", "",
        "| 条件 | 物体 | 指標 | n (PE/AE/AAE) | PE (m) | AE (rad) | AAE (rad) |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |"])
    for group in groups:
        for public, clipped, label in ((False, False, "論文・無加工"), (True, False, "公開互換・無加工"), (True, True, "公開互換・表示クリップ")):
            prefix, suffix = ("public" if public else "paper"), ("_clipped" if clipped else "")
            counts = "/".join(str(group["%s_%s_error%s_%s_n" % (prefix, name, suffix, unit)]) for name, unit in (("position", "m"), ("orientation", "rad"), ("articulation", "rad")))
            lines.append("| %s | %s | %s | %s | %s |" % (group["profile"], group["object"], label, counts, _mean_cells(group, public, clipped)))

    if paired:
        lines.extend(["", "## 同じ参照ペアによる 対照 → 評価条件 の変化", "",
            "同じ物体・sample_indexを対応させた比較です。両条件の実行が完了したペアを示し、異常完了は失敗に含めます。実行前に同じsample_indexが同じ参照/初期状態へ対応していることはplan.jsonで確認できます。", "",
            "| 評価条件 | 対照 | 指標 | 完了ペア | 両方成功 | 対照のみ成功 | 評価条件のみ成功 | 両方失敗 | 未完ペア | 成功率差 (pp) | 対照成功例での達成率 (n) |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"])
        for pair in paired:
            lines.append("| %s | %s | %s | %d | %d | %d | %d | %d | %d | %s | %s (%d) |" % (
                pair["profile"], pair["reference_profile"], pair["metric"], pair["paired_trials"],
                pair["both_success"], pair["id_only_success"], pair["ood_only_success"],
                pair["both_failure"], pair["missing_pairs"], _fmt(pair["success_difference_pp"], 1),
                _fmt(pair["target_success_given_reference_success"]), pair["reference_success_count"]))

    lines.extend(["", "## 定性的な動画", "",
        "成功/失敗は指定した指標に従います。各条件・物体で最大5件ずつを保存します。0成功の条件で成功例を生成することはできないため、不足をそのまま記録します。"])
    indexes = sorted((directory / "videos").glob("*/index.json"))
    if (directory / "videos" / "index.json").exists():
        indexes.append(directory / "videos" / "index.json")
    if indexes:
        for index in indexes:
            md_path = index.with_suffix(".md")
            relative = (md_path if md_path.exists() else index).relative_to(directory).as_posix()
            lines.append("- [%s の動画一覧](%s)" % (index.parent.name, relative))
    else:
        lines.append("動画インデックスはまだありません。定量評価後にrecordコマンドを実行してください。")
    lines.extend(["", "| 条件 | 物体 | 正常な成功候補 | 正常な失敗候補 | 成功不足 (5件基準) | 失敗不足 (5件基準) |",
        "| --- | --- | ---: | ---: | ---: | ---: |"])
    video_metric = plan.get("config", {}).get("video_metric", "paper_success")
    for group in groups:
        valid = [row for row in completed.values() if row["object"] == group["object"] and
                 row["profile"] == group["profile"] and row.get("valid", True) and
                 isinstance(row.get(video_metric), bool)]
        successes = sum(_is_success(row, video_metric) for row in valid)
        failures = len(valid) - successes
        lines.append("| %s | %s | %d | %d | %d | %d |" % (group["profile"], group["object"], successes, failures, max(0, 5 - successes), max(0, 5 - failures)))
    lines.extend(["", "上表の候補と不足は `%s` を基準にしています。実際の録画件数・再生検証は各動画インデックスを参照してください。" % video_metric, "",
        "## 再現用ファイル", "",
        "- [計画と固定seed](plan.json)", "- [試行ごとの全結果](episodes.jsonl)", "- [機械可読集計](summary.json)", "",
        "計画は条件ごとに分けており、物体数や関節上限が異なる対象を黙って落としません。未実行・異常・範囲外目標の有無を確認してから結果を使用してください。", ""])
    (directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return {"report": str(directory / "report.md"), "summary": str(directory / "summary.json")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir")
    args = parser.parse_args(argv)
    paths = generate_report(args.results_dir)
    print(paths["report"])


if __name__ == "__main__":
    main()
