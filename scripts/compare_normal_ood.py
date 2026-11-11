#!/usr/bin/env python3
"""Compare completed, provenance-matched paired normal and five-condition runs.

This reads saved artifacts only; it neither changes a run nor rescores a rollout.
It deliberately refuses incomplete or incompatible input before creating output.
"""

import argparse
import csv
import html
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from ood_suite.five_report import summarize
from ood_suite.io import file_sha256, read_json, read_rows, write_json

OOD_PROFILES = ("horizontal", "height", "orientation", "articulation", "all")
LABELS = {"normal": "normal（通常）", "horizontal": "水平", "height": "高さ",
          "orientation": "向き", "articulation": "関節角度", "all": "全て"}
PAPER_OBJECTS = {"box", "laptop", "notebook", "waffleiron", "mixer", "ketchup",
                 "espressomachine", "microwave"}
IDENTITY = ("object", "sample_index", "seed", "split", "paper_object",
            "grasp_label_index", "arti_label_index", "grasp_label", "arti_label")
ERRORS = {"PE": "paper_position_error_m", "AE": "paper_orientation_error_rad",
          "AAE": "paper_articulation_error_rad"}
CORE_SOURCES = {"tools/ood_suite/" + name + ".py" for name in
                ("runtime", "metrics", "targets", "constraints", "geometry", "paths", "io")}
PAPER_URL = "https://arxiv.org/html/2309.03891v2"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(spec):
    return {key: spec[key] for key in IDENTITY}


def load_run(directory, expected_profiles):
    directory = Path(directory).resolve()
    for name in ("plan.json", "episodes.jsonl", "provenance.json", "execution.json"):
        require((directory / name).is_file(), "Missing input: {}".format(directory / name))
    plan = read_json(directory / "plan.json")
    provenance = read_json(directory / "provenance.json")
    execution = read_json(directory / "execution.json")
    require(provenance.get("plan_sha256") == file_sha256(directory / "plan.json"),
            "Saved plan hash mismatch: {}".format(directory))
    require(plan.get("profiles") == list(expected_profiles),
            "Unexpected profiles in {}: {}".format(directory, plan.get("profiles")))
    require(plan.get("config", {}).get("evaluation_protocol") == "five_v1",
            "Comparison requires the five_v1 scoring protocol")
    objects = plan["objects"]
    count = plan["samples_per_object"]
    require(objects and len(objects) == len(set(objects)), "Empty or duplicate object list")
    require(isinstance(count, int) and not isinstance(count, bool) and count > 0,
            "Invalid samples_per_object")
    planned = {}
    groups = {}
    for spec in plan["episodes"]:
        episode_id = spec["episode_id"]
        require(episode_id not in planned, "Duplicate planned episode: " + episode_id)
        require(spec["object"] in objects and spec["profile"] in expected_profiles,
                "Episode outside declared object/profile grid")
        require(type(spec["sample_index"]) is int and 0 <= spec["sample_index"] < count,
                "Episode sample index outside declared grid")
        require(spec["split"] == plan["split"], "Episode split differs from saved plan")
        require(spec["paper_object"] == (spec["object"] in PAPER_OBJECTS),
                "Incorrect paper-object membership")
        identity(spec)  # Required pairing fields must be present.
        group = (spec["object"], spec["profile"], spec["sample_index"])
        require(group not in groups, "Duplicate planned object/profile/sample")
        planned[episode_id] = spec
        groups[group] = spec
    require(len(planned) == len(objects) * len(expected_profiles) * count,
            "Saved plan does not cover the complete declared grid")
    rows = read_rows(directory / "episodes.jsonl")
    require({row["episode_id"] for row in rows} == set(planned),
            "Incomplete results or episodes outside saved plan: {} ({}/{})".format(
                directory, len(rows), len(planned)))
    indexed = {}
    for row in rows:
        spec = planned[row["episode_id"]]
        require(identity(row) == identity(spec) and row["profile"] == spec["profile"],
                "Result identity differs from saved plan: " + row["episode_id"])
        for key in ("valid", "ood_goal_success", "paper_success"):
            require(type(row.get(key)) is bool, "Missing Boolean {} in {}".format(key, row["episode_id"]))
        if row["valid"]:
            for field in ERRORS.values():
                value = row.get(field)
                require(isinstance(value, (int, float)) and not isinstance(value, bool)
                        and math.isfinite(value) and value >= 0,
                        "Missing/nonfinite final error: {} {}".format(row["episode_id"], field))
        for key in ("timing", "environment_config", "policy_device", "initial_articulation_rad"):
            require(key in row, "Missing runtime field {} in {}".format(key, row["episode_id"]))
        indexed[(row["object"], row["profile"], row["sample_index"])] = row
    return dict(path=directory, plan=plan, provenance=provenance, execution=execution,
                rows=rows, indexed=indexed, planned=groups)


def verify_comparable(normal, ood):
    np, op = normal["plan"], ood["plan"]
    for key in ("objects", "samples_per_object", "seed", "split", "label_counts", "sampling"):
        require(np.get(key) == op.get(key), "Run plan differs: " + key)
    for key in ("checkpoints", "normal_goal", "physics_validation", "runtime", "video_metric"):
        require(np["config"].get(key) == op["config"].get(key), "Run configuration differs: " + key)
    require(np["config"].get("normal_goal") == {
        "source": "paper_supplement_b4", "articulation_rad": [0.5, 0.6], "yaw_rad": [-0.4, 0.4]},
        "Normal target definition is not the paper B.4 distribution")
    nv, ov = normal["provenance"], ood["provenance"]
    require(nv.get("input_sha256") and nv["input_sha256"] == ov.get("input_sha256"),
            "Checkpoint/data/geometry input hashes differ or are missing")
    checkpoints = np["config"].get("checkpoints", {})
    require(checkpoints, "Checkpoint configuration is missing")
    for path in checkpoints.values():
        require(path in nv["input_sha256"], "Checkpoint input hash is missing: " + path)
    ns, os = nv.get("source_sha256", {}), ov.get("source_sha256", {})
    sources = CORE_SOURCES | {p for p in ns if p.startswith("raisimGymTorch/")}
    require(sources > CORE_SOURCES, "RaiSim source provenance is missing")
    require({p for p in ns if p.startswith("raisimGymTorch/")} ==
            {p for p in os if p.startswith("raisimGymTorch/")}, "RaiSim source file sets differ")
    for path in sorted(sources):
        require(path in ns and path in os and ns[path] == os[path],
                "Evaluation source hash differs or is missing: " + path)
    ne, oe = normal["execution"], ood["execution"]
    for key in ("python", "platform", "device"):
        require(ne.get(key) and ne[key] == oe.get(key), "Execution environment differs: " + key)
    for key in ("numpy", "torch", "ruamel.yaml", "raisimGymTorch.env.bin.compose_eval", "compose_eval_sha256"):
        require(ne.get("runtime", {}).get(key) and ne["runtime"][key] == oe.get("runtime", {}).get(key),
                "Runtime version/binary differs: " + key)
    pairs = 0
    for key, baseline in normal["planned"].items():
        obj, _, sample = key
        target = baseline["target"]
        require(target.get("changed_axes") == [] and target.get("normal_goal_source") == "paper_supplement_b4",
                "Normal profile contains an OOD intervention")
        require(target.get("articulation_rule") == "absolute" and .5 <= target["articulation_rad"] <= .6,
                "Normal articulation is outside paper range")
        require(len(target["translation_m"]) == 2 and all(-.1 <= v <= -.05 for v in target["translation_m"]),
                "Normal translation is outside paper range")
        require(target["lift_m"] == (.3 if obj == "ketchup" else .15), "Normal lift differs from public implementation")
        rotation = target.get("orientation_rotation", {})
        require(rotation.get("axis_name") == "z" and rotation.get("frame") == "world"
                and rotation.get("stage") == "both" and -.4 <= rotation.get("angle_rad", float("inf")) <= .4,
                "Normal yaw is outside paper range")
        nr = normal["indexed"][key]
        for profile in OOD_PROFILES:
            other = ood["planned"][(obj, profile, sample)]
            require(identity(baseline) == identity(other),
                    "Paired sample identity mismatch: {} {} {}".format(obj, profile, sample))
            row = ood["indexed"][(obj, profile, sample)]
            for field in ("timing", "environment_config", "policy_device", "initial_articulation_rad"):
                require(nr[field] == row[field], "Paired runtime/state mismatch: {} {} {} {}".format(
                    obj, profile, sample, field))
            pairs += 1
    return dict(paired_trials_verified=pairs, evaluation_source_files_verified=len(sources),
                input_files_verified=len(nv["input_sha256"]),
                compose_eval_sha256=ne["runtime"]["compose_eval_sha256"],
                matched_sources={p: ns[p] for p in sorted(sources)})


def difference(value, baseline):
    return value - baseline if value is not None and baseline is not None else None


def metric_record(scope, obj, profile, rows, baseline, planned):
    record = dict(scope=scope, object=obj, profile=profile, **summarize(rows, planned))
    for metric in ("SucT", "paper_SucT", "PE", "AE", "AAE"):
        suffix = "_pp" if "SucT" in metric else ""
        value = difference(record[metric], baseline[metric])
        record["delta_" + metric + suffix] = value * 100 if suffix and value is not None else value
    return record


def paired_record(scope, obj, profile, pairs):
    result = dict(scope=scope, object=obj, profile=profile, pairs=len(pairs),
                  normal_invalid=sum(not n["valid"] for n, _ in pairs),
                  ood_invalid=sum(not o["valid"] for _, o in pairs))
    for name, field in (("SucT", "ood_goal_success"), ("paper_SucT", "paper_success")):
        counts = dict(both_success=0, normal_only=0, ood_only=0, both_failure=0)
        for normal, ood in pairs:
            n = bool(normal["valid"] and normal[field])
            o = bool(ood["valid"] and ood[field])
            key = "both_success" if n and o else "normal_only" if n else "ood_only" if o else "both_failure"
            counts[key] += 1
        result.update({name + "_" + k: v for k, v in counts.items()})
        result[name + "_delta_pp"] = 100 * (counts["ood_only"] - counts["normal_only"]) / len(pairs)
    for name, field in ERRORS.items():
        values = [o[field] - n[field] for n, o in pairs if n["valid"] and o["valid"]]
        result[name + "_paired_n"] = len(values)
        result[name + "_paired_delta"] = sum(values) / len(values) if values else None
    return result


def build_comparison(normal_directory, ood_directory):
    normal = load_run(normal_directory, ("normal",))
    ood = load_run(ood_directory, OOD_PROFILES)
    verified = verify_comparable(normal, ood)
    aggregates, objects, paired = [], [], []
    names = normal["plan"]["objects"]
    scopes = [("all", "全{}物体".format(len(names)), names),
              ("paper", "論文対象{}物体".format(len(set(names) & PAPER_OBJECTS)),
               [obj for obj in names if obj in PAPER_OBJECTS])]
    groups = [(scope, "", members) for scope, _, members in scopes if members]
    groups += [("object", obj, [obj]) for obj in names]
    count = normal["plan"]["samples_per_object"]
    for scope, obj, members in groups:
        normal_rows = [r for r in normal["rows"] if r["object"] in members]
        baseline = summarize(normal_rows, count * len(members))
        target_records = objects if scope == "object" else aggregates
        target_records.append(metric_record(scope, obj, "normal", normal_rows, baseline, count * len(members)))
        for profile in OOD_PROFILES:
            rows = [r for r in ood["rows"] if r["profile"] == profile and r["object"] in members]
            target_records.append(metric_record(scope, obj, profile, rows, baseline, count * len(members)))
            pairs = [(r, ood["indexed"][(r["object"], profile, r["sample_index"])]) for r in normal_rows]
            paired.append(paired_record(scope, obj, profile, pairs))
    inputs = {}
    for label, run in (("normal", normal), ("ood", ood)):
        inputs[label] = dict(path=str(run["path"]),
                            artifacts_sha256={p: file_sha256(run["path"] / p) for p in
                                              ("plan.json", "episodes.jsonl", "provenance.json", "execution.json")})
    return dict(complete=True, normal_completed=len(normal["rows"]), ood_completed=len(ood["rows"]),
                normal_planned=len(normal["planned"]), ood_planned=len(ood["planned"]),
                inputs=inputs, verification=verified, scopes=[dict(scope=s, title=t, objects=m) for s,t,m in scopes],
                conditions=dict(seed=normal["plan"]["seed"], split=normal["plan"]["split"], samples_per_object=count,
                                checkpoints=normal["plan"]["config"]["checkpoints"],
                                runtime=normal["plan"]["config"]["runtime"], execution=normal["execution"]),
                primary_success="final PE<0.05m AND SO(3) AE<0.2rad AND AAE<0.5rad AND lift height error<0.05m",
                error_definition="Unclipped final-state means over all valid trials including failures; invalid trials are failures in success denominators.",
                delta_definition="OOD minus normal; success differences in percentage points; errors in m/rad.",
                aggregates=aggregates, objects=objects, paired=paired)


def value(x, percentage=False, signed=False):
    if x is None:
        return "—"
    if percentage:
        return "{:.1%}".format(x)
    return ("{:+.3f}" if signed else "{:.3f}").format(x)


def markdown_table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] + ["---:"] * (len(headers)-1)) + "|"] + [
        "| " + " | ".join(str(v) for v in row) + " |" for row in rows]


def render_report(summary):
    c = summary["conditions"]
    lines = ["# normal と OOD の比較", "", "**normal {}/{}・OOD {}/{} 試行完了。対応と実行条件の検査済み。**".format(
        summary["normal_completed"], summary["normal_planned"], summary["ood_completed"], summary["ood_planned"]), "",
        "Suc.T は最終 PE < 0.05 m・AE < 0.2 rad・AAE < 0.5 rad と、中間の要求上昇量との誤差 < 0.05 m をすべて満たす成功率です。論文Suc.T は最終状態の3条件だけです。", "",
        "PE/AE/AAE は正常に測定できた全試行の最終誤差平均です。失敗例を含み、クリップしません。異常試行は成功率の分母に含めて失敗とし、誤差平均から除外します。PE は m、AE/AAE は rad。測定数を各表と CSV に明示します。", "",
        "差はすべて **OOD − normal**。成功率の差はパーセントポイント（pp）です。", ""]
    for scope in summary["scopes"]:
        records = [r for r in summary["aggregates"] if r["scope"] == scope["scope"]]
        if not records:
            continue
        lines += ["## " + scope["title"], ""]
        rows = [[LABELS[r["profile"]], "{}/{}".format(r["completed"],r["planned"]), r["successes"],
                 value(r["SucT"],True),value(r["PE"]),value(r["AE"]),value(r["AAE"]),
                 value(r["paper_SucT"],True),r["invalid"],"{}/{}/{}".format(r["PE_n"],r["AE_n"],r["AAE_n"])] for r in records]
        lines += markdown_table(["条件","完了/予定","成功数","Suc.T","PE","AE","AAE","論文Suc.T","異常","PE/AE/AAE測定数"],rows)
        lines += ["", "### normal からの差", ""]
        lines += markdown_table(["OOD条件","ΔSuc.T (pp)","ΔPE (m)","ΔAE (rad)","ΔAAE (rad)","Δ論文Suc.T (pp)"],[
            [LABELS[r["profile"]]] + [value(r[k],signed=True) for k in
             ("delta_SucT_pp","delta_PE","delta_AE","delta_AAE","delta_paper_SucT_pp")]
            for r in records if r["profile"] != "normal"])
        lines += ["", "### 対応する試行の成功・失敗", ""]
        pairs = [r for r in summary["paired"] if r["scope"] == scope["scope"]]
        lines += markdown_table(["OOD条件","対応数","双方成功","normalのみ成功","OODのみ成功","双方失敗"],[
            [LABELS[r["profile"]],r["pairs"]] + [r["SucT_"+k] for k in
             ("both_success","normal_only","ood_only","both_failure")] for r in pairs])
        lines += [""]
    lines += ["## 物体別の比較", "", "各物体も同じ分母・指標で集計しています。", ""]
    for obj in dict.fromkeys(r["object"] for r in summary["objects"]):
        lines += ["### " + obj, ""]
        lines += markdown_table(["条件","成功/予定","Suc.T","PE","AE","AAE","論文Suc.T","異常","ΔSuc.T (pp)","PE/AE/AAE測定数"],[
            [LABELS[r["profile"]],"{}/{}".format(r["successes"],r["planned"]),value(r["SucT"],True),
             value(r["PE"]),value(r["AE"]),value(r["AAE"]),value(r["paper_SucT"],True),r["invalid"],
             value(r["delta_SucT_pp"],signed=True),"{}/{}/{}".format(r["PE_n"],r["AE_n"],r["AAE_n"])]
            for r in summary["objects"] if r["object"] == obj])
        lines += [""]
    e = c["execution"]
    lines += ["## 実行条件と検査", "",
              "seed={}、split={}、各物体・各条件{}試行。方策は {}、物理計算は CPU、{}スレッドです。".format(
                  c["seed"],c["split"],c["samples_per_object"],e["device"],c["runtime"].get("num_threads")), "",
              "normal と OOD で object/sample/seed/split/参照ラベル・初期関節角・制御段階・環境設定を照合しました。checkpoint と入力データ、RaiSim binary、Python/NumPy/PyTorch/ruamel.yaml、評価・制御ソースのハッシュも一致しています。追加した設定・レポートコードは一致検査の対象外です。", "",
              "- Python: " + e["python"].replace("\n"," "),
              "- NumPy: " + e["runtime"]["numpy"] + "; PyTorch: " + e["runtime"]["torch"] + "; ruamel.yaml: " + e["runtime"]["ruamel.yaml"],
              "- RaiSim compose_eval SHA256: `" + e["runtime"]["compose_eval_sha256"] + "`", ""]
    lines += ["- " + name + ": `" + path + "`" for name,path in c["checkpoints"].items()]
    lines += ["", "## 論文との対応", "",
              "normal は論文 Supplementary B.4 の目標分布です。水平 x/y 各 −0.10〜−0.05 m、配置時 z 変位0 m、世界Z軸 yaw ±0.4 rad、絶対関節角0.5〜0.6 rad。途中の持ち上げは公開実装の0.15 m（ketchupは0.30 m）です。配置目標には既存 OOD と同じ衝突形状による最大2 cmの支持面補正を適用しています。", "",
              "公開 runner は初期角による相対関節角、ketchup の別範囲、Euler 回転、独立した成功ラッチや誤差クリップを含みます。この比較では既存 OOD と同じ論文式の最終状態指標を使います。公開 runner の標準出力との数値一致は意味しません。", "",
              "論文 Table 3 は grasp×articulation の全組合せ約7,500試行です。本検証は参照ラベル対を復元抽出した各物体{}試行で、厳密な全組合せ再現ではありません。論文対象8物体と全10物体の集計を分けています。".format(c["samples_per_object"]), "",
              "論文の ArtiGrasp は Suc.T 50%、PE 0.03 m、AE 0.10 rad、AAE 0.41 rad。比較するときは追加の高さ条件を含まない「論文Suc.T」を使用します。[論文 B.4・6.1・Table 3]("+PAPER_URL+")", "",
              "## 成果物", "", "- [集計と差のCSV](comparison.csv)", "- [物体別CSV](per-object.csv)",
              "- [対応試行の成功不一致と誤差差](paired_comparison.csv)", "- [入力ハッシュ・検査記録・全数値](summary.json)", "",
              "paired_comparison.csv の誤差差は、双方が正常に測定できた対応試行だけの OOD−normal 平均です。対象集合が異なるため、各条件の全有効試行平均の差とは異なる場合があります。", ""]
    return "\n".join(lines)


def markdown_html(markdown):
    """Render our restricted Markdown using only the standard library."""
    import re
    output = []
    in_table = False
    for line in markdown.splitlines():
        if line.startswith("|"):
            cells = [part.strip() for part in line.strip("|").split("|")]
            if all(re.fullmatch(r":?-+:?", part) for part in cells):
                continue
            if not in_table:
                output.append('<div class="table"><table>')
                in_table = True
            output.append("<tr>" + "".join("<td>" + html.escape(cell) + "</td>" for cell in cells) + "</tr>")
            continue
        if in_table:
            output.append("</table></div>")
            in_table = False
        if not line:
            continue
        escaped = html.escape(line)
        escaped = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', escaped)
        escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
        escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            output.append("<h{0}>{1}</h{0}>".format(level,escaped[level:].strip()))
        else:
            output.append("<p>" + escaped + "</p>")
    if in_table:
        output.append("</table></div>")
    return ('<!doctype html><html lang="ja"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>normal と OOD の比較</title><style>'
            'body{font:16px system-ui,sans-serif;background:#f5f7fa;color:#213047;margin:0}'
            'main{max-width:1280px;margin:auto;padding:32px}p{line-height:1.8}'
            'h2{margin-top:40px}table{border-collapse:collapse;width:100%;background:white}'
            'td{padding:10px;border-bottom:1px solid #dde3ea;text-align:right;white-space:nowrap}'
            'td:first-child{text-align:left}tr:first-child{background:#e8eef6;font-weight:bold}'
            '.table{overflow:auto;margin:16px 0}a{color:#175aa9}code{overflow-wrap:anywhere}'
            '</style><main>' + "\n".join(output) + '</main></html>')


def write_csv(path, records):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def generate(normal_directory, ood_directory, output):
    output = Path(output).resolve()
    inputs = [Path(p).resolve() for p in (normal_directory, ood_directory)]
    require(not any(output == p or p in output.parents for p in inputs),
            "Comparison output must be separate from immutable input runs")
    summary = build_comparison(*inputs)
    markdown = render_report(summary)
    page = markdown_html(markdown)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", summary)
    write_csv(output / "comparison.csv", summary["aggregates"])
    write_csv(output / "per-object.csv", summary["objects"])
    write_csv(output / "paired_comparison.csv", summary["paired"])
    (output / "report.md").write_text(markdown, encoding="utf-8")
    (output / "report.html").write_text(page, encoding="utf-8")
    return output / "report.md"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal", type=Path, default=ROOT / "results/normal_seed1")
    parser.add_argument("--ood", type=Path, default=ROOT / "results/ood5_seed1")
    parser.add_argument("--output", type=Path, default=ROOT / "results/normal_vs_ood_seed1")
    args = parser.parse_args(argv)
    try:
        result = generate(args.normal, args.ood, args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, "Comparison failed: {}\n".format(error))
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
