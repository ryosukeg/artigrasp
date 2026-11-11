#!/usr/bin/env python3
"""Independently verify adaptive playback metadata, sources and decoded media.

Never simulates, renders, or imports the renderer. Only media_verification.json
is written. The state-to-frame mapping is reconstructed from raw NPZ stop steps.
Pixel meaning additionally requires a visual/source review; video hashing and
full decoding alone cannot establish which meshes were drawn into its pixels.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
METHODS = ("baseline", "guided_absolute", "guided_delta")
EXPECTED_IDS = (
    "box__normal__0017", "box__normal__0086", "box__ood_articulation__0019",
    "box__ood_articulation__0041", "box__ood_articulation__0037", "box__ood_articulation__0053",
)
START, DT = 1260, .01
METRIC_FIELDS = ("paper_position_error_m", "paper_articulation_error_rad", "paper_orientation_error_rad")


def require(value, message):
    if not value:
        raise AssertionError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_path(root, relative):
    result = (root / relative).resolve()
    require(root.resolve() in result.parents, "Path escapes root: " + str(relative))
    return result


def close(actual, expected, name, tolerance=1e-10):
    require(math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance),
            "{}: {} != {}".format(name, actual, expected))


def arrays(actual, expected, name):
    try:
        np.testing.assert_array_equal(actual, expected)
    except AssertionError as error:
        raise AssertionError(name + ": " + str(error)) from error


def nodes(value):
    yield value
    if isinstance(value, dict):
        for item in value.values():
            yield from nodes(item)
    elif isinstance(value, list):
        for item in value:
            yield from nodes(item)


def physical_metrics(state, goal):
    state, goal = np.asarray(state, float), np.asarray(goal, float)
    actual = state[122:130]
    a, b = actual[3:7], goal[3:7]
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return [float(np.linalg.norm(actual[:3] - goal[:3])),
            abs(float(actual[7] - goal[7])),
            2 * math.acos(float(np.clip(abs(a @ b), 0., 1.)))]


def png_dimensions(path):
    with Path(path).open("rb") as stream:
        header = stream.read(24)
    require(header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR", "Invalid PNG " + str(path))
    return struct.unpack(">II", header[16:24])


def inspect_video(path):
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,nb_frames,avg_frame_rate,codec_name,pix_fmt:format=duration",
        "-of", "json", str(path)], check=True, capture_output=True, text=True)
    value = json.loads(probe.stdout)
    require(len(value["streams"]) == 1, "Expected one video stream")
    stream = value["streams"][0]
    numerator, denominator = [int(x) for x in stream["avg_frame_rate"].split("/")]
    return dict(width=stream["width"], height=stream["height"], frames=int(stream["nb_frames"]),
                duration=float(value["format"]["duration"]), fps=numerator / denominator,
                codec=stream["codec_name"], pixel_format=stream["pix_fmt"])


def check_sources(manifest, audit, run):
    hashes = manifest["source_sha256"]
    resolved = {}
    for name, digest in hashes.items():
        if name in ("plan.json", "results.json", "independent_audit.json"):
            path = run / name
        elif name == "previous_qualitative_manifest":
            path = Path(read(run / "plan.json")["source_run"]) / "qualitative" / "manifest.json"
        else:
            path = Path(name) if Path(name).is_absolute() else ROOT / name
        require(sha(path) == digest, "Changed rendering source/asset: " + str(path))
        resolved[str(path.resolve())] = digest
    for name, digest in audit["verified_source_and_input_sha256"].items():
        if "/rsc/" in name:
            require(resolved.get(str(Path(name).resolve())) == digest, "Audited asset omitted/changed: " + name)
    for path in (HERE / "render_adaptive.py", HERE / "render_threeway.py", ROOT / "scripts/render_ood_five.py"):
        require(str(path) in resolved, "Missing renderer/helper source hash: " + str(path))
    return resolved


def verify_case(run, output, case, result_rows, source_hashes, required_hashes):
    episode = case["episode_id"]
    paths = {name: safe_path(output, case[name]) for name in ("video", "poster", "terminal_image")}
    for name, path in paths.items():
        require(path.is_file() and sha(path) == case[name + "_sha256"], episode + " changed " + name)
    require(read(paths["video"].with_suffix(".json")) == case, episode + " manifest/per-video metadata differ")
    dimensions = (int(case["encoded"]["width"]), int(case["encoded"]["height"]))
    require(min(dimensions) > 0 and png_dimensions(paths["poster"]) == dimensions and
            png_dimensions(paths["terminal_image"]) == dimensions, episode + " poster/video size")
    cache = case["cache_key"]
    cache_nodes = list(nodes(cache))
    strings = {value for value in cache_nodes if isinstance(value, str)}
    require(set(required_hashes.values()).issubset(strings), episode + " cache omits plan/results/audit binding")
    require(all(digest in strings for digest in source_hashes.values()),
            episode + " cache omits source/asset hashes")
    require(any(isinstance(node, dict) and node == case["terminations"] for node in cache_nodes),
            episode + " cache omits terminations")
    series = case.get("series")
    if series is not None:
        times = np.asarray(series["time_s"], dtype=float)
        require(times.ndim == 1 and len(times) > 1 and np.isfinite(times).all() and
                times[0] == 0 and np.all(np.diff(times) > 0), episode + " invalid angle-series timeline")
        series_indices = np.rint(times / DT).astype(int)
        require(np.allclose(series_indices * DT, times, rtol=0, atol=1e-10), episode + " angle-series time not physical steps")
    metrics = {item["method"]: item for item in case["metrics"]}
    require(set(metrics) == set(METHODS) and len(case["metrics"]) == 3, episode + " metric methods")
    require(set(case["terminations"]) == set(METHODS), episode + " termination methods")
    stop_steps, trajectories, prefixes = {}, {}, {}
    for method in METHODS:
        row = result_rows[(method, episode)]
        require(row["valid"] and row["profile"] == case["profile"], episode + " invalid/mismatched selected row")
        require(case["terminations"][method] == row["termination"], episode + "/" + method + " termination metadata")
        trajectory = safe_path(run, row["trajectory_path"])
        require(sha(trajectory) == row["trajectory_sha256"] and row["trajectory_sha256"] in strings,
                episode + "/" + method + " trajectory/cache hash")
        with np.load(trajectory, allow_pickle=False) as archive:
            states, final = archive["states"].copy(), archive["final_states"].copy()
            goal = archive["goal"].copy()
            opening = int(archive["stop_opening_step"])
            reason = str(archive["stop_reason"].item())
        stop = START + opening
        stop_steps[method] = stop
        require(states.shape == (stop + 1, 138) and final.shape == (opening + 1, 138), episode + " variable state length")
        require(np.isfinite(states).all() and np.isfinite(final).all() and np.isfinite(goal).all(), episode + " nonfinite selected trajectory")
        require(opening == row["termination"]["opening_step"] and reason == row["termination"]["reason"], episode + " stop NPZ/row mismatch")
        close(row["termination"]["total_seconds"], stop * DT, episode + " physical stop seconds")
        arrays(states[START + 1:], final[1:], episode + " final-state history")
        # The hand reset is instantaneous at physical t=12.6; it is not an extra step.
        states[START] = final[0]
        prefixes[method] = (states[:START + 1].copy(), goal.copy())
        close(case["target_angle_rad"], goal[7], episode + " target")
        computed = physical_metrics(states[-1], goal)
        for key, field, value in zip(("PE", "AAE", "AE"), METRIC_FIELDS, computed):
            close(row[field], value, episode + "/" + method + " row " + field)
            close(metrics[method][key], value, episode + "/" + method + " displayed " + key)
        success = computed[0] < .05 and computed[1] < .5 and computed[2] < .2
        require(metrics[method]["paper_success"] == row["paper_success"] == success,
                episode + "/" + method + " displayed physical success")
        require(metrics[method]["composite_success"] == row["ood_goal_success"], episode + " displayed lift success")
        if series is not None:
            mapped_series = np.minimum(series_indices, stop)
            arrays(series["angle_rad"][method], states[mapped_series, 129].astype(float), episode + "/" + method + " angle series")
        trajectories[method] = (states, goal, row["trajectory_sha256"])
    for method in METHODS[1:]:
        arrays(prefixes[method][0], prefixes[METHODS[0]][0], episode + " common prefix/post-reset entry")
        arrays(prefixes[method][1], prefixes[METHODS[0]][1], episode + " common goal")
    longest = max(stop_steps.values())
    close(case["duration"], longest * DT, episode + " longest physical duration")
    close(case["opening_start"], START * DT, episode + " opening start")
    if series is not None:
        close(times[-1], longest * DT, episode + " series terminal inclusion")
    fps = case["fps"]
    require(isinstance(fps, int) and fps > 0, episode + " fps")
    # Integer arithmetic avoids ceil errors such as ceil(19.69*20)=394.
    frames = (longest * fps + 99) // 100
    common = np.rint(np.arange(frames) * 100. / fps).astype(int)
    common = np.minimum(common, longest)
    common[-1] = longest
    arrays(case["frame_state_indices"], common, episode + " independent common frame mapping")
    require(START in common and np.all(np.diff(common) > 0), episode + " reset frame absent/nonmonotone timeline")
    require(set(case["method_frame_state_indices"]) == set(METHODS), episode + " method mapping keys")
    frozen = {}
    for method, stop in stop_steps.items():
        mapped = np.minimum(common, stop)
        arrays(case["method_frame_state_indices"][method], mapped, episode + "/" + method + " clamp mapping")
        terminal_frames = np.flatnonzero(mapped == stop)
        require(len(terminal_frames) > 0 and np.all(mapped[terminal_frames[0]:] == stop), episode + " missing/unfrozen terminal")
        require(mapped[np.flatnonzero(common == START)[0]] == START, episode + " wrong post-reset frame")
        states = trajectories[method][0]
        require(all(np.array_equal(states[index], states[-1]) for index in mapped[terminal_frames[0]:]), episode + " post-stop state differs")
        frozen[method] = dict(stop_step=stop, first_terminal_frame=int(terminal_frames[0]),
                              terminal_frames=int(len(terminal_frames)), terminal_state_included=True)
    encoded = inspect_video(paths["video"])
    require((encoded["width"], encoded["height"]) == dimensions and encoded["frames"] == frames,
            episode + " encoded size/frame count")
    close(encoded["fps"], fps, episode + " encoded FPS")
    close(encoded["duration"], frames / fps, episode + " encoded duration", .001)
    for name in ("width", "height", "frames", "duration"):
        close(case["encoded"][name], encoded[name], episode + " saved ffprobe " + name, .001)
    require(encoded["codec"] == "h264" and encoded["pixel_format"] == "yuv420p", episode + " browser video encoding")
    decoded = subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-threads", "2", "-i", str(paths["video"]),
                              "-map", "0:v:0", "-f", "null", "-"], capture_output=True, text=True)
    require(decoded.returncode == 0 and not decoded.stderr.strip(), episode + " full decode failed: " + decoded.stderr)
    return dict(episode_id=episode, rollouts=3, physical_duration=longest * DT,
                encoded=encoded, video_bytes=paths["video"].stat().st_size,
                video_sha256=sha(paths["video"]), methods=frozen,
                physical_metrics_verified=True, angle_series_verified=series is not None,
                source_hashes_verified=True,
                fully_decoded=True)


def verify(results, allow_partial=False):
    run = Path(results).resolve()
    output = run / "qualitative"
    started = time.monotonic()
    report = dict(status="running", verifier_sha256=sha(__file__), output=str(output),
                  cases=[], simulation_performed=False, rendering_performed=False,
                  pixel_semantics_automatically_verified=False)
    try:
        manifest_path = output / "manifest.json"
        manifest = read(manifest_path)
        require(manifest["status"] == "complete" or (allow_partial and manifest["status"] == "partial"),
                "Complete manifest required (or --allow-partial)")
        require(manifest["simulation_performed"] is False, "Media claims a new simulation")
        cases = manifest["cases"]
        ids = [case["episode_id"] for case in cases]
        require(len(set(ids)) == len(ids) and set(ids).issubset(EXPECTED_IDS) and ids, "Unexpected/duplicate representative IDs")
        if manifest["status"] == "complete":
            require(set(ids) == set(EXPECTED_IDS), "Missing representative cases")
        result, audit, plan = read(run / "results.json"), read(run / "independent_audit.json"), read(run / "plan.json")
        hashes = {name: sha(run / filename) for name, filename in (
            ("source_plan_sha256", "plan.json"), ("source_results_sha256", "results.json"),
            ("source_audit_sha256", "independent_audit.json"))}
        for key, value in hashes.items():
            require(manifest[key] == value, "Media source hash differs: " + key)
        require(audit["status"] == "completed" and audit["results_sha256"] == hashes["source_results_sha256"] and
                audit["plan_sha256"] == hashes["source_plan_sha256"], "Independent trajectory audit not current")
        require(result["complete"] and result["plan_sha256"] == hashes["source_plan_sha256"], "Results incomplete")
        rows = {(row["method"], row["episode_id"]): row for row in result["episodes"]}
        require(len(rows) == len(result["episodes"]), "Duplicate result rows")
        source_hashes = check_sources(manifest, audit, run)
        report.update(manifest_sha256=sha(manifest_path), source_hashes=hashes,
                      source_assets_verified=len([p for p in source_hashes if "/rsc/" in p]))
        old_manifest = read(Path(plan["source_run"]) / "qualitative" / "manifest.json")
        require(set(EXPECTED_IDS) == {case["episode_id"] for case in old_manifest["cases"]},
                "Selected episodes differ from fixed comparison")
        for case in cases:
            report["cases"].append(verify_case(run, output, case, rows, source_hashes, hashes))
            print("Verified media " + case["episode_id"], flush=True)
        require(check_sources(manifest, audit, run) == source_hashes, "Sources changed during media verification")
        require(sha(manifest_path) == report["manifest_sha256"], "Manifest changed during verification")
        for key, filename in (("source_plan_sha256", "plan.json"), ("source_results_sha256", "results.json"),
                              ("source_audit_sha256", "independent_audit.json")):
            require(sha(run / filename) == hashes[key], "Source changed during media verification: " + filename)
        report.update(status="passed", complete=manifest["status"] == "complete",
                      trajectories_verified=3 * len(cases), total_frames=sum(x["encoded"]["frames"] for x in report["cases"]),
                      videos_fully_decoded=ids, notes=[
                          "No simulation, policy/model inference or rendering performed",
                          "Frames independently reconstructed from saved stop steps and common physical timeline",
                          "Exact post-reset state at 12.6 seconds; all 18 terminals included when complete",
                          "Each method freezes at its own terminal state; no post-stop physics represented",
                          "Hashes/metadata/full decoding do not replace source and visual review of pixel semantics",
                      ])
    except BaseException as error:
        report.update(status="failed", error="{}: {}".format(type(error).__name__, error))
        raise
    finally:
        report["verified_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["elapsed_seconds"] = time.monotonic() - started
        destination = output / "media_verification.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(destination)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "runs/comparison_seed202_adaptive_v1")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    try:
        report = verify(args.results, args.allow_partial)
    except (Exception, KeyboardInterrupt) as error:
        print("Media verification failed: {}".format(error), file=sys.stderr, flush=True)
        return 1
    print(json.dumps({key: report[key] for key in ("status", "complete", "trajectories_verified", "total_frames")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
