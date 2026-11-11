#!/usr/bin/env python3
"""Record selected, already evaluated OOD episodes through RaiSimUnity on Linux.

The simulation is replayed by ``ood_eval.py``. This program collects the actual
Unity recording, checks the replay result and the video stream, and writes a
video index. It deliberately never synthesizes a video from plots or stills.
Only Python's standard library is needed here; ffprobe must be on PATH.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import time
import uuid


BASE_METRICS = ("paper_success", "public_success")
METRICS = BASE_METRICS + ("ood_goal_success", "strict_goal_success")


def load_episodes(path):
    rows = []
    seen = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("episode_id", "object", "profile"):
                if not isinstance(row.get(key), str) or not row[key]:
                    raise ValueError("{}:{}: invalid {}".format(path, line_number, key))
            if row["episode_id"] in seen:
                raise ValueError("Duplicate episode_id: " + row["episode_id"])
            seen.add(row["episode_id"])
            if not isinstance(row.get("valid"), bool):
                raise ValueError("Episode {} has no Boolean valid flag".format(row["episode_id"]))
            if row["valid"]:
                for key in BASE_METRICS:
                    if not isinstance(row.get(key), bool):
                        raise ValueError("Episode {} has no Boolean {}".format(row["episode_id"], key))
            rows.append(row)
    if not rows:
        raise ValueError("No evaluated episodes in " + str(path))
    return rows


def select_episodes(rows, metric="paper_success", per_outcome=5, objects=None, profiles=None):
    """Return deterministic selections and counts, including empty outcome bins."""
    if metric not in METRICS:
        raise ValueError("Unknown success metric: " + metric)
    if per_outcome < 1:
        raise ValueError("per_outcome must be positive")
    filtered = [row for row in rows
                if (not objects or row["object"] in objects)
                and (not profiles or row["profile"] in profiles)]
    for row in filtered:
        if row.get("valid") and not isinstance(row.get(metric), bool):
            raise ValueError("Episode {} has no Boolean {}; select a recorded metric".format(
                row["episode_id"], metric))
    groups = sorted(set((row["object"], row["profile"]) for row in filtered))
    selections = []
    for object_name, profile in groups:
        group = [row for row in filtered
                 if row["object"] == object_name and row["profile"] == profile]
        valid = [row for row in group if row["valid"]]
        for outcome, success in (("success", True), ("failure", False)):
            candidates = sorted(
                (row for row in valid if row[metric] is success),
                key=lambda row: (int(row.get("sample_index", 0)), row["episode_id"]))
            selections.append({
                "object": object_name, "profile": profile, "outcome": outcome,
                "metric": metric, "requested": per_outcome,
                "available": len(candidates), "invalid_excluded": len(group) - len(valid),
                "selected": candidates[:per_outcome],
                "shortage": max(0, per_outcome - len(candidates)),
            })
    return selections


def safe_component(value):
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")[:100]
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:8]
    if cleaned == value and cleaned:
        return cleaned
    return (cleaned or "episode") + "_" + digest


def default_capture_dirs(repo_root):
    """Bundled Unity versions use different capitalization for this directory."""
    linux_dir = Path(repo_root) / "raisimUnity" / "linux"
    candidates = [linux_dir / name for name in ("screenshots", "Screenshot", "Screenshots")]
    candidates += [linux_dir, Path.cwd() / "screenshots", Path.cwd() / "Screenshot"]
    return list(dict.fromkeys(path.resolve() for path in candidates))


def probe_video(path, ffprobe="ffprobe", minimum_duration=0.1):
    """Reject missing, unfinished, empty, or non-video captures."""
    result = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_type,width,height,duration,nb_frames,avg_frame_rate:format=duration",
         "-of", "json", str(path)],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True, timeout=15)
    if result.returncode != 0:
        raise ValueError("ffprobe rejected {}: {}".format(path, result.stderr.strip()[-1000:]))
    info = json.loads(result.stdout)
    streams = info.get("streams", [])
    if not streams or streams[0].get("codec_type") != "video":
        raise ValueError("Capture has no video stream: " + str(path))
    stream = streams[0]
    if int(stream.get("width", 0)) <= 0 or int(stream.get("height", 0)) <= 0:
        raise ValueError("Capture has invalid frame dimensions: " + str(path))
    duration = float(info.get("format", {}).get("duration", stream.get("duration", 0)))
    if not math.isfinite(duration) or duration < minimum_duration:
        raise ValueError("Capture too short ({:.3f}s; expected >= {:.3f}s): {}".format(
            duration, minimum_duration, path))
    info = {"duration_seconds": duration, "width": int(stream["width"]),
            "height": int(stream["height"]), "bytes": Path(path).stat().st_size,
            "average_frame_rate": stream.get("avg_frame_rate")}
    frames = str(stream.get("nb_frames", ""))
    if frames.isdigit():
        info["frames"] = int(frames)
        if info["frames"] < 2:
            raise ValueError("Capture has fewer than two video frames: " + str(path))
    return info


def find_capture(capture_dirs, basename):
    """Search only the exact unique basename, never choose an unrelated video."""
    for directory in capture_dirs:
        directory = Path(directory)
        direct = directory / basename
        if direct.is_file():
            return direct
        if directory.is_dir():
            # Recorder builds sometimes place MP4s in one timestamped subdir.
            for candidate in sorted(directory.glob("*/" + basename)):
                if candidate.is_file():
                    return candidate
    return None


def wait_for_capture(capture_dirs, basename, timeout=30, ffprobe="ffprobe", minimum_duration=0.1):
    deadline = time.monotonic() + timeout
    last_signature = None
    last_error = "No file appeared"
    while time.monotonic() < deadline:
        candidate = find_capture(capture_dirs, basename)
        if candidate is not None:
            stat = candidate.stat()
            signature = (str(candidate), stat.st_size, stat.st_mtime_ns)
            if signature == last_signature and stat.st_size > 0:
                try:
                    return candidate, probe_video(candidate, ffprobe, minimum_duration)
                except (ValueError, subprocess.SubprocessError, json.JSONDecodeError) as error:
                    last_error = str(error)
            last_signature = signature
        time.sleep(0.5)
    raise RuntimeError("Unity capture {} was not completed in {}s. Searched: {}. {}. "
                       "Keep RaiSimUnity connected with Auto-connect and specify "
                       "--unity-capture-dir if its recording folder differs.".format(
                           basename, timeout, ", ".join(str(path) for path in capture_dirs), last_error))


def verify_replay(original, replay):
    if replay.get("episode_id") != original["episode_id"]:
        raise ValueError("Replay result belongs to another episode")
    if replay.get("verified") is not True:
        raise ValueError("Replay diverged from the quantitative trajectory")
    for metric in METRICS:
        if metric in original and replay.get(metric) is not original[metric]:
            raise ValueError("Replay changed {} for {}".format(metric, original["episode_id"]))


def run_replay(command, cwd, log, timeout):
    """Allow the replay's finally block to deliver STOP_RECORD_VIDEO on timeout."""
    process = subprocess.Popen(command, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT)
    try:
        return process.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        # subprocess.run(timeout=...) sends SIGKILL immediately. That would leave
        # Unity recording because the simulation could not run its cleanup.
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        raise


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def markdown_index(manifest):
    lines = ["# OOD qualitative examples", "",
             "Classification: `{}`. These are replays of the evaluated episodes, "
             "verified against their quantitative trajectories.".format(manifest["metric"]), "",
             "| Object | Profile | Outcome | Available | Requested | Selected | Shortage |",
             "|---|---|---|---:|---:|---:|---:|"]
    for selection in manifest["selection"]:
        lines.append("| {object} | {profile} | {outcome} | {available} | {requested} | {selected_count} | {shortage} |".format(
            selected_count=len(selection["episode_ids"]), **selection))
    lines += ["", "A shortage means the evaluated samples contain fewer examples of that outcome. "
              "No extra or relabeled episodes are substituted.", "",
              "| Object | Profile | Outcome | Episode | Video | Replay verified |",
              "|---|---|---|---|---|---|"]
    for entry in manifest["videos"]:
        link = "[MP4]({})".format(entry["relative_path"]) if entry.get("verified") else "ERROR"
        lines.append("| {} | {} | {} | {} | {} | {} |".format(
            entry["object"], entry["profile"], entry["outcome"], entry["episode_id"],
            link, "yes" if entry.get("verified") else "no"))
    errors = [entry for entry in manifest["videos"] if entry.get("error")]
    if errors:
        lines.append("")
        for entry in errors:
            lines.append("- `{}`: {}".format(entry["episode_id"], entry["error"].replace("\n", " ")))
    return "\n".join(lines) + "\n"


def retain_unselected_groups(previous_manifest, selections, rows):
    """A filtered update preserves the index for other evaluated groups."""
    updated = {(selection["object"], selection["profile"]) for selection in selections}
    known = {(row["object"], row["profile"]) for row in rows}
    episodes = {row["episode_id"] for row in rows}
    retained_selections = [dict(selection) for selection in previous_manifest.get("selection", [])
                           if (selection["object"], selection["profile"]) in known - updated]
    retained_groups = {(selection["object"], selection["profile"]) for selection in retained_selections}
    for selection in retained_selections:
        selection["episode_ids"] = [episode_id for episode_id in selection.get("episode_ids", [])
                                     if episode_id in episodes]
    retained_videos = [dict(entry) for entry in previous_manifest.get("videos", [])
                       if (entry["object"], entry["profile"]) in retained_groups
                       and entry["episode_id"] in episodes]
    return retained_selections, retained_videos


def record_selected(args):
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = Path(args.results).resolve()
    if args.metric is None:
        plan_path = results_dir / "plan.json"
        config = json.loads(plan_path.read_text(encoding="utf-8")).get("config", {}) if plan_path.exists() else {}
        args.metric = config.get("video_metric", "paper_success")
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is missing from PATH; use the existing Ubuntu ffmpeg installation")
    capture_dirs = [Path(path).resolve() for path in (args.unity_capture_dir or [])]
    capture_dirs += default_capture_dirs(repo_root)
    capture_dirs = list(dict.fromkeys(capture_dirs))
    rows = load_episodes(results_dir / "episodes.jsonl")
    selections = select_episodes(rows, args.metric, args.per_outcome, args.objects, args.profiles)
    if not selections:
        raise ValueError("No evaluated object/profile groups match the requested filters")
    video_root = results_dir / "videos" / args.metric
    video_root.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "metric": args.metric,
                "per_outcome": args.per_outcome, "selection": [], "videos": [],
                "capture_dirs": [str(path) for path in capture_dirs]}
    old_index = video_root / "index.json"
    previous = {}
    if old_index.exists():
        try:
            previous_manifest = json.loads(old_index.read_text(encoding="utf-8"))
            retained_selections, retained_videos = retain_unselected_groups(previous_manifest, selections, rows)
            manifest["selection"].extend(retained_selections)
            manifest["videos"].extend(retained_videos)
            if not args.overwrite:
                previous = {entry["episode_id"]: entry for entry in previous_manifest.get("videos", [])}
        except (ValueError, KeyError):
            previous = {}
    error_count = sum(not entry.get("verified", False) for entry in manifest["videos"])
    for selection in selections:
        selected_rows = selection["selected"]
        selection_info = {key: value for key, value in selection.items() if key != "selected"}
        selection_info["episode_ids"] = [row["episode_id"] for row in selected_rows]
        manifest["selection"].append(selection_info)
        output_dir = video_root / safe_component(selection["object"]) / safe_component(
            selection["profile"]) / selection["outcome"]
        output_dir.mkdir(parents=True, exist_ok=True)
        for row in selected_rows:
            stem = safe_component(row["episode_id"])
            output_path = output_dir / (stem + ".mp4")
            replay_path = output_dir / (stem + ".replay.json")
            log_path = output_dir / (stem + ".log")
            entry = {"episode_id": row["episode_id"], "object": row["object"],
                     "profile": row["profile"], "outcome": selection["outcome"],
                     "paper_success": row["paper_success"], "public_success": row["public_success"],
                     "relative_path": output_path.relative_to(video_root).as_posix(), "verified": False}
            for key in ("ood_goal_success", "strict_goal_success"):
                if key in row:
                    entry[key] = row[key]
            entry["episode_sha256"] = hashlib.sha256(json.dumps(
                row, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
            trajectory_path = results_dir / "trajectories" / (row["episode_id"] + ".npz")
            if trajectory_path.is_file():
                entry["trajectory_sha256"] = sha256_file(trajectory_path)
            try:
                prior = previous.get(row["episode_id"], {})
                if (not args.overwrite and prior.get("verified") is True
                        and prior.get("relative_path") == entry["relative_path"]
                        and prior.get("episode_sha256") == entry["episode_sha256"]
                        and prior.get("trajectory_sha256") == entry.get("trajectory_sha256")
                        and output_path.is_file() and replay_path.is_file()):
                    verify_replay(row, json.loads(replay_path.read_text(encoding="utf-8")))
                    if sha256_file(output_path) != prior.get("sha256"):
                        raise ValueError("Previously recorded MP4 checksum changed; use --overwrite to replace")
                    entry.update(probe_video(output_path, ffprobe))
                    entry.update({"verified": True, "sha256": prior["sha256"], "reused": True})
                    print("Reuse verified video: " + row["episode_id"], flush=True)
                else:
                    basename = "ood_{}_{}.mp4".format(stem[:75], uuid.uuid4().hex[:12])
                    command = [sys.executable, str(repo_root / "tools" / "ood_eval.py"), "replay",
                               "--results", str(results_dir), "--episode-id", row["episode_id"],
                               "--record-video", basename, "--replay-result", str(replay_path)]
                    print("Record {} ({}/{}/{})".format(row["episode_id"], row["object"],
                                                           row["profile"], selection["outcome"]), flush=True)
                    # A fresh replay must produce a fresh verification record.
                    if replay_path.exists():
                        replay_path.unlink()
                    with log_path.open("w", encoding="utf-8") as log:
                        returncode = run_replay(command, repo_root, log, args.replay_timeout)
                    if returncode:
                        raise RuntimeError("Replay exited with code {}; see {}".format(returncode, log_path))
                    replay = json.loads(replay_path.read_text(encoding="utf-8"))
                    verify_replay(row, replay)
                    expected = replay.get("simulation_seconds", 0)
                    minimum_duration = max(0.1, float(expected) * 0.75)
                    source_path, video_info = wait_for_capture(
                        capture_dirs, basename, args.capture_timeout, ffprobe, minimum_duration)
                    temporary = output_path.with_suffix(".mp4.tmp")
                    shutil.copyfile(str(source_path), str(temporary))
                    os.replace(str(temporary), str(output_path))
                    entry.update(video_info)
                    entry.update({"verified": True, "sha256": sha256_file(output_path),
                                  "capture_source": str(source_path), "reused": False})
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                error_count += 1
                entry["error"] = str(error)
                print("ERROR {}: {}".format(row["episode_id"], error), file=sys.stderr, flush=True)
            manifest["videos"].append(entry)
            manifest["capture_errors"] = error_count
            write_json(video_root / "index.json", manifest)
            (video_root / "index.md").write_text(markdown_index(manifest), encoding="utf-8")
    # Also write indexes when every selected bin is empty (e.g. all episodes invalid).
    manifest["capture_errors"] = error_count
    write_json(video_root / "index.json", manifest)
    (video_root / "index.md").write_text(markdown_index(manifest), encoding="utf-8")
    print("Video index: " + str(video_root / "index.md"), flush=True)
    return 1 if error_count else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="OOD run directory containing episodes.jsonl")
    parser.add_argument("--per-outcome", type=int, default=5,
                        help="Maximum success and failure examples per object/profile (default: 5)")
    parser.add_argument("--metric", choices=METRICS, default=None,
                        help="Success definition; default: plan video_metric, or paper_success for legacy runs")
    parser.add_argument("--objects", nargs="+", help="Optional object name filter")
    parser.add_argument("--profiles", nargs="+", help="Optional target profile filter")
    parser.add_argument("--unity-capture-dir", action="append", default=[],
                        help="Unity's MP4 output directory; repeat to search multiple locations")
    parser.add_argument("--capture-timeout", type=float, default=30,
                        help="Seconds to wait for Unity to finish an MP4 after replay")
    parser.add_argument("--replay-timeout", type=float, default=600,
                        help="Maximum wall time per real-time replay in seconds")
    parser.add_argument("--overwrite", action="store_true", help="Re-record previously verified videos")
    args = parser.parse_args(argv)
    if (args.per_outcome < 1 or not math.isfinite(args.capture_timeout)
            or not math.isfinite(args.replay_timeout)
            or args.capture_timeout <= 0 or args.replay_timeout <= 0):
        parser.error("Counts and timeouts must be positive")
    try:
        return record_selected(args)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, "error: {}\n".format(error))


if __name__ == "__main__":
    sys.exit(main())
