#!/usr/bin/env python3
"""Plan, run, resume, replay and summarize ArtiGrasp OOD experiments.

Run from any directory; no C++ changes or additional Python packages are needed
beyond the existing paper environment. See docs/OOD_FIVE.md.
"""

import argparse
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from ood_suite.io import (code_fingerprint, file_sha256, read_json, read_rows,
                          write_json)
from ood_suite.paths import configure_package_path, missing_extension_hint
from ood_suite.targets import build_plan, load_config


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs/ood_five.json"


def _positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _plan_options(parser):
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True, help="New experiment directory")
    parser.add_argument("--objects", nargs="+", help="Default: all ten released objects")
    parser.add_argument("--profiles", nargs="+", help="Target profiles; default uses the config's default_suite or all profiles")
    parser.add_argument("--suite", help="Named profile set (default: five requested OOD conditions)")
    parser.add_argument("--samples", type=_positive, help="Samples per object AND condition; default 100")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--split", choices=("test", "train"))


def parser_for_cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="Read-only check of existing Ubuntu runtime/assets")
    check.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    check.add_argument("--videos", action="store_true", help="Also check Linux, Unity and ffprobe")
    plan = commands.add_parser("plan", help="Generate targets/labels without starting the simulator")
    _plan_options(plan)
    run = commands.add_parser("run", help="Run numerical evaluation and generate report.md")
    _plan_options(run)
    run.add_argument("--resume", action="store_true", help="Continue existing immutable plan")
    run.add_argument("--batch-size", type=_positive)
    run.add_argument("--num-threads", type=_positive)
    run.add_argument("--device", choices=("cpu", "cuda"))
    run.add_argument("--record-videos", action="store_true", help="After evaluation, replay selected episodes in Unity")
    run.add_argument("--unity-capture-dir", type=Path, action="append", default=[])
    report = commands.add_parser("report", help="Rebuild report from saved measurements")
    report.add_argument("--results", type=Path, required=True)
    replay = commands.add_parser("replay", help="Replay saved actions and verify the original trajectory")
    replay.add_argument("--results", type=Path, required=True)
    replay.add_argument("--episode-id", required=True)
    replay.add_argument("--record-video", help="Unique basename.mp4 sent to RaiSimUnity (Linux)")
    replay.add_argument("--replay-result", type=Path, required=True)
    worker = commands.add_parser("worker", help="Internal: execute one simulator batch")
    worker.add_argument("--results", type=Path, required=True)
    worker.add_argument("--episode-ids", nargs="+", required=True)
    worker.add_argument("--num-threads", type=_positive)
    worker.add_argument("--device", choices=("cpu", "cuda"))
    return parser


def check_runtime(config, videos=False, verbose=True):
    errors, versions = [], {}
    package_ready = True
    try:
        configure_package_path(REPO_ROOT)
        package = importlib.import_module("raisimGymTorch")
        versions["raisimGymTorch_path"] = str(Path(package.__file__).resolve())
    except Exception as error:
        package_ready = False
        errors.append("raisimGymTorch package: {}".format(error))
    for name in ("numpy", "torch", "ruamel.yaml", "raisimGymTorch.env.bin.compose_eval"):
        if name.endswith("compose_eval") and not package_ready:
            continue
        try:
            module = importlib.import_module(name)
            versions[name] = str(getattr(module, "__version__", "available"))
            if name.endswith("compose_eval"):
                versions["compose_eval_path"] = str(Path(module.__file__).resolve())
                versions["compose_eval_sha256"] = file_sha256(module.__file__)
        except Exception as error:
            errors.append("{}: {}".format(name, error))
            if (name.endswith("compose_eval") and isinstance(error, ModuleNotFoundError)
                    and error.name in ("raisimGymTorch.env", "raisimGymTorch.env.bin",
                                       "raisimGymTorch.env.bin.compose_eval")):
                errors.append(missing_extension_hint(REPO_ROOT))
    for name, filename in config.get("checkpoints", {}).items():
        if not (REPO_ROOT / filename).is_file():
            errors.append("Missing checkpoint {}: {}".format(name, filename))
    if set(config.get("checkpoints", {})) != {"cooperative_left", "cooperative_right", "articulation_right"}:
        errors.append("checkpoints must define cooperative_left, cooperative_right and articulation_right")
    for name in config.get("objects", []):
        for path in (
                REPO_ROOT / "rsc/arctic" / name / (name + ".urdf"),
                REPO_ROOT / "rsc/mano_double/grasp_label/final_labels" /
                (config.get("split", "test") + "_set") / (name + ".npy")):
            if not path.is_file():
                errors.append("Missing asset: {}".format(path))
    if videos:
        import shutil
        if platform.system() != "Linux":
            errors.append("Native Unity video recording requires Linux")
        if not (REPO_ROOT / "raisimUnity/linux/raisimUnity.x86_64").is_file():
            errors.append("Missing bundled Linux Unity executable")
        if shutil.which("ffprobe") is None:
            errors.append("ffprobe not found (provided by the existing ffmpeg installation)")
    if verbose:
        print("Python {} ({})".format(platform.python_version(), sys.executable))
        for name, version in versions.items():
            print("  {}: {}".format(name, version))
        for error in errors:
            print("  ERROR: " + error, file=sys.stderr)
        print("Runtime check: {}".format("FAILED" if errors else "OK"), flush=True)
    return errors, versions


def _asset_hashes(plan):
    """Hash concrete input files used by the sampled labels for resume checks."""
    paths = set()
    for spec in plan["episodes"]:
        for label in (spec["grasp_label"], spec["arti_label"]):
            subject, sequence, obj = label["seq_num"], label["seq_name"], spec["object"]
            base = "rsc/mano_double"
            paths.add("{}/s{}/{}.mano.npy".format(base, subject, sequence))
            paths.add("rsc/arctic/{}/s{}/{}.object.npy".format(obj, subject, sequence))
            for suffix in ("r", "l", "min"):
                paths.add("{}/ee/s{}/{}_{}.npy".format(base, subject, sequence, suffix))
            for hand in ("r", "l"):
                paths.add("{}/contact/s{}/{}_a_{}.npy".format(base, subject, sequence, hand))
        paths.add("rsc/arctic/{0}/{0}.urdf".format(spec["object"]))
    paths.update("rsc/mano_double/" + name for name in (
        "right_pose_mean.txt", "left_pose_mean.txt", "rhand_mano_demo.urdf", "lhand_mano_demo.urdf"))
    paths.update(plan["config"]["checkpoints"].values())
    result = {}
    for relative in sorted(paths):
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError("Missing input for selected labels: {}".format(path))
        result[relative] = file_sha256(path)
    return result


def create_plan(args):
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output is not empty: {}. Use run --resume or a new directory.".format(output))
    config = load_config(args.config)
    plan = build_plan(REPO_ROOT, config, objects=args.objects, profiles=args.profiles,
                      samples=args.samples, seed=args.seed, split=args.split,
                      suite=getattr(args, "suite", None))
    constraints = None
    if config.get("evaluation_protocol") == "five_v1":
        from ood_suite.constraints import validate_plan
        constraints = validate_plan(plan, REPO_ROOT)
    for option, name in (("batch_size", "batch_size"), ("num_threads", "num_threads"), ("device", "device")):
        if getattr(args, option, None) is not None:
            plan["config"].setdefault("runtime", {})[name] = getattr(args, option)
    # Hash before creating files so incomplete input cannot leave a misleading plan.
    provenance = code_fingerprint(REPO_ROOT)
    provenance["input_sha256"] = _asset_hashes(plan)
    if constraints is not None:
        provenance["input_sha256"].update(constraints["mesh_sha256"])
    provenance["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(output / "plan.json", plan)
    if constraints is not None:
        write_json(output / "physics_constraints.json", constraints)
    provenance["plan_sha256"] = file_sha256(output / "plan.json")
    write_json(output / "provenance.json", provenance)
    print("Plan: {} episodes -> {}".format(len(plan["episodes"]), output / "plan.json"), flush=True)
    print("{} objects x {} conditions x {} samples | suite: {}".format(
        len(plan["objects"]), len(plan["profiles"]), plan["samples_per_object"],
        plan["config"].get("selected_suite") or "explicit/legacy"), flush=True)
    return plan


def verify_provenance(results_dir):
    provenance = read_json(Path(results_dir) / "provenance.json")
    if provenance.get("plan_sha256") != file_sha256(Path(results_dir) / "plan.json"):
        raise RuntimeError("plan.json changed since creation. Use a new experiment directory.")
    current = code_fingerprint(REPO_ROOT)
    if provenance["source_sha256"] != current["source_sha256"]:
        raise RuntimeError("Evaluator/source changed since this plan was generated. "
                           "Use a new output directory to avoid mixing implementations.")
    for relative, expected in provenance["input_sha256"].items():
        path = REPO_ROOT / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError("Experiment input changed or disappeared: " + relative)


def _load_or_create(args):
    output = args.output.resolve()
    if not args.resume:
        return create_plan(args)
    plan = read_json(output / "plan.json")
    # Resume consumes exactly the existing targets, irrespective of random state.
    for option, config_key in (("samples", "samples_per_object"), ("seed", "seed"), ("split", "split")):
        value = getattr(args, option)
        if value is not None and value != plan["config"][config_key]:
            raise ValueError("--{} differs from the saved plan".format(option))
    if args.objects is not None and args.objects != plan["config"]["objects"]:
        raise ValueError("--objects differs from the saved plan")
    if args.profiles is not None and args.profiles != list(plan["config"]["profiles"]):
        raise ValueError("--profiles differs from the saved plan")
    if getattr(args, "suite", None) is not None and args.suite != plan["config"].get("selected_suite"):
        raise ValueError("--suite differs from the saved plan")
    verify_provenance(output)
    return plan


def run_experiment(args):
    from ood_suite.report import generate_report
    plan = _load_or_create(args)
    output = args.output.resolve()
    errors, versions = check_runtime(plan["config"], args.record_videos)
    if errors:
        raise RuntimeError("Existing runtime is not ready; see check output above.")
    execution_path = output / "execution.json"
    if execution_path.exists():
        original = read_json(execution_path)
        if _runtime_identity(original["runtime"]) != _runtime_identity(versions):
            raise RuntimeError("Runtime/binary differs from previous batches; use the original environment or a new run")
        if original["device"] != (args.device or plan["config"].get("runtime", {}).get("device", "cpu")):
            raise RuntimeError("Policy device differs from previous batches; use a new run for a different device")
    else:
        write_json(execution_path, {"runtime": versions, "platform": platform.platform(),
                                   "python": sys.version, "interpreter": sys.executable,
                                   "device": args.device or plan["config"].get("runtime", {}).get("device", "cpu"),
                                   "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    rows = read_rows(output / "episodes.jsonl")
    complete = {row["episode_id"] for row in rows}
    all_ids = {spec["episode_id"] for spec in plan["episodes"]}
    if complete - all_ids:
        raise ValueError("episodes.jsonl contains trials absent from plan.json")
    for row in rows:
        if not (output / row["trajectory_path"]).is_file():
            raise RuntimeError("Completed result has no saved trajectory: " + row["episode_id"])
    settings = plan["config"].get("runtime", {})
    batch_size = args.batch_size or settings.get("batch_size", 100)
    if batch_size <= 0:
        raise ValueError("runtime.batch_size must be positive")
    print("Requested policy inference device: {} | RaiSim physics: CPU | threads: {} | batch size: {}".format(
        args.device or settings.get("device", "cpu"),
        args.num_threads or settings.get("num_threads", 4), batch_size), flush=True)
    remaining = [spec for spec in plan["episodes"] if spec["episode_id"] not in complete]
    groups = {}
    for spec in remaining:
        groups.setdefault((spec["object"], spec["profile"]), []).append(spec["episode_id"])
    failure = None
    try:
        for (object_name, profile), ids in groups.items():
            for start in range(0, len(ids), batch_size):
                batch = ids[start:start + batch_size]
                print("{} / {} / {} trials ({} already completed)".format(
                    object_name, profile, len(batch), len(complete)), flush=True)
                command = [sys.executable, str(Path(__file__).resolve()), "worker", "--results", str(output),
                           "--episode-ids"] + batch
                for option, value in (("--num-threads", args.num_threads), ("--device", args.device)):
                    if value is not None:
                        command.extend([option, str(value)])
                environment = os.environ.copy()
                environment.setdefault("OMP_NUM_THREADS", str(args.num_threads or settings.get("num_threads", 4)))
                subprocess.run(command, cwd=str(REPO_ROOT), env=environment, check=True)
                complete.update(batch)
    except (subprocess.CalledProcessError, KeyboardInterrupt) as error:
        failure = error
    finally:
        generated = generate_report(output)
        print("Report: {}".format(generated), flush=True)
    if failure is not None:
        raise RuntimeError("Evaluation interrupted/failed; completed trials are saved. Resume with --resume.") from failure
    if args.record_videos:
        command = [sys.executable, str(REPO_ROOT / "tools/ood_record.py"), "--results", str(output),
                   "--per-outcome", str(plan["config"].get("videos_per_outcome", 5)),
                   "--metric", plan["config"].get("video_metric", "paper_success")]
        for directory in args.unity_capture_dir:
            command.extend(["--unity-capture-dir", str(directory.resolve())])
        try:
            subprocess.run(command, cwd=str(REPO_ROOT), check=True)
        finally:
            generate_report(output)


def main(argv=None):
    args = parser_for_cli().parse_args(argv)
    try:
        if args.command == "check":
            return 1 if check_runtime(load_config(args.config), args.videos)[0] else 0
        if args.command == "plan":
            create_plan(args)
        elif args.command == "run":
            run_experiment(args)
        elif args.command == "report":
            from ood_suite.report import generate_report
            print(generate_report(args.results.resolve()))
        elif args.command == "worker":
            from ood_suite.runtime import run_worker
            run_worker(REPO_ROOT, args.results.resolve(), args.episode_ids,
                       device=args.device, threads=args.num_threads)
        elif args.command == "replay":
            from ood_suite.runtime import replay_episode
            # Saved actions and initialization remove checkpoint/data dependence,
            # but replay must use the same evaluator and simulator binary.
            provenance = read_json(args.results / "provenance.json")
            if provenance.get("plan_sha256") != file_sha256(args.results / "plan.json"):
                raise RuntimeError("plan.json changed since creation")
            if provenance["source_sha256"] != code_fingerprint(REPO_ROOT)["source_sha256"]:
                raise RuntimeError("Source changed since evaluation; restore it before replaying")
            plan = read_json(args.results / "plan.json")
            errors, versions = check_runtime(plan["config"], bool(args.record_video), verbose=False)
            if errors:
                raise RuntimeError("; ".join(errors))
            execution = read_json(args.results / "execution.json")
            if _runtime_identity(execution["runtime"]) != _runtime_identity(versions):
                raise RuntimeError("Replay runtime/binary differs from the evaluated runtime")
            result = replay_episode(REPO_ROOT, args.results.resolve(), args.episode_id, args.record_video)
            write_json(args.replay_result, result)
            print(json.dumps(result, ensure_ascii=False, allow_nan=False))
            return 0 if result["verified"] else 2
        return 0
    except (Exception, KeyboardInterrupt) as error:
        if os.environ.get("ARTIGRASP_OOD_DEBUG"):
            raise
        print("ERROR: {}".format(error or "Interrupted"), file=sys.stderr)
        return 1


def _runtime_identity(versions):
    return {name: value for name, value in versions.items() if not name.endswith("_path")}


if __name__ == "__main__":
    sys.exit(main())
