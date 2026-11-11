#!/usr/bin/env python3
"""Plan or run the three-stage ArtiGrasp baseline without public policy weights.

Planning uses the standard library only. Training is opt-in with --execute.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/training/scratch_baseline_v1.json"
STAGE_ORDER = ("left_fixed", "multi_obj_arti", "general_two")
HANDS = {"left_fixed": ["left"], "multi_obj_arti": ["right"],
         "general_two": ["left", "right"]}
PAPER_OBJECTS = ("box", "laptop", "notebook", "waffleiron", "mixer", "ketchup",
                 "espressomachine", "microwave")
LEFT_INPUT = "@LEFT_PRETRAIN_FINAL@"
RIGHT_INPUT = "@RIGHT_PRETRAIN_FINAL@"
TRAINING_PACKAGE = ROOT / "raisimGymTorch/raisimGymTorch"
SHARED_TRAINING_SOURCES = (
    "helper/training_runtime.py", "helper/training_support.py", "helper/label_gen_final.py",
    "helper/rotations.py", "helper/all_frame_dict.py", "algo/ppo/module.py",
    "algo/ppo/ppo.py", "algo/ppo/storage.py", "env/RaisimGymVecEnv.py",
    "env/VectorizedEnvironment.hpp", "env/RaisimGymEnv.hpp", "env/Reward.hpp")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive_integer(value, field):
    require(type(value) is int and value > 0, field + " must be a positive integer")
    return value


def project_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def load_config(path=DEFAULT_CONFIG):
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text())
    require(config.get("version") == 1, "Unsupported training configuration version")
    require(config.get("split") == "train", "Scratch training requires the train split")
    objects = config.get("objects")
    require(isinstance(objects, list) and objects and len(objects) == len(set(objects)),
            "Training objects must be a nonempty list without duplicates")
    require(set(objects) <= set(PAPER_OBJECTS), "Training contains a non-paper object")
    stages = config.get("stages", [])
    require([s.get("id") for s in stages] == list(STAGE_ORDER),
            "Stages must be left_fixed, multi_obj_arti, general_two in this order")
    for stage in stages:
        positive_integer(stage.get("iterations"), stage["id"] + ".iterations")
        positive_integer(stage.get("environments"), stage["id"] + ".environments")
        require(stage.get("required_hands") == HANDS[stage["id"]],
                "Incorrect expected checkpoints for " + stage["id"])
        divisor = len(objects) * (1 if stage["id"] == "left_fixed" else 2)
        require(stage["environments"] % divisor == 0,
                stage["id"] + ": environments must divide evenly across objects and tasks")
    runtime = config.get("runtime", {})
    require(runtime.get("device") in ("auto", "cpu", "cuda"), "Invalid training device")
    positive_integer(runtime.get("num_threads"), "runtime.num_threads")
    positive_integer(runtime.get("torch_threads"), "runtime.torch_threads")
    require(runtime["torch_threads"] == runtime["num_threads"],
            "RaiSim and PyTorch share OpenMP; use the same physics and torch thread counts")
    require(runtime.get("headless") is True, "Pipeline requires headless training")
    require(runtime.get("observation_normalization") is False,
            "This baseline requires observation normalization off")
    require(runtime.get("torch_deterministic") is True,
            "This baseline requires deterministic PyTorch operations")
    return config


def build_plan(config, config_path, output, seed=1, smoke=False, python=sys.executable,
               eval_seed=1, eval_samples_per_object=100):
    require(type(seed) is int and 0 <= seed < 2 ** 32, "seed must be in [0, 2**32)")
    require(type(eval_seed) is int and 0 <= eval_seed < 2 ** 32,
            "eval seed must be in [0, 2**32)")
    positive_integer(eval_samples_per_object, "eval samples per object")
    output = Path(output).expanduser().resolve()
    interpreter = Path(python).expanduser().resolve()
    require(interpreter.is_file() and os.access(interpreter, os.X_OK),
            "Python interpreter is not executable: " + str(interpreter))
    objects = config["objects"]
    runtime = deepcopy(config["runtime"])
    if smoke:
        runtime.update(device="cpu", num_threads=1, torch_threads=1)
    stages = []
    source_hashes = {str(Path(config_path).resolve()): sha256(config_path),
                     str(Path(__file__).resolve()): sha256(__file__)}
    for relative in SHARED_TRAINING_SOURCES:
        source = TRAINING_PACKAGE / relative
        require(source.is_file(), "Missing shared training source: " + str(source))
        source_hashes[str(source)] = sha256(source)
    for spec in config["stages"]:
        stage = deepcopy(spec)
        name = stage["id"]
        cfg = project_path(stage["cfg"])
        runner = TRAINING_PACKAGE / "env/envs" / name / "runner.py"
        environment_source = runner.parent / "Environment.hpp"
        extensions = sorted((TRAINING_PACKAGE / "env/bin").glob(name + "*.so"))
        require(cfg.is_file(), "Missing stage configuration: " + str(cfg))
        require(runner.is_file(), "Missing stage runner: " + str(runner))
        require(environment_source.is_file(), "Missing stage environment source: " + str(environment_source))
        require(len(extensions) == 1 and extensions[0].is_file(),
                "Expected exactly one built extension for {} under {}; found {}".format(
                    name, TRAINING_PACKAGE / "env/bin", len(extensions)))
        source_hashes[str(cfg)] = sha256(cfg)
        source_hashes[str(runner)] = sha256(runner)
        source_hashes[str(environment_source)] = sha256(environment_source)
        source_hashes[str(extensions[0])] = sha256(extensions[0])
        if smoke:
            stage["iterations"] = 1
            stage["environments"] = len(objects) * (1 if name == "left_fixed" else 2)
        stage.update(cfg=str(cfg), runner=str(runner), output_dir=str(output / name),
                     seed=seed, objects=list(objects))
        command = [str(interpreter), str(runner), "--cfg", str(cfg),
                   "--seed", str(seed), "--num-threads", str(runtime["num_threads"]),
                   "--torch-threads", str(runtime["torch_threads"]), "--headless",
                   "--device", runtime["device"], "--output-dir", stage["output_dir"],
                   "--num_iterations", str(stage["iterations"]),
                   "--num_repeats", str(stage["environments"]), "--objects", *objects]
        if name == "general_two":
            command += ["--pretrained-left", LEFT_INPUT, "--pretrained-right", RIGHT_INPUT]
        stage["command"] = command
        stage["environment_steps"] = stage["iterations"] * stage["environments"] * 300
        stages.append(stage)
    template = project_path(config["evaluation_template"])
    require(template.is_file(), "Missing evaluation template: " + str(template))
    source_hashes[str(template)] = sha256(template)
    return {
        "version": 1, "baseline_id": config["baseline_id"], "smoke": bool(smoke),
        "training_seed": seed, "split": "train", "objects": list(objects),
        "runtime": runtime, "output_dir": str(output), "project_root": str(ROOT),
        "config_path": str(Path(config_path).resolve()), "python": str(interpreter),
        "stages": stages, "source_sha256": source_hashes,
        "total_environment_steps": sum(s["environment_steps"] for s in stages),
        "evaluation": {"template": str(template), "seed": eval_seed,
                       "samples_per_object": eval_samples_per_object,
                       "output_config": str(output / "normal_evaluation.json")},
        "notes": ["No files are created and no training starts unless --execute is set.",
                  "The cooperative stage uses final checkpoints returned by this run's pretraining stages.",
                  "The schedule freezes repository defaults; it does not establish convergence.",
                  "Resume is unsupported; use a new output directory for each run."]}


def read_stage_result(stage):
    directory = Path(stage["output_dir"])
    result_file = directory / "training_result.json"
    require(result_file.is_file(), "Stage did not write completion record: " + str(result_file))
    result = json.loads(result_file.read_text())
    require(result.get("complete") is True, "Stage is incomplete: " + stage["id"])
    require(result.get("stage") == stage["id"],
            "Stage completion record identifies a different stage: " + stage["id"])
    require(type(result.get("seed")) is int and result["seed"] == stage["seed"],
            "Stage seed does not match its plan: " + stage["id"])
    require(result.get("objects") == stage["objects"],
            "Stage objects do not match its plan: " + stage["id"])
    require(type(result.get("num_envs")) is int and result["num_envs"] == stage["environments"],
            "Stage environment count does not match its plan: " + stage["id"])
    require(result.get("checkpoint_timing") == "after_completed_ppo_updates",
            "Stage did not identify checkpoints saved after PPO updates: " + stage["id"])
    require(result.get("device") in ("cpu", "cuda"),
            "Stage did not identify its concrete device: " + stage["id"])
    require(type(result.get("iterations_completed")) is int
            and result["iterations_completed"] == stage["iterations"],
            "Stage iteration count does not match its plan: " + stage["id"])
    checkpoints = result.get("checkpoints", {})
    resolved = {}
    for hand in stage["required_hands"]:
        checkpoint = checkpoints.get(hand)
        require(isinstance(checkpoint, str) and checkpoint,
                "Missing final " + hand + " checkpoint: " + stage["id"])
        path = Path(checkpoint).expanduser()
        path = (path if path.is_absolute() else directory / path).resolve()
        require(path.is_file(), "Missing final checkpoint file: " + str(path))
        require(directory.resolve() in path.parents,
                "Final checkpoint must belong to the stage output: " + str(path))
        resolved[hand] = str(path)
    return {"result_file": str(result_file), "iterations_completed": result["iterations_completed"],
            "device": result.get("device"),
            "checkpoints": resolved,
            "checkpoint_sha256": {hand: sha256(path) for hand, path in resolved.items()}}


def run_stage(command, log_file, cwd, environment):
    with Path(log_file).open("w") as log:
        process = subprocess.Popen(command, cwd=cwd, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
            return process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            process.stdout.close()


def make_evaluation_config(plan, completed):
    template = plan["evaluation"]["template"]
    snapshot = plan.get("source_snapshots", {}).get(template)
    template_path = Path(plan["output_dir"]) / snapshot if snapshot else Path(template)
    if snapshot:
        require(sha256(template_path) == plan["source_sha256"][template],
                "Saved evaluation template snapshot changed: " + str(template_path))
    config = json.loads(template_path.read_text())
    config.update(objects=plan["objects"], split="test", seed=plan["evaluation"]["seed"],
                  samples_per_object=plan["evaluation"]["samples_per_object"])
    config["checkpoints"] = {
        "cooperative_left": completed["general_two"]["checkpoints"]["left"],
        "cooperative_right": completed["general_two"]["checkpoints"]["right"],
        "articulation_right": completed["multi_obj_arti"]["checkpoints"]["right"]}
    device = completed["general_two"].get("device") or plan["runtime"]["device"]
    # The evaluator accepts concrete devices only; CPU is a portable fallback for auto.
    if device == "auto":
        device = "cpu"
    config["runtime"].update(device=device,
                            num_threads=plan["runtime"]["num_threads"])
    if plan["smoke"]:
        config["runtime"]["batch_size"] = 1
    return config


def verify_plan_sources(plan):
    for name, digest in plan["source_sha256"].items():
        require(sha256(name) == digest, "Planned source changed before execution: " + name)


def snapshot_plan_sources(plan, output):
    snapshots = {}
    root = Path(plan["project_root"]).resolve()
    for name, digest in plan["source_sha256"].items():
        source = Path(name).resolve()
        try:
            relative = source.relative_to(root)
        except ValueError:
            path_digest = hashlib.sha256(str(source).encode()).hexdigest()[:16]
            relative = Path("_external") / path_digest / source.name
        relative = Path("pipeline_sources") / relative
        content = source.read_bytes()
        require(hashlib.sha256(content).hexdigest() == digest,
                "Planned source changed while saving snapshot: " + name)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as snapshot:
            snapshot.write(content)
        snapshots[name] = str(relative)
    return snapshots


def execute_plan(plan):
    plan = deepcopy(plan)
    output = Path(plan["output_dir"])
    require(not output.exists(), "Output already exists; refusing to overwrite: " + str(output))
    verify_plan_sources(plan)
    output.mkdir(parents=True, exist_ok=False)
    status = {"complete": False, "baseline_id": plan["baseline_id"], "smoke": plan["smoke"],
              "training_seed": plan["training_seed"], "started_at": datetime.now(timezone.utc).isoformat(),
              "stages": {}}
    completed = {}
    environment = os.environ.copy()
    environment.update(PYTHONUNBUFFERED="1", PYTHONHASHSEED=str(plan["training_seed"]),
                       CUBLAS_WORKSPACE_CONFIG=":4096:8",
                       OMP_NUM_THREADS=str(plan["runtime"]["num_threads"]),
                       MKL_NUM_THREADS=str(plan["runtime"]["torch_threads"]))
    write_json(output / "execution.json", status)
    try:
        plan["source_snapshots"] = snapshot_plan_sources(plan, output)
        write_json(output / "training_plan.json", plan)
        for stage in plan["stages"]:
            replacements = {}
            if stage["id"] == "general_two":
                replacements = {LEFT_INPUT: completed["left_fixed"]["checkpoints"]["left"],
                                RIGHT_INPUT: completed["multi_obj_arti"]["checkpoints"]["right"]}
            command = [replacements.get(token, token) for token in stage["command"]]
            status["active_stage"] = stage["id"]
            status["stages"][stage["id"]] = {"command": command, "complete": False}
            write_json(output / "execution.json", status)
            verify_plan_sources(plan)
            print("Starting " + stage["id"] + ": " + shlex.join(command), flush=True)
            code = run_stage(command, output / (stage["id"] + ".log"), plan["project_root"], environment)
            status["stages"][stage["id"]]["returncode"] = code
            require(code == 0, "Stage failed (exit {}): {}".format(code, stage["id"]))
            completed[stage["id"]] = read_stage_result(stage)
            status["stages"][stage["id"]].update(completed[stage["id"]], complete=True)
            write_json(output / "execution.json", status)
        verify_plan_sources(plan)
        config = make_evaluation_config(plan, completed)
        write_json(plan["evaluation"]["output_config"], config)
        status.update(complete=True, active_stage=None,
                      evaluation_config=plan["evaluation"]["output_config"],
                      finished_at=datetime.now(timezone.utc).isoformat())
        write_json(output / "execution.json", status)
        return status
    except BaseException as exc:
        status.update(error=str(exc), finished_at=datetime.now(timezone.utc).isoformat())
        write_json(output / "execution.json", status)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--python", default=sys.executable,
                        help="Python executable with the ArtiGrasp environment")
    parser.add_argument("--smoke", action="store_true", help="One small CPU iteration per stage")
    parser.add_argument("--execute", action="store_true", help="Create output and actually train")
    parser.add_argument("--eval-seed", type=int, default=1)
    parser.add_argument("--eval-samples-per-object", type=int, default=100)
    args = parser.parse_args(argv)
    try:
        config_path = args.config.expanduser().resolve()
        config = load_config(config_path)
        suffix = "_smoke" if args.smoke else ""
        output = args.output or ROOT / "results/training" / (
            "{}_seed{}{}".format(config["baseline_id"], args.seed, suffix))
        plan = build_plan(config, config_path, output, args.seed, args.smoke, args.python,
                          args.eval_seed, args.eval_samples_per_object)
        if args.execute:
            result = execute_plan(plan)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print("Scratch training error: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
