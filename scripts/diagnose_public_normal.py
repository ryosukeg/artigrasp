#!/usr/bin/env python3
"""Isolated reproduction of released composed-controller target conventions.

This diagnostic does not modify the OOD evaluator or its historical artifacts.
Goals follow compose_eval.py / ketchup_eval.py, including intrinsic Euler-Y
perturbation and ketchup's unchanged relative-angle observation at step 1300.
Label pairs use the existing deterministic Monte Carlo sampler; they are not
the original exhaustive Cartesian product, and original random goal draws are
not available. Invalid numerical trials count as failures in both metrics.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from ood_suite.io import code_fingerprint, file_sha256, read_json, write_json
from ood_suite.targets import PAPER_OBJECTS, build_plan


def configuration(args):
    config = copy.deepcopy(read_json(ROOT / "configs/normal_five.json"))
    config.pop("normal_goal", None)
    config.pop("physics_validation", None)
    config.update(evaluation_protocol="literal_public_diagnostic_v1",
                  default_suite="public_normal", suites={"public_normal": ["public_normal"]},
                  profiles={"public_normal": {}}, objects=args.objects,
                  samples_per_object=args.samples, seed=args.seed, split=args.split)
    config["runtime"].update(batch_size=args.samples, num_threads=args.num_threads,
                             device=args.device)
    return config


def literal_controller_class():
    from ood_suite.runtime import Controller

    class LiteralPublicController(Controller):
        def phase(self, step):
            g, t = self.timing["grasp_steps"], self.timing["transit_steps"]
            if self.object == "ketchup" and step == g + 2 * t:
                # Literal public ketchup_eval.py only changes object goal here.
                # set_obj_goal does not update final_obj_angle_ used by the actor.
                self.env.set_obj_goal(self.angle, self.goal)
                self.constants(12, 11)
                return
            super().phase(step)

    return LiteralPublicController


def summarize(rows):
    from ood_suite.metrics import wilson_interval

    def group(items):
        n = len(items)
        result = {"trials": n, "valid": sum(bool(r["valid"]) for r in items)}
        for key in ("public_success", "paper_success", "public_arrived", "public_opened",
                    "public_orientation_reached", "paper_position_success",
                    "paper_orientation_success", "paper_articulation_success"):
            successes = sum(bool(r.get(key, False)) and r["valid"] for r in items)
            result[key] = {"successes": successes, "rate": successes / n if n else None,
                           "wilson_95": wilson_interval(successes, n)}
        for key in ("paper_position_error_m", "paper_orientation_error_rad",
                    "paper_articulation_error_rad", "public_position_error_clipped_m",
                    "public_orientation_error_clipped_rad", "public_articulation_error_clipped_rad"):
            values = [r[key] for r in items if r["valid"] and r.get(key) is not None]
            result[key] = sum(values) / len(values) if values else None
        return result

    by_object = {name: group([r for r in rows if r["object"] == name])
                 for name in dict.fromkeys(r["object"] for r in rows)}
    return {"overall": group(rows), "by_object": by_object,
            "primary_metric": "public_success (literal released code)",
            "physical_final_metric": "paper_success (PE<0.05m, SO3 AE<0.2rad, AAE<0.5rad)",
            "lift_is_not_required_by_either_reported_metric": True,
            "objects_macro_public_success": (
                sum(v["public_success"]["rate"] for v in by_object.values()) / len(by_object)
                if by_object else None)}


def worker(args):
    from ood_suite.runtime import (Policies, load_environment_config, make_environment,
                                   prepare_data, simulate, timing_for, _save_trajectory)
    from ood_suite.metrics import evaluate_episode
    import numpy as np

    out = args.output.resolve()
    plan = read_json(out / "plan.json")
    specs = [s for s in plan["episodes"] if s["object"] == args.worker_object]
    if not specs:
        raise RuntimeError("No planned episodes for " + args.worker_object)
    started = time.perf_counter()
    data = prepare_data(ROOT, specs)
    config = load_environment_config(ROOT, len(specs), False, args.num_threads)
    env = make_environment(ROOT, data, config, args.worker_object)
    try:
        env.seed(plan["seed"])
        policy = Policies(ROOT, plan["config"]["checkpoints"], config, env, args.device, plan["seed"])
        timing = timing_for(args.worker_object)
        controller = literal_controller_class()(env, data, specs, timing, policy)
        performance = {}
        states, right, left, invalid, first_invalid = simulate(env, controller, policy, timings=performance)
        rows = []
        for i, spec in enumerate(specs):
            metric_spec = controller.metric_spec(i)
            row = copy.deepcopy(metric_spec)
            row.update(evaluate_episode(states[:, i], metric_spec, timing))
            if invalid[i]:
                row.update(valid=False, paper_success=False, public_success=False,
                           invalid_reason="nonfinite_or_simulator_auto_reset",
                           first_invalid_step=int(first_invalid[i]))
                for key in row:
                    if "_error_" in key:
                        row[key] = None
            row.update(timing=timing, environment_config=config, policy_device=str(policy.device),
                       controller_changes_from_public=[],
                       controller_difference_from_ood_runtime=(
                           ["suppress_ketchup_forward_requested_articulation"]
                           if spec["object"] == "ketchup" else []))
            if args.save_all_trajectories or i < args.save_trajectories:
                relative = Path("trajectories") / (spec["episode_id"] + ".npz")
                _save_trajectory(out / relative, i, states, right, left, data)
                row.update(trajectory_path=str(relative), trajectory_sha256=file_sha256(out / relative))
            rows.append(row)
        module = sys.modules["raisimGymTorch.env.bin.compose_eval"]
        info = {"object": args.worker_object, "worker_pid": os.getpid(),
                "wall_s": time.perf_counter() - started, "rollout": performance,
                "policy_device": str(policy.device), "torch_cpu_threads": policy.torch.get_num_threads(),
                "numpy_version": np.__version__, "torch_version": policy.torch.__version__,
                "interpreter": sys.executable, "python": sys.version, "platform": platform.platform(),
                "compose_eval_path": module.__file__, "compose_eval_sha256": file_sha256(module.__file__)}
        write_json(out / "batches" / (args.worker_object + ".json"), {"rows": rows, "execution": info})
        print(json.dumps({"object": args.worker_object, **summarize(rows)["overall"]}, indent=2), flush=True)
    finally:
        env.close()


def aggregate(out):
    plan = read_json(out / "plan.json")
    rows, executions = [], []
    for name in plan["objects"]:
        batch = out / "batches" / (name + ".json")
        if batch.exists():
            value = read_json(batch)
            rows.extend(value["rows"])
            executions.append(value["execution"])
    temporary = out / "episodes.jsonl.tmp"
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(str(temporary), str(out / "episodes.jsonl"))
    write_json(out / "summary.json", summarize(rows))
    write_json(out / "execution.json", {"batches": executions})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("test", "train"), default="test")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--objects", nargs="+", default=list(PAPER_OBJECTS))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-threads", type=int, default=24)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--save-trajectories", type=int, default=3,
                        help="Save first N trajectories per object (default 3)")
    parser.add_argument("--save-all-trajectories", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--worker-object", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.samples < 1 or args.num_threads < 1 or args.save_trajectories < 0:
        parser.error("samples/threads must be positive and save-trajectories nonnegative")
    if args.worker_object:
        worker(args)
        return
    out = args.output.resolve()
    config = configuration(args)
    plan = build_plan(ROOT, config)
    script_hash = file_sha256(Path(__file__))
    if out.exists() and any(out.iterdir()):
        if not args.resume:
            raise RuntimeError("Output is nonempty; use a new directory or --resume")
        if read_json(out / "plan.json") != plan:
            raise RuntimeError("Resume arguments differ from the saved immutable plan")
        if read_json(out / "provenance.json")["diagnostic_script_sha256"] != script_hash:
            raise RuntimeError("Diagnostic script changed since experiment creation")
    else:
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "plan.json", plan)
        provenance = code_fingerprint(ROOT)
        provenance.update(diagnostic_script=str(Path(__file__).resolve().relative_to(ROOT)),
                          diagnostic_script_sha256=script_hash,
                          target_convention="released scripts, not paper-intent world yaw",
                          label_sampling="same deterministic build_plan Monte Carlo sampler; not exhaustive",
                          unavailable_original_goal_draws=True,
                          original_ketchup_relative_angle_observation_preserved=True)
        provenance["checkpoint_sha256"] = {key: file_sha256(ROOT / value)
                                             for key, value in config["checkpoints"].items()}
        write_json(out / "provenance.json", provenance)
    if args.plan_only:
        print("Saved {} episodes to {}".format(len(plan["episodes"]), out / "plan.json"))
        return
    for name in plan["objects"]:
        batch = out / "batches" / (name + ".json")
        if batch.exists():
            continue
        command = [sys.executable, str(Path(__file__)), "--output", str(out),
                   "--worker-object", name, "--num-threads", str(args.num_threads),
                   "--device", args.device, "--save-trajectories", str(args.save_trajectories)]
        if args.save_all_trajectories:
            command.append("--save-all-trajectories")
        subprocess.run(command, cwd=str(ROOT), check=True)
        aggregate(out)
    rows = aggregate(out)
    print(json.dumps(summarize(rows)["overall"], indent=2))


if __name__ == "__main__":
    main()
