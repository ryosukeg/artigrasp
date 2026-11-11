"""Run the released composed-task controller without altering the C++ environment.

The phase schedule and control kinds follow compose_eval.py / ketchup_eval.py.
Only targets and selection of data labels change. Numerical trials are headless;
videos replay the saved actions and verify the resulting simulation trajectory.
"""

import contextlib
import io
import os
import platform
import random
import re
import time
from pathlib import Path

import numpy as np

from .io import append_row, file_sha256, read_json, read_rows, write_json
from .paths import configure_package_path


def timing_for(object_name):
    if object_name == "ketchup":
        return {"protocol": "ketchup", "grasp_steps": 300,
                "transit_steps": 500, "arti_steps": 300}
    return {"protocol": "standard", "grasp_steps": 260,
            "transit_steps": 500 if object_name in
            ("microwave", "notebook", "waffleiron") else 300,
            "arti_steps": 400}


def total_steps(timing):
    return (timing["grasp_steps"] + timing["arti_steps"] +
            (2 if timing["protocol"] == "ketchup" else 3) * timing["transit_steps"])


def _paths(repo_root):
    return configure_package_path(repo_root)


def _float(values):
    return np.ascontiguousarray(values, dtype=np.float32)


def prepare_data(repo_root, specs):
    _paths(repo_root)
    from raisimGymTorch.helper import label_gen_final
    # pose_gen is deterministic, but prints the first label on every call.
    with contextlib.redirect_stdout(io.StringIO()):
        grasp, objects, left, right = label_gen_final.pose_gen(
            [spec["grasp_label"] for spec in specs], len(specs), False)
        articulation, _, arti_left, arti_right = label_gen_final.pose_gen(
            [spec["arti_label"] for spec in specs], len(specs), False)
    expected = [spec["object"] for spec in specs]
    if objects != expected:
        raise ValueError("pose_gen returned wrong objects: {} != {}".format(objects, expected))
    return {"grasp": [_float(a) for a in grasp],
            "articulation": [_float(a) for a in articulation],
            "left": _float(left), "right": _float(right),
            "arti_left": _float(arti_left), "arti_right": _float(arti_right)}


def load_environment_config(repo_root, count, render, threads, saved=None):
    if saved is not None:
        config = saved
    else:
        from ruamel.yaml import YAML
        path = (Path(repo_root) / "raisimGymTorch/raisimGymTorch/env/envs/"
                "compose_eval/cfgs/cfg_reg.yaml")
        with path.open() as handle:
            config = YAML(typ="safe").load(handle)
    # Copy so replay-specific rendering/parallelism does not mutate the artifact.
    import copy
    config = copy.deepcopy(config)
    config["environment"].update({
        "num_envs": count, "num_threads": threads, "render": bool(render),
        "hand_model_r": "rhand_mano_demo.urdf", "hand_model_l": "lhand_mano_demo.urdf"})
    return config


def make_environment(repo_root, data, config, object_name):
    from ruamel.yaml import YAML
    from raisimGymTorch.env.bin import compose_eval
    from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv
    stream = io.StringIO()
    YAML().dump(config["environment"], stream)
    env = RaisimGymVecEnv(compose_eval.RaisimGymEnv(
        str(Path(repo_root) / "rsc"), stream.getvalue()), config["environment"],
        normalize_ob=False)
    if env.num_gs != 138:
        raise RuntimeError("Expected released compose_eval global-state dimension 138; got {}. "
                           "Rebuild/use the matching released environment.".format(env.num_gs))
    stage_dim = data["grasp"][0].copy()
    stage_dim[:, :2] = 3.0
    env.add_stage(stage_dim, data["grasp"][1])
    env.load_multi_articulated(["{0}/{0}.urdf".format(object_name)] * env.num_envs)
    return env


def policy_parameter_device(models, requested):
    """Inspect actual parameter placement rather than echoing the CLI option."""
    devices = {parameter.device for model in models.values() for parameter in model.parameters()}
    if len(devices) != 1:
        raise RuntimeError("Expected all policy parameters on one device; got {}".format(
            sorted(str(device) for device in devices)))
    actual = next(iter(devices))
    if (actual.type != requested.type or
            (requested.index is not None and actual.index != requested.index)):
        raise RuntimeError("Policy device mismatch: requested {}, actual {}".format(requested, actual))
    return actual


def policy_tensor_devices(tensors, expected):
    """Verify actual forward inputs/outputs before converting actions to NumPy."""
    if any(tensor.device != expected for tensor in tensors.values()):
        raise RuntimeError("Policy forward device mismatch: expected {}, got {}".format(
            expected, {name: str(tensor.device) for name, tensor in tensors.items()}))
    return {name: str(tensor.device) for name, tensor in tensors.items()}


@contextlib.contextmanager
def _timed_stage(name, performance):
    print("  Starting {}".format(name), flush=True)
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        performance["stage_seconds"][name] = elapsed
        print("  {}: {:.2f} s".format(name, elapsed), flush=True)


class Policies:
    """Deterministic actors only; critic/optimizer/scaling are unused by released eval."""

    def __init__(self, repo_root, checkpoints, config, env, device="cpu", seed=1):
        import torch
        import torch.nn as nn
        from raisimGymTorch.algo.ppo.module import MLP
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; select --device cpu.")
        print("Loading checkpoints on CPU; requested policy inference device: {}".format(
            self.device), flush=True)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # Torch's OpenMP backend calls omp_set_num_threads here. Setting this to
        # 1 after RaiSim initialization can also serialize its parallel loops
        # when the libraries share an OpenMP runtime. Keep their settings aligned.
        torch.set_num_threads(int(config["environment"]["num_threads"]))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        self.models = {}
        for name, obs_dim in (("cooperative_left", env.num_obs_l),
                              ("cooperative_right", env.num_obs_r),
                              ("articulation_right", env.num_obs_r)):
            model = MLP(config["architecture"]["policy_net"], nn.LeakyReLU,
                        obs_dim, env.num_acts)
            path = Path(repo_root) / checkpoints[name]
            # Supports the original paper-era PyTorch. Load the user's local checkpoints.
            checkpoint = torch.load(str(path), map_location="cpu")
            model.load_state_dict(checkpoint["actor_architecture_state_dict"])
            model.eval().to(self.device)
            self.models[name] = model
        self.device = policy_parameter_device(self.models, self.device)
        description = str(self.device)
        if self.device.type == "cuda":
            description += " ({})".format(torch.cuda.get_device_name(self.device))
        print("Policy inference device (verified parameters): {}".format(description), flush=True)
        print("RaiSim physics device: CPU | environments: {} | requested threads: {}".format(
            env.num_envs, config["environment"]["num_threads"]), flush=True)
        print("PyTorch CPU threads: {} (aligned with requested physics threads)".format(
            torch.get_num_threads()), flush=True)
        self.articulation = False
        self.forward_devices = None

    def actions(self, obs_r, obs_l):
        with self.torch.no_grad():
            right_name = "articulation_right" if self.articulation else "cooperative_right"
            input_r = self.torch.from_numpy(_float(obs_r)).to(self.device)
            output_r = self.models[right_name].architecture(input_r)
            right = output_r.cpu().numpy()
            input_l = self.torch.from_numpy(_float(obs_l)).to(self.device)
            output_l = self.models["cooperative_left"].architecture(input_l)
            left = output_l.cpu().numpy()
        if self.forward_devices is None:
            self.forward_devices = policy_tensor_devices(
                dict(input_r=input_r, input_l=input_l, output_r=output_r, output_l=output_l), self.device)
            print("Policy forward (verified tensors): {} | worker PID: {}".format(
                self.forward_devices, os.getpid()), flush=True)
            if self.device.type == "cuda":
                print("CUDA tensor memory allocated: {:.2f} MiB".format(
                    self.torch.cuda.memory_allocated(self.device) / (1024 ** 2)), flush=True)
        return _float(right), _float(left)


class Controller:
    """Explicit copy of released phase transitions; shared by trials and replay."""

    def __init__(self, env, data, specs, timing, policy=None):
        from raisimGymTorch.helper import rotations
        from .targets import resolve_articulation
        self.env, self.data, self.specs, self.timing = env, data, specs, timing
        self.policy = policy
        self.object = specs[0]["object"]
        if policy is not None:
            # ketchup_eval.py uses the pretrained articulation actor from step 0.
            policy.articulation = self.object == "ketchup"
        self.grasp, self.arti = data["grasp"], data["articulation"]
        self.count = len(specs)
        self.left, self.right = data["left"].copy(), data["right"].copy()
        self.arti_left, self.arti_right = data["arti_left"].copy(), data["arti_right"].copy()
        self.zero_angle = np.zeros((self.count, 1), dtype=np.float32)
        self.angle = _float([[resolve_articulation(spec["target"], float(self.grasp[2][i, 7]))]
                             for i, spec in enumerate(specs)])
        self.goal = self.grasp[2].copy()
        self.goal[:, :2] += _float([spec["target"]["translation_m"] for spec in specs])
        euler = rotations.quat2euler(self.goal[:, 3:7])
        euler[:, 1] += [spec["target"]["orientation_delta_rad"] for spec in specs]
        self.euler_y = euler[:, 1].copy()
        self.goal[:, 3:7] = rotations.euler2quat(euler)
        self.lift_goal = self.goal.copy()
        self.lift_goal[:, 2] += [spec["target"]["lift_m"] for spec in specs]
        # A small, planned clearance correction keeps collision meshes above
        # the support surface. It changes placement only, not the lift height.
        self.goal[:, 2] += [spec["target"].get("placement_z_offset_m", 0.0) for spec in specs]
        from .geometry import orientation_goal_pair
        from .metrics import public_euler_y_rad
        for i, spec in enumerate(specs):
            rotation = spec["target"].get("orientation_rotation")
            if rotation is not None:
                placement_q, lift_q = orientation_goal_pair(self.goal[i, 3:7], rotation)
                self.goal[i, 3:7], self.lift_goal[i, 3:7] = placement_q, lift_q
                if rotation["stage"] in ("placement", "both"):
                    self.euler_y[i] = public_euler_y_rad(placement_q)

    def reset(self):
        zero_velocity = np.zeros((self.count, 51), dtype=np.float32)
        self.env.reset_state(self.grasp[3], self.grasp[4], zero_velocity,
                             zero_velocity.copy(), self.grasp[2])

    def goals(self, data, angle):
        self.env.set_goals_r(data[6], data[8], data[10], data[12])
        self.env.set_goals(angle, data[6], data[8], data[9], data[10], data[11],
                           data[12], data[13], data[14], data[15])

    def switch(self, left, right):
        self.env.control_switch_all(_float(left), _float(right))

    def constants(self, left, right):
        self.switch(np.full(self.count, left), np.full(self.count, right))

    def phase(self, step):
        env, grasp, arti = self.env, self.grasp, self.arti
        g, t = self.timing["grasp_steps"], self.timing["transit_steps"]
        ketchup = self.timing["protocol"] == "ketchup"
        if step == 0:
            self.goals(grasp, self.zero_angle)
            env.set_obj_goal(self.zero_angle, self.lift_goal)
            env.set_imitation_goals(grasp[4], grasp[3], grasp[2])
            if ketchup:
                self.constants(9, 7)
            else:
                self.left[self.left == 7] = 11
                self.right[self.right == 7] = 11
                self.switch(self.left, self.right)
        elif step == g:
            if ketchup:
                self.constants(1, 7)
            else:
                self.switch(np.where(self.left == 9, 1, self.left),
                            np.where(self.right == 9, 1, self.right))
        elif step == g + t:
            env.set_obj_goal(self.zero_angle, self.goal)
            if ketchup:
                self.constants(1, 7)
            else:
                self.switch(np.where(self.left == 9, 1, self.left),
                            np.where(self.right == 9, 1, self.right))
        elif ketchup:
            if step == g + 2 * t:
                # set_obj_goal alone changes the visual/position target but not
                # final_obj_angle_, which supplies the policy's relative-angle
                # observation. Forward the actual experimental target too.
                self.goals(grasp, self.angle)
                env.set_obj_goal(self.angle, self.goal)
                self.constants(12, 11)
            elif step == g + 2 * t + 100:
                env.reset_right_hand(grasp[4], grasp[3], grasp[2])
                self.constants(12, 8)
        else:
            release = g + 2 * t
            # C++ scripts use fractional boundaries; ceil reproduces step < boundary.
            boundaries = [release, release + int(np.ceil(t / 4.0)),
                          release + int(np.ceil(t / 2.0)),
                          release + int(np.ceil(3 * t / 4.0)), g + 3 * t + 100]
            if step not in boundaries:
                return
            phase = boundaries.index(step)
            if phase == 0:
                self.goals(arti, self.angle)
                env.set_obj_goal(self.angle, self.goal)
            if phase in (0, 1, 2):
                left, right = np.zeros_like(arti[4]), np.zeros_like(arti[3])
                if phase == 1:
                    left[:, 0] += 0.5
                    right[:, 0] -= 0.5
                elif phase == 2:
                    left[:, 1:3] += 0.3
                    right[:, 1:3] += 0.3
                env.set_imitation_goals(left, right, arti[2])
                self.constants(7 if self.object in ("mixer", "ketchup") else 13, 13)
            elif phase == 3:
                if self.policy is not None:
                    self.policy.articulation = self.object in ("notebook", "mixer", "ketchup")
                env.set_imitation_goals(arti[4], arti[3], arti[2])
                self.constants(7, 11)
            else:
                # Argument order follows public compose_eval.py (C++ ignores inputs).
                env.reset_right_hand(arti[4], arti[3], arti[2])
                self.arti_left[:] = 7
                self.switch(self.arti_left, self.arti_right)

    def metric_spec(self, index):
        return dict(self.specs[index], target_position_m=self.goal[index, :3].tolist(),
                    target_quaternion_wxyz=self.goal[index, 3:7].tolist(),
                    target_articulation_rad=float(self.angle[index, 0]),
                    target_euler_y_rad=float(self.euler_y[index]),
                    initial_articulation_rad=float(self.grasp[2][index, 7]),
                    lift_target_position_m=self.lift_goal[index, :3].tolist(),
                    lift_target_quaternion_wxyz=self.lift_goal[index, 3:7].tolist())


def simulate(env, controller, policy=None, replay_actions=None, realtime=False, timings=None):
    count = controller.count
    steps = total_steps(controller.timing)
    states = np.empty((steps + 1, count, 138), dtype=np.float32)
    actions_r = np.empty((steps, count, 51), dtype=np.float32)
    actions_l = np.empty_like(actions_r)
    invalid = np.zeros(count, dtype=bool)
    first_invalid = np.full(count, -1, dtype=np.int32)
    wall_start = time.perf_counter()
    policy_seconds = physics_seconds = 0.0
    controller.reset()
    states[0] = env.get_global_state()
    start = time.monotonic()
    for step in range(steps):
        controller.phase(step)
        if replay_actions is None:
            obs_r, obs_l = env.observe(update_mean=False)
            bad_observation = (~np.isfinite(obs_r).all(axis=1) |
                               ~np.isfinite(obs_l).all(axis=1))
            invalid |= bad_observation
            obs_r = np.nan_to_num(obs_r, nan=0.0, posinf=0.0, neginf=0.0)
            obs_l = np.nan_to_num(obs_l, nan=0.0, posinf=0.0, neginf=0.0)
            policy_start = time.perf_counter() if timings is not None else 0.0
            right, left = policy.actions(obs_r, obs_l)
            if timings is not None:
                # actions() returns CPU NumPy arrays: CUDA transfers have already
                # synchronized. This is wall time including transfers, not GPU
                # kernel time or GPU utilization.
                policy_seconds += time.perf_counter() - policy_start
        else:
            right, left = replay_actions[0][step], replay_actions[1][step]
        bad_action = ~np.isfinite(right).all(axis=1) | ~np.isfinite(left).all(axis=1)
        invalid |= bad_action
        right = _float(np.nan_to_num(right, nan=0.0, posinf=0.0, neginf=0.0))
        left = _float(np.nan_to_num(left, nan=0.0, posinf=0.0, neginf=0.0))
        actions_r[step], actions_l[step] = right, left
        physics_start = time.perf_counter() if timings is not None else 0.0
        _, _, done = env.step(right, left)
        states[step + 1] = env.get_global_state()
        if timings is not None:
            physics_seconds += time.perf_counter() - physics_start
        # VecEnv silently resets numerical failures. Never score a reset as a success.
        invalid |= np.asarray(done, dtype=bool) | ~np.isfinite(states[step + 1]).all(axis=1)
        first_invalid[(first_invalid < 0) & invalid] = step
        if realtime:
            remaining = start + (step + 1) * controller.control_dt - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
        if step and step % 500 == 0 and not realtime:
            print("  {}/{} steps | {:.1f} s elapsed".format(
                step, steps, time.monotonic() - start), flush=True)
    if timings is not None:
        elapsed = time.perf_counter() - wall_start
        timings.update(wall_s=elapsed, policy_and_transfer_s=policy_seconds,
                       physics_and_state_s=physics_seconds,
                       other_s=max(0.0, elapsed - policy_seconds - physics_seconds),
                       vector_steps=steps, environments=count)
    return states, actions_r, actions_l, invalid, first_invalid


def lift_metrics(states, spec, timing, valid=True):
    """Measure the intermediate OOD lift separately from the paper's final goal."""
    names = ("lift_max_height_m", "lift_final_height_m", "lift_target_error_m")
    if not valid:
        return dict({name: None for name in names}, lift_target_success=False,
                    ood_goal_success=False)
    g, t = timing["grasp_steps"], timing["transit_steps"]
    heights = states[g + 1:g + t + 1, 124] - states[0, 124]
    target = float(spec["target"]["lift_m"])
    error = float(np.min(np.abs(heights - target)))
    return {"lift_max_height_m": float(np.max(heights)),
            "lift_final_height_m": float(heights[-1]),
            "lift_target_error_m": error, "lift_target_success": error < 0.05}


def _save_trajectory(path, index, states, right, left, data):
    arrays = {"states": states[:, index], "actions_r": right[:, index],
              "actions_l": left[:, index]}
    for key in ("grasp", "articulation"):
        arrays.update({"{}_{}".format(key, j): values[index]
                       for j, values in enumerate(data[key])})
    for key in ("left", "right", "arti_left", "arti_right"):
        arrays[key] = data[key][index]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(str(temporary), str(path))


def run_worker(repo_root, results_dir, episode_ids, device=None, threads=None):
    from .metrics import evaluate_episode, evaluate_goal_diagnostics
    from .targets import articulation_is_feasible, articulation_outside_randomized_training
    _paths(repo_root)
    results_dir = Path(results_dir)
    plan = read_json(results_dir / "plan.json")
    index = {spec["episode_id"]: spec for spec in plan["episodes"]}
    specs = [index[episode_id] for episode_id in episode_ids]
    if not specs or len({(s["object"], s["profile"]) for s in specs}) != 1:
        raise ValueError("Worker requires one nonempty object/profile batch")
    settings = plan["config"].get("runtime", {})
    physics_threads = threads or settings.get("num_threads", 4)
    requested_device = device or settings.get("device", "cpu")
    performance = dict(worker_pid=os.getpid(), episode_ids=episode_ids,
                       requested_policy_device=requested_device, requested_physics_threads=physics_threads,
                       stage_seconds={}, rollout={})
    with _timed_stage("reference_data", performance):
        data = prepare_data(repo_root, specs)
    config = load_environment_config(repo_root, len(specs), False, physics_threads)
    timing = timing_for(specs[0]["object"])
    with _timed_stage("environment_setup", performance):
        env = make_environment(repo_root, data, config, specs[0]["object"])
    try:
        env.seed(int(plan["config"].get("seed", 1)))
        with _timed_stage("policy_setup", performance):
            policy = Policies(repo_root, plan["config"]["checkpoints"], config, env,
                              requested_device, plan["config"].get("seed", 1))
        controller = Controller(env, data, specs, timing, policy)
        # Kept in the batch artifact for exact replay if environment YAML later changes.
        with _timed_stage("rollout", performance):
            states, right, left, invalid, first_invalid = simulate(
                env, controller, policy, timings=performance["rollout"])
        performance.update(policy_device=str(policy.device), policy_forward_devices=policy.forward_devices,
                           torch_cpu_threads=policy.torch.get_num_threads())
        print("  Rollout breakdown: CPU physics + state {:.2f} s | policy + transfers {:.2f} s | other {:.2f} s".format(
            performance["rollout"]["physics_and_state_s"], performance["rollout"]["policy_and_transfer_s"],
            performance["rollout"]["other_s"]), flush=True)
        scoring_start = time.perf_counter()
        for i, spec in enumerate(specs):
            metric_spec = controller.metric_spec(i)
            row = dict(spec)
            row.update(evaluate_episode(states[:, i], metric_spec, timing))
            row.update({key: metric_spec[key] for key in (
                "target_position_m", "target_quaternion_wxyz", "target_articulation_rad",
                "target_euler_y_rad", "initial_articulation_rad", "lift_target_position_m",
                "lift_target_quaternion_wxyz")})
            row["target_reachable"] = articulation_is_feasible(
                row["target_articulation_rad"], spec["joint_limits_rad"])
            row["articulation_within_joint_limits"] = row["target_reachable"]
            row["articulation_randomized_target_ood"] = articulation_outside_randomized_training(
                row["target_articulation_rad"])
            row["articulation_requires_reset_resolution"] = False
            row["target"] = dict(spec["target"],
                                 articulation_rad=row["target_articulation_rad"],
                                 resolved_articulation_rad=row["target_articulation_rad"])
            if invalid[i]:
                row.update(valid=False, paper_success=False, public_success=False,
                           failure_reason="nonfinite_or_simulator_auto_reset",
                           first_invalid_step=int(first_invalid[i]))
                for key in row:
                    if "_error_" in key:
                        row[key] = None
            row.update(lift_metrics(states[:, i], spec, timing, row["valid"]))
            row.update(evaluate_goal_diagnostics(states[:, i], metric_spec, timing, row["valid"]))
            row["timing"] = timing
            row["controller_changes"] = (["ketchup_forward_requested_articulation"]
                                         if spec["object"] == "ketchup" else [])
            row["environment_config"] = config
            row["policy_device"] = str(policy.device)
            trajectory = "trajectories/{}.npz".format(spec["episode_id"])
            _save_trajectory(results_dir / trajectory, i, states, right, left, data)
            row["trajectory_path"] = trajectory
            row["trajectory_sha256"] = file_sha256(results_dir / trajectory)
            append_row(results_dir / "episodes.jsonl", row)
            print("{} paper={} public={} valid={}".format(
                spec["episode_id"], row["paper_success"], row["public_success"],
                row.get("valid", True)), flush=True)
        performance["stage_seconds"]["scoring_and_save"] = time.perf_counter() - scoring_start
        performance_path = results_dir / "performance" / (specs[0]["episode_id"] + ".json")
        write_json(performance_path, performance)
        print("  scoring_and_save: {:.2f} s | timings: {}".format(
            performance["stage_seconds"]["scoring_and_save"], performance_path), flush=True)
    finally:
        env.close()


def replay_episode(repo_root, results_dir, episode_id, video_name=None):
    from .metrics import evaluate_episode, evaluate_goal_diagnostics
    _paths(repo_root)
    results_dir = Path(results_dir)
    plan = read_json(results_dir / "plan.json")
    matches = [row for row in read_rows(results_dir / "episodes.jsonl")
               if row["episode_id"] == episode_id]
    if len(matches) != 1:
        raise ValueError("No unique completed trial for {}".format(episode_id))
    row = matches[0]
    if not row.get("valid", True):
        raise ValueError("Cannot select an invalid numerical trial for qualitative playback")
    if video_name and (platform.system() != "Linux" or
                       re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.mp4", video_name) is None):
        raise ValueError("Native Unity recording requires Linux and an alphanumeric .mp4 basename "
                         "(letters, digits, _, - and . only)")
    if file_sha256(results_dir / row["trajectory_path"]) != row["trajectory_sha256"]:
        raise RuntimeError("Saved trajectory checksum differs from the evaluated episode")
    with np.load(str(results_dir / row["trajectory_path"]), allow_pickle=False) as saved:
        data = {key: [saved["{}_{}".format(key, j)][None].copy() for j in range(17)]
                for key in ("grasp", "articulation")}
        data.update({key: saved[key].reshape(1).copy()
                     for key in ("left", "right", "arti_left", "arti_right")})
        recorded_states = saved["states"].copy()
        actions = (saved["actions_r"][:, None].copy(), saved["actions_l"][:, None].copy())
    config = load_environment_config(repo_root, 1, bool(video_name), 1, row["environment_config"])
    env = make_environment(repo_root, data, config, row["object"])
    recording = False
    try:
        env.seed(int(plan["config"].get("seed", 1)))
        controller = Controller(env, data, [row], row["timing"])
        controller.control_dt = float(config["environment"]["control_dt"])
        if video_name:
            env.turn_on_visualization()
            controller.reset()
            time.sleep(2.0)
            env.start_video_recording(video_name)
            recording = True
            time.sleep(0.5)
        states, _, _, invalid, _ = simulate(env, controller, replay_actions=actions,
                                           realtime=bool(video_name))
        result = evaluate_episode(states[:, 0], row, row["timing"])
        result.update(lift_metrics(states[:, 0], row, row["timing"],
                                   bool(result["valid"] and not invalid[0])))
        result.update(evaluate_goal_diagnostics(states[:, 0], row, row["timing"],
                                               bool(result["valid"] and not invalid[0])))
        equal = np.allclose(states[:, 0], recorded_states, atol=1e-4, rtol=1e-4,
                            equal_nan=False)
        classification_equal = all(bool(result[key]) == bool(row[key])
                                   for key in ("paper_success", "public_success", "ood_goal_success",
                                               "strict_goal_success") if key in row)
        result.update(episode_id=episode_id,
                      verified=bool(equal and classification_equal and not invalid[0]),
                      trajectory_matches=bool(equal),
                      classification_matches=classification_equal,
                      simulation_seconds=total_steps(row["timing"]) * controller.control_dt,
                      max_state_difference=float(np.max(np.abs(states[:, 0] - recorded_states))))
        return result
    finally:
        if recording:
            env.stop_video_recording()
            time.sleep(2.0)
        env.close()
