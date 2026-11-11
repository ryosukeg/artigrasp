"""Deterministic box response-collection plans, without Torch or RaiSim.

The entry point imports only the existing target utilities. NumPy is needed
only when those utilities load the official train labels for ``random`` plans.
The caller must add the repository's ``tools`` directory to ``sys.path``.
"""

import copy
import hashlib
import itertools
import json
import math
import random
from pathlib import Path

from ood_suite.targets import (
    articulation_is_feasible,
    joint_limits,
    load_label_sets,
    sample_target,
    stable_seed,
)


HORIZON = 20
START_STEP = 1260
TOTAL_STEPS = 1560
NUM_WINDOWS = (TOTAL_STEPS - START_STEP) // HORIZON
COLLECTION_GOAL_SOURCE = "box_dynamics_collection_v1"
CALIBRATION_BASELINE_IDS = tuple("box__normal__{:04d}".format(i) for i in (0, 2, 4, 6))


def _file_source(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("{} must be a positive integer".format(name))
    return value


def _group_key(label):
    return int(label["seq_num"]), str(label["seq_name"])


def _group_id(key):
    return "s{:02d}/{}".format(*key)


def _reference_groups(labels):
    groups = {}
    for role in ("grasp", "articulation"):
        for index, label in enumerate(labels[role]):
            if label["obj_name"] != "box":
                raise ValueError("Only box labels are valid for box collection")
            key = _group_key(label)
            groups.setdefault(key, {"grasp": [], "articulation": []})[role].append(index)
    return groups


def _split_references(labels, seed):
    """Assign whole source sequences before sampling either reference role.

    Choose a seeded partition nearest to 20% validation source groups, with
    both roles represented on both sides. The released box train set has 18
    groups, so exhaustive fallback over these small group combinations is cheap.
    No episode, frame, goal, outcome, or test label affects the partition.
    """
    groups = _reference_groups(labels)
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError("Reference-disjoint train/val requires at least two source groups")
    for role in ("grasp", "articulation"):
        if sum(bool(groups[key][role]) for key in keys) < 2:
            raise ValueError("Reference-disjoint train/val requires {} in at least two source groups".format(role))
    split_seed = stable_seed(seed, "box_collection_reference_split")
    random.Random(split_seed).shuffle(keys)
    wanted = max(1, min(len(keys) - 1, int(math.ceil(0.2 * len(keys)))))
    sizes = sorted(range(1, len(keys)), key=lambda size: (abs(size - wanted), size))
    all_keys = set(keys)
    validation = None
    for size in sizes:
        for selected in itertools.combinations(keys, size):
            candidate = set(selected)
            if all(any(groups[key][role] for key in side)
                   for side in (candidate, all_keys - candidate)
                   for role in ("grasp", "articulation")):
                validation = candidate
                break
        if validation is not None:
            break
    if validation is None:
        raise ValueError("Cannot partition box source groups with both reference roles on each side")
    assignments = {key: ("val" if key in validation else "train") for key in all_keys}
    pools = {
        split: {role: [i for i, label in enumerate(labels[role])
                       if assignments[_group_key(label)] == split]
                for role in ("grasp", "articulation")}
        for split in ("train", "val")
    }
    manifest_groups = []
    for key in sorted(groups):
        references = []
        for role in ("grasp", "articulation"):
            for index in groups[key][role]:
                references.append({"role": role, "label_index": index,
                                   "label": copy.deepcopy(labels[role][index])})
        manifest_groups.append({"group_id": _group_id(key), "seq_num": key[0],
                                "seq_name": key[1], "dataset_split": assignments[key],
                                "references": references})
    manifest = {
        "official_split": "train",
        "group_keys": ["seq_num", "seq_name"],
        "seed": split_seed,
        "requested_validation_group_fraction": 0.2,
        "method": "seeded source-group partition, nearest feasible 20% val; then sample both roles within split",
        "groups": manifest_groups,
        "counts": {
            split: {"groups": sum(value == split for value in assignments.values()),
                    "grasp": len(pools[split]["grasp"]),
                    "articulation": len(pools[split]["articulation"]),
                    "possible_pairs": len(pools[split]["grasp"]) * len(pools[split]["articulation"])}
            for split in ("train", "val")
        },
        "sampling": "label pairs sampled with replacement; source groups never cross train/val",
        "test_split_used": False,
    }
    return pools, manifest


def _baseline_reference_manifest(specs, dataset_split):
    groups = {}
    for spec in specs:
        for role, field in (("grasp", "grasp_label"), ("articulation", "arti_label")):
            label = spec[field]
            key = _group_key(label)
            entry = groups.setdefault(key, {
                "group_id": _group_id(key), "seq_num": key[0], "seq_name": key[1],
                "dataset_split": dataset_split, "references": [],
            })
            reference = {"role": role, "label": copy.deepcopy(label)}
            if reference not in entry["references"]:
                entry["references"].append(reference)
    return {"official_split": "train", "group_keys": ["seq_num", "seq_name"],
            "groups": [groups[key] for key in sorted(groups)],
            "method": "unchanged baseline references; excluded from predictor train/val",
            "dataset_split": dataset_split, "test_split_used": False}


def _candidates(amplitude):
    values = [[0.0, 0.0, 0.0]]
    for axis in range(3):
        for sign in (1.0, -1.0):
            vector = [0.0, 0.0, 0.0]
            vector[axis] = sign * amplitude
            values.append(vector)
    return values


def _calibration_variants():
    variants = [("zero", [0.0, 0.0, 0.0])]
    for axis, name in enumerate(("x", "y", "z")):
        for amplitude, magnitude in ((0.5, "0p5"), (1.0, "1p0")):
            for sign, sign_name in ((1.0, "pos"), (-1.0, "neg")):
                vector = [0.0, 0.0, 0.0]
                vector[axis] = sign * amplitude
                variants.append(("{}_{}_{}".format(name, sign_name, magnitude), vector))
    return variants


def _load_baseline(repo_root, baseline_dir):
    directory = Path(baseline_dir)
    if not directory.is_absolute():
        directory = Path(repo_root) / directory
    path = directory / "plan.json"
    source = _file_source(path)
    baseline = json.loads(path.read_text(encoding="utf-8"))
    specs = baseline.get("episodes", [])
    if len(specs) != 8 or len({spec["episode_id"] for spec in specs}) != 8:
        raise ValueError("Expected the eight unique baseline smoke episodes")
    if baseline.get("split", baseline["config"].get("split")) != "train":
        raise ValueError("Collection baseline must use the official train split")
    if any(spec.get("object") != "box" or spec.get("split") != "train" for spec in specs):
        raise ValueError("Every baseline episode must be box from the official train split")
    source["directory"] = str(directory.resolve())
    source["seed"] = baseline.get("seed", baseline["config"].get("seed"))
    source["episode_ids"] = [spec["episode_id"] for spec in specs]
    return baseline, source


def build_collection_plan(repo_root, kind, baseline_dir, seed=301, episodes=40,
                          amplitude=1.0, batch_size=8, num_threads=4, device="cuda"):
    """Return a JSON-compatible plan without executing any simulation.

    ``zero`` always preserves all eight baseline specs, targets, IDs and seeds.
    ``calibration`` repeats baseline 0000/0002/0004/0006 for 13 fixed residuals
    (52 episodes), preserving their targets and seeds. ``episodes`` and
    ``amplitude`` select the random collection size and its seven-point action
    support only. For 40 random episodes the split is 32 train / 8 validation.
    Zero/calibration also preserve the baseline runtime seed: the supplied
    ``seed`` is recorded as requested, but only changes random collection.
    Baseline paths relative to ``repo_root`` are accepted. No test labels are
    loaded, and no learning or simulation occurs in this module.
    """
    repo_root = Path(repo_root).resolve()
    if kind not in ("zero", "calibration", "random"):
        raise ValueError("kind must be zero, calibration, or random")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    _positive_int(episodes, "episodes")
    _positive_int(batch_size, "batch_size")
    _positive_int(num_threads, "num_threads")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be a nonempty string")
    if isinstance(amplitude, bool) or not isinstance(amplitude, (int, float)) or not math.isfinite(amplitude) or amplitude <= 0:
        raise ValueError("amplitude must be finite and positive")
    if kind == "random" and episodes < 2:
        raise ValueError("random collection requires at least two episodes for train/val")
    baseline, baseline_source = _load_baseline(repo_root, baseline_dir)
    config = copy.deepcopy(baseline["config"])
    execution_seed = seed if kind == "random" else int(config["seed"])
    config.update(seed=execution_seed, split="train", objects=["box"])
    config["runtime"] = dict(config.get("runtime", {}), batch_size=batch_size,
                             num_threads=num_threads, device=device)
    specs = []
    collection = {
        "horizon": HORIZON, "start_step": START_STEP, "total_steps": TOTAL_STEPS,
        "num_windows": NUM_WINDOWS, "control_dt": 0.01,
        "right_action_indices": [0, 1, 2], "right_mode": 8,
        "policy_recomputed_each_step": True,
        "residual_units": "policy action coordinates",
        "nominal_target_delta_m_per_action_unit": 0.001,
        "window_end_convention": "exclusive; residual is held for exactly horizon steps",
        "seed": execution_seed, "requested_seed": seed,
        "seed_note": ("requested seed controls random collection" if kind == "random" else
                      "requested seed is ignored; baseline runtime and episode seeds are preserved for paired replay"),
    }
    if kind == "zero":
        for original in baseline["episodes"]:
            spec = copy.deepcopy(original)
            spec.update(dataset_split="audit", baseline_episode_id=original["episode_id"],
                        residuals=[[0.0, 0.0, 0.0] for _ in range(NUM_WINDOWS)])
            specs.append(spec)
        collection.update(amplitude=0.0, candidate_residuals=[[0.0, 0.0, 0.0]],
                          sampling="zero residual; unchanged baseline episode seeds and targets")
        reference_split = _baseline_reference_manifest(specs, "audit")
    elif kind == "calibration":
        by_id = {spec["episode_id"]: spec for spec in baseline["episodes"]}
        if not set(CALIBRATION_BASELINE_IDS) <= set(by_id):
            raise ValueError("Calibration needs baseline episodes 0000, 0002, 0004 and 0006")
        variants = _calibration_variants()
        for episode_id in CALIBRATION_BASELINE_IDS:
            for variant, residual in variants:
                spec = copy.deepcopy(by_id[episode_id])
                spec.update(episode_id=episode_id + "__calibration__" + variant,
                            dataset_split="calibration", calibration_pair_id=episode_id,
                            baseline_episode_id=episode_id, variant=variant,
                            residuals=[residual[:] for _ in range(NUM_WINDOWS)])
                specs.append(spec)
        collection.update(amplitude=1.0, calibration_amplitudes=[0.5, 1.0],
                          candidate_residuals=[vector for _, vector in variants],
                          sampling="each fixed variant repeated through all 15 windows; baseline seed paired")
        reference_split = _baseline_reference_manifest(specs, "calibration")
    else:
        labels = load_label_sets(repo_root, ["box"], split="train")["box"]
        pools, reference_split = _split_references(labels, seed)
        label_path = repo_root / "rsc/mano_double/grasp_label/final_labels/train_set/box.npy"
        reference_split["source"] = _file_source(label_path)
        limits = joint_limits(repo_root, "box")
        if limits[0] > 0.5 or limits[1] < 1.5:
            raise ValueError("Collection support [0.5,1.5] must lie within the unmodified box joint limits")
        normal_definition = copy.deepcopy(config.get("normal_goal"))
        if not normal_definition or normal_definition.get("source") != "paper_supplement_b4":
            raise ValueError("Baseline normal_goal is required to preserve its spatial distribution")
        # The paper normal definition is used only to obtain its spatial draw.
        # It is never relabeled with the wider collection-angle support.
        goal_definition = {
            "source": COLLECTION_GOAL_SOURCE, "articulation_rad": [0.5, 1.5],
            "articulation_rule": "absolute", "articulation_sampling": "uniform",
            "spatial_source": "unchanged baseline normal spatial distribution",
            "lift_m": 0.15, "translation_xy_m": [[-0.1, -0.05], [-0.1, -0.05]],
            "yaw_rad": copy.deepcopy(normal_definition.get("yaw_rad", [-0.4, 0.4])),
            "orientation_frame": "world", "orientation_stage": "both",
        }
        config["baseline_normal_goal"] = config.pop("normal_goal")
        config["collection_goal"] = copy.deepcopy(goal_definition)
        config["profiles"] = {"collection": {"description": "box response collection; not paper normal evaluation"}}
        config["suites"] = {"collection": ["collection"]}
        config["default_suite"] = config["selected_suite"] = "collection"
        validation_count = min(episodes - 1, max(1, int(round(episodes * 0.2))))
        counts = {"train": episodes - validation_count, "val": validation_count}
        candidates = _candidates(float(amplitude))
        for split in ("train", "val"):
            reference_split["counts"][split]["episodes"] = counts[split]
            for index in range(counts[split]):
                episode_seed = stable_seed(seed, "box_collection_episode", split, index)
                label_rng = random.Random(stable_seed(episode_seed, "labels"))
                grasp_index = label_rng.choice(pools[split]["grasp"])
                arti_index = label_rng.choice(pools[split]["articulation"])
                target = sample_target("box", episode_seed, {}, limits, index, normal_definition)
                angle_rng = random.Random(stable_seed(episode_seed, "collection_articulation"))
                target.update(articulation_rad=angle_rng.uniform(0.5, 1.5),
                              articulation_rule="absolute", changed_axes=["articulation"],
                              goal_source=COLLECTION_GOAL_SOURCE,
                              spatial_goal_source="baseline normal spatial distribution",
                              collection_goal_source=COLLECTION_GOAL_SOURCE)
                target.pop("normal_goal_source", None)
                target.pop("public_delta_rad", None)
                residual_seed = stable_seed(episode_seed, "collection_residuals")
                residual_rng = random.Random(residual_seed)
                candidate_indices = [residual_rng.randrange(7) for _ in range(NUM_WINDOWS)]
                spec = {
                    "episode_id": "box__collection__{}__{:04d}".format(split, index),
                    "object": "box", "profile": "collection", "sample_index": index,
                    "seed": episode_seed, "split": "train", "dataset_split": split,
                    "paper_object": True, "grasp_label_index": grasp_index,
                    "arti_label_index": arti_index, "grasp_label": copy.deepcopy(labels["grasp"][grasp_index]),
                    "arti_label": copy.deepcopy(labels["articulation"][arti_index]),
                    "target": target, "joint_limits_rad": limits[:],
                    "target_reachable": articulation_is_feasible(target["articulation_rad"], limits),
                    "articulation_within_joint_limits": True,
                    "target_reachability_basis": "articulation limits only; spatial reachability not inferred",
                    "articulation_requires_reset_resolution": False,
                    "evaluation_protocol": config.get("evaluation_protocol", "factorial_v1"),
                    "require_lift_pose_success": False, "family": "collection", "severity": None,
                    "compare_to": "normal", "stress_test": False,
                    "articulation_randomized_target_ood": False,
                    "factor_flags": {"horizontal": False, "lift": False, "articulation": True},
                    "residual_seed": residual_seed, "residual_candidate_indices": candidate_indices,
                    "residuals": [candidates[i][:] for i in candidate_indices],
                    "reference_group_ids": {"grasp": _group_id(_group_key(labels["grasp"][grasp_index])),
                                            "articulation": _group_id(_group_key(labels["articulation"][arti_index]))},
                }
                specs.append(spec)
        collection.update(amplitude=float(amplitude), candidate_residuals=candidates,
                          candidate_probabilities=[1.0 / 7.0] * 7,
                          sampling="independent uniform seven-candidate draw per window, with replacement",
                          goal_distribution=goal_definition)
    config["samples_per_object"] = len(specs)
    return {"schema_version": 1, "kind": kind, "config": config, "seed": execution_seed,
            "split": "train", "objects": ["box"], "episodes": specs,
            "collection": collection, "reference_split": reference_split,
            "baseline": baseline_source}
