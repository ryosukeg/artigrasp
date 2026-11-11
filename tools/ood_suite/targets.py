"""Reproducible target plans for the composed ArtiGrasp evaluation.

Everything except loading the existing ``.npy`` labels uses the Python standard
library.  In particular, inspecting a plan does not import Torch or RaiSim.
Spatial OOD refers to a deliberate extrapolation from the public evaluation
targets; the public repository does not describe the complete training support.
"""

import copy
import hashlib
import json
import math
import random
from pathlib import Path
from xml.etree import ElementTree

from .geometry import axis_angle_quaternion


ALL_OBJECTS = (
    "box", "laptop", "notebook", "waffleiron", "mixer", "ketchup",
    "espressomachine", "microwave", "capsulemachine", "phone",
)
PAPER_OBJECTS = ALL_OBJECTS[:8]
DEFAULT_PROFILES = (
    "id_control", "ood_height", "ood_translation",
    "ood_articulation_180", "ood_combined",
)


def load_config(path):
    """Load the experiment JSON without requiring scientific dependencies."""
    with Path(path).open(encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("Experiment configuration must be a JSON object")
    return config


def stable_seed(seed, *parts):
    """Use an explicit hash, independent of PYTHONHASHSEED or process order."""
    payload = json.dumps([int(seed)] + list(parts), separators=(",", ":"),
                         ensure_ascii=True).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7fffffff


def joint_limits(repo_root, object_name):
    """Read the one movable object's hinge limit, never changing its URDF."""
    path = Path(repo_root) / "rsc" / "arctic" / object_name / (object_name + ".urdf")
    movable = [joint for joint in ElementTree.parse(str(path)).getroot().findall("joint")
               if joint.get("type") not in ("fixed", "floating")]
    if len(movable) != 1:
        raise ValueError("Expected exactly one articulation joint in {}".format(path))
    joint = movable[0]
    limit = joint.find("limit")
    if joint.get("type") == "continuous" or limit is None:
        raise ValueError("Expected finite articulation limits in {}".format(path))
    bounds = [float(limit.attrib["lower"]), float(limit.attrib["upper"])]
    if not all(math.isfinite(value) for value in bounds) or bounds[0] > bounds[1]:
        raise ValueError("Invalid articulation limits in {}".format(path))
    return bounds


def articulation_is_feasible(angle, limits):
    """Tolerate float32 roundoff at exactly 180 degrees, without clipping."""
    return bool(limits[0] - 1e-7 <= float(angle) <= limits[1] + 1e-7)


def resolve_articulation(target, reset_angle):
    """Evaluate the public target rule using the actual reset articulation.

    The ordinary public evaluator closes an initially open hinge by 0.5--0.6
    radians, or opens an initially closed one to that angle. Ketchup uses an
    absolute 0.5--1.0-radian target.  Absolute OOD requests always stay literal.
    """
    if target.get("articulation_rule", "absolute") == "public_relative":
        delta = float(target["public_delta_rad"])
        return float(reset_angle) - delta if float(reset_angle) > 1.0 else delta
    return float(target["articulation_rad"])


def _range(value, name):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        bounds = [float(value), float(value)]
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        bounds = [float(item) for item in value]
    else:
        raise ValueError("{} must be a number or [minimum, maximum]".format(name))
    if not all(math.isfinite(item) for item in bounds) or bounds[0] > bounds[1]:
        raise ValueError("{} must have finite, ordered bounds".format(name))
    return bounds


def _sample(rng, value, name):
    low, high = _range(value, name)
    return rng.uniform(low, high)


def select_profiles(config, profiles=None, suite=None):
    """Select a declared suite or explicit profiles without resampling targets."""
    configured = config.get("profiles", {})
    if profiles is not None and suite is not None:
        raise ValueError("Select either --profiles or --suite")
    if profiles is not None:
        return list(profiles)
    selected_suite = suite or config.get("default_suite")
    if selected_suite is None:
        return list(configured)
    suites = config.get("suites", {})
    if not suites:
        raise ValueError("This configuration has no named suites; use --profiles or configs/ood_multiaxis.json")
    if selected_suite not in suites:
        raise ValueError("Unknown suite {}; choose {}".format(selected_suite, ", ".join(suites)))
    return list(suites[selected_suite])


def articulation_outside_randomized_training(angle):
    return bool(float(angle) < 0.5 - 1e-6 or float(angle) > 1.5 + 1e-6)


def _paper_normal(target, sample_seed, definition):
    """Use the paper's composed-goal angle and world-yaw ranges for all objects."""
    if definition.get("source") != "paper_supplement_b4":
        raise ValueError("normal_goal.source must be paper_supplement_b4")
    angle_rng = random.Random(stable_seed(sample_seed, "paper_normal_articulation"))
    yaw_rng = random.Random(stable_seed(sample_seed, "paper_normal_yaw"))
    angle = _sample(angle_rng, definition.get("articulation_rad", [0.5, 0.6]), "normal articulation_rad")
    yaw = _sample(yaw_rng, definition.get("yaw_rad", [-0.4, 0.4]), "normal yaw_rad")
    target.update(articulation_rad=angle, articulation_rule="absolute", orientation_delta_rad=0.0,
                  normal_goal_source="paper_supplement_b4",
                  orientation_rotation={"axis": [0.0, 0.0, 1.0], "axis_name": "z", "angle_rad": yaw,
                                        "frame": "world", "stage": "both",
                                        "quaternion_wxyz": axis_angle_quaternion([0, 0, 1], yaw)})


def _random_ood_articulation(sample_seed, definition, joint_bounds):
    """Sample uniformly over the feasible union of declared OOD intervals.

    Intersect intervals with joint limits before drawing. Never clamp a drawn
    angle into the learned range or silently call an in-range target OOD.
    """
    if not isinstance(definition, dict) or joint_bounds is None:
        raise ValueError("articulation_ood requires an interval definition and joint bounds")
    unknown = set(definition) - {"intervals_rad", "joint_margin_rad", "prefer_upper"}
    if unknown:
        raise ValueError("Unknown articulation_ood option(s): {}".format(", ".join(sorted(unknown))))
    margin = float(definition.get("joint_margin_rad", 0.05))
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("joint_margin_rad must be finite and nonnegative")
    intervals = definition.get("intervals_rad")
    if not isinstance(intervals, list) or not intervals:
        raise ValueError("articulation_ood.intervals_rad must be nonempty")
    eligible = []
    previous_high = None
    for index, interval in enumerate(intervals):
        low, high = _range(interval, "articulation_ood interval")
        if low == high or not (high < 0.5 - 1e-6 or low > 1.5 + 1e-6):
            raise ValueError("Every OOD interval must have positive width entirely outside [0.5,1.5] rad")
        if previous_high is not None and low < previous_high:
            raise ValueError("OOD intervals must be sorted and non-overlapping")
        previous_high = high
        feasible_low = max(low, joint_bounds[0] + margin)
        feasible_high = min(high, joint_bounds[1] - margin)
        if feasible_high > feasible_low:
            eligible.append((index, feasible_low, feasible_high))
    if not eligible:
        raise ValueError("No feasible OOD articulation interval for joint limits {}".format(joint_bounds))
    if definition.get("prefer_upper", False):
        upper = [entry for entry in eligible if entry[1] > 1.5]
        if upper:
            eligible = upper
    rng = random.Random(stable_seed(sample_seed, "ood_articulation_union"))
    offset = rng.random() * sum(high - low for _, low, high in eligible)
    for index, low, high in eligible:
        if offset < high - low:
            return low + offset, index, [[a, b] for _, a, b in eligible]
        offset -= high - low
    # Rounding at the total length can only affect the last interval endpoint.
    index, low, high = eligible[-1]
    return high, index, [[a, b] for _, a, b in eligible]


def _rotation(sample_seed, definition, sample_index):
    if not isinstance(definition, dict):
        raise ValueError("orientation must define axis, angle_deg, frame and stage")
    if set(definition) - {"axis", "angle_deg", "frame", "stage", "signed"}:
        raise ValueError("Unknown orientation option(s)")
    axis_name = definition.get("axis", "z")
    frame = definition.get("frame", "world")
    stage = definition.get("stage", "placement")
    if not isinstance(definition.get("signed", True), bool):
        raise ValueError("orientation.signed must be Boolean")
    if frame not in ("world", "object") or stage not in ("placement", "lift", "both"):
        raise ValueError("Invalid orientation frame or stage")
    if axis_name not in ("x", "y", "z", "random"):
        raise ValueError("orientation.axis must be x, y, z or random")
    bounds = _range(definition.get("angle_deg", [60, 90]), "orientation.angle_deg")
    if bounds[0] < 0 or bounds[1] > 180:
        raise ValueError("orientation.angle_deg must lie in [0,180]")
    rng = random.Random(stable_seed(sample_seed, "rotation_magnitude"))
    angle = math.radians(rng.uniform(*bounds))
    if definition.get("signed", True):
        sign = (1 if sample_index % 2 == 0 else -1) if sample_index is not None else (
            1 if random.Random(stable_seed(sample_seed, "rotation_sign")).random() < 0.5 else -1)
        angle *= sign
    if axis_name == "random":
        # Uniform on the sphere; uniformly sampled Euler angles are not used.
        rng = random.Random(stable_seed(sample_seed, "rotation_axis"))
        z = rng.uniform(-1.0, 1.0)
        phi = rng.uniform(-math.pi, math.pi)
        radius = math.sqrt(max(0.0, 1.0 - z * z))
        axis = [radius * math.cos(phi), radius * math.sin(phi), z]
    else:
        axis = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}[axis_name]
    return {"axis": axis, "axis_name": axis_name, "angle_rad": angle,
            "frame": frame, "stage": stage, "quaternion_wxyz": axis_angle_quaternion(axis, angle)}


def sample_target(object_name, sample_seed, profile, joint_bounds=None, sample_index=None, normal_definition=None):
    """Sample a paired baseline, then alter only the requested target axes.

    The profile-independent random stream ensures, for example, that a height
    sweep keeps translation, orientation, and articulation identical to control.
    A separate stream samples the additional OOD values.
    """
    allowed = {"lift_m", "translation_distance_m", "translation_direction",
               "translation_directions_deg", "orientation_delta_deg", "orientation",
               "articulation_deg", "articulation_limit_fraction", "family", "severity",
               "articulation_ood", "compare_to", "description", "stress_test", "factors", "factor_flags"}
    unknown = set(profile) - allowed
    if unknown:
        raise ValueError("Unknown target option(s): {}".format(", ".join(sorted(unknown))))
    if "orientation" in profile and "orientation_delta_deg" in profile:
        raise ValueError("Choose quaternion orientation or legacy orientation_delta_deg")
    if sum(key in profile for key in ("articulation_deg", "articulation_limit_fraction", "articulation_ood")) > 1:
        raise ValueError("Choose only one articulation target definition")
    direction_rule = profile.get("translation_direction", "random")
    if direction_rule not in ("random", "control"):
        raise ValueError("translation_direction must be random or control")
    rng = random.Random(stable_seed(sample_seed, "public_target"))
    ketchup = object_name == "ketchup"
    translation = [rng.uniform(-0.1, -0.05), rng.uniform(-0.1, -0.05)]
    delta = rng.uniform(0.5, 1.0 if ketchup else 0.6)
    orientation = rng.uniform(-0.3 if ketchup else -0.4, 0.3 if ketchup else 0.4)
    target = {
        "lift_m": 0.3 if ketchup else 0.15,
        "translation_m": translation,
        "orientation_delta_rad": orientation,
        "articulation_rad": delta,
        "articulation_rule": "absolute" if ketchup else "public_relative",
        "public_delta_rad": delta,
    }
    if normal_definition is not None:
        _paper_normal(target, sample_seed, normal_definition)
    changed_axes = []
    if "lift_m" in profile:
        rng_axis = random.Random(stable_seed(sample_seed, "ood_lift"))
        target["lift_m"] = _sample(rng_axis, profile["lift_m"], "lift_m")
        if target["lift_m"] < 0:
            raise ValueError("lift_m must be nonnegative")
        changed_axes.append("lift")
    if "translation_distance_m" in profile:
        rng_axis = random.Random(stable_seed(sample_seed, "ood_translation"))
        distance = _sample(rng_axis, profile["translation_distance_m"], "translation_distance_m")
        if distance <= 0:
            raise ValueError("translation_distance_m must be positive")
        # Balanced directions test extrapolation independent of a preferred axis.
        # All directions have exactly the sampled norm; no axis can cancel it.
        direction = (math.atan2(translation[1], translation[0]) if direction_rule == "control"
                     else rng_axis.uniform(-math.pi, math.pi))
        target["translation_m"] = [distance * math.cos(direction), distance * math.sin(direction)]
        changed_axes.append("translation")
    if "translation_directions_deg" in profile:
        directions = profile["translation_directions_deg"]
        if not isinstance(directions, list) or not directions or not all(
                isinstance(value, (int, float)) and math.isfinite(value) for value in directions):
            raise ValueError("translation_directions_deg must be a nonempty list of finite angles")
        direction_index = (sample_index % len(directions) if sample_index is not None else
                           random.Random(stable_seed(sample_seed, "direction_bin")).randrange(len(directions)))
        direction_deg = float(directions[direction_index])
        radius = math.hypot(*target["translation_m"])
        direction = math.radians(direction_deg)
        target["translation_m"] = [radius * math.cos(direction), radius * math.sin(direction)]
        target.setdefault("strata", {})["translation_direction_deg"] = direction_deg
        changed_axes.append("translation_direction")
    if "orientation_delta_deg" in profile:
        rng_axis = random.Random(stable_seed(sample_seed, "ood_orientation"))
        target["orientation_delta_rad"] = math.radians(_sample(
            rng_axis, profile["orientation_delta_deg"], "orientation_delta_deg"))
        changed_axes.append("orientation")
    if "orientation" in profile:
        rotation = _rotation(sample_seed, profile["orientation"], sample_index)
        target["orientation_rotation"] = rotation
        target.setdefault("strata", {})["rotation_sign"] = "positive" if rotation["angle_rad"] >= 0 else "negative"
        changed_axes.append("orientation_" + rotation["stage"])
    if "articulation_deg" in profile:
        rng_axis = random.Random(stable_seed(sample_seed, "ood_articulation"))
        target["articulation_rad"] = math.radians(_sample(
            rng_axis, profile["articulation_deg"], "articulation_deg"))
        target["articulation_rule"] = "absolute"
        changed_axes.append("articulation")
    if "articulation_limit_fraction" in profile:
        if joint_bounds is None:
            raise ValueError("articulation_limit_fraction requires URDF joint bounds")
        bounds = _range(profile["articulation_limit_fraction"], "articulation_limit_fraction")
        if bounds[0] < 0 or bounds[1] > 1:
            raise ValueError("articulation_limit_fraction must lie in [0,1]")
        rng_axis = random.Random(stable_seed(sample_seed, "joint_limit_fraction"))
        fraction = rng_axis.uniform(*bounds)
        target["articulation_rad"] = joint_bounds[0] + fraction * (joint_bounds[1] - joint_bounds[0])
        target["articulation_rule"] = "absolute"
        target["articulation_limit_fraction"] = fraction
        changed_axes.append("articulation")
    if "articulation_ood" in profile:
        angle, band, intervals = _random_ood_articulation(sample_seed, profile["articulation_ood"], joint_bounds)
        target.update(articulation_rad=angle, articulation_rule="absolute",
                      articulation_ood_band_index=band, articulation_ood_support_rad=intervals)
        changed_axes.append("articulation")
    target["changed_axes"] = changed_axes
    return target


def _plain_label(label):
    """Normalize NumPy scalar values into stable JSON-compatible label fields."""
    return {
        "seq_num": int(label["seq_num"]),
        "seq_name": str(label["seq_name"]),
        "obj_name": str(label["obj_name"]),
        "init_frame": int(label["init_frame"]),
        "grasp_frame": int(label["grasp_frame"]),
        "kind": str(label["kind"]),
    }


def load_label_sets(repo_root, objects, split="test"):
    """Read existing label dictionaries without importing label_gen's Torch code.

    Selection follows ``label_eval_compose`` for ordinary objects and the
    released ``ketchup_eval`` special approach for ketchup.  The latter uses
    each articulation label again with kind ``right_special`` for grasping.
    ``label_eval_floating`` accidentally removes several objects' articulation
    labels and therefore cannot enumerate this full suite.
    """
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    import numpy as np
    root = Path(repo_root) / "rsc" / "mano_double" / "grasp_label" / "final_labels" / (split + "_set")
    result = {}
    for object_name in objects:
        path = root / (object_name + ".npy")
        labels = [_plain_label(item) for item in np.load(str(path), allow_pickle=True)]
        grasp_kinds = {"left_grasp"} if object_name in ("mixer", "ketchup", "phone") else {
            "left_grasp", "right_grasp", "two_grasp"}
        result[object_name] = {
            "grasp": [item for item in labels if item["kind"] in grasp_kinds],
            "articulation": [item for item in labels if item["kind"] == "right_arti"],
        }
        if object_name == "ketchup":
            result[object_name]["grasp"] = [dict(item, kind="right_special")
                                             for item in result[object_name]["articulation"]]
    return result


def build_plan(repo_root, config, objects=None, profiles=None, samples=None,
               seed=None, split=None, label_sets=None, suite=None):
    """Build a deterministic list of paired episodes; simulation is separate.

    ``label_sets`` is optional so tests can use in-memory labels. Repeated label
    pairs are intentional Monte Carlo samples with independently drawn goals.
    Plans record this sampling design and never pretend all initial states are
    unique. ``articulation_rad`` for a ``public_relative`` target is a provisional
    value for a closed initial hinge; call ``resolve_articulation`` after reset
    and persist the resolved value in each quantitative result.
    """
    selected_objects = list(config.get("objects", ALL_OBJECTS) if objects is None else objects)
    configured_profiles = config.get("profiles", {})
    selected_profiles = select_profiles(config, profiles, suite)
    samples = config.get("samples_per_object", 100) if samples is None else samples
    seed = config.get("seed", 1) if seed is None else seed
    split = config.get("split", "test") if split is None else split
    if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        raise ValueError("samples_per_object must be a positive integer")
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    if not selected_objects or len(selected_objects) != len(set(selected_objects)):
        raise ValueError("Select at least one object, with no duplicate names")
    if not selected_profiles or len(selected_profiles) != len(set(selected_profiles)):
        raise ValueError("Select at least one profile, with no duplicate names")
    unknown = set(selected_objects) - set(ALL_OBJECTS)
    if unknown:
        raise ValueError("Unknown object(s): {}".format(", ".join(sorted(unknown))))
    unknown = set(selected_profiles) - set(configured_profiles)
    if unknown:
        raise ValueError("Unknown profile(s): {}".format(", ".join(sorted(unknown))))
    if label_sets is None:
        label_sets = load_label_sets(repo_root, selected_objects, split)

    episodes = []
    label_counts = {}
    for object_name in selected_objects:
        pools = label_sets[object_name]
        grasp_labels = [_plain_label(item) for item in pools["grasp"]]
        arti_labels = [_plain_label(item) for item in pools["articulation"]]
        if object_name == "ketchup":
            # ketchup_eval.py uses the same articulation frame for the left
            # grasp and subsequent right-hand articulation; independent left
            # grasp labels produce a materially different controller input.
            grasp_labels = [dict(item, kind="right_special") for item in arti_labels]
        if not grasp_labels or not arti_labels:
            raise ValueError("{} {} split needs both grasp and articulation labels ({} / {})".format(
                object_name, split, len(grasp_labels), len(arti_labels)))
        for label in grasp_labels + arti_labels:
            if label["obj_name"] != object_name:
                raise ValueError("Mismatched object in {} labels".format(object_name))
        limits = joint_limits(repo_root, object_name)
        label_counts[object_name] = {"grasp": len(grasp_labels), "articulation": len(arti_labels),
                                     "possible_pairs": (len(arti_labels) if object_name == "ketchup"
                                                        else len(grasp_labels) * len(arti_labels))}
        if object_name == "ketchup":
            label_counts[object_name]["sampling_note"] = "paired articulation label; right_special grasp approach"
        for sample_index in range(samples):
            sample_seed = stable_seed(seed, split, object_name, sample_index)
            label_rng = random.Random(stable_seed(sample_seed, "labels"))
            if object_name == "ketchup":
                arti_index = label_rng.randrange(len(arti_labels))
                grasp_index = arti_index
            else:
                grasp_index = label_rng.randrange(len(grasp_labels))
                arti_index = label_rng.randrange(len(arti_labels))
            for profile_name in selected_profiles:
                definition = copy.deepcopy(configured_profiles[profile_name])
                definition.update(config.get("object_profile_overrides", {}).get(object_name, {}).get(profile_name, {}))
                target = sample_target(object_name, sample_seed, definition, limits, sample_index,
                                       config.get("normal_goal"))
                feasible = articulation_is_feasible(target["articulation_rad"], limits)
                episode = {
                    "episode_id": "{}__{}__{:04d}".format(object_name, profile_name, sample_index),
                    "object": object_name,
                    "profile": profile_name,
                    "sample_index": sample_index,
                    "seed": sample_seed,
                    "split": split,
                    "paper_object": object_name in PAPER_OBJECTS,
                    "grasp_label_index": grasp_index,
                    "arti_label_index": arti_index,
                    "grasp_label": copy.deepcopy(grasp_labels[grasp_index]),
                    "arti_label": copy.deepcopy(arti_labels[arti_index]),
                    "target": target,
                    "joint_limits_rad": limits[:],
                    "target_reachable": feasible,
                    "articulation_within_joint_limits": feasible,
                    "target_reachability_basis": "articulation limits only; spatial reachability not inferred",
                    "articulation_requires_reset_resolution": target["articulation_rule"] == "public_relative",
                }
                protocol = config.get("evaluation_protocol")
                if protocol in ("multiaxis_v1", "factorial_v1"):
                    episode.update({"evaluation_protocol": protocol,
                                    "require_lift_pose_success": protocol == "multiaxis_v1",
                                    "family": definition.get("family", "unspecified"),
                                    "severity": definition.get("severity"),
                                    "compare_to": definition.get("compare_to", "id_control"),
                                    "stress_test": bool(definition.get("stress_test", False)),
                                    "articulation_randomized_target_ood": (
                                        articulation_outside_randomized_training(target["articulation_rad"])
                                        if target["articulation_rule"] == "absolute" else None)})
                if protocol == "factorial_v1":
                    flags = definition.get("factor_flags", {})
                    if set(flags) != {"horizontal", "lift", "articulation"} or not all(
                            isinstance(value, bool) for value in flags.values()):
                        raise ValueError("factorial_v1 requires Boolean horizontal/lift/articulation factor_flags")
                    expected_axes = {axis for factor, axis in (("horizontal", "translation"),
                                     ("lift", "lift"), ("articulation", "articulation")) if flags[factor]}
                    if set(target["changed_axes"]) != expected_axes:
                        raise ValueError("factor_flags must agree with the actual target overrides")
                    episode["factor_flags"] = copy.deepcopy(flags)
                episodes.append(episode)
    effective_config = copy.deepcopy(config)
    effective_config.update({
        "objects": selected_objects[:],
        "profiles": {name: copy.deepcopy(configured_profiles[name]) for name in selected_profiles},
        "samples_per_object": samples,
        "seed": int(seed),
        "split": split,
    })
    effective_config["selected_suite"] = suite or (config.get("default_suite") if profiles is None else None)
    if "default_suite" in effective_config:
        effective_config["default_suite"] = effective_config["selected_suite"]
    return {
        "schema_version": 1,
        "config": effective_config,
        "seed": int(seed),
        "split": split,
        "objects": selected_objects,
        "profiles": selected_profiles,
        "samples_per_object": samples,
        "episodes": episodes,
        "label_counts": label_counts,
        "sampling": "label pairs sampled with replacement; independent random targets; paired across profiles",
        "ood_reference": (
            "Spatial targets extrapolate from the public composed-evaluation target envelope. "
            "The full training support is not published. Articulation OOD is relative to "
            "the public randomized training target range [0.5, 1.5] radians. "
            "URDF limits remain unchanged and infeasible requests remain in the denominator."
        ),
    }
