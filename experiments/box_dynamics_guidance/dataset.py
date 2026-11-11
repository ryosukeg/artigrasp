"""Verified loading of fixed-PPO, 20-step box response windows.

This module deliberately does not filter on task success. Numerically valid
failures are useful response-model training examples.
"""

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np


SCHEMA_VERSION = 1
PHASE_START = 1260
TOTAL_STEPS = 1560
HORIZON = 20
INPUT_DIM = 571
OUTPUT_DIM = 9
SPLITS = ("train", "val", "audit", "calibration")

FEATURE_SPEC = {
    "input_dim": INPUT_DIM,
    "inputs": [
        {"name": "start_obs_right_raw", "slice": [0, 276]},
        {"name": "start_obs_left_raw", "slice": [276, 552]},
        {"name": "start_object_position_quat_wxyz_angle", "slice": [552, 560],
         "source": "final_states[start_step - 1260, 122:130]"},
        {"name": "original_goal_position_quat_wxyz_angle", "slice": [560, 568]},
        {"name": "constant_right_wrist_action_residual_z", "slice": [568, 571]},
    ],
    "output_dim": OUTPUT_DIM,
    "outputs": [
        "end_articulation_angle_rad", "end_articulation_velocity_rad_s",
        "end_base_position_x_m", "end_base_position_y_m", "end_base_position_z_m",
        "end_base_quaternion_w", "end_base_quaternion_x",
        "end_base_quaternion_y", "end_base_quaternion_z",
    ],
    "output_sources": {
        "angle_velocity": "final_obs_r[end_step - 1260, 273:275]",
        "base_position_quaternion": "final_states[end_step - 1260, 122:129]",
    },
    "horizon_steps": HORIZON,
    "phase_start_step": PHASE_START,
    "quaternion_representation": (
        "Direct wxyz components, no rotation-invariant loss. Start quaternion is "
        "recorded directly; end and goal quaternion signs are aligned to the start "
        "quaternion within each sample. Raw observations are unchanged."
    ),
    "persistence_baseline": "Start angle, angular velocity, base position and quaternion.",
    "task_success_filter": False,
}


class DatasetError(ValueError):
    """The dataset is incomplete, inconsistent, or violates its schema."""


@dataclass
class WindowSplit:
    x: np.ndarray
    y: np.ndarray
    persistence: np.ndarray
    episode_ids: np.ndarray
    window_start_steps: np.ndarray


@dataclass
class LoadedDataset:
    root: Path
    splits: Dict[str, WindowSplit]
    metadata: Dict[str, Any]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path):
    try:
        with Path(path).open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, ValueError) as exc:
        raise DatasetError("Cannot read {}: {}".format(path, exc)) from exc
    if not isinstance(value, dict):
        raise DatasetError("{} must contain a JSON object".format(path))
    return value


def _rows_by_id(rows, context):
    if not isinstance(rows, list):
        raise DatasetError("{}.episodes must be a list".format(context))
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("episode_id"), str):
            raise DatasetError("{} has an invalid episode row".format(context))
        episode_id = row["episode_id"]
        if not episode_id or episode_id in result:
            raise DatasetError("{} has duplicate/empty episode_id {}".format(context, episode_id))
        result[episode_id] = row
    return result


def _reference_key(label):
    if not isinstance(label, dict):
        return None
    subject = label.get("seq_num", label.get("subject"))
    sequence = label.get("seq_name", label.get("sequence"))
    if subject is None or sequence is None:
        return None
    return (str(subject), str(sequence))


def _check_references(plan, rows):
    """Prevent train/val leakage at subject + sequence level when labels exist."""
    observed = {"train": set(), "val": set()}
    reference_split = plan.get("reference_split")
    groups = {}
    if reference_split is not None:
        if not isinstance(reference_split, dict) or reference_split.get("official_split") != "train":
            raise DatasetError("reference_split must use the official train split")
        if reference_split.get("test_split_used") is not False:
            raise DatasetError("reference_split must explicitly exclude the test split")
        if not isinstance(reference_split.get("groups"), list):
            raise DatasetError("reference_split.groups must be a list")
        for group in reference_split["groups"]:
            key = _reference_key(group)
            if key is None or key in groups or group.get("dataset_split") not in SPLITS:
                raise DatasetError("Invalid or duplicate reference group")
            groups[key] = group
    checked = 0
    membership_checked = 0
    for row in rows.values():
        split = row.get("dataset_split")
        # source split is the official dataset split, not the model train/val split.
        if row.get("split", "train") != "train":
            raise DatasetError("Official test references cannot enter training/validation")
        for name, role in (("grasp_label", "grasp"), ("arti_label", "articulation")):
            label = row.get(name)
            key = _reference_key(label)
            if reference_split is not None:
                group = groups.get(key)
                if group is None or group["dataset_split"] != split:
                    raise DatasetError("Episode reference does not belong to its planned split")
                if not any(reference.get("role") == role and reference.get("label") == label
                           for reference in group.get("references", [])):
                    raise DatasetError("Episode label is absent from reference_split group")
                membership_checked += 1
            if key is not None and split in observed:
                observed[split].add(key)
                checked += 1
    overlap = observed["train"] & observed["val"]
    if overlap:
        raise DatasetError("Train/val reference sequence overlap: {}".format(sorted(overlap)))
    return {
        "plan_reference_split_present": "reference_split" in plan,
        "reference_memberships_checked": membership_checked,
        "observed_reference_labels_checked": checked,
        "train_reference_sequences": sorted([list(x) for x in observed["train"]]),
        "val_reference_sequences": sorted([list(x) for x in observed["val"]]),
        "status": ("manifest_membership_verified" if membership_checked else
                   "observed_labels_disjoint" if checked else "no_reference_labels_available"),
    }


def _shape(array, expected, key, episode_id):
    if array.shape != expected:
        raise DatasetError("{}: {} shape {}, expected {}".format(
            episode_id, key, array.shape, expected))
    if array.dtype.kind not in "biuf":
        raise DatasetError("{}: {} must be a numeric array".format(episode_id, key))


def _flag(array, expected, key, episode_id):
    _shape(array, expected, key, episode_id)
    if array.dtype.kind != "b":
        raise DatasetError("{}: {} must have boolean dtype".format(episode_id, key))


def _load_episode(path, episode_id):
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "states": (1561, 138), "actions_r": (1560, 51),
                "actions_l": (1560, 51), "ppo_actions_r": (1560, 51),
                "ppo_actions_l": (1560, 51), "residuals": (1560, 3),
                "final_obs_r": (301, 276), "final_obs_l": (301, 276),
                "final_states": (301, 138), "modes_r": (1560,),
                "modes_l": (1560,), "window_start_step": (15,),
                "window_end_step": (15,), "window_z": (15, 3), "goal": (8,),
                "schema_version": (), "horizon": (), "control_dt": (),
            }
            flags = {name: (1560,) for name in (
                "bad_observation", "bad_action", "bad_state", "auto_reset")}
            flags.update({"window_valid": (15,), "initial_invalid": (),
                          "terminal_bad_observation": ()})
            missing = (set(required) | set(flags)) - set(archive.files)
            if missing:
                raise DatasetError("{}: missing arrays {}".format(episode_id, sorted(missing)))
            data = {key: archive[key] for key in set(required) | set(flags)}
            for key, expected in required.items():
                _shape(data[key], expected, key, episode_id)
            for key, expected in flags.items():
                _flag(data[key], expected, key, episode_id)
    except (OSError, ValueError) as exc:
        if isinstance(exc, DatasetError):
            raise
        raise DatasetError("Cannot load {}: {}".format(path, exc)) from exc

    starts = np.arange(PHASE_START, TOTAL_STEPS, HORIZON)
    if data["schema_version"].item() != SCHEMA_VERSION:
        raise DatasetError("{}: unsupported trajectory schema_version".format(episode_id))
    if data["horizon"].item() != HORIZON or not np.isclose(data["control_dt"].item(), .01):
        raise DatasetError("{}: expected horizon=20 and control_dt=0.01".format(episode_id))
    if not np.array_equal(data["window_start_step"], starts):
        raise DatasetError("{}: window starts must be 1260,1280,...,1540".format(episode_id))
    if not np.array_equal(data["window_end_step"], starts + HORIZON):
        raise DatasetError("{}: windows must contain exactly 20 actions".format(episode_id))
    return data


def _window(data, index, episode_id):
    start = int(data["window_start_step"][index])
    end = int(data["window_end_step"][index])
    local_start, local_end = start - PHASE_START, end - PHASE_START
    reasons = []
    if bool(data["initial_invalid"]):
        reasons.append("initial_invalid")
    if not data["window_valid"][index]:
        reasons.append("collector_window_invalid")
    for key in ("bad_observation", "bad_action", "bad_state", "auto_reset"):
        if np.any(data[key][:end]):
            reasons.append("prior_or_current_" + key)
    if end == TOTAL_STEPS and bool(data["terminal_bad_observation"]):
        reasons.append("terminal_bad_observation")
    for key in ("final_obs_r", "final_obs_l", "final_states"):
        if not np.isfinite(data[key][local_start:local_end + 1]).all():
            reasons.append("nonfinite_" + key)
    for key in ("actions_r", "actions_l", "ppo_actions_r", "ppo_actions_l", "residuals"):
        if not np.isfinite(data[key][start:end]).all():
            reasons.append("nonfinite_" + key)
    if not np.isfinite(data["window_z"][index]).all() or not np.isfinite(data["goal"]).all():
        reasons.append("nonfinite_goal_or_z")
    if reasons:
        return None, reasons

    z = data["window_z"][index]
    if not np.all(data["modes_r"][start:end] == 8):
        raise DatasetError("{}: valid final window is not entirely right mode 8".format(episode_id))
    if not np.allclose(data["residuals"][start:end], z, atol=1e-7, rtol=1e-6):
        raise DatasetError("{}: residual changes inside window {}".format(episode_id, start))
    expected_r = data["ppo_actions_r"][start:end].copy()
    expected_r[:, :3] += z
    if not np.allclose(data["actions_r"][start:end], expected_r, atol=2e-6, rtol=1e-5):
        raise DatasetError("{}: executed right actions do not match PPO + z".format(episode_id))
    if not np.allclose(data["actions_l"][start:end], data["ppo_actions_l"][start:end],
                       atol=2e-6, rtol=1e-5):
        raise DatasetError("{}: left actions changed".format(episode_id))

    start_object = data["final_states"][local_start, 122:130].copy()
    end_object = data["final_states"][local_end, 122:130].copy()
    goal = data["goal"].copy()
    start_quat = start_object[3:7]
    if any(np.linalg.norm(q) < 1e-8 for q in (start_quat, end_object[3:7], goal[3:7])):
        return None, ["invalid_zero_quaternion"]
    if np.dot(end_object[3:7], start_quat) < 0:
        end_object[3:7] *= -1
    if np.dot(goal[3:7], start_quat) < 0:
        goal[3:7] *= -1
    x = np.concatenate((data["final_obs_r"][local_start], data["final_obs_l"][local_start],
                        start_object, goal, z)).astype(np.float32)
    y = np.concatenate((data["final_obs_r"][local_end, 273:275], end_object[:7])).astype(np.float32)
    persistence = np.concatenate((data["final_obs_r"][local_start, 273:275],
                                  start_object[:7])).astype(np.float32)
    if not all(np.isfinite(array).all() for array in (x, y, persistence)):
        return None, ["nonfinite_float32_conversion"]
    return (x, y, persistence, start), []


def load_dataset(path: Union[str, Path]) -> LoadedDataset:
    """Read a completed schema-v1 dataset, verifying hashes before loading NPZs.

    Returns ``LoadedDataset.splits[name]`` with x(N,571), y(N,9), persistence
    predictions, episode IDs and absolute window start steps. Audit/calibration
    windows remain separate and must not be used to fit normalization or weights.
    """
    supplied = Path(path).expanduser().resolve()
    root = supplied.parent if supplied.name == "dataset.json" else supplied
    manifest_path = root / "dataset.json"
    plan_path = root / "plan.json"
    manifest, plan = _read_json(manifest_path), _read_json(plan_path)
    for name, document in (("dataset", manifest), ("plan", plan)):
        if document.get("schema_version") != SCHEMA_VERSION:
            raise DatasetError("Unsupported {} schema_version".format(name))
    if manifest.get("complete") is not True:
        raise DatasetError("Dataset completion marker is missing or false")
    plan_hash = file_sha256(plan_path)
    if manifest.get("plan_sha256") != plan_hash:
        raise DatasetError("plan.json SHA256 mismatch")
    rows = _rows_by_id(manifest.get("episodes"), "dataset")
    planned = _rows_by_id(plan.get("episodes"), "plan")
    if set(rows) != set(planned):
        raise DatasetError("Completed dataset episode IDs differ from plan")
    for episode_id, row in rows.items():
        split = row.get("dataset_split")
        if split not in SPLITS or planned[episode_id].get("dataset_split") != split:
            raise DatasetError("{}: dataset split is invalid or differs from plan".format(episode_id))
        for key in ("grasp_label", "arti_label", "residuals", "target"):
            if key in planned[episode_id] and row.get(key) != planned[episode_id][key]:
                raise DatasetError("{}: {} differs from plan".format(episode_id, key))
    reference_check = _check_references(plan, planned)

    collected = {split: [] for split in SPLITS}
    skipped = Counter()
    episodes_by_split = Counter()
    trajectory_paths = set()
    for episode_id, row in rows.items():
        trajectory_name = row.get("trajectory_path")
        if not isinstance(trajectory_name, str):
            raise DatasetError("{}: missing trajectory_path".format(episode_id))
        trajectory = (root / trajectory_name).resolve()
        try:
            trajectory.relative_to(root)
        except ValueError as exc:
            raise DatasetError("Trajectory path escapes dataset directory") from exc
        if trajectory in trajectory_paths:
            raise DatasetError("Duplicate trajectory_path in dataset")
        trajectory_paths.add(trajectory)
        try:
            actual_hash = file_sha256(trajectory)
        except OSError as exc:
            raise DatasetError("Missing trajectory {}".format(trajectory)) from exc
        if row.get("trajectory_sha256") != actual_hash:
            raise DatasetError("{}: trajectory SHA256 mismatch".format(episode_id))
        data = _load_episode(trajectory, episode_id)
        if "residuals" in planned[episode_id]:
            scheduled = np.asarray(planned[episode_id]["residuals"])
            if scheduled.shape != (15, 3) or not np.allclose(
                    data["window_z"], scheduled, atol=1e-7, rtol=1e-6):
                raise DatasetError("{}: recorded residual schedule differs from plan".format(episode_id))
        split = row["dataset_split"]
        episodes_by_split[split] += 1
        for index in range(15):
            window, reasons = _window(data, index, episode_id)
            if reasons:
                skipped.update(reasons)
            else:
                collected[split].append((episode_id, window))

    result = {}
    for split, samples in collected.items():
        result[split] = WindowSplit(
            x=np.stack([window[0] for _, window in samples]) if samples else np.empty((0, INPUT_DIM), np.float32),
            y=np.stack([window[1] for _, window in samples]) if samples else np.empty((0, OUTPUT_DIM), np.float32),
            persistence=np.stack([window[2] for _, window in samples]) if samples else np.empty((0, OUTPUT_DIM), np.float32),
            episode_ids=np.asarray([episode_id for episode_id, _ in samples], dtype=str),
            window_start_steps=np.asarray([window[3] for _, window in samples], dtype=np.int64),
        )
    valid_count = sum(len(split.x) for split in result.values())
    metadata = {
        "dataset_path": str(root), "schema_version": SCHEMA_VERSION,
        "dataset_manifest_sha256": file_sha256(manifest_path), "plan_sha256": plan_hash,
        "episode_counts": dict(episodes_by_split),
        "window_counts": {split: len(data.x) for split, data in result.items()},
        "total_windows": len(rows) * 15, "valid_windows": valid_count,
        "excluded_windows": len(rows) * 15 - valid_count,
        "excluded_reason_counts": dict(skipped),
        "exclusion_counts_can_overlap": True,
        "reference_validation": reference_check,
        "feature_spec": FEATURE_SPEC,
    }
    return LoadedDataset(root=root, splits=result, metadata=metadata)
