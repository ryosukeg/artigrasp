"""Angle/position delta teachers with the absolute model's loss weights.

Only output indices 0, 2, 3 and 4 change representation. Their normalized
network outputs decode as ``persistence + absolute_train_std * output``.
Velocity and sign-aligned quaternion components remain absolute targets.
"""

from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np

try:
    from .dataset import FEATURE_SPEC, OUTPUT_DIM, DatasetError, file_sha256
    from .train_smoke import _normalization
except ImportError:
    from dataset import FEATURE_SPEC, OUTPUT_DIM, DatasetError, file_sha256
    from train_smoke import _normalization


DELTA_INDICES = np.asarray([0, 2, 3, 4], dtype=np.int64)
TARGET_SPEC = {
    "representation": "delta_angle_position",
    "delta_indices": DELTA_INDICES.tolist(),
    "absolute_indices": [1, 5, 6, 7, 8],
    "definition": "target[j] = future[j] - persistence[j] for delta_indices; future[j] otherwise",
    "normalization": {
        "fit_split": "train",
        "center": "zero for delta_indices; original absolute training mean otherwise",
        "scale": "original absolute training standard deviation for every output",
        "constant_dimension_scale": 1.0,
        "constant_dimension_threshold": 1e-6,
    },
    "decoder": {
        "network_to_target": "target = network_output * absolute_train_std + center",
        "target_to_absolute": "future[j] = target[j] + persistence[j] for delta_indices; target[j] otherwise",
    },
    "quaternion_representation": "Unchanged sign-aligned absolute wxyz components",
}
DELTA_FEATURE_SPEC = deepcopy(FEATURE_SPEC)
DELTA_FEATURE_SPEC["target_representation"] = deepcopy(TARGET_SPEC)


def _values(values, name):
    array = np.asarray(values)
    if array.ndim < 1 or array.shape[-1] != OUTPUT_DIM:
        raise DatasetError("{} must have final dimension {}".format(name, OUTPUT_DIM))
    if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
        raise DatasetError("{} must contain finite floating-point values".format(name))
    return array


def _pair(values, persistence):
    values = _values(values, "values")
    persistence = _values(persistence, "persistence")
    if values.shape != persistence.shape:
        raise DatasetError("Values and persistence must have identical shapes")
    return values, persistence


def make_targets(split):
    """Return a copy of split.y with angle and position made relative to start."""
    future, persistence = _pair(split.y, split.persistence)
    target = future.copy()
    target[..., DELTA_INDICES] -= persistence[..., DELTA_INDICES]
    if not np.isfinite(target).all():
        raise DatasetError("Non-finite delta targets")
    return target


def decode_targets(values, persistence):
    """Decode unnormalized mixed targets into a new absolute-state array."""
    values, persistence = _pair(values, persistence)
    future = values.copy()
    future[..., DELTA_INDICES] += persistence[..., DELTA_INDICES]
    if not np.isfinite(future).all():
        raise DatasetError("Non-finite decoded targets")
    return future


def target_normalization(train_y):
    """Return (mixed center, absolute std, absolute mean), fit on train only.

    Pass the original absolute train.y, never make_targets(train). Preserving
    its standard deviations preserves every physical squared-error weight.
    A zero normalized output reproduces persistence for the four delta outputs.
    """
    train_y = _values(train_y, "absolute training targets")
    if train_y.ndim != 2 or not len(train_y):
        raise DatasetError("Normalization requires nonempty (N,9) absolute training targets")
    absolute_mean, scale = _normalization(train_y)
    center = absolute_mean.copy()
    center[DELTA_INDICES] = 0.0
    if not all(np.isfinite(value).all() for value in (center, scale, absolute_mean)) or np.any(scale <= 0):
        raise DatasetError("Non-finite or nonpositive target normalization")
    return center, scale, absolute_mean


def _sync_directory(path):
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_npz(path, arrays):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _sync_directory(path.parent)


def _write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _sync_directory(path.parent)


def _archive_arrays(split, name):
    target = make_targets(split)
    if target.ndim != 2 or not len(target):
        raise DatasetError("{} requires nonempty (N,9) targets".format(name))
    episode_ids = np.asarray(split.episode_ids, dtype=str)
    steps = np.asarray(split.window_start_steps)
    if episode_ids.shape != (len(target),) or not np.all(episode_ids != ""):
        raise DatasetError("{} has invalid episode IDs".format(name))
    if steps.shape != (len(target),) or not np.issubdtype(steps.dtype, np.integer):
        raise DatasetError("{} has invalid window start steps".format(name))
    return {
        "target": target,
        "absolute_target": np.asarray(split.y).copy(),
        "persistence": np.asarray(split.persistence).copy(),
        "episode_ids": episode_ids,
        "window_start_steps": steps.astype(np.int64, copy=True),
    }


def save_targets(dataset, output_path):
    """Durably save train/val teachers from an already verified LoadedDataset.

    The caller owns initial loading and verification via load_dataset. This
    function never edits the source dataset. A completed manifest is published
    only after both pickle-free NPZs are durable. Existing nonempty output
    directories (including interrupted attempts) are deliberately rejected.
    """
    output = Path(output_path).expanduser().resolve()
    source = Path(dataset.root).expanduser().resolve()
    if output == source or source in output.parents:
        raise ValueError("Target artifacts must be outside the source dataset directory")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Use a fresh or empty target output directory")
    if any(name not in dataset.splits for name in ("train", "val")):
        raise DatasetError("Target export requires train and val splits")
    arrays = {name: _archive_arrays(dataset.splits[name], name) for name in ("train", "val")}
    center, scale, absolute_mean = target_normalization(dataset.splits["train"].y)
    identity_keys = ("dataset_path", "dataset_manifest_sha256", "plan_sha256")
    if any(key not in dataset.metadata for key in identity_keys):
        raise DatasetError("Source dataset identity is incomplete")
    identity = {key: deepcopy(dataset.metadata[key]) for key in identity_keys}
    module_directory = Path(__file__).resolve().parent
    sources = {name: file_sha256(module_directory / name)
               for name in ("delta_targets.py", "dataset.py", "train_smoke.py")}

    output.mkdir(parents=True, exist_ok=True)
    # Exclusive marker prevents concurrent writers from publishing to one path.
    marker = output / ".writing"
    with marker.open("x", encoding="utf-8") as stream:
        stream.write("Incomplete target export; manifest.json is the completion marker.\n")
        stream.flush()
        os.fsync(stream.fileno())
    _sync_directory(output)
    split_info = {}
    for name, values in arrays.items():
        path = output / (name + ".npz")
        _write_npz(path, values)
        split_info[name] = {
            "path": path.name,
            "sha256": file_sha256(path),
            "window_count": len(values["target"]),
            "episode_count": len(np.unique(values["episode_ids"])),
        }
    manifest = {
        "schema_version": 1,
        "complete": True,
        "artifact_type": "box_response_delta_targets",
        "dataset_identity": identity,
        "target_spec": deepcopy(TARGET_SPEC),
        "feature_spec": deepcopy(DELTA_FEATURE_SPEC),
        "normalization": {
            "fit_split": "train",
            "center": center.tolist(),
            "scale": scale.tolist(),
            "absolute_mean": absolute_mean.tolist(),
        },
        "splits": split_info,
        "sources": sources,
    }
    _write_json(output / "manifest.json", manifest)
    marker.unlink()
    _sync_directory(output)
    return manifest
