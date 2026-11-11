#!/usr/bin/env python3
"""Compare absolute and delta predictors on the same original validation windows.

This reads completed training artifacts and the verified source dataset. It does
not train a model, execute the simulator, or evaluate OOD policy performance.
"""

import argparse
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

try:
    from .dataset import FEATURE_SPEC, file_sha256, load_dataset
    from .delta_targets import DELTA_FEATURE_SPEC, TARGET_SPEC, target_normalization
    from .train_delta import _predict as predict_delta, _prepare_targets
    from .train_model import ResponseMLP, _predict as predict_absolute, _now
    from .train_smoke import _metrics, _normalization
except ImportError:
    from dataset import FEATURE_SPEC, file_sha256, load_dataset
    from delta_targets import DELTA_FEATURE_SPEC, TARGET_SPEC, target_normalization
    from train_delta import _predict as predict_delta, _prepare_targets
    from train_model import ResponseMLP, _predict as predict_absolute, _now
    from train_smoke import _metrics, _normalization


HERE = Path(__file__).resolve().parent
BATCH_SIZE = 256
METRIC_RTOL = 1e-6
METRIC_ATOL = 1e-8


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _equal_arrays(left, right, context):
    _require(set(left) == set(right), context + ": keys differ")
    for key in left:
        _require(np.array_equal(_array(left[key]), _array(right[key])), context + ": " + key + " differs")


def _local_file(directory, name):
    path = (directory / name).resolve()
    _require(path.parent == directory.resolve(), "Unexpected artifact/source path: " + name)
    return path


def _load_run(directory, delta):
    directory = Path(directory).expanduser().resolve()
    report_path = directory / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(report.get("schema_version") == 1 and report.get("status") == "completed",
             "Run must have a completed schema-1 report: " + str(directory))
    _require(report.get("evaluated_checkpoint") == "best.pt", "Report must evaluate best.pt")
    hashes = {str(report_path): file_sha256(report_path)}
    artifacts = report["artifact_sha256"]
    _require({"best.pt", "last.pt", "history.jsonl"} <= set(artifacts), "Missing training artifact hashes")
    for name, expected in artifacts.items():
        path = _local_file(directory, name)
        _require(file_sha256(path) == expected, "Training artifact hash mismatch: " + str(path))
        hashes[str(path)] = expected
    for name, expected in report["sources"].items():
        path = _local_file(HERE, name)
        _require(file_sha256(path) == expected, "Training source hash mismatch: " + str(path))
        hashes[str(path)] = expected
    best = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    last = torch.load(directory / "last.pt", map_location="cpu", weights_only=True)
    prefix = "box_response_delta" if delta else "box_response"
    for checkpoint, suffix in ((best, "best_predictor"), (last, "training_resume")):
        _require(checkpoint.get("schema_version") == 1 and
                 checkpoint.get("artifact_type") == prefix + "_" + suffix,
                 "Incorrect checkpoint type in " + str(directory))
        for field in ("config", "sources"):
            _require(checkpoint[field] == report[field], "Report/checkpoint " + field + " mismatch")
        _require(checkpoint["model_config"] == report["model"], "Report/checkpoint model mismatch")
        _require(checkpoint["best_epoch"] == report["best_epoch"], "Report/checkpoint best epoch mismatch")
        _require(checkpoint["feature_spec"] == (DELTA_FEATURE_SPEC if delta else FEATURE_SPEC),
                 "Unexpected feature/target specification")
    _require(best["dataset_identity"] == last["dataset_identity"], "Best/last dataset identities differ")
    _equal_arrays(best["normalization"], last["normalization"], "Best/last normalization")
    _equal_arrays(best["model_state_dict"], last["best_model_state_dict"], "Best/last selected weights")
    history = [json.loads(line) for line in (directory / "history.jsonl").read_text().splitlines() if line.strip()]
    _require(history == last["history"], "History file differs from committed last checkpoint")
    _require(len(history) == last["completed_epoch"] == report["completed_epochs"], "Completed epoch mismatch")
    _require(last["requested_max_epochs"] == report["requested_max_epochs"], "Requested epoch limit mismatch")
    _require(bool(history) and 1 <= best["best_epoch"] <= len(history), "Invalid best epoch")
    _require(best["best_val_normalized_mse"] == last["best_val_normalized_mse"] ==
             report["best_val_normalized_mse"] == min(row["val_normalized_mse"] for row in history),
             "Best validation score does not match committed history")
    if delta:
        for source in (report, best, last):
            _require(source["target_spec"] == TARGET_SPEC, "Delta target specification mismatch")
        _require(report["target_identity"] == best["target_identity"] == last["target_identity"],
                 "Delta teacher identities disagree")
    return {"directory": directory, "report": report, "best": best, "last": last, "hashes": hashes}


def _optimizer_info(run, train_windows):
    last = run["last"]
    optimizer = last["optimizer_state_dict"]
    groups = optimizer["param_groups"]
    parameter_ids = [key for group in groups for key in group["params"]]
    _require(set(parameter_ids) == set(optimizer["state"]), "Missing Adam parameter state")
    steps = [float(_array(optimizer["state"][key]["step"]).item()) for key in parameter_ids]
    _require(bool(steps) and all(math.isfinite(step) and step >= 0 and step.is_integer() for step in steps),
             "Invalid Adam update counter")
    _require(len(set(steps)) == 1, "Adam update counts differ across parameters")
    step_count = int(steps[0])
    batches_per_epoch = math.ceil(train_windows / last["config"]["batch_size"])
    _require(step_count == batches_per_epoch * last["completed_epoch"],
             "Actual Adam steps disagree with completed epochs and batch size")
    return {
        "optimizer": "Adam", "parameter_groups": [dict(group) for group in groups],
        "actual_update_steps": step_count, "parameter_step_counters": [int(step) for step in steps],
        "batches_per_epoch": batches_per_epoch,
        "completed_epochs": last["completed_epoch"], "best_epoch": last["best_epoch"],
        "requested_max_epochs": last["requested_max_epochs"],
        "elapsed_seconds_last_checkpoint": last["elapsed_seconds"],
        "elapsed_seconds_final_report": run["report"]["elapsed_seconds"],
        "summed_epoch_seconds": sum(row["epoch_seconds"] for row in last["history"]),
        "stop_reason": run["report"]["stop_reason"],
    }


def _runtime(config):
    """Match training inference settings; record an explicit CPU fallback."""
    requested = torch.device(config["device"])
    actual = requested
    fallback = None
    if requested.type == "cuda" and (not torch.cuda.is_available() or
                                     (requested.index is not None and requested.index >= torch.cuda.device_count())):
        actual = torch.device("cpu")
        fallback = "Recorded CUDA device is unavailable; both predictors evaluated on CPU."
    _require(actual.type in ("cpu", "cuda"), "Only CPU/CUDA inference is supported")
    if actual.type == "cuda":
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = config["cublas_workspace_config"]
        _require(os.environ["CUBLAS_WORKSPACE_CONFIG"] in (":4096:8", ":16:8"), "Invalid deterministic CUDA workspace")
        torch.cuda.set_device(actual)
    torch.set_num_threads(config["torch_cpu_threads"])
    torch.use_deterministic_algorithms(config["deterministic_algorithms"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if actual.type == "cuda":
        torch.cuda.manual_seed(config["seed"])
    info = {"requested_training_device": str(requested), "actual_device": str(actual),
            "fallback": fallback, "batch_size": BATCH_SIZE,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "torch_cpu_threads": torch.get_num_threads(), "tf32_enabled": False,
            "torch_version": str(torch.__version__), "numpy_version": str(np.__version__),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG") if actual.type == "cuda" else None}
    info["training_library_versions_match"] = (
        info["torch_version"] == config["torch_version"] and info["numpy_version"] == config["numpy_version"])
    if actual.type == "cuda":
        info["cuda_device_name"] = torch.cuda.get_device_name(actual)
        info["cuda_capability"] = list(torch.cuda.get_device_capability(actual))
        info["training_hardware_matches"] = (info["cuda_device_name"] == config.get("cuda_device_name") and
                                              info["cuda_capability"] == config.get("cuda_capability"))
    return actual, info


def _metric_check(recalculated, recorded):
    _require(set(recalculated) == set(recorded), "Recorded metric fields differ")
    differences = {}
    for key in recalculated:
        new, old = np.asarray(recalculated[key]), np.asarray(recorded[key])
        _require(new.shape == old.shape and np.allclose(new, old, rtol=METRIC_RTOL, atol=METRIC_ATOL),
                 "Recalculated metric exceeds report tolerance: " + key)
        differences[key] = float(np.max(np.abs(new - old)))
    return {"matched": True, "rtol": METRIC_RTOL, "atol": METRIC_ATOL,
            "maximum_absolute_difference_by_metric": differences}


def _changes(new, reference):
    result = {}
    for key in new:
        newer, older = np.asarray(new[key], dtype=float), np.asarray(reference[key], dtype=float)
        change = newer - older
        relative = np.divide(100.0 * change, older, out=np.full_like(change, np.nan), where=older != 0)
        values = relative.tolist()
        values = ([None if not math.isfinite(value) else value for value in values]
                  if isinstance(values, list) else None if not math.isfinite(values) else values)
        result[key] = {"absolute_change": change.tolist(), "percent_change": values}
    return result


def compare(absolute_dir, delta_dir, output):
    output = Path(output).expanduser().resolve()
    _require(output.suffix.lower() == ".json", "--output must name a new JSON file")
    predictions_path = output.with_suffix(".predictions.npz")
    if output.exists() or predictions_path.exists():
        raise FileExistsError("Comparison output exists; choose a new JSON path")
    absolute, delta = _load_run(absolute_dir, False), _load_run(delta_dir, True)
    absolute_config, delta_config = absolute["best"]["config"], delta["best"]["config"]
    _require(absolute_config["target_representation"] == "absolute" and
             delta_config["target_representation"] == "delta_angle_position", "Wrong target representations")
    _require({k: v for k, v in absolute_config.items() if k != "target_representation"} ==
             {k: v for k, v in delta_config.items() if k != "target_representation"},
             "Training configurations differ beyond target representation")
    _require(absolute["best"]["model_config"] == delta["best"]["model_config"], "Model capacities differ")
    _require(absolute["report"]["dataset"] == delta["report"]["dataset"], "Dataset metadata/splits differ")
    identity = absolute["best"]["dataset_identity"]
    _require(identity == delta["best"]["dataset_identity"], "Training dataset identities differ")
    _require(absolute["last"]["requested_max_epochs"] == delta["last"]["requested_max_epochs"],
             "Training epoch limits differ")
    # Expensive trajectory verification is performed once, shared by both models.
    dataset = load_dataset(identity["dataset_path"])
    _require(dataset.metadata == absolute["report"]["dataset"], "Verified dataset differs from training metadata")
    _require({key: dataset.metadata[key] for key in identity} == identity, "Verified dataset identity differs")
    train, val = dataset.splits["train"], dataset.splits["val"]
    _require(len(train.x) > 0 and len(val.x) > 0, "Nonempty original train/val splits are required")
    normal_a = {key: _array(value) for key, value in absolute["best"]["normalization"].items()}
    normal_d = {key: _array(value) for key, value in delta["best"]["normalization"].items()}
    for key in ("x_mean", "x_std", "y_std"):
        _require(np.array_equal(normal_a[key], normal_d[key]), "Normalization differs: " + key)
    _require(np.array_equal(normal_a["y_mean"], normal_d["absolute_y_mean"]), "Absolute target means differ")
    x_mean, x_std = _normalization(train.x)
    y_mean, y_std = _normalization(train.y)
    _equal_arrays(normal_a, {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std},
                  "Original normalization against train-only data")
    d_mean, d_std, absolute_mean = target_normalization(train.y)
    _equal_arrays(normal_d, {"x_mean": x_mean, "x_std": x_std, "y_mean": d_mean,
                            "y_std": d_std, "absolute_y_mean": absolute_mean}, "Delta normalization")
    _, target_identity = _prepare_targets(dataset, delta["directory"] / "targets", resume=True)
    _require(target_identity == delta["best"]["target_identity"], "Derived teacher hash/association mismatch")
    training = {"absolute": _optimizer_info(absolute, len(train.x)), "delta": _optimizer_info(delta, len(train.x))}
    _require(training["absolute"]["parameter_groups"] == training["delta"]["parameter_groups"],
             "Actual optimizer configurations differ")
    device, runtime = _runtime(absolute_config)
    val_x = (val.x - x_mean) / x_std
    _require(np.isfinite(val_x).all(), "Nonfinite normalized validation inputs")
    predictions, inference_seconds, consistency = {}, {}, {}
    for name, run, normalization in (("absolute", absolute, normal_a), ("delta", delta, normal_d)):
        model = ResponseMLP(run["best"]["model_config"]["hidden_dim"], run["best"]["model_config"]["layers"]).to(device)
        model.load_state_dict(run["best"]["model_state_dict"], strict=True)
        _require(all(torch.isfinite(parameter).all().item() for parameter in model.parameters()), "Nonfinite weights")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        args = (model, val_x, normalization["y_mean"], normalization["y_std"], BATCH_SIZE, device)
        predictions[name] = predict_delta(*args, val.persistence) if name == "delta" else predict_absolute(*args)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds[name] = time.perf_counter() - started
        metrics = _metrics(predictions[name], val.y, y_std)
        consistency[name] = {"report": _metric_check(metrics, run["report"]["val_model"]),
                             "checkpoint": _metric_check(metrics, run["best"]["best_val_metrics"])}
        del model
    metrics = {"persistence": _metrics(val.persistence, val.y, y_std)}
    metrics.update({name: _metrics(value, val.y, y_std) for name, value in predictions.items()})
    for name, run in (("absolute", absolute), ("delta", delta)):
        consistency[name]["persistence_report"] = _metric_check(metrics["persistence"], run["report"]["val_persistence"])
    hashes = dict(absolute["hashes"], **delta["hashes"])
    for path, expected in hashes.items():
        _require(file_sha256(path) == expected, "Input artifact/source changed during comparison: " + path)
    report = {
        "schema_version": 1, "status": "completed", "created_at": _now(),
        "purpose": "Paired absolute-state prediction comparison on original validation windows only.",
        "dataset": dataset.metadata, "validation_windows": len(val.x),
        "checks": {"same_dataset_and_splits": True, "same_model_capacity": True,
                   "same_training_config_except_targets": True, "same_optimizer_configuration": True,
                   "same_x_mean_x_std_y_std_exact": True, "same_absolute_y_mean_exact": True,
                   "normalization_verified_against_train_only": True, "teacher_window_association_verified": True,
                   "source_and_artifact_hashes_verified": True, "metric_recalculation": consistency},
        "runtime": runtime, "training": training, "validation_inference_seconds": inference_seconds,
        "timing_note": "Training report time includes loading and final evaluation; inference is one sequential pass per model, not a speed benchmark.",
        "metrics": metrics,
        "changes": {"delta_vs_absolute": _changes(metrics["delta"], metrics["absolute"]),
                    "delta_vs_persistence": _changes(metrics["delta"], metrics["persistence"]),
                    "absolute_vs_persistence": _changes(metrics["absolute"], metrics["persistence"])},
        "change_convention": "new minus reference; percent change is 100*(new-reference)/reference. Negative means lower error; zero reference yields null.",
        "absolute_run": str(absolute["directory"]), "delta_run": str(delta["directory"]),
        "verified_input_sha256": hashes, "target_identity": target_identity,
        "comparison_source_sha256": file_sha256(__file__),
        "guidance_evaluation_performed": False, "ood_evaluation_performed": False,
        "limitations": ["Both best epochs were selected on this validation split; it is not an untouched test set.",
                        "Windows from one episode are correlated; no independent-sample significance claim is made.",
                        "Quaternion error is on sign-aligned components, not a geodesic rotation error.",
                        "Prediction errors do not establish candidate ranking or OOD control improvements."],
    }
    arrays = {"schema_version": np.asarray(1), "target": val.y, "persistence": val.persistence,
              "absolute_prediction": predictions["absolute"], "delta_prediction": predictions["delta"],
              "episode_ids": np.asarray(val.episode_ids, dtype=str), "window_start_steps": val.window_start_steps}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=output.parent, prefix=predictions_path.name + ".", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, predictions_path)
    report["predictions"] = {"path": str(predictions_path), "sha256": file_sha256(predictions_path),
                             "rows": len(val.x), "split": "val", "arrays": sorted(arrays)}
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                     prefix=output.name + ".", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--absolute-dir", required=True, type=Path)
    parser.add_argument("--delta-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Fresh JSON path; paired predictions use .predictions.npz")
    args = parser.parse_args()
    report = compare(args.absolute_dir, args.delta_dir, args.output)
    print(json.dumps({"status": report["status"], "validation_windows": report["validation_windows"],
                      "metrics": report["metrics"], "output": str(args.output.resolve())}, allow_nan=False))


if __name__ == "__main__":
    main()
