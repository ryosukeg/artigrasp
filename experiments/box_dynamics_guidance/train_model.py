#!/usr/bin/env python3
"""Train the box response predictor; completion does not certify guidance quality.

Resume uses last.pt at an epoch boundary. best.pt contains the lowest-validation-
MSE weights for inference. Neither artifact changes or evaluates the PPO policy.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import sys
import time

# Must be set before the first CUDA context / GEMM is initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from .dataset import FEATURE_SPEC, INPUT_DIM, OUTPUT_DIM, DatasetError, file_sha256, load_dataset
    from .progress import Progress
    from .train_smoke import _metrics, _normalization
except ImportError:
    from dataset import FEATURE_SPEC, INPUT_DIM, OUTPUT_DIM, DatasetError, file_sha256, load_dataset
    from progress import Progress
    from train_smoke import _metrics, _normalization


SCHEMA_VERSION = 1


class ResponseMLP(nn.Module):
    def __init__(self, hidden_dim=256, layers=3):
        super().__init__()
        modules = []
        width = INPUT_DIM
        for _ in range(layers):
            modules.extend((nn.Linear(width, hidden_dim), nn.ReLU()))
            width = hidden_dim
        modules.append(nn.Linear(width, OUTPUT_DIM))
        self.network = nn.Sequential(*modules)

    def forward(self, x):
        return self.network(x)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _atomic_text(path, text):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _json(path, value):
    _atomic_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def _save(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    return value


def _finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def _rng_state(generator, device):
    numpy_state = np.random.get_state()
    result = {
        "python": random.getstate(), "torch_cpu": torch.get_rng_state(),
        "shuffle_generator": generator.get_state(),
        "numpy": {"algorithm": numpy_state[0], "keys": numpy_state[1].tolist(),
                  "position": numpy_state[2], "has_gauss": numpy_state[3],
                  "cached_gaussian": numpy_state[4]},
    }
    if device.type == "cuda":
        result["torch_cuda"] = torch.cuda.get_rng_state(device)
    return result


def _restore_rng(saved, generator, device):
    random.setstate(saved["python"])
    state = saved["numpy"]
    np.random.set_state((state["algorithm"], np.asarray(state["keys"], dtype=np.uint32),
                         state["position"], state["has_gauss"], state["cached_gaussian"]))
    torch.set_rng_state(saved["torch_cpu"])
    generator.set_state(saved["shuffle_generator"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(saved["torch_cuda"], device)


def _predict(model, values, mean, std, batch_size, device):
    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            prediction = model(torch.from_numpy(values[start:start + batch_size]).to(device))
            pieces.append(prediction.cpu().numpy())
    result = np.concatenate(pieces) * std + mean
    if not np.isfinite(result).all():
        raise RuntimeError("Non-finite model predictions")
    return result


def _publish_epoch(output, checkpoint):
    """Materialize recoverable artifacts from the authoritative last checkpoint."""
    _atomic_text(output / "history.jsonl", "".join(
        json.dumps(row, allow_nan=False) + "\n" for row in checkpoint["history"]))
    if checkpoint["best_epoch"] is not None:
        best = {key: checkpoint[key] for key in (
            "schema_version", "config", "sources", "dataset_identity", "normalization",
            "model_config", "feature_spec", "best_epoch", "best_val_normalized_mse", "best_val_metrics")}
        best.update(artifact_type="box_response_best_predictor",
                    model_state_dict=checkpoint["best_model_state_dict"],
                    resume_from="last.pt")
        _save(output / "best.pt", best)


def _source_hashes():
    directory = Path(__file__).resolve().parent
    return {name: file_sha256(directory / name) for name in (
        "train_model.py", "dataset.py", "train_smoke.py", "progress.py")}


def train_model(dataset_path, output_path, epochs=100, patience=20, min_delta=1e-4,
                batch_size=256, learning_rate=1e-3, seed=301, device="cuda",
                hidden_dim=256, layers=3, resume=False, progress=True):
    """Fit using train only; select the absolute minimum validation MSE epoch.

    min_delta controls patience resets, not best-weight selection. epochs is a
    total limit and may be increased when resuming; other settings must match.
    Interrupted partial epochs are rerun from the previous complete checkpoint.
    """
    output = Path(output_path).expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise FileExistsError("Output must be a directory")
    if not resume and output.exists() and any(output.iterdir()):
        raise FileExistsError("Output is not empty; choose a new directory or use --resume")
    if resume and not (output / "last.pt").is_file():
        raise FileNotFoundError("--resume requires an existing last.pt")
    output.mkdir(parents=True, exist_ok=True)
    completed_epoch = 0
    bar = None
    stage = "validation"
    stopped_reason = None
    checkpoint = None
    invocation_start = time.monotonic()
    previous_elapsed = 0.0
    try:
        for name, value, minimum in (("epochs", epochs, 1), ("patience", patience, 0),
                                     ("batch_size", batch_size, 1), ("hidden_dim", hidden_dim, 1),
                                     ("layers", layers, 1), ("seed", seed, 0)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError("{} must be an integer >= {}".format(name, minimum))
        if not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError("min_delta must be finite and nonnegative")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        target_device = torch.device(device)
        if target_device.type not in ("cpu", "cuda"):
            raise ValueError("Only cpu and cuda devices are supported")
        if target_device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; explicitly use --device cpu")
            target_device = torch.device("cuda", target_device.index if target_device.index is not None
                                         else torch.cuda.current_device())
            torch.cuda.set_device(target_device)
            if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in (":4096:8", ":16:8"):
                raise RuntimeError("Deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8")
        config = {
            "patience": patience, "min_delta": min_delta, "batch_size": batch_size,
            "learning_rate": learning_rate, "seed": seed, "device": str(target_device),
            "hidden_dim": hidden_dim, "layers": layers, "target_representation": "absolute",
            "deterministic_algorithms": True, "torch_cpu_threads": 1,
            "torch_version": str(torch.__version__), "numpy_version": str(np.__version__),
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"] if target_device.type == "cuda" else None,
        }
        if target_device.type == "cuda":
            config["cuda_device_name"] = torch.cuda.get_device_name(target_device)
            config["cuda_capability"] = list(torch.cuda.get_device_capability(target_device))
        sources = _source_hashes()
        _json(output / "status.json", {"status": "loading", "updated_at": _now(),
                                      "resume": resume, "requested_max_epochs": epochs})
        print("Loading and verifying completed dataset: {}".format(dataset_path), file=sys.stderr, flush=True)
        dataset = load_dataset(dataset_path)
        train, val = dataset.splits["train"], dataset.splits["val"]
        if not len(train.x) or not len(val.x):
            raise DatasetError("Training requires nonempty valid train and val splits")
        identity = {key: dataset.metadata[key] for key in (
            "dataset_path", "dataset_manifest_sha256", "plan_sha256")}
        if resume:
            checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
            if checkpoint.get("schema_version") != SCHEMA_VERSION or checkpoint.get("artifact_type") != "box_response_training_resume":
                raise ValueError("Unsupported resume checkpoint")
            completed_epoch = checkpoint["completed_epoch"]
            for name, expected in (("config", config), ("sources", sources), ("dataset_identity", identity)):
                if checkpoint.get(name) != expected:
                    raise ValueError("Resume {} mismatch; use the original settings/source/dataset or a new output directory".format(name))
            if epochs < completed_epoch:
                raise ValueError("--epochs is a total limit and cannot be below completed epoch {}".format(completed_epoch))
            if not all(_finite(checkpoint[key]) for key in (
                    "model_state_dict", "optimizer_state_dict", "best_model_state_dict", "best_val_metrics", "history")):
                raise ValueError("Resume checkpoint contains non-finite weights or optimizer state")
            previous_elapsed = checkpoint.get("elapsed_seconds", 0.0)

        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if target_device.type == "cuda":
            torch.cuda.manual_seed(seed)
        model = ResponseMLP(hidden_dim, layers).to(target_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        generator = torch.Generator().manual_seed(seed)
        if resume:
            normalization = checkpoint["normalization"]
        else:
            x_mean, x_std = _normalization(train.x)
            y_mean, y_std = _normalization(train.y)
            normalization = {name: torch.from_numpy(value) for name, value in (
                ("x_mean", x_mean), ("x_std", x_std), ("y_mean", y_mean), ("y_std", y_std))}
        if not _finite(normalization):
            raise RuntimeError("Non-finite normalization statistics")
        x_mean, x_std, y_mean, y_std = [normalization[name].numpy() for name in (
            "x_mean", "x_std", "y_mean", "y_std")]
        if x_mean.shape != (INPUT_DIM,) or y_mean.shape != (OUTPUT_DIM,) or x_std.shape != x_mean.shape or y_std.shape != y_mean.shape:
            raise ValueError("Invalid normalization dimensions")
        if np.any(x_std <= 0) or np.any(y_std <= 0):
            raise ValueError("Normalization scales must be positive")
        train_x, train_y = (train.x - x_mean) / x_std, (train.y - y_mean) / y_std
        val_x = (val.x - x_mean) / x_std
        if not all(np.isfinite(array).all() for array in (train_x, train_y, val_x)):
            raise RuntimeError("Non-finite normalized data")
        loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
                            batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
        if resume:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            _restore_rng(checkpoint["rng_state"], generator, target_device)
            history = checkpoint["history"]
            best_epoch, best_value = checkpoint["best_epoch"], checkpoint["best_val_normalized_mse"]
            best_state, best_metrics = checkpoint["best_model_state_dict"], checkpoint["best_val_metrics"]
            anchor, stale = checkpoint["early_stopping_anchor"], checkpoint["stale_epochs"]
            if len(history) != completed_epoch or any(row["epoch"] != i + 1 for i, row in enumerate(history)):
                raise ValueError("Resume history does not match completed epochs")
            if history and (best_epoch is None or not 1 <= best_epoch <= completed_epoch
                            or best_value != min(row["val_normalized_mse"] for row in history)
                            or history[best_epoch - 1]["val_normalized_mse"] != best_value):
                raise ValueError("Resume best epoch does not match validation history")
            if not isinstance(stale, int) or not 0 <= stale <= completed_epoch:
                raise ValueError("Invalid early-stopping state")
            _publish_epoch(output, checkpoint)
        else:
            history = []
            best_epoch = best_value = best_state = best_metrics = anchor = None
            stale = 0

        def snapshot(epoch):
            return {
                "schema_version": SCHEMA_VERSION, "artifact_type": "box_response_training_resume",
                "completed_epoch": epoch, "requested_max_epochs": epochs, "config": config,
                "sources": sources, "dataset_identity": identity, "normalization": normalization,
                "model_config": {"type": "ResponseMLP", "input_dim": INPUT_DIM, "output_dim": OUTPUT_DIM,
                                 "hidden_dim": hidden_dim, "layers": layers},
                "feature_spec": FEATURE_SPEC, "model_state_dict": _cpu(model.state_dict()),
                "optimizer_state_dict": _cpu(optimizer.state_dict()),
                "rng_state": _rng_state(generator, target_device), "history": list(history),
                "best_epoch": best_epoch, "best_val_normalized_mse": best_value,
                "best_model_state_dict": best_state, "best_val_metrics": best_metrics,
                "early_stopping_anchor": anchor, "stale_epochs": stale,
                "elapsed_seconds": previous_elapsed + time.monotonic() - invocation_start,
            }

        if not resume:
            checkpoint = snapshot(0)
            _save(output / "last.pt", checkpoint)
            _publish_epoch(output, checkpoint)
        stage = "training"
        if progress:
            bar = Progress(epochs, "Predictor training", "epoch")
            bar.update(completed_epoch)
        _json(output / "status.json", {"status": "running", "updated_at": _now(),
                                      "completed_epoch": completed_epoch, "requested_max_epochs": epochs,
                                      "best_epoch": best_epoch, "best_val_normalized_mse": best_value})
        # A committed final epoch may precede an interrupted report/best write.
        # Resume must repair/finalize it without forcing additional optimization.
        already_stopped = bool(patience and stale >= patience)
        if already_stopped:
            stopped_reason = "early_stopping"
        epoch_numbers = () if already_stopped else range(completed_epoch + 1, epochs + 1)
        for epoch in epoch_numbers:
            epoch_start = time.monotonic()
            model.train()
            loss_sum = 0.0
            for x, y in loader:
                x, y = x.to(target_device), y.to(target_device)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(model(x), y)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                loss.backward()
                if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                       for parameter in model.parameters()):
                    raise RuntimeError("Non-finite model gradients")
                optimizer.step()
                loss_sum += float(loss.detach().cpu()) * len(x)
            if not _finite(model.state_dict()) or not _finite(optimizer.state_dict()):
                raise RuntimeError("Non-finite model weights or optimizer state")
            prediction = _predict(model, val_x, y_mean, y_std, batch_size, target_device)
            metrics = _metrics(prediction, val.y, y_std)
            value = metrics["normalized_mse"]
            if not _finite(metrics) or not math.isfinite(loss_sum):
                raise RuntimeError("Non-finite epoch metrics")
            if best_value is None or value < best_value:
                best_value, best_epoch = value, epoch
                best_state, best_metrics = _cpu(model.state_dict()), metrics
            if anchor is None or value < anchor - min_delta:
                anchor, stale = value, 0
            else:
                stale += 1
            row = {"epoch": epoch, "train_normalized_mse": loss_sum / len(train.x),
                   "val_normalized_mse": value, "val_angle_rad_rmse": metrics["angle_rad_rmse"],
                   "val_base_position_m_rmse": metrics["base_position_m_rmse"],
                   "best_val_normalized_mse": best_value, "best_epoch": best_epoch,
                   "stale_epochs": stale, "epoch_seconds": time.monotonic() - epoch_start}
            history.append(row)
            checkpoint = snapshot(epoch)
            # last.pt commits the epoch first; history/best can be repaired from it.
            _save(output / "last.pt", checkpoint)
            completed_epoch = epoch
            _publish_epoch(output, checkpoint)
            _json(output / "status.json", {"status": "running", "updated_at": _now(),
                                          "completed_epoch": epoch, "requested_max_epochs": epochs,
                                          "best_epoch": best_epoch, "best_val_normalized_mse": best_value,
                                          "stale_epochs": stale, "latest": row})
            detail = "train={:.6g} val={:.6g} best={:.6g}@{} angle_RMSE={:.5g}rad".format(
                row["train_normalized_mse"], value, best_value, best_epoch, metrics["angle_rad_rmse"])
            if bar is not None:
                bar.update(epoch, detail)
            else:
                print("epoch {}/{} {}".format(epoch, epochs, detail), file=sys.stderr, flush=True)
            if patience and stale >= patience:
                stopped_reason = "early_stopping"
                break
        stopped_reason = stopped_reason or "max_epochs"
        stage = "final_evaluation"
        model.load_state_dict(best_state)
        train_prediction = _predict(model, train_x, y_mean, y_std, batch_size, target_device)
        val_prediction = _predict(model, val_x, y_mean, y_std, batch_size, target_device)
        val_result = _metrics(val_prediction, val.y, y_std)
        persistence = _metrics(val.persistence, val.y, y_std)
        report = {
            "schema_version": SCHEMA_VERSION, "status": "completed", "completed_at": _now(),
            "stop_reason": stopped_reason, "completed_epochs": completed_epoch,
            "requested_max_epochs": epochs, "best_epoch": best_epoch,
            "best_val_normalized_mse": best_value, "config": config,
            "dataset": dataset.metadata, "sources": sources,
            "normalization": {"fit_split": "train", "training_windows": len(train.x),
                              "constant_dimension_scale": 1.0, "values_saved_in": ["best.pt", "last.pt"]},
            "model": checkpoint["model_config"], "evaluated_checkpoint": "best.pt",
            "train_model": _metrics(train_prediction, train.y, y_std), "val_model": val_result,
            "train_persistence": _metrics(train.persistence, train.y, y_std), "val_persistence": persistence,
            "val_model_beats_persistence_normalized_mse": val_result["normalized_mse"] < persistence["normalized_mse"],
            "prediction_quality_passed": None, "guidance_evaluation_performed": False,
            "limitations": [
                "Completion means optimization and validation finished, not that prediction quality is sufficient.",
                "Validation MSE and persistence comparison do not establish action-candidate ranking or OOD policy improvement.",
                "Targets are absolute states; quaternion loss uses sign-aligned components, not a rotation-invariant metric.",
                "PPO and the simulator are not modified or evaluated by this script.",
            ],
            "history_file": "history.jsonl", "history_epochs": len(history),
            "elapsed_seconds": previous_elapsed + time.monotonic() - invocation_start,
            "artifact_sha256": {name: file_sha256(output / name) for name in ("best.pt", "last.pt", "history.jsonl")},
        }
        if not _finite(report):
            raise RuntimeError("Non-finite final report")
        _json(output / "report.json", report)
        _json(output / "status.json", {"status": "completed", "updated_at": _now(),
                                      "stop_reason": stopped_reason, "completed_epoch": completed_epoch,
                                      "requested_max_epochs": epochs, "best_epoch": best_epoch,
                                      "best_val_normalized_mse": best_value, "report": "report.json",
                                      "prediction_quality_passed": None})
        if bar is not None:
            bar.close("{}; best epoch {}".format(stopped_reason, best_epoch))
            bar = None
        return report
    except BaseException as exc:
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        _json(output / "status.json", {"status": status, "updated_at": _now(), "stage": stage,
                                      "completed_epoch": completed_epoch, "requested_max_epochs": epochs,
                                      "resume": resume, "error_type": type(exc).__name__, "error": str(exc),
                                      "resume_checkpoint_available": (output / "last.pt").is_file()})
        if bar is not None:
            bar.close("{}; last committed epoch {}".format(status, completed_epoch))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=100, help="Total epoch limit, including completed epochs on resume")
    parser.add_argument("--patience", type=int, default=20, help="0 disables early stopping")
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=301)
    parser.add_argument("--device", default="cuda", help="cuda, cuda:N, or cpu")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    try:
        report = train_model(args.dataset, args.output, args.epochs, args.patience, args.min_delta,
                             args.batch_size, args.learning_rate, args.seed, args.device,
                             args.hidden_dim, args.layers, args.resume, not args.no_progress)
    except KeyboardInterrupt:
        print("Interrupted. Resume from the last committed epoch with --resume.", file=sys.stderr)
        return 130
    except Exception as exc:
        print("Training failed: {}".format(exc), file=sys.stderr)
        return 1
    print(json.dumps({"status": report["status"], "stop_reason": report["stop_reason"],
                      "best_epoch": report["best_epoch"], "val_normalized_mse": report["best_val_normalized_mse"],
                      "report": str(args.output.resolve() / "report.json")}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
