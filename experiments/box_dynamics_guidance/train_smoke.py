#!/usr/bin/env python3
"""Small supervised smoke run; this does not establish guidance performance."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from .dataset import FEATURE_SPEC, INPUT_DIM, OUTPUT_DIM, DatasetError, file_sha256, load_dataset
except ImportError:
    from dataset import FEATURE_SPEC, INPUT_DIM, OUTPUT_DIM, DatasetError, file_sha256, load_dataset


class ResponseMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(INPUT_DIM, 128), nn.ReLU(),
                                     nn.Linear(128, 128), nn.ReLU(),
                                     nn.Linear(128, OUTPUT_DIM))

    def forward(self, x):
        return self.network(x)


def _normalization(train_array):
    mean = train_array.astype(np.float64).mean(axis=0)
    std = train_array.astype(np.float64).std(axis=0)
    # Constant training dimensions cannot supply an empirical scale.
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _metrics(prediction, truth, target_std):
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    rmse = np.sqrt(np.mean(error ** 2, axis=0))
    values = {
        "normalized_mse": float(np.mean((error / target_std) ** 2)),
        "per_output_rmse": rmse.tolist(),
        "angle_rad_rmse": float(rmse[0]),
        "angular_velocity_rad_s_rmse": float(rmse[1]),
        "base_position_m_rmse": float(np.sqrt(np.mean(error[:, 2:5] ** 2))),
        "base_quaternion_component_rmse": float(np.sqrt(np.mean(error[:, 5:9] ** 2))),
    }
    if not np.isfinite(error).all() or not np.isfinite(rmse).all():
        raise RuntimeError("Non-finite evaluation results")
    return values


def train_smoke(dataset_path, output_path, epochs=5, seed=0, batch_size=128, device="cpu"):
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    dataset = load_dataset(dataset_path)
    train, val = dataset.splits["train"], dataset.splits["val"]
    if len(train.x) == 0 or len(val.x) == 0:
        raise DatasetError("Smoke training needs nonempty, valid train and val splits")
    output = Path(output_path).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "checkpoint.pt").exists() or (output / "report.json").exists():
        raise FileExistsError("Use a fresh output directory; existing smoke artifacts are preserved")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.use_deterministic_algorithms(True)
    model = ResponseMLP().to(device)
    x_mean, x_std = _normalization(train.x)
    y_mean, y_std = _normalization(train.y)
    train_x, train_y = (train.x - x_mean) / x_std, (train.y - y_mean) / y_std
    val_x = (val.x - x_mean) / x_std
    if not all(np.isfinite(array).all() for array in (train_x, train_y, val_x)):
        raise RuntimeError("Non-finite normalized data")
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
                        batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
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
        model.eval()
        with torch.no_grad():
            val_prediction = model(torch.from_numpy(val_x).to(device)).cpu().numpy() * y_std + y_mean
        val_metrics = _metrics(val_prediction, val.y, y_std)
        row = {"epoch": epoch + 1, "train_normalized_mse": loss_sum / len(train.x),
               "val_normalized_mse": val_metrics["normalized_mse"]}
        history.append(row)
        print(json.dumps(row, allow_nan=False), flush=True)
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise RuntimeError("Non-finite model weights")
    model.eval()
    with torch.no_grad():
        train_prediction = model(torch.from_numpy(train_x).to(device)).cpu().numpy() * y_std + y_mean
    report = {
        "schema_version": 1, "status": "smoke_passed", "epochs": epochs,
        "seed": seed, "device": device, "batch_size": batch_size,
        "purpose": "Verify dataset reload, supervised optimization and held-out inference only.",
        "limitations": [
            "This smoke run is not evidence of prediction quality or OOD guidance improvement.",
            "Quaternion loss uses direct sign-aligned components, not a rotation-invariant metric.",
            "No simulator or policy is changed or evaluated by this script.",
        ],
        "normalization": {"fit_split": "train", "constant_dimension_scale": 1.0,
                          "x_mean": x_mean.tolist(), "x_std": x_std.tolist(),
                          "y_mean": y_mean.tolist(), "y_std": y_std.tolist()},
        "model": {"type": "ResponseMLP", "hidden_dimensions": [128, 128],
                  "input_dim": INPUT_DIM, "output_dim": OUTPUT_DIM},
        "dataset": dataset.metadata, "history": history,
        "train_model": _metrics(train_prediction, train.y, y_std),
        "val_model": val_metrics,
        "val_persistence": _metrics(val.persistence, val.y, y_std),
        "source_sha256": {"train_smoke.py": file_sha256(__file__),
                          "dataset.py": file_sha256(Path(__file__).with_name("dataset.py"))},
    }
    checkpoint = {
        "schema_version": 1, "model_state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        "model_config": report["model"], "feature_spec": FEATURE_SPEC,
        "normalization": {key: torch.from_numpy(value) for key, value in (
            ("x_mean", x_mean), ("x_std", x_std), ("y_mean", y_mean), ("y_std", y_std))},
        "dataset_manifest_sha256": dataset.metadata["dataset_manifest_sha256"],
        "plan_sha256": dataset.metadata["plan_sha256"], "seed": seed, "epochs": epochs,
    }
    temporary_checkpoint = output / "checkpoint.pt.tmp"
    torch.save(checkpoint, temporary_checkpoint)
    os.replace(temporary_checkpoint, output / "checkpoint.pt")
    report["checkpoint_sha256"] = file_sha256(output / "checkpoint.pt")
    temporary_report = output / "report.json.tmp"
    temporary_report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary_report, output / "report.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = train_smoke(args.dataset, args.output, args.epochs, args.seed, args.batch_size, args.device)
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve()),
                      "train_windows": report["dataset"]["window_counts"]["train"],
                      "val_windows": report["dataset"]["window_counts"]["val"]}, allow_nan=False))


if __name__ == "__main__":
    main()
