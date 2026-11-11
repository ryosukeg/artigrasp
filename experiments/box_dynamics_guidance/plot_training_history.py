#!/usr/bin/env python3
"""Render recorded training losses without training a model or modifying its files."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/box_training_plot_mplconfig")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(checkpoint_dir, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report_path = checkpoint_dir / "report.json"
    history_path = checkpoint_dir / "history.jsonl"
    report = json.loads(report_path.read_text())
    rows = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
    if sha256(history_path) != report["artifact_sha256"]["history.jsonl"]:
        raise ValueError("Recorded history hash differs from the completed report")
    if not rows or [row["epoch"] for row in rows] != list(range(1, report["completed_epochs"] + 1)):
        raise ValueError("History must contain every completed epoch exactly once")
    for row in rows:
        for key in ("train_normalized_mse", "val_normalized_mse", "epoch_seconds"):
            if not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError("Invalid history value: " + key)
    epochs = [row["epoch"] for row in rows]
    train = [row["train_normalized_mse"] for row in rows]
    val = [row["val_normalized_mse"] for row in rows]
    best = report["best_epoch"]
    best_val = report["best_val_normalized_mse"]
    if not math.isclose(val[best - 1], min(val), rel_tol=1e-9) or not math.isclose(best_val, min(val), rel_tol=1e-9):
        raise ValueError("Best epoch does not agree with the recorded validation history")
    baseline = report["val_persistence"]["normalized_mse"]
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#243247", "text.color": "#243247",
                         "axes.edgecolor": "#b0b8c4", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.8))
    fig.subplots_adjust(left=.075, right=.97, top=.79, bottom=.27, wspace=.25)
    fig.suptitle("Box response predictor: learning curves", x=.075, y=.97,
                 ha="left", fontsize=20, fontweight="bold")
    counts = report["dataset"]["window_counts"]
    representation = report["config"]["target_representation"]
    target_label = {"absolute": "Absolute-state targets",
                    "delta_angle_position": "Angle/position delta targets"}.get(representation, representation)
    fig.text(.075, .9, "{} | {:,} train / {:,} validation transitions | 0.2-second horizon".format(
        target_label, counts["train"], counts["val"]), fontsize=11)
    stopped = "Early stopped" if report["stop_reason"] == "early_stopping" else "Stopped"
    fig.text(.075, .86, "Best validation: epoch {} ({:.4f})  |  {} at epoch {} / {}".format(
        best, best_val, stopped, epochs[-1], report["requested_max_epochs"]), fontsize=11)

    train_color, val_color, reference_color = "#2563eb", "#c65d14", "#607080"
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Normalized MSE (lower is better)")
        ax.set_xlim(0, max(epochs) + 2)
        ax.set_xticks(sorted(set([1] + list(range(10, max(epochs) + 1, 10)) + [epochs[-1]])))
        ax.grid(axis="y", color="#dce1e7", linewidth=.7)
        ax.set_axisbelow(True)
        ax.axhline(baseline, color=reference_color, linestyle="--", linewidth=1.5)
        ax.axvline(best, color=val_color, linestyle=":", linewidth=1, alpha=.7)
        ax.scatter([best], [best_val], s=45, color=val_color, edgecolors="white", zorder=5)
    train_line, = axes[0].plot(epochs, train, color=train_color, linewidth=1.8,
                              label="Training loss (during updates)")
    val_line, = axes[0].plot(epochs, val, color=val_color, linewidth=1.8,
                            label="Validation loss (after epoch)")
    reference_line = axes[0].lines[0]
    reference_line.set_label("Validation: copy current state ({:.4f})".format(baseline))
    axes[0].set_ylim(0, max(max(train), max(val), baseline) * 1.12)
    axes[0].set_title("Training and validation", loc="left", fontweight="bold", pad=12)
    axes[1].plot(epochs, val, color=val_color, linewidth=1.8)
    axes[1].set_ylim(min(baseline, min(val)) * .82, max(max(val), baseline) * 1.12)
    axes[1].set_title("Validation detail", loc="left", fontweight="bold", pad=12)
    axes[1].annotate("Best epoch {}\n{:.4f}".format(best, best_val),
                     xy=(best, best_val), xytext=(best - 15, best_val + .065),
                     arrowprops={"arrowstyle": "->", "color": val_color, "lw": 1},
                     fontsize=10, color=val_color)
    fig.legend(handles=[train_line, val_line, reference_line], loc="lower left",
               bbox_to_anchor=(.065, .12), ncol=2, frameon=False, fontsize=10)
    fig.text(.075, .09, "Training: sample-weighted mini-batch loss during weight updates. Validation: all held-out samples after each epoch.", fontsize=9)
    fig.text(.075, .055, "Dashed reference: predict the current state unchanged. Linear axes; no smoothing. Source: history.jsonl + report.json.", fontsize=9)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_dir / ("loss_curve." + suffix), dpi=180, facecolor="white")
    plt.close(fig)

    with (output_dir / "loss_history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    train_windows = counts["train"]
    batches = math.ceil(train_windows / report["config"]["batch_size"])
    model = report["model"]
    width, layers = model["hidden_dim"], model["layers"]
    parameters = (model["input_dim"] + 1) * width + (layers - 1) * (width + 1) * width + (width + 1) * model["output_dim"]
    metadata = {
        "source_report": str(report_path.resolve()), "source_history": str(history_path.resolve()),
        "source_sha256": {"report.json": sha256(report_path), "history.jsonl": sha256(history_path),
                          "plot_training_history.py": sha256(Path(__file__))},
        "matplotlib_version": matplotlib.__version__,
        "completed_epochs": epochs[-1], "best_epoch": best, "best_val_normalized_mse": best_val,
        "val_persistence_normalized_mse": baseline, "stop_reason": report["stop_reason"],
        "model_parameters": parameters, "training_windows_per_epoch": train_windows,
        "validation_windows": counts["val"], "optimizer_updates_per_epoch": batches,
        "optimizer_updates_total": batches * epochs[-1], "training_sample_presentations": train_windows * epochs[-1],
        "training_process_elapsed_seconds": report["elapsed_seconds"],
        "epoch_training_and_validation_seconds_sum": sum(row["epoch_seconds"] for row in rows),
        "timing_note": "The remaining time includes loading/verification, initialization, checkpoint writes and final evaluation; components were not individually timed.",
        "train_curve_definition": "Sample-weighted average of minibatch losses computed during updates, not a fixed-weight full-training-set evaluation.",
        "val_curve_definition": "Full validation split evaluated after each epoch with fixed weights.",
        "no_smoothing": True, "axis_scale": "linear", "new_training_performed": False,
        "outputs": ["loss_curve.png", "loss_curve.pdf", "loss_curve.svg", "loss_history.csv"],
    }
    (output_dir / "plot_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    render(args.checkpoint_dir.resolve(), (args.output or args.checkpoint_dir / "plots").resolve())
