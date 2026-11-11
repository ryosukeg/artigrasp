#!/usr/bin/env python3
"""Check saved collection invariants and measure paired residual responses."""
import argparse
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from ood_suite.io import file_sha256, read_json, read_rows, write_json
from ood_suite.metrics import quaternion_geodesic_rad
from dataset import load_dataset


def load_arrays(root, row):
    with np.load(root / row["trajectory_path"], allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def max_difference(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def audit(root, baseline=None):
    loaded = load_dataset(root)
    manifest, plan = read_json(root / "dataset.json"), read_json(root / "plan.json")
    rows = manifest["episodes"]
    counts = {}
    invalid_reasons = {key: 0 for key in ("bad_observation", "bad_action", "bad_state", "auto_reset")}
    for row in rows:
        arrays = load_arrays(root, row)
        counts[row["dataset_split"]] = counts.get(row["dataset_split"], 0) + int(arrays["window_valid"].sum())
        for key in invalid_reasons:
            invalid_reasons[key] += int(arrays[key].sum())
        np.testing.assert_array_equal(arrays["residuals"][:1260], 0)
        good = ~arrays["bad_action"]
        np.testing.assert_array_equal(arrays["actions_r"][:1260][good[:1260]],
                                      arrays["ppo_actions_r"][:1260][good[:1260]])
        np.testing.assert_array_equal(arrays["actions_l"][good], arrays["ppo_actions_l"][good])
        np.testing.assert_array_equal(arrays["actions_r"][good, 3:], arrays["ppo_actions_r"][good, 3:])
        expected = arrays["ppo_actions_r"][:, :3] + arrays["residuals"]
        np.testing.assert_array_equal(arrays["actions_r"][good, :3], expected[good])
        np.testing.assert_array_equal(arrays["residuals"][1260:], np.repeat(arrays["window_z"], 20, axis=0))
        np.testing.assert_array_equal(arrays["modes_r"][1260:], 8)
        np.testing.assert_array_equal(arrays["modes_l"][1260:], 7)
        for name in ("r", "l"):
            if arrays["final_obs_" + name].shape != (301, 276):
                raise ValueError("Missing post-step endpoint observation")
        np.testing.assert_array_equal(arrays["final_states"][1:], arrays["states"][1261:])
        finite = np.isfinite(arrays["final_obs_r"]).all(axis=1) & np.isfinite(arrays["final_states"]).all(axis=1)
        np.testing.assert_allclose(arrays["final_obs_r"][finite, 273],
                                   arrays["final_states"][finite, 129], atol=2e-5)
    report = {"schema_version": 1, "checks_passed": True, "kind": plan["kind"],
              "episodes": len(rows), "valid_episodes": sum(row["valid"] for row in rows),
              "valid_windows": sum(counts.values()), "windows_by_split": counts,
              "invalid_windows": sum(row["invalid_windows"] for row in rows),
              "invalid_event_counts": invalid_reasons,
              "source_dataset_sha256": file_sha256(root / "dataset.json"),
              "batch_wall_seconds": manifest["batch_wall_seconds"],
              "final_paper_successes": sum(row["paper_success"] for row in rows),
            "source_sha256": {name: file_sha256(HERE / name) for name in ("audit_collection.py", "dataset.py")},
            "interpretation": "Collection and data-pipeline verification; not an OOD improvement result."}
    if plan["kind"] == "zero":
        if baseline is None:
            raise ValueError("Zero audit requires --baseline")
        original = {row["episode_id"]: row for row in read_rows(baseline / "episodes.jsonl")}
        checks = []
        for row in rows:
            source = original[row["episode_id"]]
            if file_sha256(baseline / source["trajectory_path"]) != source["trajectory_sha256"]:
                raise ValueError("Original baseline trajectory hash mismatch")
            a, b = load_arrays(root, row), load_arrays(baseline, source)
            differences = {key: max_difference(a[key], b[key]) for key in ("states", "actions_r", "actions_l")}
            for key in differences:
                np.testing.assert_array_equal(a[key], b[key])
            checks.append(dict(episode_id=row["episode_id"], max_absolute_difference=differences))
        report["baseline_exact_checks"] = checks
    elif plan["kind"] == "calibration":
        groups = {}
        for row in rows:
            groups.setdefault(row["calibration_pair_id"], []).append(row)
        responses = []
        for group, variants in groups.items():
            zero = [row for row in variants if np.all(np.asarray(row["residuals"]) == 0)]
            if len(zero) != 1:
                raise ValueError("Calibration group needs exactly one zero control")
            control = load_arrays(root, zero[0])
            for row in variants:
                arrays = load_arrays(root, row)
                # Same initial conditions through the phase reset; the only
                # intervention is the right translation residual after this.
                for key in ("states", "actions_r", "actions_l"):
                    stop = 1261 if key == "states" else 1260
                    np.testing.assert_array_equal(arrays[key][:stop], control[key][:stop])
                np.testing.assert_array_equal(arrays["final_states"][0], control["final_states"][0])
                z = arrays["window_z"][0]
                response = {"pair_id": group, "episode_id": row["episode_id"], "z": z.tolist(),
                            "valid": bool(row["valid"] and zero[0]["valid"]),
                            "first_window_valid": bool(arrays["window_valid"][0] and control["window_valid"][0]),
                            "entry": row["opening_entry"]}
                for step, prefix in ((20, "first_window"), (300, "final")):
                    if not response["first_window_valid" if step == 20 else "valid"]:
                        continue
                    a, b = arrays["final_states"][step], control["final_states"][step]
                    response.update({prefix + "_angle_delta_vs_zero_rad": float(a[129] - b[129]),
                                     prefix + "_base_displacement_vs_zero_m": float(np.linalg.norm(a[122:125] - b[122:125])),
                                     prefix + "_base_rotation_vs_zero_rad": quaternion_geodesic_rad(a[125:129], b[125:129]),
                                     prefix + "_wrist_displacement_vs_zero_m": float(np.linalg.norm(a[6:9] - b[6:9])),
                                     prefix + "_wrist_delta_world_m": (a[6:9] - b[6:9]).tolist()})
                if response["valid"]:
                    response["max_angle_difference_rad"] = max_difference(arrays["final_states"][:, 129], control["final_states"][:, 129])
                q = arrays["action_frame_quaternion_wxyz"].astype(float)
                if np.isfinite(q).all() and np.linalg.norm(q) > 0:
                    q /= np.linalg.norm(q)
                    v = .001 * z.astype(float)
                    response["nominal_pd_target_offset_world_m"] = (v + 2 * np.cross(q[1:], np.cross(q[1:], v) + q[0] * v)).tolist()
                responses.append(response)
        report["paired_responses"] = responses
        report["exact_prefix_checks"] = len(responses)
        report["amplitude_summary"] = []
        for amplitude in (.5, 1.):
            selected = [r for r in responses if np.max(np.abs(r["z"])) == amplitude]
            short = [r for r in selected if r["first_window_valid"]]
            full = [r for r in selected if r["valid"]]
            report["amplitude_summary"].append({
                "amplitude_action": amplitude, "episodes": len(selected),
                "valid_episodes": len(full), "valid_first_windows": len(short),
                "first_window_abs_angle_delta_median_rad": float(np.median([abs(r["first_window_angle_delta_vs_zero_rad"]) for r in short])) if short else None,
                "first_window_abs_angle_delta_max_rad": max(abs(r["first_window_angle_delta_vs_zero_rad"]) for r in short) if short else None,
                "first_window_wrist_displacement_median_m": float(np.median([r["first_window_wrist_displacement_vs_zero_m"] for r in short])) if short else None,
                "max_trajectory_angle_difference_rad": max(r["max_angle_difference_rad"] for r in full) if full else None})
    write_json(root / "audit.json", report)
    print("Audit passed: {} episodes / {} valid windows; {}".format(len(rows), report["valid_windows"], root / "audit.json"))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    audit(args.dataset.resolve(), args.baseline.resolve() if args.baseline else None)
