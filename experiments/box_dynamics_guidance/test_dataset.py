"""Synthetic contract checks; no simulator or existing trajectories are modified."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dataset import DatasetError, file_sha256, load_dataset


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def synthetic_episode(offset=0.0):
    rng = np.random.RandomState(9)
    states = np.zeros((1561, 138), np.float32)
    final_states = np.zeros((301, 138), np.float32)
    final_states[:, 122] = offset + 7.0 + np.arange(301) * .001
    final_states[:, 125] = 1
    final_states[:, 129] = np.arange(301) * .001
    states[1260:] = final_states
    # The collector records final_states after phase(1260) resets the hands.
    states[1260, 122] = 99
    right = rng.normal(0, .01, (301, 276)).astype(np.float32) + offset
    left = rng.normal(0, .01, (301, 276)).astype(np.float32) + offset
    right[:, 273] = final_states[:, 129]
    right[:, 274] = np.arange(301) * .01
    z = np.zeros((15, 3), np.float32)
    z[::2, 0] = .5
    residuals = np.zeros((1560, 3), np.float32)
    residuals[1260:] = np.repeat(z, 20, axis=0)
    action_r = np.zeros((1560, 51), np.float32)
    action_r[:, :3] = residuals
    modes_r = np.zeros(1560, np.int32)
    modes_l = np.zeros(1560, np.int32)
    modes_r[1260:], modes_l[1260:] = 8, 7
    data = {
        "schema_version": np.asarray(1), "horizon": np.asarray(20),
        "control_dt": np.asarray(.01), "states": states,
        "actions_r": action_r, "actions_l": np.zeros_like(action_r),
        "ppo_actions_r": np.zeros_like(action_r), "ppo_actions_l": np.zeros_like(action_r),
        "residuals": residuals, "final_obs_r": right, "final_obs_l": left,
        "final_states": final_states, "modes_r": modes_r, "modes_l": modes_l,
        "window_start_step": np.arange(1260, 1560, 20, dtype=np.int32),
        "window_end_step": np.arange(1280, 1561, 20, dtype=np.int32),
        "window_z": z, "window_valid": np.ones(15, bool),
        "goal": np.asarray([7., 0., 0., 1., 0., 0., 0., 1.], np.float32),
        "initial_invalid": np.asarray(False), "terminal_bad_observation": np.asarray(False),
    }
    for key in ("bad_observation", "bad_action", "bad_state", "auto_reset"):
        data[key] = np.zeros(1560, bool)
    return data


class DatasetContractTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="box_response_dataset_test_")
        self.root = Path(self.directory.name)
        (self.root / "trajectories").mkdir()
        self.plan = {"schema_version": 1, "episodes": [], "reference_split": {
            "official_split": "train", "test_split_used": False, "groups": []}}
        self.manifest = {"schema_version": 1, "complete": True, "episodes": []}
        self.arrays = {}
        for number, split in enumerate(("train", "val")):
            episode_id = "synthetic_" + split
            label = {"seq_num": number + 1, "seq_name": "box_use_01", "obj_name": "box",
                     "init_frame": 0, "grasp_frame": 1, "kind": "synthetic"}
            arrays = synthetic_episode(number * .5)
            self.arrays[episode_id] = arrays
            spec = {"episode_id": episode_id, "dataset_split": split, "split": "train",
                    "grasp_label": dict(label), "arti_label": dict(label),
                    "residuals": arrays["window_z"].tolist()}
            self.plan["episodes"].append(spec)
            self.plan["reference_split"]["groups"].append({
                "seq_num": number + 1, "seq_name": "box_use_01", "dataset_split": split,
                "references": [{"role": role, "label": dict(label)} for role in ("grasp", "articulation")]})
            path = Path("trajectories") / (episode_id + ".npz")
            np.savez_compressed(self.root / path, **arrays)
            self.manifest["episodes"].append(dict(
                spec, trajectory_path=str(path), trajectory_sha256=file_sha256(self.root / path),
                paper_success=False, ood_goal_success=False, valid=True))
        self.commit_plan()

    def tearDown(self):
        self.directory.cleanup()

    def commit_plan(self):
        write_json(self.root / "plan.json", self.plan)
        self.manifest["plan_sha256"] = file_sha256(self.root / "plan.json")
        write_json(self.root / "dataset.json", self.manifest)

    def commit_episode(self, episode_id="synthetic_train"):
        row = next(row for row in self.manifest["episodes"] if row["episode_id"] == episode_id)
        path = self.root / row["trajectory_path"]
        np.savez_compressed(path, **self.arrays[episode_id])
        row["trajectory_sha256"] = file_sha256(path)
        write_json(self.root / "dataset.json", self.manifest)

    def test_features_targets_failed_examples_and_terminal_observation(self):
        data = load_dataset(self.root)
        train = data.splits["train"]
        self.assertEqual(train.x.shape, (15, 571))
        self.assertEqual(train.y.shape, (15, 9))
        self.assertAlmostEqual(float(train.x[0, 552]), 7.)
        self.assertNotEqual(train.x[0, 552], 99.)
        self.assertAlmostEqual(float(train.y[0, 0]), .02)
        self.assertAlmostEqual(float(train.y[0, 1]), .2)
        self.assertAlmostEqual(float(train.y[-1, 1]), 3.)
        self.assertAlmostEqual(float(train.persistence[0, 1]), 0.)
        self.assertEqual(data.metadata["reference_validation"]["reference_memberships_checked"], 4)
        self.assertEqual(data.metadata["excluded_windows"], 0)

    def test_quaternion_sign_is_aligned_per_window(self):
        self.arrays["synthetic_train"]["final_states"][20, 125:129] *= -1
        self.commit_episode()
        train = load_dataset(self.root).splits["train"]
        self.assertEqual(train.y[0, 5], 1.)
        self.assertEqual(train.y[1, 5], -1.)
        self.assertEqual(train.x[1, 563], -1.)  # Goal sign follows this start quaternion.

    def test_incomplete_manifest_rejected(self):
        self.manifest["complete"] = False
        write_json(self.root / "dataset.json", self.manifest)
        with self.assertRaisesRegex(DatasetError, "completion"):
            load_dataset(self.root)

    def test_changed_plan_hash_rejected(self):
        with (self.root / "plan.json").open("a") as stream:
            stream.write(" ")
        with self.assertRaisesRegex(DatasetError, "plan.json SHA256"):
            load_dataset(self.root)

    def test_changed_trajectory_hash_rejected(self):
        with (self.root / "trajectories/synthetic_train.npz").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(DatasetError, "trajectory SHA256"):
            load_dataset(self.root)

    def test_reference_membership_leakage_rejected(self):
        train_label = self.plan["episodes"][0]["grasp_label"]
        self.plan["episodes"][1]["grasp_label"] = train_label
        self.manifest["episodes"][1]["grasp_label"] = train_label
        self.commit_plan()
        with self.assertRaisesRegex(DatasetError, "reference"):
            load_dataset(self.root)

    def test_prior_reset_latches_invalidity_even_if_window_marked_valid(self):
        self.arrays["synthetic_train"]["auto_reset"][100] = True
        self.commit_episode()
        data = load_dataset(self.root)
        self.assertEqual(len(data.splits["train"].x), 0)
        self.assertEqual(len(data.splits["val"].x), 15)
        self.assertEqual(data.metadata["excluded_reason_counts"]["prior_or_current_auto_reset"], 15)

    def test_terminal_bad_flag_excludes_last_window(self):
        self.arrays["synthetic_train"]["terminal_bad_observation"] = np.asarray(True)
        self.commit_episode()
        self.assertEqual(len(load_dataset(self.root).splits["train"].x), 14)

    def test_nonfinite_endpoint_excluded_even_without_flag(self):
        self.arrays["synthetic_train"]["final_obs_r"][-1, 274] = np.nan
        self.commit_episode()
        self.assertEqual(len(load_dataset(self.root).splits["train"].x), 14)

    def test_changed_residual_inside_valid_window_rejected(self):
        self.arrays["synthetic_train"]["residuals"][1261, 0] += .1
        self.commit_episode()
        with self.assertRaisesRegex(DatasetError, "residual changes"):
            load_dataset(self.root)

    def test_wrong_horizon_rejected(self):
        self.arrays["synthetic_train"]["horizon"] = np.asarray(19)
        self.commit_episode()
        with self.assertRaisesRegex(DatasetError, "horizon=20"):
            load_dataset(self.root)

    def test_supervised_smoke_and_checkpoint_reload(self):
        import torch
        from train_smoke import ResponseMLP, train_smoke
        output = self.root / "smoke"
        report = train_smoke(self.root, output, epochs=2)
        self.assertEqual(report["status"], "smoke_passed")
        self.assertTrue((output / "report.json").exists())
        dataset = load_dataset(self.root)
        np.testing.assert_allclose(report["normalization"]["x_mean"],
                                   dataset.splits["train"].x.astype(np.float64).mean(axis=0), atol=1e-6)
        saved = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
        model = ResponseMLP()
        model.load_state_dict(saved["model_state_dict"])
        normalized = ((torch.from_numpy(dataset.splits["val"].x) - saved["normalization"]["x_mean"])
                      / saved["normalization"]["x_std"])
        with torch.no_grad():
            self.assertTrue(torch.isfinite(model(normalized)).all())
        self.assertEqual(saved["feature_spec"]["input_dim"], 571)
        self.assertTrue(np.isfinite(report["val_persistence"]["normalized_mse"]))


if __name__ == "__main__":
    unittest.main()
