"""Tests for collection provenance, paired calibration and source isolation."""

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import planning
from ood_suite.targets import sample_target


class CollectionPlanTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        label_path = self.root / "rsc/mano_double/grasp_label/final_labels/train_set/box.npy"
        label_path.parent.mkdir(parents=True)
        labels = []
        for subject, roles in ((1, ("two_grasp", "right_arti")),
                               (2, ("two_grasp", "right_arti")),
                               (3, ("two_grasp",)), (4, ("right_arti",)),
                               (5, ("two_grasp", "right_arti"))):
            for role in roles:
                for frame in (20, 40):
                    labels.append({"seq_num": subject, "seq_name": "box_use_01",
                                   "obj_name": "box", "init_frame": frame,
                                   "grasp_frame": frame + 10, "kind": role})
        np.save(str(label_path), np.array(labels, dtype=object))
        urdf = self.root / "rsc/arctic/box/box.urdf"
        urdf.parent.mkdir(parents=True)
        urdf.write_text('<robot><joint name="hinge" type="revolute">'
                        '<limit lower="0" upper="3.141592653589793"/></joint></robot>')
        self.baseline_dir = self.root / "baseline"
        self.baseline_dir.mkdir()
        normal = {"source": "paper_supplement_b4", "articulation_rad": [0.5, 0.6],
                  "yaw_rad": [-0.4, 0.4]}
        self.baseline = {
            "schema_version": 1, "seed": 101, "split": "train",
            "config": {"seed": 101, "split": "train", "objects": ["box"],
                       "normal_goal": normal, "profiles": {"normal": {}},
                       "runtime": {"batch_size": 8, "num_threads": 4, "device": "cuda"},
                       "evaluation_protocol": "factorial_v1"},
            "episodes": [],
        }
        for index in range(8):
            self.baseline["episodes"].append({
                "episode_id": "box__normal__{:04d}".format(index),
                "object": "box", "profile": "normal", "sample_index": index,
                "seed": 1000 + index, "split": "train",
                "grasp_label": copy.deepcopy(labels[0]),
                "arti_label": copy.deepcopy(labels[2]),
                "target": {"lift_m": 0.15, "translation_m": [-0.08, -0.07],
                           "orientation_delta_rad": 0.0, "articulation_rad": 0.51 + index * 0.01,
                           "articulation_rule": "absolute", "normal_goal_source": "paper_supplement_b4"},
            })
        self.path = self.baseline_dir / "plan.json"
        self.path.write_text(json.dumps(self.baseline), encoding="utf-8")

    def plan(self, kind, **kwargs):
        return planning.build_collection_plan(self.root, kind, self.baseline_dir, **kwargs)

    def test_zero_preserves_baseline_and_runtime_seed(self):
        plan = self.plan("zero", seed=999)
        self.assertEqual(8, len(plan["episodes"]))
        self.assertEqual(101, plan["config"]["seed"])
        self.assertEqual(999, plan["collection"]["requested_seed"])
        for original, actual in zip(self.baseline["episodes"], plan["episodes"]):
            self.assertEqual(original, {key: actual[key] for key in original})
            self.assertEqual("audit", actual["dataset_split"])
            self.assertEqual([[0.0] * 3 for _ in range(15)], actual["residuals"])
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), plan["baseline"]["sha256"])
        self.assertEqual(str(self.path.resolve()), plan["baseline"]["path"])
        self.assertEqual(self.baseline, json.loads(self.path.read_text()))

    def test_calibration_pairs_and_constant_windows(self):
        plan = self.plan("calibration", seed=999, amplitude=2.0)
        self.assertEqual(52, len(plan["episodes"]))
        self.assertEqual(52, len({spec["episode_id"] for spec in plan["episodes"]}))
        self.assertEqual(101, plan["config"]["seed"])
        originals = {spec["episode_id"]: spec for spec in self.baseline["episodes"]}
        for pair_id in planning.CALIBRATION_BASELINE_IDS:
            paired = [spec for spec in plan["episodes"] if spec["calibration_pair_id"] == pair_id]
            self.assertEqual(13, len(paired))
            self.assertEqual(13, len({spec["variant"] for spec in paired}))
            self.assertEqual(13, len({tuple(spec["residuals"][0]) for spec in paired}))
            for spec in paired:
                self.assertEqual("calibration", spec["dataset_split"])
                for field in ("target", "seed", "grasp_label", "arti_label"):
                    self.assertEqual(originals[pair_id][field], spec[field])
                self.assertEqual([spec["residuals"][0]] * 15, spec["residuals"])
                nonzero = [abs(value) for value in spec["residuals"][0] if value]
                self.assertLessEqual(len(nonzero), 1)
                self.assertTrue(all(value in (0.5, 1.0) for value in nonzero))
        plan["episodes"][0]["residuals"][0][0] = 123
        self.assertEqual([0.0, 0.0, 0.0], plan["episodes"][0]["residuals"][1])

    def test_random_source_split_precedes_pair_sampling_and_never_reads_test(self):
        with mock.patch.object(planning, "load_label_sets", wraps=planning.load_label_sets) as loader:
            plan = self.plan("random")
        loader.assert_called_once_with(self.root, ["box"], split="train")
        self.assertEqual({"train": 32, "val": 8}, {
            split: sum(spec["dataset_split"] == split for spec in plan["episodes"])
            for split in ("train", "val")})
        manifest = plan["reference_split"]
        assignments = {(group["seq_num"], group["seq_name"]): group["dataset_split"]
                       for group in manifest["groups"]}
        self.assertEqual(5, len(assignments))
        for group in manifest["groups"]:
            for reference in group["references"]:
                label = reference["label"]
                self.assertEqual(group["dataset_split"], assignments[(label["seq_num"], label["seq_name"])])
        for spec in plan["episodes"]:
            self.assertEqual("train", spec["split"])
            for role in ("grasp_label", "arti_label"):
                label = spec[role]
                self.assertEqual(spec["dataset_split"], assignments[(label["seq_num"], label["seq_name"])])
        self.assertFalse(manifest["test_split_used"])
        self.assertEqual(manifest["groups"], self.plan("random", episodes=100, amplitude=0.5)["reference_split"]["groups"])

    def test_random_distribution_sources_spatial_draws_and_residual_support(self):
        plan = self.plan("random", amplitude=0.75)
        self.assertNotIn("normal_goal", plan["config"])
        self.assertEqual(planning.COLLECTION_GOAL_SOURCE, plan["config"]["collection_goal"]["source"])
        self.assertEqual([1.0 / 7.0] * 7, plan["collection"]["candidate_probabilities"])
        candidates = {tuple(vector) for vector in plan["collection"]["candidate_residuals"]}
        self.assertEqual(7, len(candidates))
        for spec in plan["episodes"]:
            self.assertTrue(0.5 <= spec["target"]["articulation_rad"] <= 1.5)
            self.assertNotIn("normal_goal_source", spec["target"])
            expected = sample_target("box", spec["seed"], {}, [0.0, np.pi],
                                     spec["sample_index"], self.baseline["config"]["normal_goal"])
            for field in ("lift_m", "translation_m", "orientation_delta_rad", "orientation_rotation"):
                self.assertEqual(expected[field], spec["target"][field])
            self.assertEqual(15, len(spec["residuals"]))
            self.assertTrue(all(tuple(vector) in candidates for vector in spec["residuals"]))
        self.assertEqual(plan, self.plan("random", amplitude=0.75))
        self.assertNotEqual(plan["episodes"], self.plan("random", seed=302, amplitude=0.75)["episodes"])
        json.dumps(plan, allow_nan=False)

    def test_rejects_test_baseline_and_invalid_inputs(self):
        for kwargs in ({"episodes": 1}, {"amplitude": 0}, {"amplitude": float("nan")},
                       {"batch_size": 0}, {"num_threads": True}, {"seed": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.plan("random", **kwargs)
        self.baseline["split"] = "test"
        self.path.write_text(json.dumps(self.baseline))
        with self.assertRaisesRegex(ValueError, "train split"):
            self.plan("zero")

    def test_insufficient_articulation_source_groups_fail_instead_of_leaking(self):
        labels = planning.load_label_sets(self.root, ["box"], split="train")["box"]
        labels["articulation"] = [item for item in labels["articulation"] if item["seq_num"] == 1]
        with mock.patch.object(planning, "load_label_sets", return_value={"box": labels}):
            with self.assertRaisesRegex(ValueError, "articulation in at least two"):
                self.plan("random")


if __name__ == "__main__":
    unittest.main()
