import copy
import json
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ood_suite import targets


ROOT = Path(__file__).resolve().parents[2]


def labels(object_name):
    base = {"seq_num": 1, "seq_name": object_name + "_use_01", "obj_name": object_name,
            "init_frame": 10, "grasp_frame": 20}
    return {
        "grasp": [dict(base, kind="two_grasp"), dict(base, kind="left_grasp", init_frame=25)],
        "articulation": [dict(base, kind="right_arti"), dict(base, kind="right_arti", grasp_frame=30)],
    }


class TargetTests(unittest.TestCase):
    def setUp(self):
        self.config = targets.load_config(ROOT / "configs" / "ood_eval.json")

    def plan(self, **kwargs):
        kwargs.setdefault("samples", 4)
        kwargs.setdefault("objects", ["box", "ketchup", "espressomachine"])
        kwargs["label_sets"] = {name: labels(name) for name in kwargs["objects"]}
        return targets.build_plan(ROOT, self.config, **kwargs)

    def test_identical_repeat_and_json_roundtrip(self):
        first = self.plan()
        self.assertEqual(first, self.plan())
        self.assertEqual(first, json.loads(json.dumps(first, allow_nan=False)))

    def test_labels_and_seed_paired_across_profiles(self):
        rows = [row for row in self.plan()["episodes"] if row["object"] == "box" and row["sample_index"] == 0]
        self.assertEqual(len(rows), 5)
        for row in rows[1:]:
            for key in ("grasp_label", "arti_label", "seed"):
                self.assertEqual(rows[0][key], row[key])
        by_profile = {row["profile"]: row for row in rows}
        base = by_profile["id_control"]["target"]
        height = by_profile["ood_height"]["target"]
        self.assertEqual(base["translation_m"], height["translation_m"])
        self.assertEqual(base["articulation_rad"], height["articulation_rad"])
        self.assertEqual(base["orientation_delta_rad"], height["orientation_delta_rad"])
        combined = by_profile["ood_combined"]["target"]
        self.assertEqual(combined["lift_m"], height["lift_m"])
        self.assertEqual(combined["translation_m"], by_profile["ood_translation"]["target"]["translation_m"])

    def test_selection_order_does_not_change_episodes(self):
        first = {row["episode_id"]: row for row in self.plan()["episodes"]}
        selected = self.plan(objects=["espressomachine", "box"], profiles=["ood_combined", "id_control"])
        for row in selected["episodes"]:
            self.assertEqual(first[row["episode_id"]], row)

    def test_extremes_and_infeasible_targets_not_clamped(self):
        for row in self.plan(samples=100)["episodes"]:
            profile, target = row["profile"], row["target"]
            if profile in ("ood_height", "ood_combined"):
                self.assertTrue(0.6 <= target["lift_m"] <= 0.9)
            if profile in ("ood_translation", "ood_combined"):
                self.assertTrue(0.5 <= math.hypot(*target["translation_m"]) <= 0.8)
            if profile in ("ood_articulation_180", "ood_combined"):
                self.assertEqual(target["articulation_rad"], math.pi)
                self.assertEqual(row["target_reachable"], row["object"] == "box")

    def test_public_target_rules(self):
        target = targets.sample_target("box", 1234, {})
        self.assertEqual(targets.resolve_articulation(target, 0), target["public_delta_rad"])
        self.assertEqual(targets.resolve_articulation(target, 2.0), 2.0 - target["public_delta_rad"])
        absolute = targets.sample_target("box", 1234, {"articulation_deg": 180})
        self.assertEqual(targets.resolve_articulation(absolute, 2.0), math.pi)
        ketchup = targets.sample_target("ketchup", 1234, {})
        self.assertEqual(ketchup["lift_m"], 0.3)
        self.assertEqual(targets.resolve_articulation(ketchup, 2.0), ketchup["articulation_rad"])

    def test_float32_pi_roundoff_allowed(self):
        self.assertTrue(targets.articulation_is_feasible(3.1415927410125732, [0, math.pi]))
        self.assertFalse(targets.articulation_is_feasible(math.pi + 1e-5, [0, math.pi]))

    def test_real_urdf_limits_and_paper_set(self):
        self.assertEqual(targets.joint_limits(ROOT, "espressomachine"), [0.0, 1.5])
        self.assertEqual(targets.joint_limits(ROOT, "phone"), [0.0, 2.5])
        self.assertEqual(len(targets.PAPER_OBJECTS), 8)
        self.assertNotIn("phone", targets.PAPER_OBJECTS)
        self.assertNotIn("capsulemachine", targets.PAPER_OBJECTS)

    def test_ketchup_uses_released_right_special_approach(self):
        plan = self.plan(objects=["ketchup"], samples=30)
        self.assertEqual(plan["label_counts"]["ketchup"]["possible_pairs"], 2)
        for episode in plan["episodes"]:
            self.assertEqual(episode["grasp_label_index"], episode["arti_label_index"])
            self.assertEqual(episode["grasp_label"]["kind"], "right_special")
            effective = dict(episode["grasp_label"], kind="right_arti")
            self.assertEqual(effective, episode["arti_label"])

    def test_invalid_inputs_fail_before_simulation(self):
        for kwargs in ({"samples": 0}, {"samples": -3}, {"samples": True},
                       {"objects": ["box", "box"]}, {"objects": ["scissors"]},
                       {"profiles": ["does_not_exist"]}, {"split": "validation"}):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, KeyError)):
                self.plan(**kwargs)
        with self.assertRaises(ValueError):
            targets.sample_target("box", 4, {"lift_m": [1, 0.1]})
        with self.assertRaises(ValueError):
            targets.sample_target("box", 4, {"translation_distance_m": 0})

    def test_config_is_not_mutated(self):
        previous = copy.deepcopy(self.config)
        self.plan()
        self.assertEqual(previous, self.config)

    def test_overrides_are_normalized_into_effective_config(self):
        plan = self.plan(objects=["ketchup", "box"], profiles=["ood_combined", "id_control"],
                         samples=2, seed=4321, split="train")
        config = plan["config"]
        self.assertEqual(config["objects"], ["ketchup", "box"])
        self.assertEqual(list(config["profiles"]), ["ood_combined", "id_control"])
        self.assertEqual(config["samples_per_object"], 2)
        self.assertEqual(config["seed"], 4321)
        self.assertEqual(config["split"], "train")
        self.assertEqual(config["checkpoints"], self.config["checkpoints"])
        self.assertEqual(config["runtime"], self.config["runtime"])
        self.assertEqual(len(plan["episodes"]), 8)


if __name__ == "__main__":
    unittest.main()
