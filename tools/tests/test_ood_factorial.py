"""Eight paired combinations, genuinely out-of-range draws and factorial reports."""

from collections import Counter
import itertools
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import ood_eval
from ood_suite.metrics import evaluate_goal_diagnostics
from ood_suite.report import _factorial_summaries, generate_report
from ood_suite.runtime import Controller, timing_for
from ood_suite.targets import build_plan, joint_limits, load_config, sample_target, select_profiles
from tools.tests.test_ood_multiaxis import diagnostic_rollout, label_set
from tools.tests.test_ood_runtime import _data, _fake_rotations, EventEnvironment


class FactorialTargetTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "configs/ood_multiaxis.json")

    def plan(self, objects=None, samples=2, **kwargs):
        objects = objects or self.config["objects"]
        return build_plan(ROOT, self.config, objects=objects, samples=samples,
                          label_sets={name: label_set(name) for name in objects}, **kwargs)

    def test_legacy_factorial_is_exactly_eight_combinations_and_8000_trials(self):
        self.assertEqual(self.config["evaluation_protocol"], "factorial_v1")
        self.assertEqual(len(self.config["profiles"]), 8)
        masks = {tuple(value["factor_flags"][key] for key in ("horizontal", "lift", "articulation"))
                 for value in self.config["profiles"].values()}
        self.assertEqual(masks, set(itertools.product((False, True), repeat=3)))
        for suite in (None, "all", "core"):
            self.assertEqual(select_profiles(self.config, suite=suite), list(self.config["profiles"]))
        plan = self.plan(samples=100)
        self.assertEqual(len(plan["episodes"]), 8000)
        counts = Counter((item["object"], item["profile"]) for item in plan["episodes"])
        self.assertEqual(len(counts), 80)
        self.assertEqual(set(counts.values()), {100})

    def test_random_ood_draws_are_feasible_outside_support_and_paired(self):
        plan = self.plan(samples=100)
        by_sample = {}
        values = {}
        for item in plan["episodes"]:
            by_sample.setdefault((item["object"], item["sample_index"]), {})[item["profile"]] = item
        for (name, index), items in by_sample.items():
            base, on = items["normal"], items["ood_all"]
            for item in items.values():
                for field in ("seed", "grasp_label", "arti_label"):
                    self.assertEqual(item[field], base[field])
                self.assertFalse(item["require_lift_pose_success"])
                target, normal, flags = item["target"], base["target"], item["factor_flags"]
                self.assertEqual(target["orientation_rotation"], normal["orientation_rotation"])
                for factor, field in (("horizontal", "translation_m"), ("lift", "lift_m"),
                                      ("articulation", "articulation_rad")):
                    self.assertEqual(target[field], (on if flags[factor] else base)["target"][field])
                self.assertTrue(item["articulation_within_joint_limits"])
                self.assertEqual(item["articulation_randomized_target_ood"], flags["articulation"])
                if flags["horizontal"]:
                    self.assertTrue(0.30 <= math.hypot(*target["translation_m"]) <= 0.80)
                    self.assertAlmostEqual(math.atan2(*reversed(target["translation_m"])),
                                           math.atan2(*reversed(normal["translation_m"])))
                if flags["lift"]:
                    self.assertTrue(0.60 <= target["lift_m"] <= 0.90)
                if flags["articulation"]:
                    angle = target["articulation_rad"]
                    self.assertTrue(0.20 <= angle <= 0.40 or 1.60 <= angle <= 3.10)
                    limits = item["joint_limits_rad"]
                    self.assertTrue(limits[0] + 0.05 <= angle <= limits[1] - 0.05)
                    if name == "espressomachine":
                        self.assertLessEqual(angle, 0.4)
            draws = values.setdefault(name, {key: [] for key in ("horizontal", "lift", "articulation")})
            draws["horizontal"].append(math.hypot(*on["target"]["translation_m"]))
            draws["lift"].append(on["target"]["lift_m"])
            draws["articulation"].append(on["target"]["articulation_rad"])
        for draws in values.values():
            for samples in draws.values():
                self.assertEqual(len(set(samples)), 100)
        # Sampling is uniform on the union, not a 50/50 choice of unequal bands.
        box_angles = values["box"]["articulation"]
        self.assertGreater(sum(value > 1.5 for value in box_angles), 70)
        self.assertLess(sum(value > 1.5 for value in box_angles), 100)

    def test_normal_uses_paper_ranges_even_for_ketchup(self):
        for item in self.plan(samples=20, profiles=["normal"])["episodes"]:
            target = item["target"]
            self.assertEqual(target["articulation_rule"], "absolute")
            self.assertTrue(0.5 <= target["articulation_rad"] <= 0.6)
            self.assertTrue(all(-0.1 <= value <= -0.05 for value in target["translation_m"]))
            self.assertEqual(target["orientation_delta_rad"], 0)
            rotation = target["orientation_rotation"]
            self.assertEqual(rotation["axis"], [0, 0, 1])
            self.assertEqual(rotation["frame"], "world")
            self.assertTrue(-0.4 <= rotation["angle_rad"] <= 0.4)
            self.assertEqual(target["lift_m"], 0.3 if item["object"] == "ketchup" else 0.15)

    def test_seed_repeats_and_subset_retains_draws(self):
        full = self.plan(objects=["box"], samples=3)
        subset = self.plan(objects=["box"], samples=3, profiles=["ood_all", "normal"])
        lookup = {item["episode_id"]: item for item in full["episodes"]}
        self.assertEqual(subset["episodes"], [lookup[item["episode_id"]] for item in subset["episodes"]])
        repeated = build_plan(ROOT, subset["config"], label_sets={"box": label_set("box")})
        self.assertEqual(repeated["episodes"], subset["episodes"])
        other_seed = self.plan(objects=["box"], seed=2)
        self.assertNotEqual(other_seed["episodes"][0]["target"], full["episodes"][0]["target"])

    def test_invalid_ood_intervals_and_mislabeled_flags_fail(self):
        for intervals in ([[0.4, 0.6]], [[0.5, 0.5]], [[1.6, 2.0], [1.8, 2.2]], [[2.0, 2.1]]):
            with self.subTest(intervals=intervals), self.assertRaises(ValueError):
                sample_target("espressomachine", 1, {"articulation_ood": {"intervals_rad": intervals}},
                              joint_limits(ROOT, "espressomachine"))
        self.config["profiles"]["normal"]["factor_flags"]["lift"] = True
        with self.assertRaisesRegex(ValueError, "actual target overrides"):
            self.plan(objects=["box"], samples=1)

    def test_controller_receives_world_yaw_and_requested_angles(self):
        specs = self.plan(objects=["box"], samples=1)["episodes"]
        data, env = _data(len(specs)), EventEnvironment()
        with patch.dict(sys.modules, _fake_rotations()):
            controller = Controller(env, data, specs, timing_for("box"))
        for i, spec in enumerate(specs):
            yaw = spec["target"]["orientation_rotation"]["angle_rad"]
            np.testing.assert_allclose(controller.goal[i, 3:7], [math.cos(yaw/2), 0, 0, math.sin(yaw/2)], atol=1e-6)
            self.assertAlmostEqual(float(controller.angle[i, 0]), spec["target"]["articulation_rad"], places=6)
            self.assertAlmostEqual(float(controller.lift_goal[i, 2] - controller.goal[i, 2]),
                                   spec["target"]["lift_m"], places=6)

    def test_factorial_height_gate_does_not_require_removed_air_rotation_condition(self):
        states, spec, timing = diagnostic_rollout()
        spec["require_lift_pose_success"] = False
        result = evaluate_goal_diagnostics(states, spec, timing)
        self.assertFalse(result["lift_pose_success"])
        self.assertTrue(result["ood_goal_success"])


class FactorialReportTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "configs/ood_multiaxis.json")
        self.plan = build_plan(ROOT, self.config, objects=["box"], samples=2,
                               label_sets={"box": label_set("box")})
        self.completed = {item["episode_id"]: dict(item, valid=True, paper_success=True, public_success=True,
                          ood_goal_success=True, strict_goal_success=True) for item in self.plan["episodes"]}

    def effects(self):
        return _factorial_summaries(self.plan["episodes"], self.completed, self.config["profiles"])

    def test_main_effect_averages_over_other_factors(self):
        for row in self.completed.values():
            row["paper_success"] = not row["factor_flags"]["horizontal"]
        effects = [row for row in self.effects() if row["metric"] == "paper_success"]
        self.assertEqual(len(effects), 7)
        for effect in effects:
            self.assertEqual(effect["complete_octuples"], 2)
            self.assertEqual(effect["difference_pp"], -100 if effect["factors"] == ["horizontal"] else 0)

    def test_pair_and_triple_effects_have_difference_of_differences_scale(self):
        for row in self.completed.values():
            row["paper_success"] = row["profile"] != "ood_all"
        effects = {tuple(row["factors"]): row["difference_pp"] for row in self.effects() if row["metric"] == "paper_success"}
        self.assertEqual(effects[("horizontal",)], -25)
        self.assertEqual(effects[("horizontal", "lift")], -50)
        self.assertEqual(effects[("horizontal", "lift", "articulation")], -100)

    def test_missing_octuples_and_invalid_completed_failures(self):
        row = self.completed.pop("box__ood_all__0000")
        effects = self.effects()
        self.assertTrue(all(effect["complete_octuples"] == effect["missing_octuples"] == 1 for effect in effects))
        row.update(valid=False, paper_success=False)
        self.completed[row["episode_id"]] = row
        effect = next(item for item in self.effects() if item["factors"] == ["horizontal"] and item["metric"] == "paper_success")
        self.assertEqual(effect["difference_pp"], -12.5)
        self.assertEqual(effect["complete_octuples"], 2)
        next(item for item in self.plan["episodes"] if item["episode_id"] == "box__normal__0000")["seed"] += 1
        with self.assertRaisesRegex(ValueError, "matching initial references"):
            self.effects()

    def test_report_has_eight_conditions_paired_normal_and_well_formed_tables(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory / "plan.json").write_text(json.dumps(self.plan))
            (directory / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in self.completed.values()))
            generate_report(directory)
            summary = json.loads((directory / "summary.json").read_text())
            self.assertEqual(len(summary["profiles"]), 8)
            self.assertTrue(all(row["reference_profile"] == "normal" for row in summary["paired_comparisons"]))
            self.assertEqual(len(summary["factorial_comparisons"]), 28)
            self.assertEqual(summary["completed"], 16)
            md = (directory / "report.md").read_text()
            self.assertIn("3因子・8通りの組合せ", md)
            self.assertIn("[0.2, 0.4]", md)
            self.assertIn("論文 Table 3", md)
            self.assertNotIn("air_control", md)
            widths = []
            for line in md.splitlines() + [""]:
                if line.startswith("|"):
                    widths.append(line.count("|"))
                elif widths:
                    self.assertEqual(len(set(widths)), 1)
                    widths = []


if __name__ == "__main__":
    unittest.main()
