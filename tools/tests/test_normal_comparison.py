"""Contract checks for completed, paired normal/OOD result comparisons."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("normal_comparison", ROOT / "scripts/compare_normal_ood.py")
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


class NormalComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.normal = self.root / "normal"
        self.ood = self.root / "ood"
        self.output = self.root / "comparison"
        self._run(self.normal, ["normal"])
        self._run(self.ood, list(comparison.OOD_PROFILES))

    def _run(self, directory, profiles):
        directory.mkdir()
        checkpoint = "weights/policy.pt"
        config = dict(evaluation_protocol="five_v1", checkpoints={"cooperative_left": checkpoint},
                      normal_goal=dict(source="paper_supplement_b4", articulation_rad=[.5,.6], yaw_rad=[-.4,.4]),
                      physics_validation={}, runtime=dict(batch_size=2, num_threads=2, device="cuda"),
                      video_metric="ood_goal_success")
        plan = dict(objects=["box","phone"], profiles=profiles, samples_per_object=2, seed=1,
                    split="test", label_counts={"box":{},"phone":{}}, sampling="paired with replacement",
                    config=config, episodes=[])
        rows = []
        for obj in plan["objects"]:
            for sample in range(2):
                for profile in profiles:
                    spec = dict(episode_id="{}__{}__{:04d}".format(obj,profile,sample), object=obj,
                                profile=profile, sample_index=sample, seed=sample+10, split="test",
                                paper_object=obj == "box", grasp_label_index=sample, arti_label_index=sample,
                                grasp_label={"frame":sample}, arti_label={"frame":sample+2},
                                target=dict(changed_axes=[] if profile == "normal" else ["intervention"],
                                            normal_goal_source="paper_supplement_b4", articulation_rule="absolute",
                                            articulation_rad=.55, translation_m=[-.08,-.07], lift_m=.15,
                                            orientation_rotation=dict(axis_name="z", frame="world", stage="both", angle_rad=.1)))
                    plan["episodes"].append(spec)
                    normal = profile == "normal"
                    if normal:
                        valid = not (obj == "box" and sample == 1)
                        success = obj == "box" and sample == 0 or obj == "phone" and sample == 1
                        pe = .01 if obj == "box" else .2 if sample == 0 else .03
                    else:
                        valid = not (obj == "phone" and sample == 1)
                        success = obj == "box" and sample == 1 or obj == "phone" and sample == 0
                        pe = .09 if obj == "box" and sample == 0 else .02 if obj == "box" else .04
                    row = copy.deepcopy(spec)
                    # Deliberately set invalid flags true: invalid trials must still
                    # enter both ordinary and paired success summaries as failures.
                    row.update(valid=valid, ood_goal_success=success if valid else True,
                               paper_success=success if valid else True,
                               paper_position_error_m=pe if valid else None,
                               paper_orientation_error_rad=.1 if valid else None,
                               paper_articulation_error_rad=.2 if valid else None,
                               timing=dict(grasp_steps=260), environment_config={"control_dt":.01},
                               policy_device="cuda:0", initial_articulation_rad=.1)
                    rows.append(row)
        comparison.write_json(directory / "plan.json", plan)
        (directory / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in rows)+"\n")
        source = {p:"a"*64 for p in comparison.CORE_SOURCES}
        source["raisimGymTorch/raisimGymTorch/env/VectorizedEnvironment.hpp"] = "b"*64
        comparison.write_json(directory / "provenance.json", dict(source_sha256=source,
            input_sha256={checkpoint:"c"*64}, plan_sha256=comparison.file_sha256(directory / "plan.json")))
        comparison.write_json(directory / "execution.json", dict(python="3.8.20", platform="Linux-test", device="cuda",
            runtime={"numpy":"1.23.1","torch":"2.4.1+cu121","ruamel.yaml":"0.17.16",
                     "raisimGymTorch.env.bin.compose_eval":"available", "compose_eval_sha256":"d"*64}))

    def _replace_plan(self, directory, change):
        plan = comparison.read_json(directory / "plan.json")
        change(plan)
        comparison.write_json(directory / "plan.json",plan)
        provenance = comparison.read_json(directory / "provenance.json")
        provenance["plan_sha256"] = comparison.file_sha256(directory / "plan.json")
        comparison.write_json(directory / "provenance.json",provenance)

    def test_deltas_paired_discordance_and_invalid_denominators(self):
        before = {str(p):comparison.file_sha256(p) for directory in (self.normal,self.ood) for p in directory.iterdir()}
        comparison.generate(self.normal,self.ood,self.output)
        summary = comparison.read_json(self.output / "summary.json")
        self.assertTrue(summary["complete"])
        all_rows = {r["profile"]:r for r in summary["aggregates"] if r["scope"] == "all"}
        self.assertEqual(all_rows["normal"]["planned"],4)
        self.assertEqual(all_rows["normal"]["invalid"],1)
        self.assertEqual(all_rows["normal"]["PE_n"],3)
        self.assertEqual(all_rows["normal"]["successes"],2)
        self.assertEqual(all_rows["horizontal"]["delta_SucT_pp"],0)
        self.assertAlmostEqual(all_rows["horizontal"]["delta_PE"],-.03)
        pair = next(r for r in summary["paired"] if r["scope"] == "all" and r["profile"] == "horizontal")
        self.assertEqual(pair["SucT_normal_only"],2)
        self.assertEqual(pair["SucT_ood_only"],2)
        self.assertEqual(pair["SucT_both_success"],0)
        self.assertEqual(pair["PE_paired_n"],2)
        self.assertAlmostEqual(pair["PE_paired_delta"],-.04)
        paper = next(r for r in summary["aggregates"] if r["scope"] == "paper" and r["profile"] == "horizontal")
        self.assertEqual(paper["planned"],2)
        self.assertAlmostEqual(paper["delta_PE"],.045)
        self.assertEqual({p.name for p in self.output.iterdir()},
                         {"summary.json","comparison.csv","per-object.csv","paired_comparison.csv","report.md","report.html"})
        self.assertEqual(before,{str(p):comparison.file_sha256(p) for directory in (self.normal,self.ood) for p in directory.iterdir()})
        self.assertIn("normal からの差",(self.output / "report.html").read_text())

    def test_missing_or_incomplete_results_leave_no_output(self):
        rows = comparison.read_rows(self.normal / "episodes.jsonl")
        (self.normal / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in rows[:-1])+"\n")
        with self.assertRaisesRegex(ValueError,"Incomplete results"):
            comparison.generate(self.normal,self.ood,self.output)
        self.assertFalse(self.output.exists())
        (self.normal / "episodes.jsonl").unlink()
        with self.assertRaisesRegex(ValueError,"Missing input"):
            comparison.generate(self.normal,self.ood,self.output)
        self.assertFalse(self.output.exists())

    def test_duplicate_result_is_rejected(self):
        path = self.normal / "episodes.jsonl"
        with path.open("a") as stream:
            stream.write(path.read_text().splitlines()[0]+"\n")
        with self.assertRaisesRegex(ValueError,"Duplicate result"):
            comparison.generate(self.normal,self.ood,self.output)
        self.assertFalse(self.output.exists())

    def test_paired_label_mismatch_is_rejected(self):
        self._replace_plan(self.normal,lambda plan:plan["episodes"][0]["grasp_label"].update(frame=999))
        rows = comparison.read_rows(self.normal / "episodes.jsonl")
        rows[0]["grasp_label"]["frame"] = 999
        (self.normal / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in rows)+"\n")
        with self.assertRaisesRegex(ValueError,"Paired sample identity mismatch"):
            comparison.generate(self.normal,self.ood,self.output)
        self.assertFalse(self.output.exists())

    def test_scoring_source_change_is_rejected(self):
        path = self.normal / "provenance.json"
        provenance = comparison.read_json(path)
        provenance["source_sha256"]["tools/ood_suite/metrics.py"] = "e"*64
        comparison.write_json(path,provenance)
        with self.assertRaisesRegex(ValueError,"source hash differs"):
            comparison.generate(self.normal,self.ood,self.output)
        self.assertFalse(self.output.exists())

    def test_runtime_binary_change_is_rejected(self):
        path = self.normal / "execution.json"
        execution = comparison.read_json(path)
        execution["runtime"]["compose_eval_sha256"] = "e"*64
        comparison.write_json(path,execution)
        with self.assertRaisesRegex(ValueError,"Runtime version/binary differs"):
            comparison.generate(self.normal,self.ood,self.output)
        self.assertFalse(self.output.exists())

    def test_missing_valid_error_is_not_silently_imputed(self):
        path = self.normal / "episodes.jsonl"
        rows = comparison.read_rows(path)
        rows[0]["paper_position_error_m"] = None
        path.write_text("\n".join(json.dumps(r) for r in rows)+"\n")
        with self.assertRaisesRegex(ValueError,"Missing/nonfinite final error"):
            comparison.generate(self.normal,self.ood,self.output)
        self.assertFalse(self.output.exists())

    def test_tampered_plan_hash_is_rejected(self):
        path = self.normal / "plan.json"
        path.write_text(path.read_text()+"\n")
        with self.assertRaisesRegex(ValueError,"plan hash mismatch"):
            comparison.generate(self.normal,self.ood,self.output)
        self.assertFalse(self.output.exists())

    def test_new_report_source_is_allowed(self):
        path = self.normal / "provenance.json"
        provenance = comparison.read_json(path)
        provenance["source_sha256"]["tools/ood_suite/five_report.py"] = "f"*64
        comparison.write_json(path,provenance)
        result = comparison.build_comparison(self.normal,self.ood)
        self.assertTrue(result["complete"])


if __name__ == "__main__":
    unittest.main()
