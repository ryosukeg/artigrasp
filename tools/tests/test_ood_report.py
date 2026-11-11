import json
from pathlib import Path
import tempfile
import unittest

from tools.ood_suite.report import generate_report


def planned_row(identifier, obj, profile="ood_combined", sample_index=0):
    return dict(episode_id=identifier, object=obj, profile=profile, sample_index=sample_index)


def result_row(planned, success=True, valid=True):
    return dict(planned, valid=valid, paper_success=success, public_success=success,
                paper_position_error_m=0.02 if valid else None,
                paper_orientation_error_rad=0.1 if valid else None,
                paper_articulation_error_rad=0.3 if valid else None,
                public_position_error_m=0.12 if valid else None,
                public_orientation_error_rad=0.4 if valid else None,
                public_articulation_error_rad=1.2 if valid else None,
                public_position_error_clipped_m=0.1 if valid else None,
                public_orientation_error_clipped_rad=0.3 if valid else None,
                public_articulation_error_clipped_rad=1.0 if valid else None,
                target_reachable=False)


class ReportTests(unittest.TestCase):
    def make_run(self, directory, plan_rows, rows):
        (directory / "plan.json").write_text(json.dumps({"episodes": plan_rows}), encoding="utf-8")
        (directory / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_denominators_include_invalid_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            planned = [planned_row(str(i), "box", sample_index=i) for i in range(3)]
            self.make_run(directory, planned, [result_row(planned[0]), result_row(planned[1], valid=False)])
            generate_report(directory)
            summary = json.loads((directory / "summary.json").read_text())
            group = summary["groups"][0]
            self.assertEqual(group["paper_success_count"], 1)
            self.assertEqual(group["paper_success_rate"], 1 / 3)
            self.assertEqual(group["invalid"], 1)
            self.assertEqual(group["missing"], 1)
            self.assertIsNone(group["paper_success_wilson95"])
            self.assertEqual(group["paper_position_error_m_n"], 1)
            self.assertAlmostEqual(group["paper_position_error_m_mean"], 0.02)
            report = (directory / "report.md").read_text()
            self.assertIn("1/3", report)
            self.assertIn("暫定", report)
            self.assertIn("物体別のTable 3値は掲載されていません", report)

    def test_paper_scope_excludes_phone_and_capsulemachine(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            planned = [planned_row("box", "box"), planned_row("phone", "phone"), planned_row("capsule", "capsulemachine")]
            self.make_run(directory, planned, [result_row(row) for row in planned])
            generate_report(directory)
            summaries = json.loads((directory / "summary.json").read_text())["aggregates"]
            self.assertEqual(summaries[0]["planned"], 3)
            self.assertEqual(summaries[1]["scope"], "paper_objects")
            self.assertEqual(summaries[1]["planned"], 1)

    def test_paired_comparisons_and_video_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            planned = [planned_row("id", "box", "id_control"), planned_row("ood", "box")]
            self.make_run(directory, planned, [result_row(planned[0], success=True), result_row(planned[1], success=False)])
            video_dir = directory / "videos" / "paper_success"
            video_dir.mkdir(parents=True)
            (video_dir / "index.json").write_text("{}")
            (video_dir / "index.md").write_text("# Videos")
            generate_report(directory)
            pair = json.loads((directory / "summary.json").read_text())["paired_comparisons"][0]
            self.assertEqual(pair["id_only_success"], 1)
            self.assertEqual(pair["success_difference_pp"], -100.0)
            self.assertIn("videos/paper_success/index.md", (directory / "report.md").read_text())

    def test_duplicate_results_are_not_double_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            planned = [planned_row("box", "box")]
            result = result_row(planned[0])
            self.make_run(directory, planned, [result, result])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                generate_report(directory)

    def test_ketchup_controller_fix_is_disclosed_separately_from_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            planned = [planned_row("ketchup", "ketchup")]
            result = result_row(planned[0])
            result["controller_changes"] = ["ketchup_forward_requested_articulation"]
            self.make_run(directory, planned, [result])
            generate_report(directory)
            summary = json.loads((directory / "summary.json").read_text())
            self.assertEqual(summary["controller_changes"], ["ketchup_forward_requested_articulation"])
            report = (directory / "report.md").read_text()
            self.assertIn("ketchupの目標角度入力の補正", report)
            self.assertIn("軌跡は公開スクリプトと異なり得ます", report)


if __name__ == "__main__":
    unittest.main()
