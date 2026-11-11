"""Normal baseline reports use the existing five-condition metric definitions."""

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from ood_suite.five_report import generate_report


class NormalReportTests(unittest.TestCase):
    def write_plan(self, root, objects=('box',), profiles=('normal',), samples=3):
        episodes = [dict(episode_id='{}__{}__{:04d}'.format(obj, profile, index))
                    for obj in objects for profile in profiles for index in range(samples)]
        plan = dict(samples_per_object=samples, objects=list(objects),
                    profiles=list(profiles), episodes=episodes)
        (root / 'plan.json').write_text(json.dumps(plan), encoding='utf-8')

    def test_normal_report_uses_dynamic_counts_and_only_existing_artifact_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_plan(root)
            rows = [dict(episode_id='box__normal__0000', object='box', profile='normal',
                         valid=True, ood_goal_success=True, paper_success=True,
                         paper_position_error_m=.01, paper_orientation_error_rad=.1,
                         paper_articulation_error_rad=.2),
                    dict(episode_id='box__normal__0001', object='box', profile='normal',
                         valid=True, ood_goal_success=False, paper_success=True,
                         paper_position_error_m=.59, paper_orientation_error_rad=1.1,
                         paper_articulation_error_rad=1.2),
                    dict(episode_id='box__normal__0002', object='box', profile='normal',
                         valid=False, ood_goal_success=True, paper_success=True,
                         paper_position_error_m=99)]
            (root / 'episodes.jsonl').write_text(
                '\n'.join(json.dumps(row) for row in rows) + '\n', encoding='utf-8')
            generate_report(root)

            summary = json.loads((root / 'summary.json').read_text())
            result = summary['aggregates'][0]
            self.assertTrue(summary['complete'])
            self.assertEqual(result['successes'], 1)
            self.assertAlmostEqual(result['SucT'], 1 / 3)
            self.assertAlmostEqual(result['paper_SucT'], 2 / 3)
            self.assertAlmostEqual(result['PE'], .30)  # Includes the failed trial, no clipping.
            self.assertEqual(result['PE_n'], 2)
            self.assertEqual(result['invalid'], 1)
            with (root / 'metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
                csv_rows = list(csv.DictReader(stream))
            self.assertEqual(len(csv_rows), 1)
            self.assertEqual(csv_rows[0]['profile'], 'normal')

            markdown = (root / 'report.md').read_text(encoding='utf-8')
            page = (root / 'report.html').read_text(encoding='utf-8')
            for text in (markdown, page):
                self.assertIn('通常条件（normal）の評価', text)
                self.assertIn('全1物体', text)
                self.assertIn('各物体・条件3シーケンス', text)
                self.assertNotIn('5条件', text)
                self.assertNotIn('全10物体', text)
                self.assertNotIn('100シーケンス', text)
                self.assertNotIn('videos/index.html', text)
                self.assertNotIn('physics_constraints.json', text)
            self.assertIn('全1組のCSV', markdown)
            self.assertIn('絶対関節角0.50〜0.60 rad', markdown)
            self.assertIn('ketchupは0.30 m', markdown)
            self.assertNotIn('水平0.30–0.80 m', markdown)

    def test_missing_measurements_remain_explicitly_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_plan(root, objects=('box', 'phone'), samples=4)
            (root / 'physics_constraints.json').write_text('{}')
            (root / 'videos').mkdir()
            (root / 'videos/index.html').write_text('preview')
            generate_report(root)
            summary = json.loads((root / 'summary.json').read_text())
            self.assertFalse(summary['complete'])
            self.assertEqual(summary['aggregates'][0]['planned'], 8)
            self.assertIsNone(summary['aggregates'][0]['PE'])
            markdown = (root / 'report.md').read_text(encoding='utf-8')
            self.assertIn('**0/8試行完了**（計算中。', markdown)
            self.assertIn('全2物体の集計', markdown)
            self.assertIn('全2組のCSV', markdown)
            self.assertIn('](physics_constraints.json)', markdown)
            self.assertIn('](videos/index.html)', markdown)

    def test_normal_and_ood_profiles_can_share_a_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_plan(root, profiles=('normal', 'height'), samples=2)
            generate_report(root)
            markdown = (root / 'report.md').read_text(encoding='utf-8')
            self.assertIn('# 通常条件とOOD評価：2条件', markdown)
            self.assertIn('## 通常（normal）', markdown)
            self.assertIn('## 高さ', markdown)
            self.assertIn('水平0.30–0.80 m', markdown)


if __name__ == '__main__':
    unittest.main()
