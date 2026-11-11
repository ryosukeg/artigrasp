"""Checks for paired five-condition goals and the requested success definition."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from ood_suite.targets import build_plan,load_config
from ood_suite.runtime import Controller,timing_for
from ood_suite.five_report import summarize,generate_report
from tools.tests.test_ood_runtime import _data,_fake_rotations,_specs,EventEnvironment


class FiveConditionTests(unittest.TestCase):
    def test_only_requested_conditions_and_paired_targets(self):
        config=load_config(ROOT/'configs/ood_five.json')
        plan=build_plan(ROOT,config,samples=4)
        self.assertEqual(plan['profiles'],['horizontal','height','orientation','articulation','all'])
        self.assertEqual(len(plan['episodes']),200)
        for obj in plan['objects']:
            for i in range(4):
                group={e['profile']:e for e in plan['episodes'] if e['object']==obj and e['sample_index']==i}
                for e in group.values():
                    self.assertEqual(e['grasp_label'],group['all']['grasp_label'])
                    self.assertEqual(e['arti_label'],group['all']['arti_label'])
                    self.assertTrue(e['target_reachable'])
                for profile,key in [('horizontal','translation_m'),('height','lift_m'),
                                    ('orientation','orientation_rotation'),('articulation','articulation_rad')]:
                    self.assertEqual(group[profile]['target'][key],group['all']['target'][key])
                angle=group['articulation']['target']['articulation_rad']
                if obj=='espressomachine': self.assertTrue(.2<=angle<=.4)
                elif obj=='phone': self.assertTrue(2.4<=angle<=2.45)
                else: self.assertGreater(angle,1.5)

    def test_clearance_changes_placement_without_changing_lift(self):
        specs=_specs()
        specs[0]['target']['placement_z_offset_m']=.012
        with patch.dict(sys.modules,_fake_rotations()):
            c=Controller(EventEnvironment(),_data(),specs,timing_for('box'))
        self.assertAlmostEqual(float(c.goal[0,2]),.612,places=6)
        self.assertAlmostEqual(float(c.lift_goal[0,2]),1.3,places=6)

    def test_primary_success_requires_height_and_errors_include_failures(self):
        rows=[dict(valid=True,ood_goal_success=False,paper_success=True,
                   paper_position_error_m=.01,paper_orientation_error_rad=.1,paper_articulation_error_rad=.2),
              dict(valid=True,ood_goal_success=True,paper_success=True,
                   paper_position_error_m=.03,paper_orientation_error_rad=.2,paper_articulation_error_rad=.4),
              dict(valid=False,ood_goal_success=True,paper_success=True)]
        r=summarize(rows,3)
        self.assertAlmostEqual(r['SucT'],1/3)
        self.assertAlmostEqual(r['paper_SucT'],2/3)
        self.assertAlmostEqual(r['PE'],.02)
        self.assertEqual(r['PE_n'],2)
        self.assertEqual(r['invalid'],1)

    def test_empty_report_is_explicitly_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            plan=dict(samples_per_object=100,objects=['box'],profiles=['height'],
                      episodes=[dict(episode_id='box__height__{:04d}'.format(i)) for i in range(100)])
            (root/'plan.json').write_text(json.dumps(plan))
            generate_report(root)
            result=json.loads((root/'summary.json').read_text())
            self.assertFalse(result['complete'])
            self.assertIsNone(result['objects'][0]['PE'])
            self.assertTrue((root/'metrics.csv').is_file())


if __name__=='__main__': unittest.main()
