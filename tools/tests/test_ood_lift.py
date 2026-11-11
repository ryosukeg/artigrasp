"""Intermediate OOD-height success must remain distinct from final paper success."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ood_suite.runtime import lift_metrics


class LiftMetricTests(unittest.TestCase):
    def test_final_placement_does_not_hide_missed_lift(self):
        states = np.zeros((9, 138), dtype=np.float32)
        states[:, 124] = [0.6, 0.6, 0.6, 0.7, 0.8, 0.7, 0.6, 0.6, 0.6]
        result = lift_metrics(states, {"target": {"lift_m": 0.8}},
                              {"grasp_steps": 2, "transit_steps": 3})
        self.assertFalse(result["lift_target_success"])
        self.assertAlmostEqual(result["lift_max_height_m"], 0.2, places=5)
        self.assertAlmostEqual(result["lift_target_error_m"], 0.6, places=5)

    def test_reaching_height_after_lift_window_does_not_pass(self):
        states = np.zeros((9, 138), dtype=np.float32)
        states[:, 124] = [0.6, 0.6, 0.6, 0.7, 1.1, 0.8, 1.4, 0.6, 0.6]
        spec = {"target": {"lift_m": 0.8}}
        timing = {"grasp_steps": 2, "transit_steps": 3}
        self.assertFalse(lift_metrics(states, spec, timing)["lift_target_success"])
        states[4, 124] = 1.4
        self.assertTrue(lift_metrics(states, spec, timing)["lift_target_success"])

    def test_invalid_lift_is_failure_without_fabricated_zero_error(self):
        result = lift_metrics(None, None, None, valid=False)
        self.assertFalse(result["lift_target_success"])
        self.assertIsNone(result["lift_target_error_m"])


if __name__ == "__main__":
    unittest.main()
