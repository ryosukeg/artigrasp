import copy
import math
import unittest

from tools.ood_suite.metrics import evaluate_episode, public_euler_y_rad, quaternion_geodesic_rad, wilson_interval


def rollout(protocol="standard", angle=0.55):
    timing = {"protocol": protocol, "grasp_steps": 2, "transit_steps": 4, "arti_steps": 102}
    n = 2 + (2 if protocol == "ketchup" else 3) * 4 + 102
    row = [0.0] * 138
    row[3] = row[129] = row[137] = angle
    row[125] = row[133] = 1.0
    return [row.copy() for _ in range(n + 1)], timing


class MetricTests(unittest.TestCase):
    def test_paper_uses_final_physical_pose_not_target_pose(self):
        states, timing = rollout()
        states[-1][125:129] = [math.cos(0.5), 0.0, 0.0, math.sin(0.5)]
        result = evaluate_episode(states, {"object": "box"}, timing)
        self.assertTrue(result["valid"])
        self.assertFalse(result["paper_success"])
        self.assertTrue(result["public_success"])
        self.assertAlmostEqual(result["paper_orientation_error_rad"], 1.0)
        self.assertAlmostEqual(result["public_orientation_error_rad"], 0.0)

    def test_paper_tests_post_last_step_and_public_tests_previous_state(self):
        states, timing = rollout()
        states[-1][122] = 0.5
        states[-1][3] = states[-1][129] = 2.5
        result = evaluate_episode(states, {"object": "box"}, timing)
        self.assertFalse(result["paper_success"])
        self.assertTrue(result["public_success"])
        self.assertAlmostEqual(result["paper_position_error_m"], 0.5)
        self.assertAlmostEqual(result["public_articulation_error_rad"], 0.0)

    def test_public_independent_latches_do_not_require_final_target(self):
        states, timing = rollout()
        for row in states:
            row[0] = 0.7
        # Arrival at original loop step g+t reads physical state g+t-1.
        states[timing["grasp_steps"] + timing["transit_steps"] - 1][0] = 0.01
        states[-1][122] = 0.7
        result = evaluate_episode(states, {"object": "box"}, timing)
        self.assertTrue(result["public_arrived"])
        self.assertTrue(result["public_success"])
        self.assertFalse(result["paper_success"])

    def test_standard_two_radian_guard_rejects_perfect_180_degree_goal(self):
        states, timing = rollout(angle=math.pi)
        result = evaluate_episode(states, {"object": "box"}, timing)
        self.assertTrue(result["paper_success"])
        self.assertTrue(result["public_opened"])
        self.assertFalse(result["public_terminal_guard"])
        self.assertFalse(result["public_success"])

    def test_ketchup_latches_change_from_initial_not_distance_to_target(self):
        states, timing = rollout(protocol="ketchup", angle=0.8)
        for row in states:
            row[137] = math.pi
        states[0][129] = states[0][3] = 0.0
        result = evaluate_episode(states, {"object": "ketchup", "initial_articulation_rad": 0.0}, timing)
        self.assertTrue(result["public_opened"])
        self.assertTrue(result["public_success"])
        self.assertFalse(result["paper_success"])
        self.assertGreater(result["paper_articulation_error_rad"], 2.0)

    def test_public_position_sampling_and_clipping(self):
        for protocol in ("standard", "ketchup"):
            states, timing = rollout(protocol)
            # Snapshot at arrive_end (standard), arrive_end+99 (ketchup),
            # with one-step-old physical state in both cases.
            snapshot = 2 + 2 * 4 + (99 if protocol == "ketchup" else 0) - 1
            states[snapshot][0] = 0.25
            result = evaluate_episode(states, {"object": "ketchup" if protocol == "ketchup" else "box"}, timing)
            self.assertAlmostEqual(result["public_position_error_m"], 0.25)
            self.assertAlmostEqual(result["public_position_error_clipped_m"], 0.1)

    def test_strict_position_threshold(self):
        states, timing = rollout()
        states[-1][122] = 0.05
        result = evaluate_episode(states, {"object": "box"}, timing)
        self.assertFalse(result["paper_position_success"])

    def test_articulation_is_not_wrapped_modulo_2pi(self):
        states, timing = rollout(angle=0.0)
        result = evaluate_episode(states, {"object": "box", "target_articulation_rad": 2 * math.pi}, timing)
        self.assertAlmostEqual(result["paper_articulation_error_rad"], 2 * math.pi)
        self.assertFalse(result["paper_success"])

    def test_nonfinite_and_incomplete_are_failures(self):
        states, timing = rollout()
        for invalid_states in (states[:-1], [[0.0]], copy.deepcopy(states)):
            if len(invalid_states) == len(states):
                invalid_states[4][122] = float("nan")
            result = evaluate_episode(invalid_states, {"object": "box"}, timing)
            self.assertFalse(result["valid"])
            self.assertFalse(result["paper_success"])
            self.assertFalse(result["public_success"])
            self.assertIsNone(result["paper_position_error_m"])

    def test_quaternion_sign_and_normalization(self):
        self.assertAlmostEqual(quaternion_geodesic_rad([2, 0, 0, 0], [-1, 0, 0, 0]), 0.0)
        self.assertAlmostEqual(quaternion_geodesic_rad([1, 0, 0, 0], [0, 1, 0, 0]), math.pi)
        self.assertAlmostEqual(public_euler_y_rad([math.cos(0.15), 0, math.sin(0.15), 0]), 0.3)

    def test_wilson_extreme_rates_have_nonzero_uncertainty(self):
        self.assertIsNone(wilson_interval(0, 0))
        low, high = wilson_interval(0, 100)
        self.assertAlmostEqual(low, 0)
        self.assertGreater(high, 0.03)
        low, high = wilson_interval(100, 100)
        self.assertLess(low, 0.97)
        self.assertAlmostEqual(high, 1)


if __name__ == "__main__":
    unittest.main()
