"""Synthetic stopping-rule boundary tests; no simulator, GPU, or policy."""

import copy
import unittest

import numpy as np

from adaptive_stop import AdaptiveStop, STOP_SETTINGS, object_metrics


def fixture(count=1, target=1.):
    states = np.zeros((count, 138), dtype=np.float64)
    states[:, 125] = 1.
    goals = np.zeros((count, 8), dtype=np.float64)
    goals[:, 3] = 1.
    goals[:, 7] = target
    return states, goals


def quat_z(angle):
    return np.array([np.cos(angle / 2), 0., 0., np.sin(angle / 2)])


class ObjectMetricsTests(unittest.TestCase):
    def test_units_order_normalization_and_quaternion_sign(self):
        states, goals = fixture(2, target=.8)
        states[:, 122:125] = [.03, .04, 0]
        states[:, 129] = .6
        states[0, 125:129] = quat_z(.3) * 7
        states[1, 125:129] = -quat_z(.3) * 2
        goals[:, 3:7] *= -3
        np.testing.assert_allclose(object_metrics(states, goals), [[.05, .2, .3]] * 2, atol=1e-14)

    def test_invalid_quaternions_goals_and_object_values_are_nan_rows(self):
        states, goals = fixture(5)
        states[0, 125:129] = 0
        states[1, 125] = np.nan
        states[2, 129] = np.inf
        goals[3, 3:7] = 0
        goals[4, 7] = np.nan
        self.assertTrue(np.isnan(object_metrics(states, goals)).all())


class AdaptiveStopTests(unittest.TestCase):
    def test_defaults_are_three_to_ten_seconds_and_closed_windows(self):
        self.assertEqual(STOP_SETTINGS["control_dt"], .01)
        self.assertEqual(STOP_SETTINGS["opening_start_step"], 1260)
        self.assertEqual(STOP_SETTINGS["min_opening_steps"], 300)
        self.assertEqual(STOP_SETTINGS["max_opening_steps"], 1000)
        self.assertEqual(STOP_SETTINGS["success_hold_steps"] + 1, 51)
        self.assertEqual(STOP_SETTINGS["stall_window_steps"] + 1, 201)

    def test_earliest_success_at_300_and_success_precedes_stall(self):
        states, goals = fixture(target=0.)
        stop = AdaptiveStop(goals)
        for step in range(300):
            self.assertFalse(stop.update(step, states)[0])
        self.assertTrue(stop.update(300, states)[0])
        self.assertEqual(stop.reasons, ["success_hold"])
        self.assertEqual(stop.stop_step.tolist(), [300])
        self.assertEqual(stop.first_success_step.tolist(), [0])
        self.assertEqual(len(stop.metric_history), 301)

    def test_hold_includes_both_endpoints_and_one_failure_breaks_continuity(self):
        states, goals = fixture(target=0.)
        stop = AdaptiveStop(goals)
        # The failure at 250 remains in the [250,300] closed window, but not
        # [251,301]. Object motion also prevents an unrelated stall at 300.
        for step in range(302):
            states[0, 129] = .5 if step == 250 else .001 * (step % 100)
            stop.update(step, states)
            if step == 300:
                self.assertFalse(stop.done[0])
        self.assertEqual(stop.stop_step[0], 301)
        self.assertEqual(stop.reasons[0], "success_hold")

    def test_instantaneous_success_is_strict_for_all_three_metrics(self):
        states, goals = fixture(4, target=0.)
        states[0, 122] = .05
        states[1, 129] = .5
        states[2, 125:129] = quat_z(.200000001)
        states[3, 122] = np.nextafter(.05, 0.)
        stop = AdaptiveStop(goals)
        stop.update(0, states)
        self.assertEqual(stop.first_success_step.tolist(), [-1, -1, -1, 0])

    def test_current_success_suppresses_stall_until_full_hold(self):
        states, goals = fixture(target=0.)
        stop = AdaptiveStop(goals)
        for step in range(341):
            states[0, 129] = .5001 if step < 290 else .4999
            stop.update(step, states)
            if 300 <= step < 340:
                self.assertFalse(stop.done[0])
        self.assertEqual(stop.first_success_step[0], 290)
        self.assertEqual(stop.stop_step[0], 340)
        self.assertEqual(stop.reasons[0], "success_hold")

    def test_pending_success_hold_can_timeout_or_exit_to_stall(self):
        states, goals = fixture(2, target=0.)
        stop = AdaptiveStop(goals)
        for step in range(1001):
            # Row zero enters success too late to finish its 50-step hold.
            states[0, 129] = .6 + .1 * (step % 2) if step < 960 else .4999
            # Row one exits success at the first eligible step: its tiny
            # object-motion window qualifies for stall immediately.
            states[1, 129] = .4999 if step < 300 else .5001
            stop.update(step, states)
        self.assertEqual(stop.reasons, ["timeout", "stalled"])
        self.assertEqual(stop.stop_step.tolist(), [1000, 300])

    def test_stall_includes_first_of_201_states_and_ignores_hand_motion(self):
        states, goals = fixture()
        stop = AdaptiveStop(goals)
        for step in range(302):
            states[0, 129] = .03 if step <= 100 else 0.
            states[0, 6] = step * .1  # Moving hand does not defeat object stall.
            stop.update(step, states)
            if step == 300:
                self.assertFalse(stop.done[0])
        self.assertEqual(stop.stop_step[0], 301)
        self.assertEqual(stop.reasons[0], "stalled")

    def test_stall_checks_max_excursion_not_just_endpoint_difference(self):
        states, goals = fixture(3)
        stop = AdaptiveStop(goals)
        for step in range(301):
            states[:, 122:125] = 0
            states[:, 125:129] = [1, 0, 0, 0]
            states[:, 129] = 0
            if step == 200:
                states[0, 122] = .003
                states[1, 125:129] = quat_z(.03)
                states[2, 129] = .03
            stop.update(step, states)
        self.assertFalse(stop.done.any())

    def test_stall_thresholds_are_strict_and_quaternion_sign_is_irrelevant(self):
        states, goals = fixture(4)
        stop = AdaptiveStop(goals)
        for step in range(301):
            states[:, 122:125] = 0
            states[:, 129] = 0
            states[:, 125:129] = [1 if step % 2 else -3, 0, 0, 0]
            if step == 200:
                states[0, 129] = .02
                states[1, 122] = .002
                states[2, 125:129] = quat_z(.020000001)
            stop.update(step, states)
        self.assertEqual(stop.done.tolist(), [False, False, False, True])
        self.assertEqual(stop.reasons, [None, None, None, "stalled"])

    def test_invalid_is_immediate_and_precedes_success(self):
        states, goals = fixture(4, target=0.)
        stop = AdaptiveStop(goals)
        for step in range(301):
            invalid = np.zeros(4, bool)
            if step == 0:
                states[0, 125:129] = 0
                states[1, 12] = np.nan
                invalid[2] = True
            if step == 300:
                invalid[3] = True
            stop.update(step, states, invalid)
            states[:3] = fixture(3, target=0.)[0]
        self.assertEqual(stop.reasons, ["invalid"] * 4)
        self.assertEqual(stop.stop_step.tolist(), [0, 0, 0, 300])
        self.assertEqual(stop.first_success_step.tolist(), [-1, -1, -1, 0])

    def test_latched_stop_and_history_ignore_later_states_and_invalid(self):
        states, goals = fixture(target=0.)
        stop = AdaptiveStop(goals)
        for step in range(301):
            stop.update(step, states)
        saved = stop.metric_history[-1].copy()
        states[:] = np.nan
        returned = stop.update(301, states, np.array([True]))
        returned[:] = False
        self.assertTrue(stop.done[0])
        self.assertEqual(stop.reasons[0], "success_hold")
        self.assertEqual(stop.stop_step[0], 300)
        self.assertEqual(stop.first_success_step[0], 0)
        np.testing.assert_array_equal(stop.metric_history[-1], saved)

    def test_timeout_at_1000_and_higher_priority_events_on_same_step(self):
        states, goals = fixture(4, target=2.)
        stop = AdaptiveStop(goals)
        for step in range(1001):
            states[:, 129] = .1 * (step % 2)
            if step >= 950:
                states[1, 129] = 2.
            if step >= 800:
                states[2, 129] = 0.
            if step == 799:
                states[2, 129] = .1
            stop.update(step, states, np.array([False, False, False, step == 1000]))
            if step == 999:
                self.assertFalse(stop.done.any())
        self.assertEqual(stop.stop_step.tolist(), [1000] * 4)
        self.assertEqual(stop.reasons, ["timeout", "success_hold", "stalled", "invalid"])
        self.assertEqual(stop.first_success_step.tolist(), [-1, 950, -1, -1])

    def test_nonconsecutive_steps_and_invalid_shapes_do_not_consume_history(self):
        states, goals = fixture()
        stop = AdaptiveStop(goals)
        for step in (1, -1, .0, True):
            with self.assertRaises(ValueError):
                stop.update(step, states)
        with self.assertRaises(ValueError):
            stop.update(0, states, [1])
        with self.assertRaises(ValueError):
            stop.update(0, np.zeros((2, 138)))
        self.assertEqual(len(stop.metric_history), 0)
        stop.update(0, states)
        for step in (0, 2):
            with self.assertRaises(ValueError):
                stop.update(step, states)
        stop.update(1, states)
        self.assertEqual(len(stop.metric_history), 2)

    def test_goal_and_input_arrays_are_copied(self):
        states, goals = fixture(target=0.)
        stop = AdaptiveStop(goals)
        goals[:] = np.nan
        stop.update(0, states)
        states[:] = np.nan
        np.testing.assert_array_equal(stop.metric_history[0], np.zeros((1, 3)))
        self.assertEqual(stop.first_success_step[0], 0)

    def test_invalid_settings_are_rejected(self):
        _, goals = fixture()
        for key, value in [("control_dt", 0), ("success_hold_steps", True),
                           ("min_opening_steps", 49), ("max_opening_steps", 299),
                           ("priority", ["timeout", "invalid", "stalled", "success_hold"])]:
            settings = copy.deepcopy(STOP_SETTINGS)
            settings[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                AdaptiveStop(goals, settings)


if __name__ == "__main__":
    unittest.main()
