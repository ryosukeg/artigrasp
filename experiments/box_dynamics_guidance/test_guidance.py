"""Inference contracts using the collector's independent fake closed-loop plant.

No RaiSim or local checkpoint is needed. Run with the ArtiGrasp Python:
python -m unittest discover -s experiments/box_dynamics_guidance -p test_guidance.py -v
"""
import contextlib
import io
import unittest

import numpy as np
import torch

import dataset
from guidance import SETTINGS, ScheduledPolicy, Selector, candidate_scores, features
from test_collect import FakeController, FakeEnv, FakePolicy, collect, record


CANDIDATES = np.asarray(SETTINGS["candidates"], np.float32)


def scheduled_record(selector):
    env = FakeEnv()
    schedules = np.zeros((env.count, 15, 3), np.float32)
    policy = ScheduledPolicy(FakePolicy(), env, env.goal.copy(), schedules, selector)
    with contextlib.redirect_stdout(io.StringIO()):
        result = collect.simulate_recorded(env, FakeController(env), policy, schedules)
    return env, policy, schedules, result


class RecordingSelector:
    candidates = CANDIDATES

    def __init__(self, zero=False):
        self.zero = zero
        self.calls = []

    def select(self, right, left, states, goals):
        self.calls.append((right.copy(), left.copy(), states.copy(), goals.copy()))
        count = len(states)
        selected = np.zeros(count, np.int32) if self.zero else (
            np.arange(count, dtype=np.int32) + len(self.calls)) % len(self.candidates)
        return (selected, np.zeros((count, 7, 9), np.float32),
                np.zeros((count, 7)), np.zeros(count, bool))


class FeatureContractTests(unittest.TestCase):
    def test_all_recorded_window_features_match_training_loader_exactly(self):
        schedules = CANDIDATES[np.arange(30).reshape(2, 15) % 7].copy()
        _, _, result = record(schedules)
        arrays, goals, initial, terminal, _, _ = result
        goals = goals.copy()
        goals[1, 3:7] *= -1  # Same goal rotation, opposite stored sign.
        original_goals = goals.copy()
        for episode in range(2):
            saved = {key: value[:, episode].copy() for key, value in arrays.items()}
            saved.update(goal=goals[episode], window_z=schedules[episode],
                         window_start_step=np.arange(1260, 1560, 20),
                         window_end_step=np.arange(1280, 1561, 20),
                         window_valid=np.ones(15, bool), initial_invalid=initial[episode],
                         terminal_bad_observation=terminal[episode])
            for window in range(15):
                with self.subTest(episode=episode, window=window):
                    training, reasons = dataset._window(saved, window, "synthetic")
                    self.assertEqual(reasons, [])
                    offset = 20 * window
                    inference = features(arrays["final_obs_r"][offset],
                                         arrays["final_obs_l"][offset],
                                         arrays["final_states"][offset], goals, CANDIDATES)
                    candidate = int(np.flatnonzero(
                        np.all(CANDIDATES == schedules[episode, window], axis=1))[0])
                    np.testing.assert_array_equal(inference[episode, candidate], training[0])
                    self.assertEqual(inference.shape, (2, 7, dataset.INPUT_DIM))
        np.testing.assert_array_equal(goals, original_goals)


class ScoreContractTests(unittest.TestCase):
    def setUp(self):
        self.goals = np.zeros((2, 8), np.float32)
        self.goals[:, 3] = 1
        self.goals[:, 7] = .7
        self.predictions = np.zeros((2, 7, 9), np.float32)
        self.predictions[:, :, 0] = .7
        self.predictions[:, :, 5] = 1

    def test_task_thresholds_each_contribute_one_and_residual_penalty_is_added(self):
        self.predictions[:, :, 0] += .5
        self.predictions[:, :, 2] += .05
        self.predictions[:, :, 5] = np.cos(.1)
        self.predictions[:, :, 8] = np.sin(.1)
        scores = candidate_scores(self.predictions, self.goals, CANDIDATES)
        expected = 3 + .01 * np.sum(CANDIDATES ** 2, axis=1)
        np.testing.assert_allclose(scores, np.broadcast_to(expected, scores.shape), atol=2e-6)

    def test_quaternion_sign_and_nonzero_scale_do_not_change_scores(self):
        self.predictions[:, :, 5] = np.cos(.35)
        self.predictions[:, :, 8] = np.sin(.35)
        expected = candidate_scores(self.predictions, self.goals, CANDIDATES)
        changed = self.predictions.copy()
        changed[:, :, 5:9] *= np.array([2., -.5, 3., -4., .3, 5., -1.])[:, None]
        goals = self.goals.copy()
        goals[:, 3:7] *= -3
        actual = candidate_scores(changed, goals, CANDIDATES)
        np.testing.assert_allclose(actual, expected, rtol=3e-7, atol=1e-6)

    def test_nonfinite_and_zero_quaternion_candidates_cannot_win(self):
        self.predictions[0, 0, 0] = np.nan
        self.predictions[0, 1, 5:9] = 0
        self.predictions[0, 2, 1] = np.inf
        scores = candidate_scores(self.predictions, self.goals, CANDIDATES)
        self.assertTrue(np.isinf(scores[0, :3]).all())
        self.assertTrue(np.isfinite(scores[0, 3:]).all())
        self.goals[1, 3:7] = 0
        self.assertTrue(np.isinf(candidate_scores(
            self.predictions, self.goals, CANDIDATES)[1]).all())

    def test_selector_falls_back_when_all_candidates_or_input_are_invalid(self):
        class FixedPrediction(torch.nn.Module):
            def __init__(self, predictions):
                super().__init__()
                self.values = torch.from_numpy(predictions.reshape(-1, 9))

            def forward(self, value):
                return self.values

        # Construct a selector without loading a checkpoint; test the real select path.
        selector = Selector.__new__(Selector)
        selector.device = torch.device("cpu")
        selector.candidates = CANDIDATES
        selector.seconds = 0.
        selector.normalization = {"x_mean": torch.zeros(571), "x_std": torch.ones(571),
                                  "y_mean": torch.zeros(9), "y_std": torch.ones(9)}
        env = FakeEnv(count=4)
        right, left = env.observe()
        predictions = np.zeros((4, 7, 9), np.float32)
        predictions[:, :, 5] = 1
        predictions[:, :, 0] = env.goal[:, None, 7]
        predictions[0, :, 5:9] = 0  # Every prediction invalid: fallback to zero.
        predictions[1, 0, 0] = np.nan  # First candidate invalid, others usable.
        right[2, 40] = np.nan  # Inputs invalid even though output is finite.
        selector.model = FixedPrediction(predictions)
        indices, _, scores, fallback = selector.select(
            right, left, env.get_global_state(), env.goal)
        np.testing.assert_array_equal(indices, [0, 1, 0, 0])
        np.testing.assert_array_equal(fallback, [True, False, True, False])
        self.assertTrue(np.isinf(scores[[0, 2]]).all())
        self.assertGreater(selector.seconds, 0)


class ScheduledPolicyContractTests(unittest.TestCase):
    def test_observes_post_phase_state_and_holds_every_selection_twenty_steps(self):
        selector = RecordingSelector()
        _, policy, schedules, result = scheduled_record(selector)
        arrays = result[0]
        self.assertEqual(len(selector.calls), 15)
        self.assertEqual(policy.step, 1560)
        for window, (right, left, state, goal) in enumerate(selector.calls):
            np.testing.assert_array_equal(right[:, 0], 1260 + 20 * window)
            np.testing.assert_array_equal(right[:, 1], 1)
            np.testing.assert_array_equal(state[:, 13], 0)
            np.testing.assert_array_equal(right, arrays["final_obs_r"][20 * window])
            np.testing.assert_array_equal(left, arrays["final_obs_l"][20 * window])
            np.testing.assert_array_equal(state, arrays["final_states"][20 * window])
        np.testing.assert_array_equal(arrays["states"][1260, :, 13], 7)
        expected = np.repeat(schedules.transpose(1, 0, 2), 20, axis=0)
        np.testing.assert_array_equal(arrays["residuals"][:1260], 0)
        np.testing.assert_array_equal(arrays["residuals"][1260:], expected)
        np.testing.assert_array_equal(schedules, CANDIDATES[policy.selected].transpose(1, 0, 2))
        np.testing.assert_array_equal(arrays["actions_l"], arrays["ppo_actions_l"])
        np.testing.assert_array_equal(arrays["actions_r"][:, :, 3:], arrays["ppo_actions_r"][:, :, 3:])
        # State feedback still changes PPO's own action during each held correction.
        self.assertNotEqual(float(arrays["ppo_actions_r"][1260, 0, 0]),
                            float(arrays["ppo_actions_r"][1279, 0, 0]))

    def test_zero_selector_and_no_selector_match_existing_collector_exactly(self):
        _, _, reference = record()
        for selector in (None, RecordingSelector(zero=True)):
            _, _, schedules, result = scheduled_record(selector)
            np.testing.assert_array_equal(schedules, 0)
            for key in reference[0]:
                with self.subTest(selector=type(selector).__name__, array=key):
                    np.testing.assert_array_equal(result[0][key], reference[0][key])
            for actual, expected in zip(result[1:], reference[1:]):
                np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
