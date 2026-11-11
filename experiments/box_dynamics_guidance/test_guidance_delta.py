"""Synthetic delta-selector tests; no local learned model or RaiSim required."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from dataset import FEATURE_SPEC, INPUT_DIM, OUTPUT_DIM
from delta_targets import DELTA_FEATURE_SPEC, TARGET_SPEC
from guidance import SETTINGS, Selector, candidate_scores, features
from guidance_delta import DeltaSelector
from train_model import ResponseMLP
import test_guidance as recording
from test_collect import FakeEnv


CANDIDATES = np.asarray(SETTINGS["candidates"], np.float32)


def synthetic_checkpoint():
    """A known predictor: normalized angle delta=.2*z_x, zero position delta."""
    model = ResponseMLP(hidden_dim=4, layers=1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.network[0].weight[0, 568] = 1
        model.network[0].weight[1, 568] = -1
        model.network[2].weight[0, 0] = .2
        model.network[2].weight[0, 1] = -.2
        model.network[2].bias[1] = .5
    mean = torch.zeros(OUTPUT_DIM)
    mean[1], mean[5] = .3, 1
    scale = torch.ones(OUTPUT_DIM)
    scale[0], scale[1] = 2, 3
    return {
        "schema_version": 1, "artifact_type": "box_response_delta_best_predictor",
        "config": {"target_representation": "delta_angle_position"},
        "target_spec": copy.deepcopy(TARGET_SPEC), "feature_spec": copy.deepcopy(DELTA_FEATURE_SPEC),
        "model_config": {"type": "ResponseMLP", "hidden_dim": 4, "layers": 1,
                         "input_dim": INPUT_DIM, "output_dim": OUTPUT_DIM},
        "model_state_dict": model.state_dict(),
        "normalization": {"x_mean": torch.zeros(INPUT_DIM), "x_std": torch.ones(INPUT_DIM),
                          "y_mean": mean, "y_std": scale, "absolute_y_mean": mean.clone()},
    }


class FixedPrediction(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = torch.from_numpy(values.reshape(-1, OUTPUT_DIM).copy())
        self.inputs = None

    def forward(self, value):
        self.inputs = value.detach().cpu().numpy().copy()
        return self.values


def selector_with_outputs(values, normalization=None):
    selector = DeltaSelector.__new__(DeltaSelector)
    selector.device = torch.device("cpu")
    selector.candidates = CANDIDATES.copy()
    selector.seconds = 0.
    selector.normalization = normalization or {
        "x_mean": torch.zeros(INPUT_DIM), "x_std": torch.ones(INPUT_DIM),
        "y_mean": torch.zeros(OUTPUT_DIM), "y_std": torch.ones(OUTPUT_DIM)}
    selector.model = FixedPrediction(values)
    return selector


class DeltaSelectorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="box_delta_selector_test_")
        self.root = Path(self.directory.name)
        self.checkpoint = synthetic_checkpoint()

    def tearDown(self):
        self.directory.cleanup()

    def save(self, value=None, name="delta.pt"):
        path = self.root / name
        torch.save(self.checkpoint if value is None else value, path)
        return path

    def test_loaded_model_decodes_physical_units_and_selects_expected_candidate(self):
        selector = DeltaSelector(self.save(), "cpu")
        env = FakeEnv(count=1)
        right, left = env.observe()
        goals = env.goal.copy()
        goals[:, 7] = .65  # Current .25 + .4*z_x: +x reaches this angle.
        selected, prediction, scores, fallback = selector.select(
            right, left, env.get_global_state(), goals)
        np.testing.assert_array_equal(selected, [1])
        np.testing.assert_allclose(prediction[0, :, 0], .25 + .4 * CANDIDATES[:, 0], atol=1e-7)
        np.testing.assert_allclose(prediction[:, :, 1], 1.8, atol=1e-7)
        np.testing.assert_array_equal(prediction[:, :, 2:5], 0)
        np.testing.assert_array_equal(prediction[:, :, 5], 1)
        self.assertAlmostEqual(scores[0, 1], .01, places=7)
        self.assertFalse(fallback.any())
        self.assertTrue(all(not parameter.requires_grad for parameter in selector.model.parameters()))
        self.assertFalse(selector.model.training)
        self.assertGreater(selector.seconds, 0)
        np.testing.assert_array_equal(selector.candidates, CANDIDATES)

    def test_features_denormalization_and_current_state_decode_match_contract(self):
        env = FakeEnv(count=2)
        right, left = env.observe()
        states = env.get_global_state()
        right[:, 273] = [.25, 1.125]
        states[:, 129] = [.75, 1.75]  # Distinguish the documented angle source.
        states[:, 122:125] = [[1.5, -.3, .8], [-.4, .9, 1.2]]
        goals = env.goal.copy()
        goals[1, 3:7] *= -1
        originals = [value.copy() for value in (right, left, states, goals)]
        raw = np.arange(2 * 7 * 9, dtype=np.float32).reshape(2, 7, 9) / 100
        norm = {"x_mean": torch.linspace(-.1, .1, INPUT_DIM),
                "x_std": torch.linspace(.5, 1.5, INPUT_DIM),
                "y_mean": torch.tensor([0., .25, 0., 0., 0., .9, .1, -.2, .3]),
                "y_std": torch.tensor([.2, 2., .03, .04, .05, .8, .7, .6, .5])}
        selector = selector_with_outputs(raw, norm)
        selected, prediction, scores, fallback = selector.select(right, left, states, goals)
        expected_features = features(right, left, states, goals, CANDIDATES)
        np.testing.assert_array_equal(selector.model.inputs,
            ((expected_features.reshape(-1, INPUT_DIM) - norm["x_mean"].numpy()) / norm["x_std"].numpy()))
        expected = raw * norm["y_std"].numpy() + norm["y_mean"].numpy()
        expected[:, :, 0] += right[:, None, 273]
        expected[:, :, 2:5] += states[:, None, 122:125]
        np.testing.assert_array_equal(prediction, expected)
        expected_scores = candidate_scores(expected, goals, CANDIDATES)
        np.testing.assert_array_equal(scores, expected_scores)
        np.testing.assert_array_equal(selected, np.argmin(expected_scores, axis=1))
        self.assertFalse(fallback.any())
        for source, unchanged in zip((right, left, states, goals), originals):
            np.testing.assert_array_equal(source, unchanged)

    def test_each_call_restores_from_current_state_not_initial_phase_state(self):
        selector = DeltaSelector(self.save(), "cpu")
        env = FakeEnv(count=1)
        right, left = env.observe()
        states = env.get_global_state()
        first = selector.select(right, left, states, env.goal)[1]
        right[:, 273] += .5
        states[:, 122:125] += np.array([1., 2., 3.], np.float32)
        second = selector.select(right, left, states, env.goal)[1]
        np.testing.assert_allclose(second[:, :, 0] - first[:, :, 0], .5, atol=1e-7)
        np.testing.assert_array_equal(second[:, :, 2:5] - first[:, :, 2:5],
                                      np.broadcast_to([1., 2., 3.], (1, 7, 3)))
        np.testing.assert_array_equal(second[:, :, [1, 5, 6, 7, 8]], first[:, :, [1, 5, 6, 7, 8]])

    def test_invalid_candidates_and_inputs_fall_back_without_poisoning_other_rows(self):
        env = FakeEnv(count=5)
        right, left = env.observe()
        raw = np.zeros((5, 7, 9), np.float32)
        raw[:, :, 5] = 1
        raw[:, :, 0] = env.goal[:, None, 7] - right[:, None, 273]
        raw[0, :, 5:9] = 0
        raw[1, 0, 0] = np.nan
        right[2, 40] = np.nan
        right[4, 273] = np.nan
        selector = selector_with_outputs(raw)
        indices, _, scores, fallback = selector.select(right, left, env.get_global_state(), env.goal)
        np.testing.assert_array_equal(indices, [0, 1, 0, 0, 0])
        np.testing.assert_array_equal(fallback, [True, False, True, False, True])
        self.assertTrue(np.isinf(scores[[0, 2, 4]]).all())
        self.assertTrue(np.isfinite(scores[1, 1:]).all())
        self.assertTrue(np.isfinite(selector.model.inputs).all())

    def test_checkpoint_kind_specs_and_dimensions_are_validated(self):
        changes = (
            ("artifact_type", "box_response_best_predictor"),
            ("artifact_type", "box_response_delta_training_resume"),
            ("schema_version", 2),
            ("feature_spec", FEATURE_SPEC),
            ("target_spec", dict(TARGET_SPEC, delta_indices=[0])),
            ("config", {"target_representation": "absolute"}),
            ("model_config", dict(self.checkpoint["model_config"], input_dim=570)),
        )
        for key, value in changes:
            with self.subTest(key=key, value=value):
                bad = copy.deepcopy(self.checkpoint)
                bad[key] = value
                with self.assertRaises(ValueError):
                    DeltaSelector(self.save(bad), "cpu")
        with self.assertRaisesRegex(ValueError, "matching trained box-response"):
            Selector(self.save(), "cpu")

    def test_invalid_scales_centers_and_weights_are_rejected(self):
        changes = (
            ("x_mean", torch.zeros(INPUT_DIM - 1)),
            ("x_std", torch.zeros(INPUT_DIM)),
            ("y_std", torch.full((OUTPUT_DIM,), np.nan)),
            ("y_mean", torch.ones(OUTPUT_DIM)),
            ("absolute_y_mean", torch.zeros(OUTPUT_DIM)),
        )
        for key, value in changes:
            with self.subTest(key=key):
                bad = copy.deepcopy(self.checkpoint)
                bad["normalization"][key] = value
                with self.assertRaises(ValueError):
                    DeltaSelector(self.save(bad), "cpu")
        bad = copy.deepcopy(self.checkpoint)
        bad["model_state_dict"]["network.0.weight"][0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite delta predictor weights"):
            DeltaSelector(self.save(bad), "cpu")

    def test_scheduled_policy_uses_fifteen_selections_and_holds_each_twenty_steps(self):
        selector = DeltaSelector(self.save(), "cpu")
        snapshots = []
        original_select = selector.select

        def capture(*args):
            snapshots.append(tuple(value.copy() for value in args))
            return original_select(*args)

        with patch.object(selector, "select", side_effect=capture):
            _, policy, schedules, result = recording.scheduled_record(selector)
        arrays = result[0]
        self.assertEqual(len(snapshots), 15)
        self.assertEqual(policy.step, 1560)
        for window, (right, left, states, goals) in enumerate(snapshots):
            np.testing.assert_array_equal(right, arrays["final_obs_r"][window * 20])
            np.testing.assert_array_equal(states, arrays["final_states"][window * 20])
            np.testing.assert_array_equal(right[:, 0], 1260 + window * 20)
            np.testing.assert_array_equal(states[:, 13], 0)
        np.testing.assert_array_equal(arrays["residuals"][:1260], 0)
        np.testing.assert_array_equal(arrays["residuals"][1260:],
                                      np.repeat(schedules.transpose(1, 0, 2), 20, axis=0))
        np.testing.assert_array_equal(schedules, CANDIDATES[policy.selected].transpose(1, 0, 2))
        np.testing.assert_array_equal(arrays["actions_l"], arrays["ppo_actions_l"])
        np.testing.assert_array_equal(arrays["actions_r"][:, :, 3:], arrays["ppo_actions_r"][:, :, 3:])
        self.assertEqual(policy.predictions.shape, (15, 2, 7, 9))
        self.assertTrue(np.isfinite(policy.predictions).all())


if __name__ == "__main__":
    unittest.main()
