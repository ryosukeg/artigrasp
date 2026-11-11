"""Collector contract tests; no simulator, checkpoints, or planning module needed.

Run with the ArtiGrasp Python: ``python -m unittest discover -s
experiments/box_dynamics_guidance -p test_collect.py -v``.
The fake dynamics exercise recording and indexing, not physical correctness.
"""
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np


_spec = importlib.util.spec_from_file_location(
    "box_collector_under_test", Path(__file__).with_name("collect.py"))
collect = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(collect)


class FakeEnv:
    num_obs_r = num_obs_l = 276
    num_acts, num_gs = 51, 138

    def __init__(self, count=2, bad_observation=None, done_at=None):
        self.count = count
        self.bad_observation = bad_observation
        self.done_at = done_at
        self.goal = np.zeros((count, 8), np.float32)
        self.goal[:, 3] = 1
        self.goal[:, 7] = np.linspace(.6, 1.3, count)
        self.reset()

    def reset(self):
        self.step_index = 0
        self.after_phase = False
        self.received = []
        self.state = np.zeros((self.count, 138), np.float32)
        self.state[:, 9] = self.state[:, 125] = 1
        self.state[:, 13] = 7  # A local root coordinate reset at phase 1260.
        self.state[:, 129] = .25
        self.state[:, 130:138] = self.goal

    def phase_reset(self):
        self.after_phase = True
        self.state[:, 13] = 0
        self.state[:, 9:13] = [np.cos(.1), 0, 0, np.sin(.1)]

    def get_global_state(self):
        return self.state.copy()

    def observe(self, update_mean=False):
        if update_mean:
            raise AssertionError("Collection must not update normalization")
        right = np.zeros((self.count, 276), np.float32)
        right[:, 0] = self.step_index
        right[:, 1] = self.after_phase
        right[:, 2] = self.state[:, 122]
        right[:, 273] = self.state[:, 129]
        right[:, 275] = self.goal[:, 7] - right[:, 273]
        left = right.copy()
        if self.bad_observation == self.step_index:
            right[0, 50] = np.nan
        return right, left

    def step(self, right, left):
        if not (np.isfinite(right).all() and np.isfinite(left).all()):
            raise AssertionError("The simulator must receive sanitized actions")
        self.received.append((right.copy(), left.copy()))
        done = np.zeros(self.count, bool)
        if self.done_at == self.step_index:
            done[0] = True
            self.state[0, 122] = 0  # Simulate an automatic reset.
        self.state[:, 122] += right[:, 0] * .001
        self.state[:, 129] += right[:, 1] * .0005
        self.step_index += 1
        self.state[:, 0] = self.step_index
        return np.zeros(self.count), np.zeros(self.count), done


class FakeController:
    def __init__(self, env):
        self.env, self.count = env, env.count
        self.goal = env.goal.copy()
        self.angle = env.goal[:, 7:8].copy()
        self.timing = collect.timing_for("box")

    def reset(self):
        self.env.reset()
        self.phases = []
        self.modes_r = np.full(self.count, 9, np.int32)
        self.modes_l = np.full(self.count, 9, np.int32)

    def phase(self, step):
        self.phases.append(step)
        if step == 1260:
            self.env.phase_reset()
            self.modes_r[:] = 8
            self.modes_l[:] = 7


class FakePolicy:
    articulation = False

    def __init__(self, bad_action=None):
        self.bad_action = bad_action

    def actions(self, right, left):
        if not (np.isfinite(right).all() and np.isfinite(left).all()):
            raise AssertionError("Policy inputs must be sanitized")
        # State-dependent output ensures the policy is called in closed loop.
        base = np.arange(51, dtype=np.float32)[None, :] * .001
        action_r = (base + right[:, 2:3] * .01 + .1).astype(np.float32)
        action_l = (base + left[:, 2:3] * .02 - .1).astype(np.float32)
        if self.bad_action == int(right[0, 0]):
            action_r[0, 2] = np.inf
        return action_r, action_l


def record(schedules=None, **env_options):
    env = FakeEnv(**env_options)
    controller, policy = FakeController(env), FakePolicy()
    if schedules is None:
        schedules = np.zeros((env.count, 15, 3), np.float32)
    with contextlib.redirect_stdout(io.StringIO()):
        result = collect.simulate_recorded(env, controller, policy, schedules)
    return env, controller, result


class RecordingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.z = np.zeros((2, 15, 3), np.float32)
        cls.z[0, :, 0] = np.arange(15, dtype=np.float32) / 10
        cls.z[1, :, 1] = -.5
        cls.env, cls.controller, cls.result = record(cls.z)
        cls.arrays = cls.result[0]

    def test_phase_state_is_refreshed_and_last_endpoint_is_recorded(self):
        a = self.arrays
        self.assertEqual(self.controller.phases, list(range(1560)))
        self.assertEqual(len(self.env.received), 1560)
        self.assertEqual(a["states"].shape, (1561, 2, 138))
        self.assertEqual(a["final_obs_r"].shape, (301, 2, 276))
        self.assertEqual(a["final_states"].shape, (301, 2, 138))
        np.testing.assert_array_equal(a["states"][1260, :, 13], 7)
        np.testing.assert_array_equal(a["final_states"][0, :, 13], 0)
        np.testing.assert_array_equal(a["final_obs_r"][0, :, 1], 1)
        np.testing.assert_array_equal(a["final_obs_r"][:, 0, 0], np.arange(1260, 1561))
        np.testing.assert_array_equal(a["final_states"][-1], a["states"][-1])
        self.assertFalse(self.result[4].any())
        np.testing.assert_array_equal(self.result[5], -1)

    def test_each_residual_lasts_twenty_steps_and_changes_only_right_translation(self):
        a = self.arrays
        expected = np.repeat(self.z.transpose(1, 0, 2), 20, axis=0)
        np.testing.assert_array_equal(a["residuals"][:1260], 0)
        np.testing.assert_array_equal(a["residuals"][1260:], expected)
        np.testing.assert_allclose(a["actions_r"][1260:, :, :3] -
                                   a["ppo_actions_r"][1260:, :, :3], expected, atol=1e-7)
        np.testing.assert_array_equal(a["actions_r"][:1260], a["ppo_actions_r"][:1260])
        np.testing.assert_array_equal(a["actions_r"][:, :, 3:], a["ppo_actions_r"][:, :, 3:])
        np.testing.assert_array_equal(a["actions_l"], a["ppo_actions_l"])
        np.testing.assert_array_equal(a["actions_r"], np.stack([x[0] for x in self.env.received]))
        np.testing.assert_array_equal(a["modes_r"][1260:], 8)
        np.testing.assert_array_equal(a["modes_l"][1260:], 7)

    def test_zero_residual_matches_an_independent_plain_policy_rollout(self):
        _, _, result = record()
        a = result[0]
        env, policy = FakeEnv(), FakePolicy()
        controller = FakeController(env)
        controller.reset()
        states = [env.get_global_state()]
        for step in range(1560):
            controller.phase(step)
            right, left = policy.actions(*env.observe())
            env.step(right, left)
            states.append(env.get_global_state())
        np.testing.assert_array_equal(a["states"], np.stack(states))
        np.testing.assert_array_equal(a["actions_r"], a["ppo_actions_r"])
        np.testing.assert_array_equal(a["actions_l"], a["ppo_actions_l"])
        # Perturbations cannot alter the preceding shared approach trajectory.
        np.testing.assert_array_equal(a["states"][:1261], self.arrays["states"][:1261])

    def test_raw_bad_observation_is_preserved_and_invalidity_is_latched(self):
        _, _, result = record(bad_observation=1280)
        a, _, initial, terminal, invalid, first = result
        self.assertTrue(np.isnan(a["final_obs_r"][20, 0, 50]))
        self.assertTrue(a["bad_observation"][1280, 0])
        np.testing.assert_array_equal(invalid, [True, False])
        np.testing.assert_array_equal(first, [1280, -1])
        # The first window also uses the corrupted state/observation at its endpoint.
        self.assertFalse(collect.window_validity(a, 0, initial[0], terminal[0]).any())
        self.assertTrue(collect.window_validity(a, 1, initial[1], terminal[1]).all())

    def test_nonfinite_action_is_recorded_raw_but_sanitized_before_execution(self):
        env = FakeEnv()
        with contextlib.redirect_stdout(io.StringIO()):
            result = collect.simulate_recorded(
                env, FakeController(env), FakePolicy(bad_action=1280), self.z)
        a = result[0]
        self.assertTrue(np.isinf(a["ppo_actions_r"][1280, 0, 2]))
        self.assertEqual(a["actions_r"][1280, 0, 2], 0)
        self.assertTrue(a["bad_action"][1280, 0])
        self.assertEqual(result[5][0], 1280)

    def test_terminal_observation_is_checked_without_an_extra_action(self):
        env, _, result = record(bad_observation=1560)
        a, _, initial, terminal, invalid, first = result
        self.assertEqual(len(env.received), 1560)
        np.testing.assert_array_equal(terminal, [True, False])
        np.testing.assert_array_equal(first, [1560, -1])
        valid = collect.window_validity(a, 0, initial[0], terminal[0])
        np.testing.assert_array_equal(valid, [True] * 14 + [False])

    def test_auto_reset_invalidates_current_and_all_later_windows(self):
        _, _, result = record(done_at=1280)
        a, _, initial, terminal, invalid, first = result
        self.assertTrue(a["auto_reset"][1280, 0])
        np.testing.assert_array_equal(first, [1280, -1])
        valid = collect.window_validity(a, 0, initial[0], terminal[0])
        np.testing.assert_array_equal(valid, [True] + [False] * 14)

    def test_mismatched_layout_and_nonfinite_schedule_are_rejected(self):
        env = FakeEnv()
        env.num_gs = 137
        with self.assertRaisesRegex(ValueError, "dimensions"):
            collect.simulate_recorded(env, FakeController(env), FakePolicy(), self.z)
        env.num_gs = 138
        bad = self.z.copy()
        bad[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite residual"):
            collect.simulate_recorded(env, FakeController(env), FakePolicy(), bad)
        states = env.get_global_state()
        right, _ = env.observe()
        right[0, 273] += .1
        with self.assertRaises(AssertionError):
            collect.check_observation_layout(right, states, env.goal, np.ones(2, bool))


class WindowValidityTests(unittest.TestCase):
    def setUp(self):
        self.a = {key: np.zeros((1560, 2), bool) for key in
                  ("bad_observation", "bad_action", "bad_state", "auto_reset")}
        for key, size in (("final_obs_r", 276), ("final_obs_l", 276), ("final_states", 138)):
            self.a[key] = np.zeros((301, 2, size), np.float32)

    def valid(self, initial=False, terminal=False):
        return collect.window_validity(self.a, 0, initial, terminal)

    def test_clean_and_initial_invalid_episodes(self):
        self.assertTrue(self.valid().all())
        self.assertFalse(self.valid(initial=True).any())

    def test_every_failure_kind_in_prefix_invalidates_all_windows(self):
        for key in ("bad_observation", "bad_action", "bad_state", "auto_reset"):
            with self.subTest(key=key):
                self.a[key][1259, 0] = True
                self.assertFalse(self.valid().any())
                self.assertTrue(collect.window_validity(self.a, 1, False, False).all())
                self.a[key][1259, 0] = False

    def test_last_action_of_window_included_next_action_excluded(self):
        self.a["bad_action"][1279, 0] = True
        self.assertFalse(self.valid().any())
        self.a["bad_action"][1279, 0] = False
        self.a["bad_action"][1280, 0] = True
        np.testing.assert_array_equal(self.valid(), [True] + [False] * 14)

    def test_observation_and_state_endpoints_must_be_finite(self):
        for key in ("final_obs_r", "final_obs_l", "final_states"):
            with self.subTest(key=key):
                self.a[key][20, 0, 0] = np.nan
                np.testing.assert_array_equal(self.valid(), [False, False] + [True] * 13)
                self.a[key][20, 0, 0] = 0
        np.testing.assert_array_equal(self.valid(terminal=True), [True] * 14 + [False])


class ArtifactTests(unittest.TestCase):
    def test_committed_batch_checks_plan_ids_and_trajectory_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            collect.write_json(output / "plan.json", {"episodes": [{"episode_id": "example"}]})
            path = output / "trajectories/example.npz"
            collect.atomic_npz(path, {"window_valid": np.ones(15, bool), "goal": np.arange(8)})
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["goal"], np.arange(8))
            self.assertFalse(path.with_suffix(".npz.tmp").exists())
            row = {"episode_id": "example", "trajectory_path": "trajectories/example.npz",
                   "trajectory_sha256": collect.file_sha256(path)}
            record = {"complete": True, "plan_sha256": collect.file_sha256(output / "plan.json"),
                      "episodes": [row]}
            collect.write_json(collect.batch_path(output, 0), record)
            self.assertEqual(collect.check_batch(output, 0, [row]), record)
            with self.assertRaisesRegex(ValueError, "IDs disagree"):
                collect.check_batch(output, 0, [{"episode_id": "other"}])
            path.write_bytes(b"interrupted or corrupted file")
            with self.assertRaisesRegex(ValueError, "Trajectory changed"):
                collect.check_batch(output, 0, [row])
            collect.write_json(output / "plan.json", {"episodes": []})
            with self.assertRaisesRegex(ValueError, "mismatched batch"):
                collect.check_batch(output, 0, [row])

    def test_resume_refuses_changed_plan_or_runtime_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            plan = {"episodes": []}
            collect.write_json(output / "plan.json", plan)
            collect.write_json(output / "provenance.json", {
                "source_sha256": {"collect.py": "original"},
                "plan_sha256": collect.file_sha256(output / "plan.json")})
            with mock.patch.object(collect, "provenance", return_value={
                    "source_sha256": {"collect.py": "original"}}):
                self.assertEqual(collect.check_provenance(output), plan)
            with mock.patch.object(collect, "provenance", return_value={
                    "source_sha256": {"collect.py": "changed"}}):
                with self.assertRaisesRegex(ValueError, "Provenance changed"):
                    collect.check_provenance(output)
            collect.write_json(output / "plan.json", {"episodes": ["changed"]})
            with mock.patch.object(collect, "provenance") as fingerprint:
                with self.assertRaisesRegex(ValueError, "Plan changed"):
                    collect.check_provenance(output)
                fingerprint.assert_not_called()


if __name__ == "__main__":
    unittest.main()
