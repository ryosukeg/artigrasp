"""Protocol regression tests with a fake environment, without RaiSim or Torch."""

import math
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ood_suite.runtime import Controller, simulate, timing_for, total_steps


def _fake_rotations():
    rotations = types.ModuleType("raisimGymTorch.helper.rotations")

    def quat2euler(quaternions):
        result = np.zeros((len(quaternions), 3))
        result[:, 1] = 2 * np.arctan2(quaternions[:, 2], quaternions[:, 0])
        return result

    def euler2quat(eulers):
        result = np.zeros((len(eulers), 4))
        result[:, 0] = np.cos(eulers[:, 1] / 2)
        result[:, 2] = np.sin(eulers[:, 1] / 2)
        return result

    rotations.quat2euler = quat2euler
    rotations.euler2quat = euler2quat
    package = types.ModuleType("raisimGymTorch")
    helper = types.ModuleType("raisimGymTorch.helper")
    helper.rotations = rotations
    package.helper = helper
    return {"raisimGymTorch": package, "raisimGymTorch.helper": helper,
            "raisimGymTorch.helper.rotations": rotations}


def _data(count=1):
    dimensions = [3, 3, 8, 51, 51, 1, 8, 8, 63, 63, 48, 48, 51, 51, 16, 16, 3]
    grasp = [np.full((count, size), i / 10, dtype=np.float32)
             for i, size in enumerate(dimensions)]
    articulation = [np.full((count, size), 10 + i / 10, dtype=np.float32)
                    for i, size in enumerate(dimensions)]
    for values in (grasp, articulation):
        values[2][:] = [0.75, 0.1, 0.6, 1, 0, 0, 0, 0]
    return {"grasp": grasp, "articulation": articulation,
            "left": np.resize(np.array([7, 9], dtype=np.float32), count),
            "right": np.resize(np.array([9, 7], dtype=np.float32), count),
            "arti_left": np.full(count, 7, dtype=np.float32),
            "arti_right": np.full(count, 8, dtype=np.float32)}


def _specs(object_name="box", count=1):
    return [{"episode_id": "fake{}".format(i), "object": object_name,
             "target": {"translation_m": [0.5, -0.6], "lift_m": 0.7,
                        "orientation_delta_rad": 0.2, "articulation_rad": math.pi,
                        "articulation_rule": "absolute"}}
            for i in range(count)]


class EventEnvironment:
    def __init__(self):
        self.step_number = -1
        self.events = []

    def __getattr__(self, name):
        def record(*args):
            saved = [arg.copy() if isinstance(arg, np.ndarray) else arg for arg in args]
            self.events.append((self.step_number, name, saved))
        return record


class RuntimeProtocolTests(unittest.TestCase):
    def controller(self, name="box", count=1):
        env, data = EventEnvironment(), _data(count)
        policy = types.SimpleNamespace(articulation=False)
        with patch.dict(sys.modules, _fake_rotations()):
            controller = Controller(env, data, _specs(name, count), timing_for(name), policy)
        return controller, env, data, policy

    def advance(self, controller, env):
        for step in range(total_steps(controller.timing)):
            env.step_number = step
            controller.phase(step)

    def test_standard_event_boundaries(self):
        controller, env, _, policy = self.controller(count=2)
        self.advance(controller, env)
        self.assertEqual(total_steps(controller.timing), 1560)
        self.assertEqual(sorted(set(event[0] for event in env.events)),
                         [0, 260, 560, 860, 935, 1010, 1085, 1260])
        self.assertFalse(policy.articulation)
        switches = {step: args for step, name, args in env.events if name == "control_switch_all"}
        np.testing.assert_array_equal(switches[0][0], [11, 9])
        np.testing.assert_array_equal(switches[0][1], [9, 11])
        np.testing.assert_array_equal(switches[260][0], [11, 1])
        np.testing.assert_array_equal(switches[260][1], [1, 11])
        for step in (860, 935, 1010):
            np.testing.assert_array_equal(switches[step][0], [13, 13])
            np.testing.assert_array_equal(switches[step][1], [13, 13])
        np.testing.assert_array_equal(switches[1085][0], [7, 7])
        np.testing.assert_array_equal(switches[1085][1], [11, 11])
        np.testing.assert_array_equal(switches[1260][1], [8, 8])

    def test_long_transit_and_pretraining_policy_switch(self):
        for name in ("notebook", "microwave", "waffleiron"):
            with self.subTest(object=name):
                controller, env, _, policy = self.controller(name)
                self.advance(controller, env)
                self.assertEqual(total_steps(controller.timing), 2160)
                self.assertEqual(sorted(set(event[0] for event in env.events)),
                                 [0, 260, 760, 1260, 1385, 1510, 1635, 1860])
                self.assertEqual(policy.articulation, name == "notebook")
        controller, env, _, policy = self.controller("mixer")
        self.advance(controller, env)
        self.assertTrue(policy.articulation)
        left_release = [args[0] for step, name, args in env.events
                        if step == 860 and name == "control_switch_all"][0]
        np.testing.assert_array_equal(left_release, [7])

    def test_ketchup_has_its_original_short_schedule(self):
        controller, env, _, policy = self.controller("ketchup")
        self.assertTrue(policy.articulation)
        self.advance(controller, env)
        self.assertEqual(total_steps(controller.timing), 1600)
        self.assertEqual(sorted(set(event[0] for event in env.events)),
                         [0, 300, 800, 1300, 1400])
        switches = {step: tuple(int(a[0]) for a in args)
                    for step, name, args in env.events if name == "control_switch_all"}
        self.assertEqual(switches, {0: (9, 7), 300: (1, 7), 800: (1, 7),
                                   1300: (12, 11), 1400: (12, 8)})

    def test_ketchup_requested_angle_conditions_actor(self):
        controller, env, data, _ = self.controller("ketchup")
        self.advance(controller, env)
        initial = [args[0] for step, name, args in env.events if step == 0 and name == "set_goals"][0]
        np.testing.assert_array_equal(initial, [[0]])
        preparation = [(name, args) for step, name, args in env.events if step == 1300]
        calls = [name for name, _ in preparation]
        self.assertIn("set_goals", calls)
        self.assertLess(calls.index("set_goals"), calls.index("set_obj_goal"))
        args = [args for name, args in preparation if name == "set_goals"][0]
        np.testing.assert_allclose(args[0], [[math.pi]], atol=1e-6)
        expected_indices = [6, 8, 9, 10, 11, 12, 13, 14, 15]
        for received, index in zip(args[1:], expected_indices):
            np.testing.assert_array_equal(received, data["grasp"][index])

    def test_goals_and_transition_imitation_offsets(self):
        controller, env, data, _ = self.controller()
        self.advance(controller, env)
        object_goals = {step: args for step, name, args in env.events if name == "set_obj_goal"}
        np.testing.assert_allclose(object_goals[0][1][0, :3], [1.25, -0.5, 1.3], atol=1e-6)
        np.testing.assert_allclose(object_goals[560][1][0, :3], [1.25, -0.5, 0.6], atol=1e-6)
        np.testing.assert_allclose(object_goals[860][0], [[math.pi]], atol=1e-6)
        np.testing.assert_allclose(object_goals[860][1][0, :3], [1.25, -0.5, 0.6], atol=1e-6)
        imitation = {step: args for step, name, args in env.events if name == "set_imitation_goals"}
        np.testing.assert_array_equal(imitation[860][0], np.zeros((1, 51)))
        np.testing.assert_array_equal(imitation[935][0][0, :4], [0.5, 0, 0, 0])
        np.testing.assert_array_equal(imitation[935][1][0, :4], [-0.5, 0, 0, 0])
        np.testing.assert_allclose(imitation[1010][0][0, :4], [0, 0.3, 0.3, 0])
        np.testing.assert_array_equal(imitation[1085][0], data["articulation"][4])
        right_goals = {step: args for step, name, args in env.events if name == "set_goals_r"}
        np.testing.assert_array_equal(right_goals[0][0], data["grasp"][6])
        np.testing.assert_array_equal(right_goals[860][0], data["articulation"][6])
        metric = controller.metric_spec(0)
        np.testing.assert_allclose(metric["target_position_m"], [1.25, -0.5, 0.6], atol=1e-6)
        self.assertAlmostEqual(metric["target_articulation_rad"], math.pi, places=6)


class FakeSimulationEnvironment:
    def __init__(self, count=2, failures=True):
        self.count = count
        self.step_number = 0
        self.failures = failures
        self.received_actions = []

    def get_global_state(self):
        return np.full((self.count, 138), self.step_number, dtype=np.float32)

    def observe(self, update_mean=False):
        right = np.zeros((self.count, 3), dtype=np.float32)
        if self.failures and self.step_number == 1:
            right[1, 0] = np.nan
        return right, np.zeros_like(right)

    def step(self, right, left):
        self.received_actions.append((right.copy(), left.copy()))
        done = np.zeros(self.count, dtype=bool)
        if self.failures and self.step_number == 0:
            done[0] = True
        self.step_number += 1
        return np.zeros(self.count), np.zeros(self.count), done


class FakeSimulationPolicy:
    def __init__(self, failures=True):
        self.step_number = 0
        self.failures = failures

    def actions(self, right, left):
        if not np.isfinite(right).all() or not np.isfinite(left).all():
            raise AssertionError("Nonfinite observations reached policy")
        action = np.zeros((len(right), 51), dtype=np.float32)
        if self.failures and self.step_number == 2:
            action[0, 0] = np.inf
        self.step_number += 1
        return action, np.zeros_like(action)


class SimulationTests(unittest.TestCase):
    def controller(self, env):
        return types.SimpleNamespace(count=env.count, reset=lambda: None, phase=lambda step: None,
                                     timing={"protocol": "ketchup", "grasp_steps": 1,
                                             "transit_steps": 1, "arti_steps": 1})

    def test_numerical_failures_and_auto_reset_latch(self):
        env = FakeSimulationEnvironment()
        states, right, left, invalid, first = simulate(
            env, self.controller(env), FakeSimulationPolicy())
        np.testing.assert_array_equal(invalid, [True, True])
        np.testing.assert_array_equal(first, [0, 1])
        self.assertEqual(states.shape, (5, 2, 138))
        self.assertEqual(right.shape, (4, 2, 51))
        self.assertEqual(left.shape, right.shape)
        for action_r, action_l in env.received_actions:
            self.assertTrue(np.isfinite(action_r).all())
            self.assertTrue(np.isfinite(action_l).all())

    def test_replay_uses_saved_actions_exactly(self):
        env = FakeSimulationEnvironment(count=1, failures=False)
        right = np.arange(4 * 51, dtype=np.float32).reshape(4, 1, 51) / 100
        left = -right
        _, actual_r, actual_l, invalid, first = simulate(
            env, self.controller(env), replay_actions=(right, left))
        np.testing.assert_array_equal(actual_r, right)
        np.testing.assert_array_equal(actual_l, left)
        np.testing.assert_array_equal(invalid, [False])
        np.testing.assert_array_equal(first, [-1])

    def test_profiling_separates_wall_time_and_preserves_the_trajectory(self):
        clock = [0.0]

        class TimedEnvironment(FakeSimulationEnvironment):
            def get_global_state(self):
                clock[0] += 1.0
                return super().get_global_state()

            def step(self, right, left):
                clock[0] += 3.0
                return super().step(right, left)

        class TimedPolicy(FakeSimulationPolicy):
            def actions(self, right, left):
                clock[0] += 2.0
                return super().actions(right, left)

        env = TimedEnvironment(failures=False)
        timings = {}
        with patch("ood_suite.runtime.time.perf_counter", side_effect=lambda: clock[0]):
            measured = simulate(env, self.controller(env), TimedPolicy(failures=False), timings=timings)
        self.assertEqual(timings["policy_and_transfer_s"], 8.0)
        self.assertEqual(timings["physics_and_state_s"], 16.0)
        self.assertEqual(timings["wall_s"], 25.0)
        self.assertEqual(timings["other_s"], 1.0)
        env = FakeSimulationEnvironment(failures=False)
        original = simulate(env, self.controller(env), FakeSimulationPolicy(failures=False))
        for before, after in zip(original, measured):
            np.testing.assert_array_equal(before, after)


if __name__ == "__main__":
    unittest.main()
