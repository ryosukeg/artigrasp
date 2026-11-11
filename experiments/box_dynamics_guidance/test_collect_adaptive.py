import unittest
import numpy as np
from collect_adaptive import simulate_adaptive, episode_arrays
from collect import START


class FakeEnv:
    num_obs_r, num_obs_l, num_acts, num_gs = 276, 276, 51, 138

    def __init__(self):
        self.step_index = 0
        self.state = np.zeros((2, 138), np.float32)
        self.state[:, 125] = 1.
        self.state[:, 133] = 1.
        self.state[1, 137] = 4.

    def get_global_state(self):
        return self.state.copy()

    def observe(self, update_mean=False):
        obs = np.zeros((2, 276), np.float32)
        obs[:, 273] = self.state[:, 129]
        obs[:, 275] = self.state[:, 137] - self.state[:, 129]
        return obs.copy(), obs.copy()

    def step(self, right, left):
        self.step_index += 1
        self.state[1, 129] = self.step_index * .001
        if self.step_index > 1560:
            self.state[0, 20] = np.nan
        return None, None, np.zeros(2, bool)


class FakePolicy:
    articulation = False

    def actions(self, right, left):
        return np.zeros((2, 51), np.float32), np.zeros((2, 51), np.float32)


class FakeController:
    count = 2
    modes_r, modes_l = np.full(2, 8), np.full(2, 7)
    goal = np.array([[0, 0, 0, 1, 0, 0, 0]] * 2, np.float32)
    angle = np.array([[0.], [4.]], np.float32)

    def reset(self):
        pass

    def phase(self, step):
        pass


class CollectorTest(unittest.TestCase):
    def test_per_slot_stop_latches_while_batch_continues(self):
        result = simulate_adaptive(FakeEnv(), FakeController(), FakePolicy(), np.zeros((2, 50, 3), np.float32))
        arrays, goals, initial, first, monitor, batch_end = result
        self.assertEqual(monitor.reasons, ['success_hold', 'timeout'])
        self.assertEqual(monitor.stop_step.tolist(), [300, 1000])
        self.assertEqual(batch_end, 2260)
        self.assertGreaterEqual(first[0], 1560)
        saved = episode_arrays(arrays, 0, 300)
        self.assertEqual(len(saved['states']), 1561)
        self.assertEqual(len(saved['actions_r']), 1560)
        self.assertEqual(len(saved['final_states']), 301)
        self.assertTrue(np.isfinite(saved['states']).all())
        self.assertFalse(saved['bad_state'].any())
        self.assertTrue(np.isfinite(monitor.metric_history[-1][0]).all())
        saved = episode_arrays(arrays, 1, 1000)
        self.assertEqual(len(saved['final_states']), 1001)
        self.assertEqual(len(saved['states']), START + 1001)


if __name__ == '__main__':
    unittest.main()
