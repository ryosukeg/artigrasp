"""Training reproducibility and transfer checks without a physics simulator."""

import importlib
from pathlib import Path
import random
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "raisimGymTorch"))

try:
    import numpy as np
    import torch
except ImportError:
    np = torch = None


@unittest.skipIf(torch is None, "Requires the ArtiGrasp NumPy/PyTorch environment")
class SeedAndSamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.support = importlib.import_module("raisimGymTorch.helper.training_support")

    def setUp(self):
        self.original_threads = torch.get_num_threads()

    def tearDown(self):
        torch.set_num_threads(self.original_threads)

    def draws(self, seed, stage):
        streams = self.support.seed_training(seed, stage)
        return streams, (random.random(), np.random.rand(), torch.rand(4))

    def test_same_training_seed_repeats_python_numpy_and_model_rng(self):
        first_seeds, first = self.draws(13, "general_two")
        second_seeds, second = self.draws(13, "general_two")
        self.assertEqual(first_seeds, second_seeds)
        self.assertEqual(first[:2], second[:2])
        self.assertTrue(torch.equal(first[2], second[2]))

    def test_other_training_seeds_and_stages_change_random_streams(self):
        reference_seeds, reference = self.draws(13, "general_two")
        for seed, stage in ((14, "general_two"), (13, "left_fixed")):
            seeds, actual = self.draws(seed, stage)
            self.assertNotEqual(reference_seeds["environment"], seeds["environment"])
            self.assertNotEqual(reference[:2], actual[:2])
            self.assertFalse(torch.equal(reference[2], actual[2]))

    def test_environment_seed_fits_the_native_signed_cpp_integer(self):
        for master in (0, 1, 2, 17, 2 ** 32 - 1):
            for stage in ("left_fixed", "multi_obj_arti", "general_two"):
                with self.subTest(master=master, stage=stage):
                    seeds = self.support.seed_training(master, stage)
                    self.assertGreaterEqual(seeds["environment"], 0)
                    self.assertLess(seeds["environment"], 2 ** 31)
                    for component in ("python", "numpy", "torch", "action_left", "action_right"):
                        self.assertGreaterEqual(seeds[component], 0)
                        self.assertLess(seeds[component], 2 ** 32)

    def sample(self, sampler, mean=None, std=None):
        mean = np.zeros((5, 3), np.float32) if mean is None else mean
        std = np.ones(3, np.float32) if std is None else std
        actions = np.empty_like(mean)
        log_prob = np.empty(len(mean), np.float32)
        sampler.sample(mean, std, actions, log_prob)
        return actions, log_prob

    def test_left_and_right_sampling_are_independent_reproducible_streams(self):
        seeds = self.support.seed_training(13, "general_two")
        self.assertNotEqual(seeds["action_left"], seeds["action_right"])
        left, right, reference = [self.support.NumpyNormalSampler(3) for _ in range(3)]
        left.seed(seeds["action_left"])
        right.seed(seeds["action_right"])
        reference.seed(seeds["action_left"])
        left_first, _ = self.sample(left)
        right_first, _ = self.sample(right)
        self.assertFalse(np.array_equal(left_first, right_first))
        np.testing.assert_array_equal(left_first, self.sample(reference)[0])
        right.seed(987)  # Seeding the other hand must not reset this hand.
        self.sample(right)
        np.testing.assert_array_equal(self.sample(left)[0], self.sample(reference)[0])

    def test_sample_log_probability_matches_diagonal_gaussian(self):
        sampler = self.support.NumpyNormalSampler(3)
        sampler.seed(7)
        mean = np.arange(15, dtype=np.float32).reshape(5, 3) / 7
        std = np.array([0.2, 0.7, 1.3], np.float32)
        actions, log_prob = self.sample(sampler, mean, std)
        expected = torch.distributions.Normal(torch.from_numpy(mean), torch.from_numpy(std))
        expected = expected.log_prob(torch.from_numpy(actions)).sum(dim=1).numpy()
        np.testing.assert_allclose(log_prob, expected, rtol=2e-6, atol=2e-6)
        self.assertEqual(actions.dtype, np.float32)
        self.assertEqual(log_prob.dtype, np.float32)

    def test_action_sampling_does_not_consume_numpy_global_goal_rng(self):
        np.random.seed(91)
        expected = np.random.rand(4)
        np.random.seed(91)
        sampler = self.support.NumpyNormalSampler(3)
        sampler.seed(7)
        self.sample(sampler)
        np.testing.assert_array_equal(np.random.rand(4), expected)

    def test_invalid_standard_deviations_fail_before_a_rollout(self):
        sampler = self.support.NumpyNormalSampler(3)
        for invalid in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(std=invalid), self.assertRaises(ValueError):
                self.sample(sampler, std=np.array([1.0, invalid, 1.0], np.float32))

    def test_wrong_action_dimension_is_rejected(self):
        sampler = self.support.NumpyNormalSampler(3)
        with self.assertRaises(ValueError):
            self.sample(sampler, mean=np.zeros((5, 2), np.float32))


@unittest.skipIf(torch is None, "Requires the ArtiGrasp NumPy/PyTorch environment")
class TrainingCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.support = importlib.import_module("raisimGymTorch.helper.training_support")
        cls.ppo_module = importlib.import_module("raisimGymTorch.algo.ppo.module")

    def models(self, seed, learning_rate=0.01):
        torch.manual_seed(seed)
        actor = self.ppo_module.Actor(
            self.ppo_module.MLP([4], torch.nn.LeakyReLU, 3, 2),
            self.ppo_module.MultivariateGaussianDiagonalCovariance(
                2, 2, 1.0, self.support.NumpyNormalSampler(2), seed=seed), "cpu")
        critic = self.ppo_module.Critic(
            self.ppo_module.MLP([4], torch.nn.LeakyReLU, 3, 1), "cpu")
        optimizer = torch.optim.Adam(
            [*actor.parameters(), *critic.parameters()], lr=learning_rate)
        return actor, critic, optimizer

    def update(self, actor, critic, optimizer):
        obs = torch.tensor([[0.3, 0.2, -0.1], [0.1, 0.7, 0.5]])
        loss = actor.architecture.architecture(obs).square().sum()
        loss = loss + critic.architecture.architecture(obs).square().sum()
        loss = loss + actor.distribution.std.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        actor.update()

    def assert_same_weights(self, source, target):
        for a, b in ((source[0].architecture, target[0].architecture),
                     (source[0].distribution, target[0].distribution),
                     (source[1].architecture, target[1].architecture)):
            self.assertEqual(set(a.state_dict()), set(b.state_dict()))
            for key, value in a.state_dict().items():
                self.assertTrue(torch.equal(value, b.state_dict()[key]), key)

    def test_saved_checkpoint_contains_post_update_weights_and_optimizer(self):
        actor, critic, optimizer = self.models(7)
        before = {key: value.clone() for key, value in actor.architecture.state_dict().items()}
        self.update(actor, critic, optimizer)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(self.support.save_training_checkpoint(
                folder, 1, "l", actor, critic, optimizer))
            self.assertEqual(path.name, "full_1_l.pt")
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        self.assertTrue(checkpoint["optimizer_state_dict"]["state"])
        self.assertTrue(any(not torch.equal(before[key], value)
                            for key, value in checkpoint["actor_architecture_state_dict"].items()))
        for key, value in actor.architecture.state_dict().items():
            self.assertTrue(torch.equal(value, checkpoint["actor_architecture_state_dict"][key]))

    def test_stage_transfer_loads_weights_and_keeps_a_fresh_optimizer(self):
        source = self.models(7)
        self.update(*source)
        target = self.models(19, learning_rate=0.003)
        with tempfile.TemporaryDirectory() as folder:
            path = self.support.save_training_checkpoint(folder, 1, "r", *source)
            metadata = self.support.load_training_checkpoint(path, *target)
        self.assert_same_weights(source, target)
        self.assertFalse(target[2].state)
        self.assertEqual(target[2].param_groups[0]["lr"], 0.003)
        self.assertFalse(metadata["optimizer_loaded"])
        # The Gaussian's NumPy cache must also use the loaded noise scale.
        np.testing.assert_array_equal(target[0].distribution.std_np,
                                      source[0].distribution.std.detach().numpy())

    def test_optimizer_resume_requires_an_explicit_request(self):
        source = self.models(7)
        self.update(*source)
        target = self.models(19, learning_rate=0.003)
        with tempfile.TemporaryDirectory() as folder:
            path = self.support.save_training_checkpoint(folder, 1, "r", *source)
            metadata = self.support.load_training_checkpoint(path, *target, load_optimizer=True)
        self.assert_same_weights(source, target)
        self.assertTrue(metadata["optimizer_loaded"])
        self.assertEqual(target[2].param_groups[0]["lr"], 0.01)
        self.assertTrue(target[2].state)
        source_state = source[2].state_dict()["state"]
        for key, state in target[2].state_dict()["state"].items():
            for name in ("step", "exp_avg", "exp_avg_sq"):
                self.assertTrue(torch.equal(state[name], source_state[key][name]))

    def test_missing_checkpoint_stops_transfer(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises((FileNotFoundError, ValueError)):
                self.support.load_training_checkpoint(Path(folder) / "missing.pt", *self.models(3))


if __name__ == "__main__":
    unittest.main()
