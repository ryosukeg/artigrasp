"""Runner setup and balanced train-only labels, without constructing RaiSim."""

from collections import Counter
import importlib
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "raisimGymTorch"))

try:
    import numpy as np
    import torch
    import ruamel.yaml
except ImportError:
    np = torch = None


@unittest.skipIf(torch is None, "Requires the ArtiGrasp training dependencies")
class TrainingRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = importlib.import_module("raisimGymTorch.helper.training_runtime")

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.cfg = self.root / "external_config.yaml"
        self.cfg.write_text("seed: 99\nrecord_video: true\nenvironment:\n"
                            "  num_envs: 16\n  num_threads: 4\n  render: true\n"
                            "  eval_every_n: 100\n")
        old_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, old_threads)

    def args(self, **overrides):
        values = dict(cfg=str(self.cfg), seed=1, num_iterations=2, num_repeats=16,
                      objects=list(self.runtime.PAPER_OBJECTS), torch_threads=1,
                      num_threads=2, checkpoint_every=7, debug=False, evaluate=False,
                      test=False, output_dir=str(self.root / "new_output"), headless=True,
                      device="cpu", pretrained_left=None, pretrained_right=None,
                      load_trained_policy=False, weight=None)
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_seed_one_overrides_yaml_and_absolute_config_is_loaded(self):
        args = self.args()
        cfg, seeds, device = self.runtime.configure_training(args, "general_two", self.root / "task")
        self.assertEqual(args.cfg, str(self.cfg.resolve()))
        self.assertEqual(cfg["seed"], 1)
        self.assertEqual(cfg["environment"]["seed"], seeds["environment"])
        self.assertEqual(cfg["environment"]["num_threads"], 2)
        self.assertEqual(cfg["environment"]["eval_every_n"], 7)
        self.assertFalse(cfg["environment"]["render"])
        self.assertFalse(cfg["record_video"])
        self.assertEqual(str(device), "cpu")

    def test_half_a_cooperative_checkpoint_pair_is_rejected(self):
        checkpoint = self.root / "left.pt"
        checkpoint.write_bytes(b"weights")
        args = self.args(pretrained_left=str(checkpoint))
        with self.assertRaisesRegex(ValueError, "both"):
            self.runtime.configure_training(args, "general_two", self.root)

    def test_legacy_pretrained_switch_cannot_silently_use_public_defaults(self):
        for stage in ("left_fixed", "multi_obj_arti", "general_two"):
            with self.subTest(stage=stage), self.assertRaises(ValueError):
                self.runtime.configure_training(self.args(load_trained_policy=True), stage, self.root)

    def test_test_split_or_existing_output_is_refused(self):
        with self.assertRaisesRegex(ValueError, "train split"):
            self.runtime.configure_training(self.args(test=True), "left_fixed", self.root)
        output = self.root / "new_output"
        output.mkdir()
        with self.assertRaises(FileExistsError):
            self.runtime.configure_training(self.args(), "left_fixed", self.root)

    def test_duplicate_objects_are_rejected_before_training(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.runtime.configure_training(self.args(objects=["box", "box"]), "left_fixed", self.root)

    def test_next_rollout_uses_the_updated_clamped_tensor_std_and_log_probability(self):
        module = importlib.import_module("raisimGymTorch.algo.ppo.module")
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                distribution = module.MultivariateGaussianDiagonalCovariance(
                    2, 4, 1.0, self.runtime.training_support.NumpyNormalSampler(2), seed=17).to(device)
                with torch.no_grad():
                    distribution.std.copy_(torch.tensor([0.1, 0.7], device=device))
                actor = SimpleNamespace(distribution=distribution, update=distribution.update)
                self.runtime.update_action_distribution(actor)
                tensor_std = distribution.std.detach().cpu()
                np.testing.assert_array_equal(distribution.std_np, tensor_std.numpy())
                np.testing.assert_allclose(tensor_std.numpy(), [0.2, 0.7], rtol=1e-7)
                mean = np.zeros((4, 2), np.float32)
                actions, log_prob = distribution.sample(mean)
                expected = torch.distributions.Normal(torch.from_numpy(mean), tensor_std)
                expected = expected.log_prob(torch.from_numpy(actions)).sum(dim=1).numpy()
                np.testing.assert_allclose(log_prob, expected, rtol=2e-6, atol=2e-6)

    def test_finite_rewards_do_not_hide_a_numerical_simulator_reset(self):
        reward = np.ones(2, np.float32)
        self.runtime.validate_training_step(np.array([False, False]), reward_r=reward)
        with self.assertRaisesRegex(FloatingPointError, "reset"):
            self.runtime.validate_training_step(np.array([False, True]), reward_r=reward)
        with self.assertRaisesRegex(FloatingPointError, "Non-finite"):
            self.runtime.validate_training_step(np.array([False, False]),
                                                reward_r=np.array([1.0, float("nan")]))

    def label_module(self, missing=None):
        directory = self.root / "grasp_label/final_labels/train_set"
        directory.mkdir(parents=True, exist_ok=True)
        for obj in self.runtime.PAPER_OBJECTS:
            labels = []
            for index, kind in enumerate(("two_grasp", "left_grasp", "right_grasp", "right_arti")):
                if (obj, kind) == missing:
                    continue
                labels.append(dict(seq_num=1, seq_name=obj + "_use_01", obj_name=obj,
                                   init_frame=10 + index, grasp_frame=20 + index, kind=kind))
            np.save(str(directory / (obj + ".npy")), np.array(labels, dtype=object))
        return SimpleNamespace(hand_path=str(self.root))

    def test_eight_objects_and_both_tasks_fill_the_exact_requested_batch(self):
        labels = self.label_module()
        for stage, count in (("left_fixed", 24), ("multi_obj_arti", 48), ("general_two", 48)):
            random.seed(17)
            _, _, cases = self.runtime.select_training_labels(
                stage, self.runtime.PAPER_OBJECTS, count, labels)
            self.assertEqual(len(cases), count)
            by_object = Counter(row["obj_name"] for row in cases)
            self.assertEqual(set(by_object), set(self.runtime.PAPER_OBJECTS))
            self.assertEqual(set(by_object.values()), {count // 8})
            if stage != "left_fixed":
                tasks = Counter((row["obj_name"], row["kind"] == "right_arti") for row in cases)
                self.assertEqual(len(tasks), 16)
                self.assertEqual(set(tasks.values()), {count // 16})

    def test_nondivisible_batch_still_has_exact_count_and_balanced_quotas(self):
        _, _, cases = self.runtime.select_training_labels(
            "general_two", self.runtime.PAPER_OBJECTS, 21, self.label_module())
        tasks = Counter((row["obj_name"], row["kind"] == "right_arti") for row in cases)
        self.assertEqual(len(cases), 21)
        self.assertEqual(len(tasks), 16)
        self.assertEqual(max(tasks.values()) - min(tasks.values()), 1)

    def test_training_label_sampling_is_reproducible_for_a_seed(self):
        labels = self.label_module()
        random.seed(17)
        first = self.runtime.select_training_labels("general_two", self.runtime.PAPER_OBJECTS, 48, labels)[2]
        random.seed(17)
        repeated = self.runtime.select_training_labels("general_two", self.runtime.PAPER_OBJECTS, 48, labels)[2]
        random.seed(18)
        other = self.runtime.select_training_labels("general_two", self.runtime.PAPER_OBJECTS, 48, labels)[2]
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, other)

    def test_underfilled_batches_and_missing_required_tasks_are_rejected(self):
        labels = self.label_module()
        for stage, count in (("left_fixed", 7), ("multi_obj_arti", 15), ("general_two", 15)):
            with self.subTest(stage=stage), self.assertRaisesRegex(ValueError, "at least"):
                self.runtime.select_training_labels(stage, self.runtime.PAPER_OBJECTS, count, labels)
        labels = self.label_module(missing=("box", "right_arti"))
        with self.assertRaisesRegex(ValueError, "required"):
            self.runtime.select_training_labels("multi_obj_arti", self.runtime.PAPER_OBJECTS, 16, labels)

    def test_saver_exclusively_creates_output_and_copies_the_selected_config(self):
        task = self.root / "task"
        task.mkdir()
        (task / "Environment.hpp").write_text("environment")
        (task / "runner.py").write_text("runner")
        args = self.args()
        saver = self.runtime.create_training_saver(args, self.root, task)
        self.assertEqual(Path(saver.data_dir), Path(args.output_dir))
        self.assertEqual((Path(saver.data_dir) / self.cfg.name).read_text(), self.cfg.read_text())
        with self.assertRaises(FileExistsError):
            self.runtime.create_training_saver(args, self.root, task)

    def test_checkpoint_failure_cannot_leave_a_successful_completion_record(self):
        output = self.root / "output"
        output.mkdir()
        saver = SimpleNamespace(data_dir=str(output))
        env = SimpleNamespace(save_scaling=mock.Mock(), close=mock.Mock(), num_envs=16)
        actor = SimpleNamespace(parameters=lambda: [torch.ones(1)])
        critic = SimpleNamespace(parameters=lambda: [torch.ones(1)])
        with mock.patch.object(self.runtime.training_support, "save_training_checkpoint",
                               side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                self.runtime.finish_training(saver, env, self.args(), "left_fixed", {},
                                             {"left": (actor, critic, None)})
        self.assertFalse((output / "training_result.json").exists())


if __name__ == "__main__":
    unittest.main()
