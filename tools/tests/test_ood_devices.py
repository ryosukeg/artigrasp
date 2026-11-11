"""Reject a claimed CUDA run whose policy parameters are actually on CPU."""

from collections import namedtuple
import ctypes
from pathlib import Path
import sys
import types
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ood_suite.runtime import Policies, policy_parameter_device, policy_tensor_devices

try:
    import torch
except ImportError:
    torch = None


class Device(namedtuple("DeviceBase", "type index")):
    __slots__ = ()

    def __str__(self):
        return self.type if self.index is None else "{}:{}".format(self.type, self.index)


def model(*devices):
    return types.SimpleNamespace(parameters=lambda: iter(
        types.SimpleNamespace(device=device) for device in devices))


class DeviceTests(unittest.TestCase):
    def test_cuda_report_uses_actual_device_index(self):
        actual = Device("cuda", 1)
        models = {name: model(actual, actual) for name in ("left", "right", "articulation")}
        self.assertEqual(policy_parameter_device(models, Device("cuda", None)), actual)

    def test_cpu_parameters_cannot_be_reported_as_cuda(self):
        with self.assertRaisesRegex(RuntimeError, "requested cuda, actual cpu"):
            policy_parameter_device({"left": model(Device("cpu", None))}, Device("cuda", None))

    def test_mixed_devices_or_wrong_gpu_are_rejected(self):
        cpu, gpu = Device("cpu", None), Device("cuda", 0)
        with self.assertRaisesRegex(RuntimeError, "one device"):
            policy_parameter_device({"left": model(gpu), "right": model(cpu)}, gpu)
        with self.assertRaisesRegex(RuntimeError, "device mismatch"):
            policy_parameter_device({"left": model(gpu)}, Device("cuda", 1))


    def test_forward_checks_actual_tensors_not_only_model_parameters(self):
        cpu, gpu = Device("cpu", None), Device("cuda", 0)
        tensors = {name: types.SimpleNamespace(device=gpu) for name in
                   ("input_r", "input_l", "output_r", "output_l")}
        self.assertEqual(set(policy_tensor_devices(tensors, gpu).values()), {"cuda:0"})
        tensors["output_r"].device = cpu
        with self.assertRaisesRegex(RuntimeError, "forward device mismatch"):
            policy_tensor_devices(tensors, gpu)


@unittest.skipIf(torch is None, "Optional real PyTorch/OpenMP regression check")
class ThreadInteractionTests(unittest.TestCase):
    def test_policy_initialization_and_forward_preserve_physics_thread_setting(self):
        # Use the OpenMP symbols from Torch itself: the old unconditional
        # set_num_threads(1) reset the setting made by native simulator init.
        root = Path(__file__).resolve().parents[2]
        libraries = list((Path(torch.__file__).parent / "lib").glob("libtorch_cpu.*"))
        if not libraries:
            self.skipTest("No shared Torch CPU library")
        library = ctypes.CDLL(str(libraries[0]))
        if not hasattr(library, "omp_get_max_threads"):
            self.skipTest("Torch was built without exported OpenMP symbols")
        library.omp_get_max_threads.restype = ctypes.c_int
        library.omp_set_num_threads.argtypes = [ctypes.c_int]
        import json
        import numpy as np
        from ood_suite.paths import configure_package_path
        configure_package_path(root)
        config = json.loads((root / "configs/ood_multiaxis.json").read_text())
        original = torch.get_num_threads()
        try:
            library.omp_set_num_threads(4)  # Same call as VectorizedEnvironment::init.
            env = types.SimpleNamespace(num_obs_r=276, num_obs_l=276, num_acts=51, num_envs=2)
            policy = Policies(root, config["checkpoints"],
                              {"architecture": {"policy_net": [128, 128]},
                               "environment": {"num_threads": 4}}, env, device="cpu")
            self.assertEqual(library.omp_get_max_threads(), 4)
            right, left = policy.actions(np.zeros((2, 276), dtype=np.float32),
                                         np.zeros((2, 276), dtype=np.float32))
            self.assertEqual(library.omp_get_max_threads(), 4)
            self.assertEqual(set(policy.forward_devices.values()), {"cpu"})
            self.assertEqual(right.shape, (2, 51))
            self.assertTrue(np.isfinite(right).all() and np.isfinite(left).all())
        finally:
            torch.set_num_threads(original)


if __name__ == "__main__":
    unittest.main()
