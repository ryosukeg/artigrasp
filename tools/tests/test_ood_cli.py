"""Plan immutability and resume checks, without constructing a simulator."""

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import ood_eval
from ood_suite.io import code_fingerprint, file_sha256, write_json


class ProvenanceTests(unittest.TestCase):
    def test_environment_yaml_is_part_of_source_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = root / "raisimGymTorch/raisimGymTorch/env/envs/compose_eval/cfgs/cfg_reg.yaml"
            config.parent.mkdir(parents=True)
            config.write_text("environment: {control_dt: 0.01}\n")
            with mock.patch("ood_suite.io.subprocess.check_output", return_value="revision\n"):
                before = code_fingerprint(root)
                config.write_text("environment: {control_dt: 0.02}\n")
                after = code_fingerprint(root)
            self.assertNotEqual(before["source_sha256"], after["source_sha256"])

    def test_plan_and_input_mutations_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / "results"
            data = root / "asset.npy"
            data.write_bytes(b"original")
            write_json(output / "plan.json", {"episodes": ["fixed"]})
            source = {"source_sha256": {"evaluator.py": "source"}}
            write_json(output / "provenance.json", {
                "source_sha256": source["source_sha256"],
                "input_sha256": {"asset.npy": file_sha256(data)},
                "plan_sha256": file_sha256(output / "plan.json")})
            with mock.patch.object(ood_eval, "REPO_ROOT", root), \
                    mock.patch.object(ood_eval, "code_fingerprint", return_value=source):
                ood_eval.verify_provenance(output)
                data.write_bytes(b"changed")
                with self.assertRaisesRegex(RuntimeError, "input changed"):
                    ood_eval.verify_provenance(output)
                data.write_bytes(b"original")
                write_json(output / "plan.json", {"episodes": ["different"]})
                with self.assertRaisesRegex(RuntimeError, "plan.json changed"):
                    ood_eval.verify_provenance(output)

    def test_resume_accepts_effective_options_and_rejects_target_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            plan = {"config": {"samples_per_object": 2, "seed": 7, "split": "test",
                               "objects": ["box"], "profiles": {"id_control": {}}},
                    "episodes": []}
            write_json(output / "plan.json", plan)
            args = argparse.Namespace(output=output, resume=True, samples=2, seed=7,
                                      split="test", objects=["box"], profiles=["id_control"])
            with mock.patch.object(ood_eval, "verify_provenance"):
                self.assertEqual(ood_eval._load_or_create(args), plan)
                changed = copy.copy(args)
                changed.samples = 3
                with self.assertRaisesRegex(ValueError, "samples"):
                    ood_eval._load_or_create(changed)


if __name__ == "__main__":
    unittest.main()
