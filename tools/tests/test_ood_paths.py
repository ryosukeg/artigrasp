"""Regression coverage for PYTHONPATH/editable-install package shadowing."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

TOOLS = Path(__file__).resolve().parents[1]


class PackagePathTests(unittest.TestCase):
    def probe(self, repo, body):
        # Use a fresh interpreter so cached imports from other tests cannot hide
        # the real lookup behavior of a user's terminal invocation.
        program = "\n".join([
            "import sys, json, types",
            "from pathlib import Path",
            "repo = Path({!r})".format(str(repo)),
            "sys.path[:0] = [str(repo), {!r}, str(repo / 'raisimGymTorch')]".format(str(TOOLS)),
            textwrap.dedent(body),
        ])
        output = subprocess.check_output([sys.executable, "-c", program],
                                         universal_newlines=True)
        return json.loads(output)

    def create_checkout(self, folder, extension=True):
        repo = Path(folder).resolve()
        outer = repo / "raisimGymTorch"
        inner = outer / "raisimGymTorch"
        (inner / "env").mkdir(parents=True)
        (outer / "__init__.py").write_text("LOCATION = 'outer'\n")
        (inner / "__init__.py").write_text("LOCATION = 'inner'\n")
        (inner / "env" / "__init__.py").write_text("")
        if extension:
            (inner / "env" / "bin").mkdir()
            (inner / "env" / "bin" / "compose_eval.py").write_text("__version__ = 'probe'\n")
        for filename in ("left.pt", "right.pt", "arti.pt"):
            (repo / filename).write_bytes(b"checkpoint fixture")
        return repo

    def test_existing_package_parent_is_promoted_ahead_of_outer_package(self):
        with tempfile.TemporaryDirectory() as folder:
            repo = self.create_checkout(folder)
            result = self.probe(repo, """
                from ood_suite.paths import configure_package_path
                configure_package_path(repo)
                configure_package_path(repo)
                import raisimGymTorch.env.bin.compose_eval
                import raisimGymTorch
                print(json.dumps({'location': raisimGymTorch.LOCATION,
                                  'first': sys.path[0],
                                  'copies': sys.path.count(str(repo / 'raisimGymTorch'))}))
            """)
            self.assertEqual(result["location"], "inner")
            self.assertEqual(result["first"], str(repo / "raisimGymTorch"))
            self.assertEqual(result["copies"], 1)

    def check_probe(self, repo):
        return self.probe(repo, """
            import ood_eval
            ood_eval.REPO_ROOT = repo
            for name in ('numpy', 'torch', 'ruamel.yaml'):
                module = types.ModuleType(name)
                module.__version__ = 'probe'
                sys.modules[name] = module
            config = {'objects': [], 'checkpoints': {
                'cooperative_left': 'left.pt', 'cooperative_right': 'right.pt',
                'articulation_right': 'arti.pt'}}
            errors, versions = ood_eval.check_runtime(config, verbose=False)
            print(json.dumps({'errors': errors, 'versions': versions}))
        """)

    def test_check_uses_inner_package_with_existing_editable_path(self):
        with tempfile.TemporaryDirectory() as folder:
            repo = self.create_checkout(folder)
            result = self.check_probe(repo)
            self.assertEqual(result["errors"], [])
            self.assertEqual(result["versions"]["raisimGymTorch_path"],
                             str(repo / "raisimGymTorch/raisimGymTorch/__init__.py"))

    def test_fresh_clone_has_actionable_missing_extension_diagnostic(self):
        with tempfile.TemporaryDirectory() as folder:
            repo = self.create_checkout(folder, extension=False)
            result = self.check_probe(repo)
            errors = "\n".join(result["errors"])
            self.assertIn("No module named 'raisimGymTorch.env.bin'", errors)
            self.assertIn("python setup.py develop", errors)
            self.assertIn("not included in a clone", errors)

    def test_wrong_cached_package_is_reported_without_silently_mixing_code(self):
        with tempfile.TemporaryDirectory() as folder:
            repo = self.create_checkout(folder)
            result = self.probe(repo, """
                import raisimGymTorch
                from ood_suite.paths import configure_package_path
                try:
                    configure_package_path(repo)
                except RuntimeError as error:
                    print(json.dumps({'error': str(error)}))
            """)
            self.assertIn("already imported from", result["error"])
            self.assertIn("Start a fresh Python process", result["error"])


if __name__ == "__main__":
    unittest.main()
