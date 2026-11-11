"""Resolve the inner Python package in ArtiGrasp's nested source layout."""

import importlib
from pathlib import Path
import shlex
import sys


def configure_package_path(repo_root):
    """Promote this checkout's package parent even when it is already on sys.path.

    setup.py develop and PYTHONPATH can put it behind the repository root. That
    root also contains raisimGymTorch/__init__.py, which is the outer directory,
    not the package containing env/helper/algo. Merely testing membership in
    sys.path leaves that incorrect package first.
    """
    parent = Path(repo_root).resolve() / "raisimGymTorch"
    expected = parent / "raisimGymTorch" / "__init__.py"
    if not expected.is_file():
        raise RuntimeError("Missing ArtiGrasp Python package: {}".format(expected))
    sys.path[:] = [str(parent)] + [entry for entry in sys.path
                                  if Path(entry or ".").resolve() != parent]
    importlib.invalidate_caches()
    loaded = sys.modules.get("raisimGymTorch")
    if loaded is not None:
        loaded_file = getattr(loaded, "__file__", None)
        if loaded_file is None or Path(loaded_file).resolve() != expected.resolve():
            raise RuntimeError(
                "raisimGymTorch was already imported from {}; expected {}. "
                "Start a fresh Python process and run tools/ood_eval.py again."
                .format(loaded_file, expected))
    return parent


def missing_extension_hint(repo_root):
    parent = Path(repo_root).resolve() / "raisimGymTorch"
    return (
        "No compatible compose_eval simulator extension was found in this checkout. "
        "Compiled env/bin modules are ignored by Git and are not included in a clone. "
        "Using the active artigrasp Python environment, build/register this checkout once:\n"
        "    cd {}\n"
        "    python setup.py develop\n"
        "    cd ..\n"
        "    python tools/ood_eval.py check"
    ).format(shlex.quote(str(parent)))
