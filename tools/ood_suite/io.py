"""Small, simulator-independent helpers for experiment artifacts."""

import hashlib
import json
import os
import subprocess
from pathlib import Path


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as error:
                raise ValueError("Invalid JSONL at {}:{}: {}".format(path, number, error))
            episode_id = row["episode_id"]
            if episode_id in seen:
                raise ValueError("Duplicate result for {} in {}".format(episode_id, path))
            seen.add(episode_id)
            rows.append(row)
    return rows


def append_row(path, row):
    # Workers are sequential; fsync makes completed episodes survive interruption.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def code_fingerprint(repo_root):
    """Hash local source as well as HEAD, including uncommitted evaluator files."""
    repo_root = Path(repo_root)
    paths = list((repo_root / "tools").rglob("*.py"))
    paths += list((repo_root / "configs").glob("ood*.json"))
    paths += list((repo_root / "raisimGymTorch/raisimGymTorch").rglob("*.py"))
    paths += list((repo_root / "raisimGymTorch/raisimGymTorch/env").rglob("*.hpp"))
    paths += list((repo_root / "raisimGymTorch/raisimGymTorch/env").glob("*.cpp"))
    paths += list((repo_root / "raisimGymTorch/raisimGymTorch/env/envs/compose_eval/cfgs").glob("*.yaml"))
    hashes = {str(path.relative_to(repo_root)): file_sha256(path)
              for path in sorted(paths) if "__pycache__" not in path.parts}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), universal_newlines=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"git_commit": commit, "source_sha256": hashes}
