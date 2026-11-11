#!/usr/bin/env bash
# Run with the existing ArtiGrasp Python environment, on the host or with it mounted.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
arti_python="${ARTIGRASP_PYTHON:-/home/cvrt/miniconda3/envs/artigrasp/bin/python}"
cd -- "$project_dir"

if [[ ! -x "$arti_python" ]]; then
  printf 'ArtiGrasp の既存 Python 環境が見つかりません: %s\n' "$arti_python" >&2
  printf '同じ環境が見える場所で実行するか、ARTIGRASP_PYTHON にその python のパスを指定してください。\n' >&2
  exit 1
fi

# Check dependencies/assets before creating a result directory.
"$arti_python" tools/ood_eval.py check --config configs/normal_five.json

# Report/config additions may differ; the actual evaluator and all prior inputs must match.
"$arti_python" - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path.cwd()
source = root / 'results/ood5_seed1'
plan_file = source / 'plan.json'
provenance = json.loads((source / 'provenance.json').read_text())
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
if digest(plan_file) != provenance['plan_sha256']:
    raise SystemExit('Existing OOD plan differs from its provenance.')
plan = json.loads(plan_file.read_text())
config = json.loads((root / 'configs/normal_five.json').read_text())
for field in ('seed', 'split', 'samples_per_object', 'objects', 'normal_goal',
              'physics_validation', 'checkpoints', 'evaluation_protocol', 'runtime'):
    if config[field] != plan['config'][field]:
        raise SystemExit('normal/OOD configuration mismatch: ' + field)
for name, expected in provenance['source_sha256'].items():
    if (name.startswith('raisimGymTorch/') or name in (
            'tools/ood_suite/runtime.py', 'tools/ood_suite/metrics.py',
            'tools/ood_suite/targets.py', 'tools/ood_suite/constraints.py',
            'tools/ood_suite/geometry.py')):
        path = root / name
        if not path.is_file() or digest(path) != expected:
            raise SystemExit('Existing OOD evaluator changed: ' + name)
for name, expected in provenance['input_sha256'].items():
    path = root / name
    if not path.is_file() or digest(path) != expected:
        raise SystemExit('Existing OOD input changed: ' + name)
print('normal/OOD evaluator, inputs and configuration: matched', flush=True)
PY

resume_args=()
if [[ -f results/normal_seed1/plan.json ]]; then
  resume_args=(--resume)
fi
"$arti_python" tools/ood_eval.py run \
  --config configs/normal_five.json --output results/normal_seed1 \
  --device cuda --num-threads 24 "${resume_args[@]}"
"$arti_python" scripts/compare_normal_ood.py \
  --normal results/normal_seed1 --ood results/ood5_seed1 \
  --output results/normal_vs_ood_seed1
