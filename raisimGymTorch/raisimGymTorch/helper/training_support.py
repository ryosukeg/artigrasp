"""Reproducible training helpers; evaluation keeps its released sampler unchanged."""

import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import torch


SAMPLER_VERSION = "numpy-pcg64-float32-v1"
SEED_VERSION = "artigrasp-training-seed-v1"
CHECKPOINT_KEYS = (
    "actor_architecture_state_dict", "actor_distribution_state_dict",
    "critic_architecture_state_dict", "optimizer_state_dict",
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_root():
    return Path(__file__).resolve().parents[3]


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return str(value)


def _json_bytes(value):
    return json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _write_json(path, value):
    Path(path).write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True,
                                    ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8")


def seed_training(master_seed, stage, torch_threads=1):
    """Seed every training RNG before labels, networks, or CUDA work are created.

    The returned environment/action seeds must also be passed to their owners.
    Setting PYTHONHASHSEED here cannot reseed this interpreter's hash function;
    it is recorded for child processes only. Component derivation never uses hash().
    """
    if isinstance(master_seed, bool) or not isinstance(master_seed, (int, np.integer)):
        raise ValueError("master_seed must be an integer")
    if int(master_seed) < 0:
        raise ValueError("master_seed must be nonnegative")
    if not isinstance(stage, str) or not stage:
        raise ValueError("stage must be a nonempty string")
    if isinstance(torch_threads, bool) or int(torch_threads) != torch_threads or torch_threads < 1:
        raise ValueError("torch_threads must be a positive integer")
    seeds = {}
    for component in ("python", "numpy", "torch", "environment", "action_left", "action_right"):
        token = "{}:{}:{}:{}".format(SEED_VERSION, int(master_seed), stage, component)
        seeds[component] = int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:4], "big")
    # pybind's environment setSeed accepts a signed C++ int, unlike the
    # Python/NumPy RNG APIs, which also accept the upper half of uint32.
    seeds["environment"] &= 0x7fffffff
    # cuBLAS requires this setting before the first CUDA context/workspace use.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seeds["python"])
    random.seed(seeds["python"])
    np.random.seed(seeds["numpy"])
    torch.manual_seed(seeds["torch"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seeds["torch"])
    torch.set_num_threads(int(torch_threads))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    seeds.update(master_seed=int(master_seed), stage=stage, version=SEED_VERSION,
                 torch_threads=int(torch_threads))
    return seeds


class NumpyNormalSampler:
    """NormalSampler interface with independent RNG state for each policy.

    Array order, rather than an OpenMP schedule, assigns noise to environments.
    This intentionally defines a new training sampler, not release-bit parity.
    """

    version = SAMPLER_VERSION

    def __init__(self, dim):
        if isinstance(dim, bool) or int(dim) != dim or dim < 1:
            raise ValueError("dim must be a positive integer")
        self.dim = int(dim)
        self.seed(0)

    def seed(self, seed):
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("sampler seed must be a nonnegative integer")
        self.rng = np.random.Generator(np.random.PCG64(int(seed)))

    def sample(self, mean, std, samples, logprob):
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        if mean.ndim != 2 or mean.shape[1] != self.dim:
            raise ValueError("mean must have shape (num_envs, dim)")
        if std.shape != (self.dim,):
            raise ValueError("std must have shape (dim,)")
        if samples.shape != mean.shape or logprob.shape != (mean.shape[0],):
            raise ValueError("samples/logprob output shapes do not match mean")
        if samples.dtype != np.float32 or logprob.dtype != np.float32:
            raise ValueError("samples/logprob output arrays must be float32")
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError("mean must be finite and std must be finite and positive")
        noise = self.rng.standard_normal(mean.shape, dtype=np.float32)
        samples[:] = mean + noise * std
        logprob[:] = -np.sum(np.float32(0.5) * noise * noise + np.log(std) +
                            np.float32(0.5 * np.log(2 * np.pi)), axis=1, dtype=np.float32)


def load_training_checkpoint(path, actor, critic, optimizer, load_optimizer=False):
    """Transfer explicit weights, optionally optimizer state, with no CSV fallback."""
    if path is None or str(path).strip() == "":
        raise ValueError("An explicit training checkpoint path is required")
    if load_optimizer and optimizer is None:
        raise ValueError("optimizer is required when load_optimizer=True")
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Training checkpoint does not exist: {}".format(checkpoint_path))
    digest = _sha256(checkpoint_path)
    # Public checkpoints contain only tensors and built-in containers. Restrict
    # deserialization while supporting the older torch used by the release.
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    except TypeError as error:
        if "weights_only" not in str(error):
            raise
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    required = CHECKPOINT_KEYS if load_optimizer else CHECKPOINT_KEYS[:3]
    if not isinstance(checkpoint, dict) or any(key not in checkpoint for key in required):
        raise ValueError("Checkpoint must contain {}".format(", ".join(required)))
    actor.architecture.load_state_dict(checkpoint[CHECKPOINT_KEYS[0]])
    actor.distribution.load_state_dict(checkpoint[CHECKPOINT_KEYS[1]])
    critic.architecture.load_state_dict(checkpoint[CHECKPOINT_KEYS[2]])
    # The distribution's cached std_np must agree before the next rollout.
    if hasattr(actor.distribution, "update"):
        actor.distribution.update()
    if load_optimizer:
        optimizer.load_state_dict(checkpoint[CHECKPOINT_KEYS[3]])
    return dict(path=str(checkpoint_path), sha256=digest,
                optimizer_loaded=bool(load_optimizer), observation_scaling_loaded=False,
                keys=sorted(checkpoint))


def save_training_checkpoint(data_dir, iteration, hand, actor, critic, optimizer):
    """Save the caller's current parameters (call after PPO.update) atomically."""
    if hand not in ("l", "r"):
        raise ValueError("hand must be 'l' or 'r'")
    if isinstance(iteration, bool) or int(iteration) != iteration or iteration < 0:
        raise ValueError("iteration must be a nonnegative integer")
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "full_{}_{}.pt".format(int(iteration), hand)
    checkpoint = dict(zip(CHECKPOINT_KEYS, (
        actor.architecture.state_dict(), actor.distribution.state_dict(),
        critic.architecture.state_dict(), optimizer.state_dict(),
    )))
    with tempfile.NamedTemporaryFile(prefix=".checkpoint-", suffix=".pt", dir=str(data_dir), delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(checkpoint, str(temporary))
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _input_paths(root, stage, args, cfg, labels):
    rsc = root / "rsc"
    mano = rsc / "mano_double"
    environment = cfg.get("environment", {})
    paths = {mano / "right_pose_mean.txt", mano / "left_pose_mean.txt"}
    for key, default in (("hand_model_r", "rhand_mano.urdf"), ("hand_model_l", "lhand_mano.urdf")):
        paths.add(mano / environment.get(key, default))
    for label in labels:
        seq = str(label["seq_name"])
        subject = "s{}".format(label["seq_num"])
        obj = seq.split("_")[0]
        suffix = "" if stage == "general_two" else "_fixed_base"
        paths.update((rsc / "arctic" / obj / (obj + suffix + ".urdf"),
                      mano / "grasp_label/final_labels/train_set" / (obj + ".npy"),
                      mano / subject / (seq + ".mano.npy"),
                      rsc / "arctic" / obj / subject / (seq + ".object.npy")))
        for side in ("r", "l", "min"):
            paths.add(mano / "ee" / subject / (seq + "_" + side + ".npy"))
        kind = label["kind"]
        if kind in ("two_grasp", "right_grasp", "right_arti", "two_arti", "right_special", "left_special"):
            paths.add(mano / "contact" / subject / (seq + "_a_r.npy"))
        if kind in ("two_grasp", "left_grasp", "left_arti", "two_arti", "right_special", "left_special"):
            paths.add(mano / "contact" / subject / (seq + "_a_l.npy"))
    # URDF meshes/textures are physical inputs too. Resolve paths relative to
    # their URDF, as the distributed files do; flag unresolved package URIs.
    for urdf in sorted(p for p in paths if p.suffix == ".urdf" and p.is_file()):
        for element in ET.parse(str(urdf)).iter():
            filename = element.attrib.get("filename")
            if filename:
                if filename.startswith("file://"):
                    paths.add(Path(filename[7:]))
                elif "://" in filename:
                    raise ValueError("Unresolved URDF resource URI: {} in {}".format(filename, urdf))
                else:
                    paths.add((urdf.parent / filename).resolve())
    return paths


def write_training_manifest(data_dir, stage, args, cfg, seeds, labels, processed_data, checkpoints=None):
    """Snapshot effective settings and hash all known selected training inputs.

    Missing known inputs fail before training. A manifest records software,
    seeds, sampler, and artifacts, not a guarantee of cross-hardware bit parity.
    """
    if stage not in ("multi_obj_arti", "left_fixed", "general_two"):
        raise ValueError("Unknown training stage: {}".format(stage))
    root = _project_root()
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    args_dict = _jsonable(args if isinstance(args, dict) else vars(args))
    config = _jsonable(cfg)
    environment = config.get("environment", {})
    package = root / "raisimGymTorch/raisimGymTorch"
    env_dir = package / "env/envs" / stage
    sources = [Path(__file__).resolve(), env_dir / "runner.py", env_dir / "Environment.hpp",
               package / "env/RaisimGymVecEnv.py", package / "env/VectorizedEnvironment.hpp",
               package / "env/RaisimGymEnv.hpp", package / "env/Reward.hpp",
               package / "helper/training_runtime.py",
               package / "helper/label_gen_final.py", package / "helper/rotations.py",
               package / "helper/all_frame_dict.py", package / "helper/raisim_gym_helper.py",
               package / "algo/ppo/module.py", package / "algo/ppo/ppo.py", package / "algo/ppo/storage.py"]
    cfg_file = env_dir / "cfgs" / str(args_dict.get("cfg", "cfg_reg.yaml"))
    sources.append(cfg_file)
    module_name = "raisimGymTorch.env.bin." + stage
    module = sys.modules.get(module_name)
    binary = Path(module.__file__).resolve() if module is not None else None
    if binary is None:
        candidates = sorted((package / "env/bin").glob(stage + ".*.so"))
        binary = candidates[0] if len(candidates) == 1 else package / "env/bin" / (stage + ".so")
    inputs = _input_paths(root, stage, args_dict, config, labels)
    missing = sorted(str(p) for p in sources + [binary] + list(inputs) if not p.is_file())
    if missing:
        _write_json(data_dir / "training_manifest.failed.json", dict(stage=stage, missing_inputs=missing))
        raise FileNotFoundError("Cannot record training provenance; {} required paths missing: {}".format(
            len(missing), ", ".join(missing[:5])))
    def key(path):
        try:
            return str(path.resolve().relative_to(root))
        except ValueError:
            return str(path.resolve())
    source_sha256 = {}
    for source in sources:
        relative = key(source)
        source_sha256[relative] = _sha256(source)
        # Include a source's tree path so same basenames never overwrite.
        target = data_dir / "sources" / relative.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(source), str(target))
    input_sha256 = {key(p): _sha256(p) for p in sorted(inputs)}
    array_records = []
    for array in processed_data:
        array = np.asarray(array)
        if array.dtype.hasobject:
            raise ValueError("processed_data must contain numeric arrays")
        metadata = dict(shape=list(array.shape), dtype=array.dtype.str)
        digest = hashlib.sha256(_json_bytes(metadata) + b"\n" + array.tobytes(order="C")).hexdigest()
        array_records.append(dict(metadata, sha256=digest))
    device = str(args_dict.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    runtime = dict(python=sys.version, executable=sys.executable, command=list(sys.argv),
                   working_directory=str(Path.cwd()), platform=platform.platform(),
                   numpy=np.__version__, torch=str(torch.__version__), cuda=torch.version.cuda,
                   cudnn=torch.backends.cudnn.version(), device=device,
                   device_source="args.device" if args_dict.get("device") else "cuda_availability",
                   num_envs=environment.get("num_envs"), batch_size=len(labels),
                   num_threads=environment.get("num_threads"), torch_threads=torch.get_num_threads(),
                   torch_threads_requested=args_dict.get("torch_threads", seeds.get("torch_threads")),
                   normalize_ob=False, sampler=SAMPLER_VERSION,
                   deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                   cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                   omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
                   mkl_num_threads=os.environ.get("MKL_NUM_THREADS"),
                   pythonhashseed=os.environ.get("PYTHONHASHSEED"),
                   pythonhashseed_applies="child interpreters; current hash seed is set at interpreter startup",
                   runtime_module=module_name, runtime_module_loaded=module is not None,
                   runtime_binary_path=str(binary), runtime_binary_sha256=_sha256(binary))
    manifest = dict(schema_version=1, stage=stage,
                    created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    args=args_dict, effective_config=config, seeds=_jsonable(seeds), runtime=runtime,
                    source_sha256=source_sha256, input_sha256=input_sha256,
                    labels_count=len(labels), labels_sha256=hashlib.sha256(_json_bytes(labels)).hexdigest(),
                    processed_arrays=array_records, checkpoints=_jsonable(checkpoints or {}))
    _write_json(data_dir / "effective_config.json", config)
    # JSON is a valid YAML document and preserves exact JSON scalar values.
    _write_json(data_dir / "effective_config.yaml", config)
    _write_json(data_dir / "training_labels.json", labels)
    _write_json(data_dir / "training_manifest.json", manifest)
    return manifest
