"""Inference adapter for the mixed angle/position-delta response predictor.

Feature construction, seven candidates, costs and selection rules are shared
with the absolute predictor. Denormalization uses float32 operations on the
model's device, as in guidance.Selector; delta restoration then adds the current
float32 angle/position on CPU. Returned predictions are absolute physical states.
No policy parameters or simulation state are modified by this adapter.
"""

import time

import numpy as np
import torch

from dataset import INPUT_DIM, OUTPUT_DIM
from delta_targets import DELTA_FEATURE_SPEC, TARGET_SPEC
from guidance import SETTINGS, candidate_scores, features
from train_model import ResponseMLP


class DeltaSelector:
    """The guidance.Selector API with an explicit delta-checkpoint contract."""

    def __init__(self, checkpoint_path, device):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (checkpoint.get("schema_version") != 1
                or checkpoint.get("artifact_type") != "box_response_delta_best_predictor"
                or checkpoint.get("target_spec") != TARGET_SPEC
                or checkpoint.get("feature_spec") != DELTA_FEATURE_SPEC
                or checkpoint.get("config", {}).get("target_representation") != "delta_angle_position"):
            raise ValueError("Expected the matching trained box-response delta best.pt")
        config = checkpoint.get("model_config", {})
        if (config.get("type") != "ResponseMLP" or config.get("input_dim") != INPUT_DIM
                or config.get("output_dim") != OUTPUT_DIM
                or any(type(config.get(key)) is not int or config[key] <= 0
                       for key in ("hidden_dim", "layers"))):
            raise ValueError("Unsupported delta predictor model configuration")
        normalization = checkpoint.get("normalization", {})
        dimensions = {"x_mean": INPUT_DIM, "x_std": INPUT_DIM, "y_mean": OUTPUT_DIM,
                      "y_std": OUTPUT_DIM, "absolute_y_mean": OUTPUT_DIM}
        for name, size in dimensions.items():
            value = normalization.get(name)
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                    or tuple(value.shape) != (size,) or not torch.isfinite(value).all()):
                raise ValueError("Invalid delta predictor normalization: " + name)
        if any(not (normalization[name] > 0).all() for name in ("x_std", "y_std")):
            raise ValueError("Invalid delta predictor normalization scale")
        if not torch.equal(normalization["y_mean"][[0, 2, 3, 4]], torch.zeros(4)):
            raise ValueError("Delta angle/position centers must be zero")
        absolute_indices = [1, 5, 6, 7, 8]
        if not torch.equal(normalization["y_mean"][absolute_indices],
                           normalization["absolute_y_mean"][absolute_indices]):
            raise ValueError("Absolute output centers disagree with training normalization")

        self.device = torch.device(device)
        self.model = ResponseMLP(config["hidden_dim"], config["layers"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval().to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
            if not torch.isfinite(parameter).all():
                raise ValueError("Nonfinite delta predictor weights")
        self.normalization = {name: normalization[name].to(self.device) for name in dimensions}
        self.candidates = np.asarray(SETTINGS["candidates"], np.float32)
        self.seconds = 0.

    def select(self, obs_r, obs_l, states, goals):
        start = time.perf_counter()
        x = features(obs_r, obs_l, states, goals, self.candidates)
        finite = np.isfinite(x).all(axis=(1, 2))
        x = np.nan_to_num(x, nan=0., posinf=0., neginf=0.)
        n = self.normalization
        with torch.no_grad():
            tensor = torch.from_numpy(x.reshape(-1, INPUT_DIM)).to(self.device)
            prediction = self.model((tensor - n["x_mean"]) / n["x_std"]) * n["y_std"] + n["y_mean"]
            prediction = prediction.cpu().numpy().reshape(len(x), len(self.candidates), OUTPUT_DIM)

        # Match dataset.persistence: angle comes from the raw right observation,
        # position from the current world-frame object state. Do not add a base
        # value to angular velocity or to the absolute quaternion components.
        # Unlike decode_targets (a training-data validator), inference must allow
        # a nonfinite row/candidate to reach the same fallback as guidance.py.
        with np.errstate(invalid="ignore", over="ignore"):
            prediction[:, :, 0] += np.asarray(obs_r[:, 273], np.float32)[:, None]
            prediction[:, :, 2:5] += np.asarray(states[:, 122:125], np.float32)[:, None, :]

        score = candidate_scores(prediction, goals, self.candidates)
        score[~finite] = np.inf
        selected = np.argmin(score, axis=1)
        fallback = ~np.isfinite(score).any(axis=1)
        selected[fallback] = 0
        self.seconds += time.perf_counter() - start
        return selected, prediction, score, fallback
