"""Frozen 20-step response-model candidate selection for the box experiment."""
import time

import numpy as np
import torch

from dataset import FEATURE_SPEC
from train_model import ResponseMLP


SETTINGS = {
    "horizon_steps": 20,
    "reselect_every_steps": 20,
    "phase_start_step": 1260,
    "candidates": [[0., 0., 0.], [1., 0., 0.], [-1., 0., 0.],
                   [0., 1., 0.], [0., -1., 0.], [0., 0., 1.], [0., 0., -1.]],
    "angle_scale_rad": .5,
    "position_scale_m": .05,
    "orientation_scale_rad": .2,
    "residual_squared_weight": .01,
    "score": "squared normalized angle + position + SO(3) errors + 0.01 * squared residual norm",
    "selection": "minimum score; first candidate (zero) wins ties; all invalid -> zero",
    "parameter_selection": "Fixed from task thresholds before test evaluation; no OOD tuning",
}


def features(obs_r, obs_l, states, goals, candidates):
    """Exactly the training feature order, with per-state goal quaternion signs."""
    objects = np.asarray(states[:, 122:130], np.float32)
    goals = np.asarray(goals, np.float32).copy()
    flip = np.sum(goals[:, 3:7] * objects[:, 3:7], axis=1) < 0
    goals[flip, 3:7] *= -1
    common = np.concatenate((obs_r, obs_l, objects, goals), axis=1)
    expanded = np.repeat(common[:, None, :], len(candidates), axis=1)
    residuals = np.broadcast_to(candidates, (len(common), len(candidates), 3))
    return np.concatenate((expanded, residuals), axis=2).astype(np.float32)


def candidate_scores(predictions, goals, candidates, settings=SETTINGS):
    """Task-scale costs; quaternion sign and predicted length cannot alter SO(3)."""
    predictions = np.asarray(predictions, np.float64)
    goals = np.asarray(goals, np.float64)
    norm = np.linalg.norm(predictions[:, :, 5:9], axis=-1)
    goal_norm = np.linalg.norm(goals[:, 3:7], axis=-1)
    good = (np.isfinite(predictions).all(axis=-1) & (norm > 1e-8) &
            np.isfinite(goals).all(axis=-1)[:, None] & (goal_norm > 1e-8)[:, None])
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        dot = np.sum(predictions[:, :, 5:9] * goals[:, None, 3:7], axis=-1)
        rotation = 2 * np.arccos(np.clip(np.abs(dot / (norm * goal_norm[:, None])), 0, 1))
        score = ((predictions[:, :, 0] - goals[:, None, 7]) / settings["angle_scale_rad"]) ** 2
        score += np.sum((predictions[:, :, 2:5] - goals[:, None, :3]) ** 2, axis=-1) / settings["position_scale_m"] ** 2
        score += (rotation / settings["orientation_scale_rad"]) ** 2
        score += settings["residual_squared_weight"] * np.sum(candidates ** 2, axis=-1)[None, :]
    return np.where(good & np.isfinite(score), score, np.inf)


class Selector:
    def __init__(self, checkpoint_path, device):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("artifact_type") != "box_response_best_predictor" or checkpoint["feature_spec"] != FEATURE_SPEC:
            raise ValueError("Expected the matching trained box-response best.pt")
        config = checkpoint["model_config"]
        self.device = torch.device(device)
        self.model = ResponseMLP(config["hidden_dim"], config["layers"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval().to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
            if not torch.isfinite(parameter).all():
                raise ValueError("Nonfinite predictor weights")
        self.normalization = {name: value.to(self.device) for name, value in checkpoint["normalization"].items()}
        if any(not torch.isfinite(value).all() for value in self.normalization.values()):
            raise ValueError("Nonfinite predictor normalization")
        if any(not (self.normalization[name] > 0).all() for name in ("x_std", "y_std")):
            raise ValueError("Invalid normalization scale")
        self.candidates = np.asarray(SETTINGS["candidates"], np.float32)
        self.seconds = 0.

    def select(self, obs_r, obs_l, states, goals):
        start = time.perf_counter()
        x = features(obs_r, obs_l, states, goals, self.candidates)
        finite = np.isfinite(x).all(axis=(1, 2))
        x = np.nan_to_num(x, nan=0., posinf=0., neginf=0.)
        n = self.normalization
        with torch.no_grad():
            tensor = torch.from_numpy(x.reshape(-1, 571)).to(self.device)
            prediction = (self.model((tensor - n["x_mean"]) / n["x_std"]) * n["y_std"] + n["y_mean"])
            prediction = prediction.cpu().numpy().reshape(len(x), 7, 9)
        score = candidate_scores(prediction, goals, self.candidates)
        score[~finite] = np.inf
        selected = np.argmin(score, axis=1)
        fallback = ~np.isfinite(score).any(axis=1)
        selected[fallback] = 0
        self.seconds += time.perf_counter() - start
        return selected, prediction, score, fallback


class ScheduledPolicy:
    """Update the collector's schedule just before it applies the current residual.

    The existing collector remains responsible for phase order, recording and
    env.step. PPO actions returned here are untouched; the collector adds z.
    """
    def __init__(self, policy, env, goals, schedules, selector=None):
        self.policy, self.env, self.goals = policy, env, goals
        self.schedules, self.selector = schedules, selector
        self.step = 0
        count = len(goals)
        self.selected = np.zeros((15, count), np.int32)
        self.predictions = np.full((15, count, 7, 9), np.nan, np.float32)
        self.scores = np.full((15, count, 7), np.nan, np.float64)
        self.fallback = np.zeros((15, count), bool)

    @property
    def articulation(self):
        return self.policy.articulation

    @articulation.setter
    def articulation(self, value):
        self.policy.articulation = value

    def actions(self, obs_r, obs_l):
        right, left = self.policy.actions(obs_r, obs_l)
        if self.selector is not None and self.step >= 1260 and (self.step - 1260) % 20 == 0:
            window = (self.step - 1260) // 20
            selected, predictions, scores, fallback = self.selector.select(
                obs_r, obs_l, self.env.get_global_state(), self.goals)
            self.schedules[:, window] = self.selector.candidates[selected]
            self.selected[window], self.predictions[window] = selected, predictions
            self.scores[window], self.fallback[window] = scores, fallback
        self.step += 1
        return right, left
