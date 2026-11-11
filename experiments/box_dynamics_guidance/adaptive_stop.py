"""Fixed, state-based stopping rules for the final free-base box articulation.

Opening step zero is the state AFTER phase(1260), before its first action.
At 0.01 s per control step, 300 and 1000 mean 3 s and 10 s of opening.
Success requires 51 consecutive states (both ends of a 50-step interval).
Stall uses 201 consecutive states, including both ends of a 200-step interval.
Its position/orientation excursions are measured from the FIRST state of that
window, not merely between endpoints and not over every pair of states.

Stall diagnoses lack of OBJECT motion; hand motion is intentionally irrelevant.
Slow progress or contact preparation can consequently be stopped prematurely.
These fixed heuristics do not establish that further control could not succeed.
An instantaneous successful state suppresses stall while the success hold builds;
the 10 s timeout still applies if that hold has not completed at the limit.
Only NumPy is required; no simulator, learned model, or policy is invoked.
"""

import copy

import numpy as np


STOP_SETTINGS = {
    "control_dt": .01,
    "opening_start_step": 1260,
    "min_opening_steps": 300,
    "max_opening_steps": 1000,
    "success_hold_steps": 50,
    "success_position_threshold_m": .05,
    "success_articulation_threshold_rad": .5,
    "success_orientation_threshold_rad": .2,
    "stall_window_steps": 200,
    "stall_articulation_range_rad": .02,
    "stall_position_displacement_m": .002,
    "stall_orientation_displacement_rad": .02,
    "stall_requires_current_failure": True,
    "quaternion_norm_epsilon": 1e-12,
    "priority": ["invalid", "success_hold", "stalled", "timeout"],
}


def _array(value, width, label):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != width or not result.shape[0]:
        raise ValueError("{} must have nonempty shape (N, {})".format(label, width))
    return result


def _unit_quaternions(value, epsilon):
    # Scaling before the norm also handles finite non-unit quaternions without
    # overflow. Zero, tiny, or nonfinite quaternions are invalid, not identities.
    finite = np.isfinite(value).all(axis=-1)
    scale = np.max(np.abs(value), axis=-1)
    safe_scale = np.where(finite & (scale > 0), scale, 1.)
    scaled = value / safe_scale[..., None]
    length = np.linalg.norm(scaled, axis=-1)
    valid = finite & (length > 0) & (scale > epsilon / np.maximum(length, 1e-300))
    unit = scaled / np.where(valid, length, 1.)[..., None]
    unit = np.where(valid[..., None], unit, np.nan)
    return unit, valid


def _metrics(states, goals, epsilon):
    if len(states) != len(goals):
        raise ValueError("states and goals must have the same batch size")
    state_q, state_valid = _unit_quaternions(states[:, 125:129], epsilon)
    goal_q, goal_valid = _unit_quaternions(goals[:, 3:7], epsilon)
    valid = (np.isfinite(states[:, 122:130]).all(axis=1) &
             np.isfinite(goals).all(axis=1) & state_valid & goal_valid)
    with np.errstate(invalid="ignore", over="ignore"):
        pe = np.linalg.norm(states[:, 122:125] - goals[:, :3], axis=1)
        aae = np.abs(states[:, 129] - goals[:, 7])
        dot = np.abs(np.sum(state_q * goal_q, axis=1))
        ae = 2 * np.arccos(np.clip(dot, 0., 1.))
    result = np.column_stack((pe, aae, ae))
    result[~valid | ~np.isfinite(result).all(axis=1)] = np.nan
    return result


def object_metrics(states, goals):
    """Return float64 (N, 3) columns PE[m], AAE[rad], AE[rad].

    Quaternions are wxyz, normalized before SO(3) comparison, and sign-invariant.
    Invalid object state/goal rows return three NaNs. Other state fields are not
    used here; AdaptiveStop additionally rejects any nonfinite state component.
    """
    return _metrics(_array(states, 138, "states"), _array(goals, 8, "goals"),
                    STOP_SETTINGS["quaternion_norm_epsilon"])


class AdaptiveStop:
    """Latch each environment's first stopping event.

    Call update with every integer opening_step, starting at zero. Public
    stop_step and first_success_step are opening-relative and initially -1.
    first_success_step records instantaneous three-threshold success, including
    before 3 s; it is not the success-hold completion time. reasons starts with
    None. metric_history contains independent (N, 3) arrays for steps 0..latest.
    Stopped rows' metrics and internal object histories remain frozen afterward.
    """

    def __init__(self, goals, settings=STOP_SETTINGS):
        self.settings = copy.deepcopy(settings)
        if set(self.settings) != set(STOP_SETTINGS):
            raise ValueError("Unexpected adaptive-stop setting keys")
        if self.settings["priority"] != ["invalid", "success_hold", "stalled", "timeout"]:
            raise ValueError("Stop priority must be invalid/success_hold/stalled/timeout")
        if self.settings["stall_requires_current_failure"] is not True:
            raise ValueError("Current success must suppress stall while its hold builds")
        integers = ("opening_start_step", "min_opening_steps", "max_opening_steps",
                    "success_hold_steps", "stall_window_steps")
        for key in integers:
            if (isinstance(self.settings[key], (bool, np.bool_)) or
                    not isinstance(self.settings[key], (int, np.integer)) or self.settings[key] < 1):
                raise ValueError("{} must be a positive integer".format(key))
        for key in set(STOP_SETTINGS) - set(integers) - {"priority", "stall_requires_current_failure"}:
            value = self.settings[key]
            if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
                raise ValueError("{} must be finite and positive".format(key))
        if not (max(self.settings["success_hold_steps"], self.settings["stall_window_steps"]) <=
                self.settings["min_opening_steps"] <= self.settings["max_opening_steps"]):
            raise ValueError("Both windows must fit before the minimum; minimum must not exceed maximum")
        self.goals = _array(goals, 8, "goals").copy()
        count = len(self.goals)
        self.done = np.zeros(count, dtype=bool)
        self.stop_step = np.full(count, -1, dtype=np.int32)
        self.reasons = [None] * count
        self.first_success_step = np.full(count, -1, dtype=np.int32)
        self.metric_history = []
        self._object_history = []
        self._next_step = 0
        self._thresholds = np.array([self.settings["success_position_threshold_m"],
                                     self.settings["success_articulation_threshold_rad"],
                                     self.settings["success_orientation_threshold_rad"]])

    def update(self, opening_step, states, invalid=None):
        """Process one consecutive state and return a copy of cumulative done."""
        if (isinstance(opening_step, (bool, np.bool_)) or
                not isinstance(opening_step, (int, np.integer)) or opening_step != self._next_step or
                opening_step > self.settings["max_opening_steps"]):
            raise ValueError("Expected consecutive opening step {} within [0, {}]".format(
                self._next_step, self.settings["max_opening_steps"]))
        states = _array(states, 138, "states")
        if len(states) != len(self.goals):
            raise ValueError("State batch size changed")
        if invalid is None:
            invalid = np.zeros(len(states), dtype=bool)
        else:
            invalid = np.asarray(invalid)
            if invalid.shape != (len(states),) or invalid.dtype.kind != "b":
                raise ValueError("invalid must be a Boolean array of shape (N,)")
        metrics = _metrics(states, self.goals, self.settings["quaternion_norm_epsilon"])
        bad = invalid | ~np.isfinite(states).all(axis=1) | ~np.isfinite(metrics).all(axis=1)
        objects = states[:, 122:130].copy()
        objects[:, 3:7], _ = _unit_quaternions(objects[:, 3:7], self.settings["quaternion_norm_epsilon"])
        if self.metric_history:
            metrics[self.done] = self.metric_history[-1][self.done]
            objects[self.done] = self._object_history[-1][self.done]
        self.metric_history.append(metrics)
        self._object_history.append(objects)
        active = ~self.done
        instant_success = active & ~bad & np.all(metrics < self._thresholds, axis=1)
        first = instant_success & (self.first_success_step < 0)
        self.first_success_step[first] = opening_step
        reasons = np.full(len(states), None, dtype=object)
        reasons[active & bad] = "invalid"
        eligible = active & ~bad
        if opening_step >= self.settings["min_opening_steps"]:
            hold = self.settings["success_hold_steps"] + 1
            success = np.all(np.asarray(self.metric_history[-hold:]) < self._thresholds, axis=(0, 2))
            reasons[eligible & success] = "success_hold"
            window = np.asarray(self._object_history[-(self.settings["stall_window_steps"] + 1):])
            with np.errstate(invalid="ignore", over="ignore"):
                angle_range = np.ptp(window[:, :, 7], axis=0)
                position_excursion = np.max(np.linalg.norm(window[:, :, :3] - window[0, :, :3], axis=2), axis=0)
                dot = np.abs(np.sum(window[:, :, 3:7] * window[0, :, 3:7], axis=2))
                orientation_excursion = np.max(2 * np.arccos(np.clip(dot, 0., 1.)), axis=0)
            stalled = ((angle_range < self.settings["stall_articulation_range_rad"]) &
                       (position_excursion < self.settings["stall_position_displacement_m"]) &
                       (orientation_excursion < self.settings["stall_orientation_displacement_rad"]))
            reasons[eligible & ~success & ~instant_success & stalled] = "stalled"
            if opening_step == self.settings["max_opening_steps"]:
                reasons[eligible & (reasons == None)] = "timeout"  # noqa: E711
        newly_done = active & (reasons != None)  # noqa: E711
        self.done[newly_done] = True
        self.stop_step[newly_done] = opening_step
        for index in np.flatnonzero(newly_done):
            self.reasons[index] = reasons[index]
        self._next_step += 1
        return self.done.copy()
