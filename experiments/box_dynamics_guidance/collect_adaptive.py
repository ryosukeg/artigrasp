"""Variable-duration box rollouts, with per-episode causal evaluation stops.

Finished slots keep participating in batch inference/physics until the batch ends.
They are never reset/reordered; their evaluation is latched and saved arrays are
truncated at their stop. This preserves the original 100-slot numerical protocol.
"""
import time
import numpy as np

from collect import START, HORIZON, check_observation_layout, _float
from guidance import ScheduledPolicy
from adaptive_stop import AdaptiveStop, STOP_SETTINGS


class AdaptiveScheduledPolicy(ScheduledPolicy):
    def __init__(self, policy, env, goals, schedules, selector=None):
        super().__init__(policy, env, goals, schedules, selector)
        windows, count = schedules.shape[1], len(goals)
        self.selected = np.zeros((windows, count), np.int32)
        self.predictions = np.full((windows, count, 7, 9), np.nan, np.float32)
        self.scores = np.full((windows, count, 7), np.nan, np.float64)
        self.fallback = np.zeros((windows, count), bool)


def simulate_adaptive(env, controller, policy, schedules, settings=STOP_SETTINGS):
    count = controller.count
    max_opening = settings['max_opening_steps']
    steps = START + max_opening
    if settings['opening_start_step'] != START or settings['control_dt'] != .01:
        raise ValueError('Opening phase/control period mismatch')
    if (env.num_obs_r, env.num_obs_l, env.num_acts, env.num_gs) != (276, 276, 51, 138):
        raise ValueError('Unexpected simulator dimensions')
    if schedules.shape != (count, (max_opening + HORIZON - 1) // HORIZON, 3):
        raise ValueError('Wrong residual schedule dimensions')
    arrays = {'states': np.empty((steps + 1, count, 138), np.float32)}
    for key in ('actions_r', 'actions_l', 'ppo_actions_r', 'ppo_actions_l'):
        arrays[key] = np.empty((steps, count, 51), np.float32)
    arrays['residuals'] = np.zeros((steps, count, 3), np.float32)
    for key in ('modes_r', 'modes_l'):
        arrays[key] = np.empty((steps, count), np.int32)
    for key in ('bad_observation', 'bad_action', 'bad_state', 'auto_reset'):
        arrays[key] = np.zeros((steps, count), bool)
    for side in ('r', 'l'):
        arrays['final_obs_' + side] = np.empty((max_opening + 1, count, 276), np.float32)
    arrays['final_states'] = np.empty((max_opening + 1, count, 138), np.float32)
    goal = np.concatenate((controller.goal[:, :7], controller.angle), axis=1).astype(np.float32)
    monitor = AdaptiveStop(goal, settings)
    controller.reset()
    arrays['states'][0] = env.get_global_state()
    initial_invalid = ~np.isfinite(arrays['states'][0]).all(axis=1)
    invalid = initial_invalid.copy()
    first_invalid = np.where(invalid, 0, -1).astype(np.int32)
    wall = time.perf_counter()
    # Observe at each physical state, including the terminal one, exactly once.
    for step in range(steps + 1):
        if step < steps:
            controller.phase(step)
        obs_r, obs_l = env.observe(update_mean=False)
        bad_obs = ~np.isfinite(obs_r).all(axis=1) | ~np.isfinite(obs_l).all(axis=1)
        invalid |= bad_obs
        if step < steps:
            arrays['bad_observation'][step] = bad_obs
            arrays['modes_r'][step] = controller.modes_r
            arrays['modes_l'][step] = controller.modes_l
        if step >= START:
            offset = step - START
            if not np.all(controller.modes_r == 8) or not np.all(controller.modes_l == 7) or policy.articulation:
                raise ValueError('Wrong opening modes/policy')
            state = env.get_global_state()
            arrays['final_obs_r'][offset], arrays['final_obs_l'][offset] = obs_r, obs_l
            arrays['final_states'][offset] = state
            bad_state = ~np.isfinite(state).all(axis=1)
            invalid |= bad_state
            if step < steps:
                arrays['bad_state'][step] |= bad_state
            check_observation_layout(obs_r, state, goal, ~invalid)
            first_invalid[(first_invalid < 0) & invalid] = step
            monitor.update(offset, state, invalid)
            invalid_stop = np.asarray([reason == 'invalid' for reason in monitor.reasons], bool)
            first_invalid[(first_invalid < 0) & invalid_stop] = step
            if monitor.done.all():
                break
        right, left = policy.actions(np.nan_to_num(obs_r, nan=0., posinf=0., neginf=0.),
                                     np.nan_to_num(obs_l, nan=0., posinf=0., neginf=0.))
        arrays['ppo_actions_r'][step], arrays['ppo_actions_l'][step] = right, left
        right, left = right.copy(), left.copy()
        if step >= START:
            z = schedules[:, (step - START) // HORIZON]
            right[:, :3] += z
            arrays['residuals'][step] = z
        bad = ~np.isfinite(right).all(axis=1) | ~np.isfinite(left).all(axis=1)
        arrays['bad_action'][step] = bad
        invalid |= bad
        right = _float(np.nan_to_num(right, nan=0., posinf=0., neginf=0.))
        left = _float(np.nan_to_num(left, nan=0., posinf=0., neginf=0.))
        arrays['actions_r'][step], arrays['actions_l'][step] = right, left
        _, _, done = env.step(right, left)
        arrays['states'][step + 1] = env.get_global_state()
        arrays['auto_reset'][step] = np.asarray(done, bool)
        arrays['bad_state'][step] |= ~np.isfinite(arrays['states'][step + 1]).all(axis=1)
        invalid |= arrays['auto_reset'][step] | arrays['bad_state'][step]
        first_invalid[(first_invalid < 0) & invalid] = step
        if step and step % 500 == 0:
            print('  {}/{} steps | stopped {}/{} | {:.1f}s'.format(
                step, steps, int(monitor.done.sum()), count, time.perf_counter() - wall), flush=True)
    if not monitor.done.all():
        raise AssertionError('Every episode must have a recorded stop')
    # Slots may have continued after their latched evaluation stop. The writer
    # slices each one independently; later invalid states never affect its score.
    return arrays, goal, initial_invalid, first_invalid, monitor, step


def episode_arrays(arrays, index, stop_step):
    end = START + int(stop_step)
    return {key: value[:(stop_step + 1 if key.startswith('final_') else
                        end + 1 if key == 'states' else end), index]
            for key, value in arrays.items()}
