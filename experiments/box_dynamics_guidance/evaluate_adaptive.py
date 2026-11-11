#!/usr/bin/env python3
"""Adaptive stopping with exactly the frozen three-way goals, models and batches."""
import argparse
import copy
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np

from collect import (HERE, ROOT, START, HORIZON, RecordingController, atomic_npz,
                     prepare_data, load_environment_config, make_environment,
                     Policies, timing_for, lift_metrics, _paths)
import evaluate_threeway as frozen
from collect_adaptive import AdaptiveScheduledPolicy, simulate_adaptive, episode_arrays
from adaptive_stop import STOP_SETTINGS
from ood_suite.io import file_sha256, read_json, write_json
from ood_suite.metrics import evaluate_episode, evaluate_goal_diagnostics, quaternion_geodesic_rad

METHODS = frozen.METHODS


def fingerprint(plan):
    value = frozen.fingerprint(plan)
    value['adaptive_sources'] = {name: file_sha256(HERE / name) for name in (
        'evaluate_adaptive.py', 'collect_adaptive.py', 'adaptive_stop.py')}
    source = Path(plan['source_run'])
    value['source_plan_sha256'] = file_sha256(source / 'plan.json')
    value['source_results_sha256'] = file_sha256(source / 'results.json')
    return value


def check_plan(output):
    plan, saved = read_json(output / 'plan.json'), read_json(output / 'provenance.json')
    if saved['plan_sha256'] != file_sha256(output / 'plan.json'):
        raise ValueError('Plan changed; use a new directory')
    for key, value in fingerprint(plan).items():
        if saved[key] != value:
            raise ValueError('Provenance changed: ' + key)
    if plan['termination'] != STOP_SETTINGS or plan['methods'] != list(METHODS):
        raise ValueError('Termination/method mismatch')
    return plan


def create_plan(args):
    output, source = args.output.resolve(), args.source_run.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output must be empty; use run to resume')
    _paths(ROOT)
    plan = copy.deepcopy(frozen.check_plan(source))
    if not read_json(source / 'results.json').get('complete'):
        raise ValueError('Source comparison must be complete')
    plan.update(source_run=str(source), termination=STOP_SETTINGS,
                purpose='exploratory_single_seed_adaptive_termination',
                evaluation_stop_semantics=(
                    'Per-slot causal stop; no score/state after that stop is used. '
                    'Finished slots remain in the same inference/physics batch until all finish; '
                    'this preserves batch size and original prefix. Saved episodes are truncated.'),
                primary_metrics=['success_hold', 'paper_success', 'ood_goal_success'])
    source_hash = fingerprint(plan)
    write_json(output / 'plan.json', plan)
    source_hash.update(plan_sha256=file_sha256(output / 'plan.json'),
                       created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    write_json(output / 'provenance.json', source_hash)
    print('Frozen adaptive plan: {} episodes x 3 methods'.format(len(plan['episodes'])), flush=True)


def metrics_at(states, spec, invalid=False):
    timing = dict(timing_for('box'))
    timing['arti_steps'] = len(states) - 1 - timing['grasp_steps'] - 3 * timing['transit_steps']
    metrics = evaluate_episode(states, spec, timing)
    valid = metrics['valid'] and not invalid
    if not valid:
        metrics.update({key: False for key in metrics if key.endswith('success')})
        metrics.update({key: None for key in metrics if '_error_' in key})
    metrics['valid'] = valid
    metrics.update(lift_metrics(states, spec, timing, valid))
    metrics.update(evaluate_goal_diagnostics(states, spec, timing, valid))
    return metrics, timing


def worker(output, index):
    _paths(ROOT)
    plan = check_plan(output)
    task = frozen.tasks(plan)[index]
    name, method, specs = task
    if (output / 'batches' / (name + '.json')).exists():
        frozen.check_batch(output, task)
        return
    wall = time.perf_counter()
    runtime = plan['config']['runtime']
    data = prepare_data(ROOT, specs)
    config = load_environment_config(ROOT, len(specs), False, runtime['num_threads'])
    if not config['environment']['root_guided'] or config['environment']['control_dt'] != .01:
        raise ValueError('Expected root-guided 0.01-second control')
    env = make_environment(ROOT, data, config, 'box')
    try:
        env.seed(plan['seed'])
        base = Policies(ROOT, plan['config']['checkpoints'], config, env, runtime['device'], plan['seed'])
        controller = RecordingController(env, data, specs, timing_for('box'), base)
        goals = np.concatenate((controller.goal[:, :7], controller.angle), axis=1).astype(np.float32)
        max_opening = STOP_SETTINGS['max_opening_steps']
        schedules = np.zeros((len(specs), (max_opening + HORIZON - 1) // HORIZON, 3), np.float32)
        selector = frozen.selector_for(method, plan, runtime['device'])
        policy = AdaptiveScheduledPolicy(base, env, goals, schedules, selector)
        before = time.perf_counter()
        arrays, goals, initial, first, monitor, batch_steps = simulate_adaptive(env, controller, policy, schedules)
        rollout_seconds = time.perf_counter() - before
        rows = []
        for i, spec in enumerate(specs):
            opening = int(monitor.stop_step[i])
            end = START + opening
            windows = (opening + HORIZON - 1) // HORIZON
            saved = episode_arrays(arrays, i, opening)
            for key in ('grasp', 'articulation'):
                saved.update({'{}_{}'.format(key, j): value[i] for j, value in enumerate(data[key])})
            for key in ('left', 'right', 'arti_left', 'arti_right'):
                saved[key] = data[key][i]
            starts = np.arange(START, end, HORIZON, dtype=np.int32)
            ends = np.minimum(starts + HORIZON, end)
            events = np.zeros(end, bool)
            for key in ('bad_observation', 'bad_action', 'bad_state', 'auto_reset'):
                events |= saved[key]
            terminal_bad = not all(np.isfinite(saved[key][-1]).all()
                                   for key in ('final_obs_r', 'final_obs_l', 'final_states'))
            window_valid = np.asarray([
                not initial[i] and not events[:int(e)].any() and int(e - s) == HORIZON and
                (int(e) < end or (not terminal_bad and monitor.reasons[i] != 'invalid'))
                for s, e in zip(starts, ends)], bool)
            saved.update(goal=goals[i], window_z=schedules[i, :windows],
                         window_start_step=starts, window_end_step=ends, window_valid=window_valid,
                         initial_invalid=initial[i], terminal_bad_observation=terminal_bad,
                         control_dt=np.asarray(.01), horizon=np.asarray(HORIZON), schema_version=np.asarray(2),
                         guidance_selected=policy.selected[:windows, i],
                         guidance_fallback=policy.fallback[:windows, i],
                         stop_opening_step=np.asarray(opening), stop_reason=np.asarray(monitor.reasons[i]),
                         first_success_step=np.asarray(monitor.first_success_step[i]),
                         action_frame_quaternion_wxyz=saved['final_states'][0, 9:13])
            if selector is not None:
                saved.update(guidance_predictions=policy.predictions[:windows, i],
                             guidance_scores=policy.scores[:windows, i])
            metric_spec = controller.metric_spec(i)
            metrics, timing = metrics_at(saved['states'], metric_spec, monitor.reasons[i] == 'invalid')
            row = dict(metric_spec, **metrics)
            entry = saved['final_states'][0]
            entry_good = np.isfinite(entry).all() and np.linalg.norm(entry[125:129]) > 1e-8
            prefix_valid = not bool(initial[i]) and entry_good and not any(
                arrays[key][:START, i].any() for key in
                ('bad_observation', 'bad_action', 'bad_state', 'auto_reset'))
            prior_lift = lift_metrics(saved['states'], metric_spec, timing, prefix_valid)
            row['opening_entry'] = {
                'position_error_m': float(np.linalg.norm(entry[122:125] - goals[i, :3])) if entry_good else None,
                'orientation_error_rad': quaternion_geodesic_rad(entry[125:129], goals[i, 3:7]) if entry_good else None,
                'articulation_rad': float(entry[129]) if entry_good else None,
                'prior_rollout_valid': bool(prefix_valid),
                'prior_lift_success': prior_lift['lift_target_success']}
            row['fixed_3s_metrics'] = metrics_at(saved['states'][:1561], metric_spec,
                bool(initial[i]) or bool(events[:1560].any()) or not all(
                    np.isfinite(saved[key][300]).all()
                    for key in ('final_obs_r', 'final_obs_l', 'final_states')))[0] if opening >= 300 else None
            row['termination'] = {'reason': monitor.reasons[i], 'opening_step': opening,
                                  'opening_seconds': opening * .01, 'total_seconds': end * .01,
                                  'first_success_step': int(monitor.first_success_step[i])}
            path = Path('trajectories') / method / (spec['episode_id'] + '.npz')
            atomic_npz(output / path, saved)
            row.update(method=method, first_invalid_step=int(first[i]) if monitor.reasons[i] == 'invalid' else -1,
                       guidance_candidate_counts=np.bincount(policy.selected[:windows, i], minlength=7).tolist(),
                       guidance_fallback_count=int(policy.fallback[:windows, i].sum()),
                       trajectory_path=str(path), trajectory_sha256=file_sha256(output / path),
                       environment_config=config, timing=timing, policy_device=str(base.device))
            rows.append(row)
        record = {'complete': True, 'plan_sha256': file_sha256(output / 'plan.json'), 'episodes': rows,
                  'performance': {'wall_seconds': time.perf_counter() - wall,
                                  'rollout_seconds': rollout_seconds,
                                  'guidance_seconds': selector.seconds if selector else 0.,
                                  'environments': len(specs), 'batch_physics_steps': int(batch_steps),
                                  'policy_forward_devices': base.forward_devices}}
        write_json(output / 'batches' / (name + '.json'), record)
        print('Committed {}: {} trials, {:.1f}s'.format(name, len(rows), record['performance']['wall_seconds']), flush=True)
    finally:
        env.close()


def run(output):
    _paths(ROOT)
    plan = check_plan(output)
    with (output / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = []
        tasks = frozen.tasks(plan)
        for index, task in enumerate(tasks):
            name = task[0]
            if not (output / 'batches' / (name + '.json')).exists():
                log = output / 'logs' / (name + '.log')
                log.parent.mkdir(parents=True, exist_ok=True)
                print('Batch {}/{}: {}'.format(index + 1, len(tasks), name), flush=True)
                with log.open('w') as handle:
                    process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                        'worker', '--output', str(output), '--batch-index', str(index)],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True,
                        bufsize=1, start_new_session=True)
                    try:
                        for line in process.stdout:
                            print(line, end='', flush=True)
                            handle.write(line)
                            handle.flush()
                        if process.wait() != 0:
                            raise RuntimeError('Worker failed: ' + str(log))
                    except BaseException:
                        import signal
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGTERM)
                            process.wait()
                        raise
            records.append(frozen.check_batch(output, task))
        write_json(output / 'results.json', {'complete': True,
                   'plan_sha256': file_sha256(output / 'plan.json'),
                   'episodes': [row for rec in records for row in rec['episodes']],
                   'batch_performance': [rec['performance'] for rec in records]})
        from report_adaptive import write_report
        write_report(output)
        print('Adaptive evaluation complete: ' + str(output), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'run', 'worker', 'report'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-run', type=Path, default=HERE / 'runs/comparison_seed202_threeway_v1')
    parser.add_argument('--batch-index', type=int, default=0)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == 'plan':
        create_plan(args)
    elif args.command == 'run':
        run(output)
    elif args.command == 'worker':
        worker(output, args.batch_index)
    else:
        from report_adaptive import write_report
        write_report(output)


if __name__ == '__main__':
    main()
