"""Shared runner setup for controlled, multi-object scratch experiments."""

import datetime
import json
import os
from pathlib import Path
import random
import shutil
from types import SimpleNamespace

import numpy as np
import torch
from ruamel.yaml import YAML

from raisimGymTorch.helper import training_support


PAPER_OBJECTS = ("box", "laptop", "notebook", "waffleiron", "mixer",
                 "ketchup", "espressomachine", "microwave")
ALL_OBJECTS = PAPER_OBJECTS + ("capsulemachine", "phone")


def add_training_arguments(parser, cooperative=False):
    parser.add_argument('--objects', nargs='+', choices=ALL_OBJECTS,
                        default=list(PAPER_OBJECTS), help='Objects mixed in one training run')
    parser.add_argument('--output-dir', help='New, exclusive directory for this stage')
    parser.add_argument('--headless', action='store_true', help='Disable rendering and video')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--num-threads', type=int, help='Physics threads')
    parser.add_argument('--torch-threads', type=int, default=1)
    parser.add_argument('--checkpoint-every', type=int, help='Save every N completed updates')
    parser.add_argument('--load-optimizer', action='store_true',
                        help='Also transfer optimizer state; default is fresh optimizers')
    if cooperative:
        parser.add_argument('--pretrained-left', help='Own left fixed-base checkpoint')
        parser.add_argument('--pretrained-right', help='Own right fixed-base checkpoint')


def configure_training(args, stage, task_path):
    if args.num_iterations < 1 or args.num_repeats < 1:
        raise ValueError('num_iterations and num_repeats must be positive')
    if len(set(args.objects)) != len(args.objects):
        raise ValueError('Duplicate objects are not allowed')
    if args.torch_threads < 1 or (args.num_threads is not None and args.num_threads < 1):
        raise ValueError('Thread counts must be positive')
    if args.checkpoint_every is not None and args.checkpoint_every < 1:
        raise ValueError('checkpoint-every must be positive')
    # Debug/evaluate change only num_envs in the historical runners, leaving the
    # generated arrays at the full batch shape. Use a small real batch instead.
    if getattr(args, 'debug', False) or getattr(args, 'evaluate', False):
        raise ValueError('Use a small --num_repeats batch instead of --debug/--evaluate')
    if args.test:
        raise ValueError('Training uses the train split; use the evaluation CLI for test')
    if args.output_dir and Path(args.output_dir).exists():
        raise FileExistsError('Refusing to reuse training output: ' + args.output_dir)
    cfg_path = Path(args.cfg).expanduser()
    if not cfg_path.is_absolute():
        candidate = Path(task_path) / 'cfgs' / cfg_path
        cfg_path = candidate if candidate.is_file() else cfg_path.resolve()
    args.cfg = str(cfg_path.resolve(strict=True))
    with cfg_path.open() as stream:
        cfg = YAML().load(stream)
    seeds = training_support.seed_training(args.seed, stage, args.torch_threads)
    cfg['seed'] = args.seed
    cfg['environment']['seed'] = seeds['environment']
    if args.num_threads is not None:
        cfg['environment']['num_threads'] = args.num_threads
    if args.headless:
        cfg['environment']['render'] = False
        cfg['record_video'] = False
    if args.checkpoint_every is not None:
        cfg['environment']['eval_every_n'] = args.checkpoint_every
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    args.device = str(device)
    if stage == 'general_two':
        pair = (args.pretrained_left, args.pretrained_right)
        if any(pair) and not all(pair):
            raise ValueError('Provide both --pretrained-left and --pretrained-right')
        if args.load_trained_policy and not all(pair):
            raise ValueError('-re requires --pretrained-left and --pretrained-right')
        if args.weight:
            raise ValueError('Use --pretrained-left/--pretrained-right for cooperation')
        paths = [p for p in pair if p]
    else:
        if args.load_trained_policy and not args.weight:
            raise ValueError('-re requires an explicit --weight checkpoint')
        paths = [args.weight] if args.weight else []
    for path in paths:
        if not Path(path).expanduser().is_file():
            raise FileNotFoundError('Checkpoint does not exist: ' + path)
    return cfg, seeds, device


def select_training_labels(stage, objects, count, label_module):
    """Balance objects and task kinds, producing exactly count train cases.

    This is separate from the legacy evaluation sampler, which assumes ten
    objects and silently underfills batches when that set changes.
    """
    grasp_kinds = {'left_grasp', 'two_grasp'} if stage == 'left_fixed' else {
        'right_grasp', 'two_grasp'}
    if stage == 'general_two':
        grasp_kinds.add('left_grasp')
    groups, articulation, grasp = [], {}, {}
    for obj in objects:
        path = Path(label_module.hand_path) / 'grasp_label/final_labels/train_set' / (obj + '.npy')
        rows = np.load(str(path), allow_pickle=True)
        grasp[obj + '_list'] = [dict(row) for row in rows if row['kind'] in grasp_kinds]
        articulation[obj + '_list'] = [dict(row) for row in rows if row['kind'] == 'right_arti']
        groups.append(grasp[obj + '_list'])
        if stage != 'left_fixed':
            groups.append(articulation[obj + '_list'])
    if count < len(groups):
        raise ValueError('num_repeats must be at least %d to cover every object/task kind' % len(groups))
    if any(not group for group in groups):
        raise ValueError('Selected objects lack a required grasp/articulation train reference')
    cases = []
    base, remainder = divmod(count, len(groups))
    for index, group in enumerate(groups):
        quota = base + (index < remainder)
        repeats, rest = divmod(quota, len(group))
        cases.extend(dict(row) for row in group for _ in range(repeats))
        cases.extend(dict(row) for row in random.sample(group, rest))
    random.shuffle(cases)
    return articulation, grasp, cases


def create_training_saver(args, exp_path, task_path):
    if args.output_dir:
        out = Path(args.output_dir).expanduser().resolve()
    else:
        stamp = datetime.datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S-%f')
        out = Path(exp_path) / 'raisimGymTorch' / args.storedir / args.exp_name / stamp
    out.mkdir(parents=True, exist_ok=False)
    for source in (Path(args.cfg), Path(task_path) / 'Environment.hpp', Path(task_path) / 'runner.py'):
        shutil.copy2(str(source), str(out / source.name))
    return SimpleNamespace(data_dir=str(out))


def ensure_finite(**arrays):
    for name, array in arrays.items():
        if not np.isfinite(array).all():
            raise FloatingPointError('Non-finite training values: ' + name)


def validate_training_step(dones, **rewards):
    ensure_finite(**rewards)
    # These three fixed-horizon environments mark only numerical instability
    # as terminal. The native wrapper resets such a world before observe().
    if np.any(dones):
        raise FloatingPointError('Numerical instability triggered an environment reset')


def update_action_distribution(actor, minimum_std=0.2):
    actor.distribution.enforce_minimum_std(
        torch.full_like(actor.distribution.std, minimum_std))
    # sample() uses a CPU NumPy cache; both GPU updates and the std clamp can
    # leave it detached from the current trainable tensor.
    actor.update()


def save_completed_update(saver, env, completed, actors):
    paths = {}
    for hand, (actor, critic, optimizer) in actors.items():
        for parameter in actor.parameters() + critic.parameters():
            if not torch.isfinite(parameter).all():
                raise FloatingPointError('Non-finite trained parameters for ' + hand)
        suffix = {'left': 'l', 'right': 'r'}[hand]
        paths[hand] = str(training_support.save_training_checkpoint(
            saver.data_dir, completed, suffix, actor, critic, optimizer))
    env.save_scaling(saver.data_dir, str(completed))
    return paths


def finish_training(saver, env, args, stage, seeds, actors):
    paths = save_completed_update(saver, env, args.num_iterations, actors)
    result = {'complete': True, 'stage': stage, 'seed': args.seed, 'seeds': seeds,
              'iterations_completed': args.num_iterations, 'objects': args.objects,
              'checkpoints': paths, 'checkpoint_timing': 'after_completed_ppo_updates',
              'num_envs': env.num_envs, 'device': args.device}
    output = Path(saver.data_dir) / 'training_result.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('Training result: ' + str(output), flush=True)
    env.close()
