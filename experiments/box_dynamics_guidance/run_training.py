#!/usr/bin/env python3
"""User-launched collection -> validation -> supervised fitting, with progress.

Default: 1,000 box episodes (20 parallel), then up to 100 fitting epochs.
No OOD simulation or action guidance is started by this command.
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time

from progress import Progress

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@contextlib.contextmanager
def workflow_lock(dataset):
    identity = hashlib.sha256(str(dataset).encode("utf-8")).hexdigest()[:16]
    path = HERE / "logs" / ("workflow_" + identity + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("A workflow for this dataset is already running; watch its original terminal/logs")
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(str(os.getpid()) + "\n")
            handle.flush()
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def parser_for_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--episodes", type=positive, default=1000)
    parser.add_argument("--seed", type=int, default=301)
    parser.add_argument("--amplitude", type=float, default=1.)
    parser.add_argument("--collect-batch-size", type=positive, default=20)
    parser.add_argument("--num-threads", type=positive, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=positive, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--train-batch-size", type=positive, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=positive, default=256)
    parser.add_argument("--layers", type=positive, default=3)
    parser.add_argument("--dry-run", action="store_true", help="Print stages/commands without writing files or starting work")
    return parser


def paths_for(args):
    name = "box_response_{}_seed{}".format(args.episodes, args.seed)
    dataset = (args.dataset or HERE / "datasets" / name).expanduser().resolve()
    output = (args.output or HERE / "checkpoints" / (name + "_full")).expanduser().resolve()
    if dataset == output or dataset in output.parents or output in dataset.parents:
        raise ValueError("Dataset and model output must be separate directories")
    return dataset, output


def validate_existing_plan(dataset, args):
    path = dataset / "plan.json"
    if not path.exists():
        if dataset.exists() and any(dataset.iterdir()):
            raise ValueError("Dataset directory is nonempty but has no plan.json; use a new path")
        return None
    plan = read_json(path)
    expected = {
        "kind": (plan.get("kind"), "random"),
        "episodes": (len(plan.get("episodes", [])), args.episodes),
        "seed": (plan["config"]["seed"], args.seed),
        "amplitude": (plan["collection"]["amplitude"], args.amplitude),
        "collect-batch-size": (plan["config"]["runtime"]["batch_size"], args.collect_batch_size),
        "num-threads": (plan["config"]["runtime"]["num_threads"], args.num_threads),
        "device": (plan["config"]["runtime"]["device"], args.device),
        "goal-range": (plan["collection"]["goal_distribution"]["articulation_rad"], [.5, 1.5]),
        "horizon": (plan["collection"]["horizon"], 20),
    }
    differences = {name: values for name, values in expected.items() if values[0] != values[1]}
    if differences:
        raise ValueError("Existing immutable plan differs from requested options (saved, requested): {}. "
                         "Use matching options or new dataset/output paths.".format(differences))
    return plan


def commands_for(args, dataset, output):
    python = sys.executable
    plan = [python, "-u", str(HERE / "collect.py"), "plan", "--kind", "random",
            "--episodes", str(args.episodes), "--amplitude", str(args.amplitude),
            "--seed", str(args.seed), "--batch-size", str(args.collect_batch_size),
            "--num-threads", str(args.num_threads), "--device", args.device,
            "--output", str(dataset)]
    collect = [python, "-u", str(HERE / "collect.py"), "run", "--output", str(dataset)]
    audit = [python, "-u", str(HERE / "audit_collection.py"), "--dataset", str(dataset)]
    train = [python, "-u", str(HERE / "train_model.py"), "--dataset", str(dataset),
             "--output", str(output), "--epochs", str(args.epochs), "--patience", str(args.patience),
             "--batch-size", str(args.train_batch_size), "--learning-rate", str(args.learning_rate),
             "--seed", str(args.seed), "--device", args.device,
             "--hidden-dim", str(args.hidden_dim), "--layers", str(args.layers)]
    if (output / "last.pt").exists():
        train.append("--resume")
    return plan, collect, audit, train


def group_alive(process):
    # Reap the direct child, but also check workers that outlive that child.
    process.poll()
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return False
    return True


def stop_group(process):
    """Collector starts a worker grandchild; interrupt the entire owned group."""
    for signum, grace in ((signal.SIGINT, 8), (signal.SIGTERM, 3), (signal.SIGKILL, 1)):
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            process.wait()
            return
        deadline = time.monotonic() + grace
        while group_alive(process):
            if time.monotonic() >= deadline:
                break
            time.sleep(.1)
        else:
            process.wait()
            return
    process.wait(timeout=3)


def run_child(command, log_path, on_line=None, echo=True):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n{} COMMAND {}\n".format(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), shlex.join(command)))
        log.flush()
        process = subprocess.Popen(command, cwd=str(ROOT), stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, universal_newlines=True,
                                   bufsize=1, start_new_session=True)
        recent = []
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                recent.append(line.rstrip())
                recent = recent[-20:]
                if echo:
                    print(line, end="", flush=True)
                if on_line is not None:
                    on_line(line)
            status = process.wait()
        except BaseException:
            stop_group(process)
            raise
        finally:
            process.stdout.close()
    if status != 0:
        if not echo:
            print("\n" + "\n".join(recent), file=sys.stderr, flush=True)
        raise RuntimeError("Stage failed (exit {}); see {}".format(status, log_path))


class CollectionProgress:
    def __init__(self, dataset, plan, stream=None):
        self.dataset = dataset
        self.count = len(plan["episodes"])
        size = plan["config"]["runtime"]["batch_size"]
        self.sizes = [min(size, self.count - i) for i in range(0, self.count, size)]
        self.steps = plan["collection"]["total_steps"]
        # Only batch manifests are completion markers. Orphan NPZs are ignored.
        self.committed = set()
        for index in range(len(self.sizes)):
            path = dataset / "batches" / "batch_{:04d}.json".format(index)
            if path.exists():
                marker = read_json(path)
                if marker.get("complete"):
                    self.committed.add(index)
        self.active = None
        self.bar = Progress(self.count * self.steps, "Collect", "control steps", stream=stream)
        self.bar.update(self.saved_steps(), "existing batches will be verified; {} trials".format(self.count))

    def saved_steps(self):
        return sum(self.sizes[i] for i in self.committed) * self.steps

    def line(self, line):
        match = re.search(r"Running batch (\d+)/(\d+) \((\d+) episodes\)", line)
        if match:
            self.active = int(match.group(1)) - 1
            self.bar.update(self.saved_steps(), "batch {}/{} preparing".format(match.group(1), match.group(2)))
        match = re.search(r"(\d+)/(\d+) steps", line)
        if match and self.active is not None:
            completed = self.saved_steps() + int(match.group(1)) * self.sizes[self.active]
            self.bar.update(completed, "batch {}/{}; saved {}/{} trials".format(
                self.active + 1, len(self.sizes), self.saved_steps() // self.steps, self.count))
        match = re.search(r"Committed batch (\d+):", line)
        if match:
            self.committed.add(int(match.group(1)))
            self.active = None
            self.bar.update(self.saved_steps(), "saved {}/{} trials".format(self.saved_steps() // self.steps, self.count))

    def finish(self):
        manifest = read_json(self.dataset / "dataset.json")
        if not manifest.get("complete") or len(manifest["episodes"]) != self.count:
            raise ValueError("Collector exited without a complete dataset")
        self.bar.update(self.count * self.steps, "complete; {} valid windows".format(manifest["valid_windows"]))
        self.bar.close()


def run_pipeline(args):
    if args.episodes < 2 or args.seed < 0 or args.patience < 0:
        raise ValueError("episodes >= 2, seed >= 0 and patience >= 0 are required")
    import math
    if not math.isfinite(args.amplitude) or not math.isfinite(args.learning_rate) or min(args.amplitude, args.learning_rate) <= 0:
        raise ValueError("amplitude and learning-rate must be positive and finite")
    dataset, output = paths_for(args)
    plan = validate_existing_plan(dataset, args)
    if output.exists() and any(output.iterdir()) and not (output / "last.pt").exists():
        raise ValueError("Model output is nonempty without last.pt; select a new --output")
    commands = commands_for(args, dataset, output)
    print("Dataset: {}\nModel output: {}".format(dataset, output), flush=True)
    print("Collection goal angles: 0.5..1.5 rad | {} episodes | fit up to {} epochs".format(args.episodes, args.epochs), flush=True)
    if args.dry_run:
        print("DRY RUN: no files, simulation, or training will be started.")
        for index, command in enumerate(commands):
            if index == 0 and plan is not None:
                print("Reuse existing immutable plan (collector verifies sources/assets).")
            else:
                print(shlex.join(command))
        return
    with workflow_lock(dataset):
        return execute_workflow(args, dataset, output, plan, commands)


def execute_workflow(args, dataset, output, plan, commands):
    # Keep pipeline logs OUTSIDE the collection path so plan can create it.
    log_dir = HERE / "logs" / (dataset.name + "__" + output.name)
    print("Logs: {}".format(log_dir), flush=True)
    print("[1/3] Collecting physical responses. Completed batches are reused.", flush=True)
    if plan is None:
        run_child(commands[0], log_dir / "plan.log")
        plan = read_json(dataset / "plan.json")
    display = CollectionProgress(dataset, plan)
    try:
        run_child(commands[1], log_dir / "collection.log", on_line=display.line, echo=False)
        display.finish()
    except BaseException:
        display.bar.close("stopped; rerun the same command to verify/reuse completed batches")
        raise
    print("[2/3] Verifying recorded actions, response windows and reference splits.", flush=True)
    run_child(commands[2], log_dir / "audit.log")
    print("[3/3] Supervised training. Watch validation loss as well as training loss.", flush=True)
    # Run fitting in this process so its progress bar sees the user's real TTY.
    from train_model import train_model
    report = train_model(dataset, output, epochs=args.epochs, patience=args.patience,
                         batch_size=args.train_batch_size, learning_rate=args.learning_rate,
                         seed=args.seed, device=args.device, hidden_dim=args.hidden_dim,
                         layers=args.layers, resume=(output / "last.pt").exists())
    print("Finished. Results: {}".format(output / "report.json"), flush=True)
    return report


def main():
    args = parser_for_cli().parse_args()
    try:
        run_pipeline(args)
    except KeyboardInterrupt:
        print("\nStopped. Completed data batches and epoch checkpoints are retained.", file=sys.stderr)
        raise SystemExit(130)
    except (ValueError, RuntimeError, OSError) as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
