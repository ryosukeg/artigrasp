"""Progress, immutable collection settings, failure gating and dry-run checks."""
import contextlib
import io
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from progress import Progress
import run_training as workflow


class Terminal(io.StringIO):
    def isatty(self):
        return True


def plan_for(episodes=2):
    return {"kind": "random", "config": {"seed": 301, "runtime": {
        "batch_size": 20, "num_threads": 4, "device": "cuda"}},
        "episodes": [{"episode_id": str(i)} for i in range(episodes)],
        "collection": {"amplitude": 1., "horizon": 20, "total_steps": 1560,
                       "goal_distribution": {"articulation_rad": [.5, 1.5]}}}


class ProgressTests(unittest.TestCase):
    def test_terminal_bar_and_early_stop_keep_actual_fraction(self):
        stream = Terminal()
        progress = Progress(100, "Epoch", "epochs", stream=stream)
        progress.update(0)
        progress.update(12, "train=0.5 val=0.8")
        progress.close("early stopped")
        text = stream.getvalue()
        self.assertIn("\rEpoch [", text)
        self.assertIn("12/100", text)
        self.assertIn("train=0.5 val=0.8", text)
        self.assertNotIn("100.0%", text)
        self.assertTrue(text.endswith("\n"))

    def test_resume_eta_excludes_already_completed_work(self):
        stream = io.StringIO()
        with mock.patch("progress.time.monotonic", side_effect=[0., 10., 12.]):
            progress = Progress(10, "Epoch", "epochs", stream=stream)
            progress.update(5)
            progress.update(6)
        self.assertIn("ETA 00:08", stream.getvalue())
        self.assertNotIn("\r", stream.getvalue())


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="box_training_workflow_")
        self.root = Path(self.directory.name)
        self.dataset = self.root / "dataset"
        self.output = self.root / "output"
        self.args = workflow.parser_for_cli().parse_args([
            "--episodes", "2", "--dataset", str(self.dataset), "--output", str(self.output)])

    def tearDown(self):
        self.directory.cleanup()

    def save_plan(self):
        self.dataset.mkdir()
        (self.dataset / "plan.json").write_text(json.dumps(plan_for()))

    def test_dry_run_creates_nothing_and_starts_no_children(self):
        self.args.dry_run = True
        with contextlib.redirect_stdout(io.StringIO()) as stream, mock.patch.object(workflow, "run_child") as child:
            workflow.run_pipeline(self.args)
        child.assert_not_called()
        self.assertIn("DRY RUN", stream.getvalue())
        self.assertIn("train_model.py", stream.getvalue())
        self.assertFalse(self.dataset.exists())
        self.assertFalse(self.output.exists())

    def test_existing_plan_mismatch_rejected_without_modification(self):
        self.save_plan()
        before = (self.dataset / "plan.json").read_bytes()
        self.args.collect_batch_size = 8
        with self.assertRaisesRegex(ValueError, "collect-batch-size"):
            workflow.validate_existing_plan(self.dataset, self.args)
        self.assertEqual((self.dataset / "plan.json").read_bytes(), before)

    def test_no_plan_in_nonempty_dataset_is_rejected(self):
        self.dataset.mkdir()
        (self.dataset / "orphan.npz").write_bytes(b"incomplete")
        with self.assertRaisesRegex(ValueError, "no plan.json"):
            workflow.validate_existing_plan(self.dataset, self.args)

    def test_completion_uses_batch_markers_not_orphan_trajectories(self):
        self.save_plan()
        (self.dataset / "trajectories").mkdir()
        (self.dataset / "trajectories/orphan.npz").write_bytes(b"incomplete")
        display = workflow.CollectionProgress(self.dataset, plan_for(), stream=io.StringIO())
        self.assertEqual(display.saved_steps(), 0)
        display.line("Running batch 1/1 (2 episodes)")
        display.line(" 500/1560 steps | 1.0 s")
        self.assertEqual(display.bar.completed, 1000)
        display.line("Committed batch 0: 2 episodes")
        self.assertEqual(display.saved_steps(), 3120)
        with self.assertRaises(FileNotFoundError):
            display.finish()

    def test_different_output_directory_required(self):
        self.args.output = self.dataset / "model"
        with self.assertRaisesRegex(ValueError, "separate"):
            workflow.paths_for(self.args)

    def test_audit_failure_prevents_training(self):
        self.save_plan()
        fake_train = mock.Mock()
        def child(command, *_args, **_kwargs):
            if any(value.endswith("audit_collection.py") for value in command):
                raise RuntimeError("audit failed")
            (self.dataset / "dataset.json").write_text(json.dumps({
                "complete": True, "episodes": plan_for()["episodes"], "valid_windows": 30}))
        with mock.patch.object(workflow, "HERE", self.root), mock.patch.object(workflow, "run_child", side_effect=child), \
             mock.patch.dict(sys.modules, {"train_model": types.SimpleNamespace(train_model=fake_train)}), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "audit failed"):
                workflow.run_pipeline(self.args)
        fake_train.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_same_dataset_cannot_run_twice_concurrently(self):
        with mock.patch.object(workflow, "HERE", self.root):
            with workflow.workflow_lock(self.dataset):
                with self.assertRaisesRegex(RuntimeError, "already running"):
                    with workflow.workflow_lock(self.dataset):
                        pass
            with workflow.workflow_lock(self.dataset):
                pass

    def test_interrupt_signals_whole_collector_process_group(self):
        process = mock.Mock(pid=12345)
        # The parent already exited, but its worker still belongs to this group.
        process.poll.return_value = 130
        process.wait.return_value = 130
        with mock.patch.object(workflow.os, "killpg") as kill, \
             mock.patch.object(workflow, "group_alive", side_effect=[True, False]), \
             mock.patch.object(workflow.time, "monotonic", side_effect=[0., 9., 9.]):
            workflow.stop_group(process)
        self.assertEqual(kill.call_args_list, [mock.call(12345, signal.SIGINT), mock.call(12345, signal.SIGTERM)])

    def test_group_lookup_race_is_harmless(self):
        process = mock.Mock(pid=12345)
        with mock.patch.object(workflow.os, "killpg", side_effect=ProcessLookupError):
            workflow.stop_group(process)
        process.wait.assert_called_once()


if __name__ == "__main__":
    unittest.main()
