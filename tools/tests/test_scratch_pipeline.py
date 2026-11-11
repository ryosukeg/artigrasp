"""Fail-closed scratch-stage orchestration, without launching training."""

from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("train_scratch", ROOT / "scripts/train_scratch.py")
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


class ScratchPipelineTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.config_path = self.root / "config.json"
        config = json.loads(pipeline.DEFAULT_CONFIG.read_text())
        # Minimal valid budgets keep these orchestration tests independent of
        # the long-run schedule while preserving the eight-object task counts.
        for stage in config["stages"]:
            stage["iterations"] = 2
            stage["environments"] = 8 if stage["id"] == "left_fixed" else 16
        self.config_path.write_text(json.dumps(config))
        self.config = pipeline.load_config(self.config_path)

    def plan(self, smoke=False):
        return pipeline.build_plan(self.config, self.config_path, self.root / "run", seed=17,
                                   smoke=smoke, python=sys.executable, eval_seed=91,
                                   eval_samples_per_object=13)

    def execute(self, plan):
        with redirect_stdout(io.StringIO()):
            return pipeline.execute_plan(plan)

    def write_stage_result(self, stage_spec, **overrides):
        stage = stage_spec
        directory = Path(stage["output_dir"])
        directory.mkdir(parents=True, exist_ok=True)
        checkpoints = {}
        for hand in stage["required_hands"]:
            suffix = "l" if hand == "left" else "r"
            path = directory / "full_{}_{}.pt".format(stage["iterations"], suffix)
            path.write_bytes((stage["id"] + ":" + hand).encode())
            checkpoints[hand] = str(path)
        result = {"complete": True, "stage": stage["id"], "seed": stage.get("seed", 17),
                  "objects": stage.get("objects", list(pipeline.PAPER_OBJECTS)),
                  "num_envs": stage["environments"], "device": "cpu",
                  "checkpoint_timing": "after_completed_ppo_updates",
                  "iterations_completed": stage["iterations"], "checkpoints": checkpoints}
        result.update(overrides)
        (directory / "training_result.json").write_text(json.dumps(result))
        return result

    def test_dry_run_cli_emits_a_plan_without_creating_output(self):
        output = self.root / "dry-run"
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/train_scratch.py"), "--config",
             str(self.config_path), "--output", str(output), "--smoke", "--seed", "17"],
            capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["training_seed"], 17)
        self.assertFalse(output.exists())
        self.assertEqual([s["id"] for s in plan["stages"]], list(pipeline.STAGE_ORDER))

    def test_smoke_uses_all_eight_objects_and_keeps_train_eval_seeds_separate(self):
        plan = self.plan(smoke=True)
        self.assertEqual(set(plan["objects"]), set(pipeline.PAPER_OBJECTS))
        self.assertEqual(plan["training_seed"], 17)
        self.assertEqual(plan["evaluation"]["seed"], 91)
        self.assertEqual(plan["runtime"]["device"], "cpu")
        for stage in plan["stages"]:
            self.assertEqual(stage["iterations"], 1)
            self.assertEqual(stage["environments"], 8 if stage["id"] == "left_fixed" else 16)
            command = stage["command"]
            offset = command.index("--objects") + 1
            self.assertEqual(command[offset:offset + 8], list(pipeline.PAPER_OBJECTS))
            self.assertIn("--headless", command)
            self.assertEqual(command[command.index("--seed") + 1], "17")

    def test_training_configuration_rejects_test_split_and_nonpaper_objects(self):
        for changes in ({"split": "test"}, {"objects": ["phone"]},
                        {"objects": ["box", "box"]}):
            config = deepcopy(self.config)
            config.update(changes)
            self.config_path.write_text(json.dumps(config))
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                pipeline.load_config(self.config_path)

    def test_default_schedule_allocates_whole_batches_to_each_object_and_task(self):
        config = pipeline.load_config(pipeline.DEFAULT_CONFIG)
        self.assertEqual(set(config["objects"]), set(pipeline.PAPER_OBJECTS))
        for stage in config["stages"]:
            divisor = 8 if stage["id"] == "left_fixed" else 16
            self.assertEqual(stage["environments"] % divisor, 0)

    def test_shared_openmp_runtime_rejects_mismatched_torch_and_physics_threads(self):
        config = deepcopy(self.config)
        config["runtime"].update(num_threads=32, torch_threads=1)
        self.config_path.write_text(json.dumps(config))
        with self.assertRaises(ValueError):
            pipeline.load_config(self.config_path)

    def test_even_an_empty_existing_output_is_refused_before_any_stage(self):
        plan = self.plan()
        Path(plan["output_dir"]).mkdir()
        with mock.patch.object(pipeline, "run_stage") as run:
            with self.assertRaisesRegex(ValueError, "already exists"):
                self.execute(plan)
        run.assert_not_called()

    def test_source_change_is_refused_before_creating_any_run_artifacts(self):
        plan = self.plan()
        self.config_path.write_text(self.config_path.read_text() + "\n")
        with mock.patch.object(pipeline, "run_stage") as run:
            with self.assertRaisesRegex(ValueError, "source changed"):
                self.execute(plan)
        run.assert_not_called()
        self.assertFalse(Path(plan["output_dir"]).exists())

    def test_failed_stage_stops_before_downstream_training_or_evaluation(self):
        plan = self.plan()
        with mock.patch.object(pipeline, "run_stage", return_value=9) as run:
            with self.assertRaisesRegex(ValueError, "Stage failed"):
                self.execute(plan)
        self.assertEqual(run.call_count, 1)
        execution = json.loads((Path(plan["output_dir"]) / "execution.json").read_text())
        self.assertFalse(execution["complete"])
        self.assertEqual(list(execution["stages"]), ["left_fixed"])
        self.assertFalse(Path(plan["evaluation"]["output_config"]).exists())

    def test_zero_exit_without_a_completion_record_is_a_failure(self):
        plan = self.plan()
        with mock.patch.object(pipeline, "run_stage", return_value=0) as run:
            with self.assertRaisesRegex(ValueError, "completion record"):
                self.execute(plan)
        self.assertEqual(run.call_count, 1)
        self.assertFalse(Path(plan["evaluation"]["output_config"]).exists())

    def test_source_edit_during_a_run_stops_before_the_next_stage(self):
        plan = self.plan()

        def changed_source(command, log_file, cwd, environment):
            self.write_stage_result(plan["stages"][0])
            self.config_path.write_text(self.config_path.read_text() + "\n")
            return 0

        with mock.patch.object(pipeline, "run_stage", side_effect=changed_source) as run:
            with self.assertRaisesRegex(ValueError, "source changed"):
                self.execute(plan)
        self.assertEqual(run.call_count, 1)
        self.assertFalse(Path(plan["evaluation"]["output_config"]).exists())

    def test_incomplete_or_short_stage_is_not_accepted_as_pretraining(self):
        stage = self.plan()["stages"][0]
        for overrides in ({"complete": False}, {"iterations_completed": 1},
                          {"iterations_completed": True}):
            self.write_stage_result(stage, **overrides)
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                pipeline.read_stage_result(stage)

    def test_completion_must_match_the_planned_stage_seed_objects_and_batch(self):
        stage = self.plan()["stages"][0]
        changes = ({"stage": "general_two"}, {"seed": 99}, {"objects": ["box"]},
                   {"num_envs": 1}, {"checkpoint_timing": "before_update"}, {"device": "auto"})
        for overrides in changes:
            self.write_stage_result(stage, **overrides)
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                pipeline.read_stage_result(stage)

    def test_missing_or_foreign_checkpoint_cannot_be_used_as_the_stage_result(self):
        stage = self.plan()["stages"][0]
        foreign = self.root / "public_weights.pt"
        foreign.write_bytes(b"foreign")
        for path in (str(foreign), str(Path(stage["output_dir"]) / "missing.pt")):
            self.write_stage_result(stage, checkpoints={"left": path})
            with self.subTest(path=path), self.assertRaises(ValueError):
                pipeline.read_stage_result(stage)

    def test_auto_training_device_becomes_a_concrete_evaluation_device(self):
        plan = self.plan()
        self.assertEqual(plan["runtime"]["device"], "auto")
        completed = {}
        for stage in plan["stages"]:
            self.write_stage_result(stage, device="cuda")
            completed[stage["id"]] = pipeline.read_stage_result(stage)
        self.assertEqual(pipeline.make_evaluation_config(plan, completed)["runtime"]["device"], "cuda")
        completed["general_two"]["device"] = None
        self.assertEqual(pipeline.make_evaluation_config(plan, completed)["runtime"]["device"], "cpu")

    def test_cooperative_stage_consumes_this_run_final_pretraining_checkpoints(self):
        plan = self.plan(smoke=True)
        stages = {stage["id"]: stage for stage in plan["stages"]}
        commands = []

        def fake_stage(command, log_file, cwd, environment):
            name = Path(command[1]).parent.name
            commands.append(command)
            self.assertEqual(cwd, str(ROOT))
            self.assertEqual(environment["OMP_NUM_THREADS"], "1")
            self.assertEqual(environment["PYTHONHASHSEED"], "17")
            self.write_stage_result(stages[name])
            return 0

        with mock.patch.object(pipeline, "run_stage", side_effect=fake_stage):
            result = self.execute(plan)
        self.assertTrue(result["complete"])
        output = Path(plan["output_dir"])
        saved_plan = json.loads((output / "training_plan.json").read_text())
        self.assertEqual(set(saved_plan["source_snapshots"]), set(plan["source_sha256"]))
        for source, snapshot in saved_plan["source_snapshots"].items():
            self.assertEqual(pipeline.sha256(output / snapshot), plan["source_sha256"][source])
        self.assertEqual(len(commands), 3)
        cooperative = commands[-1]
        left = cooperative[cooperative.index("--pretrained-left") + 1]
        right = cooperative[cooperative.index("--pretrained-right") + 1]
        self.assertEqual(left, str(Path(stages["left_fixed"]["output_dir"]) / "full_1_l.pt"))
        self.assertEqual(right, str(Path(stages["multi_obj_arti"]["output_dir"]) / "full_1_r.pt"))
        self.assertNotIn(pipeline.LEFT_INPUT, cooperative)
        self.assertNotIn(pipeline.RIGHT_INPUT, cooperative)
        evaluation = json.loads(Path(plan["evaluation"]["output_config"]).read_text())
        self.assertEqual(evaluation["seed"], 91)
        self.assertEqual(evaluation["split"], "test")
        self.assertEqual(evaluation["samples_per_object"], 13)
        self.assertEqual(evaluation["objects"], list(pipeline.PAPER_OBJECTS))
        self.assertEqual(evaluation["checkpoints"]["articulation_right"], right)
        for hand in ("left", "right"):
            key = "cooperative_" + hand
            self.assertEqual(Path(evaluation["checkpoints"][key]).parent,
                             Path(stages["general_two"]["output_dir"]))


if __name__ == "__main__":
    unittest.main()
