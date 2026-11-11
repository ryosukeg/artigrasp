"""Tests for selecting and validating actual Unity video replays, without RaiSim."""

import json
from pathlib import Path
import subprocess
import signal
import tempfile
import unittest
from unittest import mock

from tools import ood_record


def episode(episode_id, success, index=0, valid=True, object_name="box", profile="ood_all"):
    return {"episode_id": episode_id, "sample_index": index, "object": object_name,
            "profile": profile, "valid": valid,
            "paper_success": success, "public_success": not success}


class SelectionTests(unittest.TestCase):
    def test_selection_is_stable_and_does_not_fill_shortages_from_other_outcomes(self):
        rows = [episode("late", True, 3), episode("failure", False, 2),
                episode("first", True, 0), episode("invalid", False, 1, valid=False)]
        selected = ood_record.select_episodes(rows, per_outcome=5)
        self.assertEqual([row["episode_id"] for row in selected[0]["selected"]], ["first", "late"])
        self.assertEqual([row["episode_id"] for row in selected[1]["selected"]], ["failure"])
        self.assertEqual(selected[0]["shortage"], 3)
        self.assertEqual(selected[1]["shortage"], 4)
        self.assertEqual(selected[1]["invalid_excluded"], 1)

    def test_public_metric_changes_selection_and_object_filter_is_respected(self):
        rows = [episode("paper_success", True), episode("public_success", False),
                episode("other", False, object_name="mixer")]
        selected = ood_record.select_episodes(rows, metric="public_success", objects=["box"])
        self.assertEqual([row["episode_id"] for row in selected[0]["selected"]], ["public_success"])
        self.assertEqual(len(selected), 2)

    def test_duplicate_episode_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.jsonl"
            path.write_text("\n".join(json.dumps(episode("duplicate", True)) for _ in range(2)))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                ood_record.load_episodes(path)

    def test_filtered_update_retains_other_object_and_profile_indexes(self):
        rows = [episode("box_lift", True, profile="lift"),
                episode("box_all", True), episode("mixer_all", False, object_name="mixer")]
        previous = {"selection": [], "videos": []}
        for row in rows:
            previous["selection"].append({"object": row["object"], "profile": row["profile"],
                                           "episode_ids": [row["episode_id"]]})
            previous["videos"].append(dict(row, verified=True))
        selected = ood_record.select_episodes(rows, objects=["box"], profiles=["lift"])
        retained_selection, retained_videos = ood_record.retain_unselected_groups(previous, selected, rows)
        self.assertEqual({entry["episode_id"] for entry in retained_videos}, {"box_all", "mixer_all"})
        self.assertEqual({(group["object"], group["profile"]) for group in retained_selection},
                         {("box", "ood_all"), ("mixer", "ood_all")})
        # Updating all evaluated groups replaces every old index entry.
        selections = ood_record.select_episodes(rows)
        self.assertEqual(ood_record.retain_unselected_groups(previous, selections, rows), ([], []))


class VideoValidationTests(unittest.TestCase):
    def test_capture_error_details_follow_the_entire_video_table(self):
        failed = dict(episode("failed", False), outcome="failure", verified=False, error="capture failed")
        succeeded = dict(episode("succeeded", True), outcome="success", verified=True, relative_path="ok.mp4")
        text = ood_record.markdown_index({"metric": "paper_success", "selection": [],
                                          "videos": [failed, succeeded]})
        lines = text.splitlines()
        failed_row = next(i for i, line in enumerate(lines) if "| failed |" in line)
        succeeded_row = next(i for i, line in enumerate(lines) if "| succeeded |" in line)
        error_detail = next(i for i, line in enumerate(lines) if line.startswith("- `failed`"))
        self.assertEqual(succeeded_row, failed_row + 1)
        self.assertGreater(error_detail, succeeded_row)

    def test_replay_cannot_be_misclassified_or_divergent(self):
        original = episode("box_0", True)
        replay = {"episode_id": "box_0", "verified": True,
                  "paper_success": True, "public_success": False}
        ood_record.verify_replay(original, replay)
        for update in ({"verified": False}, {"episode_id": "box_1"}, {"public_success": True}):
            with self.assertRaises(ValueError):
                ood_record.verify_replay(original, dict(replay, **update))

    def test_capture_search_uses_unique_filename_not_any_mp4(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "unrelated.mp4").write_bytes(b"unrelated")
            self.assertIsNone(ood_record.find_capture([root], "requested.mp4"))
            timestamp = root / "timestamp"
            timestamp.mkdir()
            expected = timestamp / "requested.mp4"
            expected.write_bytes(b"video")
            self.assertEqual(ood_record.find_capture([root], "requested.mp4"), expected)

    def test_video_stream_and_duration_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.mp4"
            path.write_bytes(b"real bytes are checked by ffprobe")
            valid = {"streams": [{"codec_type": "video", "width": 1920, "height": 1080}],
                     "format": {"duration": "40.0"}}
            with mock.patch.object(ood_record.subprocess, "run", return_value=subprocess.CompletedProcess(
                    [], 0, stdout=json.dumps(valid), stderr="")):
                self.assertEqual(ood_record.probe_video(path, minimum_duration=30)["duration_seconds"], 40)
                with self.assertRaisesRegex(ValueError, "too short"):
                    ood_record.probe_video(path, minimum_duration=50)
            with mock.patch.object(ood_record.subprocess, "run", return_value=subprocess.CompletedProcess(
                    [], 0, stdout=json.dumps({"streams": [], "format": {"duration": "40"}}), stderr="")):
                with self.assertRaisesRegex(ValueError, "no video stream"):
                    ood_record.probe_video(path)

    def test_safe_path_component_prevents_traversal_and_preserves_valid_ids(self):
        self.assertEqual(ood_record.safe_component("box_ood_0001"), "box_ood_0001")
        component = ood_record.safe_component("../../outside")
        self.assertNotIn("/", component)
        self.assertFalse(component.startswith("."))

    def test_timeout_interrupts_replay_before_forced_kill_so_video_can_stop(self):
        process = mock.Mock()
        process.wait.side_effect = [subprocess.TimeoutExpired(["replay"], 1), 130]
        process.poll.return_value = None
        with mock.patch.object(ood_record.subprocess, "Popen", return_value=process):
            with self.assertRaises(subprocess.TimeoutExpired):
                ood_record.run_replay(["replay"], ".", None, 1)
        process.send_signal.assert_called_once_with(signal.SIGINT)
        process.kill.assert_not_called()


if __name__ == "__main__":
    unittest.main()
