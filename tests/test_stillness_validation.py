import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from behavision.behavior.stillness_validation import (
    frame_indices, load_manifest, main, parse_prediction, summarize,
)


class StillnessValidationTests(unittest.TestCase):
    def test_sampling_stays_in_half_open_window_and_includes_edges(self):
        indices = frame_indices(0.15, 2.15, 30, 300, 8)
        self.assertEqual(len(indices), 16)
        self.assertEqual((indices[0], indices[-1]), (5, 64))
        self.assertEqual(indices, sorted(set(indices)))
        self.assertTrue(all(0.15 <= i / 30 < 2.15 for i in indices))

    def test_low_fps_does_not_repeat_frames(self):
        self.assertEqual(frame_indices(0, 3, 2, 10, 8), list(range(6)))

    def test_bad_video_or_window_is_rejected(self):
        for args in [(0, 1, 0, 30, 8), (0, 2, 30, 30, 8), (0, .01, 30, 30, 8)]:
            with self.assertRaises(ValueError):
                frame_indices(*args)

    def test_strict_response_validation(self):
        self.assertEqual(parse_prediction('{"label":"ACTIVE","reason":"Head turns"}')["label"], "ACTIVE")
        for raw in ['[]', '{}', '{"label":"still","reason":"x"}', '{"label":"STILL","reason":""}', 'not JSON']:
            with self.assertRaises(ValueError):
                parse_prediction(raw)

    def test_errors_and_unlabeled_rows_are_not_hidden_in_metrics(self):
        rows = [
            dict(clip_id="a", expected_label="STILL", label="STILL", status="ok"),
            dict(clip_id="b", expected_label="ACTIVE", label="STILL", status="ok"),
            dict(clip_id="c", expected_label="STILL", label=None, status="error"),
            dict(clip_id="d", expected_label="", label="STILL", status="ok"),
        ]
        summary = summarize(rows)
        self.assertEqual(summary["still_precision"], .5)
        self.assertEqual(summary["still_recall_including_errors"], .5)
        self.assertEqual(summary["agreement_including_errors"], 1 / 3)
        self.assertEqual(summary["false_still_clip_ids"], ["b"])

    def write_manifest(self, root):
        (root / "video.avi").touch()
        manifest = root / "clips.csv"
        with manifest.open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(["clip_id", "video_path", "start_s", "end_s", "expected_label"])
            writer.writerow(["clip01", "video.avi", "0", "3", "ACTIVE"])
        return manifest

    def test_manifest_relative_paths_and_nonfinite_times(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = self.write_manifest(root)
            self.assertEqual(load_manifest(path)[0]["video_path"], str((root / "video.avi").resolve()))
            path.write_text(path.read_text().replace(",0,3,", ",nan,3,"))
            with self.assertRaises(ValueError):
                load_manifest(path)

    def test_mocked_end_to_end_preserves_errors_raw_text_and_prompt_separation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self.write_manifest(root)
            output = root / "output"
            responses = [{"models": [{"name": "qwen3-vl:4b", "digest": "test"}]},
                         {"message": {"content": "bad model response"}}]
            with patch("behavision.behavior.stillness_validation.sample_clip", return_value=(["image"], [0, 1])), \
                 patch("behavision.behavior.stillness_validation.local_request", side_effect=responses) as request:
                self.assertEqual(main(["--manifest", str(manifest), "--output-dir", str(output)]), 1)
            prompt = request.call_args.args[3]["messages"][0]["content"]
            self.assertNotIn("expected_label", prompt)
            result = json.loads((output / "predictions.jsonl").read_text())
            self.assertEqual(result["raw_response"], "bad model response")
            self.assertEqual(result["status"], "error")
            self.assertIsNone(result["label"])

    def test_successful_classification_and_existing_output_protection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self.write_manifest(root)
            output = root / "output"
            responses = [{"models": [{"name": "qwen3-vl:4b", "digest": "test"}]},
                         {"message": {"content": '{"label":"ACTIVE","reason":"Head turns"}'}}]
            with patch("behavision.behavior.stillness_validation.sample_clip", return_value=(["image"], [0, 1])), \
                 patch("behavision.behavior.stillness_validation.local_request", side_effect=responses):
                self.assertEqual(main(["--manifest", str(manifest), "--output-dir", str(output)]), 0)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["agreement_including_errors"], 1)
            self.assertIsNone(summary["still_precision"])
            config = json.loads((output / "config.json").read_text())
            self.assertEqual(config["model_info"]["digest"], "test")
            original = (output / "predictions.csv").read_bytes()
            with self.assertRaises(SystemExit) as exit_status:
                main(["--manifest", str(manifest), "--output-dir", str(output), "--dry-run"])
            self.assertEqual(exit_status.exception.code, 2)
            self.assertEqual((output / "predictions.csv").read_bytes(), original)

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV not installed")
    def test_real_video_decoding_dry_run(self):
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self.write_manifest(root)
            writer = cv2.VideoWriter(str(root / "video.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 64))
            self.assertTrue(writer.isOpened())
            for index in range(30):
                writer.write(np.full((64, 64, 3), index * 7, dtype=np.uint8))
            writer.release()
            output = root / "output"
            with patch("behavision.behavior.stillness_validation.local_request") as request:
                self.assertEqual(main(["--manifest", str(manifest), "--output-dir", str(output), "--dry-run"]), 0)
                request.assert_not_called()
            self.assertEqual(len(list((output / "frames" / "clip01").glob("*.jpg"))), 24)
            summary = json.loads((output / "summary.json").read_text())
            self.assertTrue(summary["dry_run"])
            self.assertNotIn("still_precision", summary)


if __name__ == "__main__":
    unittest.main()
