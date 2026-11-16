import copy
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from behavision.behavior import activity
from behavision.config import DEFAULTS, load_config
from behavision.detection.postprocess import drop_excluded
from behavision.detection.types import Detection
from behavision.pipeline.inactivity import MouseTracker, box_iou, mask_box
from behavision.segmentation.sam2_video import TrackedMask


def square(x, y, size=12, shape=(60, 80)):
    mask = np.zeros(shape, bool)
    mask[y:y + size, x:x + size] = True
    return mask


class FrameMotionTests(unittest.TestCase):
    def test_moving_square_changes_roi_but_not_background(self):
        a, b = square(10, 10), square(14, 10)
        prev, cur = np.where(a, 0, 200).astype(np.uint8), np.where(b, 0, 200).astype(np.uint8)
        row = activity.frame_motion(prev, cur, a, b, dilation_px=1, background_margin_px=5)
        self.assertGreater(row["chg_d20"], 0)
        self.assertEqual(row["bg_d4"], 0.0)
        self.assertEqual(row["bg_p999"], 0)

    def test_identical_frames_have_no_motion(self):
        a = square(10, 10)
        frame = np.where(a, 0, 200).astype(np.uint8)
        row = activity.frame_motion(frame, frame, a, a, 1, 5)
        self.assertEqual(row["chg_d4"], 0)


class StateTests(unittest.TestCase):
    def test_runs(self):
        self.assertEqual(activity.runs([0, 1, 1, 0, 1]), [(1, 3), (4, 5)])
        self.assertEqual(activity.runs([]), [])

    def test_short_still_runs_are_active_and_uncertain_breaks_bouts(self):
        motion = np.array([0.0] * 5 + [0.5] + [0.0] * 2 + [0.0] * 3)
        uncertain = np.zeros(len(motion), bool)
        uncertain[8] = True
        states = activity.frame_states(motion, uncertain, fps=2, threshold=0.1, min_bout_s=2.0)
        self.assertEqual(list(states[:5]), ["inactive"] * 5)
        self.assertEqual(states[5], "active")
        self.assertEqual(list(states[6:8]), ["active"] * 2)  # 1 s still is shorter than the 2 s minimum
        self.assertEqual(states[8], "uncertain")

    def test_min_bout_frames(self):
        motion = np.zeros(3)
        states = activity.frame_states(motion, np.zeros(3, bool), fps=2, threshold=0.1, min_bout_s=2.0)
        self.assertEqual(list(states), ["active"] * 3)

    def test_otsu_separates_modes(self):
        rng = np.random.default_rng(0)
        values = np.concatenate([rng.uniform(0.0005, 0.002, 500), rng.uniform(0.1, 0.4, 500)])
        threshold = activity.otsu_threshold(values, (1e-5, 1.0))
        self.assertTrue(0.002 < threshold < 0.1)
        self.assertEqual(activity.otsu_threshold(values, (0.2, 0.3)), 0.2)

    def test_calibrate_delta_uses_background_99th_percentile(self):
        # 2% of background pixels exceed 4 and 6, 0.5% exceed 8: noise level 8.
        fractions = {4: 0.02, 6: 0.02, 8: 0.005}
        frame = pd.DataFrame({f"bg_d{d}": [fractions.get(d, 0.001)] * 3 for d in activity.DELTAS})
        self.assertEqual(activity.calibrate_delta(frame, 1.5), (12, 8))
        noisy = pd.DataFrame({f"bg_d{d}": [0.5] for d in activity.DELTAS})
        self.assertEqual(activity.calibrate_delta(noisy, 1.5), (40, 40))

    def test_smoothing_removes_single_frame_spikes(self):
        frame = pd.DataFrame({"roi_px": [100] * 5, "chg_d6": [0, 0, 50, 0, 0], "bg_d6": [0.0] * 5})
        self.assertEqual(activity.motion_fraction(frame, 6, 3).tolist(), [0.0] * 5)
        self.assertEqual(activity.motion_fraction(frame, 6, 1).tolist(), [0, 0, 0.5, 0, 0])


class AnalyzeTests(unittest.TestCase):
    def tracking(self):
        n, fps = 100, 10
        moving = np.r_[np.ones(30), np.zeros(50), np.ones(20)].astype(bool)
        rows = {"frame": np.arange(n), "time_s": np.arange(n) / fps, "tracked": True,
                "mask_area": 1000, "x": 1.0, "y": 1.0, "roi_px": 1000, "bg_p999": 4}
        for d in activity.DELTAS:
            rows[f"chg_d{d}"] = np.where(moving, 300, 2)
            rows[f"bg_d{d}"] = 0.001
        frame = pd.DataFrame(rows)
        frame.loc[0, ["roi_px"] + [f"chg_d{d}" for d in activity.DELTAS]] = np.nan
        return frame, fps

    def test_single_bout(self):
        tracking, fps = self.tracking()
        result = activity.analyze(tracking, fps, copy.deepcopy(DEFAULTS))
        bouts = result["bouts"]
        self.assertEqual(len(bouts), 1)
        self.assertEqual((bouts.loc[0, "start_frame"], bouts.loc[0, "end_frame"]), (30, 79))
        self.assertAlmostEqual(bouts.loc[0, "duration_s"], 5.0)
        summary = result["summary"]
        self.assertEqual(summary["uncertain_s"], 0.1)
        self.assertAlmostEqual(summary["inactive_s"], 5.0)
        self.assertEqual(summary["calibration"]["pixel_delta"], 6)
        self.assertEqual(len(result["sensitivity"]), 20)

    def test_area_outliers_are_uncertain(self):
        tracking, fps = self.tracking()
        tracking.loc[50, "mask_area"] = 100
        result = activity.analyze(tracking, fps, copy.deepcopy(DEFAULTS))
        self.assertEqual(result["timeline"].loc[50, "state"], "uncertain")
        self.assertEqual(len(result["bouts"]), 2)


class SmallUnitTests(unittest.TestCase):
    def test_drop_excluded(self):
        inside = Detection("m", 0.9, 290, 420, 390, 479)
        outside = Detection("m", 0.5, 100, 100, 150, 150)
        self.assertEqual(drop_excluded([inside, outside], [[280, 405, 405, 480]]), [outside])

    def test_box_helpers(self):
        self.assertEqual(mask_box(square(5, 6, 4)), (5.0, 6.0, 9.0, 10.0))
        self.assertIsNone(mask_box(np.zeros((4, 4), bool)))
        self.assertAlmostEqual(box_iou((0, 0, 2, 2), (1, 0, 3, 2)), 1 / 3)

    def test_unknown_config_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.yaml"
            path.write_text("activity:\n  min_bout: 2\n")
            with self.assertRaises(ValueError):
                load_config(path)


class FakeDetector:
    def __init__(self):
        self.calls = 0

    def detect(self, image, prompt):
        self.calls += 1
        ys, xs = np.nonzero(image[:, :, 0] < 60)
        return [Detection("m", 0.9, xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)] if xs.size else []


class FakeSegmenter:
    """Segments the dark square by thresholding each chunk frame.

    ``lose_after`` (SAM2 frame index) makes the first propagation return empty
    masks from that frame on, imitating a SAM2 failure.
    """

    def __init__(self, lose_after=None):
        self.seeds = []
        self.lose_after = lose_after
        self.resets = 0

    def reset(self, state):
        self.resets += 1

    def initialize(self, directory):
        return sorted(Path(directory).glob("*.jpg"))

    def add_mask(self, state, frame_index, mask, object_id=1):
        self.seeds.append(("mask", frame_index))

    def add_box(self, state, frame_index, box, object_id=1):
        self.seeds.append(("box", frame_index))

    def propagate(self, state, start_frame_index, max_frames=None):
        for i, path in enumerate(state[start_frame_index:], start_frame_index):
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) < 60
            if self.lose_after is not None and self.resets == 0 and i >= self.lose_after:
                mask = np.zeros_like(mask)
            yield TrackedMask(i, 1, mask)


def write_video(path, positions):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (80, 60))
    for x in positions:
        frame = np.full((60, 80, 3), 200, np.uint8)
        frame[20:32, x:x + 12] = 0
        writer.write(frame)
    writer.release()


class TrackerTests(unittest.TestCase):
    def test_recovers_from_lost_mask_inside_chunk(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_video(tmp / "v.avi", [5 + i for i in range(30)])
            config = load_config(None)
            config["tracking"].update(chunk_size=30, redetect_stride=2)
            segmenter = FakeSegmenter(lose_after=12)
            tracker = MouseTracker(config, detector=FakeDetector(), segmenter=segmenter)
            tracking = tracker.track(str(tmp / "v.avi"), tmp / "out", tmp / "work", log=lambda _: None)
            self.assertEqual(segmenter.seeds, [("box", 0), ("box", 12)])
            self.assertTrue(tracking["tracked"].all())
            self.assertEqual(pd.read_csv(tmp / "out" / "tracking_chunks.csv").loc[0, "recoveries"], 1)

    def test_chunked_tracking_resume_and_bout(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            video = tmp / "v.avi"
            write_video(video, [5 + 2 * i for i in range(10)] + [25] * 25)  # moves, then still 2.5 s

            config = load_config(None)
            config["tracking"]["chunk_size"] = 10
            config["activity"]["motion_threshold"] = 0.05
            segmenter = FakeSegmenter()
            tracker = MouseTracker(config, detector=FakeDetector(), segmenter=segmenter)
            out = tmp / "out"
            tracking = tracker.track(str(video), out, tmp / "work", log=lambda _: None)
            self.assertEqual(len(tracking), 35)
            self.assertTrue(tracking["tracked"].all())
            self.assertEqual(segmenter.seeds, [("box", 0), ("mask", 0), ("mask", 0), ("mask", 0)])
            chunks = pd.read_csv(out / "tracking_chunks.csv")
            self.assertEqual(list(chunks["seed"]), ["detection", "mask", "mask", "mask"])
            self.assertFalse((tmp / "work" / "chunk_00000").exists())

            result = activity.analyze(tracking, 10, config)
            self.assertEqual(len(result["bouts"]), 1)
            self.assertEqual(int(result["bouts"].loc[0, "start_frame"]), 11)

            again = MouseTracker(config, detector=FakeDetector(), segmenter=FakeSegmenter())
            resumed = again.track(str(video), out, tmp / "work", log=lambda _: None)
            self.assertEqual(again._segmenter.seeds, [])
            pd.testing.assert_frame_equal(resumed.reset_index(drop=True), tracking.reset_index(drop=True))


if __name__ == "__main__":
    unittest.main()
