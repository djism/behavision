"""Chunked single-mouse SAM2 tracking that records per-frame motion statistics.

Frames are decoded once, sequentially. Each chunk is written to a temporary
JPEG directory for SAM2 and removed afterwards, so disk use is bounded by one
chunk. A chunk is seeded with the previous chunk's last mask (one-frame
overlap) unless that mask is missing or implausible, or a confident detection
disagrees with it. Completed chunks are saved and skipped on restart.
"""

import gc
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from behavision.behavior.activity import frame_motion
from behavision.detection.postprocess import drop_excluded, non_max_suppression, select_best_detection
from behavision.video.reader import VideoReader


def mask_box(mask) -> tuple[float, float, float, float] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def box_iou(a, b) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / union if union > 0 else 0.0


def _write_atomic(path: Path, write) -> None:
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    tmp.replace(path)


def _save_mask(path: Path, mask) -> None:
    with open(path, "wb") as handle:
        np.savez_compressed(handle, valid=mask is not None, mask=mask if mask is not None else np.zeros(1, bool))


def _empty_cache() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


class MouseTracker:
    def __init__(self, config: dict, detector=None, segmenter=None):
        self.config = config
        self._detector = detector
        self._segmenter = segmenter

    @property
    def detector(self):
        if self._detector is None:
            from behavision.detection.grounding_dino import GroundingDINODetector

            cfg = self.config["detection"]
            self._detector = GroundingDINODetector(box_threshold=cfg["box_threshold"],
                                                   text_threshold=cfg["text_threshold"])
        return self._detector

    @property
    def segmenter(self):
        if self._segmenter is None:
            from behavision.segmentation.sam2_video import SAM2VideoSegmenter

            self._segmenter = SAM2VideoSegmenter()
        return self._segmenter

    def detect(self, image):
        cfg = self.config["detection"]
        detections = non_max_suppression(self.detector.detect(image=image, prompt=cfg["prompt"]), iou_threshold=0.5)
        detections = drop_excluded(detections, cfg["exclude_regions"], cfg["max_excluded_overlap"])
        return select_best_detection(detections)

    def track(self, video_path: str, out_dir: Path, work_dir: Path,
              start_s: float = 0.0, end_s: float | None = None, log=print) -> pd.DataFrame:
        cfg_t, cfg_a = self.config["tracking"], self.config["activity"]
        chunk_dir = out_dir / "tracking_chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        reader = VideoReader(video_path)
        fps = reader.fps
        first = max(0, round(start_s * fps))
        last = reader.frame_count if end_s is None else min(reader.frame_count, round(end_s * fps))
        if last - first < 2:
            raise ValueError("Selected range contains fewer than two frames")
        for _ in range(first):
            if not reader.cap.grab():
                raise RuntimeError(f"Could not skip to frame {first}")
        reader.current_frame = first

        kernel = cfg_a["blur_kernel"]
        prev_gray = prev_mask = prev_image = None
        areas: list[int] = []
        size = cfg_t["chunk_size"]
        starts = range(first, last, size)
        log(f"Tracking frames {first}-{last - 1} at {fps:.3f} fps in {len(starts)} chunks")

        try:
            for k, chunk_start in enumerate(starts):
                chunk_end = min(chunk_start + size, last)
                images = []
                for _ in range(chunk_end - chunk_start):
                    frame = reader.read_frame()
                    if frame is None:
                        break
                    images.append(frame.image)
                if not images:
                    log(f"Video ended early at frame {chunk_start}")
                    break
                chunk_end = chunk_start + len(images)
                grays = [cv2.GaussianBlur(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY), (kernel, kernel), 0) for i in images]
                info_path = chunk_dir / f"chunk_{k:05d}.json"
                if info_path.exists():
                    saved = np.load(chunk_dir / f"chunk_{k:05d}_last.npz")
                    prev_mask = saved["mask"] if saved["valid"] else None
                    saved_rows = pd.read_csv(chunk_dir / f"chunk_{k:05d}.csv")
                    if "mask_area" in saved_rows:
                        areas.extend(saved_rows["mask_area"].dropna().astype(int).tolist())
                    log(f"Chunk {k + 1}/{len(starts)} already done; skipped")
                else:
                    rows, prev_mask, info = self._track_chunk(
                        images, grays, chunk_start, fps, prev_image, prev_gray, prev_mask, areas,
                        work_dir / f"chunk_{k:05d}")
                    _write_atomic(chunk_dir / f"chunk_{k:05d}.csv", lambda p: pd.DataFrame(rows).to_csv(p, index=False))
                    _write_atomic(chunk_dir / f"chunk_{k:05d}_last.npz", lambda p: _save_mask(p, prev_mask))
                    _write_atomic(info_path, lambda p: p.write_text(json.dumps({"chunk": k, **info}) + "\n"))
                    tracked = sum(r["tracked"] for r in rows)
                    log(f"Chunk {k + 1}/{len(starts)} frames {chunk_start}-{chunk_end - 1}: "
                        f"seed={info['seed']} tracked={tracked}/{len(rows)}")
                prev_image, prev_gray = images[-1], grays[-1]
        finally:
            reader.release()

        tracking = pd.concat([pd.read_csv(p) for p in sorted(chunk_dir.glob("chunk_*.csv"))], ignore_index=True)
        tracking = tracking[(tracking["frame"] >= first) & (tracking["frame"] < last)].sort_values("frame")
        chunks = [json.loads(p.read_text()) for p in sorted(chunk_dir.glob("chunk_*.json"))]
        pd.DataFrame(chunks).to_csv(out_dir / "tracking_chunks.csv", index=False)
        tracking.to_csv(out_dir / "tracking.csv", index=False)
        return tracking

    def _seed(self, images, prev_mask, areas):
        """Choose (seed kind, local frame offset in the chunk, mask or box, info)."""
        cfg = self.config["tracking"]
        low, high = self.config["activity"]["area_ratio_bounds"]
        detection = self.detect(images[0])
        info = {"detection_confidence": round(detection.confidence, 3) if detection else None}
        if prev_mask is not None and areas:
            ratio = prev_mask.sum() / np.median(areas)
            if not low <= ratio <= high:
                prev_mask = None
        if prev_mask is not None:
            iou = box_iou(mask_box(prev_mask), (detection.x1, detection.y1, detection.x2, detection.y2)) if detection else None
            info["mask_detection_iou"] = round(iou, 3) if iou is not None else None
            if not (detection and detection.confidence >= cfg["reseed_min_confidence"] and iou < cfg["reseed_max_iou"]):
                return "mask", 0, prev_mask, info
        found = self._find_detection(images, 0, detection)
        if found is None:
            return "none", None, None, info
        j, detection = found
        box = (detection.x1, detection.y1, detection.x2, detection.y2)
        return "detection", j, box, {**info, "detection_frame_offset": j,
                                     "detection_confidence": round(detection.confidence, 3)}

    def _find_detection(self, images, start: int, first=None):
        """First (local index, detection) at start, start + stride, ...; ``first`` is images[start]'s."""
        for j in range(start, len(images), self.config["tracking"]["redetect_stride"]):
            detection = first if j == start and first is not None else self.detect(images[j])
            if detection is not None:
                return j, detection
        return None

    def _plausible(self, mask, reference_area) -> bool:
        area = int(mask.sum())
        if area == 0:
            return False
        if reference_area is None:
            return area <= 0.1 * mask.size
        low, high = self.config["activity"]["area_ratio_bounds"]
        return low <= area / reference_area <= high

    def _propagate(self, state, kind, frame, prompt, shift, n, reference_area):
        """Masks by local index from a prompt; stops at a sustained failure.

        Returns (masks, local index where the failure began or None).
        """
        if kind == "mask":
            self.segmenter.add_mask(state=state, frame_index=frame, mask=prompt)
        else:
            self.segmenter.add_box(state=state, frame_index=frame, box=prompt)
        failure_frames = self.config["tracking"]["failure_frames"]
        masks, bad = {}, []
        for tracked in self.segmenter.propagate(state=state, start_frame_index=frame):
            local = tracked.frame_index - shift
            if not 0 <= local < n:
                continue
            masks[local] = tracked.mask
            if reference_area is None and self._plausible(tracked.mask, None) and len(masks) >= 10:
                reference_area = float(np.median([m.sum() for m in masks.values() if m.any()]))
            bad = bad + [local] if not self._plausible(tracked.mask, reference_area) else []
            if len(bad) >= failure_frames:
                for failed in bad:
                    del masks[failed]
                return masks, bad[0]
        return masks, None

    def _track_chunk(self, images, grays, chunk_start, fps, prev_image, prev_gray, prev_mask, areas, work):
        cfg_t, cfg_a = self.config["tracking"], self.config["activity"]
        kind, offset, seed, info = self._seed(images, prev_mask, areas)
        info.update(seed=kind, recoveries=0)
        rows = [{"frame": chunk_start + i, "time_s": (chunk_start + i) / fps, "tracked": False} for i in range(len(images))]
        masks = {}
        if kind != "none":
            # Mask seeds use the previous chunk's last frame as SAM2 frame 0.
            shift = 1 if kind == "mask" else 0
            shutil.rmtree(work, ignore_errors=True)
            work.mkdir(parents=True)
            for i, image in enumerate([prev_image] + images if shift else images):
                if not cv2.imwrite(str(work / f"{i:06d}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError(f"Could not write {work}")
            reference_area = float(np.median(areas)) if areas else None
            state = self.segmenter.initialize(str(work))
            prompt_kind, prompt_frame, prompt = ("mask", 0, seed) if kind == "mask" else ("box", offset, seed)
            while True:
                found, failed_at = self._propagate(state, prompt_kind, prompt_frame, prompt, shift, len(images),
                                                   reference_area)
                masks.update(found)
                if failed_at is None or info["recoveries"] >= cfg_t["max_recoveries"]:
                    break
                # Search strictly after the last prompt: while the mouse is hidden
                # (e.g. under nesting paper) each recovery can fail at its own
                # frame, and must not be retried there.
                if prompt_kind == "box":
                    failed_at = max(failed_at, prompt_frame - shift + cfg_t["redetect_stride"])
                recovery = self._find_detection(images, failed_at)
                if recovery is None:
                    break
                j, detection = recovery
                info["recoveries"] += 1
                self.segmenter.reset(state)
                prompt_kind, prompt_frame = "box", j + shift
                prompt = (detection.x1, detection.y1, detection.x2, detection.y2)
            del state
            shutil.rmtree(work, ignore_errors=True)
            _empty_cache()
            for local, mask in sorted(masks.items()):
                area = int(mask.sum())
                if area == 0:
                    continue
                ys, xs = np.nonzero(mask)
                rows[local].update(tracked=True, mask_area=area, x=round(float(xs.mean()), 2), y=round(float(ys.mean()), 2),
                                   **dict(zip(("box_x1", "box_y1", "box_x2", "box_y2"), mask_box(mask))))
                areas.append(area)
        last_mask = masks.get(len(images) - 1)
        if last_mask is not None and not last_mask.any():
            last_mask = None

        # Motion needs the previous frame; across a chunk boundary that is prev_gray/prev_mask.
        before_gray, before_mask = prev_gray, prev_mask
        for local, gray in enumerate(grays):
            mask = masks.get(local)
            if rows[local]["tracked"] and before_gray is not None:
                rows[local].update(frame_motion(before_gray, gray, before_mask if before_mask is not None else mask,
                                                mask, cfg_a["mask_dilation_px"], cfg_a["background_margin_px"]))
            before_gray = gray
            before_mask = mask if rows[local]["tracked"] else None
        return rows, last_mask, info
