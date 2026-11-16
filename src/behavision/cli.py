"""Whole-video inactivity analysis: ``behavision-run VIDEO --config C --out DIR``.

Stages (all by default): track -> analyze -> review. Tracking resumes from
completed chunks in an existing output directory; analyze and review are cheap
and can be re-run with new activity/review settings without re-tracking.
"""

import argparse
import json
import os

# The tokenizer must not use threads before ffmpeg subprocesses fork.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import sys
import time
from pathlib import Path

import pandas as pd

from behavision.config import load_config


STAGES = ("track", "analyze", "review")
# Settings that change tracking.csv; a resumed run must not mix them.
TRACKING_KEYS = ("detection", "tracking")
TRACKING_ACTIVITY_KEYS = ("blur_kernel", "mask_dilation_px", "background_margin_px", "area_ratio_bounds")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def video_metadata(video_path: str) -> dict:
    from behavision.video.reader import VideoReader

    reader = VideoReader(video_path)
    try:
        return {"video_path": str(Path(video_path).resolve()), "fps": reader.fps, "frame_count": reader.frame_count,
                "width": reader.width, "height": reader.height, "duration_s": round(reader.duration, 3)}
    finally:
        reader.release()


def tracking_settings(config: dict, start_s, end_s) -> dict:
    return {**{k: config[k] for k in TRACKING_KEYS},
            "activity": {k: config["activity"][k] for k in TRACKING_ACTIVITY_KEYS},
            "start_s": start_s, "end_s": end_s}


def preview(video_path: str, out_dir: Path, config: dict) -> Path:
    """Exclusion regions and chosen detection on frames across the video."""
    import cv2
    import numpy as np

    from behavision.pipeline.inactivity import MouseTracker
    from behavision.video.reader import VideoReader

    tracker = MouseTracker(config)
    reader = VideoReader(video_path)
    panels = []
    try:
        for fraction in (0.0, 0.25, 0.5, 0.75, 0.98):
            frame = reader.get_frame(int(fraction * (reader.frame_count - 1)))
            image = frame.image.copy()
            for x1, y1, x2, y2 in config["detection"]["exclude_regions"]:
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            best = tracker.detect(frame.image)
            if best is not None:
                cv2.rectangle(image, (int(best.x1), int(best.y1)), (int(best.x2), int(best.y2)), (214, 120, 42), 2)
            text = f"t={frame.timestamp / 60:.1f} min  " + (f"mouse {best.confidence:.2f}" if best else "NO DETECTION")
            cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            panels.append(image)
    finally:
        reader.release()
    panels.append(np.full_like(panels[0], 255))
    grid = np.vstack([np.hstack(panels[i:i + 3]) for i in (0, 3)])
    path = out_dir / "preview.jpg"
    cv2.imwrite(str(path), grid)
    return path


def quality_report(tracking: pd.DataFrame, chunks: pd.DataFrame, result: dict, config: dict) -> dict:
    timeline, summary = result["timeline"], result["summary"]
    calibration = summary["calibration"]
    frames = len(tracking)
    tracked = float(tracking["tracked"].mean()) if frames else 0.0
    uncertain = float((timeline["state"] == "uncertain").mean()) if frames else 0.0
    seeds = chunks["seed"].value_counts().to_dict() if "seed" in chunks else {}
    reseeded = (int(((chunks["seed"] == "detection") & chunks["mask_detection_iou"].notna()).sum())
                if {"seed", "mask_detection_iou"} <= set(chunks) else 0)
    warnings = []
    if tracked < 0.95:
        warnings.append(f"Mouse tracked in only {tracked:.1%} of frames")
    if uncertain > 0.10:
        warnings.append(f"{uncertain:.1%} of frames are uncertain")
    if seeds.get("none"):
        warnings.append(f"{seeds['none']} chunk(s) had no mouse detection")
    bounds = config["activity"]["motion_threshold_bounds"]
    if calibration["motion_threshold_source"] == "otsu" and calibration["motion_threshold"] in [round(b, 5) for b in bounds]:
        warnings.append("Automatic motion threshold hit a configured bound; inspect activity_overview.png")
    return {"frames": frames, "tracked_fraction": round(tracked, 4), "uncertain_fraction": round(uncertain, 4),
            "chunks": int(len(chunks)), "chunk_seeds": seeds,
            "chunks_reseeded_by_detection": reseeded,
            "calibration": calibration, "warnings": warnings}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Detect inactive bouts of one mouse in a video.")
    parser.add_argument("video")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Output directory (reused to resume)")
    parser.add_argument("--stages", default=",".join(STAGES), help="Comma-separated subset of: " + ", ".join(STAGES))
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--end-s", type=float, default=None)
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="Scratch for temporary chunk frames (default: $TMPDIR or OUT/_work)")
    parser.add_argument("--annotated-video", action="store_true", help="Also write annotated.mp4 for the whole range")
    parser.add_argument("--preview", action="store_true", help="Only save preview.jpg of detections, then exit")
    args = parser.parse_args(argv)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    if not stages or any(s not in STAGES for s in stages):
        parser.error(f"--stages must be a subset of {','.join(STAGES)}")
    if not Path(args.video).is_file():
        parser.error(f"video not found: {args.video}")

    config = load_config(args.config)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    title = Path(args.video).stem
    if args.preview:
        log(f"Preview: {preview(args.video, out, config)}")
        return 0

    meta = video_metadata(args.video)
    (out / "video_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    log(f"{title}: {meta['frame_count']} frames, {meta['fps']:.3f} fps, {meta['width']}x{meta['height']}, "
        f"{meta['duration_s'] / 60:.1f} min")

    if "track" in stages:
        settings = tracking_settings(config, args.start_s, args.end_s)
        settings_path = out / "tracking_settings.json"
        if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
            parser.exit(2, f"{out} holds tracking from different settings or range; use a new --out directory\n")
        settings_path.write_text(json.dumps(settings, indent=2) + "\n")
        from behavision.pipeline.inactivity import MouseTracker

        work = args.work_dir or (Path(os.environ["TMPDIR"]) / f"behavision_{title}" if os.environ.get("TMPDIR")
                                 else out / "_work")
        started = time.monotonic()
        MouseTracker(config).track(args.video, out, work, args.start_s, args.end_s, log=log)
        log(f"Tracking finished in {(time.monotonic() - started) / 60:.1f} min")

    if "analyze" in stages or "review" in stages:
        if not (out / "tracking.csv").exists():
            parser.exit(2, f"{out / 'tracking.csv'} not found; run the track stage first\n")
        tracking = pd.read_csv(out / "tracking.csv")
        (out / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
        from behavision.behavior.activity import analyze
        from behavision.visualization.activity_report import plot_overview, write_annotated_video, write_review

        result = analyze(tracking, meta["fps"], config)
        if "analyze" in stages:
            chunks = pd.read_csv(out / "tracking_chunks.csv") if (out / "tracking_chunks.csv").exists() else pd.DataFrame()
            summary = {"video": meta["video_path"], **result["summary"]}
            result["timeline"].to_csv(out / "timeline.csv", index=False)
            result["bouts"].to_csv(out / "inactive_bouts.csv", index=False)
            result["sensitivity"].to_csv(out / "sensitivity.csv", index=False)
            (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            qc = quality_report(tracking, chunks, result, config)
            (out / "qc.json").write_text(json.dumps(qc, indent=2) + "\n")
            plot_overview(result["timeline"], result["summary"], out / "activity_overview.png", title)
            log(f"Inactive {summary['inactive_s'] / 60:.1f} min in {summary['bouts']} bouts "
                f"({summary['inactive_fraction_of_observed']} of observed time); uncertain {summary['uncertain_s']:.0f} s")
            for warning in qc["warnings"]:
                log(f"QC warning: {warning}")
        if "review" in stages:
            page = write_review(args.video, result["timeline"], tracking, result["bouts"], out, config, title)
            log(f"Review page: {page}")
        if args.annotated_video:
            write_annotated_video(args.video, result["timeline"], tracking, out / "annotated.mp4")
            log(f"Annotated video: {out / 'annotated.mp4'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
