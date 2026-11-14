"""Small, independent VLM pilot; no tracking or centroid inputs are used."""

import argparse
import base64
import csv
import hashlib
import http.client
import json
import math
import time
from pathlib import Path


LABELS = ("STILL", "ACTIVE", "AMBIGUOUS")
PROMPT = """Classify a short laboratory mouse clip using the ordered images below.
STILL means no observable voluntary behavior throughout the sampled sequence.
ACTIVE includes grooming, sniffing, head movement, walking/running, turning,
rearing, scratching, eating/licking, or clear posture adjustment, even if the
body centroid stays in the same place. Any clear voluntary action makes the
whole window ACTIVE. Tiny involuntary movement or video noise may be ignored.
Use AMBIGUOUS if visibility, resolution, occlusion, temporal sampling, or an
uncertain movement prevents distinguishing STILL from ACTIVE. Absence of
centroid displacement is not evidence of STILL. Do not infer intent or sleep.
These are sampled frames, not continuous video: do not claim unseen movements
are absent. Compare the head, paws, body posture, and orientation across frames.
Return only JSON with label (STILL, ACTIVE, AMBIGUOUS) and reason (a short
description of visible evidence and any limitation). No confidence score.
"""
SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": list(LABELS)},
        "reason": {"type": "string"},
    },
    "required": ["label", "reason"],
    "additionalProperties": False,
}


def load_manifest(path: Path, max_clips: int = 30) -> list[dict]:
    """Times are seconds in the referenced video; paths are manifest-relative."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"clip_id", "video_path", "start_s", "end_s", "expected_label"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest requires columns: {sorted(required)}")
        rows = list(reader)
    if not 1 <= len(rows) <= max_clips:
        raise ValueError(f"Select 1–{max_clips} short clips for this pilot")
    seen = set()
    for row in rows:
        clip_id = row["clip_id"].strip()
        if not clip_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in clip_id):
            raise ValueError("clip_id must contain only letters, digits, _ or -")
        if clip_id in seen:
            raise ValueError(f"Duplicate clip_id: {clip_id}")
        seen.add(clip_id)
        row["clip_id"] = clip_id
        start, end = float(row["start_s"]), float(row["end_s"])
        if not all(map(math.isfinite, (start, end))) or start < 0 or not 0 < end - start <= 10:
            raise ValueError(f"{clip_id}: require 0 <= start < end, duration <= 10 s")
        row.update(start_s=start, end_s=end, duration_s=end - start)
        expected = row["expected_label"].strip().upper()
        if expected and expected not in LABELS:
            raise ValueError(f"{clip_id}: invalid expected_label {expected!r}")
        row["expected_label"] = expected
        if not row["video_path"].strip():
            raise ValueError(f"{clip_id}: video_path is empty")
        video = Path(row["video_path"]).expanduser()
        if not video.is_absolute():
            video = path.parent / video
        if not video.is_file():
            raise ValueError(f"{clip_id}: video does not exist: {video}")
        row["video_path"] = str(video.resolve())
    return rows


def frame_indices(start: float, end: float, fps: float, count: int, sample_fps: float) -> list[int]:
    if not math.isfinite(fps) or fps <= 0 or count <= 0:
        raise ValueError("Video lacks valid FPS/frame-count metadata")
    if end > count / fps + 1e-6:
        raise ValueError("Selected window extends beyond the video")
    first, last = math.ceil(start * fps), math.ceil(end * fps) - 1
    if last - first < 1:
        raise ValueError("Window must contain at least two distinct frames")
    n = min(last - first + 1, max(2, math.ceil((end - start) * sample_fps)))
    return sorted({round(first + i * (last - first) / (n - 1)) for i in range(n)})


def sample_clip(row: dict, directory: Path, sample_fps: float, max_side: int) -> tuple[list[str], list[float]]:
    # Lazy imports keep manifest/response tests and --help independent of OpenCV.
    import cv2
    from behavision.video.reader import VideoReader

    reader = VideoReader(row["video_path"])
    images, timestamps = [], []
    try:
        indices = frame_indices(row["start_s"], row["end_s"], reader.fps, reader.frame_count, sample_fps)
        directory.mkdir(parents=True)
        for index in indices:
            frame = reader.get_frame(index)
            if frame is None:
                raise ValueError(f"Cannot decode frame {index}")
            image = frame.image
            height, width = image.shape[:2]
            scale = min(1.0, max_side / max(height, width))
            if scale < 1:
                image = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
            success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not success:
                raise ValueError(f"Cannot encode frame {index}")
            data = encoded.tobytes()
            (directory / f"{index:09d}.jpg").write_bytes(data)
            images.append(base64.b64encode(data).decode("ascii"))
            timestamps.append(index / reader.fps)
    finally:
        reader.release()
    return images, timestamps


def local_request(port: int, method: str, path: str, payload=None, timeout=300):
    """Loopback only, no proxy, remote endpoint, automatic download, or redirect."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        connection.request(method, path, body=json.dumps(payload) if payload is not None else None,
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        body = response.read().decode("utf-8")
        if response.status != 200:
            raise RuntimeError(f"Ollama HTTP {response.status}: {body[:1000]}")
        return json.loads(body)
    finally:
        connection.close()


def parse_prediction(raw: str) -> dict:
    result = json.loads(raw)
    if not isinstance(result, dict) or result.get("label") not in LABELS:
        raise ValueError("Model did not return a valid label")
    if not isinstance(result.get("reason"), str) or not result["reason"].strip():
        raise ValueError("Model did not return a nonempty reason")
    return {"label": result["label"], "reason": result["reason"]}


def summarize(results: list[dict]) -> dict:
    labeled = [r for r in results if r["expected_label"]]
    matrix = {label: {prediction: 0 for prediction in (*LABELS, "ERROR")} for label in LABELS}
    for row in labeled:
        prediction = row["label"] if row["status"] == "ok" else "ERROR"
        matrix[row["expected_label"]][prediction] += 1
    tp = matrix["STILL"]["STILL"]
    predicted_still = sum(matrix[label]["STILL"] for label in LABELS)
    actual_still = sum(matrix["STILL"].values())
    return {
        "clips": len(results),
        "errors": sum(r["status"] == "error" for r in results),
        "labeled_clips": len(labeled),
        "confusion_matrix_actual_rows_predicted_columns": matrix,
        "agreement_including_errors": sum(matrix[label][label] for label in LABELS) / len(labeled) if labeled else None,
        "still_precision": tp / predicted_still if predicted_still else None,
        "still_recall_including_errors": tp / actual_still if actual_still else None,
        "false_still_clip_ids": [r["clip_id"] for r in labeled if r.get("label") == "STILL" and r["expected_label"] != "STILL"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate local VLM stillness on 1–30 manually selected short windows.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing directories are refused")
    parser.add_argument("--model", default="qwen3-vl:4b")
    parser.add_argument("--port", type=int, default=11434)
    parser.add_argument("--sample-fps", type=float, default=8)
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--dry-run", action="store_true", help="Decode/save frames without contacting Ollama")
    args = parser.parse_args(argv)
    if not math.isfinite(args.sample_fps) or not 0 < args.sample_fps <= 8 or args.max_side < 64:
        parser.error("require 0 < sample-fps <= 8 and max-side >= 64")
    if not 1 <= args.port <= 65535 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("require valid port and positive finite timeout")
    if "cloud" in args.model.lower() or "/" in args.model or not args.model.strip():
        parser.error("use an installed local model tag, e.g. qwen3-vl:4b")
    try:
        rows = load_manifest(args.manifest)
        model_info = None
        if not args.dry_run:
            tags = local_request(args.port, "GET", "/api/tags", timeout=args.timeout)
            model_info = next((m for m in tags["models"] if m["name"] == args.model), None)
            if model_info is None:
                raise ValueError(f"Model is not installed. Run: ollama pull {args.model}")
        args.output_dir.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(2, f"Validation setup failed: {exc}\n")

    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(model_info=model_info, prompt=PROMPT, schema=SCHEMA,
                  manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  options={"temperature": 0, "seed": 0, "num_predict": 256, "num_ctx": 16384})
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    results = []
    with (args.output_dir / "predictions.jsonl").open("w") as output:
        for row in rows:
            result = {**row, "status": "error", "label": None, "reason": "", "raw_response": ""}
            started = time.monotonic()
            try:
                images, timestamps = sample_clip(row, args.output_dir / "frames" / row["clip_id"], args.sample_fps, args.max_side)
                result["sample_timestamps_s"] = timestamps
                if args.dry_run:
                    result["status"] = "dry_run"
                else:
                    timing = "\nImages are chronological, at video times (seconds): " + json.dumps(timestamps)
                    response = local_request(args.port, "POST", "/api/chat", {
                        "model": args.model, "stream": False, "think": False,
                        "messages": [{"role": "user", "content": PROMPT + timing, "images": images}],
                        "format": SCHEMA, "options": config["options"],
                    }, timeout=args.timeout)
                    result["raw_response"] = response["message"]["content"]
                    result.update(parse_prediction(result["raw_response"]), status="ok")
            except (OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError, http.client.HTTPException) as exc:
                result["error"] = str(exc)
            result["elapsed_s"] = round(time.monotonic() - started, 3)
            results.append(result)
            output.write(json.dumps(result) + "\n")
            output.flush()
            print(f"{row['clip_id']}: {result['label'] or result['status']} {result.get('error', '')}", flush=True)

    fields = ["clip_id", "video_path", "start_s", "end_s", "duration_s", "expected_label", "status", "label", "reason", "error", "elapsed_s"]
    with (args.output_dir / "predictions.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    summary = {"dry_run": True, "clips": len(results), "errors": sum(r["status"] == "error" for r in results)} if args.dry_run else summarize(results)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved validation artifacts to {args.output_dir}")
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
