import csv
import os

import cv2
import numpy as np

from behavision.detection.grounding_dino import (
    GroundingDINODetector,
)
from behavision.detection.postprocess import (
    non_max_suppression,
    select_best_detection,
)
from behavision.segmentation.sam2_video import (
    SAM2VideoSegmenter,
)
from behavision.tracking.trajectory import (
    mask_centroid,
    mask_area,
    displacement,
    speed,
    area_ratio,
)


FRAME_DIR = "data/sam2_test_frames"

OUTPUT_DIR = "data/outputs/tracking_test"

PROMPT = "black laboratory mouse"

OBJECT_ID = 1

MAX_FRAMES = 265

FPS = 26.53257978723404

VIDEO_START_SECONDS = 175.0

JUMP_THRESHOLD_PX = 80.0

AREA_RATIO_LOW = 0.5
AREA_RATIO_HIGH = 2.0


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


first_frame_path = os.path.join(
    FRAME_DIR,
    "00000.jpg",
)


first_frame = cv2.imread(
    first_frame_path
)

if first_frame is None:
    raise RuntimeError(
        f"Could not load {first_frame_path}"
    )


print("=== Initial Detection ===")


detector = GroundingDINODetector()


detections = detector.detect(
    image=first_frame,
    prompt=PROMPT,
)


print(
    "Raw detections:",
    len(detections),
)


detections = non_max_suppression(
    detections,
    iou_threshold=0.5,
)


print(
    "After NMS:",
    len(detections),
)


best = select_best_detection(
    detections
)


if best is None:
    raise RuntimeError(
        "Could not detect mouse "
        "on initialization frame"
    )


print(
    "Selected detection:",
    best
)


box = (
    best.x1,
    best.y1,
    best.x2,
    best.y2,
)


print(
    "\n=== SAM2 Video Initialization ==="
)


segmenter = SAM2VideoSegmenter()


state = segmenter.initialize(
    FRAME_DIR
)


segmenter.add_box(
    state=state,
    frame_index=0,
    box=box,
    object_id=OBJECT_ID,
)


print(
    "\n=== Propagating Mouse Mask ==="
)


trajectory = []

previous_centroid = None
previous_area = None


for tracked in segmenter.propagate(
    state=state,
    start_frame_index=0,
    max_frames=MAX_FRAMES,
):

    centroid = mask_centroid(
        tracked.mask
    )

    if centroid is None:
        print(
            f"Frame {tracked.frame_index}: "
            f"No centroid"
        )
        continue


    cx, cy = centroid

    current_area = mask_area(
        tracked.mask
    )


    frame_displacement = displacement(
        previous_centroid,
        centroid,
    )


    delta_time = (
        1.0 / FPS
    )


    speed_px_s = speed(
        frame_displacement,
        delta_time,
    )


    current_area_ratio = area_ratio(
        previous_area,
        current_area,
    )


    jump_flag = False

    if (
        frame_displacement is not None
        and frame_displacement > JUMP_THRESHOLD_PX
    ):
        jump_flag = True


    area_flag = False

    if current_area_ratio is not None:
        if (
            current_area_ratio < AREA_RATIO_LOW
            or current_area_ratio > AREA_RATIO_HIGH
        ):
            area_flag = True


    clip_time = (
        tracked.frame_index
        / FPS
    )


    video_time = (
        VIDEO_START_SECONDS
        + clip_time
    )


    trajectory.append(
        {
            "frame": tracked.frame_index,
            "clip_time": clip_time,
            "video_time": video_time,
            "x": cx,
            "y": cy,
            "mask_area": current_area,
            "displacement": frame_displacement,
            "speed_px_s": speed_px_s,
            "area_ratio": current_area_ratio,
            "jump_flag": jump_flag,
            "area_flag": area_flag,
        }
    )


    print(
        f"Frame: {tracked.frame_index} | "
        f"Centroid: ({cx:.1f}, {cy:.1f}) | "
        f"Area: {current_area} | "
        f"Disp: "
        f"{frame_displacement if frame_displacement is not None else 0:.2f} | "
        f"Jump: {jump_flag} | "
        f"Area flag: {area_flag}"
    )


    if tracked.frame_index % 20 == 0:

        frame_path = os.path.join(
            FRAME_DIR,
            f"{tracked.frame_index:05d}.jpg",
        )

        image = cv2.imread(
            frame_path
        )

        if image is not None:

            overlay = image.copy()

            overlay[
                tracked.mask
            ] = (
                0,
                255,
                0,
            )

            output = cv2.addWeighted(
                image,
                0.7,
                overlay,
                0.3,
                0,
            )

            cv2.circle(
                output,
                (
                    int(cx),
                    int(cy),
                ),
                6,
                (
                    0,
                    0,
                    255,
                ),
                -1,
            )

            cv2.imwrite(
                os.path.join(
                    OUTPUT_DIR,
                    f"{tracked.frame_index:05d}.jpg",
                ),
                output,
            )


    previous_centroid = centroid
    previous_area = current_area


print(
    "\n=== Saving Trajectory ==="
)


csv_path = os.path.join(
    OUTPUT_DIR,
    "trajectory.csv",
)


with open(
    csv_path,
    "w",
    newline="",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "frame",
            "clip_time",
            "video_time",
            "x",
            "y",
            "mask_area",
            "displacement",
            "speed_px_s",
            "area_ratio",
            "jump_flag",
            "area_flag",
        ],
    )

    writer.writeheader()

    writer.writerows(
        trajectory
    )


print(
    "Saved trajectory:",
    csv_path
)

print(
    "Trajectory rows:",
    len(trajectory)
)