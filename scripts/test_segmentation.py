import os
import cv2

from behavision.video.reader import (
    VideoReader,
)
from behavision.detection.grounding_dino import (
    GroundingDINODetector,
)
from behavision.detection.postprocess import (
    non_max_suppression,
    select_best_detection,
)
from behavision.segmentation.sam2 import (
    SAM2Segmenter,
)
from behavision.tracking.trajectory import (
    mask_centroid,
)
from behavision.visualization.overlay import (
    draw_detections,
    draw_mask,
    draw_centroid,
)


VIDEO_PATH = (
    "/Users/dhananjaysharma/"
    "Desktop/videos/Trial1.mp4"
)

OUTPUT_DIR = (
    "data/outputs/segmentation_test"
)

PROMPT = (
    "black laboratory mouse"
)

TEST_TIME = 175


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


reader = VideoReader(
    VIDEO_PATH
)

detector = (
    GroundingDINODetector()
)

segmenter = (
    SAM2Segmenter()
)


frame_index = int(
    TEST_TIME
    * reader.fps
)

frame = reader.get_frame(
    frame_index
)

if frame is None:
    raise RuntimeError(
        "Could not read frame"
    )


detections = detector.detect(
    image=frame.image,
    prompt=PROMPT,
)


print(
    "Raw detections:",
    len(detections),
)


filtered = non_max_suppression(
    detections,
    iou_threshold=0.5,
)


print(
    "After NMS:",
    len(filtered),
)


best = select_best_detection(
    filtered
)


if best is None:
    raise RuntimeError(
        "No mouse detected"
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


segmentation = (
    segmenter.segment_box(
        image=frame.image,
        box=box,
    )
)


print(
    "SAM2 score:",
    round(
        segmentation.score,
        3,
    )
)


centroid = mask_centroid(
    segmentation.mask
)


print(
    "Mask centroid:",
    centroid
)


output = draw_mask(
    frame.image,
    segmentation.mask,
)

output = draw_detections(
    output,
    [best],
)

output = draw_centroid(
    output,
    centroid,
)


output_path = os.path.join(
    OUTPUT_DIR,
    f"{TEST_TIME:03d}s_mask.jpg",
)


cv2.imwrite(
    output_path,
    output,
)


print(
    "Saved:",
    output_path
)


reader.release()