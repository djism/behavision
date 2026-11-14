import cv2
import numpy as np
from behavision.detection.types import Detection


def draw_detections(
    image,
    detections: list[Detection],
):
    output = image.copy()

    for detection in detections:
        x1 = int(detection.x1)
        y1 = int(detection.y1)
        x2 = int(detection.x2)
        y2 = int(detection.y2)

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        label = (
            f"{detection.class_name} "
            f"{detection.confidence:.2f}"
        )

        cv2.putText(
            output,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    return output

def draw_mask(
    image,
    mask,
):
    output = image.copy()

    overlay = np.zeros_like(
        output
    )

    overlay[mask] = (
        0,
        255,
        0,
    )

    output = cv2.addWeighted(
        output,
        1.0,
        overlay,
        0.4,
        0,
    )

    return output


def draw_centroid(
    image,
    centroid,
):
    output = image.copy()

    if centroid is None:
        return output

    cx, cy = centroid

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

    return output