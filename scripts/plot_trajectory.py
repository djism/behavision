import csv
import os

import cv2
import numpy as np


FRAME_DIR = "data/sam2_test_frames"

CSV_PATH = (
    "data/outputs/tracking_test/"
    "trajectory.csv"
)

OUTPUT_PATH = (
    "data/outputs/tracking_test/"
    "trajectory_overlay.jpg"
)


first_frame_path = os.path.join(
    FRAME_DIR,
    "00000.jpg",
)


image = cv2.imread(
    first_frame_path
)

if image is None:
    raise RuntimeError(
        "Could not load first frame"
    )


points = []


with open(
    CSV_PATH,
    "r",
) as file:

    reader = csv.DictReader(
        file
    )

    for row in reader:

        x = float(
            row["x"]
        )

        y = float(
            row["y"]
        )

        points.append(
            (
                int(x),
                int(y),
            )
        )


for i in range(
    1,
    len(points),
):

    cv2.line(
        image,
        points[i - 1],
        points[i],
        (
            0,
            0,
            255,
        ),
        2,
    )


if points:

    cv2.circle(
        image,
        points[0],
        8,
        (
            0,
            255,
            0,
        ),
        -1,
    )

    cv2.circle(
        image,
        points[-1],
        8,
        (
            255,
            0,
            0,
        ),
        -1,
    )


cv2.imwrite(
    OUTPUT_PATH,
    image,
)


print(
    "Saved trajectory overlay:",
    OUTPUT_PATH
)