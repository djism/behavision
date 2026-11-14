import cv2

from behavision.detection.grounding_dino import (
    GroundingDINODetector,
)


FRAME_PATH = (
    "data/cache/Trial1/"
    "003990.jpg"
)


PROMPTS = [
    "green square object",
    "green cube",
    "green laboratory object",
    "white circular object",
    "white round object",
    "white laboratory object",
    "circular object with black center",
    "round object with dark center",
]


image = cv2.imread(
    FRAME_PATH
)


if image is None:
    raise RuntimeError(
        f"Could not load "
        f"{FRAME_PATH}"
    )


detector = (
    GroundingDINODetector()
)


for prompt in PROMPTS:

    print(
        "\n===================="
    )

    print(
        "Prompt:",
        prompt
    )

    detections = (
        detector.detect(
            image=image,
            prompt=prompt,
        )
    )


    print(
        "Detections:",
        len(detections)
    )


    for detection in detections:

        print(
            detection
        )