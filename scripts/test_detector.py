import os
import cv2

from behavision.video.reader import VideoReader
from behavision.detection.grounding_dino import (
    GroundingDINODetector,
)
from behavision.visualization.overlay import (
    draw_detections,
)


VIDEO_PATH = "/Users/dhananjaysharma/Desktop/videos/Trial1.mp4"

OUTPUT_DIR = "data/outputs/detection_test"

PROMPT = "black laboratory mouse"

TEST_TIMES = [
    0,
    25,
    50,
    75,
    100,
    125,
    150,
    175,
    200,
    225,
    250,
    275,
    295,
]


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


reader = VideoReader(VIDEO_PATH)

detector = GroundingDINODetector()


print("=== Video Info ===")
print("FPS:", reader.fps)
print("Frames:", reader.frame_count)
print("Duration:", round(reader.duration, 2))
print("Prompt:", PROMPT)


print("\n=== Detection Test ===")


for seconds in TEST_TIMES:

    frame_index = int(
        seconds * reader.fps
    )

    # Make sure we never request a frame
    # outside the video.
    frame_index = min(
        frame_index,
        reader.frame_count - 1
    )

    frame = reader.get_frame(
        frame_index
    )

    if frame is None:
        print(
            f"[{seconds}s] "
            f"Could not read frame "
            f"{frame_index}"
        )
        continue


    detections = detector.detect(
        image=frame.image,
        prompt=PROMPT,
    )


    print(
        f"\nTime: {seconds}s | "
        f"Frame: {frame.index} | "
        f"Actual timestamp: "
        f"{frame.timestamp:.2f}s"
    )

    print(
        f"Found {len(detections)} "
        f"detections"
    )


    for i, detection in enumerate(
        detections
    ):

        print(
            f"  Detection {i + 1}:",
            "class:",
            detection.class_name,
            "confidence:",
            round(
                detection.confidence,
                3
            ),
            "bbox:",
            (
                round(
                    detection.x1,
                    1
                ),
                round(
                    detection.y1,
                    1
                ),
                round(
                    detection.x2,
                    1
                ),
                round(
                    detection.y2,
                    1
                ),
            ),
        )


    output_image = draw_detections(
        frame.image,
        detections,
    )


    output_path = os.path.join(
        OUTPUT_DIR,
        f"{seconds:03d}s.jpg",
    )


    success = cv2.imwrite(
        output_path,
        output_image,
    )

    if success:
        print(
            "Saved:",
            output_path
        )
    else:
        print(
            "Failed to save:",
            output_path
        )


reader.release()


print(
    "\n=== Test Complete ==="
)

print(
    f"Check images inside: "
    f"{OUTPUT_DIR}"
)