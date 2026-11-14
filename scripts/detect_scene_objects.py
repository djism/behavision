import json
import cv2

from behavision.detection.grounding_dino import GroundingDINODetector
from behavision.detection.postprocess import non_max_suppression
from behavision.scene.transform import load_arena_calibration


FRAME_PATH = "data/cache/Trial1/003990.jpg"
SCENE_PATH = "data/outputs/Trial1/scene.json"


OBJECT_PROMPTS = [
    {
        "id": "object_1",
        "prompt": "green cube",
    },
    {
        "id": "object_2",
        "prompt": "white circular object",
    },
]


image = cv2.imread(FRAME_PATH)

if image is None:
    raise RuntimeError(f"Could not load {FRAME_PATH}")


calibration = load_arena_calibration(
    SCENE_PATH
)

detector = GroundingDINODetector()

scene_objects = []


for config in OBJECT_PROMPTS:

    object_id = config["id"]
    prompt = config["prompt"]

    print(
        f"\nDetecting {object_id}: {prompt}"
    )

    detections = detector.detect(
        image=image,
        prompt=prompt,
    )

    detections = non_max_suppression(
        detections,
        iou_threshold=0.5,
    )

    valid_detections = []

    for detection in detections:

        cx, cy = detection.center

        if calibration.contains_camera_point(
            cx,
            cy,
            margin=0.03,
        ):
            valid_detections.append(
                detection
            )

    if not valid_detections:

        print(
            f"No valid detection for {object_id}"
        )
        continue

    best = max(
        valid_detections,
        key=lambda d: d.confidence,
    )

    cx, cy = best.center

    arena_x, arena_y = (
        calibration.camera_to_arena(
            cx,
            cy,
        )
    )

    print("Selected:", best)

    print(
        "Camera center:",
        round(cx, 2),
        round(cy, 2),
    )

    print(
        "Arena center:",
        round(arena_x, 3),
        round(arena_y, 3),
    )

    scene_objects.append(
        {
            "id": object_id,
            "label": prompt,
            "confidence": best.confidence,

            "bbox": [
                best.x1,
                best.y1,
                best.x2,
                best.y2,
            ],

            "center": {
                "x": cx,
                "y": cy,
            },

            "arena_center": {
                "x": arena_x,
                "y": arena_y,
            },
        }
    )


with open(
    SCENE_PATH,
    "r",
) as file:
    scene = json.load(file)


scene["objects"] = scene_objects


with open(
    SCENE_PATH,
    "w",
) as file:
    json.dump(
        scene,
        file,
        indent=2,
    )


print(
    "\nDetected objects:",
    len(scene_objects)
)

print(
    "Saved:",
    SCENE_PATH
)