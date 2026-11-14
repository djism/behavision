import os
import cv2

from behavision.detection.grounding_dino import GroundingDINODetector


IMAGE_PATH = "data/cache/Trial1/003990.jpg"
OUTPUT_DIR = "data/outputs/Trial1/mouse_part_prompts"

PROMPTS = [
    "mouse nose",
    "mouse snout",
    "mouse head",
    "head of a mouse",
]

# Temporary crop for this single-frame model test.
# If this crop is wrong, we'll replace it with your detector bbox next.
x1, y1, x2, y2 = 1130, 180, 1330, 500

os.makedirs(OUTPUT_DIR, exist_ok=True)

image = cv2.imread(IMAGE_PATH)
if image is None:
    raise RuntimeError(f"Could not read {IMAGE_PATH}")

crop = image[y1:y2, x1:x2]

# Enlarge the mouse crop so tiny anatomical regions are easier to ground.
scale = 4
crop_big = cv2.resize(
    crop,
    None,
    fx=scale,
    fy=scale,
    interpolation=cv2.INTER_CUBIC,
)

cv2.imwrite(
    os.path.join(OUTPUT_DIR, "mouse_crop.jpg"),
    crop_big,
)

detector = GroundingDINODetector()

for prompt in PROMPTS:
    print(f"\nPrompt: {prompt}")

    detections = detector.detect(
        crop_big,
        prompt=prompt,
    )

    print(detections)

    vis = crop_big.copy()

    for det in detections:
        x1d = int(det.x1)
        y1d = int(det.y1)
        x2d = int(det.x2)
        y2d = int(det.y2)

        cv2.rectangle(
            vis,
            (x1d, y1d),
            (x2d, y2d),
            (0, 255, 0),
            2,
        )

        label = f"{det.class_name} {det.confidence:.2f}"

        cv2.putText(
            vis,
            label,
            (x1d, max(20, y1d - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    safe_name = prompt.replace(" ", "_")

    out_path = os.path.join(
        OUTPUT_DIR,
        f"{safe_name}.jpg",
    )

    cv2.imwrite(out_path, vis)

    print("Saved:", out_path)