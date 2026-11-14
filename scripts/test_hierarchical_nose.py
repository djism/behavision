import os
import cv2

from behavision.detection.grounding_dino import GroundingDINODetector


IMAGE_PATH = "data/cache/Trial1/003990.jpg"
OUTPUT_DIR = "data/outputs/Trial1/hierarchical_nose"

MOUSE_PROMPT = "black laboratory mouse"
HEAD_PROMPT = "head of a mouse"

PART_PROMPTS = [
    "nose",
    "snout",
    "tip of snout",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

detector = GroundingDINODetector()

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise RuntimeError(f"Could not read {IMAGE_PATH}")


def best_detection(detections, label=None):
    if label is not None:
        filtered = [
            d for d in detections
            if d.class_name.lower() == label.lower()
        ]

        if filtered:
            detections = filtered

    if not detections:
        return None

    return max(
        detections,
        key=lambda d: d.confidence,
    )


def crop_detection(image, det, padding=0.15):
    h, w = image.shape[:2]

    bw = det.x2 - det.x1
    bh = det.y2 - det.y1

    pad_x = bw * padding
    pad_y = bh * padding

    x1 = max(0, int(det.x1 - pad_x))
    y1 = max(0, int(det.y1 - pad_y))
    x2 = min(w, int(det.x2 + pad_x))
    y2 = min(h, int(det.y2 + pad_y))

    return image[y1:y2, x1:x2]


print("\n=== Stage 1: Mouse ===")

mouse_detections = detector.detect(
    image,
    prompt=MOUSE_PROMPT,
)

mouse = best_detection(mouse_detections)

if mouse is None:
    raise RuntimeError("Mouse was not detected.")

print(mouse)

mouse_crop = crop_detection(
    image,
    mouse,
    padding=0.20,
)

# Enlarge because source resolution is weak.
mouse_crop = cv2.resize(
    mouse_crop,
    None,
    fx=4,
    fy=4,
    interpolation=cv2.INTER_CUBIC,
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "01_mouse_crop.jpg",
    ),
    mouse_crop,
)


print("\n=== Stage 2: Head ===")

head_detections = detector.detect(
    mouse_crop,
    prompt=HEAD_PROMPT,
)

for detection in head_detections:
    print(detection)

head = best_detection(
    head_detections,
    label="head",
)

if head is None:
    raise RuntimeError(
        "No separate 'head' detection found."
    )


head_vis = mouse_crop.copy()

cv2.rectangle(
    head_vis,
    (int(head.x1), int(head.y1)),
    (int(head.x2), int(head.y2)),
    (0, 255, 0),
    3,
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "02_head_detection.jpg",
    ),
    head_vis,
)


head_crop = crop_detection(
    mouse_crop,
    head,
    padding=0.20,
)

# Enlarge head again.
head_crop = cv2.resize(
    head_crop,
    None,
    fx=3,
    fy=3,
    interpolation=cv2.INTER_CUBIC,
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "03_head_crop.jpg",
    ),
    head_crop,
)


print("\n=== Stage 3: Nose/Snout ===")

for prompt in PART_PROMPTS:

    print(f"\nPrompt: {prompt}")

    detections = detector.detect(
        head_crop,
        prompt=prompt,
    )

    for detection in detections:
        print(detection)

    vis = head_crop.copy()

    for det in detections:

        cv2.rectangle(
            vis,
            (int(det.x1), int(det.y1)),
            (int(det.x2), int(det.y2)),
            (0, 255, 0),
            3,
        )

        label = (
            f"{det.class_name} "
            f"{det.confidence:.2f}"
        )

        cv2.putText(
            vis,
            label,
            (
                int(det.x1),
                max(25, int(det.y1) - 10),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    filename = (
        "04_"
        + prompt.replace(" ", "_")
        + ".jpg"
    )

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            filename,
        ),
        vis,
    )


print(
    "\nSaved results to:",
    OUTPUT_DIR,
)