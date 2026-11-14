import json
import cv2


FRAME_PATH = "data/cache/Trial1/003990.jpg"
SCENE_PATH = "data/outputs/Trial1/scene.json"
OUTPUT_PATH = "data/outputs/Trial1/scene_overlay.jpg"


# -----------------------------
# Load frame
# -----------------------------

image = cv2.imread(FRAME_PATH)

if image is None:
    raise RuntimeError(
        f"Could not load {FRAME_PATH}"
    )


# -----------------------------
# Load scene calibration
# -----------------------------

with open(SCENE_PATH, "r") as file:
    scene = json.load(file)


# -----------------------------
# Draw arena boundary
# -----------------------------

corners = scene["arena"]["corners"]

arena_points = [
    (
        int(corners["top_left"]["x"]),
        int(corners["top_left"]["y"]),
    ),
    (
        int(corners["top_right"]["x"]),
        int(corners["top_right"]["y"]),
    ),
    (
        int(corners["bottom_right"]["x"]),
        int(corners["bottom_right"]["y"]),
    ),
    (
        int(corners["bottom_left"]["x"]),
        int(corners["bottom_left"]["y"]),
    ),
]


for i in range(4):
    start = arena_points[i]
    end = arena_points[(i + 1) % 4]

    cv2.line(
        image,
        start,
        end,
        (255, 0, 0),
        3,
    )


# -----------------------------
# Draw detected objects
# -----------------------------

for obj in scene.get("objects", []):

    x1, y1, x2, y2 = obj["bbox"]

    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    # Bounding box
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3,
    )

    # Object center
    cx = int(obj["center"]["x"])
    cy = int(obj["center"]["y"])

    cv2.circle(
        image,
        (cx, cy),
        7,
        (0, 0, 255),
        -1,
    )

    # Label
    label = (
        f'{obj["id"]}: '
        f'{obj["label"]} '
        f'({obj["confidence"]:.2f})'
    )

    cv2.putText(
        image,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )


# -----------------------------
# Save output
# -----------------------------

success = cv2.imwrite(
    OUTPUT_PATH,
    image,
)

if not success:
    raise RuntimeError(
        f"Could not save {OUTPUT_PATH}"
    )


print("Saved:", OUTPUT_PATH)