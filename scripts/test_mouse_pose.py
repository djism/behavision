import cv2

from mmpose.apis import MMPoseInferencer


IMAGE_PATH = "data/cache/Trial1/003990.jpg"
OUTPUT_DIR = "data/outputs/Trial1/pose_test"


# We already know the mouse bbox on this frame from Grounding DINO.
# Crop generously around it so the pose model sees mostly the mouse.
x1 = 1130
y1 = 180
x2 = 1330
y2 = 500


image = cv2.imread(IMAGE_PATH)

if image is None:
    raise RuntimeError(
        f"Could not load {IMAGE_PATH}"
    )


crop = image[
    y1:y2,
    x1:x2,
]


crop_path = (
    "data/outputs/Trial1/"
    "mouse_pose_crop.jpg"
)

cv2.imwrite(
    crop_path,
    crop,
)


print(
    "Saved mouse crop:",
    crop_path
)


inferencer = MMPoseInferencer(
    pose2d="animal",
    det_model="whole_image",
    device="cpu",
)


results = inferencer(
    crop_path,
    return_vis=True,
    out_dir=OUTPUT_DIR,
)


result = next(results)


print(
    "\n=== Prediction ==="
)

print(
    result["predictions"]
)

print(
    "\nVisualization saved under:"
)

print(
    OUTPUT_DIR
)