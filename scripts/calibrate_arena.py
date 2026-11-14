import json
import os

import cv2


FRAME_PATH = (
    "data/cache/Trial1/"
    "000000.jpg"
)

OUTPUT_PATH = (
    "data/outputs/Trial1/"
    "scene.json"
)


points = []


image = cv2.imread(
    FRAME_PATH
)

if image is None:
    raise RuntimeError(
        f"Could not read "
        f"{FRAME_PATH}"
    )


display = image.copy()


def mouse_callback(
    event,
    x,
    y,
    flags,
    param,
):
    if (
        event
        == cv2.EVENT_LBUTTONDOWN
    ):
        if len(points) >= 4:
            return

        points.append(
            (x, y)
        )

        print(
            f"Point {len(points)}: "
            f"({x}, {y})"
        )

        cv2.circle(
            display,
            (x, y),
            8,
            (0, 0, 255),
            -1,
        )

        cv2.putText(
            display,
            str(len(points)),
            (
                x + 10,
                y - 10,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )


cv2.namedWindow(
    "Arena Calibration",
    cv2.WINDOW_NORMAL,
)


cv2.resizeWindow(
    "Arena Calibration",
    1200,
    700,
)


cv2.setMouseCallback(
    "Arena Calibration",
    mouse_callback,
)


print(
    "\nClick the arena FLOOR corners "
    "in this exact order:"
)

print(
    "1. Top-left"
)

print(
    "2. Top-right"
)

print(
    "3. Bottom-right"
)

print(
    "4. Bottom-left"
)

print(
    "\nPress ENTER after selecting "
    "all four points."
)

print(
    "Press R to reset."
)

print(
    "Press ESC to cancel."
)


while True:

    cv2.imshow(
        "Arena Calibration",
        display,
    )

    key = cv2.waitKey(
        20
    ) & 0xFF


    if key == 27:
        print(
            "Calibration cancelled."
        )

        cv2.destroyAllWindows()

        raise SystemExit


    if key in (
        ord("r"),
        ord("R"),
    ):
        points.clear()

        display = (
            image.copy()
        )

        print(
            "Points reset."
        )


    if key in (
        10,
        13,
    ):
        if len(points) != 4:
            print(
                "Please select "
                "exactly 4 points."
            )

            continue

        break


cv2.destroyAllWindows()


scene = {
    "arena": {
        "coordinate_system": (
            "normalized_homography"
        ),

        "corners": {
            "top_left": {
                "x": points[0][0],
                "y": points[0][1],
            },

            "top_right": {
                "x": points[1][0],
                "y": points[1][1],
            },

            "bottom_right": {
                "x": points[2][0],
                "y": points[2][1],
            },

            "bottom_left": {
                "x": points[3][0],
                "y": points[3][1],
            },
        }
    }
}


os.makedirs(
    os.path.dirname(
        OUTPUT_PATH
    ),
    exist_ok=True,
)


with open(
    OUTPUT_PATH,
    "w",
) as file:

    json.dump(
        scene,
        file,
        indent=2,
    )


print(
    "\nSaved arena calibration:"
)

print(
    OUTPUT_PATH
)