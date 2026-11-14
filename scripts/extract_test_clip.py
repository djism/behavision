import os

import cv2

from behavision.video.reader import VideoReader


VIDEO_PATH = (
    "/Users/dhananjaysharma/"
    "Desktop/videos/Trial1.mp4"
)

OUTPUT_DIR = (
    "data/sam2_test_frames"
)

START_SECONDS = 175
DURATION_SECONDS = 10


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


reader = VideoReader(
    VIDEO_PATH
)


start_frame = int(
    START_SECONDS * reader.fps
)

end_frame = int(
    (
        START_SECONDS
        + DURATION_SECONDS
    )
    * reader.fps
)


reader.seek(
    start_frame
)


local_index = 0


while (
    reader.current_frame
    < end_frame
):

    frame = reader.read_frame()

    if frame is None:
        break

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{local_index:05d}.jpg",
    )

    cv2.imwrite(
        output_path,
        frame.image,
    )

    local_index += 1


reader.release()


print(
    f"Saved {local_index} frames "
    f"to {OUTPUT_DIR}"
)