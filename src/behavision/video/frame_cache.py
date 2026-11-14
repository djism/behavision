import os

import cv2

from behavision.video.reader import (
    VideoReader,
)


class FrameCache:
    def __init__(
        self,
        output_dir: str,
    ):
        self.output_dir = (
            output_dir
        )

    def build(
        self,
        video_path: str,
        overwrite: bool = False,
    ) -> dict:

        os.makedirs(
            self.output_dir,
            exist_ok=True,
        )

        reader = VideoReader(
            video_path
        )

        metadata = {
            "fps": reader.fps,
            "frame_count": (
                reader.frame_count
            ),
            "width": reader.width,
            "height": reader.height,
            "duration": reader.duration,
        }

        existing_frames = [
            name
            for name in os.listdir(
                self.output_dir
            )
            if name.endswith(".jpg")
        ]

        if (
            existing_frames
            and not overwrite
        ):
            print(
                "Frame cache already exists:",
                self.output_dir,
            )

            reader.release()

            return metadata

        print(
            "Building frame cache..."
        )

        count = 0

        while True:
            frame = reader.read_frame()

            if frame is None:
                break

            output_path = os.path.join(
                self.output_dir,
                f"{frame.index:06d}.jpg",
            )

            success = cv2.imwrite(
                output_path,
                frame.image,
            )

            if not success:
                reader.release()

                raise RuntimeError(
                    f"Could not save "
                    f"frame {frame.index}"
                )

            count += 1

            if count % 500 == 0:
                print(
                    f"Cached {count}/"
                    f"{reader.frame_count}"
                    f" frames"
                )

        reader.release()

        print(
            f"Cached {count} frames "
            f"to {self.output_dir}"
        )

        return metadata