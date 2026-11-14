from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class VideoFrame:
    index: int
    timestamp: float
    image: np.ndarray


class VideoReader:
    def __init__(self, video_path: str):
        self.video_path = video_path

        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(
                f"Could not open video: {video_path}"
            )

        self.fps = float(
            self.cap.get(cv2.CAP_PROP_FPS)
        )

        self.frame_count = int(
            self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        self.width = int(
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        self.height = int(
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        self.current_frame = 0

    @property
    def duration(self) -> float:
        if self.fps <= 0:
            return 0.0

        return self.frame_count / self.fps

    def read_frame(
        self,
        rgb: bool = False
    ) -> VideoFrame | None:

        success, frame = self.cap.read()

        if not success:
            return None

        index = self.current_frame
        timestamp = index / self.fps

        self.current_frame += 1

        if rgb:
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

        return VideoFrame(
            index=index,
            timestamp=timestamp,
            image=frame
        )

    def frames(
        self,
        stride: int = 1,
        rgb: bool = False
    ):
        if stride <= 0:
            raise ValueError(
                "stride must be greater than 0"
            )

        while True:
            video_frame = self.read_frame(rgb=rgb)

            if video_frame is None:
                break

            if video_frame.index % stride == 0:
                yield video_frame

    def frames_every(
        self,
        seconds: float,
        rgb: bool = False
    ):
        if seconds <= 0:
            raise ValueError(
                "seconds must be greater than 0"
            )

        stride = max(
            1,
            round(seconds * self.fps)
        )

        yield from self.frames(
            stride=stride,
            rgb=rgb
        )

    def seek(self, frame_index: int):
        if frame_index < 0 or frame_index >= self.frame_count:
            raise ValueError(
                f"Frame index must be between 0 and "
                f"{self.frame_count - 1}"
            )

        success = self.cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_index
        )

        if not success:
            raise RuntimeError(
                f"Could not seek to frame {frame_index}"
            )

        self.current_frame = frame_index

    def get_frame(
        self,
        frame_index: int,
        rgb: bool = False
    ) -> VideoFrame | None:

        self.seek(frame_index)

        return self.read_frame(rgb=rgb)

    def release(self):
        self.cap.release()