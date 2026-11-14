from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ArenaPoint:
    x: float
    y: float


class ArenaCalibration:
    def __init__(
        self,
        top_left: ArenaPoint,
        top_right: ArenaPoint,
        bottom_right: ArenaPoint,
        bottom_left: ArenaPoint,
    ):
        self.top_left = top_left
        self.top_right = top_right
        self.bottom_right = bottom_right
        self.bottom_left = bottom_left

        self._homography = (
            self._compute_homography()
        )

    def _compute_homography(
        self,
    ) -> np.ndarray:

        source = np.array(
            [
                [
                    self.top_left.x,
                    self.top_left.y,
                ],
                [
                    self.top_right.x,
                    self.top_right.y,
                ],
                [
                    self.bottom_right.x,
                    self.bottom_right.y,
                ],
                [
                    self.bottom_left.x,
                    self.bottom_left.y,
                ],
            ],
            dtype=np.float32,
        )

        destination = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )

        homography = cv2.getPerspectiveTransform(
            source,
            destination,
        )

        return homography

    def camera_to_arena(
        self,
        x: float,
        y: float,
    ) -> tuple[float, float]:

        point = np.array(
            [
                [
                    [x, y]
                ]
            ],
            dtype=np.float32,
        )

        transformed = cv2.perspectiveTransform(
            point,
            self._homography,
        )

        arena_x = float(
            transformed[0][0][0]
        )

        arena_y = float(
            transformed[0][0][1]
        )

        return arena_x, arena_y
    
    def contains_camera_point(
        self,
        x: float,
        y: float,
        margin: float = 0.0,
    ) -> bool:

        arena_x, arena_y = self.camera_to_arena(
            x,
            y,
        )

        return (
            -margin <= arena_x <= 1.0 + margin
            and
            -margin <= arena_y <= 1.0 + margin
        )

    @property
    def homography(
        self,
    ) -> np.ndarray:
        return self._homography.copy()