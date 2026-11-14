import math

import numpy as np


def mask_centroid(
    mask: np.ndarray,
) -> tuple[float, float] | None:

    y_coords, x_coords = np.where(mask)

    if len(x_coords) == 0:
        return None

    cx = float(np.mean(x_coords))
    cy = float(np.mean(y_coords))

    return cx, cy


def mask_area(
    mask: np.ndarray,
) -> int:
    return int(np.count_nonzero(mask))


def displacement(
    previous: tuple[float, float] | None,
    current: tuple[float, float] | None,
) -> float | None:

    if previous is None or current is None:
        return None

    x1, y1 = previous
    x2, y2 = current

    return math.sqrt(
        (x2 - x1) ** 2
        + (y2 - y1) ** 2
    )


def speed(
    displacement_pixels: float | None,
    delta_time: float,
) -> float | None:

    if (
        displacement_pixels is None
        or delta_time <= 0
    ):
        return None

    return (
        displacement_pixels
        / delta_time
    )


def area_ratio(
    previous_area: int | None,
    current_area: int,
) -> float | None:

    if (
        previous_area is None
        or previous_area <= 0
    ):
        return None

    return (
        current_area
        / previous_area
    )