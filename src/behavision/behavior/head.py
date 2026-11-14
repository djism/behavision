import numpy as np


def mask_principal_axis(mask):
    """
    Estimate the major body axis of the mouse mask.

    Returns:
        {
            "centroid": (x, y),
            "axis": (ax, ay),
            "endpoint_a": (x, y),
            "endpoint_b": (x, y),
        }

    or None if the mask is too small or numerically invalid.
    """

    y_coords, x_coords = np.where(mask)

    if len(x_coords) < 10:
        return None

    # Use float64 for numerical stability.
    points = np.column_stack(
        [x_coords, y_coords]
    ).astype(np.float64)

    centroid = points.mean(axis=0)

    centered = points - centroid

    covariance = np.cov(
        centered,
        rowvar=False,
    )

    if not np.all(np.isfinite(covariance)):
        return None

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    if not np.all(np.isfinite(eigenvalues)):
        return None

    if not np.all(np.isfinite(eigenvectors)):
        return None

    major_axis = eigenvectors[
        :, np.argmax(eigenvalues)
    ]

    if not np.all(np.isfinite(major_axis)):
        return None

    projections = centered @ major_axis

    if not np.all(np.isfinite(projections)):
        return None

    min_projection = float(
        projections.min()
    )

    max_projection = float(
        projections.max()
    )

    endpoint_a = (
        centroid
        + major_axis * min_projection
    )

    endpoint_b = (
        centroid
        + major_axis * max_projection
    )

    if not np.all(
        np.isfinite(endpoint_a)
    ):
        return None

    if not np.all(
        np.isfinite(endpoint_b)
    ):
        return None

    return {
        "centroid": (
            float(centroid[0]),
            float(centroid[1]),
        ),

        "axis": (
            float(major_axis[0]),
            float(major_axis[1]),
        ),

        "endpoint_a": (
            float(endpoint_a[0]),
            float(endpoint_a[1]),
        ),

        "endpoint_b": (
            float(endpoint_b[0]),
            float(endpoint_b[1]),
        ),
    }


def compute_recent_motion(
    centroid_history,
    min_displacement_px=2.0,
):
    """
    Estimate motion direction using several recent centroids.

    This is more stable than using only one-frame displacement.
    """

    if len(centroid_history) < 2:
        return None

    old = np.array(
        centroid_history[0],
        dtype=np.float64,
    )

    current = np.array(
        centroid_history[-1],
        dtype=np.float64,
    )

    motion = current - old

    if not np.all(
        np.isfinite(motion)
    ):
        return None

    magnitude = float(
        np.linalg.norm(motion)
    )

    if magnitude < min_displacement_px:
        return None

    return motion / magnitude


def _endpoint_direction(
    endpoint,
    centroid,
):
    vector = np.array(
        [
            endpoint[0] - centroid[0],
            endpoint[1] - centroid[1],
        ],
        dtype=np.float64,
    )

    if not np.all(
        np.isfinite(vector)
    ):
        return None

    norm = float(
        np.linalg.norm(vector)
    )

    if norm <= 1e-12:
        return None

    return vector / norm


def choose_front_endpoint(
    endpoint_a,
    endpoint_b,
    centroid,
    motion_direction=None,
    previous_front=None,
):
    """
    Decide which principal-axis endpoint is the mouse front.

    Strategy:
    1. When there is clear recent motion, choose the endpoint
       aligned with motion.
    2. When the animal becomes stationary, preserve the previous
       front orientation by selecting the current endpoint closest
       to the previous front position.
    3. If there is no motion history and no previous front,
       orientation is unknown.
    """

    # --------------------------------------------
    # Case 1: clear recent movement
    # --------------------------------------------

    if motion_direction is not None:

        direction_a = _endpoint_direction(
            endpoint_a,
            centroid,
        )

        direction_b = _endpoint_direction(
            endpoint_b,
            centroid,
        )

        if (
            direction_a is not None
            and direction_b is not None
        ):

            score_a = float(
                np.dot(
                    direction_a,
                    motion_direction,
                )
            )

            score_b = float(
                np.dot(
                    direction_b,
                    motion_direction,
                )
            )

            if score_a >= score_b:
                return endpoint_a

            return endpoint_b

    # --------------------------------------------
    # Case 2: stationary animal
    #
    # Maintain temporal orientation.
    # --------------------------------------------

    if previous_front is not None:

        previous_front_array = np.array(
            previous_front,
            dtype=np.float64,
        )

        endpoint_a_array = np.array(
            endpoint_a,
            dtype=np.float64,
        )

        endpoint_b_array = np.array(
            endpoint_b,
            dtype=np.float64,
        )

        if not np.all(
            np.isfinite(
                previous_front_array
            )
        ):
            return None

        distance_a = float(
            np.linalg.norm(
                endpoint_a_array
                - previous_front_array
            )
        )

        distance_b = float(
            np.linalg.norm(
                endpoint_b_array
                - previous_front_array
            )
        )

        if distance_a <= distance_b:
            return endpoint_a

        return endpoint_b

    # --------------------------------------------
    # Not enough information yet.
    # --------------------------------------------

    return None