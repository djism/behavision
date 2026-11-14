import numpy as np
import pandas as pd


def add_motion_features(
    df: pd.DataFrame,
    fps: float,
    smoothing_window: int = 5,
    movement_threshold: float = 0.002,
) -> pd.DataFrame:

    if smoothing_window <= 0:
        raise ValueError(
            "smoothing_window must be > 0"
        )

    result = df.copy()

    # --------------------------------
    # 1. Smooth normalized coordinates
    # --------------------------------

    result["arena_x_smooth"] = (
        result["arena_x"]
        .rolling(
            window=smoothing_window,
            center=True,
            min_periods=1,
        )
        .mean()
    )

    result["arena_y_smooth"] = (
        result["arena_y"]
        .rolling(
            window=smoothing_window,
            center=True,
            min_periods=1,
        )
        .mean()
    )

    # --------------------------------
    # 2. Frame-to-frame movement
    # --------------------------------

    dx = result[
        "arena_x_smooth"
    ].diff()

    dy = result[
        "arena_y_smooth"
    ].diff()

    result["arena_displacement"] = (
        np.sqrt(
            dx ** 2
            + dy ** 2
        )
    )

    result.loc[
        result.index[0],
        "arena_displacement",
    ] = 0.0

    # --------------------------------
    # 3. Speed
    # --------------------------------

    result["arena_speed_per_s"] = (
        result["arena_displacement"]
        * fps
    )

    # --------------------------------
    # 4. Cumulative distance
    # --------------------------------

    result["cumulative_distance"] = (
        result[
            "arena_displacement"
        ].cumsum()
    )

    # --------------------------------
    # 5. Moving / stationary
    # --------------------------------

    result["moving"] = (
        result["arena_displacement"]
        >= movement_threshold
    )

    return result


def summarize_motion(
    df: pd.DataFrame,
    fps: float,
) -> dict:

    total_frames = len(df)

    moving_frames = int(
        df["moving"].sum()
    )

    stationary_frames = (
        total_frames
        - moving_frames
    )

    return {
        "total_frames":
            total_frames,

        "duration_s":
            total_frames / fps,

        "total_normalized_distance":
            float(
                df[
                    "arena_displacement"
                ].sum()
            ),

        "mean_speed":
            float(
                df[
                    "arena_speed_per_s"
                ].mean()
            ),

        "median_speed":
            float(
                df[
                    "arena_speed_per_s"
                ].median()
            ),

        "max_speed":
            float(
                df[
                    "arena_speed_per_s"
                ].max()
            ),

        "moving_time_s":
            moving_frames / fps,

        "stationary_time_s":
            stationary_frames / fps,

        "moving_fraction":
            moving_frames
            / total_frames,
    }