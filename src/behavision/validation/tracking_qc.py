from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class TrackingQCSummary:
    total_frames: int

    first_frame: int
    last_frame: int

    missing_frames: int

    median_displacement: float
    p95_displacement: float
    p99_displacement: float
    max_displacement: float

    median_mask_area: float
    min_mask_area: int
    max_mask_area: int

    jump_flags: int
    area_flags: int


def validate_trajectory(
    csv_path: str,
) -> tuple[
    TrackingQCSummary,
    pd.DataFrame,
]:

    df = pd.read_csv(
        csv_path
    )

    if len(df) == 0:
        raise ValueError(
            "Trajectory CSV is empty"
        )

    # ---------------------------------
    # Frame continuity
    # ---------------------------------

    expected_frames = np.arange(
        int(df["frame"].iloc[0]),
        int(df["frame"].iloc[-1]) + 1,
    )

    actual_frames = (
        df["frame"]
        .astype(int)
        .to_numpy()
    )

    missing = np.setdiff1d(
        expected_frames,
        actual_frames,
    )

    # ---------------------------------
    # Movement statistics
    # ---------------------------------

    displacement_values = (
        df["displacement"]
        .dropna()
    )

    # ---------------------------------
    # Mask statistics
    # ---------------------------------

    area_values = (
        df["mask_area"]
        .dropna()
    )

    # ---------------------------------
    # Existing flags
    # ---------------------------------

    jump_flags = int(
        df["jump_flag"]
        .astype(str)
        .str.lower()
        .eq("true")
        .sum()
    )

    area_flags = int(
        df["area_flag"]
        .astype(str)
        .str.lower()
        .eq("true")
        .sum()
    )

    summary = TrackingQCSummary(
        total_frames=len(df),

        first_frame=int(
            df["frame"].iloc[0]
        ),

        last_frame=int(
            df["frame"].iloc[-1]
        ),

        missing_frames=len(missing),

        median_displacement=float(
            displacement_values.median()
        ),

        p95_displacement=float(
            displacement_values.quantile(
                0.95
            )
        ),

        p99_displacement=float(
            displacement_values.quantile(
                0.99
            )
        ),

        max_displacement=float(
            displacement_values.max()
        ),

        median_mask_area=float(
            area_values.median()
        ),

        min_mask_area=int(
            area_values.min()
        ),

        max_mask_area=int(
            area_values.max()
        ),

        jump_flags=jump_flags,
        area_flags=area_flags,
    )

    return summary, df


def analyze_chunk_boundaries(
    df: pd.DataFrame,
    chunk_size: int,
) -> pd.DataFrame:

    boundaries = []

    last_frame = int(
        df["frame"].max()
    )

    boundary = chunk_size

    while boundary <= last_frame:

        previous_frame = (
            boundary - 1
        )

        previous_row = df[
            df["frame"]
            == previous_frame
        ]

        current_row = df[
            df["frame"]
            == boundary
        ]

        if (
            len(previous_row) == 0
            or len(current_row) == 0
        ):
            boundary += chunk_size
            continue

        previous_row = (
            previous_row.iloc[0]
        )

        current_row = (
            current_row.iloc[0]
        )

        dx = (
            current_row["x"]
            - previous_row["x"]
        )

        dy = (
            current_row["y"]
            - previous_row["y"]
        )

        boundary_displacement = (
            float(
                np.sqrt(
                    dx ** 2
                    + dy ** 2
                )
            )
        )

        previous_area = float(
            previous_row["mask_area"]
        )

        current_area = float(
            current_row["mask_area"]
        )

        boundary_area_ratio = (
            current_area
            / previous_area
        )

        boundaries.append(
            {
                "previous_frame":
                    previous_frame,

                "current_frame":
                    boundary,

                "displacement":
                    boundary_displacement,

                "area_ratio":
                    boundary_area_ratio,
            }
        )

        boundary += chunk_size

    return pd.DataFrame(
        boundaries
    )


def summary_to_dict(
    summary: TrackingQCSummary,
):
    return asdict(summary)