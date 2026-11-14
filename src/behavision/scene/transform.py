import json

import pandas as pd

from behavision.scene.arena import (
    ArenaCalibration,
    ArenaPoint,
)


def load_arena_calibration(
    scene_path: str,
) -> ArenaCalibration:

    with open(
        scene_path,
        "r",
    ) as file:

        scene = json.load(
            file
        )


    corners = (
        scene[
            "arena"
        ][
            "corners"
        ]
    )


    return ArenaCalibration(
        top_left=ArenaPoint(
            **corners[
                "top_left"
            ]
        ),

        top_right=ArenaPoint(
            **corners[
                "top_right"
            ]
        ),

        bottom_right=ArenaPoint(
            **corners[
                "bottom_right"
            ]
        ),

        bottom_left=ArenaPoint(
            **corners[
                "bottom_left"
            ]
        ),
    )


def transform_trajectory(
    trajectory_path: str,
    scene_path: str,
    output_path: str,
):

    calibration = (
        load_arena_calibration(
            scene_path
        )
    )


    df = pd.read_csv(
        trajectory_path
    )


    arena_x = []
    arena_y = []


    for _, row in df.iterrows():

        x_norm, y_norm = (
            calibration.camera_to_arena(
                x=float(
                    row["x"]
                ),

                y=float(
                    row["y"]
                ),
            )
        )

        arena_x.append(
            x_norm
        )

        arena_y.append(
            y_norm
        )


    df["arena_x"] = (
        arena_x
    )

    df["arena_y"] = (
        arena_y
    )


    df.to_csv(
        output_path,
        index=False,
    )


    return df