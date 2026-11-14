import json

import pandas as pd

from behavision.behavior.motion import (
    add_motion_features,
    summarize_motion,
)


TRAJECTORY_PATH = (
    "data/outputs/Trial1/"
    "trajectory_arena.csv"
)

OUTPUT_PATH = (
    "data/outputs/Trial1/"
    "trajectory_features.csv"
)

SUMMARY_PATH = (
    "data/outputs/Trial1/"
    "motion_summary.json"
)


FPS = 26.53257978723404


df = pd.read_csv(
    TRAJECTORY_PATH
)


features = add_motion_features(
    df=df,
    fps=FPS,
    smoothing_window=5,
    movement_threshold=0.002,
)


summary = summarize_motion(
    df=features,
    fps=FPS,
)


features.to_csv(
    OUTPUT_PATH,
    index=False,
)


with open(
    SUMMARY_PATH,
    "w",
) as file:

    json.dump(
        summary,
        file,
        indent=2,
    )


print(
    "\n=== Behavision Motion Analysis ==="
)


print(
    "Frames:",
    summary["total_frames"]
)


print(
    "Duration:",
    round(
        summary["duration_s"],
        2,
    ),
    "seconds",
)


print(
    "Total normalized distance:",
    round(
        summary[
            "total_normalized_distance"
        ],
        3,
    ),
)


print(
    "Mean speed:",
    round(
        summary["mean_speed"],
        4,
    ),
)


print(
    "Median speed:",
    round(
        summary["median_speed"],
        4,
    ),
)


print(
    "Maximum speed:",
    round(
        summary["max_speed"],
        4,
    ),
)


print(
    "Moving time:",
    round(
        summary["moving_time_s"],
        2,
    ),
    "seconds",
)


print(
    "Stationary time:",
    round(
        summary["stationary_time_s"],
        2,
    ),
    "seconds",
)


print(
    "Moving fraction:",
    round(
        summary["moving_fraction"],
        3,
    ),
)


print(
    "\nSaved:",
    OUTPUT_PATH
)


print(
    "Saved:",
    SUMMARY_PATH
)