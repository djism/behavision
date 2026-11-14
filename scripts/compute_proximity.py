import json

import pandas as pd

from behavision.behavior.proximity import (
    load_object_centers,
    add_object_distances,
    summarize_proximity,
)


FEATURES_PATH = (
    "data/outputs/Trial1/"
    "trajectory_features.csv"
)

SCENE_PATH = (
    "data/outputs/Trial1/"
    "scene.json"
)

OUTPUT_PATH = (
    "data/outputs/Trial1/"
    "trajectory_features.csv"
)

SUMMARY_PATH = (
    "data/outputs/Trial1/"
    "proximity_summary.json"
)


FPS = 26.53257978723404


df = pd.read_csv(
    FEATURES_PATH
)


object_centers = (
    load_object_centers(
        SCENE_PATH
    )
)


features = add_object_distances(
    df=df,
    object_centers=object_centers,
)


summary = summarize_proximity(
    df=features,
    object_centers=object_centers,
    fps=FPS,
    proximity_threshold=0.10,
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
    "\n=== Behavision Object Proximity ==="
)


for object_id, stats in (
    summary["objects"].items()
):

    print(
        f"\n{object_id}"
    )

    print(
        "  Minimum distance:",
        round(
            stats["minimum_distance"],
            4,
        ),
    )

    print(
        "  Mean distance:",
        round(
            stats["mean_distance"],
            4,
        ),
    )

    print(
        "  Time within threshold:",
        round(
            stats["time_near_s"],
            2,
        ),
        "seconds",
    )


print(
    "\nSaved:",
    OUTPUT_PATH
)

print(
    "Saved:",
    SUMMARY_PATH
)