from behavision.scene.transform import (
    transform_trajectory,
)


TRAJECTORY_PATH = (
    "data/outputs/Trial1/"
    "trajectory.csv"
)

SCENE_PATH = (
    "data/outputs/Trial1/"
    "scene.json"
)

OUTPUT_PATH = (
    "data/outputs/Trial1/"
    "trajectory_arena.csv"
)


df = transform_trajectory(
    trajectory_path=(
        TRAJECTORY_PATH
    ),

    scene_path=(
        SCENE_PATH
    ),

    output_path=(
        OUTPUT_PATH
    ),
)


print(
    "Transformed frames:",
    len(df)
)


print(
    "\nArena coordinate ranges:"
)


print(
    "X:",
    round(
        df["arena_x"].min(),
        3,
    ),
    "->",
    round(
        df["arena_x"].max(),
        3,
    ),
)


print(
    "Y:",
    round(
        df["arena_y"].min(),
        3,
    ),
    "->",
    round(
        df["arena_y"].max(),
        3,
    ),
)


print(
    "\nSaved:",
    OUTPUT_PATH
)