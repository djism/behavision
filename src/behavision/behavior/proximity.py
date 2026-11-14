import json
import numpy as np
import pandas as pd


def load_object_centers(scene_path: str) -> dict:
    with open(scene_path, "r") as file:
        scene = json.load(file)

    centers = {}

    for obj in scene.get("objects", []):
        centers[obj["id"]] = (
            float(obj["arena_center"]["x"]),
            float(obj["arena_center"]["y"]),
        )

    return centers


def add_object_distances(
    df: pd.DataFrame,
    object_centers: dict,
) -> pd.DataFrame:

    result = df.copy()

    x = result["arena_x_smooth"].to_numpy()
    y = result["arena_y_smooth"].to_numpy()

    distance_columns = []

    for object_id, (obj_x, obj_y) in object_centers.items():

        column = f"distance_to_{object_id}"

        result[column] = np.sqrt(
            (x - obj_x) ** 2
            + (y - obj_y) ** 2
        )

        distance_columns.append(column)

    if distance_columns:
        distance_matrix = result[
            distance_columns
        ].to_numpy()

        nearest_indices = np.argmin(
            distance_matrix,
            axis=1,
        )

        object_ids = list(
            object_centers.keys()
        )

        result["closest_object"] = [
            object_ids[index]
            for index in nearest_indices
        ]

        result["closest_object_distance"] = (
            np.min(
                distance_matrix,
                axis=1,
            )
        )

    return result


def summarize_proximity(
    df: pd.DataFrame,
    object_centers: dict,
    fps: float,
    proximity_threshold: float = 0.10,
) -> dict:

    summary = {
        "proximity_threshold":
            proximity_threshold,
        "objects": {},
    }

    for object_id in object_centers:

        column = (
            f"distance_to_{object_id}"
        )

        near = (
            df[column]
            <= proximity_threshold
        )

        frames_near = int(
            near.sum()
        )

        summary["objects"][object_id] = {
            "minimum_distance":
                float(
                    df[column].min()
                ),

            "mean_distance":
                float(
                    df[column].mean()
                ),

            "frames_near":
                frames_near,

            "time_near_s":
                frames_near / fps,
        }

    return summary