import json
import os

from behavision.validation.tracking_qc import (
    validate_trajectory,
    analyze_chunk_boundaries,
    summary_to_dict,
)


TRAJECTORY_PATH = (
    "data/outputs/Trial1/"
    "trajectory.csv"
)

OUTPUT_DIR = (
    "data/outputs/Trial1"
)

CHUNK_SIZE = 300


summary, df = validate_trajectory(
    TRAJECTORY_PATH
)


boundaries = analyze_chunk_boundaries(
    df=df,
    chunk_size=CHUNK_SIZE,
)


print(
    "\n=== Tracking QC ==="
)

print(
    "Frames:",
    summary.total_frames
)

print(
    "Frame range:",
    summary.first_frame,
    "->",
    summary.last_frame,
)

print(
    "Missing frames:",
    summary.missing_frames
)


print(
    "\n=== Movement ==="
)

print(
    "Median displacement:",
    round(
        summary.median_displacement,
        3,
    ),
)

print(
    "95th percentile:",
    round(
        summary.p95_displacement,
        3,
    ),
)

print(
    "99th percentile:",
    round(
        summary.p99_displacement,
        3,
    ),
)

print(
    "Maximum displacement:",
    round(
        summary.max_displacement,
        3,
    ),
)


print(
    "\n=== Mask Area ==="
)

print(
    "Median:",
    round(
        summary.median_mask_area,
        1,
    ),
)

print(
    "Min:",
    summary.min_mask_area,
)

print(
    "Max:",
    summary.max_mask_area,
)


print(
    "\n=== QC Flags ==="
)

print(
    "Jump flags:",
    summary.jump_flags,
)

print(
    "Area flags:",
    summary.area_flags,
)


print(
    "\n=== Chunk Boundaries ==="
)

print(
    boundaries.to_string(
        index=False
    )
)


# -----------------------------------
# Save QC summary
# -----------------------------------

summary_path = os.path.join(
    OUTPUT_DIR,
    "qc_summary.json",
)


with open(
    summary_path,
    "w",
) as file:

    json.dump(
        summary_to_dict(
            summary
        ),
        file,
        indent=2,
    )


# -----------------------------------
# Save boundary report
# -----------------------------------

boundary_path = os.path.join(
    OUTPUT_DIR,
    "chunk_boundaries.csv",
)


boundaries.to_csv(
    boundary_path,
    index=False,
)


print(
    "\nSaved:",
    summary_path
)

print(
    "Saved:",
    boundary_path
)