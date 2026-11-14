import json
from pathlib import Path
from typing import Any

from mcp.server import MCPServer


mcp = MCPServer(
    "Behavision",
    instructions=(
        "Provides access to Behavision experiment results "
        "such as run summaries and quality-control information."
    ),
)


@mcp.tool()
def get_run_summary(run_id: str) -> dict[str, Any]:
    """
    Return the motion summary for a completed Behavision run.

    Args:
        run_id:
            Name of the run under data/outputs.
            Example: Trial1
    """

    summary_path = (
        Path("data")
        / "outputs"
        / run_id
        / "motion_summary.json"
    )

    if not summary_path.exists():
        return {
            "success": False,
            "run_id": run_id,
            "error": (
                f"No motion_summary.json found for run '{run_id}'."
            ),
        }

    with summary_path.open("r") as f:
        summary = json.load(f)

    return {
        "success": True,
        "run_id": run_id,
        "summary": summary,
    }

@mcp.tool()
def get_qc_failures(run_id: str) -> dict[str, Any]:
    """
    Return tracking QC information for a completed Behavision run.
    """

    qc_path = (
        Path("data")
        / "outputs"
        / run_id
        / "qc_summary.json"
    )

    if not qc_path.exists():
        return {
            "success": False,
            "run_id": run_id,
            "error": (
                f"No qc_summary.json found for run '{run_id}'."
            ),
        }

    with qc_path.open("r") as f:
        qc = json.load(f)

    return {
        "success": True,
        "run_id": run_id,
        "qc": qc,
    }


if __name__ == "__main__":
    mcp.run()