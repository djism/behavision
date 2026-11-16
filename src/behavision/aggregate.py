"""Combine per-video outputs: ``behavision-combine RESULTS_ROOT``.

Writes ``all_videos_summary.csv`` (one row per video) and ``all_bouts.csv``
(every inactive bout, tagged by video) into RESULTS_ROOT. Videos without a
``summary.json`` (unfinished or failed) are listed, not silently dropped.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def combine(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    summaries, bouts, missing = [], [], []
    for run in sorted(p for p in root.iterdir() if p.is_dir()):
        if not (run / "video_meta.json").exists():
            continue
        if not (run / "summary.json").exists():
            missing.append(run.name)
            continue
        summary = json.loads((run / "summary.json").read_text())
        qc = json.loads((run / "qc.json").read_text()) if (run / "qc.json").exists() else {}
        calibration = summary.pop("calibration", {})
        summary.pop("bins", None)
        summaries.append({"video_id": run.name, **summary,
                          **{f"calibration_{k}": v for k, v in calibration.items()},
                          "tracked_fraction": qc.get("tracked_fraction"),
                          "qc_warnings": " | ".join(qc.get("warnings", []))})
        table = pd.read_csv(run / "inactive_bouts.csv")
        table.insert(0, "video_id", run.name)
        bouts.append(table)
    return pd.DataFrame(summaries), (pd.concat(bouts, ignore_index=True) if bouts else pd.DataFrame()), missing


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Combine per-video inactivity results.")
    parser.add_argument("root", type=Path, help="Directory containing one output directory per video")
    args = parser.parse_args(argv)
    summaries, bouts, missing = combine(args.root)
    summaries.to_csv(args.root / "all_videos_summary.csv", index=False)
    bouts.to_csv(args.root / "all_bouts.csv", index=False)
    print(f"Combined {len(summaries)} videos and {len(bouts)} bouts into {args.root}")
    if missing:
        print("Not finished (no summary.json): " + ", ".join(missing))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
