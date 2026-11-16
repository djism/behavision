"""Overview figure, review clips/page, and optional annotated video."""

import html
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from behavision.behavior.activity import runs
from behavision.video.reader import VideoReader


STATE_HEX = {"inactive": "#2a78d6", "active": "#eb6834", "uncertain": "#bdbbb5"}
STATE_BGR = {state: tuple(int(h[i:i + 2], 16) for i in (5, 3, 1)) for state, h in STATE_HEX.items()}


def clock(seconds: float) -> str:
    minutes, rest = divmod(max(0.0, seconds), 60)
    return f"{int(minutes):02d}:{rest:04.1f}"


def plot_overview(timeline: pd.DataFrame, summary: dict, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": muted, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    threshold = summary["calibration"]["motion_threshold"]
    minutes = timeline["time_s"].to_numpy() / 60
    fig = plt.figure(figsize=(12, 8.5), constrained_layout=True)
    grid_spec = fig.add_gridspec(4, 2, height_ratios=[0.6, 2, 2, 0.01], width_ratios=[2.2, 1])

    strip = fig.add_subplot(grid_spec[0, :])
    states = timeline["state"].to_numpy()
    step = float(np.median(np.diff(minutes))) if len(minutes) > 1 else 1 / 60
    for state in ("active", "uncertain", "inactive"):
        spans = [(minutes[s], minutes[e - 1] - minutes[s] + step) for s, e in runs(states == state)]
        strip.broken_barh(spans, (0, 1), facecolors=STATE_HEX[state], linewidth=0)
    strip.set_yticks([])
    strip.set_xlim(minutes[0], minutes[-1] + step)
    strip.set_xlabel("Time (min)")
    strip.legend(handles=[Patch(color=STATE_HEX[s], label=s.capitalize()) for s in ("inactive", "active", "uncertain")],
                 loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, frameon=False)
    strip.set_title(title, loc="left", color=ink, fontsize=11, pad=22)

    trace = fig.add_subplot(grid_spec[1, :], sharex=strip)
    motion = timeline["motion"].to_numpy(dtype=float)
    trace.plot(minutes, np.clip(motion, 1e-4, None), color=STATE_HEX["active"], linewidth=0.6)
    trace.axhline(threshold, color=ink, linewidth=1, linestyle="--")
    trace.annotate(f"threshold {threshold:.3f}", (minutes[-1], threshold), xytext=(0, 4),
                   textcoords="offset points", ha="right", color=ink)
    trace.set_yscale("log")
    trace.set_ylabel("Changed mouse pixels (fraction)")
    trace.grid(axis="y", color=grid, linewidth=0.6)

    hist = fig.add_subplot(grid_spec[2, 0])
    valid = motion[np.isfinite(motion) & (states != "uncertain")]
    if valid.size:
        hist.hist(np.log10(valid + 1e-4), bins=120, color=STATE_HEX["inactive"], edgecolor="#fcfcfb", linewidth=0.3)
    hist.axvline(np.log10(threshold + 1e-4), color=ink, linewidth=1, linestyle="--")
    hist.set_xlabel("log10 changed fraction (tracked frames)")
    hist.set_ylabel("Frames")
    hist.set_title("Motion distribution; dashed line is the still/moving threshold", loc="left", color=muted)

    bins = fig.add_subplot(grid_spec[2, 1])
    rows = [b for b in summary["bins"] if b["inactive_fraction_of_observed"] is not None]
    if rows:
        centers = [(b["bin_start_s"] + b["bin_end_s"]) / 120 for b in rows]
        widths = [(b["bin_end_s"] - b["bin_start_s"]) / 60 * 0.9 for b in rows]
        bins.bar(centers, [b["inactive_fraction_of_observed"] for b in rows], width=widths, color=STATE_HEX["inactive"])
    bins.set_ylim(0, 1)
    bins.set_xlabel("Time (min)")
    bins.set_ylabel("Inactive fraction")
    bins.grid(axis="y", color=grid, linewidth=0.6)
    bins.set_title("Per time bin", loc="left", color=muted)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def ffmpeg_executable() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


class H264Writer:
    """Browser-playable MP4 through an ffmpeg pipe."""

    def __init__(self, path: Path, width: int, height: int, fps: float):
        executable = ffmpeg_executable()
        if executable is None:
            raise RuntimeError("ffmpeg not found; install ffmpeg or `pip install imageio-ffmpeg`")
        self.process = subprocess.Popen(
            [executable, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE)

    def write(self, image) -> None:
        self.process.stdin.write(np.ascontiguousarray(image).tobytes())

    def close(self) -> None:
        self.process.stdin.close()
        if self.process.wait():
            raise RuntimeError("ffmpeg failed to write video")


def draw(image, row, claim: str | None = None):
    output = image.copy()
    state = row["state"] if row is not None else "uncertain"
    color = STATE_BGR[state]
    if row is not None and np.isfinite(row.get("box_x1", np.nan)):
        cv2.rectangle(output, (int(row["box_x1"]), int(row["box_y1"])), (int(row["box_x2"]), int(row["box_y2"])), color, 2)
    if claim is not None:
        cv2.rectangle(output, (2, 2), (output.shape[1] - 3, output.shape[0] - 3), STATE_BGR[claim], 6)
    label = f"{state.upper()}  {clock(row['time_s']) if row is not None else ''}"
    cv2.rectangle(output, (8, 8), (16 + 11 * len(label), 34), (255, 255, 255), -1)
    cv2.putText(output, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return output


def _frame_rows(timeline: pd.DataFrame, tracking: pd.DataFrame) -> dict:
    boxes = tracking[["frame", "box_x1", "box_y1", "box_x2", "box_y2"]] if "box_x1" in tracking else tracking[["frame"]]
    merged = timeline.merge(boxes, on="frame", how="left")
    return {int(r["frame"]): r for r in merged.to_dict("records")}


def write_annotated_video(video_path: str, timeline: pd.DataFrame, tracking: pd.DataFrame, path: Path) -> None:
    rows = _frame_rows(timeline, tracking)
    reader = VideoReader(video_path)
    first, last = int(timeline["frame"].iloc[0]), int(timeline["frame"].iloc[-1])
    writer = H264Writer(path, reader.width, reader.height, reader.fps)
    try:
        reader.seek(first)
        while reader.current_frame <= last:
            frame = reader.read_frame()
            if frame is None:
                break
            writer.write(draw(frame.image, rows.get(frame.index)))
    finally:
        writer.close()
        reader.release()


def sample_review_clips(timeline: pd.DataFrame, bouts: pd.DataFrame, fps: float, config: dict) -> list[dict]:
    cfg = config["review"]
    rng = np.random.default_rng(cfg["seed"])
    clips = []
    if len(bouts):
        for i in sorted(rng.choice(len(bouts), min(cfg["inactive_clips"], len(bouts)), replace=False)):
            bout = bouts.iloc[i]
            end = min(bout["end_s"], bout["start_s"] + cfg["max_clip_bout_s"])
            clips.append({"claim": "inactive", "claim_start_s": float(bout["start_s"]), "claim_end_s": float(end),
                          "bout_id": int(bout["bout_id"]), "bout_duration_s": float(bout["duration_s"])})
    window = round(3 * fps)
    active_runs = [(s, e) for s, e in runs(timeline["state"].to_numpy() == "active") if e - s >= window]
    times = timeline["time_s"].to_numpy()
    if active_runs:
        for i in sorted(rng.choice(len(active_runs), min(cfg["active_clips"], len(active_runs)), replace=False)):
            s, e = active_runs[i]
            start = int(rng.integers(s, e - window + 1))
            clips.append({"claim": "active", "claim_start_s": float(times[start]),
                          "claim_end_s": float(times[start + window - 1] + 1 / fps)})
    rng.shuffle(clips)
    for n, clip in enumerate(clips, 1):
        clip["clip_id"] = f"clip{n:02d}"
    return clips


def write_review(video_path: str, timeline: pd.DataFrame, tracking: pd.DataFrame, bouts: pd.DataFrame,
                 out_dir: Path, config: dict, title: str) -> Path:
    review_dir = out_dir / "review"
    clip_dir = review_dir / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    rows = _frame_rows(timeline, tracking)
    reader = VideoReader(video_path)
    fps, pad = reader.fps, config["review"]["clip_pad_s"]
    first, last = int(timeline["frame"].iloc[0]), int(timeline["frame"].iloc[-1])
    clips = sample_review_clips(timeline, bouts, fps, config)
    try:
        for clip in clips:
            start = max(first, round((clip["claim_start_s"] - pad) * fps))
            stop = min(last, round((clip["claim_end_s"] + pad) * fps))
            claim_frames = range(round(clip["claim_start_s"] * fps), round(clip["claim_end_s"] * fps))
            writer = H264Writer(clip_dir / f"{clip['clip_id']}.mp4", reader.width, reader.height, fps)
            try:
                reader.seek(start)
                while reader.current_frame <= stop:
                    frame = reader.read_frame()
                    if frame is None:
                        break
                    writer.write(draw(frame.image, rows.get(frame.index),
                                      clip["claim"] if frame.index in claim_frames else None))
            finally:
                writer.close()
    finally:
        reader.release()
    pd.DataFrame(clips).to_csv(review_dir / "clips.csv", index=False)
    page = review_dir / "index.html"
    page.write_text(_review_html(clips, title))
    return page


def _review_html(clips: list[dict], title: str) -> str:
    cards = []
    for clip in clips:
        claim = clip["claim"].upper()
        extra = (f" (bout lasts {clip['bout_duration_s']:.1f} s; clip shows up to the first 6 s)"
                 if clip["claim"] == "inactive" and clip["bout_duration_s"] > clip["claim_end_s"] - clip["claim_start_s"] + 1e-6 else "")
        cards.append(f"""
<article class="card" data-id="{clip['clip_id']}" data-claim="{clip['claim']}">
  <video src="clips/{clip['clip_id']}.mp4" controls muted loop playsinline preload="metadata"></video>
  <div class="body">
    <h2>{clip['clip_id']}</h2>
    <p>Pipeline says <strong class="{clip['claim']}">{claim}</strong> while the thick border is shown,
       {clock(clip['claim_start_s'])}–{clock(clip['claim_end_s'])}{html.escape(extra)}.</p>
    <p class="q">Is that right? Grooming, sniffing, and head movement count as active.</p>
    <div class="choices">
      <button data-v="correct">✓ Correct</button><button data-v="wrong">✗ Wrong</button><button data-v="unsure">? Can't tell</button>
    </div>
    <input class="note" placeholder="Optional note (e.g. grooming)">
  </div>
</article>""")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Inactivity Review</title>
<style>
:root {{ --bg:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --muted:#52514e; --line:#e4e3df;
  --inactive:#2a78d6; --active:#eb6834; --pick:#0b0b0b; --pick-ink:#fcfcfb; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --muted:#c3c2b7;
  --line:#383835; --inactive:#3987e5; --active:#d95926; --pick:#f0efec; --pick-ink:#1a1a19; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 -apple-system,system-ui,sans-serif; }}
header {{ position:sticky; top:0; background:var(--bg); border-bottom:1px solid var(--line); padding:12px 16px; z-index:1;
  display:flex; flex-wrap:wrap; gap:8px 16px; align-items:center; }}
header h1 {{ font-size:17px; margin:0; flex:1 1 auto; }}
main {{ max-width:1100px; margin:0 auto; padding:16px; display:grid; gap:16px; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); }}
.card {{ background:var(--surface); border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
.card.done {{ opacity:.7; }}
video {{ width:100%; display:block; background:#000; aspect-ratio:4/3; }}
.body {{ padding:12px; }} h2 {{ font-size:14px; margin:0 0 4px; color:var(--muted); }}
p {{ margin:4px 0; }} .q {{ color:var(--muted); font-size:13px; }}
strong.inactive {{ color:var(--inactive); }} strong.active {{ color:var(--active); }}
.choices {{ display:flex; gap:6px; margin:8px 0; flex-wrap:wrap; }}
button {{ font:inherit; padding:6px 10px; border-radius:6px; border:1px solid var(--line); background:transparent; color:var(--ink); cursor:pointer; }}
button.on {{ background:var(--pick); color:var(--pick-ink); border-color:var(--pick); }}
.note {{ width:100%; font:inherit; padding:6px 8px; border:1px solid var(--line); border-radius:6px; background:transparent; color:var(--ink); }}
</style></head><body>
<header><h1>{html.escape(title)} — check {len(clips)} clips</h1><span id="progress"></span>
<button id="export">Download answers (CSV)</button></header>
<main>{''.join(cards)}</main>
<script>
const KEY = "behavision-review:" + {json.dumps(title)};
let answers = {{}};
try {{ answers = JSON.parse(localStorage.getItem(KEY) || "{{}}"); }} catch (e) {{}}
const save = () => {{ try {{ localStorage.setItem(KEY, JSON.stringify(answers)); }} catch (e) {{}} render(); }};
function render() {{
  document.querySelectorAll(".card").forEach(card => {{
    const a = answers[card.dataset.id] || {{}};
    card.querySelectorAll("button").forEach(b => b.classList.toggle("on", b.dataset.v === a.verdict));
    card.querySelector(".note").value = a.note || "";
    card.classList.toggle("done", !!a.verdict);
  }});
  document.getElementById("progress").textContent =
    Object.values(answers).filter(a => a.verdict).length + " / {len(clips)} answered";
}}
document.querySelectorAll(".card").forEach(card => {{
  const id = card.dataset.id;
  card.querySelectorAll("button").forEach(b => b.onclick = () => {{ answers[id] = {{...answers[id], verdict: b.dataset.v}}; save(); }});
  card.querySelector(".note").onchange = e => {{ answers[id] = {{...answers[id], note: e.target.value}}; save(); }};
}});
document.getElementById("export").onclick = () => {{
  const rows = [["clip_id","claim","verdict","note"]];
  document.querySelectorAll(".card").forEach(c => {{ const a = answers[c.dataset.id] || {{}};
    rows.push([c.dataset.id, c.dataset.claim, a.verdict || "", (a.note || "").replaceAll('"', '""')]); }});
  const csv = rows.map(r => r.map(v => '"' + v + '"').join(",")).join("\\n");
  const link = Object.assign(document.createElement("a"), {{href: URL.createObjectURL(new Blob([csv], {{type: "text/csv"}})),
    download: "review_answers.csv"}});
  link.click();
}};
render();
</script></body></html>
"""
