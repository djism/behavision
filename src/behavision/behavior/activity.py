"""Inactivity from in-mask pixel change, calibrated per video without labels.

Tracking stores, per frame, how many pixels inside the (dilated) mouse mask
changed by more than each candidate intensity delta, and the same fraction
for background pixels far from the mouse. Analysis then picks the delta from
the background noise level, subtracts frame-wide change (flicker, compression),
median-smooths over a few frames, and calls a frame still when the remaining
changed fraction is below a threshold. Still runs of at least ``min_bout_s`` are inactive bouts; grooming,
sniffing, and head movement change in-mask pixels and therefore count as active.
"""

import math

import cv2
import numpy as np
import pandas as pd


DELTAS = (4, 6, 8, 10, 12, 16, 20, 25, 30, 40)
STATES = ("inactive", "active", "uncertain")


def frame_motion(prev_gray, gray, prev_mask, mask, dilation_px: int, background_margin_px: int) -> dict:
    """Changed-pixel counts between two blurred grayscale frames."""
    roi = (prev_mask | mask).astype(np.uint8)
    if dilation_px > 0:
        roi = cv2.dilate(roi, np.ones((2 * dilation_px + 1,) * 2, np.uint8))
    near = cv2.dilate(roi, np.ones((2 * background_margin_px + 1,) * 2, np.uint8))
    diff = cv2.absdiff(prev_gray, gray)
    roi_hist = np.bincount(diff[roi.astype(bool)], minlength=256)
    bg_hist = np.bincount(diff[~near.astype(bool)], minlength=256)
    roi_above = roi_hist[::-1].cumsum()[::-1]  # roi_above[d] = count of diff >= d
    bg_above = bg_hist[::-1].cumsum()[::-1]
    bg_total = max(int(bg_hist.sum()), 1)
    row = {"roi_px": int(roi_hist.sum())}
    for d in DELTAS:
        row[f"chg_d{d}"] = int(roi_above[d + 1])
        row[f"bg_d{d}"] = float(bg_above[d + 1] / bg_total)
    row["bg_p999"] = min(255, int(np.searchsorted(bg_hist.cumsum(), 0.999 * bg_total)))
    return row


def calibrate_delta(tracking: pd.DataFrame, noise_multiplier: float) -> tuple[int, int]:
    """Pixel delta from the background noise level (its typical 99th percentile).

    The 99th rather than a higher percentile keeps sparse moving background
    (the tether cable, disturbed bedding) from inflating the noise estimate.
    """
    noise = next((d for d in DELTAS if np.nanmedian(tracking[f"bg_d{d}"]) <= 0.01), DELTAS[-1])
    target = noise_multiplier * noise
    return next((d for d in DELTAS if d >= target), DELTAS[-1]), noise


def motion_fraction(tracking: pd.DataFrame, delta: int, smoothing_frames: int = 1) -> pd.Series:
    """Changed fraction of mouse pixels beyond frame-wide change, median-smoothed."""
    roi = tracking["roi_px"].where(tracking["roi_px"] > 0)
    motion = (tracking[f"chg_d{delta}"] / roi - tracking[f"bg_d{delta}"]).clip(lower=0)
    if smoothing_frames > 1:
        smoothed = motion.rolling(smoothing_frames, center=True, min_periods=1).median()
        motion = smoothed.where(motion.notna())
    return motion


def otsu_threshold(values, bounds: tuple[float, float], eps: float = 1e-4) -> float:
    """Valley between still and moving modes of log motion, clipped to bounds."""
    v = np.log10(np.asarray(values, dtype=float)[np.isfinite(values)] + eps)
    if v.size < 10 or v.min() == v.max():
        return float(bounds[0])
    hist, edges = np.histogram(v, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = hist.cumsum()
    w1 = w0[-1] - w0
    m0 = (hist * centers).cumsum()
    with np.errstate(divide="ignore", invalid="ignore"):
        between = w0 * w1 * (m0 / w0 - (m0[-1] - m0) / w1) ** 2
    threshold = 10 ** centers[int(np.nanargmax(between[:-1]))] - eps
    return float(min(max(threshold, bounds[0]), bounds[1]))


def runs(flags) -> list[tuple[int, int]]:
    """(start, end_exclusive) index pairs of consecutive True values."""
    padded = np.concatenate(([False], np.asarray(flags, dtype=bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def uncertain_frames(tracking: pd.DataFrame, motion: pd.Series, area_ratio_bounds) -> np.ndarray:
    area = tracking["mask_area"].where(tracking["tracked"])
    ratio = area / np.nanmedian(area)
    bad_area = (ratio < area_ratio_bounds[0]) | (ratio > area_ratio_bounds[1])
    return (~tracking["tracked"].astype(bool) | bad_area | motion.isna()).to_numpy()


def frame_states(motion, uncertain, fps: float, threshold: float, min_bout_s: float) -> np.ndarray:
    motion = np.asarray(motion, dtype=float)
    still = (motion < threshold) & ~uncertain
    states = np.where(uncertain, "uncertain", "active").astype(object)
    min_frames = max(1, math.ceil(min_bout_s * fps - 1e-9))
    for start, end in runs(still):
        if end - start >= min_frames:
            states[start:end] = "inactive"
    return states


def bout_table(frames, times, states, motion, fps: float) -> pd.DataFrame:
    rows = []
    for start, end in runs(np.asarray(states) == "inactive"):
        rows.append({
            "bout_id": len(rows) + 1,
            "start_frame": int(frames[start]),
            "end_frame": int(frames[end - 1]),
            "start_s": round(float(times[start]), 3),
            "end_s": round(float(times[end - 1]) + 1 / fps, 3),
            "duration_s": round((end - start) / fps, 3),
            "mean_motion": round(float(np.nanmean(motion[start:end])), 5),
        })
    columns = ["bout_id", "start_frame", "end_frame", "start_s", "end_s", "duration_s", "mean_motion"]
    return pd.DataFrame(rows, columns=columns)


def summarize(states, times, bouts: pd.DataFrame, fps: float, bin_s: float) -> dict:
    states = np.asarray(states)
    times = np.asarray(times, dtype=float)

    def totals(mask) -> dict:
        inactive = float((states[mask] == "inactive").sum() / fps)
        active = float((states[mask] == "active").sum() / fps)
        return {
            "inactive_s": round(inactive, 2),
            "active_s": round(active, 2),
            "uncertain_s": round(float((states[mask] == "uncertain").sum() / fps), 2),
            "inactive_fraction_of_observed": round(inactive / (inactive + active), 4) if inactive + active else None,
        }

    durations = bouts["duration_s"]
    summary = {
        "analyzed_s": round(len(states) / fps, 2),
        **totals(np.ones(len(states), dtype=bool)),
        "bouts": int(len(bouts)),
        "mean_bout_s": round(float(durations.mean()), 2) if len(bouts) else None,
        "median_bout_s": round(float(durations.median()), 2) if len(bouts) else None,
        "longest_bout_s": round(float(durations.max()), 2) if len(bouts) else None,
        "latency_to_first_bout_s": round(float(bouts["start_s"].iloc[0] - times[0]), 2) if len(bouts) else None,
    }
    bins = []
    for start in np.arange(times[0] if len(times) else 0, times[-1] + 1 / fps if len(times) else 0, bin_s):
        mask = (times >= start) & (times < start + bin_s)
        if mask.any():
            starts = bouts["start_s"].to_numpy()
            bins.append({"bin_start_s": round(float(start), 2),
                         "bin_end_s": round(float(min(start + bin_s, times[-1] + 1 / fps)), 2),
                         **totals(mask), "bouts_started": int(((starts >= start) & (starts < start + bin_s)).sum())})
    summary["bins"] = bins
    return summary


def sensitivity(motion, uncertain, times, frames, fps: float, threshold: float, scales, min_bouts) -> pd.DataFrame:
    rows = []
    for scale in scales:
        for min_bout in min_bouts:
            states = frame_states(motion, uncertain, fps, threshold * scale, min_bout)
            bouts = bout_table(frames, times, states, np.asarray(motion, dtype=float), fps)
            s = summarize(states, times, bouts, fps, bin_s=float("inf"))
            rows.append({"threshold_scale": scale, "motion_threshold": round(threshold * scale, 5),
                         "min_bout_s": min_bout, "inactive_s": s["inactive_s"],
                         "inactive_fraction_of_observed": s["inactive_fraction_of_observed"],
                         "bouts": s["bouts"], "mean_bout_s": s["mean_bout_s"]})
    return pd.DataFrame(rows)


def analyze(tracking: pd.DataFrame, fps: float, config: dict) -> dict:
    """Return timeline, bouts, summary, and sensitivity tables for one video."""
    cfg = config["activity"]
    tracking = tracking.sort_values("frame").reset_index(drop=True)
    delta, noise = calibrate_delta(tracking, cfg["noise_multiplier"])
    motion = motion_fraction(tracking, delta, cfg["smoothing_frames"])
    uncertain = uncertain_frames(tracking, motion, cfg["area_ratio_bounds"])
    if cfg["motion_threshold"] == "auto":
        threshold = otsu_threshold(motion[~uncertain], cfg["motion_threshold_bounds"])
    else:
        threshold = float(cfg["motion_threshold"])
    frames = tracking["frame"].to_numpy()
    times = tracking["time_s"].to_numpy()
    states = frame_states(motion, uncertain, fps, threshold, cfg["min_bout_s"])
    bouts = bout_table(frames, times, states, motion.to_numpy(), fps)
    summary = summarize(states, times, bouts, fps, cfg["summary_bin_s"])
    summary["calibration"] = {
        "background_noise_p99": noise, "pixel_delta": delta,
        "motion_threshold": round(threshold, 5),
        "motion_threshold_source": "otsu" if cfg["motion_threshold"] == "auto" else "config",
        "min_bout_s": cfg["min_bout_s"],
    }
    timeline = pd.DataFrame({"frame": frames, "time_s": times.round(3), "state": states,
                             "motion": motion.round(5), "x": tracking["x"], "y": tracking["y"],
                             "mask_area": tracking["mask_area"]})
    sens = sensitivity(motion, uncertain, times, frames, fps, threshold,
                       cfg["sensitivity"]["threshold_scales"], cfg["sensitivity"]["min_bout_s"])
    return {"timeline": timeline, "bouts": bouts, "summary": summary, "sensitivity": sens}
