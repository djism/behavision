"""YAML run configuration merged over defaults."""

import copy
from pathlib import Path

import yaml


DEFAULTS = {
    "detection": {
        "prompt": "black mouse",
        "box_threshold": 0.3,
        "text_threshold": 0.25,
        "exclude_regions": [],
        "max_excluded_overlap": 0.6,
    },
    "tracking": {
        "chunk_size": 300,
        "reseed_min_confidence": 0.5,
        "reseed_max_iou": 0.3,
        "redetect_stride": 10,
        "failure_frames": 5,
        "max_recoveries": 30,
    },
    "activity": {
        "blur_kernel": 5,
        "mask_dilation_px": 3,
        "background_margin_px": 25,
        "noise_multiplier": 1.5,
        "smoothing_frames": 3,
        "motion_threshold": "auto",
        "motion_threshold_bounds": [0.005, 0.10],
        "min_bout_s": 2.0,
        "area_ratio_bounds": [0.4, 2.5],
        "summary_bin_s": 300,
        "sensitivity": {
            "threshold_scales": [0.5, 0.75, 1.0, 1.5, 2.0],
            "min_bout_s": [1.0, 2.0, 3.0, 5.0],
        },
    },
    "review": {
        "inactive_clips": 15,
        "active_clips": 15,
        "clip_pad_s": 1.0,
        "max_clip_bout_s": 6.0,
        "seed": 0,
    },
}


def _merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if key not in base:
            raise ValueError(f"Unknown config key: {key}")
        if isinstance(base[key], dict) and isinstance(value, dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None) -> dict:
    config = copy.deepcopy(DEFAULTS)
    if path is None:
        return config
    with open(path, encoding="utf-8") as handle:
        override = yaml.safe_load(handle) or {}
    return _merge(config, override)
