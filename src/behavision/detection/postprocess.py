from behavision.detection.types import Detection


def compute_iou(
    a: Detection,
    b: Detection,
) -> float:

    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)

    intersection_width = max(
        0.0,
        x2 - x1,
    )

    intersection_height = max(
        0.0,
        y2 - y1,
    )

    intersection = (
        intersection_width
        * intersection_height
    )

    area_a = (
        (a.x2 - a.x1)
        * (a.y2 - a.y1)
    )

    area_b = (
        (b.x2 - b.x1)
        * (b.y2 - b.y1)
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


def non_max_suppression(
    detections: list[Detection],
    iou_threshold: float = 0.5,
) -> list[Detection]:

    if not detections:
        return []

    detections = sorted(
        detections,
        key=lambda d: d.confidence,
        reverse=True,
    )

    kept = []

    while detections:
        best = detections.pop(0)

        kept.append(best)

        remaining = []

        for detection in detections:
            iou = compute_iou(
                best,
                detection,
            )

            if iou < iou_threshold:
                remaining.append(
                    detection
                )

        detections = remaining

    return kept


def select_best_detection(
    detections: list[Detection],
) -> Detection | None:

    if not detections:
        return None

    return max(
        detections,
        key=lambda d: d.confidence,
    )

def drop_excluded(
    detections: list[Detection],
    regions: list,
    max_overlap: float = 0.6,
) -> list[Detection]:
    """Remove detections lying mostly inside fixed [x1, y1, x2, y2] regions."""

    kept = []

    for detection in detections:
        area = max(detection.width * detection.height, 1e-6)

        covered = max(
            (
                max(0.0, min(detection.x2, x2) - max(detection.x1, x1))
                * max(0.0, min(detection.y2, y2) - max(detection.y1, y1))
                for x1, y1, x2, y2 in regions
            ),
            default=0.0,
        )

        if covered / area < max_overlap:
            kept.append(detection)

    return kept
