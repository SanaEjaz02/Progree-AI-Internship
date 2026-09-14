"""Annotated frame rendering."""

from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

from tracking.centroid import TrackedDetection


def annotate_frame(frame: np.ndarray, tracked: list[TrackedDetection], fps: float, inference_ms: float) -> np.ndarray:
    output = frame.copy()
    colours = [(0, 205, 255), (255, 100, 40), (70, 220, 110)]
    counts = Counter(item.detection.label for item in tracked)
    for index, item in enumerate(tracked):
        x, y, width, height = item.detection.bbox
        colour = colours[index % len(colours)]
        cv2.rectangle(output, (x, y), (x + width, y + height), colour, 2)
        label = f"ID {item.track_id} | {item.detection.label} {item.detection.confidence:.2f}"
        cv2.putText(output, label, (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, colour, 1, cv2.LINE_AA)
    count_text = "Objects: " + ", ".join(f"{label}={count}" for label, count in sorted(counts.items()))
    cv2.rectangle(output, (10, 10), (630, 72), (12, 18, 28), -1)
    cv2.putText(output, count_text or "Objects: 0", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 242, 250), 1, cv2.LINE_AA)
    cv2.putText(output, f"FPS: {fps:.1f} | Inference: {inference_ms:.2f} ms | Classical CV", (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 205, 235), 1, cv2.LINE_AA)
    return output
