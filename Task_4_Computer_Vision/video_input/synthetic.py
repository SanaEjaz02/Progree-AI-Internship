"""Generate and read a deterministic synthetic video source."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


FRAME_WIDTH = 640
FRAME_HEIGHT = 360
FPS = 20
FRAME_COUNT = 90


def generate_synthetic_video(output_path: str | Path, frame_count: int = FRAME_COUNT) -> Path:
    """Create a reproducible scene with moving circles and rectangles."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (FRAME_WIDTH, FRAME_HEIGHT)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open video writer for {output}")
    for frame_index in range(frame_count):
        frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), (24, 30, 42), dtype=np.uint8)
        circle_center = (90 + (frame_index * 4) % 470, 105 + int(35 * np.sin(frame_index / 9)))
        rectangle_x = 470 - (frame_index * 3) % 390
        cv2.circle(frame, circle_center, 27, (0, 205, 255), -1)
        cv2.rectangle(frame, (rectangle_x, 225), (rectangle_x + 72, 285), (255, 100, 40), -1)
        cv2.circle(frame, (130 + (frame_index * 2) % 360, 270), 18, (70, 220, 110), -1)
        cv2.putText(frame, "SYNTHETIC CV BENCHMARK", (18, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (215, 225, 240), 1, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    return output


def iter_video_frames(video_path: str | Path) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (zero-based frame index, BGR frame) pairs from a video file."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video source {video_path}")
    index = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            yield index, frame
            index += 1
    finally:
        capture.release()
