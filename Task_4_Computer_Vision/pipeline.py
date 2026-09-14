"""End-to-end synthetic video detection, tracking, annotation, and metrics."""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from detection.classical import detect_contours
from metrics.collector import FrameMetric, MetricsCollector
from tracking.centroid import CentroidTracker
from video_input.synthetic import FPS, iter_video_frames
from visualization.overlay import annotate_frame


def run_pipeline(video_path: str | Path, output_dir: str | Path) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    annotated_video = output / "annotated_output.mp4"
    frames_dir = output / "sample_frames"
    frames_dir.mkdir(exist_ok=True)
    writer = cv2.VideoWriter(str(annotated_video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (640, 360))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open annotated video writer for {annotated_video}")
    collector = MetricsCollector()
    tracker = CentroidTracker()
    try:
        for frame_index, frame in iter_video_frames(video_path):
            start = time.perf_counter()
            detections, _ = detect_contours(frame)
            tracked = tracker.update(detections)
            inference_ms = (time.perf_counter() - start) * 1000
            fps = 1000.0 / inference_ms if inference_ms else 0.0
            confidence = sum(item.detection.confidence for item in tracked) / len(tracked) if tracked else 0.0
            collector.add(FrameMetric(frame_index, round(fps, 3), round(inference_ms, 3), round(confidence, 3), len(tracked)))
            annotated = annotate_frame(frame, tracked, fps, inference_ms)
            writer.write(annotated)
            if frame_index in (0, 30, 60, 89):
                cv2.imwrite(str(frames_dir / f"frame_{frame_index:03d}.png"), annotated)
    finally:
        writer.release()
    metrics_path = collector.export_csv(output / "frame_metrics.csv")
    return {"video": annotated_video, "metrics": metrics_path, "frames": frames_dir, "summary": collector.summary()}
