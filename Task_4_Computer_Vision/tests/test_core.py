from pathlib import Path

import cv2
import numpy as np

from detection.classical import detect_contours
from metrics.collector import FrameMetric, MetricsCollector
from tracking.centroid import CentroidTracker
from video_input.synthetic import generate_synthetic_video, iter_video_frames


def test_adaptive_contour_detector_finds_synthetic_objects():
    frame = np.full((360, 640, 3), (24, 30, 42), dtype=np.uint8)
    cv2.circle(frame, (100, 100), 30, (0, 205, 255), -1)
    detections, mask = detect_contours(frame)
    assert mask.shape == (360, 640)
    assert len(detections) >= 1
    assert all(item.confidence > 0 for item in detections)


def test_centroid_tracker_reuses_id_for_nearby_detection():
    frame = np.full((100, 100, 3), 20, dtype=np.uint8)
    cv2.circle(frame, (30, 30), 15, (0, 205, 255), -1)
    first, _ = detect_contours(frame, min_area=100)
    tracker = CentroidTracker()
    first_tracks = tracker.update(first)
    frame[:] = 20
    cv2.circle(frame, (35, 30), 15, (0, 205, 255), -1)
    second, _ = detect_contours(frame, min_area=100)
    second_tracks = tracker.update(second)
    assert first_tracks and second_tracks
    assert first_tracks[0].track_id == second_tracks[0].track_id


def test_metrics_export_and_summary(tmp_path: Path):
    collector = MetricsCollector()
    collector.add(FrameMetric(0, 25.0, 40.0, 0.9, 2))
    collector.add(FrameMetric(1, 20.0, 50.0, 0.8, 1))
    output = collector.export_csv(tmp_path / "metrics.csv")
    assert output.exists()
    assert collector.summary()["frames"] == 2
    assert collector.summary()["mean_objects"] == 1.5


def test_synthetic_video_generation_and_readback(tmp_path: Path):
    video = generate_synthetic_video(tmp_path / "input.mp4", frame_count=4)
    frames = list(iter_video_frames(video))
    assert len(frames) == 4
    assert frames[0][1].shape == (360, 640, 3)
