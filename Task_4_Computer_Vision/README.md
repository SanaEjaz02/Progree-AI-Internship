# Task 4: Computer Vision Object Detector & Frame-by-Frame Segmenter

This mini project implements a reproducible OpenCV pipeline for object extraction, tracking, annotation, and frame-level performance measurement.

## Video source

No live camera is required. The pipeline generates `artifacts/synthetic_input.mp4`, a deterministic 90-frame, 640x360 test scene containing moving orange and green circles and a blue rectangle. This keeps the benchmark offline, repeatable, and easy to inspect.

## Pipeline

1. Generate/read video frames.
2. Apply Gaussian smoothing, adaptive Gaussian thresholding, morphology, and contour extraction.
3. Classify detected regions by dominant BGR color as synthetic object classes.
4. Track detections with nearest-centroid IDs.
5. Overlay class counts, track IDs, confidence, FPS, and inference time.
6. Export annotated video, sample frames, and `frame_metrics.csv`.

A YOLO model is intentionally optional here: the synthetic benchmark isolates classical CV behavior and avoids downloading weights. The detector boundary can be replaced by a YOLO adapter for natural video later.

## Run

```powershell
py -3.14 main.py
py -3.14 -m pytest tests -q
```

Outputs are written under `artifacts/`.
