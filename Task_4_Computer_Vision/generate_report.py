"""Generate the Task 4 Markdown whitepaper and styled PDF report."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.image import imread


ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT.parent / "Reports"


def load_metrics() -> list[dict[str, float]]:
    with (ARTIFACTS / "frame_metrics.csv").open(encoding="utf-8") as handle:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def generate_chart(metrics: list[dict[str, float]]) -> Path:
    output = ARTIFACTS / "metrics_dashboard.png"
    frames = [row["frame_index"] for row in metrics]
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=160)
    figure.suptitle("Task 4 Frame-by-Frame Inference Metrics", fontsize=16, fontweight="bold")
    series = [("fps", "FPS", "#2563eb"), ("inference_ms", "Inference time (ms)", "#d97706"), ("mean_confidence", "Mean confidence", "#059669"), ("object_count", "Objects detected", "#b91c1c")]
    for axis, (key, title, color) in zip(axes.flat, series):
        axis.plot(frames, [row[key] for row in metrics], color=color, linewidth=1.7)
        axis.set_title(title)
        axis.set_xlabel("Frame")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return output


def write_markdown(metrics: list[dict[str, float]], chart: Path) -> Path:
    summary = {
        "mean_fps": sum(row["fps"] for row in metrics) / len(metrics),
        "mean_ms": sum(row["inference_ms"] for row in metrics) / len(metrics),
        "mean_conf": sum(row["mean_confidence"] for row in metrics) / len(metrics),
        "mean_objects": sum(row["object_count"] for row in metrics) / len(metrics),
    }
    sample_names = ["frame_000.png", "frame_030.png", "frame_060.png", "frame_089.png"]
    lines = [
        "# TASK 4: Computer Vision Object Detector & Frame-by-Frame Segmenter",
        "",
        "**Progree Remote Artificial Intelligence Internship | Technical Whitepaper**",
        "",
        "## 1. Executive Summary",
        "",
        "This project implements a reproducible OpenCV pipeline for extracting, classifying, tracking, and visualizing target objects frame by frame. It combines classical computer vision operations with a lightweight centroid tracker and exports both an annotated video and per-frame performance metrics.",
        "",
        "Because no live camera feed was configured, the benchmark uses a deterministic synthetic 90-frame video containing moving orange and green circles and a blue rectangle. This choice makes the experiment offline, repeatable, and transparent while exercising the same video I/O and overlay path used by a camera source.",
        "",
        "## 2. Pipeline Architecture",
        "",
        "```text",
        "Synthetic video source -> Gaussian blur -> adaptive threshold matrix -> morphology",
        "                     -> contours and pixel-bound regions -> color class -> centroid tracker",
        "                     -> dynamic overlay -> annotated MP4 + sample PNGs + metrics CSV",
        "```",
        "",
        "| Stage | Implementation | Output |",
        "|---|---|---|",
        "| Video input | OpenCV `VideoCapture` and deterministic MP4 generator | 90 BGR frames at 640x360 |",
        "| Shape extraction | 5x5 Gaussian filter, adaptive Gaussian threshold, morphology | Foreground mask and contours |",
        "| Object detector | Area filtering plus dominant BGR color classification | Bounding boxes, confidence proxy, centroids |",
        "| Tracking | Nearest-centroid association with 80 px gate | Stable frame-to-frame track IDs |",
        "| Visualization | OpenCV rectangles and text overlays | Counts, IDs, confidence, FPS, latency |",
        "| Evaluation | `MetricsCollector` and CSV writer | One row per processed frame |",
        "",
        "## 3. Methodology: Classical CV and Optional Deep Learning",
        "",
        "Each frame is converted to grayscale, smoothed with a Gaussian kernel, and processed by an adaptive Gaussian threshold. Unlike a single global threshold, the adaptive matrix computes a local threshold for each pixel neighborhood, making the segmentation less sensitive to illumination changes. Morphological opening removes small noise before external contours are extracted.",
        "",
        "The synthetic objects are assigned semantic labels from their dominant BGR channels. Confidence is a geometry-derived proxy based on contour fill ratio, so it is explicitly not a calibrated probability. A YOLOv8n adapter was intentionally not enabled for this benchmark because downloading model weights would make the deliverable less self-contained. The detector boundary is isolated in `detection/classical.py`, allowing a YOLO implementation to replace the contour detector for natural video without changing tracking, overlays, metrics, or report code.",
        "",
        "## 4. Frame-by-Frame Performance",
        "",
        f"The run processed **{len(metrics)} frames**. Mean inference time was **{summary['mean_ms']:.2f} ms/frame**, corresponding to a measured mean throughput of **{summary['mean_fps']:.1f} FPS**. Mean detection confidence was **{summary['mean_conf']:.3f}**, and the pipeline detected **{summary['mean_objects']:.2f} objects/frame** on average.",
        "",
        "![Frame-by-frame metrics dashboard](artifacts/metrics_dashboard.png)",
        "",
        "The CSV records `frame_index`, `fps`, `inference_ms`, `mean_confidence`, and `object_count` for every frame. Latency varies with contour complexity and OpenCV scheduling, while object count remains stable except at boundary and overlap conditions.",
        "",
        "## 5. Annotated Frame Evidence",
        "",
    ]
    for name in sample_names:
        lines.extend([f"### {name}", "", f"![Annotated {name}](artifacts/sample_frames/{name})", ""])
    lines.extend([
        "## 6. Accuracy, Performance, and Trade-offs",
        "",
        "The classical detector is fast, explainable, and has low memory overhead. Its main limitation is that color and contour rules are domain-specific: textured objects, shadows, occlusion, and visually similar backgrounds can reduce recall or produce merged contours. The reported confidence is useful for ranking detections in this controlled benchmark, but it should not be interpreted as a learned probability.",
        "",
        "A YOLOv8n-style detector would improve semantic generalization for people, vehicles, and other natural classes, especially under scale and lighting changes. That improvement would cost model startup time, weight storage, and higher per-frame latency. A practical production design can run YOLO for semantic detections and retain the classical mask as a complementary low-cost segmentation signal.",
        "",
        "## 7. Reproduction and Deliverables",
        "",
        "```powershell",
        "py -3.14 main.py",
        "py -3.14 -m pytest tests -q",
        "```",
        "",
        "Artifacts include `annotated_output.mp4`, `synthetic_input.mp4`, four sample annotated PNGs, `metrics_dashboard.png`, and `frame_metrics.csv`. The project is intentionally self-contained and does not require a live camera or downloaded deep-learning weights.",
        "",
        "## Conclusion",
        "",
        "Task 4 demonstrates a complete frame-by-frame computer vision workflow: pixels are transformed into contours, contours into labeled tracked objects, and each inference is recorded for later performance analysis. The modular boundaries make the pipeline suitable for a future YOLO-backed detector while preserving the tested OpenCV input, tracking, overlay, and metrics infrastructure.",
    ])
    output = ROOT / "REPORT.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def render_pdf(metrics: list[dict[str, float]], chart: Path, output: Path) -> None:
    navy, blue, slate, pale, border = "#1e3a8a", "#2563eb", "#4b5563", "#f1f5f9", "#cbd5e1"
    output.parent.mkdir(parents=True, exist_ok=True)

    def page_header(figure: Figure, section: str, page: int) -> None:
        figure.text(0.08, 0.955, "TASK 4: COMPUTER VISION OBJECT DETECTOR", color=navy, fontsize=15, fontweight="bold", va="top")
        figure.text(0.08, 0.925, "Progree Remote Artificial Intelligence Internship | Technical Whitepaper", color=slate, fontsize=9.5, va="top")
        figure.lines.append(matplotlib.lines.Line2D([0.08, 0.92], [0.905, 0.905], transform=figure.transFigure, color=blue, linewidth=2))
        figure.text(0.08, 0.875, section, color=navy, fontsize=12, fontweight="bold", va="top")
        figure.lines.append(matplotlib.lines.Line2D([0.08, 0.92], [0.055, 0.055], transform=figure.transFigure, color=border, linewidth=0.7))
        figure.text(0.08, 0.035, "Task 4 | Computer Vision", color=slate, fontsize=8)
        figure.text(0.92, 0.035, f"Page {page}", color=slate, fontsize=8, ha="right")

    def save(pdf: PdfPages, figure: Figure) -> None:
        FigureCanvasAgg(figure)
        pdf.savefig(figure)

    summary = (sum(row["fps"] for row in metrics) / len(metrics), sum(row["inference_ms"] for row in metrics) / len(metrics), sum(row["mean_confidence"] for row in metrics) / len(metrics), sum(row["object_count"] for row in metrics) / len(metrics))
    with PdfPages(output) as pdf:
        figure = Figure(figsize=(8.27, 11.69), dpi=150, facecolor="white")
        page_header(figure, "1. Executive Summary & Objective", 1)
        meta = figure.add_axes([0.08, 0.72, 0.84, 0.11]); meta.set_facecolor(pale); meta.set_xticks([]); meta.set_yticks([])
        for spine in meta.spines.values(): spine.set_visible(False)
        meta.spines["left"].set_visible(True); meta.spines["left"].set_color(blue); meta.spines["left"].set_linewidth(4)
        meta.text(0.03, 0.68, "Program: Progree AI Internship | Task: 4 (Computer Vision)", fontsize=9.5, fontweight="bold", color="#1e293b")
        meta.text(0.03, 0.32, "Domain: Object Detection and Segmentation | Source: Deterministic Synthetic Video", fontsize=8.8, color=slate)
        figure.text(0.08, 0.66, "This project implements a reproducible OpenCV pipeline that extracts target contours from video frames, classifies synthetic shapes, tracks them across time, overlays dynamic counts, and exports frame-level inference metrics.", fontsize=10, color="#1a1a1a", va="top", wrap=True)
        figure.text(0.08, 0.52, "Key outcomes", color=navy, fontsize=11, fontweight="bold")
        cards = [("90", "frames processed"), (f"{summary[0]:.1f}", "mean FPS"), (f"{summary[1]:.2f}", "mean ms/frame"), (f"{summary[2]:.2f}", "mean confidence")]
        for index, (value, label) in enumerate(cards):
            axis = figure.add_axes([0.08 + index * 0.215, 0.40, 0.18, 0.075]); axis.set_facecolor("#f8fafc"); axis.set_xticks([]); axis.set_yticks([])
            for spine in axis.spines.values(): spine.set_color(border)
            axis.text(0.08, 0.58, value, fontsize=17, fontweight="bold", color=blue); axis.text(0.08, 0.18, label, fontsize=8.2, color=slate)
        figure.text(0.08, 0.32, "Synthetic video -> adaptive threshold -> contours -> color classes -> centroid tracking -> annotated video + CSV", fontsize=9.5, color=slate)
        save(pdf, figure)

        figure = Figure(figsize=(8.27, 11.69), dpi=150, facecolor="white"); page_header(figure, "2. Pipeline Architecture & Methodology", 2)
        figure.text(0.08, 0.82, "Classical CV processing chain", color=navy, fontsize=11, fontweight="bold")
        steps = [("01", "Video input", "OpenCV reads a deterministic 640x360 MP4."), ("02", "Filtering", "Gaussian blur suppresses high-frequency noise."), ("03", "Segmentation", "Adaptive Gaussian threshold builds local pixel-bound masks."), ("04", "Contours", "Morphology and external contours produce regions."), ("05", "Tracking", "Nearest-centroid association preserves IDs."), ("06", "Output", "Overlay, MP4, sample PNGs, and CSV metrics.")]
        for index, (number, title, body) in enumerate(steps):
            y = 0.75 - index * 0.095
            figure.text(0.10, y, number, color=blue, fontsize=12, fontweight="bold")
            figure.text(0.18, y, title, color=navy, fontsize=10.5, fontweight="bold")
            figure.text(0.38, y, body, color="#1a1a1a", fontsize=9.4)
        figure.text(0.08, 0.16, "The detector boundary is intentionally isolated from tracking and reporting. A YOLOv8n adapter can replace contour extraction for natural classes without changing the rest of the pipeline.", color=slate, fontsize=9.4, wrap=True)
        save(pdf, figure)

        figure = Figure(figsize=(8.27, 11.69), dpi=150, facecolor="white"); page_header(figure, "3. Frame-by-Frame Performance Metrics", 3)
        axis = figure.add_axes([0.08, 0.20, 0.84, 0.64]); axis.imshow(imread(chart)); axis.axis("off")
        figure.text(0.08, 0.14, f"Mean FPS: {summary[0]:.1f} | Mean inference time: {summary[1]:.2f} ms | Mean confidence: {summary[2]:.3f} | Mean objects/frame: {summary[3]:.2f}", color=slate, fontsize=9)
        save(pdf, figure)

        sample_names = ["frame_000.png", "frame_030.png", "frame_060.png", "frame_089.png"]
        for page, names in enumerate((sample_names[:2], sample_names[2:]), start=4):
            figure = Figure(figsize=(8.27, 11.69), dpi=150, facecolor="white"); page_header(figure, "4. Annotated Frame Evidence", page)
            for index, filename in enumerate(names):
                axis = figure.add_axes([0.08, 0.51 - index * 0.30, 0.84, 0.24]); axis.imshow(imread(ARTIFACTS / "sample_frames" / filename)); axis.axis("off")
                figure.text(0.08, 0.485 - index * 0.30, filename, color=navy, fontsize=9, fontweight="bold")
            figure.text(0.08, 0.13, "Dynamic overlay includes class counts, track IDs, confidence, FPS, inference time, and the active classical CV detector label.", color=slate, fontsize=9)
            save(pdf, figure)

        figure = Figure(figsize=(8.27, 11.69), dpi=150, facecolor="white"); page_header(figure, "5. Accuracy, Performance & Reproduction", 6)
        figure.text(0.08, 0.82, "Trade-off discussion", color=navy, fontsize=11, fontweight="bold")
        text = ["Classical CV is fast, explainable, and lightweight, but its color and contour rules are domain-specific. Shadows, occlusion, texture, and similar backgrounds can reduce recall or merge regions.", "A YOLOv8n-style model would improve semantic generalization for natural classes, at the cost of model weights, startup time, and higher latency. A hybrid system can combine learned detections with classical masks.", "Reproduce with: py -3.14 main.py, then py -3.14 -m pytest tests -q. Outputs include annotated_output.mp4, four sample PNGs, metrics_dashboard.png, and frame_metrics.csv."]
        for index, paragraph in enumerate(text): figure.text(0.08, 0.75 - index * 0.16, paragraph, fontsize=10, color="#1a1a1a", wrap=True, va="top")
        save(pdf, figure)


if __name__ == "__main__":
    metric_rows = load_metrics()
    chart_path = generate_chart(metric_rows)
    write_markdown(metric_rows, chart_path)
    render_pdf(metric_rows, chart_path, REPORTS / "Task_04_Computer_Vision_Report.pdf")
    print(f"Wrote {ROOT / 'REPORT.md'}")
    print(f"Wrote {REPORTS / 'Task_04_Computer_Vision_Report.pdf'}")
