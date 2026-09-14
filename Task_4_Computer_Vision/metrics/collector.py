"""Frame-level inference metric collection and CSV export."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class FrameMetric:
    frame_index: int
    fps: float
    inference_ms: float
    mean_confidence: float
    object_count: int


class MetricsCollector:
    def __init__(self) -> None:
        self.rows: list[FrameMetric] = []

    def add(self, metric: FrameMetric) -> None:
        self.rows.append(metric)

    def export_csv(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(self.rows[0]).keys()) if self.rows else ["frame_index", "fps", "inference_ms", "mean_confidence", "object_count"])
            writer.writeheader()
            writer.writerows(asdict(row) for row in self.rows)
        return output

    def summary(self) -> dict[str, float]:
        if not self.rows:
            return {"frames": 0, "mean_fps": 0.0, "mean_inference_ms": 0.0, "mean_confidence": 0.0, "mean_objects": 0.0}
        return {"frames": len(self.rows), "mean_fps": sum(row.fps for row in self.rows) / len(self.rows), "mean_inference_ms": sum(row.inference_ms for row in self.rows) / len(self.rows), "mean_confidence": sum(row.mean_confidence for row in self.rows) / len(self.rows), "mean_objects": sum(row.object_count for row in self.rows) / len(self.rows)}
