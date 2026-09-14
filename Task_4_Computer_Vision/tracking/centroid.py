"""Simple deterministic nearest-centroid object tracker."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import List

from detection.classical import Detection


@dataclass(frozen=True)
class TrackedDetection:
    track_id: int
    detection: Detection


class CentroidTracker:
    def __init__(self, max_distance: float = 80.0) -> None:
        self.max_distance = max_distance
        self._next_id = 1
        self._centroids: dict[int, tuple[int, int]] = {}

    def update(self, detections: List[Detection]) -> List[TrackedDetection]:
        assigned: set[int] = set()
        tracked: List[TrackedDetection] = []
        for detection in detections:
            candidates = [(hypot(detection.centroid[0] - point[0], detection.centroid[1] - point[1]), track_id) for track_id, point in self._centroids.items() if track_id not in assigned]
            if candidates and min(candidates)[0] <= self.max_distance:
                _, track_id = min(candidates)
            else:
                track_id = self._next_id
                self._next_id += 1
            assigned.add(track_id)
            self._centroids[track_id] = detection.centroid
            tracked.append(TrackedDetection(track_id, detection))
        self._centroids = {item.track_id: item.detection.centroid for item in tracked}
        return tracked
