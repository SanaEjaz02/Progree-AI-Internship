"""Gaussian-smoothed adaptive threshold and contour extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    area: float
    centroid: tuple[int, int]


def classify_colour(roi: np.ndarray) -> str:
    """Assign a stable semantic class to a synthetic object's dominant hue."""
    if roi.size == 0:
        return "unknown"
    mean_bgr = roi.reshape(-1, 3).mean(axis=0)
    blue, green, red = mean_bgr
    if red > green * 1.15 and red > blue * 1.25:
        return "circle-orange"
    if blue > red * 1.35 and blue > green * 1.15:
        return "rectangle-blue"
    if green > red * 1.15 and green > blue * 1.05:
        return "circle-green"
    return "object"


def detect_contours(frame: np.ndarray, min_area: int = 250) -> tuple[List[Detection], np.ndarray]:
    """Detect foreground contours after Gaussian blur and adaptive thresholding."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        31, 7,
    )
    background_mask = cv2.inRange(frame, np.array([0, 0, 0]), np.array([45, 55, 70]))
    foreground = cv2.bitwise_and(threshold, cv2.bitwise_not(background_mask))
    kernel = np.ones((3, 3), np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections: List[Detection] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        centroid = (
            int(moments["m10"] / moments["m00"]) if moments["m00"] else x + width // 2,
            int(moments["m01"] / moments["m00"]) if moments["m00"] else y + height // 2,
        )
        roi = frame[y:y + height, x:x + width]
        fill_ratio = min(1.0, area / max(1.0, width * height))
        detections.append(Detection(classify_colour(roi), round(0.55 + 0.4 * fill_ratio, 3), (x, y, width, height), area, centroid))
    return sorted(detections, key=lambda item: item.bbox[0]), foreground
