from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


def detect_grid_textboxes(
    image_path: Path,
    min_width_ratio: float = 0.18,
    min_height_ratio: float = 0.02,
    max_ink_ratio: float = 0.01,
    inset_px: int = 2,
    horiz_k: int = 0,
    vert_k: int = 0,
) -> List[Tuple[float, float, float, float]]:
    image = cv2.imread(str(image_path))
    if image is None:
        return []
    return _detect_from_array(
        image,
        min_width_ratio=min_width_ratio,
        min_height_ratio=min_height_ratio,
        max_ink_ratio=max_ink_ratio,
        inset_px=inset_px,
        horiz_k=horiz_k,
        vert_k=vert_k,
    )


def detect_grid_textboxes_from_image(
    image: np.ndarray,
    min_width_ratio: float = 0.18,
    min_height_ratio: float = 0.02,
    max_ink_ratio: float = 0.01,
    inset_px: int = 2,
    horiz_k: int = 0,
    vert_k: int = 0,
) -> List[Tuple[float, float, float, float]]:
    return _detect_from_array(
        image,
        min_width_ratio=min_width_ratio,
        min_height_ratio=min_height_ratio,
        max_ink_ratio=max_ink_ratio,
        inset_px=inset_px,
        horiz_k=horiz_k,
        vert_k=vert_k,
    )


def _detect_from_array(
    image: np.ndarray,
    min_width_ratio: float,
    min_height_ratio: float,
    max_ink_ratio: float,
    inset_px: int,
    horiz_k: int,
    vert_k: int,
) -> List[Tuple[float, float, float, float]]:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    bin_img = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        10,
    )

    horiz_k = horiz_k or max(20, width // 30)
    vert_k = vert_k or max(20, height // 30)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_k, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_k))

    h_close = cv2.morphologyEx(
        bin_img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    )
    v_close = cv2.morphologyEx(
        bin_img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    )

    horiz = cv2.morphologyEx(h_close, cv2.MORPH_OPEN, h_kernel, iterations=1)
    horiz = cv2.dilate(horiz, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))
    vert = cv2.morphologyEx(v_close, cv2.MORPH_OPEN, v_kernel, iterations=1)
    vert = cv2.dilate(vert, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    grid = cv2.bitwise_or(horiz, vert)

    h_pos = _find_line_positions(horiz, "h", width, height)
    v_pos = _find_line_positions(vert, "v", width, height)

    hough_h, hough_v = _hough_line_positions(bin_img, width, height)
    h_pos.extend(hough_h)
    v_pos.extend(hough_v)
    h_lines = _cluster_positions(h_pos)
    v_lines = _cluster_positions(v_pos)

    if len(h_lines) < 2 or len(v_lines) < 2:
        return []

    cells = _build_cells(h_lines, v_lines)
    ink = cv2.bitwise_and(bin_img, cv2.bitwise_not(grid))
    boxes: List[Tuple[float, float, float, float]] = []

    min_w = min_width_ratio * width
    min_h = min_height_ratio * height
    rows: dict[Tuple[int, int], List[Tuple[float, int, int, int, int]]] = {}

    for x0, y0, w, h in cells:
        if w < min_w or h < min_h:
            continue
        roi = ink[y0 : y0 + h, x0 : x0 + w]
        if roi.size == 0:
            continue
        ink_ratio = cv2.countNonZero(roi) / float(w * h)
        rows.setdefault((y0, y0 + h), []).append((ink_ratio, x0, y0, w, h))

    for (y0, y1), items in rows.items():
        items.sort(key=lambda item: item[0])
        row_mid = (y0 + y1) / 2
        if row_mid < height * 0.45:
            candidates = [item for item in items if item[3] >= min_w]
            if not candidates:
                continue
            ink_ratio, x0, y0, w, h = max(candidates, key=lambda item: item[1])
            left = x0 + inset_px
            right = x0 + w - inset_px
            top = y0 + inset_px
            bottom = y0 + h - inset_px
            if right <= left or bottom <= top:
                continue
            boxes.append(
                (
                    left / width,
                    top / height,
                    right / width,
                    bottom / height,
                )
            )
            continue

        kept = 0
        for ink_ratio, x0, y0, w, h in items:
            if ink_ratio > max_ink_ratio:
                continue
            left = x0 + inset_px
            right = x0 + w - inset_px
            top = y0 + inset_px
            bottom = y0 + h - inset_px
            if right <= left or bottom <= top:
                continue
            boxes.append(
                (
                    left / width,
                    top / height,
                    right / width,
                    bottom / height,
                )
            )
            kept += 1
            if kept >= 2:
                break

    return boxes


def _find_line_positions(
    mask: np.ndarray, orientation: str, width: int, height: int
) -> List[int]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    positions: List[int] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if orientation == "h":
            if w > 10 * h and w > 0.2 * width:
                positions.append(int(y + h / 2))
        else:
            if h > 10 * w and h > 0.2 * height:
                positions.append(int(x + w / 2))
    return positions


def _hough_line_positions(
    bin_img: np.ndarray, width: int, height: int
) -> Tuple[List[int], List[int]]:
    edges = cv2.Canny(bin_img, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=60,
        minLineLength=max(30, min(width, height) // 10),
        maxLineGap=15,
    )
    h_pos: List[int] = []
    v_pos: List[int] = []
    if lines is None:
        return h_pos, v_pos
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            continue
        if abs(dy) <= max(2, int(abs(dx) * 0.15)):
            h_pos.append(int((y1 + y2) / 2))
        elif abs(dx) <= max(2, int(abs(dy) * 0.15)):
            v_pos.append(int((x1 + x2) / 2))
    return h_pos, v_pos


def _cluster_positions(values: List[int], tol: int = 4) -> List[int]:
    if not values:
        return []
    values = sorted(values)
    clusters = [[values[0]]]
    for v in values[1:]:
        if abs(v - clusters[-1][-1]) <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [int(np.median(cluster)) for cluster in clusters]


def _build_cells(h_lines: List[int], v_lines: List[int]) -> List[Tuple[int, int, int, int]]:
    cells: List[Tuple[int, int, int, int]] = []
    h_lines = sorted(h_lines)
    v_lines = sorted(v_lines)
    for yi in range(len(h_lines) - 1):
        for xi in range(len(v_lines) - 1):
            x0 = v_lines[xi]
            x1 = v_lines[xi + 1]
            y0 = h_lines[yi]
            y1 = h_lines[yi + 1]
            w = x1 - x0
            h = y1 - y0
            if w < 8 or h < 8:
                continue
            cells.append((x0, y0, w, h))
    return cells
