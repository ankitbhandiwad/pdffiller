from __future__ import annotations

import os
from typing import List, Optional, Tuple

from google.cloud import documentai_v1 as documentai


def _get_config():
    project_id = os.getenv("DOC_AI_PROJECT_ID", "")
    location = os.getenv("DOC_AI_LOCATION", "us")
    processor_id = os.getenv("DOC_AI_PROCESSOR_ID", "")
    return project_id, location, processor_id


def _process_document(
    payload: bytes, mime_type: str = "application/pdf"
) -> Optional[documentai.Document]:
    project_id, location, processor_id = _get_config()
    if not (project_id and processor_id):
        return None

    client = documentai.DocumentProcessorServiceClient()
    name = client.processor_path(project_id, location, processor_id)
    raw_document = documentai.RawDocument(content=payload, mime_type=mime_type)
    request = documentai.ProcessRequest(name=name, raw_document=raw_document)
    result = client.process_document(request=request)
    return result.document


def extract_text_from_pdf(
    pdf_bytes: bytes, mime_type: str = "application/pdf"
) -> Optional[str]:
    doc = _process_document(pdf_bytes, mime_type=mime_type)
    return (doc.text or "").strip() if doc else None


def _normalized_box(layout, page) -> Optional[Tuple[float, float, float, float]]:
    if not layout or not layout.bounding_poly:
        return None
    poly = layout.bounding_poly
    if poly.normalized_vertices:
        xs = [v.x for v in poly.normalized_vertices]
        ys = [v.y for v in poly.normalized_vertices]
        return min(xs), min(ys), max(xs), max(ys)
    if poly.vertices and getattr(page, "dimension", None):
        width = page.dimension.width or 1
        height = page.dimension.height or 1
        xs = [v.x / width for v in poly.vertices]
        ys = [v.y / height for v in poly.vertices]
        return min(xs), min(ys), max(xs), max(ys)
    return None


def _normalized_box_from_poly(poly, page) -> Optional[Tuple[float, float, float, float]]:
    if not poly:
        return None
    if getattr(poly, "normalized_vertices", None):
        xs = [v.x for v in poly.normalized_vertices]
        ys = [v.y for v in poly.normalized_vertices]
        return min(xs), min(ys), max(xs), max(ys)
    if getattr(poly, "vertices", None) and getattr(page, "dimension", None):
        width = page.dimension.width or 1
        height = page.dimension.height or 1
        xs = [v.x / width for v in poly.vertices]
        ys = [v.y / height for v in poly.vertices]
        return min(xs), min(ys), max(xs), max(ys)
    return None


def _safe_layout(value):
    try:
        return value.layout
    except AttributeError:
        return None


def _box_from_text_anchor(anchor, page) -> Optional[Tuple[float, float, float, float]]:
    if not anchor or not getattr(anchor, "text_segments", None):
        return None
    anchors = [
        (seg.start_index or 0, seg.end_index or 0) for seg in anchor.text_segments
    ]
    boxes = []
    for token in page.tokens:
        token_anchor = getattr(getattr(token, "layout", None), "text_anchor", None)
        if not token_anchor or not getattr(token_anchor, "text_segments", None):
            continue
        for seg in token_anchor.text_segments:
            t_start = seg.start_index or 0
            t_end = seg.end_index or 0
            if any(_segments_overlap(t_start, t_end, a_start, a_end) for a_start, a_end in anchors):
                box = _normalized_box(getattr(token, "layout", None), page)
                if box:
                    boxes.append(box)
                break
    if not boxes:
        return None
    xs0, ys0, xs1, ys1 = zip(*boxes)
    return min(xs0), min(ys0), max(xs1), max(ys1)


def _segments_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return start_a < end_b and start_b < end_a


def _looks_like_choice_box(box: Tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = box
    area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    width = max(0.0, x1 - x0)
    height = max(0.0, y1 - y0)
    if width <= 0 or height <= 0:
        return False
    ratio = width / height if height else 1.0
    if ratio < 0.6 or ratio > 1.7:
        return False
    return 0 < area <= 0.02


def extract_choice_boxes(
    pdf_bytes: bytes,
    mime_type: str = "application/pdf",
) -> List[Tuple[int, List[Tuple[float, float, float, float]]]]:
    doc = _process_document(pdf_bytes, mime_type=mime_type)
    if doc is None:
        return []

    by_page = {}
    for page in doc.pages:
        page_index = page.page_number - 1
        selection_found = False
        # Form fields with selection marks
        for field in page.form_fields:
            value = field.field_value
            if getattr(value, "value_type", "") != "selection_mark":
                continue
            layout = _safe_layout(value)
            box = _normalized_box(layout, page)
            if not box:
                box = _normalized_box_from_poly(
                    getattr(value, "bounding_poly", None), page
                )
            if not box:
                box = _box_from_text_anchor(getattr(value, "text_anchor", None), page)
            if box:
                selection_found = True
                by_page.setdefault(page_index, []).append(box)

        # Fallback to visual elements if available
        if page.visual_elements:
            for element in page.visual_elements:
                type_name = str(getattr(element, "type_", "")).lower()
                is_choice = any(key in type_name for key in ("checkbox", "selection", "radio"))
                box = _normalized_box(element.layout, page)
                if not box:
                    continue
                # If we already have selection marks, only accept explicit choice elements.
                if selection_found and not is_choice:
                    continue
                # Otherwise, use shape heuristics for small square-ish boxes.
                if is_choice or _looks_like_choice_box(box):
                    by_page.setdefault(page_index, []).append(box)

    # Fallback to document entities with page anchors (some processors emit these instead)
    for entity in doc.entities:
        typ = (entity.type_ or "").lower()
        if not any(key in typ for key in ("checkbox", "selection_mark", "choice")):
            continue
        for ref in entity.page_anchor.page_refs:
            page_index = ref.page - 1 if ref.page is not None else 0
            box = _normalized_box_from_poly(ref.bounding_poly, doc.pages[page_index])
            if box:
                by_page.setdefault(page_index, []).append(box)

    return [(page_ix, boxes) for page_ix, boxes in sorted(by_page.items())]


# Legacy dead/stale code moved here for reference.
#
# def extract_text_boxes(
#     pdf_bytes: bytes,
#     mime_type: str = "application/pdf",
# ) -> List[Tuple[int, List[Tuple[float, float, float, float]]]]:
#     doc = _process_document(pdf_bytes, mime_type=mime_type)
#     if doc is None:
#         return []
#
#     by_page = {}
#     for page in doc.pages:
#         page_index = page.page_number - 1
#         for field in page.form_fields:
#             value = field.field_value
#             if getattr(value, "value_type", "") == "selection_mark":
#                 continue
#             box = _extract_field_value_box(field, page)
#             if box:
#                 by_page.setdefault(page_index, []).append(box)
#
#     if not by_page:
#         by_page = _extract_table_boxes(doc)
#
#     return [(page_ix, boxes) for page_ix, boxes in sorted(by_page.items())]
#
#
# def _extract_field_value_box(field, page) -> Optional[Tuple[float, float, float, float]]:
#     value = field.field_value
#     layout = _safe_layout(value)
#     box = _normalized_box(layout, page)
#     if not box:
#         box = _normalized_box_from_poly(getattr(value, "bounding_poly", None), page)
#     if not box:
#         box = _box_from_text_anchor(getattr(value, "text_anchor", None), page)
#     if box:
#         return box
#     name = field.field_name
#     name_box = _normalized_box(_safe_layout(name), page)
#     if not name_box:
#         name_box = _normalized_box_from_poly(getattr(name, "bounding_poly", None), page)
#     if not name_box:
#         name_box = _box_from_text_anchor(getattr(name, "text_anchor", None), page)
#     if not name_box:
#         return None
#     return _infer_box_from_label(name_box)
#
#
# def _infer_box_from_label(
#     name_box: Tuple[float, float, float, float]
# ) -> Optional[Tuple[float, float, float, float]]:
#     x0, y0, x1, y1 = name_box
#     height = max(0.015, y1 - y0)
#     pad = min(0.02, height * 0.6)
#     # If label is on the left, place a box to the right; otherwise place below.
#     if x1 < 0.65:
#         left = min(x1 + pad, 0.95)
#         right = 0.95
#         if right - left < 0.05:
#             return None
#         return left, y0, right, min(y1, 0.98)
#     below_top = min(y1 + pad, 0.97)
#     below_bottom = min(below_top + height * 1.2, 0.98)
#     if below_bottom - below_top < 0.01:
#         return None
#     return 0.05, below_top, 0.95, below_bottom
#
#
# def _anchor_text(doc_text: str, anchor) -> str:
#     if not doc_text or not anchor or not getattr(anchor, "text_segments", None):
#         return ""
#     parts = []
#     for seg in anchor.text_segments:
#         start = seg.start_index or 0
#         end = seg.end_index or 0
#         if end > start:
#             parts.append(doc_text[start:end])
#     return "".join(parts)
#
#
# def _extract_table_boxes(
#     doc: documentai.Document,
# ) -> dict[int, List[Tuple[float, float, float, float]]]:
#     by_page = {}
#     doc_text = doc.text or ""
#     for page in doc.pages:
#         page_index = page.page_number - 1
#         tables = getattr(page, "tables", [])
#         for table in tables:
#             rows = list(getattr(table, "header_rows", [])) + list(
#                 getattr(table, "body_rows", [])
#             )
#             if not rows:
#                 continue
#             max_cols = max((len(row.cells) for row in rows), default=0)
#             if max_cols == 0:
#                 continue
#
#             col_stats = [
#                 {"empty": 0, "total": 0, "area_sum": 0.0} for _ in range(max_cols)
#             ]
#             cell_info = []
#
#             for row in rows:
#                 for col_idx, cell in enumerate(getattr(row, "cells", [])):
#                     layout = getattr(cell, "layout", None)
#                     box = _normalized_box(layout, page)
#                     if not box:
#                         continue
#                     cell_text = _anchor_text(
#                         doc_text, getattr(layout, "text_anchor", None)
#                     )
#                     is_empty = _is_likely_empty_cell(cell_text)
#                     area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
#                     col_stats[col_idx]["total"] += 1
#                     col_stats[col_idx]["area_sum"] += area
#                     if is_empty:
#                         col_stats[col_idx]["empty"] += 1
#                     cell_info.append((col_idx, box, is_empty, cell_text))
#
#             candidate_cols = set()
#             for col_idx, stats in enumerate(col_stats):
#                 total = stats["total"]
#                 if total == 0:
#                     continue
#                 empty_ratio = stats["empty"] / total
#                 avg_area = stats["area_sum"] / total
#                 if empty_ratio >= 0.6 and avg_area >= 0.005:
#                     candidate_cols.add(col_idx)
#
#             if not candidate_cols:
#                 candidate_cols = {idx for idx, *_ in cell_info}
#
#             for col_idx, box, is_empty, _ in cell_info:
#                 if col_idx not in candidate_cols:
#                     continue
#                 if not is_empty:
#                     continue
#                 if _looks_like_choice_box(box):
#                     continue
#                 by_page.setdefault(page_index, []).append(box)
#     return by_page
#
#
# def _is_likely_empty_cell(text: str) -> bool:
#     if not text:
#         return True
#     stripped = text.strip()
#     if not stripped:
#         return True
#     if len(stripped) <= 2 and not any(ch.isalnum() for ch in stripped):
#         return True
#     return False
