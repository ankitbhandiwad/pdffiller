from __future__ import annotations

import io
import os
from pathlib import Path

from pypdf import PdfReader

from webapp.exceptions import BadInputError, ExternalServiceError, NotFoundError


def _is_checkbox_field(field: dict) -> bool:
    name = str(field.get("name", "")).lower()
    rect = field.get("rect")
    if name.startswith("choicebutton") or "checkbox" in name:
        return True
    if isinstance(rect, list) and len(rect) == 4:
        try:
            width = abs(float(rect[2]) - float(rect[0]))
            height = abs(float(rect[3]) - float(rect[1]))
        except Exception:
            return False
        if width <= 25 and height <= 25:
            return True
        if width <= 35 and height <= 35:
            ratio = width / height if height else 1.0
            return 0.6 <= ratio <= 1.6
    return False


def _parse_rect(rect) -> tuple[float, float, float, float] | None:
    if not isinstance(rect, list) or len(rect) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(value) for value in rect)
    except Exception:
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _collect_text_targets(field_schema: list[dict]) -> tuple[list[dict], list[dict]]:
    text_targets: list[dict] = []
    checkbox_items: list[dict] = []
    for field in field_schema:
        name = str(field.get("name", "")).strip()
        if not name:
            continue
        rect = _parse_rect(field.get("rect"))
        if rect is None:
            continue
        x0, y0, x1, y1 = rect
        item = {
            "name": name,
            "label": str(field.get("label", "")).strip(),
            "page_index": field.get("page_index", 0),
            "rect": field.get("rect"),
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
        }
        if _is_checkbox_field(field):
            item["cx"] = (x0 + x1) / 2
            item["cy"] = (y0 + y1) / 2
            checkbox_items.append(item)
        else:
            item["type"] = "text"
            item["label"] = item["label"] or name
            text_targets.append(item)
    return text_targets, checkbox_items


def _group_checkbox_targets(checkbox_items: list[dict]) -> list[dict]:
    grouped: list[dict] = []
    items_by_page: dict[int, list[dict]] = {}
    for item in checkbox_items:
        items_by_page.setdefault(item["page_index"], []).append(item)

    for page_items in items_by_page.values():
        sorted_items = sorted(page_items, key=lambda item: item["cy"], reverse=True)
        rows: list[dict] = []
        tolerance = 8.0
        for item in sorted_items:
            if not rows or abs(item["cy"] - rows[-1]["center"]) > tolerance:
                rows.append({"center": item["cy"], "items": [item]})
                continue
            rows[-1]["items"].append(item)
            rows[-1]["center"] = sum(entry["cy"] for entry in rows[-1]["items"]) / len(rows[-1]["items"])

        for row in rows:
            ordered = sorted(row["items"], key=lambda item: item["cx"])
            if len(ordered) == 1:
                single = ordered[0]
                label = str(single["label"] or "").strip()
                if label.lower() in ("yes", "no", "checkbox"):
                    label = ""
                grouped.append(
                    {
                        "type": "checkbox",
                        "name": single["name"],
                        "label": label or single["name"],
                        "page_index": single["page_index"],
                        "rect": single["rect"],
                    }
                )
                continue

            index = 0
            while index + 1 < len(ordered):
                left = ordered[index]
                right = ordered[index + 1]
                yes_name = left["name"]
                no_name = right["name"]
                left_label = str(left["label"]).lower()
                right_label = str(right["label"]).lower()
                if "no" in left_label and "yes" in right_label:
                    yes_name, no_name = right["name"], left["name"]
                elif "yes" in left_label and "no" in right_label:
                    yes_name, no_name = left["name"], right["name"]

                label = "Yes/No"
                if left["label"] and right["label"] and left["label"] != right["label"]:
                    label = f"{left['label']} / {right['label']}"
                elif left["label"] or right["label"]:
                    label = left["label"] or right["label"]
                if label.strip().lower() in ("yes/no", "yes / no", "yes", "no"):
                    label = ""

                try:
                    rect = [
                        min(left["x0"], right["x0"]),
                        min(left["y0"], right["y0"]),
                        max(left["x1"], right["x1"]),
                        max(left["y1"], right["y1"]),
                    ]
                except Exception:
                    rect = left["rect"]

                grouped.append(
                    {
                        "type": "yesno",
                        "yes": yes_name,
                        "no": no_name,
                        "label": label,
                        "page_index": left["page_index"],
                        "rect": rect,
                    }
                )
                index += 2

            if index < len(ordered):
                extra = ordered[index]
                grouped.append(
                    {
                        "type": "checkbox",
                        "name": extra["name"],
                        "label": extra["label"] or extra["name"],
                        "page_index": extra["page_index"],
                        "rect": extra["rect"],
                    }
                )
    return grouped


def _overlap_ratio(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return 0.0
    intersection = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
    area_a = max(1.0, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1.0, (bx1 - bx0) * (by1 - by0))
    return intersection / min(area_a, area_b)


def _sort_target(target: dict) -> tuple[int, float, float]:
    rect = target.get("rect") or [0, 0, 0, 0]
    try:
        top = float(rect[3])
        left = float(rect[0])
    except Exception:
        top = 0.0
        left = 0.0
    try:
        page_index = int(target.get("page_index") or 0)
    except Exception:
        page_index = 0
    return (page_index, -top, left)


def build_qa_targets(field_schema: list[dict]) -> list[dict]:
    text_targets, checkbox_items = _collect_text_targets(field_schema)
    checkbox_targets = _group_checkbox_targets(checkbox_items)
    checkbox_rects = [(item["x0"], item["y0"], item["x1"], item["y1"]) for item in checkbox_items]

    filtered_text_targets = []
    for target in text_targets:
        rect = (target["x0"], target["y0"], target["x1"], target["y1"])
        if any(_overlap_ratio(rect, checkbox_rect) > 0.2 for checkbox_rect in checkbox_rects):
            continue
        filtered_text_targets.append(target)

    return sorted(filtered_text_targets + checkbox_targets, key=_sort_target)


def _label_looks_generated(text: str) -> bool:
    lowered = text.strip().lower()
    if not lowered:
        return True
    if lowered in ("yes", "no", "yes/no", "yes / no", "checkbox"):
        return True
    return lowered.startswith(("textbox_", "textbox ", "choicebutton_", "choicebutton "))


def friendly_label(target: dict) -> str:
    label = str(target.get("label", "")).strip()
    name = str(target.get("name", "")).strip()
    candidate = label if not _label_looks_generated(label) else ""
    if not candidate and name and not _label_looks_generated(name):
        candidate = name
    candidate = candidate.replace("_", " ").strip()
    return " ".join(candidate.split()).strip(" :")


def build_context_targets(targets: list[dict]) -> list[dict]:
    return [
        {
            "id": index,
            "type": str(target.get("type", "text")).strip().lower(),
            "label": friendly_label(target),
            "page_index": int(target.get("page_index", 0) or 0),
        }
        for index, target in enumerate(targets)
    ]


def generate_questions_from_context(integrations, pdf_path: Path, targets: list[dict]) -> list[str] | None:
    if not targets:
        return []
    pdf_context = integrations.extract_pdf_context(pdf_path)
    if not isinstance(pdf_context, dict):
        return None
    questions = integrations.generate_questions_for_targets(pdf_context, build_context_targets(targets))
    if len(questions) != len(targets):
        return None
    return [str(question).strip() for question in questions]


def _load_page_sizes(pdf_path: Path) -> list[tuple[float, float]]:
    reader = PdfReader(str(pdf_path))
    page_sizes = []
    for page in reader.pages:
        try:
            page_sizes.append((float(page.mediabox.width), float(page.mediabox.height)))
        except Exception:
            page_sizes.append((0.0, 0.0))
    return page_sizes


def _is_generic_question(text: str, target_type: str) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered:
        return True
    generic_text_prefixes = (
        "please provide the value",
        "please fill in this field",
        "what is the value in the field",
        "what is the value in this field",
        "please provide the value for this field",
    )
    generic_yesno_prefixes = (
        "please answer yes or no for this item",
        "please answer yes or no",
        "should this be checked",
        "should this item be checked",
    )
    if target_type in ("checkbox", "yesno"):
        return any(lowered.startswith(prefix) for prefix in generic_yesno_prefixes)
    return any(lowered.startswith(prefix) for prefix in generic_text_prefixes)


def _build_target_box(target_id: int, target: dict, page_width: float, page_height: float) -> dict | None:
    rect = _parse_rect(target.get("rect"))
    if rect is None or page_width <= 0 or page_height <= 0:
        return None
    x0, y0, x1, y1 = rect
    normalized = {
        "x0": max(0.0, min(1.0, x0 / page_width)),
        "x1": max(0.0, min(1.0, x1 / page_width)),
        "y0": max(0.0, min(1.0, (page_height - y1) / page_height)),
        "y1": max(0.0, min(1.0, (page_height - y0) / page_height)),
    }
    height = max(1.0, y1 - y0)
    if target.get("type") in ("checkbox", "yesno"):
        top_margin = max(height * 0.6, 8.0)
        bottom_margin = max(height * 0.6, 8.0)
        left_margin = 1.0
        right_margin = 0.05
    else:
        top_margin = max(height * 2.2, 20.0)
        bottom_margin = max(height * 0.5, 8.0)
        left_margin = 0.8
        right_margin = 0.2

    context_x0 = max(0.0, normalized["x0"] * page_width - left_margin * page_width)
    context_x1 = min(page_width, normalized["x1"] * page_width + right_margin * page_width)
    context_y0 = max(0.0, y0 - top_margin)
    context_y1 = min(page_height, y1 + bottom_margin)
    context_bbox = {
        "x0": max(0.0, min(1.0, context_x0 / page_width)),
        "x1": max(0.0, min(1.0, context_x1 / page_width)),
        "y0": max(0.0, min(1.0, (page_height - context_y1) / page_height)),
        "y1": max(0.0, min(1.0, (page_height - context_y0) / page_height)),
    }
    fallback_context = {
        "x0": 0.0,
        "x1": 1.0,
        "y0": max(0.0, min(1.0, (page_height - min(page_height, y1 + bottom_margin * 2.0)) / page_height)),
        "y1": max(0.0, min(1.0, (page_height - max(0.0, y0 - top_margin * 2.0)) / page_height)),
    }
    return {
        "id": target_id,
        "type": target.get("type", "text"),
        "page_index": int(target.get("page_index", 0) or 0),
        "bbox": normalized,
        "context_bbox": context_bbox,
        "fallback_context": fallback_context,
        "anchor": {
            "x": (normalized["x0"] + normalized["x1"]) / 2,
            "y": (normalized["y0"] + normalized["y1"]) / 2,
        },
    }


def _group_targets_by_page(targets: list[dict], page_sizes: list[tuple[float, float]]) -> dict[int, list[dict]]:
    pages: dict[int, list[dict]] = {}
    for index, target in enumerate(targets):
        page_index = int(target.get("page_index", 0) or 0)
        if page_index < 0 or page_index >= len(page_sizes):
            return {}
        page_width, page_height = page_sizes[page_index]
        box = _build_target_box(index, target, page_width, page_height)
        if box is None:
            return {}
        pages.setdefault(page_index, []).append(box)
    return pages


def _render_pdf_page(integrations, pdf_path: Path, page_index: int, dpi: int):
    images = integrations.convert_from_path(
        str(pdf_path),
        first_page=page_index + 1,
        last_page=page_index + 1,
        dpi=dpi,
    )
    if not images:
        return None
    return images[0]


def _generate_batch_questions(integrations, image, page_index: int, boxes: list[dict]) -> list[str] | None:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload_targets = [
        {
            "id": item["id"],
            "type": item["type"],
            "page_index": item["page_index"],
            "bbox": item["bbox"],
            "context_bbox": item["context_bbox"],
            "anchor": item["anchor"],
        }
        for item in boxes
    ]
    return integrations.questions_from_image(
        buffer.getvalue(),
        "image/png",
        payload_targets,
        page_index=page_index,
    )


def _crop_bounds(context_bbox: dict, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    left = int(max(0.0, min(1.0, context_bbox["x0"])) * image_width)
    right = int(max(0.0, min(1.0, context_bbox["x1"])) * image_width)
    top = int(max(0.0, min(1.0, context_bbox["y0"])) * image_height)
    bottom = int(max(0.0, min(1.0, context_bbox["y1"])) * image_height)
    if right <= left:
        right = min(image_width, left + 1)
    if bottom <= top:
        bottom = min(image_height, top + 1)
    return left, top, right, bottom


def _relative_bbox(box: dict, left: int, top: int, right: int, bottom: int, image_width: int, image_height: int) -> dict:
    crop_width = max(1.0, right - left)
    crop_height = max(1.0, bottom - top)
    rel_x0 = (box["bbox"]["x0"] * image_width - left) / crop_width
    rel_x1 = (box["bbox"]["x1"] * image_width - left) / crop_width
    rel_y0 = (box["bbox"]["y0"] * image_height - top) / crop_height
    rel_y1 = (box["bbox"]["y1"] * image_height - top) / crop_height
    rel_x0 = max(0.0, min(1.0, rel_x0))
    rel_x1 = max(0.0, min(1.0, rel_x1))
    rel_y0 = max(0.0, min(1.0, rel_y0))
    rel_y1 = max(0.0, min(1.0, rel_y1))
    return {"x0": rel_x0, "x1": rel_x1, "y0": rel_y0, "y1": rel_y1}


def _generate_crop_question(integrations, image, page_index: int, box: dict) -> str | None:
    image_width, image_height = image.size
    question = None
    for context_key in ("context_bbox", "fallback_context"):
        left, top, right, bottom = _crop_bounds(box[context_key], image_width, image_height)
        crop = image.crop((left, top, right, bottom))
        relative_bbox = _relative_bbox(box, left, top, right, bottom, image_width, image_height)
        payload = [
            {
                "id": box["id"],
                "type": box["type"],
                "bbox": relative_bbox,
                "context_bbox": {"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0},
                "anchor": {
                    "x": (relative_bbox["x0"] + relative_bbox["x1"]) / 2,
                    "y": (relative_bbox["y0"] + relative_bbox["y1"]) / 2,
                },
            }
        ]
        buffer = io.BytesIO()
        crop.save(buffer, format="PNG")
        page_questions = integrations.questions_from_image(
            buffer.getvalue(),
            "image/png",
            payload,
            page_index=page_index,
        )
        if not page_questions or len(page_questions) != 1:
            continue
        candidate = page_questions[0]
        if not _is_generic_question(candidate, box["type"]):
            return candidate
        question = candidate
    return question


def generate_questions_from_images(integrations, pdf_path: Path, targets: list[dict]) -> list[str] | None:
    if not targets:
        return []
    if integrations.questions_from_image is None or integrations.convert_from_path is None:
        return None

    page_sizes = _load_page_sizes(pdf_path)
    boxes_by_page = _group_targets_by_page(targets, page_sizes)
    if not boxes_by_page:
        return None

    try:
        dpi = int(os.getenv("QA_QUESTION_DPI", "150"))
    except Exception:
        dpi = 150
    qa_mode = os.getenv("QA_QUESTION_MODE", "full_page").lower()
    use_batch = qa_mode in ("1", "true", "yes", "batch", "full", "full_page")

    questions_by_id: dict[int, str] = {}
    for page_index, boxes in boxes_by_page.items():
        image = _render_pdf_page(integrations, pdf_path, page_index, dpi)
        if image is None:
            return None
        if use_batch:
            page_questions = _generate_batch_questions(integrations, image, page_index, boxes)
            if not page_questions or len(page_questions) != len(boxes):
                return None
            for box, question in zip(boxes, page_questions):
                questions_by_id[box["id"]] = question
            continue

        for box in boxes:
            question = _generate_crop_question(integrations, image, page_index, box)
            if not question:
                return None
            questions_by_id[box["id"]] = question

    if len(questions_by_id) != len(targets):
        return None
    return [questions_by_id[index] for index in range(len(targets))]


def build_question_list(integrations, pdf_path: Path, targets: list[dict]) -> list[str]:
    errors: list[str] = []
    try:
        questions = generate_questions_from_images(integrations, pdf_path, targets)
        if questions and len(questions) == len(targets):
            return questions
    except integrations.openai_question_error as exc:
        errors.append(str(exc))
    except Exception as exc:
        print(f"[qa] image question generation failed: {exc}")
        errors.append(f"OpenAI question generation failed: {exc}")

    try:
        questions = generate_questions_from_context(integrations, pdf_path, targets)
        if questions and len(questions) == len(targets):
            return questions
    except Exception as exc:
        print(f"[qa] context question generation failed: {exc}")
        errors.append(f"LLM question generation failed: {exc}")

    if errors:
        raise ExternalServiceError(errors[-1])
    raise ExternalServiceError("OpenAI returned unusable question output for this document.")


def start_qa_session(paths, integrations, store, file_id: str) -> dict:
    if not integrations.llm_available:
        raise ExternalServiceError("LLM not available")
    pdf_path = paths.data_dir / f"{file_id}.pdf"
    if not pdf_path.exists():
        raise NotFoundError(f"Unknown file id: {file_id}")

    field_schema = integrations.extract_field_schema(pdf_path)
    if not isinstance(field_schema, list):
        field_schema = []
    targets = build_qa_targets(field_schema)
    if not targets:
        raise BadInputError("No fields available for Q&A.")

    questions = build_question_list(integrations, pdf_path, targets)
    session = store.create(file_id=file_id, questions=questions, targets=targets)
    return {"session_id": session.session_id, "questions": questions, "targets": targets}


def record_qa_answer(store, session_id: str, answer: str, index: int | None) -> dict:
    session = store.get(session_id)
    if session is None:
        raise NotFoundError(f"Unknown session id: {session_id}")
    session.record_answer(answer=answer, index=index)
    return {"ok": True}


def complete_qa_session(paths, integrations, store, session_id: str, index: int | None) -> dict:
    if not integrations.llm_available:
        raise ExternalServiceError("LLM not available")
    session = store.get(session_id)
    if session is None:
        raise NotFoundError(f"Unknown session id: {session_id}")
    pdf_path = paths.data_dir / f"{session.file_id}.pdf"
    if not pdf_path.exists():
        raise NotFoundError(f"Unknown file id: {session.file_id}")

    pdf_context = integrations.extract_pdf_context(pdf_path)
    qa_pairs = []
    if isinstance(index, int) and 0 <= index < len(session.questions):
        answer = session.answers[index] if index < len(session.answers) else ""
        qa_pairs.append({"index": index, "question": session.questions[index], "answer": answer})
    else:
        for question_index, question in enumerate(session.questions):
            answer = session.answers[question_index] if question_index < len(session.answers) else ""
            qa_pairs.append({"index": question_index, "question": question, "answer": answer})

    try:
        mapping = integrations.map_answers_to_fields(
            pdf_context,
            qa_pairs,
            target_index=index if isinstance(index, int) else None,
            target_question=qa_pairs[0]["question"] if (isinstance(index, int) and qa_pairs) else None,
        )
    except Exception as exc:
        print(f"[qa] map_answers_to_fields failed: {exc}")
        raise ExternalServiceError(f"LLM answer mapping failed: {exc}") from exc

    return {"mapping": mapping}


def normalize_qa_answer(integrations, question: str, answer: str) -> dict:
    if not integrations.llm_available:
        raise ExternalServiceError("LLM not available")
    if not question:
        raise BadInputError("Missing question")
    try:
        value = integrations.normalize_answer(question, answer)
    except Exception as exc:
        print(f"[qa] normalize_answer failed: {exc}")
        raise ExternalServiceError(f"LLM answer normalization failed: {exc}") from exc
    return {"value": value}
