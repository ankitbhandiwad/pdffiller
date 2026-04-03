from __future__ import annotations

import json

from pypdf import PdfReader

from webapp.exceptions import BadInputError


def fill_form(integrations, pdf_path, values: dict[str, str]) -> bytes:
    return integrations.fill_pdf(pdf_path, values)


def read_fields_payload(integrations, pdf_path) -> dict:
    return {
        "fields": integrations.read_pdf_fields(pdf_path),
        "field_schema": integrations.extract_field_schema(pdf_path),
    }


def add_field(integrations, pdf_path, data: dict) -> dict:
    try:
        page_index = int(data.get("page_index", 0))
        x0 = float(data.get("x0", 0))
        y0 = float(data.get("y0", 0))
        x1 = float(data.get("x1", 0))
        y1 = float(data.get("y1", 0))
    except (TypeError, ValueError) as exc:
        raise BadInputError("Invalid coordinates") from exc

    label = str(data.get("label", "")).strip() or None
    field_type = str(data.get("field_type") or data.get("type") or "text").lower()
    try:
        if field_type == "checkbox":
            name = integrations.add_checkbox_field(
                pdf_path,
                page_index=page_index,
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                label=label,
            )
        else:
            name = integrations.add_textbox_field(
                pdf_path,
                page_index=page_index,
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                label=label,
            )
    except ValueError as exc:
        raise BadInputError(str(exc)) from exc

    payload = read_fields_payload(integrations, pdf_path)
    payload["name"] = name
    return payload


def remove_field(integrations, pdf_path, data: dict) -> dict:
    names = data.get("names", [])
    rect = data.get("rect")
    page_index = data.get("page_index")
    if isinstance(names, str):
        names = [names]
    if not isinstance(names, list):
        raise BadInputError("Invalid field list")

    removed = 0
    if rect and page_index is not None and names:
        try:
            removed = integrations.remove_field_by_rect(
                pdf_path,
                str(names[0]),
                int(page_index),
                [float(x) for x in rect],
            )
        except Exception:
            removed = 0
    if removed <= 0:
        removed = integrations.remove_fields(pdf_path, [str(name) for name in names])

    payload = read_fields_payload(integrations, pdf_path)
    payload["removed"] = removed
    return payload


def _event(name: str, payload: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n"


def iter_progress_events(integrations, pdf_path, file_id: str):
    if not pdf_path.exists():
        yield _event("error", {"error": "not_found"})
        return

    try:
        total_pages = len(PdfReader(str(pdf_path)).pages)
    except Exception:
        total_pages = 0

    yield _event("log", {"message": "Starting PDF analysis..."})
    print(f"[progress] start file_id={file_id} pages={total_pages}")

    if integrations.check_fields_pdf(pdf_path):
        yield _event("progress", {"current": total_pages, "total": total_pages, "percent": 100})
        yield _event("log", {"message": "Fields already exist. Skipping add."})
        print(f"[progress] fields already exist for {file_id}")
        yield "event: done\ndata: {}\n\n"
        return

    yield _event("log", {"message": "Detecting form fields..."})
    yield _event("log", {"message": "Using commonforms for text fields + DocAI for checkboxes."})
    print(f"[progress] using commonforms + docai for {file_id}")

    try:
        for current, total in integrations.add_textboxes_pdf_with_progress(
            pdf_path,
            use_openai=False,
            use_grid=False,
            skip_commonforms=False,
        ):
            percent = 100 if total == 0 else int((current / total) * 100)
            yield _event("progress", {"current": current, "total": total, "percent": percent})
            yield _event("log", {"message": f"Added field to page {current} of {total}."})
            print(f"[progress] {file_id} page={current} total={total}")
    except Exception as exc:
        yield _event("error", {"message": "Field detection failed."})
        print(f"[progress] error {file_id}: {exc}")
        return

    yield _event("log", {"message": "Finished writing fillable PDF."})
    print(f"[progress] finished {file_id}")
    yield "event: done\ndata: {}\n\n"
