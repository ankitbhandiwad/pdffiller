from __future__ import annotations

import json
from pathlib import Path

from pypdf import PdfReader

from webapp.config import AppPaths
from webapp.exceptions import ExternalServiceError, NotFoundError


def pdf_path_for(paths: AppPaths, file_id: str) -> Path:
    return paths.data_dir / f"{file_id}.pdf"


def require_pdf(paths: AppPaths, file_id: str) -> Path:
    pdf_path = pdf_path_for(paths, file_id)
    if not pdf_path.exists():
        raise NotFoundError(f"Unknown file id: {file_id}")
    return pdf_path


def page_image_path_for(paths: AppPaths, file_id: str, page_index: int) -> Path:
    return paths.data_dir / f"{file_id}_page{page_index}.png"


def read_fill_page_context(integrations, pdf_path: Path) -> dict:
    fields = integrations.read_pdf_fields(pdf_path)
    field_schema = integrations.extract_field_schema(pdf_path)
    return {
        "fields": fields,
        "has_fields": bool(fields),
        "field_schema": field_schema,
    }


def read_pdf_info(pdf_path: Path) -> dict:
    try:
        reader = PdfReader(str(pdf_path))
        page_count = len(reader.pages)
        page_sizes = []
        for page in reader.pages:
            media = page.mediabox
            width = float(media.right) - float(media.left)
            height = float(media.top) - float(media.bottom)
            page_sizes.append({"width": width, "height": height})
    except Exception:
        page_count = 0
        page_sizes = []
    return {"page_count": page_count, "page_sizes": page_sizes}


def render_page_image(paths: AppPaths, integrations, file_id: str, page_index: int) -> bytes:
    if integrations.convert_from_path is None:
        raise RuntimeError("pdf2image is unavailable.")

    pdf_path = require_pdf(paths, file_id)
    try:
        reader = PdfReader(str(pdf_path))
        total_pages = len(reader.pages)
    except Exception:
        total_pages = 0
    if page_index < 0 or page_index >= total_pages:
        raise NotFoundError(f"Page {page_index} is out of range.")

    image_path = page_image_path_for(paths, file_id, page_index)
    if not image_path.exists():
        pages = integrations.convert_from_path(
            str(pdf_path),
            first_page=page_index + 1,
            last_page=page_index + 1,
            dpi=150,
        )
        if not pages:
            raise ExternalServiceError("Unable to render PDF page.")
        pages[0].save(image_path, format="PNG")

    return image_path.read_bytes()


def json_bytes(payload: dict) -> bytes:
    return json.dumps(payload).encode("utf-8")
