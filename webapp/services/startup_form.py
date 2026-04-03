from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from webapp.config import AppPaths
from webapp.services.documents import pdf_path_for, require_pdf
from webapp.services.uploads import save_upload


STARTUP_FILE_ID = "startup_default"
_STARTUP_SOURCE_NAME = "source.pdf"
_STARTUP_METADATA_NAME = "metadata.json"


def _source_pdf_path(paths: AppPaths) -> Path:
    return paths.startup_dir / _STARTUP_SOURCE_NAME


def _metadata_path(paths: AppPaths) -> Path:
    return paths.startup_dir / _STARTUP_METADATA_NAME


def _read_metadata(paths: AppPaths) -> dict:
    metadata_path = _metadata_path(paths)
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_metadata(paths: AppPaths, *, filename: str | None, source_kind: str) -> None:
    payload = {
        "filename": (filename or "Startup form").strip() or "Startup form",
        "source_kind": source_kind,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _metadata_path(paths).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _clear_data_artifacts(paths: AppPaths, file_id: str) -> None:
    for artifact in paths.data_dir.glob(f"{file_id}*"):
        if artifact.is_file():
            artifact.unlink(missing_ok=True)


def has_startup_form(paths: AppPaths) -> bool:
    return _source_pdf_path(paths).exists()


def read_startup_form_status(paths: AppPaths, current_file_id: str | None = None) -> dict:
    metadata = _read_metadata(paths)
    configured = has_startup_form(paths)
    filename = str(metadata.get("filename") or "No startup form configured.")
    source_kind = str(metadata.get("source_kind") or "")
    updated_at = str(metadata.get("updated_at") or "")
    return {
        "configured": configured,
        "filename": filename if configured else "",
        "source_kind": source_kind if configured else "",
        "updated_at": updated_at if configured else "",
        "message": (
            f"Startup form: {filename}"
            if configured
            else "No startup form configured."
        ),
        "is_current_form": configured and current_file_id == STARTUP_FILE_ID,
        "file_id": STARTUP_FILE_ID if configured else "",
    }


def materialize_startup_form(paths: AppPaths) -> str | None:
    source_pdf = _source_pdf_path(paths)
    if not source_pdf.exists():
        return None

    _clear_data_artifacts(paths, STARTUP_FILE_ID)
    shutil.copy2(source_pdf, pdf_path_for(paths, STARTUP_FILE_ID))
    return STARTUP_FILE_ID


def startup_launch_path(paths: AppPaths, integrations) -> str | None:
    file_id = materialize_startup_form(paths)
    if not file_id:
        return None

    pdf_path = pdf_path_for(paths, file_id)
    has_fields = bool(integrations.read_pdf_fields(pdf_path))
    if has_fields:
        return f"/fill/{file_id}"
    return f"/loading/{file_id}"


def save_current_form_as_startup(
    paths: AppPaths,
    file_id: str,
    *,
    filename: str | None = None,
    source_kind: str = "current_form",
) -> str:
    current_pdf = require_pdf(paths, file_id)
    source_pdf = _source_pdf_path(paths)
    paths.startup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(current_pdf, source_pdf)
    _write_metadata(paths, filename=filename, source_kind=source_kind)
    return STARTUP_FILE_ID


def save_uploaded_startup_form(paths: AppPaths, integrations, filename: str | None, file_bytes: bytes) -> str:
    temp_file_id = save_upload(
        paths=paths,
        integrations=integrations,
        filename=filename,
        file_bytes=file_bytes,
        field_source="commonforms",
    )
    try:
        save_current_form_as_startup(
            paths,
            temp_file_id,
            filename=filename,
            source_kind="uploaded_form",
        )
    finally:
        _clear_data_artifacts(paths, temp_file_id)

    return STARTUP_FILE_ID


def sync_startup_source_from_working_copy(paths: AppPaths) -> None:
    if not has_startup_form(paths):
        return

    working_pdf = pdf_path_for(paths, STARTUP_FILE_ID)
    if not working_pdf.exists():
        return

    shutil.copy2(working_pdf, _source_pdf_path(paths))


def clear_startup_form(paths: AppPaths) -> None:
    _source_pdf_path(paths).unlink(missing_ok=True)
    _metadata_path(paths).unlink(missing_ok=True)
    _clear_data_artifacts(paths, STARTUP_FILE_ID)
