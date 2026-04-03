from __future__ import annotations

import io
import uuid

from PIL import Image

from webapp.config import AppPaths
from webapp.exceptions import BadInputError, ExternalServiceError
from webapp.services.documents import pdf_path_for


def _validate_upload_filename(filename: str | None) -> str:
    if not filename:
        raise BadInputError("Please upload a PDF or JPG file.")
    lowered = filename.lower()
    is_pdf = lowered.endswith(".pdf")
    is_jpg = lowered.endswith(".jpg") or lowered.endswith(".jpeg")
    if not (is_pdf or is_jpg):
        raise BadInputError("Please upload a PDF or JPG file.")
    return lowered


def _write_uploaded_pdf(pdf_path, pdf_bytes: bytes) -> None:
    pdf_path.write_bytes(pdf_bytes)


def _convert_image_upload_to_pdf(paths: AppPaths, file_id: str, filename: str, image_bytes: bytes) -> None:
    image_ext = ".jpg" if filename.endswith(".jpg") else ".jpeg"
    (paths.data_dir / f"{file_id}{image_ext}").write_bytes(image_bytes)
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    width, height = image.size
    image = image.resize((width * 2, height * 2), resample=Image.LANCZOS)
    image.save(pdf_path_for(paths, file_id), format="PDF")


def _run_canonicalization(paths: AppPaths, integrations, file_id: str, filename: str, file_bytes: bytes) -> None:
    if integrations.canonicalize_to_pdf is None:
        raise ExternalServiceError("Canonicalizer is unavailable.")
    pdf_path = pdf_path_for(paths, file_id)
    output_image = paths.data_dir / f"{file_id}.png"
    ok = integrations.canonicalize_to_pdf(
        file_bytes,
        filename,
        pdf_path,
        output_image=output_image,
    )
    if not ok:
        raise ExternalServiceError("Canonicalization failed.")


def save_upload(paths: AppPaths, integrations, filename: str | None, file_bytes: bytes, field_source: str) -> str:
    lowered = _validate_upload_filename(filename)
    file_id = uuid.uuid4().hex
    pdf_path = pdf_path_for(paths, file_id)

    if field_source == "canonical":
        _run_canonicalization(paths, integrations, file_id, filename or "", file_bytes)
        return file_id

    if lowered.endswith(".pdf"):
        _write_uploaded_pdf(pdf_path, file_bytes)
    else:
        _convert_image_upload_to_pdf(paths, file_id, lowered, file_bytes)

    return file_id
