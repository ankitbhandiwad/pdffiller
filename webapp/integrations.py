from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None

try:
    from webapp.canonicalize import canonicalize_to_pdf
except Exception:
    canonicalize_to_pdf = None

try:
    from webapp.openai_vision import OpenAIQuestionGenerationError, questions_from_image
except Exception:
    OpenAIQuestionGenerationError = RuntimeError
    questions_from_image = None

try:
    from webapp.pdf_forms import (
        OCR_AVAILABLE,
        add_checkbox_field,
        add_textbox_field,
        add_textboxes_pdf_with_progress,
        check_fields_pdf,
        extract_field_schema,
        extract_pdf_context,
        fill_pdf,
        read_pdf_fields,
        remove_field_by_rect,
        remove_fields,
    )
except Exception:
    OCR_AVAILABLE = False
    add_checkbox_field = None
    add_textbox_field = None
    add_textboxes_pdf_with_progress = None
    check_fields_pdf = None
    extract_field_schema = None
    extract_pdf_context = None
    fill_pdf = None
    read_pdf_fields = None
    remove_field_by_rect = None
    remove_fields = None

try:
    from webapp.transcribe import TranscribeConfig, transcribe_audio_bytes

    TRANSCRIBE_AVAILABLE = True
except Exception:
    TranscribeConfig = None
    transcribe_audio_bytes = None
    TRANSCRIBE_AVAILABLE = False

try:
    from webapp.llm_client import (
        generate_questions_for_targets,
        map_answers_to_fields,
        normalize_answer,
    )

    LLM_AVAILABLE = True
except Exception:
    generate_questions_for_targets = None
    map_answers_to_fields = None
    normalize_answer = None
    LLM_AVAILABLE = False


@dataclass(frozen=True)
class Integrations:
    canonicalize_to_pdf: Any
    convert_from_path: Any
    openai_question_error: type[Exception]
    questions_from_image: Any
    add_textboxes_pdf_with_progress: Any
    check_fields_pdf: Any
    add_textbox_field: Any
    add_checkbox_field: Any
    remove_field_by_rect: Any
    remove_fields: Any
    fill_pdf: Any
    read_pdf_fields: Any
    extract_field_schema: Any
    extract_pdf_context: Any
    ocr_available: bool
    transcribe_config: Any
    transcribe_audio_bytes: Any
    transcribe_available: bool
    generate_questions_for_targets: Any
    map_answers_to_fields: Any
    normalize_answer: Any
    llm_available: bool


def build_integrations() -> Integrations:
    """Bundle optional backend capabilities behind a small, explicit surface."""
    return Integrations(
        canonicalize_to_pdf=canonicalize_to_pdf,
        convert_from_path=convert_from_path,
        openai_question_error=OpenAIQuestionGenerationError,
        questions_from_image=questions_from_image,
        add_textboxes_pdf_with_progress=add_textboxes_pdf_with_progress,
        check_fields_pdf=check_fields_pdf,
        add_textbox_field=add_textbox_field,
        add_checkbox_field=add_checkbox_field,
        remove_field_by_rect=remove_field_by_rect,
        remove_fields=remove_fields,
        fill_pdf=fill_pdf,
        read_pdf_fields=read_pdf_fields,
        extract_field_schema=extract_field_schema,
        extract_pdf_context=extract_pdf_context,
        ocr_available=OCR_AVAILABLE,
        transcribe_config=TranscribeConfig,
        transcribe_audio_bytes=transcribe_audio_bytes,
        transcribe_available=TRANSCRIBE_AVAILABLE,
        generate_questions_for_targets=generate_questions_for_targets,
        map_answers_to_fields=map_answers_to_fields,
        normalize_answer=normalize_answer,
        llm_available=LLM_AVAILABLE,
    )
