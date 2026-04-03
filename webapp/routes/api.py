from __future__ import annotations

import json

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import Response, StreamingResponse

from webapp.exceptions import BadInputError, ExternalServiceError, NotFoundError
from webapp.services.documents import json_bytes, read_pdf_info, render_page_image, require_pdf
from webapp.services.forms import add_field, fill_form, iter_progress_events, read_fields_payload, remove_field
from webapp.services.qa import complete_qa_session, normalize_qa_answer, record_qa_answer, start_qa_session
from webapp.services.startup_form import (
    STARTUP_FILE_ID,
    clear_startup_form,
    read_startup_form_status,
    save_current_form_as_startup,
    save_uploaded_startup_form,
    startup_launch_path,
    sync_startup_source_from_working_copy,
)
from webapp.services.transcription import transcribe_audio_upload


router = APIRouter()


def _paths(request: Request):
    return request.app.state.paths


def _integrations(request: Request):
    return request.app.state.integrations


def _qa_store(request: Request):
    return request.app.state.qa_store


@router.get("/page-image/{file_id}/{page_index}")
def get_page_image(request: Request, file_id: str, page_index: int) -> Response:
    try:
        image_bytes = render_page_image(_paths(request), _integrations(request), file_id, page_index)
    except NotFoundError:
        return Response(status_code=404)
    except ExternalServiceError:
        return Response(status_code=500)
    except RuntimeError:
        return Response(status_code=501)
    return Response(content=image_bytes, media_type="image/png")


@router.get("/pdf-info/{file_id}")
def pdf_info(request: Request, file_id: str) -> Response:
    try:
        pdf_path = require_pdf(_paths(request), file_id)
    except NotFoundError:
        return Response(status_code=404)
    return Response(content=json_bytes(read_pdf_info(pdf_path)), media_type="application/json")


@router.post("/fill/{file_id}")
async def fill(request: Request, file_id: str) -> Response:
    try:
        pdf_path = require_pdf(_paths(request), file_id)
    except NotFoundError:
        return Response(status_code=404)

    form = await request.form()
    values = {key: str(value) for key, value in form.items()}
    filled = fill_form(_integrations(request), pdf_path, values)
    headers = {"Content-Disposition": f'attachment; filename="filled-{file_id}.pdf"'}
    return Response(content=filled, media_type="application/pdf", headers=headers)


@router.get("/fields/{file_id}")
def get_fields(request: Request, file_id: str) -> Response:
    try:
        pdf_path = require_pdf(_paths(request), file_id)
    except NotFoundError:
        return Response(status_code=404)
    return Response(
        content=json.dumps(read_fields_payload(_integrations(request), pdf_path)),
        media_type="application/json",
    )


@router.post("/fields/{file_id}/add")
async def add_field_route(request: Request, file_id: str) -> Response:
    try:
        pdf_path = require_pdf(_paths(request), file_id)
    except NotFoundError:
        return Response(status_code=404)

    data = await request.json()
    try:
        payload = add_field(_integrations(request), pdf_path, data)
    except BadInputError as exc:
        return Response(content=json.dumps({"error": str(exc)}), media_type="application/json", status_code=400)
    return Response(content=json.dumps(payload), media_type="application/json")


@router.post("/fields/{file_id}/remove")
async def remove_field_route(request: Request, file_id: str) -> Response:
    try:
        pdf_path = require_pdf(_paths(request), file_id)
    except NotFoundError:
        return Response(status_code=404)

    data = await request.json()
    try:
        payload = remove_field(_integrations(request), pdf_path, data)
    except BadInputError as exc:
        return Response(content=json.dumps({"error": str(exc)}), media_type="application/json", status_code=400)
    return Response(content=json.dumps(payload), media_type="application/json")


@router.get("/progress/{file_id}")
def progress(request: Request, file_id: str) -> StreamingResponse:
    pdf_path = _paths(request).data_dir / f"{file_id}.pdf"

    def event_stream():
        for event in iter_progress_events(_integrations(request), pdf_path, file_id):
            yield event
        if file_id == STARTUP_FILE_ID and pdf_path.exists():
            try:
                if _integrations(request).check_fields_pdf(pdf_path):
                    sync_startup_source_from_working_copy(_paths(request))
            except Exception:
                pass

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
    )


@router.post("/transcribe")
async def transcribe_audio(
    request: Request,
    audio: UploadFile = File(...),
    model_size: str = Form("small"),
) -> Response:
    audio_bytes = await audio.read()
    try:
        text = transcribe_audio_upload(_integrations(request), audio_bytes, model_size)
    except BadInputError as exc:
        return Response(content=str(exc), status_code=400)
    except ExternalServiceError as exc:
        status = 501 if str(exc) == "Transcription not available" else 502
        return Response(content=str(exc), status_code=status)
    return Response(content=text, media_type="text/plain")


@router.post("/settings/startup-form/upload")
async def upload_startup_form(
    request: Request,
    pdf: UploadFile = File(...),
) -> Response:
    try:
        file_bytes = await pdf.read()
        save_uploaded_startup_form(
            _paths(request),
            _integrations(request),
            pdf.filename,
            file_bytes,
        )
        redirect_url = startup_launch_path(_paths(request), _integrations(request))
        payload = {
            "message": "Startup form saved.",
            "redirect_url": redirect_url,
            "startup_form": read_startup_form_status(_paths(request), current_file_id=STARTUP_FILE_ID),
        }
    except BadInputError as exc:
        return Response(content=str(exc), status_code=400)
    except ExternalServiceError as exc:
        return Response(content=str(exc), status_code=500)
    return Response(content=json.dumps(payload), media_type="application/json")


@router.post("/settings/startup-form/current/{file_id}")
def save_current_form_route(request: Request, file_id: str) -> Response:
    try:
        save_current_form_as_startup(
            _paths(request),
            file_id,
            filename="Current workspace",
        )
        payload = {
            "message": "Current form saved as the startup form.",
            "startup_form": read_startup_form_status(_paths(request), current_file_id=file_id),
        }
    except NotFoundError:
        return Response(status_code=404)
    return Response(content=json.dumps(payload), media_type="application/json")


@router.delete("/settings/startup-form")
def clear_startup_form_route(request: Request) -> Response:
    clear_startup_form(_paths(request))
    payload = {
        "message": "Startup form cleared.",
        "startup_form": read_startup_form_status(_paths(request)),
    }
    return Response(content=json.dumps(payload), media_type="application/json")


@router.post("/qa/start/{file_id}")
async def qa_start(request: Request, file_id: str) -> Response:
    try:
        payload = start_qa_session(_paths(request), _integrations(request), _qa_store(request), file_id)
    except NotFoundError:
        return Response(status_code=404)
    except BadInputError as exc:
        return Response(content=str(exc), status_code=400)
    except ExternalServiceError as exc:
        status = 501 if str(exc) == "LLM not available" else 502
        return Response(content=str(exc), status_code=status)
    return Response(content=json.dumps(payload), media_type="application/json")


@router.post("/qa/answer/{session_id}")
async def qa_answer(request: Request, session_id: str) -> Response:
    data = await request.json()
    answer = str(data.get("answer", "")).strip()
    index = data.get("index")
    try:
        payload = record_qa_answer(_qa_store(request), session_id, answer, index if isinstance(index, int) else None)
    except NotFoundError:
        return Response(status_code=404)
    return Response(content=json.dumps(payload), media_type="application/json")


@router.post("/qa/complete/{session_id}")
async def qa_complete(request: Request, session_id: str) -> Response:
    try:
        data = await request.json()
    except Exception:
        data = {}
    index = data.get("index")
    try:
        payload = complete_qa_session(
            _paths(request),
            _integrations(request),
            _qa_store(request),
            session_id,
            index if isinstance(index, int) else None,
        )
    except NotFoundError:
        return Response(status_code=404)
    except ExternalServiceError as exc:
        status = 501 if str(exc) == "LLM not available" else 502
        return Response(content=str(exc), status_code=status)
    return Response(content=json.dumps(payload), media_type="application/json")


@router.post("/qa/normalize")
async def qa_normalize(request: Request) -> Response:
    data = await request.json()
    question = str(data.get("question", "")).strip()
    answer = str(data.get("answer", "")).strip()
    try:
        payload = normalize_qa_answer(_integrations(request), question, answer)
    except BadInputError as exc:
        return Response(content=str(exc), status_code=400)
    except ExternalServiceError as exc:
        status = 501 if str(exc) == "LLM not available" else 502
        return Response(content=str(exc), status_code=status)
    return Response(content=json.dumps(payload), media_type="application/json")
