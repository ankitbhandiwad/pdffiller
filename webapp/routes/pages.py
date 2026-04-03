from __future__ import annotations

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from webapp.exceptions import BadInputError, ExternalServiceError, NotFoundError
from webapp.services.documents import require_pdf, read_fill_page_context
from webapp.services.startup_form import read_startup_form_status, startup_launch_path
from webapp.services.uploads import save_upload


router = APIRouter()


def _templates(request: Request):
    return request.app.state.templates


def _paths(request: Request):
    return request.app.state.paths


def _integrations(request: Request):
    return request.app.state.integrations


@router.get("/", response_class=HTMLResponse)
def index(request: Request, manual: int | None = None) -> Response:
    if not manual:
        launch_path = startup_launch_path(_paths(request), _integrations(request))
        if launch_path:
            return RedirectResponse(url=launch_path, status_code=302)
    return _templates(request).TemplateResponse("index.html", {"request": request})


@router.post("/upload", response_class=HTMLResponse)
async def upload(
    request: Request,
    pdf: UploadFile = File(...),
    field_source: str = Form("commonforms"),
) -> HTMLResponse:
    try:
        file_bytes = await pdf.read()
        file_id = save_upload(
            paths=_paths(request),
            integrations=_integrations(request),
            filename=pdf.filename,
            file_bytes=file_bytes,
            field_source=field_source,
        )
    except BadInputError as exc:
        return _templates(request).TemplateResponse(
            "index.html",
            {"request": request, "error": str(exc)},
            status_code=400,
        )
    except ExternalServiceError as exc:
        return _templates(request).TemplateResponse(
            "index.html",
            {"request": request, "error": str(exc)},
            status_code=500,
        )

    return _templates(request).TemplateResponse(
        "loading.html",
        {"request": request, "file_id": file_id},
    )


@router.get("/loading/{file_id}", response_class=HTMLResponse)
def loading_page(request: Request, file_id: str) -> HTMLResponse:
    return _templates(request).TemplateResponse(
        "loading.html",
        {"request": request, "file_id": file_id},
    )


@router.get("/fill/{file_id}", response_class=HTMLResponse)
def fill_page(request: Request, file_id: str) -> HTMLResponse:
    try:
        pdf_path = require_pdf(_paths(request), file_id)
    except NotFoundError:
        return Response(status_code=404)

    context = read_fill_page_context(_integrations(request), pdf_path)
    context.update(
        {
            "request": request,
            "file_id": file_id,
            "transcribe_available": _integrations(request).transcribe_available,
            "qa_available": _integrations(request).llm_available,
            "ocr_available": _integrations(request).ocr_available,
            "startup_form": read_startup_form_status(_paths(request), current_file_id=file_id),
        }
    )
    return _templates(request).TemplateResponse("fill.html", context)
