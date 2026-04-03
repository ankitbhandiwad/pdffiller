from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Dict, List, Tuple

from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    ArrayObject,
    BooleanObject,
    DictionaryObject,
    NameObject,
    NumberObject,
    StreamObject,
    TextStringObject,
)
from PIL import Image

try:
    from commonforms import prepare_form
    from commonforms.form_creator import PyPdfFormCreator
    from commonforms.inference import FFDetrDetector, render_pdf
    from commonforms.utils import BoundingBox
except ImportError:  # commonforms may be installed outside this repo
    prepare_form = None
    PyPdfFormCreator = None
    FFDetrDetector = None
    render_pdf = None
    BoundingBox = None

try:
    from pdf2image import convert_from_path
    import pytesseract
except Exception:
    convert_from_path = None
    pytesseract = None

OCR_AVAILABLE = convert_from_path is not None and pytesseract is not None

try:
    from .openai_vision import detect_missing_fields
except Exception:
    try:
        from openai_vision import detect_missing_fields  # type: ignore
    except Exception:
        detect_missing_fields = None

OPENAI_MISSING_AVAILABLE = detect_missing_fields is not None

try:
    from .grid_detect import detect_grid_textboxes, detect_grid_textboxes_from_image
except Exception:
    try:
        from grid_detect import detect_grid_textboxes, detect_grid_textboxes_from_image  # type: ignore
    except Exception:
        detect_grid_textboxes = None
        detect_grid_textboxes_from_image = None

try:
    from .docai_client import extract_text_from_pdf, extract_choice_boxes
except Exception:
    try:
        from docai_client import extract_text_from_pdf, extract_choice_boxes  # type: ignore
    except Exception:
        extract_text_from_pdf = None
        extract_choice_boxes = None


def _docai_payload(pdf_path: Path) -> Tuple[bytes, str]:
    payload = pdf_path.read_bytes()
    mime_type = "application/pdf"
    for ext, ext_mime in ((".jpg", "image/jpeg"), (".jpeg", "image/jpeg"), (".png", "image/png")):
        image_path = pdf_path.with_suffix(ext)
        if image_path.exists():
            payload = image_path.read_bytes()
            mime_type = ext_mime
            break
    return payload, mime_type


def _add_docai_checkboxes(pdf_path: Path) -> None:
    if extract_choice_boxes is None or PyPdfFormCreator is None or BoundingBox is None:
        return
    payload, mime_type = _docai_payload(pdf_path)
    boxes_by_page = extract_choice_boxes(payload, mime_type=mime_type)
    if not boxes_by_page:
        return

    writer = PyPdfFormCreator(str(pdf_path))
    for page_ix, boxes in boxes_by_page:
        for i, (x0, y0, x1, y1) in enumerate(boxes):
            name = f"choicebutton_{page_ix}_{i}"
            writer.add_checkbox(
                name,
                page_ix,
                BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1),
            )
    tmp_path = pdf_path.with_suffix(".docai.pdf")
    writer.save(str(tmp_path))
    writer.close()
    tmp_path.replace(pdf_path)


def _normalized_to_rect(page, box: Tuple[float, float, float, float]) -> ArrayObject:
    x0, y0, x1, y1 = box
    media = page.mediabox
    width = float(media.right - media.left)
    height = float(media.top - media.bottom)
    left = float(media.left) + x0 * width
    right = float(media.left) + x1 * width
    top = float(media.bottom) + (1 - y0) * height
    bottom = float(media.bottom) + (1 - y1) * height
    return ArrayObject(
        [
            NumberObject(left),
            NumberObject(bottom),
            NumberObject(right),
            NumberObject(top),
        ]
    )


def _add_field(
    writer: PdfWriter,
    page,
    fields: ArrayObject,
    name: str,
    rect: ArrayObject,
    field_type: str,
    label: str | None = None,
) -> None:
    border = ArrayObject([NumberObject(0), NumberObject(0), NumberObject(1)])
    border_style = DictionaryObject(
        {
            NameObject("/S"): NameObject("/S"),
            NameObject("/W"): NumberObject(1),
        }
    )
    appearance = DictionaryObject(
        {
            NameObject("/BC"): ArrayObject(
                [NumberObject(0), NumberObject(0), NumberObject(0)]
            )
        }
    )
    field = DictionaryObject()
    field.update(
        {
            NameObject("/FT"): NameObject(field_type),
            NameObject("/T"): TextStringObject(name),
            NameObject("/Subtype"): NameObject("/Widget"),
            NameObject("/Rect"): rect,
            NameObject("/F"): NumberObject(4),
            NameObject("/Border"): border,
            NameObject("/BS"): border_style,
            NameObject("/MK"): appearance,
        }
    )
    if label:
        field.update({NameObject("/TU"): TextStringObject(label)})
    if field_type == "/Tx":
        field.update(
            {
                NameObject("/Ff"): NumberObject(0),
                NameObject("/DA"): TextStringObject("/Helv 10 Tf 0 g"),
                NameObject("/V"): TextStringObject(""),
                NameObject("/DV"): TextStringObject(""),
            }
        )
    else:
        field.update(
            {
                NameObject("/V"): NameObject("/Off"),
                NameObject("/AS"): NameObject("/Off"),
            }
        )

    field_ref = writer._add_object(field)
    annots = page.get("/Annots")
    if annots is None:
        annots = ArrayObject()
        page[NameObject("/Annots")] = annots
    else:
        if hasattr(annots, "get_object"):
            annots = annots.get_object()
        if not isinstance(annots, ArrayObject):
            annots = ArrayObject(list(annots))
            page[NameObject("/Annots")] = annots
    annots.append(field_ref)
    fields.append(field_ref)


def _overlaps(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
    threshold: float = 0.2,
) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return False
    inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
    a_area = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    if a_area == 0:
        return False
    return (inter_area / a_area) >= threshold


def _bbox_to_tuple(bbox) -> Optional[Tuple[float, float, float, float]]:
    if bbox is None:
        return None
    if hasattr(bbox, "x0") and hasattr(bbox, "y0"):
        return (float(bbox.x0), float(bbox.y0), float(bbox.x1), float(bbox.y1))
    if isinstance(bbox, dict):
        try:
            return (
                float(bbox.get("x0", 0)),
                float(bbox.get("y0", 0)),
                float(bbox.get("x1", 0)),
                float(bbox.get("y1", 0)),
            )
        except Exception:
            return None
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            return tuple(float(x) for x in bbox)
        except Exception:
            return None
    return None


def _widget_boxes_by_page(widgets: Dict[int, List[object]]) -> Dict[int, List[Dict[str, object]]]:
    by_page: Dict[int, List[Dict[str, object]]] = {}
    for page_ix, items in widgets.items():
        for widget in items:
            box = _bbox_to_tuple(getattr(widget, "bounding_box", None))
            if not box:
                continue
            by_page.setdefault(page_ix, []).append(
                {
                    "type": "text",
                    "bbox": {"x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3]},
                }
            )
    return by_page


def _page_image_to_bytes(page_image) -> Optional[Tuple[bytes, str]]:
    print(f"[openai] encode page image type={type(page_image)}")
    try:
        if hasattr(page_image, "image"):
            page_image = page_image.image
        if hasattr(page_image, "save"):
            buff = io.BytesIO()
            page_image.save(buff, format="PNG")
            return buff.getvalue(), "image/png"
        if isinstance(page_image, bytes):
            return page_image, "image/png"
        try:
            import numpy as np
        except Exception:
            np = None
        if np is not None and isinstance(page_image, np.ndarray):
            img = Image.fromarray(page_image)
            buff = io.BytesIO()
            img.save(buff, format="PNG")
            return buff.getvalue(), "image/png"
    except Exception as exc:
        print(f"[openai] failed to encode page image: {exc}")
        return None
    return None


def _add_openai_missing_fields(writer, pages, widgets) -> None:
    if detect_missing_fields is None:
        print("[openai] missing-fields detector unavailable")
        return
    if not os.getenv("OPENAI_API_KEY"):
        print("[openai] OPENAI_API_KEY not set for missing-fields")
        return

    existing_by_page = _widget_boxes_by_page(widgets)
    for page_ix, page_image in enumerate(pages):
        payload = _page_image_to_bytes(page_image)
        if not payload:
            print(f"[openai] failed to encode page image {page_ix}")
            continue
        image_bytes, image_mime = payload
        existing_fields = existing_by_page.get(page_ix, [])
        missing = detect_missing_fields(
            image_bytes, image_mime, existing_fields, page_index=page_ix
        )
        if not missing:
            continue
        print(f"[openai] page={page_ix} missing fields={len(missing)}")
        existing_boxes = [
            _bbox_to_tuple(item.get("bbox")) for item in existing_fields
        ]
        for i, field in enumerate(missing):
            bbox = field.get("value_bbox") or {}
            box = _bbox_to_tuple(bbox)
            if not box:
                continue
            if any(_overlaps(box, existing) for existing in existing_boxes if existing):
                continue
            name = f"textmissing_{page_ix}_{i}"
            writer.add_text_box(
                name,
                page_ix,
                BoundingBox(x0=box[0], y0=box[1], x1=box[2], y1=box[3]),
                multiline=False,
            )


def _add_grid_fields(pdf_path: Path) -> bool:
    if detect_grid_textboxes is None and detect_grid_textboxes_from_image is None:
        print("[grid] grid detection unavailable")
        return False

    image_path = pdf_path.with_suffix(".png")
    boxes: List[Tuple[float, float, float, float]] = []
    if image_path.exists() and detect_grid_textboxes is not None:
        boxes = detect_grid_textboxes(image_path)
    elif convert_from_path is not None and detect_grid_textboxes_from_image is not None:
        pages = convert_from_path(str(pdf_path))
        if pages:
            import numpy as np

            boxes = detect_grid_textboxes_from_image(np.array(pages[0]))
    if not boxes:
        print("[grid] no boxes detected")
        return False

    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()
    writer.append_pages_from_reader(reader)
    fields = ArrayObject()

    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font)

    for page_ix, page in enumerate(writer.pages):
        for i, box in enumerate(boxes):
            rect = _normalized_to_rect(page, box)
            name = f"grid_{page_ix}_{i}"
            _add_field(writer, page, fields, name, rect, "/Tx")

    acro_form = DictionaryObject(
        {
            NameObject("/Fields"): fields,
            NameObject("/NeedAppearances"): BooleanObject(True),
            NameObject("/DR"): DictionaryObject(
                {NameObject("/Font"): DictionaryObject({NameObject("/Helv"): font_ref})}
            ),
        }
    )
    writer._root_object.update({NameObject("/AcroForm"): acro_form})

    tmp_path = pdf_path.with_suffix(".fillable.pdf")
    with tmp_path.open("wb") as f:
        writer.write(f)
    tmp_path.replace(pdf_path)
    return True


def _add_grid_missing_fields(writer, pages, widgets) -> None:
    if detect_grid_textboxes is None and detect_grid_textboxes_from_image is None:
        print("[grid] grid detection unavailable")
        return

    existing_by_page = _widget_boxes_by_page(widgets)
    for page_ix, page_image in enumerate(pages):
        try:
            import numpy as np
        except Exception:
            np = None
        boxes: List[Tuple[float, float, float, float]] = []
        if np is not None:
            image = getattr(page_image, "image", page_image)
            boxes = detect_grid_textboxes_from_image(np.array(image))
        if not boxes:
            image_path = Path(getattr(page_image, "path", ""))
            if image_path.exists() and detect_grid_textboxes is not None:
                boxes = detect_grid_textboxes(image_path)
        if not boxes:
            continue

        existing_fields = existing_by_page.get(page_ix, [])
        existing_boxes = [
            _bbox_to_tuple(item.get("bbox")) for item in existing_fields
        ]
        added = 0
        for i, box in enumerate(boxes):
            if any(_overlaps(box, existing) for existing in existing_boxes if existing):
                continue
            name = f"gridmissing_{page_ix}_{i}"
            writer.add_text_box(
                name,
                page_ix,
                BoundingBox(x0=box[0], y0=box[1], x1=box[2], y1=box[3]),
                multiline=False,
            )
            added += 1
        if added:
            print(f"[grid] page={page_ix} added={added}")

def check_fields_pdf(pdf_path: Path) -> bool:
    reader = PdfReader(str(pdf_path))
    fields = reader.get_fields()
    return bool(fields)


def add_textboxes_pdf_with_progress(
    pdf_path: Path,
    use_openai: bool = False,
    use_grid: bool = False,
    skip_commonforms: bool = False,
):
    if skip_commonforms:
        yield 0, 1
        if use_grid and _add_grid_fields(pdf_path):
            _add_docai_checkboxes(pdf_path)
            yield 1, 1
            return
        _add_docai_checkboxes(pdf_path)
        yield 1, 1
        return

    if prepare_form is not None:
        if PyPdfFormCreator and FFDetrDetector and render_pdf and BoundingBox:
            yield 0, 1
            detector = FFDetrDetector("FFDetr")
            if os.getenv("COMMONFORMS_OPTIMIZE") == "1" and hasattr(
                detector.model, "optimize_for_inference"
            ):
                try:
                    detector.model.optimize_for_inference()
                except Exception as exc:
                    print(f"[commonforms] optimize_for_inference failed: {exc}")
            pages = render_pdf(str(pdf_path))
            widgets = detector.extract_widgets(pages)

            writer = PyPdfFormCreator(str(pdf_path))
            writer.clear_existing_fields()

            for page_ix, items in widgets.items():
                for i, widget in enumerate(items):
                    name = f"{widget.widget_type.lower()}_{widget.page}_{i}"
                    if widget.widget_type == "TextBox":
                        writer.add_text_box(
                            name, page_ix, widget.bounding_box, multiline=False
                        )
                    elif widget.widget_type == "Signature":
                        writer.add_text_box(
                            name, page_ix, widget.bounding_box, multiline=False
                        )

            if use_grid:
                _add_grid_missing_fields(writer, pages, widgets)

            if use_openai and OPENAI_MISSING_AVAILABLE:
                _add_openai_missing_fields(writer, pages, widgets)

            tmp_path = pdf_path.with_suffix(".fillable.pdf")
            writer.save(str(tmp_path))
            writer.close()
            tmp_path.replace(pdf_path)

            _add_docai_checkboxes(pdf_path)
            yield 1, 1
            return

        yield 0, 1
        prepare_form(str(pdf_path), str(pdf_path))
        _add_docai_checkboxes(pdf_path)
        yield 1, 1
        return

    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()
    writer.append_pages_from_reader(reader)

    fields = ArrayObject()
    margin = 36
    total_pages = len(writer.pages)

    for i, page in enumerate(writer.pages):
        media = page.mediabox
        left = float(media.left) + margin
        bottom = float(media.bottom) + margin
        right = float(media.right) - margin
        top = float(media.top) - margin

        rect = ArrayObject(
            [
                NumberObject(left),
                NumberObject(bottom),
                NumberObject(right),
                NumberObject(top),
            ]
        )

        field = DictionaryObject()
        field.update(
            {
                NameObject("/FT"): NameObject("/Tx"),
                NameObject("/T"): TextStringObject(f"page_{i + 1}_text"),
                NameObject("/Subtype"): NameObject("/Widget"),
                NameObject("/Rect"): rect,
                NameObject("/Ff"): NumberObject(4096),
                NameObject("/DA"): TextStringObject("/Helv 10 Tf 0 g"),
                NameObject("/V"): TextStringObject(""),
                NameObject("/F"): NumberObject(4),
            }
        )

        field_ref = writer._add_object(field)
        annots = page.get("/Annots")
        if annots is None:
            annots = ArrayObject()
            page[NameObject("/Annots")] = annots
        else:
            if hasattr(annots, "get_object"):
                annots = annots.get_object()
            if not isinstance(annots, ArrayObject):
                annots = ArrayObject(list(annots))
                page[NameObject("/Annots")] = annots
        annots.append(field_ref)

        fields.append(field_ref)
        yield i + 1, total_pages

    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font)

    acro_form = DictionaryObject(
        {
            NameObject("/Fields"): fields,
            NameObject("/NeedAppearances"): BooleanObject(True),
            NameObject("/DR"): DictionaryObject(
                {
                    NameObject("/Font"): DictionaryObject(
                        {NameObject("/Helv"): font_ref}
                    )
                }
            ),
        }
    )
    writer._root_object.update({NameObject("/AcroForm"): acro_form})

    tmp_path = pdf_path.with_suffix(".fillable.pdf")
    with tmp_path.open("wb") as f:
        writer.write(f)
    tmp_path.replace(pdf_path)


def _field_label(meta, name: str) -> str:
    if isinstance(meta, dict):
        alt = meta.get("/TU")
        if alt:
            return str(alt)
        title = meta.get("/T")
        if title:
            return str(title)
    return str(name)


def _field_name_from_obj(obj) -> str | None:
    if not isinstance(obj, DictionaryObject):
        return None
    name = obj.get("/T")
    if name:
        return str(name)
    parent = obj.get("/Parent")
    if parent is not None:
        try:
            parent_obj = parent.get_object()
        except Exception:
            parent_obj = None
        if isinstance(parent_obj, DictionaryObject):
            parent_name = parent_obj.get("/T")
            if parent_name:
                return str(parent_name)
    alt = obj.get("/TM")
    if alt:
        return str(alt)
    return None


def _field_type_from_obj(obj) -> str | None:
    if not isinstance(obj, DictionaryObject):
        return None
    ft = obj.get("/FT")
    if ft is None:
        parent = obj.get("/Parent")
        if parent is not None:
            try:
                parent_obj = parent.get_object()
            except Exception:
                parent_obj = None
            if isinstance(parent_obj, DictionaryObject):
                ft = parent_obj.get("/FT")
    if ft is None:
        return None
    ft_str = str(ft)
    if "Btn" in ft_str:
        return "checkbox"
    if "Tx" in ft_str:
        return "text"
    return None


def _ensure_checkbox_appearance(writer: PdfWriter, annot_obj: DictionaryObject) -> NameObject:
    rect = annot_obj.get("/Rect")
    if not rect:
        return NameObject("/Yes")
    try:
        x0, y0, x1, y1 = (float(x) for x in rect)
    except Exception:
        return NameObject("/Yes")
    width = max(1.0, abs(x1 - x0))
    height = max(1.0, abs(y1 - y0))

    def _make_stream(data: bytes) -> StreamObject:
        stream = StreamObject()
        stream._data = data
        stream.update(
            {
                NameObject("/Type"): NameObject("/XObject"),
                NameObject("/Subtype"): NameObject("/Form"),
                NameObject("/BBox"): ArrayObject(
                    [
                        NumberObject(0),
                        NumberObject(0),
                        NumberObject(width),
                        NumberObject(height),
                    ]
                ),
                NameObject("/Resources"): DictionaryObject(),
            }
        )
        return stream

    inset = 0.6
    left = inset
    bottom = inset
    right = max(left + 0.1, width - inset)
    top = max(bottom + 0.1, height - inset)
    border = (
        f"q 0 0 0 RG 1 w {left:.2f} {bottom:.2f} m "
        f"{left:.2f} {top:.2f} l {right:.2f} {top:.2f} l "
        f"{right:.2f} {bottom:.2f} l {left:.2f} {bottom:.2f} l S Q\n"
    )
    cx0 = left + (right - left) * 0.2
    cy0 = bottom + (top - bottom) * 0.55
    cx1 = left + (right - left) * 0.45
    cy1 = bottom + (top - bottom) * 0.3
    cx2 = left + (right - left) * 0.82
    cy2 = bottom + (top - bottom) * 0.78
    check = (
        f"q 0 0 0 RG 1.2 w {cx0:.2f} {cy0:.2f} m "
        f"{cx1:.2f} {cy1:.2f} l {cx2:.2f} {cy2:.2f} l S Q\n"
    )

    off_stream = _make_stream(border.encode("utf-8"))
    yes_stream = _make_stream((border + check).encode("utf-8"))
    off_ref = writer._add_object(off_stream)
    yes_ref = writer._add_object(yes_stream)

    normal = DictionaryObject(
        {
            NameObject("/Off"): off_ref,
            NameObject("/Yes"): yes_ref,
        }
    )
    annot_obj[NameObject("/AP")] = DictionaryObject({NameObject("/N"): normal})
    return NameObject("/Yes")


def _iter_widget_annots(reader: PdfReader) -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []
    for page_index, page in enumerate(reader.pages):
        annots = page.get("/Annots")
        if annots is None:
            continue
        if hasattr(annots, "get_object"):
            try:
                annots = annots.get_object()
            except Exception:
                continue
        if not isinstance(annots, (list, ArrayObject)):
            continue
        for annot_ref in list(annots):
            if annot_ref is None:
                continue
            try:
                annot_obj = annot_ref.get_object()
            except Exception:
                continue
            if not isinstance(annot_obj, DictionaryObject):
                continue
            if annot_obj.get("/Subtype") != NameObject("/Widget"):
                continue
            name = _field_name_from_obj(annot_obj)
            if not name:
                continue
            rect = annot_obj.get("/Rect")
            rect_vals = None
            if rect:
                try:
                    rect_vals = [float(x) for x in rect]
                except Exception:
                    rect_vals = None
            label = _field_label(annot_obj, name)
            field_type = _field_type_from_obj(annot_obj)
            items.append(
                {
                    "name": name,
                    "label": label,
                    "field_type": field_type,
                    "page_index": page_index,
                    "rect": rect_vals,
                }
            )
    return items


def _safe_field_names(reader: PdfReader) -> set[str]:
    names: set[str] = set()

    def add_name_from_obj(obj) -> None:
        name = _field_name_from_obj(obj)
        if name:
            names.add(name)

    def walk_kids(obj) -> None:
        if not isinstance(obj, DictionaryObject):
            return
        kids = obj.get("/Kids")
        if kids is None:
            return
        if hasattr(kids, "get_object"):
            kids = kids.get_object()
        if not isinstance(kids, (list, ArrayObject)):
            return
        for kid in list(kids):
            if kid is None:
                continue
            try:
                kid_obj = kid.get_object()
            except Exception:
                continue
            add_name_from_obj(kid_obj)
            walk_kids(kid_obj)

    try:
        acro = reader.trailer["/Root"].get("/AcroForm")
    except Exception:
        acro = None
    if acro is not None and hasattr(acro, "get_object"):
        try:
            acro = acro.get_object()
        except Exception:
            acro = None
    if isinstance(acro, DictionaryObject):
        fields = acro.get("/Fields")
        if fields is not None and hasattr(fields, "get_object"):
            try:
                fields = fields.get_object()
            except Exception:
                fields = None
        if isinstance(fields, (list, ArrayObject)):
            for field_ref in list(fields):
                if field_ref is None:
                    continue
                try:
                    field_obj = field_ref.get_object()
                except Exception:
                    continue
                add_name_from_obj(field_obj)
                walk_kids(field_obj)

    for page in reader.pages:
        annots = page.get("/Annots")
        if annots is None:
            continue
        if hasattr(annots, "get_object"):
            try:
                annots = annots.get_object()
            except Exception:
                continue
        if not isinstance(annots, (list, ArrayObject)):
            continue
        for annot_ref in list(annots):
            if annot_ref is None:
                continue
            try:
                annot_obj = annot_ref.get_object()
            except Exception:
                continue
            add_name_from_obj(annot_obj)
    return names


def _slugify_field_name(label: str) -> str:
    cleaned = []
    for ch in label.strip().lower():
        if ch.isalnum():
            cleaned.append(ch)
        elif ch in (" ", "-", "_"):
            cleaned.append("_")
    slug = "".join(cleaned).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "manual"


def add_textbox_field(
    pdf_path: Path,
    page_index: int,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    label: str | None = None,
) -> str:
    reader = PdfReader(str(pdf_path))
    if page_index < 0 or page_index >= len(reader.pages):
        raise ValueError("Invalid page index")

    x0 = max(0.0, min(1.0, float(x0)))
    x1 = max(0.0, min(1.0, float(x1)))
    y0 = max(0.0, min(1.0, float(y0)))
    y1 = max(0.0, min(1.0, float(y1)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Invalid box coordinates")

    existing = _safe_field_names(reader)
    base_name = _slugify_field_name(label or "")
    name = base_name
    counter = 1
    while name in existing:
        name = f"{base_name}_{counter}"
        counter += 1

    writer = PdfWriter()
    writer.append_pages_from_reader(reader)
    page = writer.pages[page_index]
    media = page.mediabox
    page_left = float(media.left)
    page_bottom = float(media.bottom)
    page_width = float(media.right) - page_left
    page_height = float(media.top) - page_bottom

    left = page_left + (x0 * page_width)
    right = page_left + (x1 * page_width)
    top_img = y0 * page_height
    bottom_img = y1 * page_height
    pdf_bottom = page_bottom + (page_height - bottom_img)
    pdf_top = page_bottom + (page_height - top_img)

    rect = ArrayObject(
        [
            NumberObject(left),
            NumberObject(pdf_bottom),
            NumberObject(right),
            NumberObject(pdf_top),
        ]
    )

    annots = page.get("/Annots")
    if annots is not None and hasattr(annots, "get_object"):
        annots = annots.get_object()
    existing_boxes = []
    if isinstance(annots, ArrayObject):
        for annot in annots:
            try:
                obj = annot.get_object()
            except Exception:
                continue
            if obj.get("/Subtype") != NameObject("/Widget"):
                continue
            rect_obj = obj.get("/Rect")
            if not rect_obj:
                continue
            try:
                existing_boxes.append(tuple(float(x) for x in rect_obj))
            except Exception:
                continue
    new_box = (left, pdf_bottom, right, pdf_top)
    if any(_overlaps(new_box, box, threshold=0.15) for box in existing_boxes):
        raise ValueError("New box overlaps an existing field")

    acro_form = writer._root_object.get("/AcroForm")
    if acro_form is None:
        src_acro = reader.trailer["/Root"].get("/AcroForm")
        if src_acro is not None:
            writer._root_object.update({NameObject("/AcroForm"): src_acro})
            acro_form = writer._root_object.get("/AcroForm")
        else:
            acro_form = DictionaryObject({NameObject("/Fields"): ArrayObject()})
            writer._root_object.update({NameObject("/AcroForm"): acro_form})
    if hasattr(acro_form, "get_object"):
        acro_form = acro_form.get_object()
    if not isinstance(acro_form, DictionaryObject):
        raise ValueError("AcroForm is invalid")

    fields = acro_form.get("/Fields")
    if fields is None:
        fields = ArrayObject()
    elif hasattr(fields, "get_object"):
        fields = fields.get_object()
    if not isinstance(fields, ArrayObject):
        fields = ArrayObject(list(fields))
    cleaned_fields = ArrayObject()
    for field_ref in list(fields):
        if field_ref is None:
            continue
        try:
            field_obj = field_ref.get_object()
        except Exception:
            cleaned_fields.append(field_ref)
            continue
        if field_obj is None:
            continue
        cleaned_fields.append(field_ref)
    fields = cleaned_fields

    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font)
    dr = acro_form.get("/DR")
    if dr is not None and hasattr(dr, "get_object"):
        dr = dr.get_object()
    if not isinstance(dr, DictionaryObject):
        dr = DictionaryObject()
    fonts = dr.get("/Font")
    if fonts is not None and hasattr(fonts, "get_object"):
        fonts = fonts.get_object()
    if not isinstance(fonts, DictionaryObject):
        fonts = DictionaryObject()
    if NameObject("/Helv") not in fonts:
        fonts[NameObject("/Helv")] = font_ref
    dr[NameObject("/Font")] = fonts

    acro_form.update(
        {
            NameObject("/Fields"): fields,
            NameObject("/NeedAppearances"): BooleanObject(True),
            NameObject("/DR"): dr,
        }
    )

    _add_field(writer, page, fields, name, rect, "/Tx", label=label)

    tmp_path = pdf_path.with_suffix(".manual.pdf")
    with tmp_path.open("wb") as f:
        writer.write(f)
    tmp_path.replace(pdf_path)
    return name


def add_checkbox_field(
    pdf_path: Path,
    page_index: int,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    label: str | None = None,
) -> str:
    reader = PdfReader(str(pdf_path))
    if page_index < 0 or page_index >= len(reader.pages):
        raise ValueError("Invalid page index")

    x0 = max(0.0, min(1.0, float(x0)))
    x1 = max(0.0, min(1.0, float(x1)))
    y0 = max(0.0, min(1.0, float(y0)))
    y1 = max(0.0, min(1.0, float(y1)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Invalid box coordinates")

    existing = _safe_field_names(reader)
    base_name = _slugify_field_name(label or "checkbox")
    name = base_name
    counter = 1
    while name in existing:
        name = f"{base_name}_{counter}"
        counter += 1

    writer = PdfWriter()
    writer.append_pages_from_reader(reader)
    page = writer.pages[page_index]
    media = page.mediabox
    page_left = float(media.left)
    page_bottom = float(media.bottom)
    page_width = float(media.right) - page_left
    page_height = float(media.top) - page_bottom

    left = page_left + (x0 * page_width)
    right = page_left + (x1 * page_width)
    top_img = y0 * page_height
    bottom_img = y1 * page_height
    pdf_bottom = page_bottom + (page_height - bottom_img)
    pdf_top = page_bottom + (page_height - top_img)

    rect = ArrayObject(
        [
            NumberObject(left),
            NumberObject(pdf_bottom),
            NumberObject(right),
            NumberObject(pdf_top),
        ]
    )

    annots = page.get("/Annots")
    if annots is not None and hasattr(annots, "get_object"):
        annots = annots.get_object()
    existing_boxes = []
    if isinstance(annots, ArrayObject):
        for annot in annots:
            try:
                obj = annot.get_object()
            except Exception:
                continue
            if obj.get("/Subtype") != NameObject("/Widget"):
                continue
            rect_obj = obj.get("/Rect")
            if not rect_obj:
                continue
            try:
                existing_boxes.append(tuple(float(x) for x in rect_obj))
            except Exception:
                continue
    new_box = (left, pdf_bottom, right, pdf_top)
    if any(_overlaps(new_box, box, threshold=0.15) for box in existing_boxes):
        raise ValueError("New box overlaps an existing field")

    acro_form = writer._root_object.get("/AcroForm")
    if acro_form is None:
        src_acro = reader.trailer["/Root"].get("/AcroForm")
        if src_acro is not None:
            writer._root_object.update({NameObject("/AcroForm"): src_acro})
            acro_form = writer._root_object.get("/AcroForm")
        else:
            acro_form = DictionaryObject({NameObject("/Fields"): ArrayObject()})
            writer._root_object.update({NameObject("/AcroForm"): acro_form})
    if hasattr(acro_form, "get_object"):
        acro_form = acro_form.get_object()
    if not isinstance(acro_form, DictionaryObject):
        raise ValueError("AcroForm is invalid")

    fields = acro_form.get("/Fields")
    if fields is None:
        fields = ArrayObject()
    elif hasattr(fields, "get_object"):
        fields = fields.get_object()
    if not isinstance(fields, ArrayObject):
        fields = ArrayObject(list(fields))
    cleaned_fields = ArrayObject()
    for field_ref in list(fields):
        if field_ref is None:
            continue
        try:
            field_obj = field_ref.get_object()
        except Exception:
            cleaned_fields.append(field_ref)
            continue
        if field_obj is None:
            continue
        cleaned_fields.append(field_ref)
    fields = cleaned_fields

    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font)
    dr = acro_form.get("/DR")
    if dr is not None and hasattr(dr, "get_object"):
        dr = dr.get_object()
    if not isinstance(dr, DictionaryObject):
        dr = DictionaryObject()
    fonts = dr.get("/Font")
    if fonts is not None and hasattr(fonts, "get_object"):
        fonts = fonts.get_object()
    if not isinstance(fonts, DictionaryObject):
        fonts = DictionaryObject()
    if NameObject("/Helv") not in fonts:
        fonts[NameObject("/Helv")] = font_ref
    dr[NameObject("/Font")] = fonts

    acro_form.update(
        {
            NameObject("/Fields"): fields,
            NameObject("/NeedAppearances"): BooleanObject(True),
            NameObject("/DR"): dr,
        }
    )

    _add_field(writer, page, fields, name, rect, "/Btn", label=label)

    tmp_path = pdf_path.with_suffix(".manual.pdf")
    with tmp_path.open("wb") as f:
        writer.write(f)
    tmp_path.replace(pdf_path)
    return name


def remove_fields(pdf_path: Path, names: List[str]) -> int:
    if not names:
        return 0

    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()
    writer.append_pages_from_reader(reader)
    names_set = {str(name) for name in names}

    acro_form = writer._root_object.get("/AcroForm")
    if acro_form is None:
        src_acro = reader.trailer["/Root"].get("/AcroForm")
        if src_acro is not None:
            writer._root_object.update({NameObject("/AcroForm"): src_acro})
            acro_form = writer._root_object.get("/AcroForm")
        else:
            return 0
    if hasattr(acro_form, "get_object"):
        acro_form = acro_form.get_object()
    if not isinstance(acro_form, DictionaryObject):
        return 0

    def field_has_name(obj) -> bool:
        name = _field_name_from_obj(obj)
        if name and name in names_set:
            return True
        kids = obj.get("/Kids")
        if kids is None:
            return False
        if hasattr(kids, "get_object"):
            kids = kids.get_object()
        if not isinstance(kids, (list, ArrayObject)):
            return False
        for kid in kids:
            try:
                kid_obj = kid.get_object()
            except Exception:
                continue
            if isinstance(kid_obj, DictionaryObject) and field_has_name(kid_obj):
                return True
        return False

    fields = acro_form.get("/Fields")
    if fields is None:
        return 0
    if hasattr(fields, "get_object"):
        fields = fields.get_object()
    if not isinstance(fields, ArrayObject):
        fields = ArrayObject(list(fields))

    def prune_kids(obj) -> None:
        if not isinstance(obj, DictionaryObject):
            return
        kids = obj.get("/Kids")
        if kids is None:
            return
        if hasattr(kids, "get_object"):
            try:
                kids = kids.get_object()
            except Exception:
                return
        if not isinstance(kids, (list, ArrayObject)):
            return
        cleaned = ArrayObject()
        for kid in list(kids):
            if kid is None:
                continue
            try:
                kid_obj = kid.get_object()
            except Exception:
                continue
            if kid_obj is None:
                continue
            prune_kids(kid_obj)
            cleaned.append(kid)
        obj[NameObject("/Kids")] = cleaned

    kept_fields = ArrayObject()
    removed = 0
    for field_ref in list(fields):
        if field_ref is None:
            continue
        try:
            field = field_ref.get_object()
        except Exception:
            continue
        if field is None:
            continue
        if isinstance(field, DictionaryObject) and field_has_name(field):
            removed += 1
            continue
        if isinstance(field, DictionaryObject):
            prune_kids(field)
        kept_fields.append(field_ref)

    acro_form.update({NameObject("/Fields"): kept_fields})

    for page in writer.pages:
        annots = page.get("/Annots")
        if annots is None:
            continue
        if hasattr(annots, "get_object"):
            annots = annots.get_object()
        if not isinstance(annots, ArrayObject):
            annots = ArrayObject(list(annots))
        kept_annots = ArrayObject()
        for annot_ref in list(annots):
            if annot_ref is None:
                continue
            try:
                annot = annot_ref.get_object()
            except Exception:
                kept_annots.append(annot_ref)
                continue
            if annot is None:
                continue
            annot_name = _field_name_from_obj(annot)
            if annot_name and annot_name in names_set:
                continue
            kept_annots.append(annot_ref)
        page[NameObject("/Annots")] = kept_annots

    if removed <= 0:
        return 0

    tmp_path = pdf_path.with_suffix(".manual.pdf")
    with tmp_path.open("wb") as f:
        writer.write(f)
    tmp_path.replace(pdf_path)
    return removed


def remove_field_by_rect(
    pdf_path: Path,
    name: str,
    page_index: int,
    rect: List[float],
    tol: float = 2.0,
) -> int:
    if len(rect) != 4:
        return 0
    try:
        target = tuple(float(x) for x in rect)
    except Exception:
        return 0

    reader = PdfReader(str(pdf_path))
    if page_index < 0 or page_index >= len(reader.pages):
        return 0
    writer = PdfWriter()
    writer.append_pages_from_reader(reader)

    def rect_close(a, b) -> bool:
        try:
            return all(abs(float(a[i]) - float(b[i])) <= tol for i in range(4))
        except Exception:
            return False

    def prune(obj) -> bool:
        if not isinstance(obj, DictionaryObject):
            return True
        kids = obj.get("/Kids")
        if kids is not None:
            if hasattr(kids, "get_object"):
                try:
                    kids = kids.get_object()
                except Exception:
                    kids = None
            if isinstance(kids, (list, ArrayObject)):
                cleaned = ArrayObject()
                for kid in list(kids):
                    if kid is None:
                        continue
                    try:
                        kid_obj = kid.get_object()
                    except Exception:
                        continue
                    if not prune(kid_obj):
                        continue
                    cleaned.append(kid)
                obj[NameObject("/Kids")] = cleaned
        obj_name = _field_name_from_obj(obj)
        if obj_name != name:
            return True
        rect_obj = obj.get("/Rect")
        if rect_obj and rect_close(rect_obj, target):
            return False
        kids = obj.get("/Kids")
        if isinstance(kids, (list, ArrayObject)) and len(kids) == 0:
            return False
        return True

    removed = 0
    page = writer.pages[page_index]
    annots = page.get("/Annots")
    if annots is not None:
        if hasattr(annots, "get_object"):
            annots = annots.get_object()
        if not isinstance(annots, ArrayObject):
            annots = ArrayObject(list(annots))
        kept_annots = ArrayObject()
        for annot_ref in list(annots):
            if annot_ref is None:
                continue
            try:
                annot = annot_ref.get_object()
            except Exception:
                kept_annots.append(annot_ref)
                continue
            if annot is None:
                continue
            if annot.get("/Subtype") != NameObject("/Widget"):
                kept_annots.append(annot_ref)
                continue
            annot_name = _field_name_from_obj(annot)
            rect_obj = annot.get("/Rect")
            if annot_name == name and rect_obj and rect_close(rect_obj, target):
                removed += 1
                continue
            kept_annots.append(annot_ref)
        page[NameObject("/Annots")] = kept_annots

    acro_form = writer._root_object.get("/AcroForm")
    if acro_form is None:
        src_acro = reader.trailer["/Root"].get("/AcroForm")
        if src_acro is not None:
            writer._root_object.update({NameObject("/AcroForm"): src_acro})
            acro_form = writer._root_object.get("/AcroForm")
    if acro_form is not None and hasattr(acro_form, "get_object"):
        try:
            acro_form = acro_form.get_object()
        except Exception:
            acro_form = None
    if isinstance(acro_form, DictionaryObject):
        fields = acro_form.get("/Fields")
        if fields is not None and hasattr(fields, "get_object"):
            try:
                fields = fields.get_object()
            except Exception:
                fields = None
        if isinstance(fields, (list, ArrayObject)):
            kept_fields = ArrayObject()
            for field_ref in list(fields):
                if field_ref is None:
                    continue
                try:
                    field_obj = field_ref.get_object()
                except Exception:
                    continue
                if not prune(field_obj):
                    removed += 1
                    continue
                kept_fields.append(field_ref)
            acro_form.update({NameObject("/Fields"): kept_fields})

    if removed <= 0:
        return 0
    tmp_path = pdf_path.with_suffix(".manual.pdf")
    with tmp_path.open("wb") as f:
        writer.write(f)
    tmp_path.replace(pdf_path)
    return removed


def read_pdf_fields(pdf_path: Path) -> List[Tuple[str, str]]:
    reader = PdfReader(str(pdf_path))
    try:
        fields = reader.get_fields()
    except Exception:
        fields = None
    items: List[Tuple[str, str]] = []
    seen = set()
    if fields:
        for name, meta in fields.items():
            label = _field_label(meta, str(name))
            items.append((str(name), label))
            seen.add(str(name))
    for entry in _iter_widget_annots(reader):
        name = entry["name"]
        if name in seen:
            continue
        seen.add(name)
        items.append((name, str(entry["label"])))
    return items


def extract_field_schema(pdf_path: Path) -> List[Dict[str, object]]:
    reader = PdfReader(str(pdf_path))
    try:
        fields = reader.get_fields() or {}
    except Exception:
        fields = {}
    field_list = []

    name_to_page = {}
    name_to_rect = {}
    for page_index, page in enumerate(reader.pages):
        annots = page.get("/Annots")
        if annots is None:
            continue
        if hasattr(annots, "get_object"):
            annots = annots.get_object()
        for annot in annots:
            try:
                obj = annot.get_object()
            except Exception:
                continue
            name = obj.get("/T")
            rect = obj.get("/Rect")
            if name:
                name_to_page.setdefault(str(name), page_index)
                if rect:
                    name_to_rect.setdefault(str(name), [float(x) for x in rect])

    def sort_key(item):
        name, _ = item
        key = str(name)
        page = name_to_page.get(key, 0)
        rect = name_to_rect.get(key, [0, 0, 0, 0])
        top = rect[3] if rect else 0
        left = rect[0] if rect else 0
        return (page, -top, left, key)

    seen = set()
    if fields:
        for idx, (name, meta) in enumerate(sorted(fields.items(), key=sort_key)):
            label = _field_label(meta, str(name))
            field_type = None
            if isinstance(meta, dict):
                ft = meta.get("/FT")
                if ft is not None:
                    ft_str = str(ft)
                    if "Btn" in ft_str:
                        field_type = "checkbox"
                    elif "Tx" in ft_str:
                        field_type = "text"
            field_list.append(
                {
                    "id": idx + 1,
                    "name": str(name),
                    "label": label,
                    "field_type": field_type,
                    "page_index": name_to_page.get(str(name)),
                    "rect": name_to_rect.get(str(name)),
                }
            )
            seen.add(str(name))

    annot_items = _iter_widget_annots(reader)
    dedup = {}
    for entry in annot_items:
        name = entry["name"]
        if name in dedup or name in seen:
            continue
        dedup[name] = entry

    def annot_sort(entry):
        rect = entry.get("rect") or [0, 0, 0, 0]
        top = rect[3] if rect else 0
        left = rect[0] if rect else 0
        return (entry.get("page_index", 0), -top, left, entry.get("name", ""))

    offset = len(field_list)
    for idx, entry in enumerate(sorted(dedup.values(), key=annot_sort)):
        field_list.append(
            {
                "id": offset + idx + 1,
                "name": entry["name"],
                "label": entry["label"],
                "field_type": entry.get("field_type"),
                "page_index": entry.get("page_index"),
                "rect": entry.get("rect"),
            }
        )
    return field_list


def extract_pdf_context(pdf_path: Path, max_chars: int = 4000) -> Dict[str, object]:
    reader = PdfReader(str(pdf_path))
    pages_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text.strip())
    combined = "\n\n".join(pages_text).strip()
    if not combined and extract_text_from_pdf is not None:
        try:
            docai_text = extract_text_from_pdf(pdf_path.read_bytes())
            if docai_text:
                combined = docai_text
        except Exception:
            pass
    if not combined and OCR_AVAILABLE:
        ocr_pages = []
        for image in convert_from_path(str(pdf_path), first_page=1, last_page=3):
            try:
                ocr_pages.append(pytesseract.image_to_string(image).strip())
            except Exception:
                continue
        combined = "\n\n".join([p for p in ocr_pages if p]).strip()
    if len(combined) > max_chars:
        combined = combined[:max_chars] + "..."
    return {
        "page_count": len(reader.pages),
        "text": combined,
        "fields": extract_field_schema(pdf_path),
    }


def fill_pdf(pdf_path: Path, values: Dict[str, str]) -> bytes:
    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()
    writer.append_pages_from_reader(reader)
    acro_form = writer._root_object.get("/AcroForm")
    if acro_form is None:
        src_acro = reader.trailer["/Root"].get("/AcroForm")
        if src_acro is not None:
            writer._root_object.update({NameObject("/AcroForm"): src_acro})
            acro_form = writer._root_object.get("/AcroForm")
        else:
            writer._root_object.update(
                {
                    NameObject("/AcroForm"): DictionaryObject(
                        {NameObject("/Fields"): ArrayObject()}
                    )
                }
            )
            acro_form = writer._root_object.get("/AcroForm")
    if hasattr(acro_form, "get_object"):
        acro_form = acro_form.get_object()
    if isinstance(acro_form, DictionaryObject):
        acro_form.update({NameObject("/NeedAppearances"): BooleanObject(True)})
    # Directly set field values on widget annotations for better viewer compatibility.
    checkbox_names = set()
    for page in writer.pages:
        annots = page.get("/Annots")
        if annots is None:
            continue
        if hasattr(annots, "get_object"):
            annots = annots.get_object()
        for annot in annots:
            try:
                obj = annot.get_object()
            except Exception:
                continue
            key = _field_name_from_obj(obj)
            if not key:
                continue
            field_type = _field_type_from_obj(obj)
            if field_type == "checkbox":
                checkbox_names.add(key)
                if NameObject("/AP") not in obj:
                    _ensure_checkbox_appearance(writer, obj)
                raw = values.get(key)
                checked = False
                if isinstance(raw, str):
                    checked = raw.strip().lower() in (
                        "yes",
                        "true",
                        "1",
                        "on",
                        "checked",
                    )
                else:
                    checked = bool(raw)
                on_value = NameObject("/Yes")
                ap = obj.get("/AP")
                if ap is not None and hasattr(ap, "get_object"):
                    ap = ap.get_object()
                if isinstance(ap, DictionaryObject):
                    normal = ap.get("/N")
                    if normal is not None and hasattr(normal, "get_object"):
                        normal = normal.get_object()
                    if isinstance(normal, DictionaryObject):
                        for state in normal.keys():
                            state_name = str(state)
                            if state_name != "/Off":
                                on_value = NameObject(state_name)
                                break
                parent = obj.get("/Parent")
                parent_obj = None
                if parent is not None:
                    try:
                        parent_obj = parent.get_object()
                    except Exception:
                        parent_obj = None
                target = parent_obj if isinstance(parent_obj, DictionaryObject) else obj
                if not checked:
                    target.update({NameObject("/V"): NameObject("/Off")})
                    obj.update({NameObject("/AS"): NameObject("/Off")})
                else:
                    target.update({NameObject("/V"): on_value})
                    obj.update({NameObject("/AS"): on_value})
                continue
            if key not in values:
                continue
            val = TextStringObject(values[key])
            obj.update(
                {
                    NameObject("/V"): val,
                    NameObject("/DV"): val,
                }
            )
    # Keep pypdf's helper as a fallback for other field types.
    try:
        if checkbox_names:
            non_checkbox = {k: v for k, v in values.items() if k not in checkbox_names}
        else:
            non_checkbox = values
        writer.update_page_form_field_values(writer.pages, non_checkbox)
    except Exception:
        pass
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


# Legacy dead/stale code moved here for reference.
#
# COMMONFORMS_AVAILABLE = prepare_form is not None
# OPENAI_VISION_AVAILABLE = detect_fields_from_image is not None
#
#
# def _add_openai_fields(pdf_path: Path) -> bool:
#     if detect_fields_from_image is None:
#         print("[openai] detect_fields_from_image not available")
#         return False
#     if not os.getenv("OPENAI_API_KEY"):
#         print("[openai] OPENAI_API_KEY not set")
#         return False
#
#     payload, mime_type = _docai_payload(pdf_path)
#     print(f"[openai] payload mime={mime_type} size={len(payload)}")
#     images = []
#     if mime_type != "application/pdf":
#         images = [(0, payload, mime_type)]
#     elif convert_from_path is not None:
#         for page_ix, page_img in enumerate(convert_from_path(str(pdf_path))):
#             buff = io.BytesIO()
#             page_img.save(buff, format="PNG")
#             images.append((page_ix, buff.getvalue(), "image/png"))
#         print(f"[openai] rendered {len(images)} pages to images")
#     else:
#         print("[openai] pdf2image unavailable for PDF input")
#         return False
#
#     fields_by_page: Dict[int, List[Dict[str, object]]] = {}
#     for page_ix, image_bytes, image_mime in images:
#         print(f"[openai] calling vision page={page_ix} mime={image_mime} bytes={len(image_bytes)}")
#         fields = detect_fields_from_image(
#             image_bytes, image_mime, page_index=page_ix
#         )
#         print(f"[openai] page={page_ix} fields={0 if not fields else len(fields)}")
#         if not fields:
#             continue
#         fields_by_page.setdefault(page_ix, []).extend(fields)
#
#     if not fields_by_page:
#         print("[openai] no fields returned")
#         return False
#
#     print(f"[openai] building PDF fields for pages={list(fields_by_page.keys())}")
#     reader = PdfReader(str(pdf_path))
#     writer = PdfWriter()
#     writer.append_pages_from_reader(reader)
#     fields = ArrayObject()
#
#     font = DictionaryObject(
#         {
#             NameObject("/Type"): NameObject("/Font"),
#             NameObject("/Subtype"): NameObject("/Type1"),
#             NameObject("/BaseFont"): NameObject("/Helvetica"),
#         }
#     )
#     font_ref = writer._add_object(font)
#
#     for page_ix, page in enumerate(writer.pages):
#         page_fields = fields_by_page.get(page_ix, [])
#         for i, field in enumerate(page_fields):
#             bbox = field.get("value_bbox") or {}
#             label_bbox = field.get("label_bbox") or {}
#             try:
#                 box = (
#                     float(bbox.get("x0", 0)),
#                     float(bbox.get("y0", 0)),
#                     float(bbox.get("x1", 0)),
#                     float(bbox.get("y1", 0)),
#                 )
#             except Exception:
#                 continue
#             if _overlaps_norm(box, label_bbox):
#                 print(f"[openai] skipping label-overlap box page={page_ix} idx={i}")
#                 continue
#             rect = _normalized_to_rect(page, box)
#             field_type = field.get("type")
#             if field_type == "checkbox":
#                 name = f"choicebutton_{page_ix}_{i}"
#                 _add_field(writer, page, fields, name, rect, "/Btn")
#             else:
#                 name = f"textbox_{page_ix}_{i}"
#                 _add_field(writer, page, fields, name, rect, "/Tx")
#
#     acro_form = DictionaryObject(
#         {
#             NameObject("/Fields"): fields,
#             NameObject("/NeedAppearances"): BooleanObject(True),
#             NameObject("/DR"): DictionaryObject(
#                 {NameObject("/Font"): DictionaryObject({NameObject("/Helv"): font_ref})}
#             ),
#         }
#     )
#     writer._root_object.update({NameObject("/AcroForm"): acro_form})
#
#     tmp_path = pdf_path.with_suffix(".fillable.pdf")
#     with tmp_path.open("wb") as f:
#         writer.write(f)
#     tmp_path.replace(pdf_path)
#     return True
#
#
# def _overlaps_norm(
#     a: Tuple[float, float, float, float],
#     b: Dict[str, object],
#     threshold: float = 0.15,
# ) -> bool:
#     try:
#         bx0 = float(b.get("x0", 0))
#         by0 = float(b.get("y0", 0))
#         bx1 = float(b.get("x1", 0))
#         by1 = float(b.get("y1", 0))
#     except Exception:
#         return False
#     ax0, ay0, ax1, ay1 = a
#     inter_x0 = max(ax0, bx0)
#     inter_y0 = max(ay0, by0)
#     inter_x1 = min(ax1, bx1)
#     inter_y1 = min(ay1, by1)
#     if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
#         return False
#     inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
#     a_area = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
#     if a_area == 0:
#         return False
#     return (inter_area / a_area) >= threshold
#
#
# def _add_docai_fields(pdf_path: Path) -> bool:
#     return False
#
#
# def _filter_overlapping_boxes(
#     text_boxes: List[Tuple[float, float, float, float]],
#     choice_boxes: List[Tuple[float, float, float, float]],
# ) -> List[Tuple[float, float, float, float]]:
#     if not choice_boxes:
#         return text_boxes
#     filtered = []
#     for box in text_boxes:
#         if any(_overlaps(box, choice) for choice in choice_boxes):
#             continue
#         filtered.append(box)
#     return filtered
#
#
# def add_textboxes_pdf(
#     pdf_path: Path,
#     use_openai: bool = False,
#     use_grid: bool = False,
#     skip_commonforms: bool = False,
# ) -> None:
#     for _ in add_textboxes_pdf_with_progress(
#         pdf_path, use_openai=use_openai, use_grid=use_grid, skip_commonforms=skip_commonforms
#     ):
#         pass
#
#
# def ensure_textboxes_pdf(pdf_path: Path) -> None:
#     if not check_fields_pdf(pdf_path):
#         add_textboxes_pdf(pdf_path)
#
#
# def infer_field_labels_from_ocr(
#     pdf_path: Path, field_schema: List[Dict[str, object]]
# ) -> Dict[str, str]:
#     if not OCR_AVAILABLE:
#         return {}
#     needed = {}
#     for field in field_schema:
#         name = str(field.get("name", "")).strip()
#         if not name:
#             continue
#         rect = field.get("rect")
#         if not isinstance(rect, list) or len(rect) != 4:
#             continue
#         page_index = int(field.get("page_index", 0) or 0)
#         needed.setdefault(page_index, []).append(field)
#     if not needed:
#         return {}
#
#     def valid_label(text: str) -> bool:
#         t = (text or "").strip()
#         if len(t) < 3:
#             return False
#         low = t.lower()
#         if low in ("yes", "no", "n/a", "na"):
#             return False
#         return True
#
#     def pick_label(lines, field_box):
#         x0, y0, x1, y1 = field_box
#         fh = max(1.0, y1 - y0)
#         fw = max(1.0, x1 - x0)
#         margin = 8.0
#
#         def join_lines(sorted_lines):
#             parts: List[str] = []
#             for line in sorted_lines:
#                 text = (line.get("text") or "").strip()
#                 if not text:
#                     continue
#                 if parts and parts[-1].endswith("-"):
#                     parts[-1] = parts[-1][:-1] + text
#                 else:
#                     parts.append(text)
#             return " ".join(parts).strip()
#
#         left_candidates = []
#         above_candidates = []
#         for line in lines:
#             lx0, ly0, lx1, ly1 = line["box"]
#             ltext = line["text"]
#             if not valid_label(ltext):
#                 continue
#             overlap = max(0.0, min(y1, ly1) - max(y0, ly0))
#             overlap_ratio = overlap / max(1.0, min(fh, (ly1 - ly0)))
#             if lx1 <= x0 + margin and overlap_ratio > 0.2:
#                 left_candidates.append(line)
#             horiz_overlap = max(0.0, min(x1, lx1) - max(x0, lx0))
#             horiz_ratio = horiz_overlap / max(1.0, min(fw, (lx1 - lx0)))
#             if ly1 <= y0 + margin and horiz_ratio > 0.2:
#                 above_candidates.append(line)
#
#         if left_candidates:
#             clusters = []
#             tol = 30.0
#             for line in sorted(left_candidates, key=lambda l: l["box"][0]):
#                 lx0 = line["box"][0]
#                 if not clusters or abs(lx0 - clusters[-1]["center"]) > tol:
#                     clusters.append({"center": lx0, "items": [line]})
#                 else:
#                     clusters[-1]["items"].append(line)
#                     centers = [it["box"][0] for it in clusters[-1]["items"]]
#                     clusters[-1]["center"] = sum(centers) / len(centers)
#
#             def cluster_distance(cluster):
#                 max_right = max(it["box"][2] for it in cluster["items"])
#                 return x0 - max_right
#
#             clusters.sort(key=cluster_distance)
#             chosen = clusters[0]["items"]
#             chosen.sort(key=lambda l: l["box"][1])
#             return join_lines(chosen)
#
#         if above_candidates:
#             above_candidates.sort(key=lambda l: l["box"][1])
#             return join_lines(above_candidates)
#         return ""
#
#     labels: Dict[str, str] = {}
#     reader = PdfReader(str(pdf_path))
#     for page_index, fields in needed.items():
#         if page_index < 0 or page_index >= len(reader.pages):
#             continue
#         try:
#             images = convert_from_path(
#                 str(pdf_path),
#                 first_page=page_index + 1,
#                 last_page=page_index + 1,
#                 dpi=200,
#             )
#         except Exception:
#             continue
#         if not images:
#             continue
#         image = images[0]
#         img_w, img_h = image.size
#         page = reader.pages[page_index]
#         page_w = float(page.mediabox.width)
#         page_h = float(page.mediabox.height)
#         if page_w <= 0 or page_h <= 0:
#             continue
#         scale_x = img_w / page_w
#         scale_y = img_h / page_h
#         try:
#             data = pytesseract.image_to_data(
#                 image, output_type=pytesseract.Output.DICT
#             )
#         except Exception:
#             continue
#         lines = {}
#         count = len(data.get("text", []))
#         for i in range(count):
#             text = str(data["text"][i]).strip()
#             if not text:
#                 continue
#             try:
#                 conf = float(data.get("conf", [0])[i])
#             except Exception:
#                 conf = 0.0
#             if conf != -1 and conf < 40:
#                 continue
#             key = (
#                 data.get("block_num", [0])[i],
#                 data.get("par_num", [0])[i],
#                 data.get("line_num", [0])[i],
#             )
#             left = float(data.get("left", [0])[i])
#             top = float(data.get("top", [0])[i])
#             width = float(data.get("width", [0])[i])
#             height = float(data.get("height", [0])[i])
#             entry = lines.get(key)
#             if entry is None:
#                 lines[key] = {
#                     "words": [text],
#                     "box": [left, top, left + width, top + height],
#                 }
#             else:
#                 entry["words"].append(text)
#                 entry["box"][0] = min(entry["box"][0], left)
#                 entry["box"][1] = min(entry["box"][1], top)
#                 entry["box"][2] = max(entry["box"][2], left + width)
#                 entry["box"][3] = max(entry["box"][3], top + height)
#         line_list = []
#         for entry in lines.values():
#             line_list.append(
#                 {"text": " ".join(entry["words"]).strip(), "box": entry["box"]}
#             )
#         if not line_list:
#             continue
#         for field in fields:
#             name = str(field.get("name", "")).strip()
#             rect = field.get("rect") or []
#             if not name or len(rect) != 4:
#                 continue
#             try:
#                 x0, y0, x1, y1 = (float(x) for x in rect)
#             except Exception:
#                 continue
#             img_x0 = x0 * scale_x
#             img_x1 = x1 * scale_x
#             img_top = img_h - (y1 * scale_y)
#             img_bottom = img_h - (y0 * scale_y)
#             if img_x1 <= img_x0 or img_bottom <= img_top:
#                 continue
#             label = pick_label(line_list, (img_x0, img_top, img_x1, img_bottom))
#             if label:
#                 labels[name] = label
#     return labels
