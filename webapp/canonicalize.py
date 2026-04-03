from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image
from pypdf import PdfReader, PdfWriter
from pypdf.generic import RectangleObject

try:
    from pdf2image import convert_from_bytes
except Exception:
    convert_from_bytes = None


def _load_images(source_bytes: bytes, filename: str, dpi: int) -> Optional[List[Image.Image]]:
    name = filename.lower()
    if name.endswith(".pdf"):
        if convert_from_bytes is None:
            return None
        return convert_from_bytes(source_bytes, dpi=dpi)
    image = Image.open(BytesIO(source_bytes))
    return [image]


def canonicalize_to_pdf(
    source_bytes: bytes,
    filename: str,
    output_pdf: Path,
    output_image: Optional[Path] = None,
    dpi: int = 72,
) -> bool:
    images = _load_images(source_bytes, filename, dpi)
    if not images:
        return False

    converted: List[Image.Image] = []
    sizes: List[Tuple[int, int]] = []
    for image in images:
        if image.mode != "RGB":
            image = image.convert("RGB")
        converted.append(image)
        sizes.append(image.size)

    if output_image:
        output_image.parent.mkdir(parents=True, exist_ok=True)
        converted[0].save(output_image, format="PNG")

    temp_pdf = output_pdf.with_suffix(".tmp.pdf")
    if len(converted) == 1:
        converted[0].save(temp_pdf, format="PDF", resolution=dpi)
    else:
        first, rest = converted[0], converted[1:]
        first.save(
            temp_pdf,
            format="PDF",
            resolution=dpi,
            save_all=True,
            append_images=rest,
        )

    reader = PdfReader(str(temp_pdf))
    writer = PdfWriter()
    for page_ix, page in enumerate(reader.pages):
        width, height = sizes[page_ix]
        rect = RectangleObject((0, 0, width, height))
        page.mediabox = rect
        page.cropbox = rect
        page.trimbox = rect
        page.bleedbox = rect
        page.artbox = rect
        writer.add_page(page)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with output_pdf.open("wb") as f:
        writer.write(f)
    temp_pdf.unlink(missing_ok=True)
    return True
