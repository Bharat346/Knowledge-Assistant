from __future__ import annotations

import csv
import io
import json
from pathlib import PurePosixPath

from pypdf import PdfReader

from app.ingestion.chunking import TextSection


SUPPORTED_TYPES = {"pdf", "md", "txt", "py", "js", "ts", "json", "csv"}


class DocumentParseError(ValueError):
    pass


def parse_document(filename: str, content: bytes) -> tuple[str, list[TextSection]]:
    suffix = PurePosixPath(filename).suffix.lower().lstrip(".")
    if suffix not in SUPPORTED_TYPES:
        raise DocumentParseError("Unsupported file type")
    if not content:
        raise DocumentParseError("The uploaded file is empty")
    try:
        if suffix == "pdf":
            pages = PdfReader(io.BytesIO(content)).pages
            sections = [TextSection(page.extract_text() or "", index + 1) for index, page in enumerate(pages)]
        else:
            text = content.decode("utf-8-sig")
            if suffix == "json":
                text = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
            elif suffix == "csv":
                rows = list(csv.reader(io.StringIO(text)))
                text = "\n".join(" | ".join(cell.strip() for cell in row) for row in rows)
            sections = [TextSection(text)]
    except (UnicodeDecodeError, csv.Error, json.JSONDecodeError, Exception) as exc:
        if isinstance(exc, DocumentParseError):
            raise
        raise DocumentParseError("The uploaded file could not be parsed") from exc
    if not any(section.text.strip() for section in sections):
        raise DocumentParseError("No readable text was found in the uploaded file")
    return suffix, sections