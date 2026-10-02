from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class TextSection:
    text: str
    page: int | None = None


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    metadata: dict


_CODE_TYPES = {"py", "js", "ts", "tsx", "jsx"}
_CODE_SEPARATORS = [
    "\nclass ", "\ndef ", "\nasync def ", "\nfunction ",
    "\nexport ", "\nconst ", "\nlet ", "\nvar ", "\n", " ", "",
]
_TEXT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _split_recursive(text: str, size: int, overlap: int, separators: list[str]) -> list[str]:
    if len(text) <= size:
        return [text] if text.strip() else []
    separator = next((item for item in separators if item and item in text), "")
    pieces = text.split(separator) if separator else list(text)
    if separator:
        pieces = [piece + separator for piece in pieces[:-1]] + pieces[-1:]
    pieces = [piece for piece in pieces if piece]
    if len(pieces) == 1 and len(pieces[0]) > size:
        pieces = [pieces[0][index:index + size] for index in range(0, len(pieces[0]), size)]

    merged: list[str] = []
    current = ""
    for piece in pieces:
        if len(piece) > size:
            if current.strip():
                merged.append(current.strip())
                current = current[-overlap:] if overlap else ""
            merged.extend(_split_recursive(piece, size, overlap, separators[1:] or [""]))
            continue
        if current and len(current) + len(piece) > size:
            merged.append(current.strip())
            current = current[-overlap:] if overlap else ""
        current += piece
    if current.strip():
        merged.append(current.strip())

    bounded: list[str] = []
    for part in merged:
        while len(part) > size:
            bounded.append(part[:size].strip())
            part = part[size - overlap:] if overlap else part[size:]
        if part.strip():
            bounded.append(part.strip())

    # Overlap can create identical neighbors for repetitive source text.
    unique: list[str] = []
    for part in bounded:
        if part and (not unique or part != unique[-1]):
            unique.append(part)
    return unique


def chunk_sections(
    sections: Iterable[TextSection],
    *,
    document_id: str,
    filename: str,
    file_type: str,
    source_path: str,
    chunk_size: int = 700,
    chunk_overlap: int = 100,
) -> list[Chunk]:
    if chunk_size < 1 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller than chunk_size")
    code = file_type.lower().lstrip(".") in _CODE_TYPES
    separators = _CODE_SEPARATORS if code else _TEXT_SEPARATORS
    chunks: list[Chunk] = []
    seen: set[tuple[int | None, str]] = set()
    for section in sections:
        cleaned = re.sub(r"\x00", "", section.text).strip()
        for text in _split_recursive(cleaned, chunk_size, chunk_overlap, separators):
            normalized = re.sub(r"\s+", " ", text).strip()
            identity = (section.page, normalized)
            if not normalized or identity in seen:
                continue
            seen.add(identity)
            chunk_id = hashlib.sha256(
                f"{document_id}:{section.page}:{len(chunks)}:{normalized}".encode("utf-8")
            ).hexdigest()[:24]
            metadata = {
                "document_id": document_id,
                "filename": filename,
                "file_type": file_type.lower().lstrip("."),
                "page": section.page,
                "chunk_id": chunk_id,
                "source": source_path,
                "path": source_path,
            }
            if code:
                metadata["language"] = {"py": "python", "js": "javascript", "ts": "typescript"}.get(
                    file_type.lower().lstrip("."), file_type.lower().lstrip(".")
                )
            chunks.append(Chunk(chunk_id=chunk_id, text=text, metadata=metadata))
    return chunks