#structural

from __future__ import annotations

import hashlib
from typing import Iterable

from vanka.models.chunk import Chunk


def _make_chunk(
    page: object,
    text: str,
    start_offset: int,
    end_offset: int,
    section_heading: str | None,
    strategy: str,
) -> Chunk:
    """Create a chunk while retaining its source location."""
    document_id = page.document_id
    source_file = page.source_file
    page_number = page.page_number

    identity = (
        f"{document_id}|{source_file}|{page_number}|"
        f"{start_offset}|{end_offset}|{strategy}"
    )
    chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]

    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        source_file=source_file,
        page_start=page_number,
        page_end=page_number,
        char_start=start_offset,
        char_end=end_offset,
        text=text,
        section_heading=section_heading,
        strategy=strategy,
        metadata={
            "start_offset": start_offset,
            "end_offset": end_offset,
        },
    )


def _split_oversized_span(
    page: object,
    text: str,
    span_start: int,
    section_heading: str | None,
    max_chars: int,
) -> list[Chunk]:
    """Split a long section at paragraph boundaries where possible."""
    chunks: list[Chunk] = []
    cursor = 0

    while cursor < len(text):
        remaining = text[cursor:]

        if len(remaining) <= max_chars:
            piece = remaining
        else:
            limit = cursor + max_chars
            boundary = text.rfind("\n\n", cursor, limit)

            if boundary > cursor:
                piece = text[cursor:boundary + 2]
            else:
                boundary = text.rfind("\n", cursor, limit)

                if boundary > cursor:
                    piece = text[cursor:boundary + 1]
                else:
                    piece = text[cursor:limit]

        if not piece:
            break

        piece_start = span_start + cursor
        piece_end = piece_start + len(piece)

        if piece:
            chunks.append(
                _make_chunk(
                    page=page,
                    text=piece,
                    start_offset=piece_start,
                    end_offset=piece_end,
                    section_heading=section_heading,
                    strategy="structural" if cursor == 0 else "size_fallback",
                )
            )

        cursor += len(piece)

    return chunks


def chunk_page_structurally(
    page: object,
    approved_headings: Iterable[str],
    max_chars: int = 1800,
) -> list[Chunk]:
    """
    Chunk one normalized page.

    Only exact, line-level matches from approved_headings create
    structural boundaries. Oversized sections are split by size.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1")

    approved = {heading.strip() for heading in approved_headings if heading.strip()}
    text = page.text

    if not text:
        return []

    lines = text.splitlines(keepends=True)
    spans: list[tuple[int, int, str | None]] = []

    offset = 0
    current_start = 0
    current_heading: str | None = None

    for line in lines:
        line_content = line.strip()

        if line_content in approved:
            if offset > current_start:
                spans.append((current_start, offset, current_heading))

            current_start = offset
            current_heading = line_content

        offset += len(line)

    if current_start < len(text):
        spans.append((current_start, len(text), current_heading))

    chunks: list[Chunk] = []

    for start, end, heading in spans:
        source_span = text[start:end]
        chunks.extend(
            _split_oversized_span(
                page=page,
                text=source_span,
                span_start=start,
                section_heading=heading,
                max_chars=max_chars,
            )
        )

    return chunks
