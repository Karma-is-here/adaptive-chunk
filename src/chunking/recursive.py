from __future__ import annotations

import hashlib
import re
from typing import Sequence

from chunking.base import Chunker
from models.chunk import Chunk



class RecursiveChunker(Chunker):
    """Split text at natural boundaries, preserving every source character."""

    name = "recursive"

    def __init__(self, chunk_size: int = 1200, overlap: int = 0) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        if overlap < 0:
            raise ValueError("overlap cannot be negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        if overlap:
            raise ValueError(
                "RecursiveChunker currently supports overlap=0 only"
            )

        self.chunk_size = chunk_size

    def _split_text(self, text: str) -> list[tuple[int, int]]:
        """Return source offsets for chunks, without changing the text."""
        separators = ("\n\n", "\n", r"(?<=[.!?])\s+", " ")
        spans: list[tuple[int, int]] = []

        start = 0
        while start < len(text):
            limit = min(start + self.chunk_size, len(text))

            if limit == len(text):
                spans.append((start, limit))
                break

            # Prefer the largest natural boundary within the size limit.
            boundary = -1

            for separator in separators:
                if separator.startswith("(?"):
                    matches = list(
                        re.finditer(separator, text[start:limit])
                    )
                    if matches:
                        boundary = start + matches[-1].end()
                else:
                    found = text.rfind(separator, start, limit)
                    if found >= start:
                        boundary = found + len(separator)

                if boundary > start:
                    break

            # If no natural boundary fits, split at the hard limit.
            end = boundary if boundary > start else limit
            spans.append((start, end))
            start = end

        return spans

    def chunk(self, pages: Sequence[object]) -> list[Chunk]:
        chunks: list[Chunk] = []

        for page in pages:
            text = page.text
            if not text:
                continue

            for start, end in self._split_text(text):
                piece = text[start:end]

                identity = (
                    f"{page.document_id}|{page.source_file}|"
                    f"{page.page_number}|{start}|{end}|recursive"
                )
                chunk_id = hashlib.sha256(
                    identity.encode("utf-8")
                ).hexdigest()[:16]

                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        document_id=page.document_id,
                        source_file=page.source_file,
                        page_start=page.page_number,
                        page_end=page.page_number,
                        text=piece,
                        strategy=self.name,
                        metadata={
                            "start_offset": start,
                            "end_offset": end,
                            "chunk_size": self.chunk_size,
                            "overlap": 0,
                        },
                    )
                )

        return chunks