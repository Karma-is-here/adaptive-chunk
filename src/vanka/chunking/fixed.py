#fixed

from __future__ import annotations

import hashlib
from typing import Sequence

from vanka.chunking.base import Chunker
from vanka.models.chunk import Chunk


class FixedSizeChunker(Chunker):
    """Split normalized pages into fixed-size character windows."""

    name = "fixed"

    def __init__(self, chunk_size: int = 1200, overlap: int = 150) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        if overlap < 0:
            raise ValueError("overlap cannot be negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, pages: Sequence[object]) -> list[Chunk]:
        chunks: list[Chunk] = []

        for page in pages:
            text = page.text
            if not text:
                continue

            start = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                piece = text[start:end]

                identity = (
                    f"{page.document_id}|{page.source_file}|"
                    f"{page.page_number}|{start}|{end}|fixed"
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
                            "overlap": self.overlap,
                        },
                    )
                )

                if end == len(text):
                    break

                start = end - self.overlap

        return chunks
