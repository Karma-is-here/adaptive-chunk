from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Chunk:
    """A text chunk with source provenance."""

    chunk_id: str
    document_id: str
    source_file: str

    page_start: int
    page_end: int

    char_start: int
    char_end: int

    text: str

    section_heading: str | None = None

    strategy: str = "structural"

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        if not self.chunk_id.strip():
            raise ValueError(
                "chunk_id cannot be empty"
            )

        if not self.document_id.strip():
            raise ValueError(
                "document_id cannot be empty"
            )

        if not self.source_file.strip():
            raise ValueError(
                "source_file cannot be empty"
            )

        if not self.text:
            raise ValueError(
                "Chunk text cannot be empty"
            )

        if self.page_start < 1:
            raise ValueError(
                "page_start must be at least 1"
            )

        if self.page_end < self.page_start:
            raise ValueError(
                "page_end cannot be before page_start"
            )

        if self.char_start < 0:
            raise ValueError(
                "char_start cannot be negative"
            )

        if self.char_end < self.char_start:
            raise ValueError(
                "char_end cannot be before char_start"
            )