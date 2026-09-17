import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NormalizedPage:
    document_id: str
    source_file: str
    source_path: str
    page_number: int
    page_count: int
    extraction_method: str
    text: str
    metadata: dict[str, Any]


def load_normalized_pages(file_path: str | Path) -> list[NormalizedPage]:
    """
    Load page records produced by the existing normalization pipeline.

    Accepts:
    - A JSON array of page objects
    - JSONL (one JSON object per line)
    - Consecutive JSON objects, including pretty-printed objects
    """
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Normalized data file not found: {path}")

    raw_text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    records: list[dict[str, Any]] = []

    # Support a JSON array as well as a stream of JSON objects.
    stripped = raw_text.lstrip()

    if not stripped:
        raise ValueError(f"Normalized data file is empty: {path}")

    if stripped.startswith("["):
        parsed = json.loads(raw_text)

        if not isinstance(parsed, list):
            raise ValueError("Expected a JSON array of page records.")

        records = parsed
    else:
        position = 0

        while position < len(raw_text):
            # Skip whitespace between records.
            while (
                position < len(raw_text)
                and raw_text[position].isspace()
            ):
                position += 1

            if position >= len(raw_text):
                break

            record, next_position = decoder.raw_decode(raw_text, position)

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object at character {position}."
                )

            records.append(record)
            position = next_position

    pages: list[NormalizedPage] = []

    required_fields = {
        "document_id",
        "source_file",
        "source_path",
        "page_number",
        "page_count",
        "extraction_method",
        "text",
    }

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} is not a JSON object.")

        missing = required_fields - record.keys()

        if missing:
            raise ValueError(
                f"Record {index} is missing required fields: "
                f"{', '.join(sorted(missing))}"
            )

        if not isinstance(record["text"], str):
            raise ValueError(f"Record {index}: 'text' must be a string.")

        pages.append(
            NormalizedPage(
                document_id=record["document_id"],
                source_file=record["source_file"],
                source_path=record["source_path"],
                page_number=record["page_number"],
                page_count=record["page_count"],
                extraction_method=record["extraction_method"],
                text=record["text"],
                metadata={
                    key: value
                    for key, value in record.items()
                    if key not in required_fields
                },
            )
        )

    if not pages:
        raise ValueError(f"No page records found in: {path}")

    document_ids = {page.document_id for page in pages}

    if len(document_ids) != 1:
        raise ValueError(
            "This file contains records from multiple documents. "
            "Load one document at a time."
        )

    return sorted(pages, key=lambda page: page.page_number)