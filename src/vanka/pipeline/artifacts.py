from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ArtifactWriter:
    """
    Write Vanka processing results to a reproducible output directory.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)

        self.chunks_dir = self.output_dir / "chunks"
        self.reports_dir = self.output_dir / "reports"

        self.chunks_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.reports_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _relative_document_path(
        self,
        document: str | Path,
        normalized_dir: str | Path,
    ) -> Path:
        document = Path(document).resolve()
        normalized_dir = Path(normalized_dir).resolve()

        try:
            return document.relative_to(normalized_dir)
        except ValueError:
            return Path(document.name)

    def write_document(
        self,
        report,
        chunks,
        *,
        document,
        normalized_dir,
    ):
        relative_path = self._relative_document_path(
            document,
            normalized_dir,
        )

        chunk_path = (
            self.chunks_dir
            / relative_path.with_suffix(".json")
        )

        report_path = (
            self.reports_dir
            / relative_path.with_suffix(".json")
        )

        chunk_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        chunk_records = []

        for index, chunk in enumerate(chunks):
            if not hasattr(chunk, "__dict__"):
                raise TypeError(
                    f"Expected a Chunk object, got {type(chunk).__name__}"
                )

            record = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "source": {
                    "file": chunk.source_file,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "section": chunk.section_heading,
                    "offsets": {
                        "start": chunk.char_start,
                        "end": chunk.char_end,
                    },
                },
                "text": chunk.text,
                "chunking": {
                    "strategy": chunk.strategy,
                    "chunk_index": index,
                },
                "metadata": dict(chunk.metadata),
            }

            chunk_records.append(record)

        chunk_payload = {
            "schema_version": "1.0",
            "document": str(document),
            "strategy": report["selection"]["selected_strategy"],
            "chunks": chunk_records,
        }

        chunk_path.write_text(
            json.dumps(
                chunk_payload,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        report_payload = {
            "schema_version": "1.0",
            **report,
        }

        report_path.write_text(
            json.dumps(
                report_payload,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        selection = report["selection"]

        return {
            "document": str(relative_path),
            "strategy": selection["selected_strategy"],
            "chunks": len(chunks),
            "selection_score": selection["selection_score"],
            "chunk_file": str(
                chunk_path.relative_to(self.output_dir)
            ),
            "report_file": str(
                report_path.relative_to(self.output_dir)
            ),
        }

    def write_manifest(
        self,
        entries: list[dict[str, Any]],
        *,
        failed: list[dict[str, Any]] | None = None,
    ) -> Path:
        """
        Write the top-level manifest.
        """

        failed = failed or []

        manifest = {
            "schema_version": "1.0",
            "documents_processed": len(entries),
            "documents_failed": len(failed),
            "documents": [
                {
                    **entry,
                    "status": "processed",
                }
                for entry in entries
            ],
            "failures": failed,
        }

        path = self.output_dir / "manifest.json"

        path.write_text(
            json.dumps(
                manifest,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return path