from __future__ import annotations

from pathlib import Path
from typing import Any

from vanka.chunking.fixed import FixedSizeChunker
from vanka.chunking.recursive import RecursiveChunker
from vanka.chunking.semantic import SemanticChunker
from vanka.ingestion.normalized_json import load_normalized_pages
from vanka.models.chunking_result import ChunkingResult
from vanka.pipeline.runner import VankaPipeline
from vanka.pipeline.artifacts import ArtifactWriter


class Vanka:
    """
    Public Vanka API.
    """

    SUPPORTED_STRATEGIES = {
        "fixed",
        "recursive",
        "semantic",
        "structural",
        "auto",
    }

    def __init__(
        self,
        model_name: str = (
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        ),
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device

    
    def process(
        self,
        documents: str | Path,
        benchmarks: str | Path | None = None,
        output: str | Path | None = None,
    ) -> dict[str, Any]:
        """
        Run the complete Vanka pipeline over all
        normalized documents recursively.

        If output is provided, selected chunks, reports,
        and a manifest are written to that directory.
        """

        pipeline = VankaPipeline(
            normalized_dir=documents,
            benchmarks_dir=benchmarks,
            model_name=self.model_name,
            device=self.device,
        )

        results = pipeline.process_all(
            include_chunks=output is not None,
        )

        if output is None:
            return results

        writer = ArtifactWriter(output)

        entries = []
        failures = []

        for result in results["documents"]:
            document = (
                Path(documents)
                / result["document"]
            )

            try:
                selected_chunks = result[
                    "_selected_chunks"
                ]

                report = {
                    key: value
                    for key, value in result.items()
                    if key != "_selected_chunks"
                }

                entry = writer.write_document(
                    report,
                    selected_chunks,
                    document=document,
                    normalized_dir=documents,
                )

                entries.append(entry)

            except Exception as exc:
                failures.append(
                    {
                        "document": str(document),
                        "error": str(exc),
                    }
                )

        manifest_path = writer.write_manifest(
            entries,
            failed=failures,
        )

        public_results = {
            **results,
            "documents": [
                {
                    key: value
                    for key, value in document.items()
                    if key != "_selected_chunks"
                }
                for document in results["documents"]
            ],
        }

        return {
            "results": public_results,
            "manifest": str(manifest_path),
            "output": str(Path(output)),
            "documents_processed": len(entries),
            "documents_failed": len(failures),
        }


    def process_document(
        self,
        document: str | Path,
        benchmarks: str | Path | None = None,
    ) -> dict[str, Any]:
        """
        Run the complete Vanka pipeline on one document.
        """

        document = Path(document)

        pipeline = VankaPipeline(
            normalized_dir=document.parent,
            benchmarks_dir=benchmarks,
            model_name=self.model_name,
            device=self.device,
        )

        return pipeline.process_document(document)

    def chunk(
        self,
        document: str | Path,
        strategy: str = "auto",
    ) -> ChunkingResult:
        """
        Chunk a normalized document.

        Supported strategies:
            fixed
            recursive
            semantic
            structural
            auto
        """

        strategy = strategy.lower().strip()

        if strategy not in self.SUPPORTED_STRATEGIES:
            supported = ", ".join(
                sorted(self.SUPPORTED_STRATEGIES)
            )

            raise ValueError(
                f"Unsupported chunking strategy: "
                f"{strategy!r}. "
                f"Supported strategies: {supported}"
            )

        document = Path(document)

        # --------------------------------------------------------------
        # Automatic selection
        # --------------------------------------------------------------

        if strategy == "auto":
            pipeline = VankaPipeline(
                normalized_dir=document.parent,
                model_name=self.model_name,
                device=self.device,
            )

            result, candidate_chunks = (
                pipeline._process_document_internal(
                    document
                )
            )

            selection = result["selection"]

            selected_strategy = (
                selection["selected_strategy"]
            )

            if selected_strategy not in candidate_chunks:
                raise RuntimeError(
                    "Selector returned strategy "
                    f"{selected_strategy!r}, but no "
                    "candidate chunks were produced."
                )

            return ChunkingResult(
                chunks=candidate_chunks[
                    selected_strategy
                ],
                strategy=selected_strategy,
                selection_score=selection[
                    "selection_score"
                ],
                selection_reason=selection[
                    "selection_reason"
                ],
                metrics=selection[
                    "selected_metrics"
                ],
                document=str(document),
            )

        # --------------------------------------------------------------
        # Explicit strategies
        # --------------------------------------------------------------

        pages = load_normalized_pages(
            document
        )

        if strategy == "fixed":

            chunks = FixedSizeChunker(
                chunk_size=1200,
                overlap=150,
            ).chunk(pages)

        elif strategy == "recursive":

            chunks = RecursiveChunker(
                chunk_size=1200,
            ).chunk(pages)

        elif strategy == "semantic":

            from vanka.chunking.embeddings import (
                SentenceTransformerEmbedder,
            )

            embedder = SentenceTransformerEmbedder(
                model_name=self.model_name,
                device=self.device,
            )

            chunks = SemanticChunker(
                embedder=embedder,
                similarity_threshold=0.45,
                max_chars=1200,
            ).chunk(pages)

        elif strategy == "structural":

            pipeline = VankaPipeline(
                normalized_dir=document.parent,
                model_name=self.model_name,
                device=self.device,
            )

            candidates = pipeline.build_strategies(
                pages
            )

            chunks = None

            for name, candidate_chunks in candidates:
                if name == "structural":
                    chunks = candidate_chunks
                    break

            if chunks is None:
                raise RuntimeError(
                    "Structural chunker did not "
                    "produce chunks."
                )

        else:
            raise RuntimeError(
                f"Unable to execute strategy: {strategy}"
            )

        return ChunkingResult(
            chunks=chunks,
            strategy=strategy,
            document=str(document),
        )