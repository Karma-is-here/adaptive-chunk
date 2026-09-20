from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any

from vanka.chunking.embeddings import SentenceTransformerEmbedder
from vanka.chunking.fixed import FixedSizeChunker
from vanka.chunking.recursive import RecursiveChunker
from vanka.chunking.semantic import SemanticChunker
from vanka.chunking.structural import chunk_page_structurally
from vanka.ingestion.normalized_json import load_normalized_pages
from vanka.structure.detector import detect_candidate_headings
from vanka.selection.selector import ChunkerSelector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_document_questions(
    document_path: Path,
    normalized_dir: Path,
    benchmarks_dir: Path | None,
) -> list[dict]:
    """
    Load the benchmark questions corresponding to a normalized document.

    Example:

        normalized/
            investment/
                advisory/
                    finsa.jsonl

    maps to:

        benchmarks/
            investment/
                advisory/
                    finsa_questions.json

    Benchmark questions are optional.
    """

    if benchmarks_dir is None:
        return []

    relative_path = document_path.relative_to(normalized_dir)

    question_name = f"{document_path.stem}_questions.json"

    question_path = (
        benchmarks_dir
        / relative_path.parent
        / question_name
    )

    if not question_path.exists():
        return []

    questions = json.loads(
        question_path.read_text(encoding="utf-8")
    )

    if not isinstance(questions, list):
        raise ValueError(
            f"Benchmark file must contain a JSON array: {question_path}"
        )

    return questions


def normalize(text: str) -> str:
    """
    Normalize whitespace and case for answer-span matching.
    """

    return re.sub(r"\s+", " ", text).strip().casefold()


def cosine_similarity(a, b) -> float:
    """
    Cosine similarity; works with lists or array-like vectors.
    """

    dot = sum(float(x) * float(y) for x, y in zip(a, b))

    norm_a = sum(float(x) ** 2 for x in a) ** 0.5
    norm_b = sum(float(y) ** 2 for y in b) ** 0.5

    if not norm_a or not norm_b:
        return 0.0

    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Structural chunking
# ---------------------------------------------------------------------------

def make_structural_chunks(
    pages,
    max_chars: int = 1200,
):
    """
    Create structurally-aware chunks from normalized pages.

    The current implementation intentionally preserves the existing
    Julius Baer-era structural chunking behavior: all detected candidate
    headings are approved.
    """

    chunks = []

    for page in pages:
        candidates = detect_candidate_headings(page.text)

        # Current policy:
        # approve every detected candidate heading.
        approved = {
            heading
            for _, heading, _, _ in candidates
        }

        chunks.extend(
            chunk_page_structurally(
                page,
                approved_headings=approved,
                max_chars=max_chars,
            )
        )

    return chunks


# ---------------------------------------------------------------------------
# Retrieval evaluation
# ---------------------------------------------------------------------------

def evaluate_strategy(
    name,
    chunks,
    questions,
    embedder,
):
    """
    Evaluate one chunking strategy against benchmark questions.

    Returns:
        summary
        per-question results
    """

    if not chunks:
        raise ValueError(
            f"{name} produced no chunks"
        )

    chunk_vectors = embedder.embed(
        [chunk.text for chunk in chunks]
    )

    query_vectors = embedder.embed(
        [item["question"] for item in questions]
    )

    hits_at = {
        1: 0,
        3: 0,
        5: 0,
    }

    reciprocal_ranks = []

    question_results = []

    for question, query_vector in zip(
        questions,
        query_vectors,
    ):
        # Keep both chunk index and similarity score.
        scored = [
            (
                i,
                cosine_similarity(
                    query_vector,
                    chunk_vectors[i],
                ),
            )
            for i in range(len(chunks))
        ]

        ranked = sorted(
            scored,
            key=lambda item: item[1],
            reverse=True,
        )

        answer = normalize(
            question["answer"]
        )

        first_hit_rank = None
        first_hit_chunk = None

        for rank, (chunk_index, _score) in enumerate(
            ranked,
            start=1,
        ):
            if answer in normalize(
                chunks[chunk_index].text
            ):
                first_hit_rank = rank
                first_hit_chunk = chunks[chunk_index]
                break

        reciprocal_ranks.append(
            0.0
            if first_hit_rank is None
            else 1.0 / first_hit_rank
        )

        for k in hits_at:
            if (
                first_hit_rank is not None
                and first_hit_rank <= k
            ):
                hits_at[k] += 1

        question_results.append(
            {
                "question": question["question"],
                "expected_page": question.get("page"),
                "hit_rank": first_hit_rank,
                "hit_page": (
                    first_hit_chunk.page_start
                    if first_hit_chunk is not None
                    else None
                ),
                "hit_chunk_id": (
                    first_hit_chunk.chunk_id
                    if first_hit_chunk is not None
                    else None
                ),
            }
        )

    lengths = [
        len(chunk.text)
        for chunk in chunks
    ]

    summary = {
        "strategy": name,
        "chunks": len(chunks),
        "mean_chars": round(
            statistics.mean(lengths),
            1,
        ),
        "median_chars": statistics.median(lengths),
        "under_100_chars": sum(
            length < 100
            for length in lengths
        ),
        "recall_at_1": (
            hits_at[1] / len(questions)
        ),
        "recall_at_3": (
            hits_at[3] / len(questions)
        ),
        "recall_at_5": (
            hits_at[5] / len(questions)
        ),
        "mrr": statistics.mean(
            reciprocal_ranks
        ),
    }

    return summary, question_results


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class VankaPipeline:
    """
    Core Vanka processing engine.

    This class is deliberately independent of Julius Baer or any specific
    project. It operates on normalized JSONL documents and optional
    benchmark question files.
    """

    def __init__(
        self,
        normalized_dir: str | Path,
        benchmarks_dir: str | Path | None = None,
        selector: ChunkerSelector | None = None,
        model_name: str = (
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        ),
        device: str = "cpu",
    ):
        self.normalized_dir = Path(
            normalized_dir
        ).resolve()

        self.benchmarks_dir = (
            Path(benchmarks_dir).resolve()
            if benchmarks_dir is not None
            else None
        )

        self.selector = (
            selector
            if selector is not None
            else ChunkerSelector()
        )

        self.embedder = SentenceTransformerEmbedder(
            model_name=model_name,
            device=device,
        )

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover_documents(self) -> list[Path]:
        """
        Recursively discover normalized JSONL documents.
        """

        documents = sorted(
            self.normalized_dir.rglob("*.jsonl")
        )

        if not documents:
            raise FileNotFoundError(
                "No normalized JSONL files found under: "
                f"{self.normalized_dir}"
            )

        return documents

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def build_strategies(self, pages):
        """
        Run all candidate chunking strategies.

        Returns:
            list of (strategy_name, chunks)
        """

        return [
            (
                "fixed",
                FixedSizeChunker(
                    chunk_size=1200,
                    overlap=150,
                ).chunk(pages),
            ),
            (
                "recursive",
                RecursiveChunker(
                    chunk_size=1200,
                ).chunk(pages),
            ),
            (
                "semantic",
                SemanticChunker(
                    embedder=self.embedder,
                    similarity_threshold=0.45,
                    max_chars=1200,
                ).chunk(pages),
            ),
            (
                "structural",
                make_structural_chunks(
                    pages,
                    max_chars=1200,
                ),
            ),
        ]

    # ------------------------------------------------------------------
    # Intrinsic metrics
    # ------------------------------------------------------------------

    @staticmethod
    def intrinsic_summary(
        name,
        chunks,
    ) -> dict[str, Any]:
        """
        Produce intrinsic chunk-quality metrics when no benchmark
        questions are available.
        """

        if not chunks:
            raise ValueError(
                f"{name} produced no chunks"
            )

        lengths = [
            len(chunk.text)
            for chunk in chunks
        ]

        return {
            "strategy": name,
            "chunks": len(chunks),
            "mean_chars": round(
                statistics.mean(lengths),
                1,
            ),
            "median_chars": statistics.median(
                lengths
            ),
            "under_100_chars": sum(
                length < 100
                for length in lengths
            ),
            "recall_at_1": None,
            "recall_at_3": None,
            "recall_at_5": None,
            "mrr": None,
        }

    # ------------------------------------------------------------------
    # Single document
    # ------------------------------------------------------------------

    def _process_document_internal(
        self,
        document_path: str | Path,
    ):
        """
        Internal processing method.

        Returns both the public report and the generated candidate
        chunks so callers such as Vanka.chunk(strategy="auto") can
        reuse the selected chunks without recomputing them.
        """

        document_path = Path(
            document_path
        ).resolve()

        pages = load_normalized_pages(
            document_path
        )

        strategies = self.build_strategies(
            pages
        )

        document_questions = load_document_questions(
            document_path=document_path,
            normalized_dir=self.normalized_dir,
            benchmarks_dir=self.benchmarks_dir,
        )

        summaries = []
        per_question = {}

        for name, chunks in strategies:

            if document_questions:
                summary, question_results = (
                    evaluate_strategy(
                        name=name,
                        chunks=chunks,
                        questions=document_questions,
                        embedder=self.embedder,
                    )
                )

                per_question[name] = (
                    question_results
                )

            else:
                summary = self.intrinsic_summary(
                    name,
                    chunks,
                )

            summaries.append(summary)

        selection = self.selector.select(
            summaries
        )

        try:
            relative_document = str(
                document_path.relative_to(
                    self.normalized_dir
                )
            )
        except ValueError:
            relative_document = str(
                document_path
            )

        report = {
            "document": relative_document,
            "page_count": len(pages),
            "question_count": len(
                document_questions
            ),
            "summaries": summaries,
            "selection": {
                "selected_strategy": (
                    selection.selected_strategy
                ),
                "selection_score": (
                    selection.selection_score
                ),
                "selection_reason": (
                    selection.selection_reason
                ),
                "selected_metrics": (
                    selection.selected_metrics
                ),
                "candidates": (
                    selection.candidates
                ),
                "rejected_candidates": (
                    selection.rejected_candidates
                ),
            },
            "per_question": per_question,
        }

        candidate_chunks = {
            name: chunks
            for name, chunks in strategies
        }

        return report, candidate_chunks
    
    def process_document(
        self,
        document_path: str | Path,
    ) -> dict[str, Any]:
        """
        Process one normalized document and return its report.
        """

        report, _candidate_chunks = (
            self._process_document_internal(
                document_path
            )
        )

        return report

    def process_all(
        self,
        *,
        include_chunks: bool = False,
    ) -> dict[str, Any]:
        """
        Process every normalized JSONL document recursively.

        If include_chunks is True, the selected chunks are returned
        alongside each document report so downstream consumers can
        persist them without recomputing the selected strategy.
        """

        document_paths = (
            self.discover_documents()
        )

        results = []

        for document_path in document_paths:
            if include_chunks:
                report, candidate_chunks = (
                    self._process_document_internal(
                        document_path
                    )
                )

                selected_strategy = (
                    report["selection"][
                        "selected_strategy"
                    ]
                )

                selected_chunks = candidate_chunks[
                    selected_strategy
                ]

                results.append(
                    {
                        **report,
                        "_selected_chunks": selected_chunks,
                    }
                )

            else:
                results.append(
                    self.process_document(
                        document_path
                    )
                )

        return {
            "documents": results
        }