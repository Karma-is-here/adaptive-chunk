
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

from chunking.embeddings import SentenceTransformerEmbedder
from chunking.fixed import FixedSizeChunker
from chunking.recursive import RecursiveChunker
from chunking.semantic import SemanticChunker
from chunking.structural import chunk_page_structurally
from vanka.ingestion.normalized_json import load_normalized_pages
from vanka.structure.detector import detect_candidate_headings


ROOT = Path(__file__).resolve().parents[1]
NORMALIZED_DIR = ROOT / "data" / "normalized"
BENCHMARKS_DIR = ROOT / "data" / "benchmarks"

DIAGNOSTIC_QUESTIONS = {
    "What does Julius Baer determine when exploring the client's needs?",
    "Can clients personalize their advisory mandate?",
}

def load_document_questions(document_path: Path) -> list[dict]:
    """Load the benchmark file matching a normalized JSONL file."""
    relative_path = document_path.relative_to(NORMALIZED_DIR)

    # dias.jsonl -> dias_questions.json
    question_name = f"{document_path.stem}_questions.json"
    question_path = BENCHMARKS_DIR / relative_path.parent / question_name

    if not question_path.exists():
        print(f"No benchmark questions found: {question_path.relative_to(ROOT)}")
        return []

    questions = json.loads(question_path.read_text(encoding="utf-8"))

    if not isinstance(questions, list):
        raise ValueError(f"Benchmark file must contain a JSON array: {question_path}")

    return questions


def normalize(text: str) -> str:
    """Normalize whitespace and case for answer-span matching."""
    return re.sub(r"\s+", " ", text).strip().casefold()


def cosine_similarity(a, b) -> float:
    """Cosine similarity; works with lists or array-like vectors."""
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    norm_a = sum(float(x) ** 2 for x in a) ** 0.5
    norm_b = sum(float(y) ** 2 for y in b) ** 0.5

    if not norm_a or not norm_b:
        return 0.0

    return dot / (norm_a * norm_b)


def print_retrieval_diagnostics(
    question: str,
    answer: str,
    chunks: list,
    ranked: list[tuple[int, float]],
    top_k: int = 5,
) -> None:
    """Print top-ranked chunks and the chunk containing the gold answer."""
    answer_normalized = normalize(answer)

    print(f"\n  Diagnostics: {question}")

    shown = set()

    for rank, (chunk_index, score) in enumerate(ranked[:top_k], start=1):
        chunk = chunks[chunk_index]
        has_answer = answer_normalized in normalize(chunk.text)
        marker = " [ANSWER]" if has_answer else ""

        print(
            f"    #{rank} score={score:.4f} "
            f"page={chunk.page_start} chunk_id={chunk.chunk_id}{marker}"
        )
        print(f"       {chunk.text[:300].replace(chr(10), ' ')}")
        shown.add(chunk_index)

    # Show the answer-containing chunk even if it is outside the top five.
    for rank, (chunk_index, score) in enumerate(ranked, start=1):
        if chunk_index in shown:
            continue

        chunk = chunks[chunk_index]
        if answer_normalized in normalize(chunk.text):
            print(
                f"    Gold answer at rank #{rank} "
                f"score={score:.4f} page={chunk.page_start} "
                f"chunk_id={chunk.chunk_id}"
            )
            print(f"       {chunk.text[:500].replace(chr(10), ' ')}")
            break


def make_structural_chunks(pages, max_chars: int = 1200):
    chunks = []

    for page in pages:
        candidates = detect_candidate_headings(page.text)

        # Pilot policy: approve every detected candidate heading.
        approved = {heading for _, heading, _, _ in candidates}

        chunks.extend(
            chunk_page_structurally(
                page,
                approved_headings=approved,
                max_chars=max_chars,
            )
        )

    return chunks


def evaluate_strategy(name, chunks, questions, embedder):
    if not chunks:
        raise ValueError(f"{name} produced no chunks")

    chunk_vectors = embedder.embed([chunk.text for chunk in chunks])
    query_vectors = embedder.embed([item["question"] for item in questions])

    hits_at = {1: 0, 3: 0, 5: 0}
    reciprocal_ranks = []
    question_results = []

    for question, query_vector in zip(questions, query_vectors):
        # Keep both chunk index and score so diagnostics can display them.
        scored = [
            (i, cosine_similarity(query_vector, chunk_vectors[i]))
            for i in range(len(chunks))
        ]
        ranked = sorted(scored, key=lambda item: item[1], reverse=True)

        answer = normalize(question["answer"])
        first_hit_rank = None
        first_hit_chunk = None

        for rank, (chunk_index, _score) in enumerate(ranked, start=1):
            if answer in normalize(chunks[chunk_index].text):
                first_hit_rank = rank
                first_hit_chunk = chunks[chunk_index]
                break

        if question["question"] in DIAGNOSTIC_QUESTIONS:
            print_retrieval_diagnostics(
                question=question["question"],
                answer=question["answer"],
                chunks=chunks,
                ranked=ranked,
            )

        reciprocal_ranks.append(
            0.0 if first_hit_rank is None else 1.0 / first_hit_rank
        )

        for k in hits_at:
            if first_hit_rank is not None and first_hit_rank <= k:
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

    lengths = [len(chunk.text) for chunk in chunks]

    summary = {
        "strategy": name,
        "chunks": len(chunks),
        "mean_chars": round(statistics.mean(lengths), 1),
        "median_chars": statistics.median(lengths),
        "under_100_chars": sum(length < 100 for length in lengths),
        "recall_at_1": hits_at[1] / len(questions),
        "recall_at_3": hits_at[3] / len(questions),
        "recall_at_5": hits_at[5] / len(questions),
        "mrr": statistics.mean(reciprocal_ranks),
    }

    return summary, question_results


def main():
    document_paths = sorted(NORMALIZED_DIR.rglob("*.jsonl"))

    if not document_paths:
        raise FileNotFoundError(
            f"No normalized JSONL files found under: {NORMALIZED_DIR}"
        )


    embedder = SentenceTransformerEmbedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )

    all_document_results = []

    for document_path in document_paths:
        print(f"\n\n{'#' * 72}")
        print(f"DOCUMENT: {document_path.relative_to(ROOT)}")
        print(f"{'#' * 72}")

        pages = load_normalized_pages(document_path)

        strategies = [
            ("fixed", FixedSizeChunker(chunk_size=1200, overlap=150).chunk(pages)),
            ("recursive", RecursiveChunker(chunk_size=1200).chunk(pages)),
            (
                "semantic",
                SemanticChunker(
                    embedder=embedder,
                    similarity_threshold=0.45,
                    max_chars=1200,
                ).chunk(pages),
            ),
            ("structural", make_structural_chunks(pages, max_chars=1200)),
        ]

        document_questions = load_document_questions(document_path)

        summaries = []
        per_question = {}

        for name, chunks in strategies:
            print(f"\n{'=' * 24} {name.upper()} {'=' * 24}")

            if document_questions:
                summary, question_results = evaluate_strategy(
                    name, chunks, document_questions, embedder
                )
                per_question[name] = question_results
            else:
                if not chunks:
                    raise ValueError(f"{name} produced no chunks")

                lengths = [len(chunk.text) for chunk in chunks]
                summary = {
                    "strategy": name,
                    "chunks": len(chunks),
                    "mean_chars": round(statistics.mean(lengths), 1),
                    "median_chars": statistics.median(lengths),
                    "under_100_chars": sum(length < 100 for length in lengths),
                    "recall_at_1": None,
                    "recall_at_3": None,
                    "recall_at_5": None,
                    "mrr": None,
                }

            summaries.append(summary)

        print(f"\nPages: {len(pages)}")
        if document_questions:
            print(f"Retrieval questions: {len(document_questions)}")
        else:
            print("Retrieval questions: none (chunk statistics only)")

        headers = [
            "strategy",
            "chunks",
            "mean_chars",
            "median_chars",
            "under_100_chars",
            "recall_at_1",
            "recall_at_3",
            "recall_at_5",
            "mrr",
        ]

        print("\nSUMMARY")
        print(" | ".join(headers))
        print("-" * 112)

        for result in summaries:
            print(
                " | ".join(
                    str(result[key])
                    if result[key] is None
                    or key in {
                        "strategy",
                        "chunks",
                        "mean_chars",
                        "median_chars",
                        "under_100_chars",
                    }
                    else f"{result[key]:.2f}"
                    for key in headers
                )
            )

        if per_question:
            print("\nPER-QUESTION FIRST ANSWER-HIT RANK")
            for name, question_results in per_question.items():
                print(f"\n[{name}]")
                for result in question_results:
                    rank = (
                        str(result["hit_rank"])
                        if result["hit_rank"] is not None
                        else "NOT FOUND"
                    )
                    page = (
                        str(result["hit_page"])
                        if result["hit_page"] is not None
                        else "-"
                    )
                    print(
                        f"- {result['question']} "
                        f"| rank={rank} | hit_page={page}"
                    )

        all_document_results.append(
            {
                "document": str(document_path.relative_to(ROOT)),
                "page_count": len(pages),
                "question_count": len(document_questions),
                "summaries": summaries,
                "per_question": per_question,
            }
        )

    output_path = ROOT / "data" / "benchmarks" / "retrieval_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {"documents": all_document_results},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"\nSaved detailed results to: {output_path}")


if __name__ == "__main__":
    main()
