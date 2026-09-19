
from __future__ import annotations

import statistics
from pathlib import Path

from vanka.chunking.embeddings import SentenceTransformerEmbedder
from vanka.chunking.fixed import FixedSizeChunker
from vanka.chunking.recursive import RecursiveChunker
from vanka.chunking.semantic import SemanticChunker
from vanka.chunking.structural import chunk_page_structurally
from vanka.ingestion.normalized_json import load_normalized_pages
from vanka.structure.detector import detect_candidate_headings


ROOT = Path(__file__).resolve().parents[1]
NORMALIZED_DIR = ROOT / "data" / "normalized"


def make_structural_chunks(pages, max_chars: int = 1200):
    chunks = []

    for page in pages:
        candidates = detect_candidate_headings(page.text)
        approved = {heading for _, heading, _, _ in candidates}

        chunks.extend(
            chunk_page_structurally(
                page,
                approved_headings=approved,
                max_chars=max_chars,
            )
        )

    return chunks


def report(name: str, chunks: list) -> str:
    if not chunks:
        return f"{name}: 0 chunks"

    lengths = [len(chunk.text) for chunk in chunks]

    return (
        f"{name}: {len(chunks)} chunks | "
        f"mean={statistics.mean(lengths):.0f} chars | "
        f"median={statistics.median(lengths):.0f} chars | "
        f"under_100={sum(length < 100 for length in lengths)}"
    )


def main():
    files = sorted(
        path
        for path in NORMALIZED_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}
    )

    if not files:
        print(f"No .json or .jsonl files found under {NORMALIZED_DIR}")
        return

    print(f"Found {len(files)} normalized files\n")

    embedder = SentenceTransformerEmbedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )

    successful = 0
    failed = []

    for file_path in files:
        print("=" * 80)
        print(f"FILE: {file_path.relative_to(ROOT)}")

        try:
            pages = load_normalized_pages(file_path)
            print(f"Pages: {len(pages)}")

            fixed = FixedSizeChunker(
                chunk_size=1200, overlap=150
            ).chunk(pages)

            recursive = RecursiveChunker(
                chunk_size=1200
            ).chunk(pages)

            semantic = SemanticChunker(
                embedder=embedder,
                similarity_threshold=0.45,
                min_chars=300,
                max_chars=1200,
            ).chunk(pages)

            structural = make_structural_chunks(
                pages, max_chars=1200
            )

            print(report("fixed", fixed))
            print(report("recursive", recursive))
            print(report("semantic", semantic))
            print(report("structural", structural))

            successful += 1

        except Exception as exc:
            failed.append((file_path, str(exc)))
            print(f"ERROR: {type(exc).__name__}: {exc}")

    print("\n" + "=" * 80)
    print(f"Finished: {successful}/{len(files)} files processed")

    if failed:
        print("\nFiles that failed:")
        for file_path, error in failed:
            print(f"- {file_path.relative_to(ROOT)}: {error}")


if __name__ == "__main__":
    main()
