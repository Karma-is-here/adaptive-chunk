
from types import SimpleNamespace

from src.chunking.structural import chunk_page_structurally
from src.vanka.structure.detector import get_approved_heading_texts
from src.chunking.recursive import RecursiveChunker
from src.chunking.fixed import FixedSizeChunker
from src.chunking.semantic import SemanticChunker

def test_chunk_text_reconstructs_page_without_loss():
    source_text = (
        "Intro text.\n\n"
        "SECTION ONE\n"
        "First paragraph.\n\n"
        "Second paragraph.\n\n"
        "SECTION TWO\n"
        "Final paragraph.\n"
    )

    page = SimpleNamespace(
        document_id="test-document",
        source_file="test.jsonl",
        page_number=1,
        text=source_text,
    )

    chunks = chunk_page_structurally(
        page,
        approved_headings={"SECTION ONE", "SECTION TWO"},
        max_chars=35,
    )

    reconstructed = "".join(chunk.text for chunk in chunks)

    assert reconstructed == source_text
    assert all(chunk.document_id == "test-document" for chunk in chunks)
    assert all(chunk.page_start == 1 and chunk.page_end == 1 for chunk in chunks)


def test_whitespace_only_page_is_preserved():
    source_text = " \n\t  "

    page = SimpleNamespace(
        document_id="test-document",
        source_file="test.jsonl",
        page_number=1,
        text=source_text,
    )

    chunks = chunk_page_structurally(
        page,
        approved_headings=set(),
        max_chars=2,
    )

    assert "".join(chunk.text for chunk in chunks) == source_text

def test_oversized_text_is_split_without_loss():
    source_text = "A" * 25

    page = SimpleNamespace(
        document_id="test-document",
        source_file="test.jsonl",
        page_number=1,
        text=source_text,
    )

    chunks = chunk_page_structurally(
        page,
        approved_headings=set(),
        max_chars=10,
    )

    assert len(chunks) == 3
    assert all(len(chunk.text) <= 10 for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == source_text


def test_approved_heading_starts_a_new_chunk():
    source_text = "Intro.\n\nSECTION ONE\nDetails here.\n"

    page = SimpleNamespace(
        document_id="test-document",
        source_file="test.jsonl",
        page_number=1,
        text=source_text,
    )

    chunks = chunk_page_structurally(
        page,
        approved_headings={"SECTION ONE"},
        max_chars=100,
    )

    assert len(chunks) == 2
    assert chunks[0].section_heading is None
    assert chunks[1].section_heading == "SECTION ONE"
    assert "".join(chunk.text for chunk in chunks) == source_text

def test_long_heading_respects_max_chars_without_loss():
    source_text = "A VERY LONG SECTION HEADING\nBody text."

    page = SimpleNamespace(
        document_id="test-document",
        source_file="test.jsonl",
        page_number=1,
        text=source_text,
    )

    chunks = chunk_page_structurally(
        page,
        approved_headings={"A VERY LONG SECTION HEADING"},
        max_chars=10,
    )

    assert all(len(chunk.text) <= 10 for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == source_text
    assert all(
        chunk.section_heading == "A VERY LONG SECTION HEADING"
        for chunk in chunks
    )


def test_adapter_returns_only_explicitly_approved_headings():
    text = "SECTION ONE\nControl\nHow does it work?\n"

    approved_candidates = {
        (0, "SECTION ONE"),
        (2, "How does it work?"),
    }

    result = get_approved_heading_texts(text, approved_candidates)

    assert result == {"SECTION ONE", "How does it work?"}
    assert "Control" not in result

def test_approved_detector_headings_drive_structural_chunks():
    text = "Intro.\nSECTION ONE\nDetails.\nSECTION TWO\nMore details.\n"

    approved_candidates = {
        (1, "SECTION ONE"),
        (3, "SECTION TWO"),
    }
    approved_headings = get_approved_heading_texts(text, approved_candidates)

    page = SimpleNamespace(
        document_id="test-document",
        source_file="test.jsonl",
        page_number=1,
        text=text,
    )

    chunks = chunk_page_structurally(
        page,
        approved_headings=approved_headings,
        max_chars=100,
    )

    assert len(chunks) == 3
    assert [chunk.section_heading for chunk in chunks] == [
        None,
        "SECTION ONE",
        "SECTION TWO",
    ]
    assert "".join(chunk.text for chunk in chunks) == text

from src.chunking.fixed import FixedSizeChunker


def test_fixed_size_chunker_respects_size_and_overlap():
    source_text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    page = SimpleNamespace(
        document_id="test-document",
        source_file="test.jsonl",
        page_number=1,
        text=source_text,
    )

    chunks = FixedSizeChunker(chunk_size=10, overlap=2).chunk([page])

    assert [len(chunk.text) for chunk in chunks] == [10, 10, 10]
    assert chunks[0].text == "ABCDEFGHIJ"
    assert chunks[1].text == "IJKLMNOPQR"
    assert chunks[2].text == "QRSTUVWXYZ"
    assert (
        chunks[0].text
        + chunks[1].text[2:]
        + chunks[2].text[2:]
    ) == source_text


def test_recursive_chunker_preserves_text_and_size():
    source_text = (
        "First sentence.\n\n"
        "Second sentence is longer.\n"
        "Third sentence."
    )
    page = SimpleNamespace(
        document_id="doc-1",
        source_file="sample.json",
        page_number=1,
        text=source_text,
    )

    chunks = RecursiveChunker(chunk_size=20).chunk([page])

    assert chunks
    assert all(len(chunk.text) <= 20 for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == source_text
    assert all(chunk.strategy == "recursive" for chunk in chunks)
    assert chunks[0].metadata["start_offset"] == 0
    assert chunks[-1].metadata["end_offset"] == len(source_text)

class FakeEmbedder:
    def __init__(self, vectors):
        self.vectors = vectors
        self.received_texts = None

    def embed(self, texts):
        self.received_texts = list(texts)
        return self.vectors


def test_semantic_chunker_splits_on_similarity_drop_and_preserves_text():
    source_text = "Alpha topic. Beta topic. Gamma topic."
    page = SimpleNamespace(
        document_id="doc-1",
        source_file="sample.json",
        page_number=1,
        text=source_text,
    )

    # First two sentences are similar; the third is different.
    embedder = FakeEmbedder([
        [1.0, 0.0],
        [0.99, 0.01],
        [0.0, 1.0],
    ])

    chunks = SemanticChunker(
        embedder=embedder,
        similarity_threshold=0.8,
        max_chars=100,
        min_chars=1,
    ).chunk([page])

    assert len(chunks) == 2
    assert chunks[0].text == "Alpha topic. Beta topic. "
    assert chunks[1].text == "Gamma topic."
    assert "".join(chunk.text for chunk in chunks) == source_text
    assert chunks[0].metadata["start_offset"] == 0
    assert chunks[1].metadata["start_offset"] == len(chunks[0].text)
    assert chunks[-1].metadata["end_offset"] == len(source_text)


def test_semantic_chunker_respects_max_chars():
    source_text = "One. Two. Three. Four."
    page = SimpleNamespace(
        document_id="doc-1",
        source_file="sample.json",
        page_number=1,
        text=source_text,
    )

    embedder = FakeEmbedder([[1.0, 0.0]] * 4)

    chunks = SemanticChunker(
        embedder=embedder,
        similarity_threshold=0.8,
        max_chars=100,
        min_chars=1,
    ).chunk([page])

    assert all(len(chunk.text) <= 100 for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == source_text