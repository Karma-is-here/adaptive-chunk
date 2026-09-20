from vanka import Vanka


def test_vanka_is_importable():
    vanka = Vanka()

    assert vanka is not None
    assert vanka.model_name == (
        "sentence-transformers/"
        "all-MiniLM-L6-v2"
    )
    assert vanka.device == "cpu"


def test_supported_strategies():
    assert Vanka.SUPPORTED_STRATEGIES == {
        "fixed",
        "recursive",
        "semantic",
        "structural",
        "auto",
    }


def test_invalid_strategy():
    vanka = Vanka()

    try:
        vanka.chunk(
            "does-not-exist.jsonl",
            strategy="invalid",
        )
    except ValueError as exc:
        assert "Unsupported chunking strategy" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError for invalid strategy"
        )


def test_chunk_result_type():
    from vanka.models.chunking_result import (
        ChunkingResult,
    )

    result = ChunkingResult(
        chunks=[],
        strategy="fixed",
    )

    assert result.strategy == "fixed"
    assert result.chunks == []


def test_process_writes_output_artifacts(
    tmp_path,
):
    normalized_dir = tmp_path / "normalized"
    document = normalized_dir / "test.jsonl"

    normalized_dir.mkdir(
        parents=True,
    )

    document.write_text(
        (
            '{"document_id": "test-document", '
            '"source_file": "test.pdf", '
            '"source_path": "test.pdf", '
            '"page_number": 1, '
            '"page_count": 1, '
            '"extraction_method": "text", '
            '"text": "This is a test document."}\n'
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "output"

    vanka = Vanka()

    result = vanka.process(
        documents=normalized_dir,
        output=output_dir,
    )

    assert "results" in result
    assert "manifest" in result
    assert "output" in result

    assert (
        output_dir
        / "manifest.json"
    ).exists()

    assert (
        output_dir
        / "chunks"
        / "test.json"
    ).exists()

    assert (
        output_dir
        / "reports"
        / "test.json"
    ).exists()


def test_process_does_not_rechunk_selected_strategy(
    tmp_path,
    monkeypatch,
):
    normalized_dir = tmp_path / "normalized"
    document = normalized_dir / "test.jsonl"

    normalized_dir.mkdir(
        parents=True,
    )

    document.write_text(
        (
            '{"document_id": "test-document", '
            '"source_file": "test.pdf", '
            '"source_path": "test.pdf", '
            '"page_number": 1, '
            '"page_count": 1, '
            '"extraction_method": "text", '
            '"text": "This is a test document."}\n'
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "output"

    def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "Vanka.chunk() should not be called "
            "during process(output=...)."
        )

    monkeypatch.setattr(
        Vanka,
        "chunk",
        fail_if_called,
    )

    result = Vanka().process(
        documents=normalized_dir,
        output=output_dir,
    )

    assert result["documents_processed"] == 1
    assert result["documents_failed"] == 0


def test_process_does_not_expose_internal_selected_chunks(
    tmp_path,
):
    normalized_dir = tmp_path / "normalized"
    document = normalized_dir / "test.jsonl"

    normalized_dir.mkdir(
        parents=True,
    )

    document.write_text(
        (
            '{"document_id": "test-document", '
            '"source_file": "test.pdf", '
            '"source_path": "test.pdf", '
            '"page_number": 1, '
            '"page_count": 1, '
            '"extraction_method": "text", '
            '"text": "This is a test document."}\n'
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "output"

    result = Vanka().process(
        documents=normalized_dir,
        output=output_dir,
    )

    assert result["documents_processed"] == 1
    assert result["documents_failed"] == 0

    for document_result in result["results"]["documents"]:
        assert "_selected_chunks" not in document_result