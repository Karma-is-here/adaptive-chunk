import json

from vanka.pipeline.artifacts import ArtifactWriter


class DummyChunk:
    def __init__(
        self,
        text,
        page=1,
    ):
        self.chunk_id = f"dummy-{page}"
        self.document_id = "document-1"
        self.source_file = "document.jsonl"
        self.page_start = page
        self.page_end = page
        self.char_start = 0
        self.char_end = len(text)
        self.text = text
        self.section_heading = "Test Section"
        self.strategy = "fixed"
        self.metadata = {}


def test_artifact_writer_creates_chunk_report_and_manifest(
    tmp_path,
):
    normalized_dir = tmp_path / "normalized"
    document = (
        normalized_dir
        / "investment"
        / "advisory"
        / "document.jsonl"
    )

    document.parent.mkdir(
        parents=True,
    )
    document.write_text(
        "{}\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "output"

    writer = ArtifactWriter(output_dir)

    report = {
        "document": str(document),
        "selection": {
            "selected_strategy": "fixed",
            "selection_score": 0.742,
            "selection_reason": "test",
            "selected_metrics": {},
            "rejected_candidates": [],
        },
    }

    entry = writer.write_document(
        report,
        [
            DummyChunk("first chunk"),
            DummyChunk("second chunk", page=2),
        ],
        document=document,
        normalized_dir=normalized_dir,
    )

    manifest_path = writer.write_manifest(
        [entry],
    )

    chunk_file = (
        output_dir
        / "chunks"
        / "investment"
        / "advisory"
        / "document.json"
    )

    report_file = (
        output_dir
        / "reports"
        / "investment"
        / "advisory"
        / "document.json"
    )

    assert chunk_file.exists()
    assert report_file.exists()
    assert manifest_path.exists()

    chunks = json.loads(
        chunk_file.read_text(
            encoding="utf-8",
        )
    )

    assert chunks["strategy"] == "fixed"
    assert len(chunks["chunks"]) == 2
    assert chunks["chunks"][0]["chunking"]["chunk_index"] == 0

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8",
        )
    )

    assert manifest["documents_processed"] == 1
    assert manifest["schema_version"] == "1.0"
    assert manifest["documents_failed"] == 0
    assert manifest["documents"][0]["status"] == "processed"
    assert manifest["documents"][0]["strategy"] == "fixed"
    assert manifest["documents"][0]["chunks"] == 2
    assert chunks["schema_version"] == "1.0"
    assert chunks["chunks"][0]["chunking"]["chunk_index"] == 0
    assert chunks["chunks"][0]["chunking"]["strategy"] == "fixed"

    assert chunks["chunks"][0]["chunk_id"] == "dummy-1"
    assert chunks["chunks"][0]["document_id"] == "document-1"

    assert chunks["chunks"][0]["source"]["file"] == "document.jsonl"
    assert chunks["chunks"][0]["source"]["page_start"] == 1
    assert chunks["chunks"][0]["source"]["page_end"] == 1
    assert chunks["chunks"][0]["source"]["section"] == "Test Section"

    assert chunks["chunks"][0]["source"]["offsets"]["start"] == 0
    assert chunks["chunks"][0]["source"]["offsets"]["end"] == len("first chunk")

    assert chunks["chunks"][0]["text"] == "first chunk"