from vanka.ingestion.normalized_json import load_normalized_pages
from vanka.structure.detector import get_approved_heading_texts
from vanka.chunking.structural import chunk_page_structurally



pages = load_normalized_pages(
    "data/normalized/advisory_mandated.jsonl"
)

from vanka.structure.preview import preview_page_structure

for page in pages:
    print(f"\n{'=' * 12} PAGE {page.page_number} {'=' * 12}")

    for section in preview_page_structure(page.text):
        print(
            f"\n[{section['start_line']}-{section['end_line']}] "
            f"{section['type'] or 'untitled'}: "
            f"{section['heading'] or '(text before first heading)'}"
        )
        print(str(section["text"])[:500])

from vanka.structure.detector import detect_candidate_headings

for page in pages:
    for line_index, heading, heading_type, reason in detect_candidate_headings(
        page.text
    ):
        print(
            f"Page {page.page_number}, line {line_index}, "
            f"type={heading_type}, reason={reason}: {heading}"
        )

print(f"Loaded {len(pages)} pages")
print(f"Document: {pages[0].document_id}")
print(f"Page 1 of {pages[0].page_count}")
print(f"First text characters: {pages[0].text[:200]}")

print("\nSTRUCTURAL CHUNKS")

for page in pages:
    candidates = detect_candidate_headings(page.text)

    # Temporary approval rule for inspection only:
    # approve section labels and questions, not subheadings.
    approved_candidates = {
        (line_index, heading)
        for line_index, heading, heading_type, _reason in candidates
        if heading_type in {"section_label", "question"}
    }

    approved_headings = get_approved_heading_texts(
        page.text,
        approved_candidates,
    )

    chunks = chunk_page_structurally(
        page,
        approved_headings=approved_headings,
        max_chars=1800,
    )

    for index, chunk in enumerate(chunks, start=1):
        print(
            f"\nPage {page.page_number}, chunk {index}, "
            f"chars={len(chunk.text)}, "
            f"heading={chunk.section_heading!r}, "
            f"strategy={chunk.strategy}"
        )
        print(chunk.text[:700])




