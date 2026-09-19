from vanka.structure.detector import detect_candidate_headings


BOUNDARY_TYPES = {"section_label", "question"}


def preview_page_structure(text: str) -> list[dict[str, object]]:
    """Preview page sections using only stronger candidate types."""
    lines = text.splitlines()
    candidates = [
        (line_index, heading, heading_type, reason)
        for line_index, heading, heading_type, reason
        in detect_candidate_headings(text)
        if heading_type in BOUNDARY_TYPES
    ]

    sections: list[dict[str, object]] = []
    current_start = 0
    current_heading = None
    current_type = None
    current_reason = None

    for line_index, heading, heading_type, reason in candidates:
        if line_index > current_start:
            content = "\n".join(lines[current_start:line_index]).strip()
            if content:
                sections.append({
                    "heading": current_heading,
                    "type": current_type,
                    "reason": current_reason,
                    "start_line": current_start,
                    "end_line": line_index - 1,
                    "text": content,
                })

        current_start = line_index
        current_heading = heading
        current_type = heading_type
        current_reason = reason

    content = "\n".join(lines[current_start:]).strip()
    if content:
        sections.append({
            "heading": current_heading,
            "type": current_type,
            "reason": current_reason,
            "start_line": current_start,
            "end_line": len(lines) - 1,
            "text": content,
        })

    return sections
