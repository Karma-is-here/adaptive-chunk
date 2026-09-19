
import re


def detect_candidate_headings(
    text: str,
) -> list[tuple[int, str, str, str]]:
    """Return likely headings as (line index, text, type, reason).

    This is a conservative heuristic, not a guarantee that every candidate
    is a real heading. Review candidates before approving them.
    """
    candidates = []

    for line_index, raw_line in enumerate(text.splitlines()):
        line = raw_line.strip()

        if not line or len(line) > 80:
            continue

        # Skip obvious page-number lines and common OCR/navigation noise.
        if re.fullmatch(r"(?:page\s*)?\d+(?:\s*/\s*\d+)?", line, re.I):
            continue
        if len(line) <= 3 and not line.endswith("?"):
            continue

        # Short, all-uppercase labels are often section headings.
        if (
            line.isupper()
            and len(line) >= 4
            and len(line.split()) <= 10
            and not re.fullmatch(r"[\W\d_]+", line)
        ):
            candidates.append(
                (line_index, line, "section_label", "uppercase")
            )
            continue

        # FAQ questions can be headings, but only when short and question-like.
                # Short question-like lines can be heading candidates.
        # Approval is handled separately by get_approved_heading_texts().
        if (
            line.endswith("?")
            and len(line.split()) <= 12
        ):
            candidates.append(
                (line_index, line, "question", "question_mark")
            )
            continue

        # Title-case detection is deliberately strict to avoid treating
        # ordinary short lines and names as headings.
        words = line.split()
        if (
            len(words) <= 6
            and len(line) <= 45
            and not re.search(r"[.!?;,:]$", line)
            and all(
                word[0].isupper() or word.isupper()
                for word in words
                if word.isalpha()
            )
        ):
            candidates.append(
                (line_index, line, "subheading", "title_case_pattern")
            )

    return candidates


def get_approved_heading_texts(
    text: str,
    approved_candidates: set[tuple[int, str]],
) -> set[str]:
    """Return only approved (line_index, heading_text) candidates."""
    candidates = detect_candidate_headings(text)

    return {
        heading
        for line_index, heading, _kind, _reason in candidates
        if (line_index, heading) in approved_candidates
    }


if __name__ == "__main__":
    examples = [
        "Control",
        "Dedicated specialist",
        "Portfolio monitoring and research expertise",
        "ACCESS TO GLOBAL MARKETS",
        "Can I change my investment strategy?",
        "Vv",
        "&",
    ]

    for example in examples:
        print(repr(example), "->", detect_candidate_headings(example))
