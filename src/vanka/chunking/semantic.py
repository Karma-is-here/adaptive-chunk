#semnatic

from __future__ import annotations

import hashlib
import re
from typing import Protocol, Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised without the semantic extra
    np = None  # type: ignore[assignment]

from vanka.chunking.base import Chunker
from vanka.models.chunk import Chunk


class Embedder(Protocol):
    """Interface for any sentence-embedding backend."""

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one embedding vector per input text."""
        ...


# Pre-compiled once at import time instead of per call.
# Boundaries: sentence-ending punctuation followed by horizontal space,
# OR blank-line runs (two or more newlines, possibly with whitespace
# between them). Single newlines are NOT boundaries — this avoids
# shredding OCR text that has one line per visual line.
_SENTENCE_BOUNDARY = re.compile(
    r"(?:[.!?]+[ \t]+|\n[ \t]*(?:\n[ \t]*)+)"
)


class SemanticChunker(Chunker):
    """
    Group neighboring sentences while their cosine similarity is high.

    Design notes
    ------------
    * Boundary decision: the cosine similarity between adjacent sentence
      embeddings, optionally smoothed over ``window_size`` adjacent pairs.
      A boundary is created when the (smoothed) score drops below
      ``similarity_threshold``.
    * ``min_chars`` is honored by merging undersized groups into their
      most semantically compatible, size-valid neighbor (forward or
      backward). If no merge keeps the result within ``max_chars``, the
      group is left alone and ``_enforce_max_chars`` handles it.
    * ``max_chars`` is a hard upper bound; exceptionally long sentences
      are split into hard character windows as a size fallback.

    Requires NumPy. If the package uses optional extras, install with the
    ``semantic`` extra (``pip install <pkg>[semantic]``).
    """

    name = "semantic"

    def __init__(
        self,
        embedder: Embedder,
        similarity_threshold: float = 0.25,
        max_chars: int = 1200,
        min_chars: int | None = None,
        window_size: int = 1,
    ) -> None:
        if np is None:
            raise ImportError(
                "SemanticChunker requires NumPy. "
                "Install it with: pip install <pkg>[semantic]"
            )
        if not -1.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between -1 and 1")
        if max_chars < 1:
            raise ValueError("max_chars must be at least 1")
        if min_chars is not None and min_chars < 1:
            raise ValueError("min_chars must be at least 1")
        if min_chars is not None and min_chars > max_chars:
            raise ValueError("min_chars cannot exceed max_chars")
        if window_size < 1:
            raise ValueError("window_size must be at least 1")

        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.max_chars = max_chars
        self.min_chars = min(300, max_chars) if min_chars is None else min_chars
        # How many adjacent sentence pairs the boundary score averages.
        self.window_size = window_size

    # ------------------------------------------------------------------ #
    # Sentence segmentation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sentence_spans(text: str) -> list[tuple[int, int]]:
        """
        Find sentence-like spans while retaining every source character.

        Whitespace after punctuation belongs to the preceding span. Blank
        lines (runs of 2+ newlines) are boundaries; single newlines are
        not, so OCR text with one line per visual line isn't shredded.
        This is intentionally a lightweight baseline, not a full
        linguistic sentence tokenizer.
        """
        if not text:
            return []

        boundaries = [m.end() for m in _SENTENCE_BOUNDARY.finditer(text)]

        spans: list[tuple[int, int]] = []
        start = 0
        for boundary in boundaries:
            if boundary > start:
                spans.append((start, boundary))
                start = boundary
        if start < len(text):
            spans.append((start, len(text)))
        return spans

    # ------------------------------------------------------------------ #
    # Embedding helpers (vectorized)
    # ------------------------------------------------------------------ #
    def _embed_sentences(self, texts: list[str]) -> "np.ndarray":
        """Return an L2-normalized (n, d) float32 matrix."""
        embeddings = np.asarray(self.embedder.embed(texts), dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
            raise ValueError("Embedder must return one vector per sentence")

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Embedding vectors cannot be zero vectors")
        return embeddings / norms

    @staticmethod
    def _cosine_adjacent(matrix: "np.ndarray") -> "np.ndarray":
        """
        Cosine similarity between consecutive rows: (n-1,) vector.

        Rows are already L2-normalized, so cosine == dot product.
        """
        return np.einsum("ij,ij->i", matrix[:-1], matrix[1:])

    def _smooth(self, scores: "np.ndarray") -> "np.ndarray":
        """
        Moving average over the last ``window_size`` adjacent pairs.

        Windows at the start of the text are shorter (no zero-padding),
        so index i of the result is the mean of scores[max(0, i-w+1)..i].
        """
        window = self.window_size
        if window <= 1 or len(scores) == 0:
            return scores

        idx = np.arange(len(scores))
        starts = np.maximum(0, idx - (window - 1))
        cumulative = np.concatenate([[0.0], np.cumsum(scores, dtype=np.float64)])
        sums = cumulative[idx + 1] - cumulative[starts]
        return (sums / (idx + 1 - starts)).astype(np.float32)

    @staticmethod
    def _group_centroid(
        matrix: "np.ndarray",
        first: int,
        last: int,
    ) -> "np.ndarray":
        """L2-normalized mean embedding of sentences [first..last]."""
        centroid = matrix[first : last + 1].mean(axis=0)
        norm = np.linalg.norm(centroid)
        return centroid / norm if norm > 0 else centroid

    # ------------------------------------------------------------------ #
    # Semantic grouping
    # ------------------------------------------------------------------ #
    def _semantic_groups(
        self,
        text: str,
        spans: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        if not spans:
            return []

        sentence_texts = [text[s:e].strip() for s, e in spans]

        # Blank spans can occur in unusual OCR text; skip them instead of
        # failing the whole page.
        keep = [i for i, s in enumerate(sentence_texts) if s]
        if not keep:
            return []
        spans = [spans[i] for i in keep]
        sentence_texts = [sentence_texts[i] for i in keep]

        matrix = self._embed_sentences(sentence_texts)
        scores = self._smooth(self._cosine_adjacent(matrix))
        print("sentence_texts:", sentence_texts)
        print("normalized embeddings:", matrix)
        print("similarity scores:", scores)

        # Groups carry their sentence-index range so the merge pass can
        # compute per-group centroids without re-embedding.
        # (char_start, char_end, first_sentence_idx, last_sentence_idx)
        groups: list[tuple[int, int, int, int]] = []
        group_start = spans[0][0]
        group_sent_start = 0

        for index in range(1, len(spans)):
            # scores[index - 1] is the (smoothed) similarity of the pair
            # (index - 1, index). This is the actual boundary signal.
            if float(scores[index - 1]) < self.similarity_threshold:
                groups.append(
                    (group_start, spans[index - 1][1],
                     group_sent_start, index - 1)
                )
                group_start = spans[index][0]
                group_sent_start = index

        groups.append(
            (group_start, spans[-1][1], group_sent_start, len(spans) - 1)
        )
        return self._merge_tiny_groups(groups, matrix)

    def _merge_tiny_groups(
        self,
        groups: list[tuple[int, int, int, int]],
        matrix: "np.ndarray",
    ) -> list[tuple[int, int]]:
        """
        Merge undersized groups into their best neighbor.

        A neighbor qualifies only if the merged group fits within
        ``max_chars`` (size-aware). Among qualifying neighbors, prefer
        the one with the higher group-centroid cosine similarity; ties
        break toward the smaller merged size. If no neighbor qualifies,
        the tiny group is left for ``_enforce_max_chars``.
        """
        merged = [list(g) for g in groups]
        i = 0
        while i < len(merged):
            char_start, char_end, first, last = merged[i]
            if char_end - char_start >= self.min_chars:
                i += 1
                continue

            candidate = None  # (similarity, merged_size, direction)
            if i + 1 < len(merged):
                n_start, n_end, n_first, n_last = merged[i + 1]
                merged_size = n_end - char_start
                if merged_size <= self.max_chars:
                    sim = float(
                        np.dot(
                            self._group_centroid(matrix, first, last),
                            self._group_centroid(matrix, n_first, n_last),
                        )
                    )
                    candidate = (sim, merged_size, "forward")
            if i > 0:
                p_start, p_end, p_first, p_last = merged[i - 1]
                merged_size = char_end - p_start
                if merged_size <= self.max_chars:
                    sim = float(
                        np.dot(
                            self._group_centroid(matrix, first, last),
                            self._group_centroid(matrix, p_first, p_last),
                        )
                    )
                    if candidate is None or (sim, -merged_size) > (
                        candidate[0], -candidate[1]
                    ):
                        candidate = (sim, merged_size, "backward")

            if candidate is None:
                i += 1  # no size-valid neighbor; size fallback will handle it
                continue

            if candidate[2] == "forward":
                merged[i][1] = merged[i + 1][1]
                merged[i][3] = merged[i + 1][3]
                del merged[i + 1]
                # Re-check merged[i]: it may still be undersized.
            else:
                merged[i - 1][1] = merged[i][1]
                merged[i - 1][3] = merged[i][3]
                del merged[i]
                i -= 1  # Re-check the merged group now at i-1.

        return [(g[0], g[1]) for g in merged]

    # ------------------------------------------------------------------ #
    # Size enforcement
    # ------------------------------------------------------------------ #
    def _enforce_max_chars(
        self,
        text: str,
        start: int,
        end: int,
        sentence_spans: list[tuple[int, int]],
    ) -> list[tuple[int, int, str]]:
        """
        Enforce max_chars inside a semantic group.

        Prefer sentence boundaries. If one sentence itself is too long,
        split that sentence into hard character windows.
        """
        contained = [
            (s, e) for s, e in sentence_spans if s >= start and e <= end
        ] or [(start, end)]

        output: list[tuple[int, int, str]] = []
        current_start: int | None = None
        current_end: int | None = None

        for sentence_start, sentence_end in contained:
            if sentence_end - sentence_start > self.max_chars:
                # Flush pending short sentences first.
                if current_start is not None:
                    output.append((current_start, current_end, "semantic"))
                    current_start = current_end = None
                cursor = sentence_start
                while cursor < sentence_end:
                    piece_end = min(cursor + self.max_chars, sentence_end)
                    output.append((cursor, piece_end, "size_fallback"))
                    cursor = piece_end
                continue

            if current_start is None:
                current_start, current_end = sentence_start, sentence_end
            elif sentence_end - current_start <= self.max_chars:
                current_end = sentence_end
            else:
                output.append((current_start, current_end, "semantic"))
                current_start, current_end = sentence_start, sentence_end

        if current_start is not None:
            output.append((current_start, current_end, "semantic"))

        return output

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def chunk(self, pages: Sequence[object]) -> list[Chunk]:
        chunks: list[Chunk] = []

        for page in pages:
            text = page.text
            if not text:
                continue

            sentence_spans = self._sentence_spans(text)
            semantic_groups = self._semantic_groups(text, sentence_spans)

            pieces: list[tuple[int, int, str]] = []
            for start, end in semantic_groups:
                pieces.extend(
                    self._enforce_max_chars(text, start, end, sentence_spans)
                )

            for start, end, piece_strategy in pieces:
                identity = (
                    f"{page.document_id}|{page.source_file}|"
                    f"{page.page_number}|{start}|{end}|{self.name}"
                )
                chunk_id = hashlib.sha256(
                    identity.encode("utf-8")
                ).hexdigest()[:16]

                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        document_id=page.document_id,
                        source_file=page.source_file,
                        page_start=page.page_number,
                        page_end=page.page_number,
                        text=text[start:end],
                        strategy=(
                            self.name
                            if piece_strategy == "semantic"
                            else "size_fallback"
                        ),
                        metadata={
                            "start_offset": start,
                            "end_offset": end,
                            "similarity_threshold": self.similarity_threshold,
                            "max_chars": self.max_chars,
                            "min_chars": self.min_chars,
                            "window_size": self.window_size,
                        },
                    )
                )

        return chunks
