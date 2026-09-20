from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ChunkingResult:
    """
    Result returned by Vanka's public chunking API.
    """

    chunks: list

    strategy: str

    selection_score: float | None = None

    selection_reason: str | None = None

    metrics: dict[str, Any] | None = None

    document: str | None = None