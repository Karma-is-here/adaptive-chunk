from abc import ABC, abstractmethod
from typing import Sequence

from models.chunk import Chunk


class Chunker(ABC):
    """Common interface for interchangeable chunking strategies."""

    name: str

    @abstractmethod
    def chunk(self, pages: Sequence[object]) -> list[Chunk]:
        """Chunk normalized pages while retaining source provenance."""
        raise NotImplementedError