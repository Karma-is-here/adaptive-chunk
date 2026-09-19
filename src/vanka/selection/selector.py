from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable


@dataclass(frozen=True)
class ChunkerMetrics:
    """
    Retrieval and chunk-quality metrics for one chunking strategy.
    """

    strategy: str
    chunks: int
    mean_chars: float
    median_chars: float
    under_100_chars: int

    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float

    @property
    def under_100_ratio(self) -> float:
        """Fraction of chunks shorter than 100 characters."""
        if self.chunks == 0:
            return 1.0

        return self.under_100_chars / self.chunks


@dataclass(frozen=True)
class SelectionResult:
    """
    Auditable result of dynamic chunker selection.
    """

    selected_strategy: str
    selection_score: float
    selection_reason: str

    selected_metrics: dict[str, Any]

    candidates: list[dict[str, Any]]
    rejected_candidates: list[dict[str, Any]]


class ChunkerSelector:
    """
    Select the most appropriate chunking strategy using retrieval
    performance and chunk-quality constraints.

    The selector does NOT perform retrieval or reranking.

    Expected input format is the same strategy-summary structure
    produced by the retrieval benchmark:

        {
            "strategy": "fixed",
            "chunks": 37,
            "mean_chars": 863.6,
            "median_chars": 1200,
            "under_100_chars": 3,
            "recall_at_1": 0.4667,
            "recall_at_3": 0.5333,
            "recall_at_5": 0.6,
            "mrr": 0.5056
        }
    """

    DEFAULT_WEIGHTS = {
        "recall_at_1": 0.30,
        "recall_at_3": 0.20,
        "recall_at_5": 0.15,
        "mrr": 0.35,
    }

    def __init__(
        self,
        *,
        weights: dict[str, float] | None = None,
        max_under_100_ratio: float = 0.50,
        min_median_chars: float = 100.0,
        min_chunks: int = 1,
    ) -> None:
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()

        self.max_under_100_ratio = max_under_100_ratio
        self.min_median_chars = min_median_chars
        self.min_chunks = min_chunks

        self._validate_weights()

    def _validate_weights(self) -> None:
        required = {
            "recall_at_1",
            "recall_at_3",
            "recall_at_5",
            "mrr",
        }

        if set(self.weights) != required:
            raise ValueError(
                "weights must contain exactly: "
                "recall_at_1, recall_at_3, recall_at_5, mrr"
            )

        if any(weight < 0 for weight in self.weights.values()):
            raise ValueError("weights cannot be negative")

        total = sum(self.weights.values())

        if total <= 0:
            raise ValueError("at least one weight must be positive")

    @staticmethod
    def _normalise_weights(weights: dict[str, float]) -> dict[str, float]:
        total = sum(weights.values())

        return {
            key: value / total
            for key, value in weights.items()
        }

    @staticmethod
    def _metric_value(value: Any) -> float:
        """
        Safely convert a benchmark metric to float.

        Missing/None values are treated as 0 so that a malformed
        strategy cannot accidentally win the selection.
        """
        if value is None:
            return 0.0

        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _parse_metrics(self, summary: dict[str, Any]) -> ChunkerMetrics:
        required = [
            "strategy",
            "chunks",
            "mean_chars",
            "median_chars",
            "under_100_chars",
            "recall_at_1",
            "recall_at_3",
            "recall_at_5",
            "mrr",
        ]

        missing = [
            field
            for field in required
            if field not in summary
        ]

        if missing:
            raise ValueError(
                f"Missing benchmark fields for strategy "
                f"{summary.get('strategy', '<unknown>')}: {missing}"
            )

        return ChunkerMetrics(
            strategy=str(summary["strategy"]),
            chunks=int(summary["chunks"]),
            mean_chars=float(summary["mean_chars"]),
            median_chars=float(summary["median_chars"]),
            under_100_chars=int(summary["under_100_chars"]),
            recall_at_1=self._metric_value(summary["recall_at_1"]),
            recall_at_3=self._metric_value(summary["recall_at_3"]),
            recall_at_5=self._metric_value(summary["recall_at_5"]),
            mrr=self._metric_value(summary["mrr"]),
        )

    def _quality_gate(
        self,
        metrics: ChunkerMetrics,
    ) -> tuple[bool, list[str]]:
        """
        Determine whether a chunking strategy is acceptable.

        These are safety/quality constraints, not performance scores.
        """

        reasons: list[str] = []

        if metrics.chunks < self.min_chunks:
            reasons.append(
                f"too_few_chunks ({metrics.chunks} < {self.min_chunks})"
            )

        if metrics.median_chars < self.min_median_chars:
            reasons.append(
                "median_chunk_too_small "
                f"({metrics.median_chars:.1f} < "
                f"{self.min_median_chars:.1f})"
            )

        if metrics.under_100_ratio > self.max_under_100_ratio:
            reasons.append(
                "too_many_undersized_chunks "
                f"({metrics.under_100_ratio:.2%} > "
                f"{self.max_under_100_ratio:.2%})"
            )

        return not reasons, reasons

    def _score(self, metrics: ChunkerMetrics) -> float:
        """
        Calculate the retrieval-performance score.

        Higher is better.
        """

        weights = self._normalise_weights(self.weights)

        score = (
            weights["recall_at_1"] * metrics.recall_at_1
            + weights["recall_at_3"] * metrics.recall_at_3
            + weights["recall_at_5"] * metrics.recall_at_5
            + weights["mrr"] * metrics.mrr
        )

        return round(score, 6)

    def select(
        self,
        summaries: Iterable[dict[str, Any]],
    ) -> SelectionResult:
        """
        Select one chunking strategy.

        Parameters
        ----------
        summaries:
            Strategy summaries from retrieval_benchmark.py.

        Returns
        -------
        SelectionResult
            Contains selected strategy, score, metrics, candidates,
            and rejection reasons.
        """

        parsed: list[ChunkerMetrics] = [
            self._parse_metrics(summary)
            for summary in summaries
        ]

        if not parsed:
            raise ValueError(
                "No chunker benchmark results were provided."
            )

        candidates: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []

        for metrics in parsed:
            passed, reasons = self._quality_gate(metrics)

            score = self._score(metrics)

            record = {
                "strategy": metrics.strategy,
                "score": score,
                "metrics": asdict(metrics),
                "under_100_ratio": round(
                    metrics.under_100_ratio,
                    4,
                ),
            }

            if passed:
                candidates.append(record)
            else:
                rejected.append(
                    {
                        **record,
                        "reasons": reasons,
                    }
                )

        # If every strategy fails the quality gate, choose the
        # highest-scoring strategy but explicitly record that the
        # fallback was necessary.
        if not candidates:
            fallback = max(
                rejected,
                key=lambda item: item["score"],
            )

            return SelectionResult(
                selected_strategy=fallback["strategy"],
                selection_score=fallback["score"],
                selection_reason=(
                    "All candidate strategies failed the quality "
                    "gate; selected the highest-scoring strategy "
                    "as a fallback. Review the rejection reasons."
                ),
                selected_metrics=fallback["metrics"],
                candidates=[],
                rejected_candidates=rejected,
            )

        # Highest retrieval-performance score wins among
        # strategies that pass the quality gate.
        selected = max(
            candidates,
            key=lambda item: item["score"],
        )

        return SelectionResult(
            selected_strategy=selected["strategy"],
            selection_score=selected["score"],
            selection_reason=(
                "Selected the highest-scoring strategy among "
                "candidates that passed the chunk-quality gates."
            ),
            selected_metrics=selected["metrics"],
            candidates=sorted(
                candidates,
                key=lambda item: item["score"],
                reverse=True,
            ),
            rejected_candidates=rejected,
        )
