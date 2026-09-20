from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable


@dataclass(frozen=True)
class ChunkerMetrics:
    """
    Retrieval and chunk-quality metrics for one chunking strategy.
    Retrieval metrics may be None when no benchmark questions exist.
    """

    strategy: str
    chunks: int
    mean_chars: float
    median_chars: float
    under_100_chars: int

    recall_at_1: float | None
    recall_at_3: float | None
    recall_at_5: float | None
    mrr: float | None

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
    Select the most appropriate chunking strategy.

    When benchmark retrieval metrics are available, selection uses
    retrieval performance plus chunk-quality gates.

    When retrieval metrics are unavailable, selection uses intrinsic
    chunk-quality metrics only.

    The selector does NOT perform retrieval or reranking.
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
    def _normalise_weights(
        weights: dict[str, float],
    ) -> dict[str, float]:
        total = sum(weights.values())

        return {
            key: value / total
            for key, value in weights.items()
        }

    @staticmethod
    def _metric_value(
        value: Any,
    ) -> float | None:
        """
        Convert a metric to float without turning missing values
        into zero.

        None means that the metric is unavailable.
        """
        if value is None:
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _parse_metrics(
        self,
        summary: dict[str, Any],
    ) -> ChunkerMetrics:
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
            recall_at_1=self._metric_value(
                summary["recall_at_1"]
            ),
            recall_at_3=self._metric_value(
                summary["recall_at_3"]
            ),
            recall_at_5=self._metric_value(
                summary["recall_at_5"]
            ),
            mrr=self._metric_value(
                summary["mrr"]
            ),
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

    def _has_retrieval_metrics(
        self,
        metrics: ChunkerMetrics,
    ) -> bool:
        """
        Return True when at least one retrieval metric is available.
        """

        return any(
            value is not None
            for value in (
                metrics.recall_at_1,
                metrics.recall_at_3,
                metrics.recall_at_5,
                metrics.mrr,
            )
        )

    def _retrieval_score(
        self,
        metrics: ChunkerMetrics,
    ) -> float:
        """
        Calculate retrieval-performance score using only metrics
        that are actually available.

        Missing metrics are excluded and the remaining weights
        are renormalized.
        """

        available = {
            "recall_at_1": metrics.recall_at_1,
            "recall_at_3": metrics.recall_at_3,
            "recall_at_5": metrics.recall_at_5,
            "mrr": metrics.mrr,
        }

        available = {
            key: value
            for key, value in available.items()
            if value is not None
        }

        if not available:
            raise ValueError(
                "No retrieval metrics are available."
            )

        available_weights = {
            key: self.weights[key]
            for key in available
        }

        weights = self._normalise_weights(
            available_weights
        )

        score = sum(
            weights[key] * available[key]
            for key in available
        )

        return round(score, 6)

    def _intrinsic_score(
        self,
        metrics: ChunkerMetrics,
    ) -> float:
        """
        Calculate an intrinsic chunk-quality score.

        This is used when benchmark questions are unavailable.

        The score rewards:
        - useful chunk sizes
        - fewer extremely small chunks

        The quality gates are still applied separately.
        """

        # 1200 characters is the current target chunk size used
        # by Vanka's built-in strategies.
        size_score = min(
            metrics.median_chars / 1200.0,
            1.0,
        )

        small_chunk_score = max(
            0.0,
            1.0 - metrics.under_100_ratio,
        )

        score = (
            0.60 * size_score
            + 0.40 * small_chunk_score
        )

        return round(score, 6)

    def _score(
        self,
        metrics: ChunkerMetrics,
    ) -> float:
        """
        Calculate the appropriate strategy score.

        Retrieval metrics available:
            retrieval score.

        No retrieval metrics available:
            intrinsic chunk-quality score.
        """

        if self._has_retrieval_metrics(metrics):
            return self._retrieval_score(metrics)

        return self._intrinsic_score(metrics)

    def select(
        self,
        summaries: Iterable[dict[str, Any]],
    ) -> SelectionResult:
        """
        Select one chunking strategy.

        Retrieval benchmarks are used when available.

        If no retrieval metrics exist, intrinsic chunk-quality
        metrics are used instead.
        """

        parsed: list[ChunkerMetrics] = [
            self._parse_metrics(summary)
            for summary in summaries
        ]

        if not parsed:
            raise ValueError(
                "No chunker benchmark results were provided."
            )

        retrieval_available = any(
            self._has_retrieval_metrics(metrics)
            for metrics in parsed
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

        selected = max(
            candidates,
            key=lambda item: item["score"],
        )

        if retrieval_available:
            reason = (
                "Selected the highest-scoring strategy among "
                "candidates that passed the chunk-quality gates "
                "using available retrieval metrics."
            )
        else:
            reason = (
                "No benchmark retrieval metrics were available; "
                "selected the highest-scoring strategy using "
                "intrinsic chunk-quality metrics."
            )

        return SelectionResult(
            selected_strategy=selected["strategy"],
            selection_score=selected["score"],
            selection_reason=reason,
            selected_metrics=selected["metrics"],
            candidates=sorted(
                candidates,
                key=lambda item: item["score"],
                reverse=True,
            ),
            rejected_candidates=rejected,
        )