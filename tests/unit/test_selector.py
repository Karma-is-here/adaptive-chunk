from vanka.selection.selector import ChunkerSelector


def test_selector_chooses_best_valid_strategy():
    summaries = [
        {
            "strategy": "fixed",
            "chunks": 20,
            "mean_chars": 800,
            "median_chars": 700,
            "under_100_chars": 0,
            "recall_at_1": 0.40,
            "recall_at_3": 0.70,
            "recall_at_5": 0.80,
            "mrr": 0.60,
        },
        {
            "strategy": "semantic",
            "chunks": 25,
            "mean_chars": 650,
            "median_chars": 600,
            "under_100_chars": 1,
            "recall_at_1": 0.60,
            "recall_at_3": 0.80,
            "recall_at_5": 0.90,
            "mrr": 0.75,
        },
    ]

    selector = ChunkerSelector()
    result = selector.select(summaries)

    assert result.selected_strategy == "semantic"


def test_selector_rejects_pathological_chunking():
    summaries = [
        {
            "strategy": "structural",
            "chunks": 89,
            "mean_chars": 327,
            "median_chars": 30,
            "under_100_chars": 59,
            "recall_at_1": 0.80,
            "recall_at_3": 0.90,
            "recall_at_5": 0.95,
            "mrr": 0.85,
        },
        {
            "strategy": "fixed",
            "chunks": 37,
            "mean_chars": 864,
            "median_chars": 1200,
            "under_100_chars": 3,
            "recall_at_1": 0.47,
            "recall_at_3": 0.53,
            "recall_at_5": 0.60,
            "mrr": 0.51,
        },
    ]

    selector = ChunkerSelector()
    result = selector.select(summaries)

    assert result.selected_strategy == "fixed"
    assert len(result.rejected_candidates) == 1
    assert result.rejected_candidates[0]["strategy"] == "structural"

