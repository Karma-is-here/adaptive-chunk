import json

from vanka.selection.selector import ChunkerSelector


RESULTS_PATH = "data/benchmarks/retrieval_results.json"


def main():
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)

    selector = ChunkerSelector()

    for document in results["documents"]:
        selection = selector.select(document["summaries"])

        print()
        print("=" * 70)
        print(document["document"])
        print("=" * 70)

        print(
            f"Selected strategy: {selection.selected_strategy}"
        )

        print(
            f"Selection score: {selection.selection_score:.4f}"
        )

        print(
            f"Reason: {selection.selection_reason}"
        )

        print()
        print("Candidates:")

        for candidate in selection.candidates:
            print(
                f"  {candidate['strategy']:12}"
                f" score={candidate['score']:.4f}"
            )

        print()
        print("Rejected:")

        for rejected in selection.rejected_candidates:
            print(
                f"  {rejected['strategy']:12}"
                f" reasons={rejected['reasons']}"
            )


if __name__ == "__main__":
    main()

