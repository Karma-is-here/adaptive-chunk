from pathlib import Path
import json

from vanka import VankaPipeline


ROOT = Path(__file__).resolve().parents[1]

NORMALIZED_DIR = ROOT / "data" / "normalized"
BENCHMARKS_DIR = ROOT / "data" / "benchmarks"

OUTPUT_PATH = (
    ROOT
    / "data"
    / "benchmarks"
    / "retrieval_results.json"
)


def main():
    pipeline = VankaPipeline(
        normalized_dir=NORMALIZED_DIR,
        benchmarks_dir=BENCHMARKS_DIR,
    )

    results = pipeline.process_all()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"\nSaved detailed results to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()