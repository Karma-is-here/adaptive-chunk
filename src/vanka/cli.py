from __future__ import annotations

import argparse
import json

from vanka import Vanka


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vanka",
        description="Evaluation-driven document chunking engine.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    process_parser = subparsers.add_parser(
        "process",
        help="Process normalized documents.",
    )

    process_parser.add_argument(
        "--documents",
        required=True,
        help="Directory containing normalized JSONL documents.",
    )

    process_parser.add_argument(
        "--benchmarks",
        default=None,
        help="Optional directory containing benchmark question files.",
    )

    process_parser.add_argument(
        "--output",
        required=True,
        help="Directory for generated chunks, reports, and manifest.",
    )

    chunk_parser = subparsers.add_parser(
        "chunk",
        help="Chunk a single normalized document.",
    )

    chunk_parser.add_argument(
        "document",
        help="Path to a normalized JSONL document.",
    )

    chunk_parser.add_argument(
        "--strategy",
        default="auto",
        choices=[
            "fixed",
            "recursive",
            "semantic",
            "structural",
            "auto",
        ],
        help="Chunking strategy.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    vanka = Vanka()

    if args.command == "process":
        result = vanka.process(
            documents=args.documents,
            benchmarks=args.benchmarks,
            output=args.output,
        )

        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )

    elif args.command == "chunk":
        result = vanka.chunk(
            document=args.document,
            strategy=args.strategy,
        )

        print(
            json.dumps(
                {
                    "strategy": result.strategy,
                    "chunks": len(result.chunks),
                    "selection_score": result.selection_score,
                    "selection_reason": result.selection_reason,
                    "metrics": result.metrics,
                    "document": result.document,
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )


if __name__ == "__main__":
    main()
