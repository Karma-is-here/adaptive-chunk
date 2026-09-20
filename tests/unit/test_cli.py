from vanka.cli import build_parser


def test_cli_root_parser():
    parser = build_parser()

    args = parser.parse_args(
        [
            "process",
            "--documents",
            "data/normalized",
            "--output",
            "data/output",
        ]
    )

    assert args.command == "process"
    assert args.documents == "data/normalized"
    assert args.output == "data/output"
    assert args.benchmarks is None


def test_cli_process_with_benchmarks():
    parser = build_parser()

    args = parser.parse_args(
        [
            "process",
            "--documents",
            "data/normalized",
            "--benchmarks",
            "data/benchmarks",
            "--output",
            "data/output",
        ]
    )

    assert args.command == "process"
    assert args.benchmarks == "data/benchmarks"


def test_cli_chunk_auto():
    parser = build_parser()

    args = parser.parse_args(
        [
            "chunk",
            "data/normalized/investment/advisory/advisory-mandates.jsonl",
            "--strategy",
            "auto",
        ]
    )

    assert args.command == "chunk"
    assert args.strategy == "auto"
    assert args.document.endswith("advisory-mandates.jsonl")
