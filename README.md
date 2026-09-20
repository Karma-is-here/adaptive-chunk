@'
# Vanka Chunker

**Vanka** is an evaluation-driven document chunking engine for retrieval-augmented generation (RAG) systems.

It takes normalized documents, generates multiple chunking strategies, evaluates them when benchmark questions are available, selects an appropriate strategy, and produces provenance-rich chunk artifacts.

Vanka is designed to be a reusable chunking component rather than an end-to-end RAG system.

---

## What Vanka Does

Vanka provides:

- Multiple chunking strategies
- Structural document detection
- Benchmark-based retrieval evaluation
- Intrinsic chunk-quality evaluation when benchmarks are unavailable
- Automatic chunker selection
- Chunk-level provenance
- Reproducible JSON artifacts
- A Python API
- A command-line interface
- Unit tests for the core pipeline and CLI

### Chunking strategies

Vanka currently supports:

- `fixed`
- `recursive`
- `semantic`
- `structural`
- `auto`

`auto` evaluates the available strategies and selects the appropriate candidate.

---

## Architecture

```text
Normalized Documents
        │
        ▼
┌─────────────────────┐
│ Document Ingestion  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Structure Detection │
└──────────┬──────────┘
           │
           ▼
   ┌───────┼────────┬───────────┐
   ▼       ▼        ▼           ▼
 Fixed  Recursive Semantic  Structural
   │       │        │           │
   └───────┴────────┴───────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ Evaluation      │
          │                 │
          │ Benchmarks?     │
          └────────┬────────┘
                   │
          ┌────────┴────────┐
          ▼                 ▼
     Retrieval          Intrinsic
      Metrics            Metrics
          │                 │
          └────────┬────────┘
                   ▼
          ┌─────────────────┐
          │ Chunker Selector│
          └────────┬────────┘
                   ▼
          Selected Chunks
                   │
                   ▼
       Provenance + Artifacts