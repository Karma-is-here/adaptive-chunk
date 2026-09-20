# Vanka Chunker

**Vanka** is an evaluation-driven document chunking engine designed for Retrieval-Augmented Generation (RAG) systems.

Vanka generates multiple candidate chunkings for a document, evaluates them using benchmark retrieval performance when benchmark questions are available, falls back to intrinsic chunk-quality metrics when benchmarks are unavailable, and automatically selects a chunking strategy.

Vanka is designed as a reusable **chunking and evaluation layer** that can sit upstream of a RAG pipeline.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [What Vanka Does and Does Not Do](#what-vanka-does-and-does-not-do)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Verify Installation](#verify-installation)
- [Project Structure](#project-structure)
- [Input Data Format](#input-data-format)
- [Benchmark Format](#benchmark-format)
- [Chunking Strategies](#chunking-strategies)
  - [Fixed](#1-fixed)
  - [Recursive](#2-recursive)
  - [Semantic](#3-semantic)
  - [Structural](#4-structural)
  - [Auto](#5-auto)
- [Using Vanka](#using-vanka)
  - [Python API](#python-api)
  - [Single Document](#single-document)
  - [Batch Processing](#batch-processing)
  - [CLI](#cli)
- [Benchmark-Aware Processing](#benchmark-aware-processing)
- [Processing Without Benchmarks](#processing-without-benchmarks)
- [Mixed Benchmark Collections](#mixed-benchmark-collections)
- [Strategy Selection](#strategy-selection)
- [Retrieval Evaluation](#retrieval-evaluation)
- [Intrinsic Evaluation](#intrinsic-evaluation)
- [Quality Gates](#quality-gates)
- [Chunk Provenance](#chunk-provenance)
- [Output Artifacts](#output-artifacts)
- [Output Directory Structure](#output-directory-structure)
- [Chunk Artifact Schema](#chunk-artifact-schema)
- [Report Schema](#report-schema)
- [Manifest](#manifest)
- [Testing](#testing)
- [Development](#development)
- [Integration with a RAG System](#integration-with-a-rag-system)
- [Design Principles](#design-principles)
- [Current Scope](#current-scope)
- [Future Extensions](#future-extensions)
- [Status](#status)
- [License](#license)

---

# Overview

Traditional document chunking often uses a single strategy:

```text
Document
   ↓
Fixed-size chunks
   ↓
Vector database
````

However, different documents have different structures and characteristics.

For example:

* A dense financial report may benefit from recursive chunking.
* A document with coherent paragraphs may benefit from semantic chunking.
* A structured FAQ may benefit from structural chunking.
* A document without benchmark questions may need a strategy selected using intrinsic characteristics rather than retrieval evaluation.

Vanka treats chunking as an **evaluation problem**.

Instead of assuming that one chunking strategy is always appropriate:

```text
Document
   │
   ├── Fixed
   ├── Recursive
   ├── Semantic
   └── Structural
          │
          ▼
      Evaluation
          │
          ▼
       Selection
          │
          ▼
    Selected chunks
```

---

# Key Features

Vanka currently provides:

* Fixed-size chunking
* Recursive chunking
* Semantic chunking
* Structural chunking
* Automatic chunker selection
* Benchmark-based retrieval evaluation
* Intrinsic chunk-quality evaluation
* Recall@1
* Recall@3
* Recall@5
* Mean Reciprocal Rank (MRR)
* Chunk-quality gates
* Chunk-level provenance
* Source page tracking
* Character-offset tracking
* Section/heading tracking where available
* Reproducible JSON artifacts
* Document-level selection reports
* Run-level manifest
* Python API
* Command-line interface
* Automated tests

---

# Architecture

```text
                         VANKA
                           │
                           ▼
                 Normalized Documents
                           │
                           ▼
                 ┌───────────────────┐
                 │ Document Loading  │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ Structure         │
                 │ Detection         │
                 └─────────┬─────────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
           Fixed       Recursive     Semantic
              │            │            │
              └────────────┼────────────┘
                           │
                           ▼
                      Structural
                           │
                           ▼
                 Candidate Chunk Sets
                           │
                           ▼
                 ┌───────────────────┐
                 │ Evaluation        │
                 └─────────┬─────────┘
                           │
                  ┌────────┴────────┐
                  │                 │
           Benchmarks?              │
                  │                 │
            ┌─────┴─────┐           │
            │           │           │
           YES          NO          │
            │           │           │
            ▼           ▼           │
       Retrieval    Intrinsic       │
        Metrics      Metrics        │
            │           │           │
            └─────┬─────┘           │
                  │                 │
                  ▼                 │
            Chunker Selector
                  │
                  ▼
           Selected Strategy
                  │
                  ▼
            Selected Chunks
                  │
                  ▼
        Provenance + Artifacts
```

---

# What Vanka Does and Does Not Do

## Vanka handles

```text
Document loading
      ↓
Structure detection
      ↓
Chunk generation
      ↓
Chunk evaluation
      ↓
Strategy selection
      ↓
Provenance
      ↓
Artifact generation
```

## Vanka does not handle

Vanka intentionally does **not** implement the downstream RAG system.

The following belong to the application consuming Vanka:

```text
Embedding storage
        ↓
Vector database
        ↓
Hybrid retrieval
        ↓
Reranking
        ↓
Context assembly
        ↓
LLM generation
        ↓
Agent orchestration
        ↓
Answer generation
```

This separation keeps Vanka focused on chunk generation, evaluation, and selection.

---

# Requirements

## Python

Vanka requires:

```text
Python >= 3.12
```

Check your Python version:

```bash
python --version
```

---

# Installation

Clone the repository:

```bash
git clone <repository-url>
cd vanka
```

Create a virtual environment.

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install Vanka:

```bash
python -m pip install -e .
```

The editable installation allows changes to the source code to be reflected immediately.

---

# Dependencies

The base package is intentionally lightweight.

## Core

The package metadata currently does not require external runtime dependencies for the base installation.

## Semantic Chunking

Semantic chunking requires:

```text
sentence-transformers >= 3.0
```

Install it with:

```bash
python -m pip install -e ".[semantic]"
```

## Testing

Testing requires:

```text
pytest >= 8.0
```

Install it with:

```bash
python -m pip install -e ".[test]"
```

## Development Installation

For development with semantic chunking and testing:

```bash
python -m pip install -e ".[semantic,test]"
```

---

# Verify Installation

Verify the Python API:

```bash
python -c "from vanka import Vanka; print(Vanka)"
```

Expected:

```text
<class 'vanka.api.Vanka'>
```

Verify the CLI:

```bash
vanka --help
```

Expected commands:

```text
process
chunk
```

---

# Project Structure

```text
vanka/
│
├── .gitignore
├── pyproject.toml
├── README.md
│
├── docs/
│
├── scripts/
│   ├── benchmark_retrieval.py
│   └── ...
│
├── tests/
│   └── unit/
│       ├── test_chunking.py
│       ├── test_selector.py
│       ├── test_api.py
│       ├── test_artifacts.py
│       └── test_cli.py
│
├── src/
│   └── vanka/
│       │
│       ├── ingestion/
│       │   └── ...
│       │
│       ├── structure/
│       │   └── detector.py
│       │
│       ├── models/
│       │   ├── chunk.py
│       │   ├── benchmark.py
│       │   ├── chunking_result.py
│       │   └── result.py
│       │
│       ├── chunking/
│       │   ├── fixed.py
│       │   ├── recursive.py
│       │   ├── semantic.py
│       │   ├── structural.py
│       │   └── embeddings.py
│       │
│       ├── evaluation/
│       │   └── ...
│       │
│       ├── selection/
│       │   └── selector.py
│       │
│       ├── pipeline/
│       │   ├── runner.py
│       │   ├── artifacts.py
│       │   └── __init__.py
│       │
│       ├── api.py
│       ├── cli.py
│       └── __init__.py
│
└── data/
    ├── normalized/
    ├── benchmarks/
    └── output/
```

The `data/` directory is intended for local input data, benchmarks, and generated artifacts and is ignored by Git.

---

# Input Data Format

Vanka operates on **normalized JSONL documents**.

A normalized document is represented as a `.jsonl` file.

Example:

```text
data/
└── normalized/
    └── investment/
        └── advisory/
            └── advisory-mandates.jsonl
```

Vanka preserves the directory hierarchy during processing.

Example:

```text
data/normalized/
├── investment/
│   ├── advisory/
│   │   ├── advisory-mandates.jsonl
│   │   ├── finsa.jsonl
│   │   └── investment-advice.jsonl
│   │
│   └── discretionary/
│       └── discretionary-mandates.jsonl
│
├── overview/
│   └── ...
│
└── technology/
    └── ...
```

The normalized document is the input to the chunking pipeline.

---

# Benchmark Format

Benchmark files use the naming convention:

```text
<document_name>_questions.json
```

For example:

```text
Normalized document:

data/normalized/investment/advisory/advisory-mandates.jsonl

Benchmark:

data/benchmarks/investment/advisory/advisory-mandates_questions.json
```

The benchmark directory mirrors the normalized document directory structure.

Example:

```text
data/
├── normalized/
│   └── investment/
│       └── advisory/
│           └── advisory-mandates.jsonl
│
└── benchmarks/
    └── investment/
        └── advisory/
            └── advisory-mandates_questions.json
```

A benchmark file contains a JSON array:

```json
[
  {
    "question": "Who has the final say on investment decisions?",
    "answer": "final say on any investment decision",
    "page": 1
  },
  {
    "question": "What do advisory mandates offer?",
    "answer": "expert investment strategies",
    "page": 1
  }
]
```

Each benchmark item contains:

| Field      | Description          |
| ---------- | -------------------- |
| `question` | Retrieval question   |
| `answer`   | Expected answer span |
| `page`     | Source page          |

---

# Chunking Strategies

Vanka supports four explicit chunking strategies and one automatic strategy:

```text
fixed
recursive
semantic
structural
auto
```

---

# 1. Fixed

Fixed-size chunking divides a document into chunks of approximately a configured character size.

The implementation also supports overlap between neighboring chunks.

Conceptually:

```text
Document
────────────────────────────────────────────

[──────────── Chunk 1 ────────────]
                    [──────────── Chunk 2 ────────────]
                                        [──── Chunk 3 ────]
```

### Advantages

* Simple
* Predictable
* Fast
* Stable chunk sizes
* Easy to reason about

### Limitations

* May split concepts across boundaries
* Does not inherently understand document structure
* May separate questions from their answers

### Python

```python
from vanka import Vanka

vanka = Vanka()

result = vanka.chunk(
    "data/normalized/investment/advisory/advisory-mandates.jsonl",
    strategy="fixed",
)

print(result.strategy)
print(len(result.chunks))
```

---

# 2. Recursive

Recursive chunking attempts to split text using hierarchical boundaries before falling back to smaller boundaries.

Conceptually:

```text
Document
   │
   ├── Paragraph
   │      │
   │      ├── Sentence
   │      └── Sentence
   │
   └── Paragraph
```

### Advantages

* Better boundary preservation than purely fixed-size chunking
* Attempts to preserve textual units
* Useful when explicit document structure is limited

---

# 3. Semantic

Semantic chunking uses sentence-level embeddings to identify semantically related regions.

Conceptually:

```text
Sentences
    ↓
Embeddings
    ↓
Semantic similarity
    ↓
Boundary detection
    ↓
Semantic chunks
```

Semantic chunking requires:

```text
sentence-transformers >= 3.0
```

Install it using:

```bash
python -m pip install -e ".[semantic]"
```

### Advantages

* Attempts to preserve semantic coherence
* Useful when topic boundaries are not explicitly represented

### Tradeoffs

* More computationally expensive
* Requires an embedding model
* Results depend on embedding quality and configured thresholds

---

# 4. Structural

Structural chunking uses detected document structure such as headings and sections.

Conceptually:

```text
Document
│
├── Introduction
│
├── Advisory Mandates
│   │
│   ├── Control
│   ├── Dedicated Specialist
│   └── Portfolio Monitoring
│
└── Frequently Asked Questions
```

Structural chunks can preserve section information:

```json
{
  "section": "Frequently Asked Questions"
}
```

### Advantages

* Preserves explicit document structure
* Useful for reports, FAQs, manuals, and structured documents
* Provides meaningful section provenance

### Limitations

* Depends on successful structure detection
* Poorly structured documents may produce fragmented chunks

---

# 5. Auto

`auto` is Vanka's automatic strategy-selection mode.

Vanka generates candidate chunkings:

```text
Fixed
Recursive
Semantic
Structural
```

and evaluates them.

With benchmarks:

```text
Candidate chunkings
       ↓
Retrieval evaluation
       ↓
Quality gates
       ↓
Selector
       ↓
Selected strategy
```

Without benchmarks:

```text
Candidate chunkings
       ↓
Intrinsic evaluation
       ↓
Quality gates
       ↓
Selector
       ↓
Selected strategy
```

---

# Using Vanka

Vanka can be used through:

1. Python API
2. CLI

---

# Python API

Import:

```python
from vanka import Vanka
```

Create an instance:

```python
vanka = Vanka()
```

---

# Single Document

## Fixed

```python
result = vanka.chunk(
    "data/normalized/investment/advisory/advisory-mandates.jsonl",
    strategy="fixed",
)
```

## Recursive

```python
result = vanka.chunk(
    "data/normalized/investment/advisory/advisory-mandates.jsonl",
    strategy="recursive",
)
```

## Semantic

```python
result = vanka.chunk(
    "data/normalized/investment/advisory/advisory-mandates.jsonl",
    strategy="semantic",
)
```

## Structural

```python
result = vanka.chunk(
    "data/normalized/investment/advisory/advisory-mandates.jsonl",
    strategy="structural",
)
```

## Auto

```python
result = vanka.chunk(
    "data/normalized/investment/advisory/advisory-mandates.jsonl",
    strategy="auto",
)
```

Inspect the result:

```python
print("Selected strategy:", result.strategy)
print("Chunks:", len(result.chunks))
print("Selection score:", result.selection_score)
print("Selection reason:", result.selection_reason)
print("Metrics:", result.metrics)
```

---

# `chunk()` vs `process()`

Vanka intentionally separates single-document chunking from the complete processing pipeline.

## `chunk()`

Use `chunk()` when you want to chunk one normalized document:

```python
result = vanka.chunk(
    document,
    strategy="fixed",
)
```

or:

```python
result = vanka.chunk(
    document,
    strategy="auto",
)
```

## `process()`

Use `process()` when you want the complete document-processing workflow including:

* recursive document discovery
* benchmark loading
* retrieval evaluation
* intrinsic evaluation
* strategy selection
* artifact generation
* reports
* manifest

Example:

```python
result = vanka.process(
    documents="data/normalized",
    benchmarks="data/benchmarks",
    output="data/output",
)
```

---

# Batch Processing

Process all normalized documents recursively:

```python
from vanka import Vanka

vanka = Vanka()

result = vanka.process(
    documents="data/normalized",
    output="data/output",
)
```

---

# Batch Processing With Benchmarks

```python
from vanka import Vanka

vanka = Vanka()

result = vanka.process(
    documents="data/normalized",
    benchmarks="data/benchmarks",
    output="data/output",
)
```

Vanka automatically maps normalized documents to their corresponding benchmark files.

Example:

```text
data/normalized/investment/advisory/finsa.jsonl
                         │
                         ▼
data/benchmarks/investment/advisory/finsa_questions.json
```

---

# Processing Without Benchmarks

Benchmarks are optional.

You can run:

```python
result = vanka.process(
    documents="data/normalized",
    output="data/output",
)
```

For documents without benchmark questions:

```text
question_count = 0
```

and retrieval metrics are represented as:

```json
{
  "recall_at_1": null,
  "recall_at_3": null,
  "recall_at_5": null,
  "mrr": null
}
```

This is intentional.

`null` means:

> Retrieval performance was not evaluated because benchmark questions were unavailable.

It does **not** mean retrieval performance was zero.

Vanka then falls back to intrinsic chunk-quality metrics.

---

# Mixed Benchmark Collections

Benchmark coverage does not have to be complete.

For example:

```text
data/
├── normalized/
│   ├── document-a.jsonl
│   ├── document-b.jsonl
│   └── document-c.jsonl
│
└── benchmarks/
    ├── document-a_questions.json
    └── document-c_questions.json
```

The processing behavior becomes:

```text
document-a
    ↓
retrieval evaluation

document-b
    ↓
intrinsic evaluation

document-c
    ↓
retrieval evaluation
```

Each document is evaluated independently.

---

# Benchmark-Aware Processing

When benchmark questions exist, Vanka evaluates each candidate strategy against those questions.

Conceptually:

```text
Document
   │
   ├── Fixed
   ├── Recursive
   ├── Semantic
   └── Structural
        │
        ▼
Benchmark Questions
        │
        ▼
Retrieval Evaluation
        │
        ├── Recall@1
        ├── Recall@3
        ├── Recall@5
        └── MRR
        │
        ▼
Quality Gates
        │
        ▼
Chunker Selector
        │
        ▼
Selected Strategy
```

---

# Retrieval Evaluation

Vanka currently tracks:

* Recall@1
* Recall@3
* Recall@5
* Mean Reciprocal Rank (MRR)

## Recall@1

Measures whether the relevant chunk appears in the top-1 retrieved result.

Conceptually:

```text
Recall@1 =
questions with relevant top-1 chunk
------------------------------------
total questions
```

## Recall@3

Measures whether the relevant chunk appears within the top 3 retrieved results.

## Recall@5

Measures whether the relevant chunk appears within the top 5 retrieved results.

## Mean Reciprocal Rank

MRR measures the rank of the first relevant result.

Conceptually:

```text
MRR = average(1 / rank_of_first_relevant_result)
```

---

# Intrinsic Evaluation

When benchmark questions are unavailable, Vanka evaluates candidate chunkers using intrinsic chunk-quality characteristics.

The selector considers characteristics including:

* chunk count
* mean chunk length
* median chunk length
* proportion of very small chunks

This allows Vanka to operate without treating missing retrieval metrics as zero.

The decision path is:

```text
Benchmark available
        │
        ▼
Retrieval metrics
        │
        ▼
Selector


Benchmark unavailable
        │
        ▼
Intrinsic metrics
        │
        ▼
Selector
```

---

# Strategy Selection

Vanka's selector evaluates candidate chunking strategies.

The process is:

```text
Candidate strategies
        ↓
Quality gates
        ↓
Available metrics
        ↓
Candidate scores
        ↓
Selected strategy
```

When benchmark retrieval metrics exist, the selector uses the available retrieval metrics.

When benchmark retrieval metrics do not exist, it uses intrinsic metrics.

The selection decision is recorded in the document report.

Example:

```json
{
  "selected_strategy": "fixed",
  "selection_score": 0.656667,
  "selection_reason": "Selected the highest-scoring strategy among candidates that passed the chunk-quality gates using available retrieval metrics."
}
```

---

# Quality Gates

Candidate strategies are subjected to chunk-quality constraints before selection.

The current selector considers:

* minimum median chunk size
* maximum proportion of very small chunks
* minimum number of chunks

Candidates that fail these gates are retained in the audit information but are not eligible for selection.

Example:

```json
{
  "strategy": "structural",
  "rejected": true,
  "rejection_reason": "..."
}
```

This makes the automatic selection process auditable.

---

# Selection Audit

A document report records:

```text
Selected strategy
Selection score
Selection reason
Candidate strategies
Candidate scores
Candidate metrics
Rejected candidates
Rejection reasons
```

This means the pipeline records not only:

```text
What was selected?
```

but also:

```text
What alternatives were considered?
Why were they rejected?
What evidence was used?
```

---

# Chunk Provenance

Every generated chunk carries source provenance.

Example:

```json
{
  "chunk_id": "a1058d7e0508b3b7",
  "document_id": "investment/advisory/advisory-mandates",
  "source": {
    "file": "advisory-mandates.pdf",
    "page_start": 1,
    "page_end": 1,
    "section": null,
    "offsets": {
      "start": 0,
      "end": 1200
    }
  }
}
```

Each chunk can therefore be traced back to:

```text
Chunk
  ↓
Document
  ↓
Source file
  ↓
Page
  ↓
Section
  ↓
Character offsets
```

This is useful for downstream retrieval systems that need source citations or audit trails.

---

# Provenance Fields

| Field                  | Description                              |
| ---------------------- | ---------------------------------------- |
| `chunk_id`             | Stable chunk identifier                  |
| `document_id`          | Logical document identifier              |
| `source.file`          | Original source file                     |
| `source.page_start`    | Starting page                            |
| `source.page_end`      | Ending page                              |
| `source.section`       | Detected section/heading, when available |
| `source.offsets.start` | Character start offset                   |
| `source.offsets.end`   | Character end offset                     |
| `text`                 | Chunk text                               |
| `chunking.strategy`    | Strategy used to produce the chunk       |
| `chunking.chunk_index` | Position within selected chunk set       |
| `metadata`             | Strategy-specific metadata               |

---

# Output Artifacts

When an output directory is provided:

```python
output="data/output"
```

Vanka generates:

```text
data/output/
├── chunks/
├── reports/
└── manifest.json
```

---

# Output Directory Structure

The normalized document hierarchy is preserved.

For example:

```text
data/normalized/
└── investment/
    └── advisory/
        └── advisory-mandates.jsonl
```

produces:

```text
data/output/
├── chunks/
│   └── investment/
│       └── advisory/
│           └── advisory-mandates.json
│
├── reports/
│   └── investment/
│       └── advisory/
│           └── advisory-mandates.json
│
└── manifest.json
```

---

# Chunk Artifact Schema

A generated chunk artifact has the following general structure:

```json
{
  "schema_version": "1.0",
  "document": "data/normalized/investment/advisory/advisory-mandates.jsonl",
  "strategy": "fixed",
  "chunks": [
    {
      "chunk_id": "a1058d7e0508b3b7",
      "document_id": "investment/advisory/advisory-mandates",
      "source": {
        "file": "advisory-mandates.pdf",
        "page_start": 1,
        "page_end": 1,
        "section": null,
        "offsets": {
          "start": 0,
          "end": 1200
        }
      },
      "text": "...",
      "chunking": {
        "strategy": "fixed",
        "chunk_index": 0
      },
      "metadata": {
        "start_offset": 0,
        "end_offset": 1200,
        "chunk_size": 1200,
        "overlap": 150
      }
    }
  ]
}
```

---

# Metadata

Metadata is strategy-specific.

For fixed chunking:

```json
{
  "start_offset": 0,
  "end_offset": 1200,
  "chunk_size": 1200,
  "overlap": 150
}
```

Semantic chunking may contain semantic threshold, window, and fallback information.

Structural chunking may contain section-related information.

The artifact separates:

```text
source
```

for provenance from:

```text
chunking
metadata
```

for chunk-generation information.

---

# Report Schema

Each processed document receives a report.

General structure:

```json
{
  "schema_version": "1.0",
  "document": "investment/advisory/advisory-mandates.jsonl",
  "page_count": 4,
  "question_count": 10,
  "summaries": [],
  "selection": {
    "selected_strategy": "fixed",
    "selection_score": 0.656667,
    "selection_reason": "Selected the highest-scoring strategy among candidates that passed the chunk-quality gates using available retrieval metrics.",
    "selected_metrics": {},
    "candidates": [],
    "rejected_candidates": []
  },
  "per_question": {}
}
```

The exact candidate and per-question information depends on the evaluation mode.

---

# Manifest

Every batch run produces:

```text
data/output/manifest.json
```

The manifest records run-level processing information.

Example:

```json
{
  "schema_version": "1.0",
  "documents_processed": 11,
  "documents_failed": 0,
  "documents": [],
  "failures": []
}
```

The manifest provides a concise summary of:

* documents processed
* documents that failed
* generated artifact locations
* failure information

---

# CLI

Vanka exposes two commands:

```text
vanka process
vanka chunk
```

View the CLI:

```bash
vanka --help
```

---

# CLI: Process

## Without benchmarks

```bash
vanka process \
  --documents data/normalized \
  --output data/output
```

## With benchmarks

```bash
vanka process \
  --documents data/normalized \
  --benchmarks data/benchmarks \
  --output data/output
```

The benchmark directory is optional.

---

# CLI: Single Document

Default:

```bash
vanka chunk \
  data/normalized/investment/advisory/advisory-mandates.jsonl
```

The default strategy is:

```text
auto
```

Explicit strategy:

```bash
vanka chunk \
  data/normalized/investment/advisory/advisory-mandates.jsonl \
  --strategy fixed
```

Supported strategies:

```text
fixed
recursive
semantic
structural
auto
```

---

# CLI Output

Example:

```json
{
  "strategy": "fixed",
  "chunks": 6,
  "selection_score": 0.742,
  "selection_reason": "No benchmark retrieval metrics were available; selected the highest-scoring strategy using intrinsic chunk-quality metrics.",
  "metrics": {
    "strategy": "fixed",
    "chunks": 6,
    "mean_chars": 721.7,
    "median_chars": 684.0,
    "under_100_chars": 0,
    "recall_at_1": null,
    "recall_at_3": null,
    "recall_at_5": null,
    "mrr": null
  },
  "document": "data\\normalized\\investment\\advisory\\advisory-mandates.jsonl"
}
```

---

# Complete Example: No Benchmarks

Run:

```bash
vanka process \
  --documents data/normalized \
  --output data/output
```

Pipeline:

```text
Normalized documents
        ↓
Candidate chunkers
        ↓
Intrinsic metrics
        ↓
Quality gates
        ↓
Automatic selection
        ↓
Chunks + reports + manifest
```

Expected report characteristics for documents without benchmarks:

```json
{
  "question_count": 0
}
```

and:

```json
{
  "recall_at_1": null,
  "recall_at_3": null,
  "recall_at_5": null,
  "mrr": null
}
```

---

# Complete Example: With Benchmarks

Run:

```bash
vanka process \
  --documents data/normalized \
  --benchmarks data/benchmarks \
  --output data/output
```

Pipeline:

```text
Normalized documents
        ↓
Benchmark discovery
        ↓
Candidate chunkers
        ↓
Retrieval evaluation
        ↓
Quality gates
        ↓
Automatic selection
        ↓
Chunks + reports + manifest
```

For benchmarked documents, reports contain actual retrieval metrics.

---

# Complete Example: Explicit Strategy

If you already know which strategy you want:

```bash
vanka chunk \
  data/normalized/investment/advisory/advisory-mandates.jsonl \
  --strategy semantic
```

This bypasses automatic strategy selection and directly produces semantic chunks.

Supported explicit strategies:

```text
fixed
recursive
semantic
structural
```

---

# Complete Example: Automatic Strategy

```bash
vanka chunk \
  data/normalized/investment/advisory/advisory-mandates.jsonl \
  --strategy auto
```

This runs the automatic chunking/selection path available to the single-document API.

For benchmark-aware batch selection, use:

```bash
vanka process \
  --documents data/normalized \
  --benchmarks data/benchmarks \
  --output data/output
```

---

# Testing

Run the complete test suite:

```bash
pytest -q
```

The test suite currently covers:

```text
Chunking
Selector behavior
Public API
Artifact generation
CLI parsing
```

Current test status:

```text
24 passed
```

---

# Development

Install development dependencies:

```bash
python -m pip install -e ".[semantic,test]"
```

Run tests:

```bash
pytest -q
```

Check the CLI:

```bash
vanka --help
```

Check process help:

```bash
vanka process --help
```

Check chunk help:

```bash
vanka chunk --help
```

Run a benchmark-aware smoke test:

```bash
vanka process \
  --documents data/normalized \
  --benchmarks data/benchmarks \
  --output data/output
```

---

# Integration with a RAG System

Vanka can act as the document-processing layer of a larger RAG architecture.

```text
                    DOCUMENTS
                        │
                        ▼
                 ┌─────────────┐
                 │    VANKA    │
                 │             │
                 │ Chunking    │
                 │ Evaluation  │
                 │ Selection   │
                 │ Provenance  │
                 └──────┬──────┘
                        │
                        ▼
                 Selected Chunks
                        │
                        ▼
                Embedding Pipeline
                        │
                        ▼
                   Vector DB
                        │
                        ▼
                    Retrieval
                        │
                        ▼
                    Reranking
                        │
                        ▼
                 Context Assembly
                        │
                        ▼
                     LLM/Agent
                        │
                        ▼
                     Answer
```

Vanka therefore acts as the **document intelligence and chunk-selection layer** before retrieval.

---

# Design Principles

## Evaluation-driven

Chunking is treated as an empirical problem.

Instead of assuming:

```text
one chunker works for everything
```

Vanka evaluates available strategies.

---

## Benchmark-aware

When benchmark questions exist:

```text
Benchmark
    ↓
Retrieval evaluation
    ↓
Selection
```

---

## Graceful fallback

When benchmark questions do not exist:

```text
No benchmark
     ↓
Intrinsic metrics
     ↓
Selection
```

Missing retrieval metrics are represented as `null`, rather than zero.

---

## Provenance-first

Every selected chunk remains traceable to its source document.

```text
Chunk
 ↓
Document
 ↓
Source file
 ↓
Page
 ↓
Section
 ↓
Character offsets
```

---

## Auditable Decisions

Vanka records:

```text
Selected strategy
Selection score
Selection reason
Candidate strategies
Candidate metrics
Rejected candidates
Rejection reasons
```

This allows downstream systems to understand the strategy-selection decision.

---

## Separation of Concerns

Vanka focuses on:

```text
Ingestion
Normalization consumption
Structure detection
Chunking
Evaluation
Selection
Provenance
Artifact generation
```

Downstream systems can handle:

```text
Retrieval
Reranking
Context assembly
LLM generation
Agent orchestration
Answer citations
```

---

# Current Scope

Vanka currently provides:

* Fixed chunking
* Recursive chunking
* Semantic chunking
* Structural chunking
* Automatic strategy selection
* Benchmark-aware retrieval evaluation
* Intrinsic evaluation fallback
* Recall@1
* Recall@3
* Recall@5
* MRR
* Chunk-quality gates
* Chunk provenance
* Selection audit
* JSON artifacts
* Manifest generation
* Python API
* CLI
* Automated tests

Vanka does not currently provide:

* Vector database management
* Production retrieval
* Reranking
* LLM answer generation
* Agent orchestration
* Query rewriting
* Hybrid retrieval
* Knowledge graph retrieval

These are intentionally outside the current core scope.

---

# Future Extensions

Potential future extensions include:

* Additional chunking strategies
* Additional intrinsic metrics
* Additional retrieval metrics
* More sophisticated benchmark construction
* Configurable selector weighting
* Pluggable embedding models
* Additional document formats
* Distributed evaluation
* Persistent benchmark tracking
* Experiment/version tracking
* Visualization of chunking and evaluation results

---

# Status

Vanka currently provides a working evaluation-driven chunking pipeline with:

```text
✓ Fixed chunking
✓ Recursive chunking
✓ Semantic chunking
✓ Structural chunking
✓ Automatic strategy selection
✓ Benchmark-aware evaluation
✓ No-benchmark fallback
✓ Recall@1
✓ Recall@3
✓ Recall@5
✓ MRR
✓ Chunk-quality gates
✓ Provenance
✓ Selection audit
✓ JSON artifacts
✓ Manifest generation
✓ Python API
✓ CLI
✓ Automated tests
```

Current validation:

```text
24 tests passed
11 documents processed
0 documents failed
```

---

# License

Add the project's license information here.

---

# Author

Add author/project information here.

```

**One thing I deliberately did:** I kept this README aligned with what you've actually implemented and tested, rather than claiming features like hybrid retrieval, reranking, knowledge graphs, or production vector DB integration that aren't part of Vanka yet.
```
