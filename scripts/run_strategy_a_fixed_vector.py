#!/usr/bin/env python3
"""Run Strategy A: Fixed-size chunking + vector search.

This script uses the AI Engineer FAQ corpus in data/ai_engineer_faq and writes
benchmark results for the 5 agreed group queries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking import FixedSizeChunker
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
)
from src.models import Document
from src.store import EmbeddingStore


DEFAULT_CORPUS_DIR = Path("data/ai_engineer_faq")
DEFAULT_OUTPUT_DIR = DEFAULT_CORPUS_DIR / "results"


class TokenHashEmbedder:
    """Dependency-free normalized token-vector embedder for classroom retrieval."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self._backend_name = f"token-hash-vector-{dim}d"

    def __call__(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]


def choose_embedder(provider: str | None = None) -> Callable[[str], list[float]]:
    """Pick an embedding backend.

    Provider order:
    - explicit --provider value
    - EMBEDDING_PROVIDER from .env/environment
    - token-hash fallback, which needs no external dependencies
    """
    load_dotenv(override=False)
    selected = (provider or os.getenv(EMBEDDING_PROVIDER_ENV, "hash")).strip().lower()

    if selected == "local":
        try:
            return LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception as exc:
            print(f"Local embedder unavailable, falling back to token-hash: {exc}")

    if selected == "openai":
        try:
            return OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception as exc:
            print(f"OpenAI embedder unavailable, falling back to token-hash: {exc}")

    return TokenHashEmbedder()


def load_sources(corpus_dir: Path) -> list[dict]:
    return json.loads((corpus_dir / "sources.json").read_text(encoding="utf-8"))


def load_benchmarks(corpus_dir: Path) -> list[dict]:
    benchmark_path = corpus_dir / "benchmark" / "benchmark_queries.jsonl"
    return [
        json.loads(line)
        for line in benchmark_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_chunk_documents(corpus_dir: Path, chunk_size: int, overlap: int) -> list[Document]:
    chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
    chunk_docs: list[Document] = []

    for source in load_sources(corpus_dir):
        text_path = corpus_dir / source["text_file"]
        text = text_path.read_text(encoding="utf-8")
        chunks = chunker.chunk(text)
        for index, chunk in enumerate(chunks, start=1):
            chunk_docs.append(
                Document(
                    id=f"{source['doc_id']}_chunk_{index:03d}",
                    content=chunk,
                    metadata={
                        "source_doc_id": source["doc_id"],
                        "title": source["title"],
                        "source_url": source["url"],
                        "source_file": str(text_path),
                        "chunk_index": index,
                        "chunk_size": chunk_size,
                        "overlap": overlap,
                        "strategy": "A_fixed_size_vector_search",
                    },
                )
            )

    return chunk_docs


def preview(text: str, limit: int = 180) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def run_strategy(
    corpus_dir: Path,
    output_dir: Path,
    chunk_size: int,
    overlap: int,
    top_k: int,
    provider: str | None,
) -> dict:
    embedder = choose_embedder(provider=provider)
    chunk_docs = build_chunk_documents(corpus_dir, chunk_size=chunk_size, overlap=overlap)

    store = EmbeddingStore(
        collection_name="strategy_a_fixed_vector",
        embedding_fn=embedder,
    )
    store.add_documents(chunk_docs)

    results = []
    top1_hits = 0
    topk_hits = 0

    for benchmark in load_benchmarks(corpus_dir):
        retrieved = store.search(benchmark["query"], top_k=top_k)
        relevant = set(benchmark["relevant_doc_ids"])
        retrieved_doc_ids = [item["metadata"].get("source_doc_id") for item in retrieved]
        top1_hit = bool(retrieved_doc_ids and retrieved_doc_ids[0] in relevant)
        topk_hit = any(doc_id in relevant for doc_id in retrieved_doc_ids)
        top1_hits += int(top1_hit)
        topk_hits += int(topk_hit)

        results.append(
            {
                "query_id": benchmark["query_id"],
                "query": benchmark["query"],
                "gold_answer": benchmark["gold_answer"],
                "relevant_doc_ids": benchmark["relevant_doc_ids"],
                "top1_hit": top1_hit,
                "topk_hit": topk_hit,
                "retrieved": [
                    {
                        "rank": rank,
                        "score": item["score"],
                        "chunk_id": item["id"],
                        "source_doc_id": item["metadata"].get("source_doc_id"),
                        "title": item["metadata"].get("title"),
                        "chunk_index": item["metadata"].get("chunk_index"),
                        "preview": preview(item["content"]),
                    }
                    for rank, item in enumerate(retrieved, start=1)
                ],
            }
        )

    summary = {
        "strategy": "A_fixed_size_chunking_vector_search",
        "chunk_size": chunk_size,
        "overlap": overlap,
        "top_k": top_k,
        "embedding_backend": getattr(embedder, "_backend_name", embedder.__class__.__name__),
        "num_chunks": store.get_collection_size(),
        "num_queries": len(results),
        "top1_accuracy": top1_hits / len(results) if results else 0.0,
        "topk_recall": topk_hits / len(results) if results else 0.0,
        "results": results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "strategy_A_fixed_vector_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "strategy_A_fixed_vector_results.md").write_text(
        render_markdown(summary),
        encoding="utf-8",
    )
    return summary


def escape_table_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(summary: dict) -> str:
    lines = [
        "# Strategy A - Fixed-size Chunking + Vector Search",
        "",
        "## Setup",
        "",
        f"- Chunk size: `{summary['chunk_size']}` characters",
        f"- Overlap: `{summary['overlap']}` characters",
        f"- Top-k: `{summary['top_k']}`",
        f"- Embedding backend: `{summary['embedding_backend']}`",
        f"- Total chunks indexed: `{summary['num_chunks']}`",
        "",
        "## Metrics",
        "",
        f"- Top-1 accuracy: `{summary['top1_accuracy']:.2f}`",
        f"- Top-{summary['top_k']} recall: `{summary['topk_recall']:.2f}`",
        "",
        "## Query Results",
        "",
    ]

    for result in summary["results"]:
        lines.extend(
            [
                f"### {result['query_id']}. {result['query']}",
                "",
                f"Gold answer: {result['gold_answer']}",
                "",
                f"Relevant docs: `{', '.join(result['relevant_doc_ids'])}`",
                "",
                f"Top-1 hit: `{result['top1_hit']}`; Top-k hit: `{result['topk_hit']}`",
                "",
                "| Rank | Score | Doc | Title | Chunk | Preview |",
                "|---|---:|---|---|---:|---|",
            ]
        )
        for item in result["retrieved"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        escape_table_cell(item["rank"]),
                        f"{item['score']:.4f}",
                        escape_table_cell(item["source_doc_id"]),
                        escape_table_cell(item["title"]),
                        escape_table_cell(item["chunk_index"]),
                        escape_table_cell(item["preview"]),
                    ]
                )
                + " |"
            )
        lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--provider",
        choices=["hash", "local", "openai"],
        default=None,
        help="Embedding backend. Defaults to EMBEDDING_PROVIDER or hash.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_strategy(
        corpus_dir=args.corpus_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        top_k=args.top_k,
        provider=args.provider,
    )
    print("Strategy A complete")
    print(f"Embedding backend: {summary['embedding_backend']}")
    print(f"Indexed chunks: {summary['num_chunks']}")
    print(f"Top-1 accuracy: {summary['top1_accuracy']:.2f}")
    print(f"Top-{summary['top_k']} recall: {summary['topk_recall']:.2f}")
    print(f"Results: {args.output_dir / 'strategy_A_fixed_vector_results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
