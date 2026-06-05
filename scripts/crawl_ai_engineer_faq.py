#!/usr/bin/env python3
"""Download a small AI Engineer FAQ corpus as Markdown/TXT files.

The corpus is intentionally small enough for a 5-person retrieval benchmark:
8 official OpenAI docs pages plus 5 benchmark queries with gold answers.
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_OUTPUT_DIR = Path("data/ai_engineer_faq")
USER_AGENT = "Mozilla/5.0 (compatible; ai-engineer-faq-crawler/1.0)"


@dataclass(frozen=True)
class SourceDoc:
    doc_id: str
    title: str
    url: str
    filename: str


SOURCES = [
    SourceDoc(
        "D1",
        "Prompt Engineering",
        "https://developers.openai.com/api/docs/guides/prompt-engineering.md",
        "D1_prompt_engineering",
    ),
    SourceDoc(
        "D2",
        "Embeddings",
        "https://developers.openai.com/api/docs/guides/embeddings.md",
        "D2_embeddings",
    ),
    SourceDoc(
        "D3",
        "File Search",
        "https://developers.openai.com/api/docs/guides/tools-file-search.md",
        "D3_file_search",
    ),
    SourceDoc(
        "D4",
        "Structured Outputs",
        "https://developers.openai.com/api/docs/guides/structured-outputs.md",
        "D4_structured_outputs",
    ),
    SourceDoc(
        "D5",
        "Rate Limits",
        "https://developers.openai.com/api/docs/guides/rate-limits.md",
        "D5_rate_limits",
    ),
    SourceDoc(
        "D6",
        "Batch API",
        "https://developers.openai.com/api/docs/guides/batch.md",
        "D6_batch_api",
    ),
    SourceDoc(
        "D7",
        "Safety Best Practices",
        "https://developers.openai.com/api/docs/guides/safety-best-practices.md",
        "D7_safety_best_practices",
    ),
    SourceDoc(
        "D8",
        "Agent Evals",
        "https://developers.openai.com/api/docs/guides/agent-evals.md",
        "D8_agent_evals",
    ),
]


BENCHMARKS = [
    {
        "query_id": "Q1",
        "query": "Embedding dùng để làm gì trong hệ thống AI/RAG?",
        "gold_answer": (
            "Embedding biến văn bản thành vector số để đo mức độ liên quan hoặc "
            "tương đồng ngữ nghĩa giữa các đoạn text. Trong RAG, embedding thường "
            "được dùng để tìm các chunk tài liệu liên quan nhất với câu hỏi trước "
            "khi đưa context đó cho model trả lời."
        ),
        "relevant_doc_ids": ["D2"],
    },
    {
        "query_id": "Q2",
        "query": "File Search trong OpenAI API hoạt động như thế nào?",
        "gold_answer": (
            "File Search cho phép model tìm thông tin trong các file đã upload "
            "thông qua vector store. Quy trình cơ bản là tạo vector store, upload "
            "file vào đó, rồi bật tool file_search trong request để model truy xuất "
            "các đoạn liên quan khi trả lời."
        ),
        "relevant_doc_ids": ["D3"],
    },
    {
        "query_id": "Q3",
        "query": "Khi nào nên dùng Structured Outputs?",
        "gold_answer": (
            "Nên dùng Structured Outputs khi ứng dụng cần model trả về dữ liệu theo "
            "schema cố định, ví dụ JSON có field bắt buộc. Cách này phù hợp khi "
            "output cần được parse bằng code, lưu database, gọi API khác, hoặc giảm "
            "lỗi format so với câu trả lời text tự do."
        ),
        "relevant_doc_ids": ["D4"],
    },
    {
        "query_id": "Q4",
        "query": "Rate limit của OpenAI API được tính theo những đơn vị nào?",
        "gold_answer": (
            "Rate limit có thể được tính theo request/phút, request/ngày, token/phút, "
            "token/ngày, và một số loại tài nguyên khác tùy endpoint như image/phút. "
            "Giới hạn cụ thể phụ thuộc vào model, project/organization và usage tier."
        ),
        "relevant_doc_ids": ["D5"],
    },
    {
        "query_id": "Q5",
        "query": "Batch API phù hợp với trường hợp nào?",
        "gold_answer": (
            "Batch API phù hợp với tác vụ không cần phản hồi ngay, ví dụ chạy eval, "
            "phân loại dataset lớn, hoặc xử lý nhiều request/embedding cho kho tài "
            "liệu. Nó xử lý bất đồng bộ, có rate limit riêng và thường phù hợp cho "
            "job nền hơn là tương tác realtime."
        ),
        "relevant_doc_ids": ["D6"],
    },
]


def fetch_text(url: str) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30) as response:
        encoding = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(encoding, errors="replace")


def clean_markdown(markdown: str) -> str:
    """Lightly remove MDX noise while preserving the document's retrieval value."""
    markdown = re.sub(r"\nimport\s+\{.*?\}\s+from\s+['\"].*?['\"];?\n", "\n", markdown, flags=re.S)
    markdown = re.sub(r"\nimport\s+.*?from\s+['\"].*?['\"];?\n", "\n", markdown)
    markdown = re.sub(r"\nexport\s+const\s+\w+\s*=\s*\{.*?\n\};?\n", "\n", markdown, flags=re.S)
    markdown = re.sub(r"\n<(/?)(Accordion|Cards|Card|Note|Warning|Info|CodeGroup|Tabs|Tab)[^>]*>\n", "\n", markdown)
    markdown = re.sub(r"\n\s*\{/\*.*?\*/\}\s*\n", "\n", markdown, flags=re.S)
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    return markdown.strip() + "\n"


def markdown_to_text(markdown: str) -> str:
    text = re.sub(r"```.*?```", " ", markdown, flags=re.S)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"^\s{0,3}[-*+]\s+", "- ", text, flags=re.M)
    text = re.sub(r"^\s{0,3}>\s?", "", text, flags=re.M)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def with_metadata(doc: SourceDoc, body: str) -> str:
    return textwrap.dedent(
        f"""\
        ---
        doc_id: {doc.doc_id}
        title: {doc.title}
        source_url: {doc.url}
        ---

        """
    ) + body


def write_sources(output_dir: Path, sources: Iterable[SourceDoc]) -> None:
    rows = [
        {
            "doc_id": source.doc_id,
            "title": source.title,
            "url": source.url,
            "markdown_file": f"docs_md/{source.filename}.md",
            "text_file": f"docs_txt/{source.filename}.txt",
        }
        for source in sources
    ]
    (output_dir / "sources.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_benchmark_files(output_dir: Path) -> None:
    benchmark_dir = output_dir / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    jsonl = "\n".join(json.dumps(item, ensure_ascii=False) for item in BENCHMARKS) + "\n"
    (benchmark_dir / "benchmark_queries.jsonl").write_text(jsonl, encoding="utf-8")

    lines = [
        "# AI Engineer FAQ Benchmark",
        "",
        "Dùng 5 query này cho mọi retrieval strategy trong nhóm.",
        "",
        "| Query ID | Query | Gold answer | Relevant docs |",
        "|---|---|---|---|",
    ]
    for item in BENCHMARKS:
        docs = ", ".join(item["relevant_doc_ids"])
        lines.append(
            f"| {item['query_id']} | {item['query']} | {item['gold_answer']} | {docs} |"
        )
    lines.append("")
    (benchmark_dir / "benchmark_queries.md").write_text("\n".join(lines), encoding="utf-8")


def write_readme(output_dir: Path) -> None:
    content = """# AI Engineer FAQ Corpus

Corpus này gồm 8 tài liệu OpenAI docs dạng `.md` và `.txt`, phù hợp cho bài nhóm về retrieval strategy.

## Cấu trúc

- `docs_md/`: tài liệu Markdown đã crawl.
- `docs_txt/`: bản text trơn của cùng tài liệu.
- `benchmark/benchmark_queries.md`: 5 query + gold answer để đọc nhanh.
- `benchmark/benchmark_queries.jsonl`: cùng benchmark ở dạng machine-readable.
- `sources.json`: mapping `doc_id`, tiêu đề, URL nguồn và tên file.

## Cách chạy lại crawler

```bash
python3 scripts/crawl_ai_engineer_faq.py
```

Mỗi thành viên trong nhóm nên dùng cùng corpus và cùng 5 query này, sau đó so sánh strategy riêng.
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def crawl(output_dir: Path) -> None:
    docs_md = output_dir / "docs_md"
    docs_txt = output_dir / "docs_txt"
    docs_md.mkdir(parents=True, exist_ok=True)
    docs_txt.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, str]] = []
    for source in SOURCES:
        try:
            raw = fetch_text(source.url)
        except (HTTPError, URLError, TimeoutError) as exc:
            failures.append((source.doc_id, str(exc)))
            continue

        markdown = with_metadata(source, clean_markdown(raw))
        text = with_metadata(source, markdown_to_text(markdown))
        (docs_md / f"{source.filename}.md").write_text(markdown, encoding="utf-8")
        (docs_txt / f"{source.filename}.txt").write_text(text, encoding="utf-8")
        print(f"Downloaded {source.doc_id}: {source.title}")

    write_sources(output_dir, SOURCES)
    write_benchmark_files(output_dir)
    write_readme(output_dir)

    if failures:
        print("\nSome downloads failed:")
        for doc_id, error in failures:
            print(f"- {doc_id}: {error}")
        raise SystemExit(1)

    print(f"\nDone. Corpus written to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    crawl(args.output_dir)


if __name__ == "__main__":
    main()
