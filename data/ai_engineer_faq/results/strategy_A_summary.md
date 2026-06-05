# Strategy A - Fixed-size Chunking + Vector Search

## Mục tiêu

Đánh giá retrieval strategy dùng fixed-size chunking kết hợp vector search trên cùng corpus AI Engineer FAQ của nhóm.

## Cấu hình

- Corpus: `data/ai_engineer_faq/docs_txt/`
- Benchmark: `data/ai_engineer_faq/benchmark/benchmark_queries.jsonl`
- Chunking: `FixedSizeChunker`
- Chunk size: `700` ký tự
- Overlap: `100` ký tự
- Retrieval: vector search, lấy `top_k=3`
- Embedding backend khi chạy: `token-hash-vector-512d`
- Tổng số chunk index: `116`

## Kết quả

| Metric | Value |
|---|---:|
| Top-1 accuracy | 1.00 |
| Top-3 recall | 1.00 |

## Nhận xét

Strategy này đơn giản, dễ triển khai và chạy ổn với corpus tài liệu ngắn 5-10 docs. Fixed-size chunking giúp số lượng chunk đều, dễ kiểm soát bằng `chunk_size` và `overlap`. Với benchmark hiện tại, các query đều có keyword quan trọng trùng với tài liệu như `Embedding`, `File Search`, `Structured Outputs`, `Rate limit`, `Batch API`, nên vector search tìm đúng tài liệu liên quan trong Top-1 hoặc Top-3.

Điểm yếu là fixed-size chunking có thể cắt ngang câu hoặc cắt ngang section, làm chunk đôi khi thiếu ngữ cảnh. Strategy này phù hợp làm baseline để so sánh với sentence chunking, recursive chunking, metadata filtering và hybrid search.

## Cách chạy lại

```bash
python3 scripts/run_strategy_a_fixed_vector.py
```

Kết quả chi tiết nằm ở:

- `data/ai_engineer_faq/results/strategy_A_fixed_vector_results.md`
- `data/ai_engineer_faq/results/strategy_A_fixed_vector_results.json`
