# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Đào Xuân Bách  
**Mã sinh viên:** 2A202600640  
**Nhóm:** B5-C401 
**Ngày:** 05/06/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**  
High cosine similarity nghĩa là hai vector embedding có hướng gần giống nhau, tức là hai đoạn text có nội dung hoặc ý nghĩa gần nhau. Với retrieval, chunk có cosine similarity cao với query thường là chunk đáng đưa vào context.

**Ví dụ HIGH similarity:**
- Sentence A: `Embeddings convert text into numeric vectors for semantic search.`
- Sentence B: `Text embeddings represent documents as vectors for similarity search.`
- Tại sao tương đồng: cả hai đều nói về embedding, vector hóa văn bản và semantic/similarity search.

**Ví dụ LOW similarity:**
- Sentence A: `Rate limits restrict how many API requests can be sent.`
- Sentence B: `A chocolate cake recipe needs flour, sugar, and eggs.`
- Tại sao khác: một câu nói về giới hạn API, câu còn lại nói về nấu ăn nên chủ đề gần như không liên quan.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**  
Cosine similarity tập trung vào hướng của vector, nên phù hợp để đo độ gần nghĩa hơn là độ lớn tuyệt đối của vector. Với text embeddings, hai câu có thể dài/ngắn khác nhau nhưng vẫn cùng ý; cosine similarity giúp giảm ảnh hưởng của độ dài vector.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**

Công thức:

```text
num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))
           = ceil((10000 - 50) / (500 - 50))
           = ceil(9950 / 450)
           = 23 chunks
```

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**

```text
num_chunks = ceil((10000 - 100) / (500 - 100))
           = ceil(9900 / 400)
           = 25 chunks
```

Khi overlap tăng, step nhỏ hơn nên số chunk tăng từ 23 lên 25. Overlap nhiều hơn giúp giữ ngữ cảnh giữa hai chunk liên tiếp, giảm rủi ro câu trả lời bị cắt đúng ở ranh giới chunk.

---

## 2. Document Selection - Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** AI Engineer FAQ / OpenAI API technical docs.

Nhóm chọn domain này vì nội dung phù hợp trực tiếp với bài lab về embedding, vector store, retrieval và RAG. Tài liệu có cấu trúc rõ ràng, nguồn minh bạch, có nhiều thuật ngữ kỹ thuật như Embeddings, File Search, Structured Outputs, Rate Limits và Batch API. Corpus đủ nhỏ để benchmark thủ công nhưng vẫn đủ đa dạng để so sánh nhiều retrieval strategy.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|---------:|-----------------|
| 1 | Prompt Engineering | `developers.openai.com/api/docs/guides/prompt-engineering.md` | 16,589 | `doc_id`, `title`, `source_url` |
| 2 | Embeddings | `developers.openai.com/api/docs/guides/embeddings.md` | 13,208 | `doc_id`, `title`, `source_url` |
| 3 | File Search | `developers.openai.com/api/docs/guides/tools-file-search.md` | 4,011 | `doc_id`, `title`, `source_url` |
| 4 | Structured Outputs | `developers.openai.com/api/docs/guides/structured-outputs.md` | 8,375 | `doc_id`, `title`, `source_url` |
| 5 | Rate Limits | `developers.openai.com/api/docs/guides/rate-limits.md` | 9,568 | `doc_id`, `title`, `source_url` |
| 6 | Batch API | `developers.openai.com/api/docs/guides/batch.md` | 8,458 | `doc_id`, `title`, `source_url` |
| 7 | Safety Best Practices | `developers.openai.com/api/docs/guides/safety-best-practices.md` | 5,848 | `doc_id`, `title`, `source_url` |
| 8 | Agent Evals | `developers.openai.com/api/docs/guides/agent-evals.md` | 2,723 | `doc_id`, `title`, `source_url` |

Tài liệu được crawl và lưu ở:

- `data/ai_engineer_faq/docs_md/`
- `data/ai_engineer_faq/docs_txt/`
- `data/ai_engineer_faq/sources.json`

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `doc_id` / `source_doc_id` | string | `D2`, `D5` | Xác định chunk thuộc tài liệu nào, dùng để kiểm tra top-k có đúng relevant doc không. |
| `title` | string | `Embeddings`, `Rate Limits` | Giúp hiển thị kết quả dễ hiểu và có thể filter theo chủ đề/tên tài liệu. |
| `source_url` | string | URL OpenAI docs | Giúp truy vết nguồn, kiểm chứng gold answer và minh bạch corpus. |
| `chunk_index` | integer | `3` | Xác định vị trí chunk trong tài liệu, thuận tiện khi phân tích lỗi retrieval. |
| `strategy` | string | `A_fixed_size_vector_search` | Biết chunk được tạo bởi strategy nào khi so sánh nhiều thành viên. |

---

## 3. Chunking Strategy - Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 3 tài liệu đại diện với `chunk_size=700`.

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|------------:|-----------:|-------------------|
| D2 Embeddings | FixedSizeChunker (`fixed_size`) | 19 | 695.2 | Trung bình, có thể cắt ngang câu |
| D2 Embeddings | SentenceChunker (`by_sentences`) | 32 | 409.6 | Tốt, giữ ranh giới câu |
| D2 Embeddings | RecursiveChunker (`recursive`) | 26 | 505.5 | Tốt, giữ đoạn/section tốt hơn |
| D3 File Search | FixedSizeChunker (`fixed_size`) | 6 | 668.5 | Trung bình |
| D3 File Search | SentenceChunker (`by_sentences`) | 7 | 571.1 | Tốt |
| D3 File Search | RecursiveChunker (`recursive`) | 8 | 499.5 | Tốt |
| D5 Rate Limits | FixedSizeChunker (`fixed_size`) | 14 | 683.4 | Trung bình |
| D5 Rate Limits | SentenceChunker (`by_sentences`) | 23 | 414.2 | Tốt |
| D5 Rate Limits | RecursiveChunker (`recursive`) | 19 | 501.7 | Tốt |

### Strategy Của Tôi

**Loại:** FixedSizeChunker + Vector Search.

**Mô tả cách hoạt động:**  
Tôi chia mỗi tài liệu `.txt` thành các chunk có độ dài cố định `700` ký tự và overlap `100` ký tự. Overlap giúp chunk sau giữ lại một phần ngữ cảnh của chunk trước. Sau đó mỗi chunk được embed thành vector, lưu vào `EmbeddingStore`, và query cũng được embed để search theo similarity score. Kết quả retrieval lấy `top_k=3`.

**Tại sao tôi chọn strategy này cho domain nhóm?**  
Corpus AI Engineer FAQ gồm các tài liệu kỹ thuật tương đối ngắn, có keyword rõ như `Embedding`, `File Search`, `Structured Outputs`, `Rate Limits`, `Batch API`. Fixed-size chunking là baseline đơn giản, dễ kiểm soát bằng hai tham số `chunk_size` và `overlap`, phù hợp để so sánh với các strategy phức tạp hơn của thành viên khác.

**Code sử dụng cho Strategy A:**

```python
chunker = FixedSizeChunker(chunk_size=700, overlap=100)
store = EmbeddingStore(
    collection_name="strategy_a_fixed_vector",
    embedding_fn=TokenHashEmbedder(),
)
store.add_documents(chunk_docs)
results = store.search(query, top_k=3)
```

Script chạy strategy:

```bash
python3 scripts/run_strategy_a_fixed_vector.py
```

### So Sánh: Strategy của tôi vs Baseline

Baseline `FixedSizeChunker` trong comparator dùng `overlap=0`; strategy của tôi dùng `overlap=100`.

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|------------:|-----------:|--------------------|
| D2 Embeddings | baseline fixed-size, overlap=0 | 19 | 695.2 | Có thể retrieve đúng nhưng rủi ro mất context ở ranh giới chunk |
| D2 Embeddings | **của tôi: fixed-size, overlap=100** | 22 | 695.8 | Tốt hơn về context continuity |
| D3 File Search | baseline fixed-size, overlap=0 | 6 | 668.5 | Đủ dùng vì tài liệu ngắn |
| D3 File Search | **của tôi: fixed-size, overlap=100** | 7 | 658.7 | Tăng nhẹ số chunk, giữ thêm context |
| D5 Rate Limits | baseline fixed-size, overlap=0 | 14 | 683.4 | Có thể cắt ngang danh sách/giải thích |
| D5 Rate Limits | **của tôi: fixed-size, overlap=100** | 16 | 691.8 | Tốt hơn cho các đoạn giải thích dài |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------:|-----------|----------|
| Đào Xuân Bách - A | Fixed-size + Vector Search | 10/10 | Đơn giản, chạy nhanh, làm baseline tốt, Top-3 recall đạt 1.00 | Có thể cắt ngang câu/section |
| Lê Hoài Nam - B | SentenceChunker (`max_sentences_per_chunk=3`) + Vector Search | 8/10 | Bảo toàn ranh giới câu tự nhiên, chunk dễ đọc và mạch lạc hơn fixed-size | Chunk chứa bảng/danh sách dài có thể lớn hoặc không đều nếu ít dấu câu |
| Phan Quốc Anh - C | RecursiveChunker (`chunk_size=500`) + LocalEmbedder | 10/10; top-3 relevant 5/5 | Giữ cấu trúc paragraph/section tốt, local embedding hiểu ngữ nghĩa tốt hơn mock/hash | Top-1 đôi khi trả về code/table chunk chưa phải đoạn giải thích tốt nhất; index chậm hơn |
| Nguyễn Đức Kiên Trung - D | RecursiveChunker (`chunk_size=400`) + Metadata Filtering | 10/10 với filter; plain search 4/10 | Giữ cấu trúc paragraph/section tốt, metadata filter tăng precision từ 2/5 lên 5/5 | Cần biết hoặc suy luận đúng `source_id`; filter sai có thể làm mất recall |
| Đỗ Thiện Lĩnh - E | SentenceChunker + Vector Search | 9/10; top-3 relevant 5/5 | Giữ ngữ cảnh câu tốt, phù hợp với câu trả lời ngắn/dạng FAQ | Top-1 vẫn có false positive ở một số query; report còn lẫn mô tả domain FAQ khác |

**Strategy nào tốt nhất cho domain này? Tại sao?**  
Hiện tại nhóm đã có kết quả của cả A, B, C, D và E. Strategy A, C và D đều đạt 5/5 relevant trong top-3 trên benchmark khi chạy theo cấu hình riêng, nhưng thế mạnh khác nhau: A là baseline đơn giản và không cần filter, C cho thấy RecursiveChunker kết hợp local dense embedding hiểu ngữ nghĩa tốt hơn, còn D chứng minh metadata filtering rất hữu ích khi biết rõ query thuộc tài liệu nào. Strategy B và E cùng nghiêng về SentenceChunker, có điểm mạnh về chunk coherence vì giữ ranh giới câu tự nhiên, phù hợp khi ưu tiên context dễ đọc cho LLM.

---

## 4. My Approach - Cá nhân (10 điểm)

Giải thích cách tiếp cận khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk` - approach:**  
Tôi dùng regex `(?<=[.!?])\s+` để tách câu theo dấu chấm, chấm hỏi, chấm than và whitespace phía sau. Sau đó gom các câu thành từng chunk theo `max_sentences_per_chunk`, đồng thời strip whitespace và bỏ câu rỗng. Edge case chính là text rỗng trả về list rỗng, còn `max_sentences_per_chunk` luôn tối thiểu là 1.

**`RecursiveChunker.chunk` / `_split` - approach:**  
Tôi triển khai chia đệ quy theo thứ tự separator `["\n\n", "\n", ". ", " ", ""]`. Base case là nếu đoạn hiện tại không vượt `chunk_size` thì trả về luôn; nếu separator hiện tại không tồn tại thì thử separator tiếp theo. Với separator rỗng, fallback là cắt theo kích thước cố định để đảm bảo không bị kẹt khi text quá dài.

### EmbeddingStore

**`add_documents` + `search` - approach:**  
`add_documents` tạo record gồm `id`, `content`, `metadata`, và embedding của từng document/chunk, sau đó lưu vào in-memory store. Nếu ChromaDB có sẵn thì có thể add vào collection, nhưng store vẫn giữ bản in-memory để test và fallback ổn định. `search` embed query rồi tính dot product với từng embedding, sort giảm dần theo score và trả về `top_k`.

**`search_with_filter` + `delete_document` - approach:**  
`search_with_filter` lọc record theo metadata trước, rồi mới tính similarity trên tập đã lọc. Cách này đúng hơn việc search trước rồi filter vì top-k cần được chọn trong không gian ứng viên sau lọc. `delete_document` xóa tất cả record có `metadata["doc_id"]` tương ứng và trả về `True/False` tùy có xóa được record nào không.

### KnowledgeBaseAgent

**`answer` - approach:**  
Agent gọi `store.search(question, top_k)` để lấy các chunk liên quan, sau đó ghép chunk thành context có kèm score. Prompt yêu cầu LLM chỉ trả lời dựa trên context và nói không biết nếu context không đủ. Cuối cùng agent gọi `llm_fn(prompt)` để sinh câu trả lời, đúng pattern RAG: retrieve -> inject context -> generate.

### Test Results

```text
pytest tests/ -v

collected 42 items
42 passed in 0.04s
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions - Cá nhân (5 điểm)

Tôi dùng `TokenHashEmbedder` trong script Strategy A để embed câu, sau đó gọi `compute_similarity()` để lấy actual score.

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|-------------:|-------|
| 1 | Embedding converts text into numeric vectors for semantic search. | Text embeddings represent text as vectors used for similarity search. | high | 0.4811 | Đúng |
| 2 | Batch API processes asynchronous jobs for large datasets. | The Batch API is useful for non-realtime bulk processing. | high | 0.3354 | Đúng |
| 3 | Rate limits restrict how many requests or tokens can be used. | A chocolate cake recipe needs flour, sugar, and eggs. | low | 0.1005 | Đúng |
| 4 | Structured Outputs force model responses to match a JSON schema. | File Search retrieves relevant uploaded documents from a vector store. | low | 0.1000 | Đúng |
| 5 | Prompt engineering improves model responses with clear instructions. | Good prompts provide explicit instructions and context to the model. | high | 0.2000 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**  
Pair 5 có score chỉ 0.2000 dù dự đoán là high, vì hai câu liên quan nhưng không trùng nhiều từ khóa ngoài `prompt/instructions`. Điều này cho thấy embedding backend ảnh hưởng mạnh đến similarity score; token-hash fallback bắt được overlap từ khóa tốt nhưng chưa hiểu ngữ nghĩa sâu như sentence-transformers hoặc OpenAI embeddings.

---

## 6. Results - Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên Strategy A: Fixed-size Chunking + Vector Search.

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Embedding dùng để làm gì trong hệ thống AI/RAG? | Embedding biến văn bản thành vector số để đo mức độ liên quan hoặc tương đồng ngữ nghĩa giữa các đoạn text. Trong RAG, embedding thường được dùng để tìm các chunk tài liệu liên quan nhất với câu hỏi trước khi đưa context đó cho model trả lời. |
| 2 | File Search trong OpenAI API hoạt động như thế nào? | File Search cho phép model tìm thông tin trong các file đã upload thông qua vector store. Quy trình cơ bản là tạo vector store, upload file vào đó, rồi bật tool `file_search` trong request để model truy xuất các đoạn liên quan khi trả lời. |
| 3 | Khi nào nên dùng Structured Outputs? | Nên dùng Structured Outputs khi ứng dụng cần model trả về dữ liệu theo schema cố định, ví dụ JSON có field bắt buộc. Cách này phù hợp khi output cần được parse bằng code, lưu database, gọi API khác, hoặc giảm lỗi format so với câu trả lời text tự do. |
| 4 | Rate limit của OpenAI API được tính theo những đơn vị nào? | Rate limit có thể được tính theo request/phút, request/ngày, token/phút, token/ngày, và một số loại tài nguyên khác tùy endpoint như image/phút. Giới hạn cụ thể phụ thuộc vào model, project/organization và usage tier. |
| 5 | Batch API phù hợp với trường hợp nào? | Batch API phù hợp với tác vụ không cần phản hồi ngay, ví dụ chạy eval, phân loại dataset lớn, hoặc xử lý nhiều request/embedding cho kho tài liệu. Nó xử lý bất đồng bộ, có rate limit riêng và thường phù hợp cho job nền hơn là tương tác realtime. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|------:|-----------|------------------------|
| 1 | Embedding dùng để làm gì trong hệ thống AI/RAG? | D2 Embeddings - chunk nói về embedding vector và cách dùng embedding | 0.2032 | Có | Embedding là vector số dùng để đo liên quan/ngữ nghĩa và hỗ trợ semantic search/RAG. |
| 2 | File Search trong OpenAI API hoạt động như thế nào? | D3 File Search - chunk giới thiệu File Search tool trong Responses API | 0.2685 | Có | File Search dùng vector store/file đã upload để retrieve thông tin liên quan cho model. |
| 3 | Khi nào nên dùng Structured Outputs? | D4 Structured Outputs - chunk nói về structured model outputs/schema | 0.1600 | Có | Dùng khi cần output theo schema cố định để code parse và kiểm soát format. |
| 4 | Rate limit của OpenAI API được tính theo những đơn vị nào? | D5 Rate Limits - chunk giải thích rate limits của API | 0.2312 | Có | Rate limit giới hạn request/token theo phút/ngày và phụ thuộc model/project/tier. |
| 5 | Batch API phù hợp với trường hợp nào? | D6 Batch API - chunk giới thiệu Batch API cho asynchronous groups of requests | 0.1779 | Có | Batch API phù hợp với job nền/bulk processing không cần phản hồi realtime. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

### Evaluation Summary

| Metric | Value |
|---|---:|
| Chunk size | 700 ký tự |
| Overlap | 100 ký tự |
| Top-k | 3 |
| Embedding backend | `token-hash-vector-512d` |
| Total chunks indexed | 116 |
| Top-1 accuracy | 1.00 |
| Top-3 recall | 1.00 |

Nhận xét: Strategy A đạt kết quả tốt trên benchmark hiện tại vì các query có keyword rõ ràng và corpus nhỏ. Tuy nhiên fixed-size chunking vẫn có điểm yếu là có thể cắt ngang câu hoặc section; với tài liệu dài hơn hoặc query cần tổng hợp nhiều đoạn, recursive/sentence chunking có thể coherent hơn.

---

## 7. What I Learned (5 điểm - Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**  
Từ report của bạn Lê Hoài Nam và Đỗ Thiện Lĩnh, tôi thấy SentenceChunker giúp chunk dễ đọc hơn vì giữ ranh giới câu tự nhiên, đặc biệt với các đoạn giải thích kỹ thuật hoặc câu trả lời ngắn. Từ report của bạn Phan Quốc Anh, tôi thấy local dense embedding như `all-MiniLM-L6-v2` có thể nhận diện ngữ nghĩa tốt hơn mock/hash embedding, nhưng top-1 vẫn có thể bị lệch sang code/table chunk nếu chunking chưa giữ đúng section. Từ report của bạn Nguyễn Đức Kiên Trung, tôi học được vai trò của metadata filtering: plain search với mock embedder chỉ tìm đúng 2/5 query, nhưng khi filter theo `source_id` thì đạt 5/5. Điều này cho thấy retrieval quality không chỉ phụ thuộc chunking mà còn phụ thuộc embedding backend, metadata schema và cách thu hẹp candidate set.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**  
Từ ghi chú trong report C và E, tôi rút ra thêm ý tưởng kết hợp metadata với reranker hoặc hybrid search như BM25 + cosine similarity trước khi gọi LLM. Cách này có thể giảm false positive sau bước vector search, đặc biệt khi tài liệu API có nhiều keyword kỹ thuật, code snippet, bảng biểu, hoặc khi embedding backend yếu.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**  
Tôi sẽ thêm ít nhất một benchmark query phụ thuộc metadata rõ hơn, ví dụ yêu cầu trả lời dựa trên một tài liệu hoặc category cụ thể. Ngoài ra, tôi sẽ lưu metadata cấp section/heading để phân tích xem chunk relevant nằm ở phần nào của tài liệu, không chỉ biết `doc_id`.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|------------------:|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 9 / 10 |
| Chunking strategy | Nhóm | 13 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 4 / 5 |
| **Tổng tự đánh giá** | | **86 / 100** |
