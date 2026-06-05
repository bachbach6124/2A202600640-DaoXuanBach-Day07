# Strategy A - Fixed-size Chunking + Vector Search

## Setup

- Chunk size: `700` characters
- Overlap: `100` characters
- Top-k: `3`
- Embedding backend: `token-hash-vector-512d`
- Total chunks indexed: `116`

## Metrics

- Top-1 accuracy: `1.00`
- Top-3 recall: `1.00`

## Query Results

### Q1. Embedding dùng để làm gì trong hệ thống AI/RAG?

Gold answer: Embedding biến văn bản thành vector số để đo mức độ liên quan hoặc tương đồng ngữ nghĩa giữa các đoạn text. Trong RAG, embedding thường được dùng để tìm các chunk tài liệu liên quan nhất với câu hỏi trước khi đưa context đó cho model trả lời.

Relevant docs: `D2`

Top-1 hit: `True`; Top-k hit: `True`

| Rank | Score | Doc | Title | Chunk | Preview |
|---|---:|---|---|---:|---|
| 1 | 0.2032 | D2 | Embeddings | 3 | ample: Getting embeddings The response contains the embedding vector (list of floating point numbers) along with some additional metadata. You can extract the embedding vector, ... |
| 2 | 0.1745 | D2 | Embeddings | 4 | offers two powerful third-generation embedding model (denoted by -3 in the model ID). Read the embedding v3 announcement blog post for more details. Usage is priced per input to... |
| 3 | 0.1394 | D2 | Embeddings | 22 | esult in the identical rankings Can I share my embeddings online? Yes, customers own their input and output from our models, including in the case of embeddings. You are respons... |

### Q2. File Search trong OpenAI API hoạt động như thế nào?

Gold answer: File Search cho phép model tìm thông tin trong các file đã upload thông qua vector store. Quy trình cơ bản là tạo vector store, upload file vào đó, rồi bật tool file_search trong request để model truy xuất các đoạn liên quan khi trả lời.

Relevant docs: `D3`

Top-1 hit: `True`; Top-k hit: `True`

| Rank | Score | Doc | Title | Chunk | Preview |
|---|---:|---|---|---:|---|
| 1 | 0.2685 | D3 | File Search | 1 | --- doc_id: D3 title: File Search source_url: https://developers.openai.com/api/docs/guides/tools-file-search.md --- File search File search is a tool available in the Responses... |
| 2 | 0.1732 | D6 | Batch API | 1 | --- doc_id: D6 title: Batch API source_url: https://developers.openai.com/api/docs/guides/batch.md --- Batch API Learn how to use OpenAI's Batch API to send asynchronous groups ... |
| 3 | 0.1719 | D3 | File Search | 2 | anaged by OpenAI, meaning you don't have to implement code on your end to handle its execution. When the model decides to use it, it will automatically call the tool, retrieve i... |

### Q3. Khi nào nên dùng Structured Outputs?

Gold answer: Nên dùng Structured Outputs khi ứng dụng cần model trả về dữ liệu theo schema cố định, ví dụ JSON có field bắt buộc. Cách này phù hợp khi output cần được parse bằng code, lưu database, gọi API khác, hoặc giảm lỗi format so với câu trả lời text tự do.

Relevant docs: `D4`

Top-1 hit: `True`; Top-k hit: `True`

| Rank | Score | Doc | Title | Chunk | Preview |
|---|---:|---|---|---:|---|
| 1 | 0.1600 | D4 | Structured Outputs | 1 | --- doc_id: D4 title: Structured Outputs source_url: https://developers.openai.com/api/docs/guides/structured-outputs.md --- Structured model outputs .trim(), }; .trim(), }; JSO... |
| 2 | 0.1547 | D4 | Structured Outputs | 5 | in your system, then you should use function calling - If you want to structure the model's output when it responds to the user, then you should use a structured text.format The... |
| 3 | 0.1501 | D4 | Structured Outputs | 6 | ode are supported in the Responses API, Chat Completions API, Assistants API, Fine-tuning API and Batch API. We recommend always using Structured Outputs instead of JSON mode wh... |

### Q4. Rate limit của OpenAI API được tính theo những đơn vị nào?

Gold answer: Rate limit có thể được tính theo request/phút, request/ngày, token/phút, token/ngày, và một số loại tài nguyên khác tùy endpoint như image/phút. Giới hạn cụ thể phụ thuộc vào model, project/organization và usage tier.

Relevant docs: `D5`

Top-1 hit: `True`; Top-k hit: `True`

| Rank | Score | Doc | Title | Chunk | Preview |
|---|---:|---|---|---:|---|
| 1 | 0.2312 | D5 | Rate Limits | 1 | --- doc_id: D5 title: Rate Limits source_url: https://developers.openai.com/api/docs/guides/rate-limits.md --- Rate limits Rate limits are restrictions that our API imposes on t... |
| 2 | 0.2190 | D5 | Rate Limits | 2 | ate limits, OpenAI can prevent this kind of activity. - **Rate limits help ensure that everyone has fair access to the API.** If one person or organization makes an excessive nu... |
| 3 | 0.2019 | D5 | Rate Limits | 11 | and automated social media posting - consider only enabling these for trusted customers. To protect against automated and high-volume misuse, set a usage limit for individual us... |

### Q5. Batch API phù hợp với trường hợp nào?

Gold answer: Batch API phù hợp với tác vụ không cần phản hồi ngay, ví dụ chạy eval, phân loại dataset lớn, hoặc xử lý nhiều request/embedding cho kho tài liệu. Nó xử lý bất đồng bộ, có rate limit riêng và thường phù hợp cho job nền hơn là tương tác realtime.

Relevant docs: `D6`

Top-1 hit: `True`; Top-k hit: `True`

| Rank | Score | Doc | Title | Chunk | Preview |
|---|---:|---|---|---:|---|
| 1 | 0.1779 | D6 | Batch API | 1 | --- doc_id: D6 title: Batch API source_url: https://developers.openai.com/api/docs/guides/batch.md --- Batch API Learn how to use OpenAI's Batch API to send asynchronous groups ... |
| 2 | 0.1775 | D6 | Batch API | 3 | red to using standard endpoints directly, Batch API has: 1. **Better cost efficiency:** 50% cost discount compared to synchronous APIs 2. **Higher rate limits:** Substantially m... |
| 3 | 0.1702 | D7 | Safety Best Practices | 10 | carry over between APIs or sessions. If your application already sends safety_identifier with Responses API requests, pass the same stable value separately when you create or co... |
