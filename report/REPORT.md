# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** [Huỳnh Thái Bảo]
**Nhóm:** [Nhóm 01]
**Ngày:** [10/04/2026]

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity (gần bằng 1.0) có nghĩa là hai vector đại diện cho văn bản hướng về cùng một phía trong không gian vector. Về mặt ngôn ngữ, điều này cho thấy hai đoạn văn bản có sự tương đồng rất lớn về mặt ngữ nghĩa, dù từ ngữ sử dụng có thể khác nhau.

**Ví dụ HIGH similarity:**
- Sentence A: "Hôm nay trời rất nắng và nóng."
- Sentence B: "Thời tiết hôm nay có nhiều ánh nắng và nhiệt độ cao."
- Tại sao tương đồng: Cả hai câu đều mô tả cùng một trạng thái thời tiết (nắng, nóng) bằng các cách diễn đạt khác nhau.

**Ví dụ LOW similarity:**
- Sentence A: "Hôm nay trời rất nắng và nóng."
- Sentence B: "Tôi đang học lập trình Python tại trường."
- Tại sao khác: Hai câu nói về hai chủ đề hoàn toàn khác nhau (thời tiết vs. học tập).

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Vì cosine similarity tập trung vào **hướng** của vector thay vì **độ dài** (magnitude). Trong xử lý ngôn ngữ, độ dài vector thường bị ảnh hưởng bởi độ dài văn bản hoặc tần suất từ ngữ, trong khi cosine similarity giúp so sánh ý nghĩa cốt lõi mà không bị nhiễu bởi các yếu tố này.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* `ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.11)`
> *Đáp án:* 23 chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Nếu overlap = 100, số lượng chunk tăng lên (25 chunks). Chúng ta muốn overlap nhiều hơn để đảm bảo ngữ cảnh không bị cắt ngang giữa các chunk, giúp Agent có đủ thông tin liên kết giữa các đoạn khi truy xuất.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Scientific papers về **functional brain networks**, **causal dynamics** và **neural decoding** (neuroscience + network science)

**Tại sao nhóm chọn domain này?**
> Nhóm chọn domain này vì bộ tài liệu có cùng trục nội dung (kết nối não bộ, quan hệ nhân quả, higher-order interactions) nhưng đa dạng về phương pháp (graph theory, hypergraph, state-space, controllability). Điều này rất phù hợp để kiểm tra retrieval quality: query có thể đi từ mức khái niệm (ví dụ controllability) đến mức kỹ thuật (ví dụ hypergraph attention). Ngoài ra các bài báo đều có cấu trúc học thuật rõ ràng nên thuận lợi cho thiết kế metadata filtering.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | Networks beyond pairwise interactions (Battiston et al., 2020) | arXiv/Review article | ~45,000 | `topic=higher_order_networks`, `year=2020`, `type=review`, `source=arxiv` |
| 2 | Dynamic causal brain circuits during working memory (Cai et al., 2021) | Nature Communications | ~38,000 | `topic=working_memory`, `year=2021`, `method=state_space_control`, `source=nature` |
| 3 | Causal fMRI-Mamba (Deng et al., 2025) | IEEE-style manuscript | ~35,000 | `topic=neural_decoding`, `year=2025`, `method=state_space_mamba`, `source=ieee` |
| 4 | FC-HAT Hypergraph attention network (Ji et al., 2022) | Information Sciences (Elsevier) | ~36,000 | `topic=brain_network_classification`, `year=2022`, `method=hypergraph_attention`, `source=elsevier` |
| 5 | Complex network measures of brain connectivity (Rubinov & Sporns, 2010) | NeuroImage (Elsevier) | ~40,000 | `topic=brain_connectivity_metrics`, `year=2010`, `type=survey`, `source=elsevier` |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `topic` | string | `working_memory`, `hypergraph`, `connectivity_metrics` | Giúp lọc đúng chủ đề khi query có từ khóa chuyên ngành |
| `year` | integer | `2010`, `2021`, `2025` | Dùng để lọc theo mốc thời gian (classic vs recent methods) |
| `method` | string | `state_space`, `hypergraph_attention`, `network_measures` | Trả lời các câu hỏi so sánh phương pháp chính xác hơn |
| `type` | string | `review`, `method_paper` | Tách tài liệu tổng quan và tài liệu đề xuất mô hình |
| `source` | string | `nature`, `elsevier`, `ieee`, `arxiv` | Hữu ích khi cần truy xuất theo nguồn hoặc độ tin cậy |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy script `compare_all()` trên `paper.pdf` để so sánh 5 phương pháp:

| Tài liệu | Strategy | Chunk Count | Avg Length | Min / Max | Preserves Context? |
|-----------|----------|-------------|------------|-----------|-------------------|
| paper.pdf | BaselineChunker (`fixed + overlap`) | 792 | 799 | 364 / 800 | Trung bình (dễ cắt ngang ý) |
| paper.pdf | SentenceChunker (`sentence-based`) | 1067 | 481 | 28 / 1589 | Tốt (giữ mạch theo câu) |
| paper.pdf | SectionChunker (`section-based`) | 117 | 4361 | 228 / 24384 | Theo cấu trúc lớn, ít chi tiết nhỏ |
| paper.pdf | SemanticChunker (`semantic`) | 850 | 604 | 33 / 1589 | Tốt về ngữ nghĩa (khi có model) |
| paper.pdf | RecursiveChunker (`recursive`) | 587 | 875 | 89 / 1000 | Rất tốt (cân bằng cấu trúc + độ dài) |

### Strategy Của Tôi

**Loại:** [BaselineChunker (Fixed-size + overlap)]

**Mô tả cách hoạt động:**
> Đây là phương pháp phân đoạn văn bản cơ bản nhất, thực hiện chia văn bản thành các đoạn có kích thước cố định (`chunk_size`). Để tránh việc mất thông tin tại các vị trí ngắt đoạn, phương pháp này sử dụng một khoảng chồng lấp (`overlap`) giữa các đoạn kế tiếp nhau, đảm bảo ngữ cảnh ở cuối đoạn này xuất hiện ở đầu đoạn kia.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Mặc dù đơn giản, FixedSizeChunker đảm bảo tốc độ xử lý cực nhanh cho các tài liệu khoa học dài. Trong domain Neuroscience với nhiều từ ngữ chuyên ngành phức tạp, việc chia theo kích thước cố định giúp tránh được các lỗi tách câu sai do dấu câu trong các công thức hoặc tên viết tắt, đồng thời tạo ra các vector embeddings có độ dài đồng nhất.

**Code snippet (nếu custom):**
```python
class BaselineChunker:
    def __init__(self, chunk_size: int = 800, overlap: int = 150):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[str]:
        chunks = []
        i = 0
        while i < len(text):
            end = min(i + self.chunk_size, len(text))
            chunks.append(text[i:end])
            i += self.chunk_size - self.overlap
        return chunks
```

### So Sánh: Strategy của tôi với các strategy khác

| Tài liệu | Strategy | Chunk Count | Avg Length | Đánh giá retrieval |
|-----------|----------|-------------|------------|--------------------|
| paper.pdf | **Baseline (Của tôi)** | 792 | 799 | Trung bình, đổi lại tốc độ cao |
| paper.pdf | Sentence-based | 1067 | 481 | Tốt ở truy vấn theo câu |
| paper.pdf | Section-based | 117 | 4361 | Tốt cho truy vấn theo section lớn |
| paper.pdf | Semantic | 850 | 604 | Tốt về cụm nghĩa |
| paper.pdf | Recursive | 587 | 875 | Tốt nhất tổng thể |

**Kết luận từ kết quả thực nghiệm (`Result.txt`):**
> BaselineChunker là lựa chọn đơn giản, nhanh và ổn định để làm mốc so sánh; trong khi đó RecursiveChunker cho cân bằng tốt nhất giữa độ dài chunk và khả năng giữ ngữ cảnh trên bộ tài liệu này.

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Trương Minh Tiền | Sentence-based | 8.5/10 | Giữ nội dung, trả lời tròn câu hỏi khái niệm. | Tạo ra quá nhiều chunk (1067). |
| Phạm Đoàn Phương Anh | Semantic Chunking | 9.0/10 | Ghép câu đồng nghĩa bằng LLM embeddings. | Tốn tài nguyên RAM và xử lý rất chậm. |
| Nguyễn Đức Dũng | Section-based | 7.5/10 | Bắt đúng Header lớn. | Chunk quá to dẫn đến độ nhiễu loạn cao. |
| Nguyễn Đức Trí | Recursive Chunking | 8.0/10 | Cân bằng tuyệt vời giữa logic và tốc độ. | Vẫn thỉnh thoảng cắt vỡ ý tưởng dài. |
| Huỳnh Thái Bảo | Baseline (Fixed) | 5.0/10 | Siêu nhanh, code đơn giản, chi phí bằng 0. | Tách vỡ câu, vô nghĩa hoàn toàn lúc Retrieve. |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> RecursiveChunker cho kết quả tốt nhất trên bộ tài liệu nhóm vì văn bản học thuật có cấu trúc phân cấp (Section -> Paragraph -> Sentence) và độ dài không đồng đều. Cách tách đệ quy giúp giữ coherence của chunk tốt hơn FixedSize và ít phụ thuộc vào dấu câu hơn SentenceChunker. Với các query cần bằng chứng đầy đủ trong 1 đoạn (như các phương pháp trong Ji et al. hay Deng et al.), strategy này giúp Model "hiểu" được trọn vẹn một luận điểm khoa học.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Sử dụng biểu thức chính quy `(?<=[.!?])(?:\s+|\n+)` để tách văn bản tại các dấu kết thúc câu kèm khoảng trắng, sau đó gộp các câu lại thành nhóm dựa trên `max_sentences_per_chunk`.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Thuật toán đệ quy thử tách văn bản theo danh sách ký tự phân tách ưu tiên (paragraph, newline, space). Nếu một đoạn vẫn vượt quá `chunk_size`, nó sẽ tiếp tục đệ quy cho đến khi đạt kích thước nhỏ hơn giới hạn.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Lưu trữ các chunk kèm embedding (được tạo qua `embedding_fn`) vào bộ nhớ. Khi tìm kiếm, tạo embedding cho query và tính dot product với tất cả chunk để tìm ra Top-K kết quả có điểm cao nhất.

**`search_with_filter` + `delete_document`** — approach:
> Filtering được thực hiện bằng cách lọc các record theo metadata trước khi tính toán similarity. Hàm xóa sẽ loại bỏ tất cả các record có `doc_id` trùng khớp khỏi store.

### KnowledgeBaseAgent

**`answer`** — approach:
> Agent thực hiện RAG bằng cách truy xuất context liên quan, sau đó inject vào một template prompt yêu cầu mô hình chỉ trả lời dựa trên thông tin đã cung cấp.

### Test Results

```
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.0.2, pluggy-1.6.0
rootdir: C:\assignments-main\Lab7-canhan
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED
... (42 passed total)
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED

============================= 42 passed in 1.08s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "Học máy là một nhánh của AI" | "Machine learning là một phần của trí tuệ tạo" | high | 0.92 | Yes |
| 2 | "Tôi thích ăn táo" | "Quả táo này rất ngon" | high | 0.75 | Yes |
| 3 | "Trời đang mưa to" | "Thủ đô của Pháp là Paris" | low | 0.05 | Yes |
| 4 | "Lập trình viên viết code" | "Kỹ sư phần mềm xây dựng ứng dụng" | high | 0.81 | Yes |
| 5 | "Con chó đang sủa" | "Mèo thích bắt chuột" | low | 0.22 | Yes |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Kết quả cặp số 5 có thể hơi bất ngờ vì chúng đều về động vật nhưng điểm vẫn thấp. Điều này cho thấy embeddings tập trung rất sâu vào ngữ nghĩa cụ thể của từng từ và ngữ cảnh thay vì chỉ là các từ cùng chủ đề rộng.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Theo Cai et al. (2021), những network nào tham gia chính trong working memory? | Ba network chính: **Salience Network (SN)**, **Frontoparietal Network (FPN)** và **Default Mode Network (DMN)**. |
| 2 | Trong FC-HAT (Ji et al., 2022), mô hình xử lý thông tin bậc cao bằng cách nào? | FC-HAT mô hình hóa functional brain network dưới dạng **hypergraph**, kết hợp **KNN + k-means** để tạo hyperedge và dùng **node/hyperedge attention** để tổng hợp đặc trưng. |
| 3 | Causal fMRI-Mamba (Deng et al., 2025) giải quyết hạn chế nào của CNN/attention truyền thống trên fMRI? | Mô hình nhắm tới việc nắm bắt **global spatiotemporal dependencies**, giảm dư thừa cục bộ và cải thiện tổng quát hóa trước nhiễu + khác biệt cá thể. |
| 4 | Theo Rubinov & Sporns (2010), complex network analysis dùng để làm gì trong nghiên cứu kết nối não? | Dùng các network measures để mô tả **integration/segregation**, **centrality**, độ bền mạng và so sánh connectivity giữa cá thể/nhóm bệnh lý. |
| 5 | (Metadata filter: `year >= 2022`) Kể tên các paper trong bộ dữ liệu đề xuất mô hình mới thay vì bài tổng quan. | Các bài thỏa điều kiện gồm: **Ji et al. (2022, FC-HAT)** và **Deng et al. (2025, Causal fMRI-Mamba)**; đều là method papers đề xuất mô hình mới. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Theo Cai et al. (2021), những network nào tham gia chính trong working memory? | "...dynamic signaling between distributed brain areas encompassing the salience (SN), frontoparietal (FPN), and default mode networks..." | 0.92 | Yes | Ba mạng chính gồm Salience Network (SN), Frontoparietal Network (FPN) và Default Mode Network (DMN). |
| 2 | Trong FC-HAT (Ji et al., 2022), mô hình xử lý thông tin bậc cao bằng cách nào? | "...we propose a hypergraph attention network... captured by k nearest neighbors and k-means... node and hyperedge attention layers..." | 0.89 | Yes | FC-HAT sử dụng hypergraph để lưu giữ thông tin bậc cao, dùng KNN và k-means tạo hyperedge, kết hợp node/hyperedge attention. |
| 3 | Causal fMRI-Mamba (Deng et al., 2025) giải quyết hạn chế nào của CNN/attention truyền thống trên fMRI? | "... struggle with capturing global spatiotemporal information due to high dimensionality, noise... It effectively captures global information..." | 0.88 | Yes | Mô hình giải quyết việc nắm bắt thông tin không gian-thời gian toàn cục, giảm dư thừa cục bộ và xử lý tốt long-distance dependencies. |
| 4 | Theo Rubinov & Sporns (2010), complex network analysis dùng để làm gì trong nghiên cứu kết nối não? | "...aims to characterize these brain networks with a small number of neurobiologically meaningful measures... detect functional integration and segregation..." | 0.94 | Yes | Dùng để mô tả các đặc tính tích hợp và phân tách chức năng, xác định độ trung tâm (centrality) và kiểm tra khả năng phục hồi của mạng. |
| 5 | (Metadata filter: `year >= 2022`) Kể tên các paper trong bộ dữ liệu đề xuất mô hình mới thay vì bài tổng quan. | [Metadata Filter Applied: year >= 2022] - "FC-HAT: Hypergraph attention network..." & "Causal fMRI-Mamba..." | N/A | Yes | Có 2 bài: FC-HAT (Ji et al., 2022) và Causal fMRI-Mamba (Deng et al., 2025). |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Tôi học được cách tối ưu hóa SentenceChunker của Thành viên A bằng cách tinh chỉnh thủ công các regex để nhận diện chính xác các từ viết tắt chuyên ngành trong Neuroscience (như "e.g.", "i.e."), giúp tránh ngắt câu sai vị trí.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Một nhóm nghiên cứu về Luật đã sử dụng "Contextual Chunking" bằng cách đính kèm tiêu đề chương/mục vào đầu mỗi chunk. Điều này rất hữu ích cho các bài báo khoa học khi mà metadata thường bị mất khi text bị chia nhỏ.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ thêm bước "Cleaning" kỹ hơn để loại bỏ các ký tự rác từ PDF (như header/footer lặp lại, số trang) vì chúng làm loãng vector embeddings và giảm độ chính xác khi retrieval.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **100 / 100** |
