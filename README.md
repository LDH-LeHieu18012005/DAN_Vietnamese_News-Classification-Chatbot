# 📰 NewsMind — Vietnamese News Classifier

**NewsMind** là hệ thống phân loại bài báo tiếng Việt end-to-end, kết hợp mô hình học sâu **PhoBERT** (fine-tuned) với trợ lý AI **Llama-3.1-8B** để phân loại và giải thích nội dung tin tức theo thời gian thực.

> **Nguồn dữ liệu:** Báo Lao Động (laodong.vn) | **4 chủ đề:** Công nghệ · Kinh doanh · Thế giới · Thể thao

---

## 📑 Mục lục

1. [Tính năng nổi bật](#-tính-năng-nổi-bật)
2. [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
3. [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
4. [Dữ liệu](#-dữ-liệu)
5. [Mô hình & Kết quả](#-mô-hình--kết-quả)
6. [API Reference](#-api-reference)
7. [Cài đặt & Chạy](#-cài-đặt--chạy)
8. [Quy trình ML Pipeline](#-quy-trình-ml-pipeline)
9. [Công nghệ sử dụng](#-công-nghệ-sử-dụng)

---

## ✨ Tính năng nổi bật

| Tính năng | Mô tả |
|---|---|
| 🧠 **Phân loại chính xác** | Fine-tuned `vinai/phobert-base` đạt Accuracy **96.1%** và F1 **96.1%** trên tập test |
| ⚡ **API tốc độ cao** | Backend FastAPI + Uvicorn, route `/classify_text` và `/classify_batch` |
| 💬 **Trợ lý AI tích hợp** | Chatbot Llama-3.1-8B-Instruct (qua HuggingFace Router API) giải thích kết quả và hỏi đáp nội dung bài báo |
| 📊 **Phân tích toàn diện** | Trả về confidence score (%), confidence tier (HIGH/MED/LOW), số từ, thời gian đọc ước tính, câu chủ đề |
| 🎨 **UI/UX hiện đại** | Single-page app HTML/CSS/JS thuần, dark-mode, animate, không cần framework |
| 🔒 **Bảo mật API key** | HuggingFace API key chỉ nằm ở backend (`.env`), client không bao giờ thấy key |
| 📦 **Batch classification** | Phân loại nhiều bài báo cùng lúc qua một API call duy nhất |

---

## 🏗 Kiến trúc hệ thống

```
┌─────────────────── Client (Browser) ───────────────────┐
│                    index.html                           │
│   ┌──────────────┐         ┌────────────────────────┐  │
│   │  Text Input  │         │   Chatbot UI (Llama)   │  │
│   └──────┬───────┘         └─────────────┬──────────┘  │
└──────────┼──────────────────────────────┼─────────────┘
           │ POST /classify_text           │ POST /chat
           ▼                              ▼
┌─────────────────── Backend (api.py) ───────────────────┐
│                    FastAPI + Uvicorn                    │
│                                                         │
│   ┌────────────────────────────────────────────────┐   │
│   │         PhoBERT Classifier                     │   │
│   │  vinai/phobert-base → Fine-tuned (model.pt)    │   │
│   │  · AutoTokenizer (max_len=512)                 │   │
│   │  · Expanded Position Embeddings                │   │
│   │  · Dropout(0.1) → Linear → Softmax             │   │
│   └────────────────────────────────────────────────┘   │
│                                                         │
│   ┌────────────────────────────────────────────────┐   │
│   │         LLM Proxy                              │   │
│   │  httpx → HuggingFace Router API                │   │
│   │  meta-llama/Llama-3.1-8B-Instruct:cerebras     │   │
│   └────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────┘
```

---

## 📂 Cấu trúc thư mục

> **Ghi chú Git:** Các file đánh dấu 🚫 **không được đẩy lên repository** do kích thước lớn (vượt giới hạn 100 MB của GitHub). Xem hướng dẫn tải xuống thủ công ở phần [Cài đặt & Chạy](#-cài-đặt--chạy).

```text
The News/
│
├── 📄 api.py
│       Backend chính của ứng dụng, xây dựng bằng FastAPI.
│       Thực hiện 3 nhiệm vụ chính:
│         (1) Load và phục vụ mô hình PhoBERT để phân loại văn bản;
│         (2) Expose REST API: /classify_text, /classify_batch, /chat;
│         (3) Proxy các request đến HuggingFace LLM API (ẩn API key khỏi client).
│       Được khởi động bằng: uvicorn api:app --reload --port 8000
│
├── 📄 index.html
│       Toàn bộ giao diện Web Frontend viết bằng HTML/CSS/JS thuần (single file).
│       Bao gồm: ô nhập văn bản, hiển thị kết quả phân loại với biểu đồ xác suất,
│       panel chatbot tích hợp AI (Llama-3.1), dark-mode, animations.
│       Được phục vụ trực tiếp qua route GET / của FastAPI.
│
├── 📄 .env                          ← KHÔNG commit lên git
│       File biến môi trường cục bộ. Chứa HF_API_KEY (API key HuggingFace)
│       để backend gọi Llama-3.1-8B-Instruct qua HuggingFace Router API.
│       Tạo thủ công theo mẫu: HF_API_KEY=hf_xxxxxxxxxxxxxxxxxxxx
│
├── 📄 .gitignore
│       Định nghĩa các file/folder bị loại khỏi Git tracking:
│       model weights, embedding artifacts, dataset CSV, Python cache.
│
├── 📄 Báo cáo.pptx
│       Slide PowerPoint trình bày tổng quan dự án, kiến trúc hệ thống,
│       quy trình ML pipeline và kết quả thực nghiệm.
│
├── 📓 Crawll.ipynb
│       Notebook thu thập dữ liệu (Web Scraping) từ báo Lao Động (laodong.vn).
│       Crawl 4 chuyên mục: Công nghệ, Kinh doanh, Thế giới, Thể thao.
│       Xuất ra các file CSV thô theo từng chủ đề vào thư mục data/.
│
├── 📓 base-news (1).ipynb
│       Notebook EDA (Exploratory Data Analysis) và tiền xử lý dữ liệu.
│       Phân tích phân phối nhãn, độ dài văn bản, xử lý NaN và ký tự đặc biệt.
│       Ghép 4 CSV thô, tạo train/test split (85/15%), thêm cột is_test.
│       Xuất ra laodong_all_news_with_split.csv — dataset chính cho toàn dự án.
│
├── 📓 phobert.ipynb
│       Notebook fine-tune mô hình PhoBERT cho bài toán phân loại tin tức.
│       Load vinai/phobert-base, mở rộng positional embeddings lên 512 vị trí,
│       fine-tune 3 epochs với AdamW (lr=2e-5, fp16), đánh giá trên test set.
│       Kết quả: Accuracy 96.11%, F1 96.12%. Lưu model.pt, config.json, metrics.json.
│
│
├── 📁 data/                         🚫 Không có trong Git (file quá lớn)
│   │   Chứa toàn bộ dataset tin tức tiếng Việt được crawl từ laodong.vn.
│   │   Tổng dung lượng: ~41 MB. Cần có để chạy lại các notebooks training.
│   │
│   ├── laodong_all_news.csv         (~19.7 MB)
│   │       Dataset gốc, chứa toàn bộ bài báo từ 4 chủ đề, chưa có split.
│   │       Các cột chính: title, content, category, url, date.
│   │
│   ├── laodong_all_news_with_split.csv  (~19.7 MB)  ← File dùng cho training
│   │       Dataset đã xử lý, bổ sung cột is_test (0=train, 1=test).
│   │       Tổng 6.000 bài: 5.100 train + 900 test. Đây là input cho tất cả
│   │       các notebooks embedding và fine-tuning.
│   │
│   ├── laodong_congnghe.csv         (~4.6 MB) — Nhãn: Công nghệ
│   ├── laodong_kinhdoanh.csv        (~6.3 MB) — Nhãn: Kinh doanh
│   ├── laodong_thegioi.csv          (~4.7 MB) — Nhãn: Thế giới
│   └── laodong_thethao.csv          (~4.0 MB) — Nhãn: Thể thao
│
│
├── 📁 embedding/
│   │   Chứa các notebooks và artifacts của 2 phương pháp word embedding
│   │   truyền thống được nghiên cứu để so sánh với PhoBERT.
│   │
│   ├── 📓 glove.ipynb
│   │       Cài đặt GloVe từ đầu bằng PyTorch (không dùng thư viện có sẵn).
│   │       Xây dựng co-occurrence matrix thưa với sliding window size=5,
│   │       weight decay theo khoảng cách (w=1/dist). Train GloVeModel (PyTorch nn.Module)
│   │       50 epochs, batch=1024, lr=0.05. Embedding dim=300, vocab=28.126 từ.
│   │       Training time: ~5h40m trên Kaggle GPU. Final loss: ~0.009.
│   │
│   ├── 📓 word2vec.ipynb
│   │       Huấn luyện Word2Vec Skip-gram dùng thư viện Gensim.
│   │       vector_size=300, window=5, min_count=2, epochs=20, sg=0 (CBOW).
│   │       Coverage vocabulary: 99.99% (28.124/28.126 từ).
│   │       Xuất embedding matrix (.pt) và word2idx map để dùng với LSTM.
│   │
│   ├── 📁 glove/                    🚫 Không có trong Git (file quá lớn, ~193 MB)
│   │   ├── glove_model_state.pt         (~64.6 MB) — Trọng số GloVe sau 50 epochs
│   │   ├── embedding_glove_content.pt   (~32.2 MB) — Ma trận embedding 28126×300
│   │   ├── word2idx_glove_content.pt    (~0.5 MB)  — Dict ánh xạ từ → index
│   │   └── idx2word_glove_content.pt    (~0.5 MB)  — Dict ánh xạ index → từ
│   │
│   └── 📁 skip-gram/               🚫 Không có trong Git (file quá lớn, ~130 MB)
│       ├── word2vec_content.model       (~65.2 MB) — Gensim Word2Vec model đầy đủ
│       ├── embedding_content.pt         (~32.2 MB) — Ma trận embedding 28126×300
│       └── word2idx_content.pt          (~0.5 MB)  — Dict ánh xạ từ → index
│
│
├── 📁 extend/
│   │   Tài nguyên NLP tiếng Việt bổ trợ, dùng trong quá trình tiền xử lý.
│   │
│   ├── vn_stopwords.txt             (~20 KB)
│   │       Danh sách stop words (từ dừng) tiếng Việt, dùng để lọc bỏ các từ
│   │       không mang ý nghĩa (của, và, là, thì, ...) trong bước preprocessing.
│   │
│   └── word_net_vi.json             (~1.4 MB)
│           Vietnamese WordNet — cơ sở dữ liệu quan hệ ngữ nghĩa tiếng Việt.
│           Chứa các synset (nhóm từ đồng nghĩa), hypernym/hyponym relationships.
│           Dùng để mở rộng từ vựng hoặc tra cứu semantic similarity.
│
│
└── 📁 model/
    └── 📁 bertpho/
        │   Chứa toàn bộ artifacts của mô hình PhoBERT đã fine-tune.
        │   Đây là thư mục quan trọng nhất để chạy API.
        │
        ├── model.pt                 🚫 Không có trong Git (~516 MB)
        │       File trọng số chính của mô hình PhoBERT đã fine-tune.
        │       Được load bởi api.py khi khởi động server.
        │       Tải xuống: [xem hướng dẫn bên dưới]
        │
        ├── config.json
        │       Lưu siêu tham số huấn luyện: optimizer, lr, batch_size,
        │       max_seq_length, dropout, warmup_ratio, fp16, tổng số params (135M).
        │       Dùng để tái hiện lại quá trình training nếu cần.
        │
        ├── metrics.json
        │       Kết quả đánh giá trên test set sau khi training hoàn tất.
        │       Chứa: eval_accuracy, eval_f1, eval_precision, eval_recall, eval_loss.
        │
        ├── vocab.txt                (~122 KB)
        │       Vocabulary 64.000 subword tokens của PhoBERT (BPE-based).
        │       Được AutoTokenizer đọc khi khởi động.
        │
        ├── bpe.codes                (~1.1 MB)
        │       Bảng BPE merge rules — định nghĩa cách tokenizer tách từ tiếng Việt
        │       thành các subword units (byte-pair encoding rules).
        │
        ├── tokenizer_config.json
        │       Cấu hình tokenizer: model_type, do_lower_case, bos_token, eos_token.
        │
        ├── special_tokens_map.json
        │       Ánh xạ các special tokens: [CLS], [SEP], [PAD], [UNK], [MASK].
        │
        └── added_tokens.json
                Các token đặc biệt được thêm vào vocabulary ngoài bộ vocab gốc.
```

---

## 📊 Dữ liệu

### Thu thập
Dữ liệu được crawl từ báo **Lao Động** (`laodong.vn`) theo 4 chuyên mục tương ứng với 4 nhãn phân loại.

### Thống kê tập dữ liệu chính (`laodong_all_news_with_split.csv`)

| Tập | Số lượng | Tỷ lệ |
|---|---|---|
| **Train** (`is_test=0`) | 5.100 bài | 85% |
| **Test** (`is_test=1`) | 900 bài | 15% |
| **Tổng** | **6.000 bài** | 100% |

| Chủ đề | File CSV | Kích thước |
|---|---|---|
| Công nghệ | `laodong_congnghe.csv` | ~4.8 MB |
| Kinh doanh | `laodong_kinhdoanh.csv` | ~6.7 MB |
| Thế giới | `laodong_thegioi.csv` | ~4.9 MB |
| Thể thao | `laodong_thethao.csv` | ~4.2 MB |

### Embedding Vocabulary
Cả hai mô hình embedding đều được xây dựng trên tập train:
- **Vocabulary size:** 28.126 từ (min_count ≥ 2)
- **Total tokens:** 2.681.617

---

## 🤖 Mô hình & Kết quả

### PhoBERT Fine-tuned Classifier

#### Kiến trúc
```
vinai/phobert-base
  └── AutoModel (RoBERTa-based, 12 layers, hidden=768)
        ├── Expanded Position Embedding (512 → 514 positions, interpolated)
        ├── Dropout(p=0.1)
        └── Linear(768 → 4)  →  Softmax → [cong-nghe, kinh-doanh, the-gioi, the-thao]
```

#### Siêu tham số huấn luyện (`config.json`)

| Tham số | Giá trị |
|---|---|
| Base model | `vinai/phobert-base` |
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Batch size | 8 |
| Max sequence length | 512 tokens |
| Dropout | 0.1 |
| Weight decay | 0.01 |
| Epochs | 3 |
| Warmup ratio | 6% |
| Mixed precision (fp16) | ✅ |
| Total params | 135.197.956 |

#### Kết quả đánh giá trên test set (`metrics.json`)

| Metric | Giá trị |
|---|---|
| **Accuracy** | **96.11%** |
| **F1 Score** | **96.12%** |
| F1 Macro | 96.14% |
| Precision | 96.18% |
| Recall | 96.13% |
| Eval Loss | 0.1427 |
| Throughput | 59.18 samples/s |

### Word Embeddings (Nghiên cứu)

Ngoài PhoBERT, dự án còn huấn luyện và so sánh 2 phương pháp embedding truyền thống:

#### GloVe (tự cài đặt với PyTorch)

| Tham số | Giá trị |
|---|---|
| Embedding dim | 300 |
| Window size | 5 |
| Epochs | 50 |
| Batch size | 1.024 |
| Learning rate | 0.05 |
| Co-occurrence pairs | 5.161.648 |
| Training time | ~5h40m (Kaggle GPU) |
| Final avg loss | ~0.009 |

> **Kỹ thuật:** Xây dựng co-occurrence matrix thưa (sparse) với weight decay theo khoảng cách (weight = 1/distance), loss function GloVe gốc với weighting function `f(x) = (x/x_max)^alpha`.

#### Word2Vec Skip-gram (Gensim)

| Tham số | Giá trị |
|---|---|
| Embedding dim | 300 |
| Window size | 5 |
| Min count | 2 |
| Epochs | 20 |
| Workers | 4 |
| Coverage | 28.124/28.126 (99.99%) |
| Training time | ~1.6 phút (Kaggle GPU) |

---

## 🔌 API Reference

Base URL: `http://localhost:8000`

### `GET /health`
Kiểm tra trạng thái server.
```json
// Response
{ "status": "ok", "device": "cuda" }
```

### `GET /config`
Kiểm tra cấu hình (có HF key hay không, **không** trả key ra client).
```json
{ "has_hf_key": true }
```

### `POST /classify_text`
Phân loại một bài báo đơn.

**Request:**
```json
{ "text": "Apple vừa ra mắt iPhone mới với chip A18..." }
```

**Response:**
```json
{
  "label": "cong-nghe",
  "label_vi": "Công nghệ",
  "confidence": 0.9873,
  "confidence_tier": "HIGH",
  "all_probs": {
    "cong-nghe": 98.73,
    "kinh-doanh": 0.81,
    "the-gioi": 0.27,
    "the-thao": 0.19
  },
  "word_count": 312,
  "read_time": 2,
  "top_sentences": "Apple vừa ra mắt iPhone mới với chip A18..."
}
```

**Confidence tiers:**
| Tier | Ngưỡng |
|---|---|
| `HIGH` | confidence ≥ 0.80 |
| `MED` | 0.55 ≤ confidence < 0.80 |
| `LOW` | confidence < 0.55 |

### `POST /classify_batch`
Phân loại nhiều bài báo cùng lúc.

**Request:**
```json
[
  { "text": "Bài báo thứ nhất..." },
  { "text": "Bài báo thứ hai..." }
]
```

**Response:** Mảng kết quả, mỗi phần tử tương tự `/classify_text`, thêm `index` và `preview` (80 ký tự đầu).

### `POST /chat`
Proxy gọi LLM (Llama-3.1-8B) qua HuggingFace Router, ẩn API key ở backend.

**Request:**
```json
{
  "messages": [{ "role": "user", "content": "Tại sao bài này được xếp vào Công nghệ?" }],
  "system": "Bạn là trợ lý phân tích tin tức..."
}
```

**Response:**
```json
{ "reply": "Bài báo đề cập đến chip A18..." }
```

---

## 🚀 Cài đặt & Chạy

### Yêu cầu hệ thống
- Python 3.8+
- CUDA (tùy chọn — tự động dùng CPU nếu không có GPU)
- RAM ≥ 4 GB (8 GB khuyến nghị)
- Disk: ~700 MB (cho model PhoBERT)

### 1. Cài đặt thư viện

```bash
pip install fastapi uvicorn torch transformers pydantic python-dotenv httpx
```

### 2. Cấu hình biến môi trường

Tạo file `.env` tại thư mục gốc:

```env
HF_API_KEY=hf_xxxxxxxxxxxx_your_huggingface_api_key
```

> Lấy API key miễn phí tại: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
> 
> **Lưu ý:** Nếu không có key, tính năng chatbot sẽ bị vô hiệu hóa, còn phân loại văn bản vẫn hoạt động bình thường.

### 3. Kiểm tra cấu trúc model

Đảm bảo file trọng số tồn tại tại đường dẫn:
```
model/bertpho/model.pt   (~516 MB)
```

### 4. Khởi chạy server

```bash
uvicorn api:app --reload --port 8000
```

### 5. Truy cập ứng dụng

Mở trình duyệt và vào: **[http://localhost:8000](http://localhost:8000)**

Giao diện sẽ tự động load `index.html`. Paste nội dung bài báo tiếng Việt vào ô nhập liệu để thử nghiệm.

---

## 🔬 Quy trình ML Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│  BƯỚC 1: Thu thập dữ liệu  →  Crawll.ipynb                  │
│  · Web scraping báo Lao Động theo 4 chuyên mục               │
│  · Lưu raw CSV theo từng chủ đề                              │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  BƯỚC 2: EDA & Tiền xử lý  →  base-news (1).ipynb           │
│  · Phân tích phân phối nhãn, độ dài văn bản                  │
│  · Làm sạch dữ liệu (xử lý NaN, HTML, ký tự đặc biệt)       │
│  · Tạo train/test split (85/15), thêm cột is_test            │
│  · Xuất laodong_all_news_with_split.csv                      │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  BƯỚC 3a: Word Embeddings (nghiên cứu, tùy chọn)            │
│  · GloVe  →  embedding/glove.ipynb  (PyTorch, from scratch)  │
│  · Word2Vec → embedding/word2vec.ipynb (Gensim Skip-gram)    │
│  · Embedding dim: 300, Vocab: 28.126 từ                      │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  BƯỚC 3b: Fine-tune PhoBERT  →  phobert.ipynb               │
│  · Load vinai/phobert-base                                    │
│  · Expand position embeddings (interpolation) lên 512 pos    │
│  · Fine-tune 3 epochs với AdamW, lr=2e-5, fp16               │
│  · Đạt Accuracy 96.11%, F1 96.12%                            │
│  · Lưu model.pt + config.json + metrics.json                 │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  BƯỚC 4: Triển khai  →  api.py + index.html                  │
│  · FastAPI load model một lần khi khởi động                  │
│  · Phục vụ REST API + giao diện web                          │
│  · LLM proxy ẩn HF_API_KEY khỏi client                       │
└──────────────────────────────────────────────────────────────┘
```

---

## 🛠 Công nghệ sử dụng

### Backend
| Thư viện | Mục đích |
|---|---|
| `FastAPI` | Web framework REST API |
| `Uvicorn` | ASGI server |
| `PyTorch` | Deep learning inference |
| `Transformers (HuggingFace)` | PhoBERT tokenizer & model |
| `httpx` | Async HTTP client (gọi HF LLM API) |
| `python-dotenv` | Đọc biến môi trường từ `.env` |
| `pydantic` | Validation request/response |

### Frontend
| Công nghệ | Mục đích |
|---|---|
| HTML5 / CSS3 | Giao diện web, dark-mode, animations |
| Vanilla JavaScript | Logic tương tác, gọi API |
| Fetch API | Kết nối với backend |

### ML & Research
| Thư viện | Mục đích |
|---|---|
| `vinai/phobert-base` | Pre-trained Vietnamese BERT |
| `Gensim` | Word2Vec training |
| `NumPy` | Ma trận embedding |
| `Pandas` | Xử lý dữ liệu CSV |
| `scipy.sparse` | Sparse matrix (GloVe co-occurrence) |
| `tqdm` | Progress bar training |

### Models & APIs
| Service | Mục đích |
|---|---|
| `meta-llama/Llama-3.1-8B-Instruct:cerebras` | LLM chatbot (qua HF Router) |
| `vinai/phobert-base` | Base model cho classification |
| HuggingFace Router API | Serverless LLM inference |

---

## 📁 Tài nguyên bổ sung

| File | Mô tả |
|---|---|
| `extend/vn_stopwords.txt` | Danh sách stop words tiếng Việt (~20 KB) |
| `extend/word_net_vi.json` | Vietnamese WordNet — quan hệ ngữ nghĩa từ vựng (~1.4 MB) |
| `Báo cáo.pptx` | Slide trình bày tổng quan dự án |

---

*Dự án nghiên cứu ứng dụng NLP & Xử lý ngôn ngữ tiếng Việt — Được hỗ trợ bởi **PhoBERT** (VinAI Research) & **Llama-3.1** (Meta AI).*
