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
[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue)](https://www.kaggle.com/datasets/ldhhieu18/final-data)
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
