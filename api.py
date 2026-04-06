import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import httpx
from transformers import AutoTokenizer, AutoModel
from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY", "")

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# =========================
# CONFIG
# =========================
MODEL_PATH = r"C:/Users/DELL/Desktop/The News/model/bertpho/model.pt"
BASE_MODEL = "vinai/phobert-base"
MAX_LEN = 512
HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
HF_MODEL_ROUTED = "meta-llama/Llama-3.1-8B-Instruct:cerebras"
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"

label_map = {0: "cong-nghe", 1: "kinh-doanh", 2: "the-gioi", 3: "the-thao"}
label_map_vi = {
    "cong-nghe": "Công nghệ",
    "kinh-doanh": "Kinh doanh",
    "the-gioi": "Thế giới",
    "the-thao": "Thể thao",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# EXPAND POSITION EMBEDDING
# =========================
def expand_position_embeddings(model, new_max_length=512):
    pos_emb_layer = model.embeddings.position_embeddings
    old_weight = pos_emb_layer.weight.data
    old_max, hidden = old_weight.shape
    new_max = new_max_length + 2
    if old_max == new_max:
        return
    new_weight = F.interpolate(
        old_weight.T.unsqueeze(0), size=new_max, mode="linear", align_corners=False
    ).squeeze(0).T
    new_emb = nn.Embedding(new_max, hidden)
    new_emb.weight = nn.Parameter(new_weight)
    model.embeddings.position_embeddings = new_emb
    model.embeddings.register_buffer(
        "position_ids", torch.arange(new_max).expand((1, -1)), persistent=False
    )
    model.embeddings.register_buffer(
        "token_type_ids", torch.zeros((1, new_max), dtype=torch.long), persistent=False
    )


# =========================
# MODEL
# =========================
class PhoBERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        expand_position_embeddings(self.bert, 512)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


# =========================
# LOAD MODEL (1 lần khi khởi động)
# =========================
print("Đang load model PhoBERT...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
model = PhoBERTClassifier(BASE_MODEL, 4)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()
print(f"Model sẵn sàng trên {device}")


# =========================
# PREDICT
# =========================
def predict(text: str):
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    label = label_map[pred]
    confidence = round(probs[0][pred].item(), 4)
    all_probs = {
        label_map[i]: round(probs[0][i].item() * 100, 2) for i in range(4)
    }
    return label, confidence, all_probs


# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="NewsMind API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="."), name="static")


# =========================
# SCHEMAS
# =========================
class TextIn(BaseModel):
    text: str


class ChatIn(BaseModel):
    messages: list[dict]
    system: str = ""


# =========================
# ROUTES
# =========================
@app.get("/")
def root():
    return FileResponse("index.html")


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}


@app.get("/config")
def config():
    # Không trả key ra client nữa — chỉ báo có hay không
    return {"has_hf_key": bool(HF_API_KEY)}


@app.post("/classify_text")
def classify(body: TextIn):
    if not body.text.strip():
        return {"error": "Text rỗng"}

    label, confidence, all_probs = predict(body.text)

    words = body.text.strip().split()
    word_count = len(words)
    read_time = max(1, round(word_count / 200))

    if confidence >= 0.80:
        tier = "HIGH"
    elif confidence >= 0.55:
        tier = "MED"
    else:
        tier = "LOW"

    sentences = [s.strip() for s in body.text.replace("!", ".").replace("?", ".").split(".") if len(s.strip()) > 20]
    top_sentences = ". ".join(sentences[:2]) + ("." if sentences else "")

    return {
        "label": label,
        "label_vi": label_map_vi[label],
        "confidence": confidence,
        "confidence_tier": tier,
        "all_probs": all_probs,
        "word_count": word_count,
        "read_time": read_time,
        "top_sentences": top_sentences,
    }


@app.post("/classify_batch")
def classify_batch(items: list[TextIn]):
    results = []
    for i, item in enumerate(items):
        if not item.text.strip():
            results.append({"index": i, "error": "Text rỗng"})
            continue
        label, confidence, all_probs = predict(item.text)
        words = item.text.strip().split()
        tier = "HIGH" if confidence >= 0.80 else ("MED" if confidence >= 0.55 else "LOW")
        results.append({
            "index": i,
            "label": label,
            "label_vi": label_map_vi[label],
            "confidence": confidence,
            "confidence_tier": tier,
            "all_probs": all_probs,
            "word_count": len(words),
            "preview": item.text[:80] + ("..." if len(item.text) > 80 else ""),
        })
    return results


@app.post("/chat")
async def chat(body: ChatIn):
    """Proxy LLM call qua backend — tránh CORS và ẩn API key."""
    if not HF_API_KEY:
        return {"error": "Chưa có HF_API_KEY trong file .env"}

    msgs = []
    if body.system:
        msgs.append({"role": "system", "content": body.system})
    msgs.extend(body.messages)

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                HF_API_URL,
                headers={
                    "Authorization": f"Bearer {HF_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": HF_MODEL_ROUTED,
                    "messages": msgs,
                    "max_tokens": 500,
                    "temperature": 0.65,
                },
            )
        if not r.is_success:
            try:
                err = r.json()
                if isinstance(err, dict):
                    msg = err.get("error", {})
                    if isinstance(msg, dict):
                        msg = msg.get("message", str(err))
                    else:
                        msg = str(msg)
                else:
                    msg = str(err)
            except Exception:
                msg = r.text[:300]
            return {"error": f"HF API {r.status_code}: {msg}"}
        try:
            data = r.json()
            reply = data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return {"error": f"Parse lỗi response HF: {e} | Raw: {r.text[:200]}"}
        return {"reply": reply}
    except httpx.TimeoutException:
        return {"error": "Timeout khi gọi HF API (>30s). Thử lại sau."}
    except Exception as e:
        return {"error": str(e)}