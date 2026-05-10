"""
upload_model.py — Upload model PhoBERT đã fine-tune lên HuggingFace Hub.

Chạy 1 lần duy nhất để upload model.pt + tokenizer files.

Cách dùng:
  1. pip install huggingface_hub
  2. huggingface-cli login   (nhập token từ https://huggingface.co/settings/tokens)
  3. python upload_model.py
"""

from huggingface_hub import HfApi, create_repo
import os

# ============================================================
# CẤU HÌNH
# ============================================================
REPO_ID = "Hieu18012005/newsmind-phobert"  # Model repository trên HF Hub
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model", "bertpho")

# Danh sách files cần upload
FILES_TO_UPLOAD = [
    "model.pt",                # ~516 MB — trọng số model chính
    "config.json",             # siêu tham số
    "metrics.json",            # kết quả đánh giá
    "vocab.txt",               # vocabulary
    "bpe.codes",               # BPE merge rules
    "tokenizer_config.json",   # cấu hình tokenizer
    "special_tokens_map.json", # special tokens
    "added_tokens.json",       # added tokens
]


def main():
    api = HfApi()

    # Tạo repo nếu chưa tồn tại
    print(f"📦 Tạo repository: {REPO_ID}")
    try:
        create_repo(REPO_ID, repo_type="model", exist_ok=True)
        print(f"✅ Repository sẵn sàng: https://huggingface.co/{REPO_ID}")
    except Exception as e:
        print(f"⚠️ Lỗi tạo repo: {e}")
        return

    # Upload từng file
    for filename in FILES_TO_UPLOAD:
        filepath = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(filepath):
            print(f"⏭️  Bỏ qua {filename} (không tìm thấy)")
            continue

        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"⬆️  Đang upload {filename} ({size_mb:.1f} MB)...")

        try:
            api.upload_file(
                path_or_fileobj=filepath,
                path_in_repo=filename,
                repo_id=REPO_ID,
                repo_type="model",
            )
            print(f"✅ {filename} — upload thành công!")
        except Exception as e:
            print(f"❌ {filename} — lỗi: {e}")

    print(f"\n🎉 Hoàn tất! Xem model tại: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
