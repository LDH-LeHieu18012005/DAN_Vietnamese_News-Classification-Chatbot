FROM python:3.10-slim

# Tạo user non-root (yêu cầu của HF Spaces)
RUN useradd -m -u 1000 user

# Cài đặt system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements trước (tận dụng Docker cache)
COPY requirements.txt .

# Install Python dependencies (PyTorch CPU-only)
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Copy source code
COPY api.py .
COPY index.html .
COPY .env* ./

# Copy model tokenizer files (nhẹ, < 2MB tổng)
COPY model/bertpho/vocab.txt model/bertpho/vocab.txt
COPY model/bertpho/bpe.codes model/bertpho/bpe.codes
COPY model/bertpho/config.json model/bertpho/config.json
COPY model/bertpho/metrics.json model/bertpho/metrics.json
COPY model/bertpho/tokenizer_config.json model/bertpho/tokenizer_config.json
COPY model/bertpho/special_tokens_map.json model/bertpho/special_tokens_map.json
COPY model/bertpho/added_tokens.json model/bertpho/added_tokens.json

# Tạo thư mục cache cho model weights
RUN mkdir -p /tmp/hf_cache && chown -R user:user /tmp/hf_cache
RUN chown -R user:user /app

# Set environment
ENV TRANSFORMERS_CACHE=/tmp/hf_cache
ENV HF_HOME=/tmp/hf_cache
ENV HOME=/home/user

# Chuyển sang user non-root
USER user

# Expose port 7860 (mặc định HF Spaces)
EXPOSE 7860

# Start server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
