FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

ENV HF_HOME=/data/hf_cache

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir fastapi uvicorn python-multipart pillow

RUN pip install --no-cache-dir --upgrade transformers accelerate qwen-vl-utils einops

RUN pip install --no-cache-dir torchvision==0.21.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124

COPY server_api.py .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "server_api:app", "--host", "0.0.0.0", "--port", "8000"]