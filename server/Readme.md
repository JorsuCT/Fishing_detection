# Server API: VLM Inference Backend

**Tags / Tech Stack:** `FastAPI` | `Docker` | `PyTorch` | `Hugging Face Transformers` | `CUDA` | `Accelerate`

This directory contains the backend infrastructure for the Fishing Detection project. It provides a FastAPI-based web server that dynamically loads Vision-Language Models (VLMs) into the GPU memory and processes image-text inference requests.

## Features
* **Dynamic Model Loading:** Automatically handles loading and unloading of models from VRAM to prevent Out-Of-Memory (OOM) errors during extensive benchmarking.
* **Optimized for Heavy Workloads:** Uses `accelerate`, `torch.bfloat16`, and specific memory management strategies (e.g., garbage collection, empty CUDA cache).
* **Dockerized Environment:** Fully containerized with CUDA support to avoid JetPack dependency conflicts on the host system.
* **Broad Model Support:** Capable of handling Qwen3-VL, BLIP-2, InstructBLIP, CLIP, Moondream, VideoLLaMA3, and more.

## Prerequisites
* NVIDIA GPU with appropriate drivers.
* Docker installed with `nvidia-docker2` (NVIDIA Container Toolkit).

## Setup and Deployment

### 1. Build the Docker Image
Navigate to this directory and build the container. We recommend using `--no-cache` if you update dependencies to ensure PyTorch and specific computer vision libraries are correctly compiled.

```bash
docker build -t server_api .
```

### 2. Run the Container
Run the server mapping the required ports and mounting the Hugging Face cache volumes. This prevents redownloading multi-gigabyte models on every restart.

```bash
docker run -d \
  --name vlm_api \
  --gpus all \
  -p 8000:8000 \
  --shm-size=8g \
  -e HF_HUB_DISABLE_SPACE_CHECK=1 \
  -e TMPDIR=/data/tmp \
  -e HF_HOME=/data/hf_cache \
  -e HF_HUB_CACHE=/data/hf_cache \
  -e TRANSFORMERS_CACHE=/data/hf_cache \
  -v /absolute/path/to/your/hf_cache:/data/hf_cache \
  -v /absolute/path/to/your/tmp:/data/tmp \
  server_api
```

### 3. API Usage
The server exposes a single main endpoint for inference:
* **`POST /infer`**: Accepts an image file (`file`) and a form-data payload containing the `model` name and the `prompt`. It returns a JSON object with the model's generated `caption`.

## Architecture Details
The script (`server_api.py`) utilizes a lazy-loading strategy. The AI model is only injected into VRAM upon the first API request. If a request for a different model architecture arrives, the server safely unloads the current model, clears the CUDA cache completely, and loads the newly requested architecture.
