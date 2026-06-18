# Fishing Detection using Vision-Language Models (VLM)

## Bachelor's Thesis Information
* **Author:** Jorge Cubero Toribio
* **Tutors:** Nelson Monzón López, Jonay Suárez Ramírez
* **Degree:** Grado en Ciencia e Ingeniería de Datos
* **School:** Escuela Ingeniería Informática
* **University:** Universidad de Las Palmas de Gran Canaria
* **Academic Year:** 2025-2026

## Project Overview
This repository contains the codebase for my Bachelor's Thesis (Trabajo de Fin de Título). The project explores the capabilities of state-of-the-art Vision-Language Models (VLMs) for detecting fishing activities in images. 

The system evaluates multiple architectures (such as Qwen-VL, BLIP-2, InstructBLIP, CLIP, Moondream, etc.) and compares their performance regarding accuracy (F1-Score) and inference speed (Average Time per Image). The objective is to determine the most efficient model and prompting strategy for deployment, specifically targeting NVIDIA Jetson edge devices.

## System Architecture: Why a Client-Server Split?
Instead of a monolithic script, this project is deliberately divided into a decoupled Client-Server architecture (`/local` and `/server`). This design decision addresses several critical engineering challenges associated with large AI models:

1. **VRAM Memory Management & OOM Prevention:** Vision-Language Models require vast amounts of GPU memory. Running an evaluation loop that sequentially loads and unloads multiple heavy models in a single script inevitably leads to memory fragmentation and Out-Of-Memory (OOM) crashes. By decoupling the architecture, the Server acts as an isolated sandbox. Models are loaded into VRAM once, kept alive to serve requests, and can be completely wiped from memory using garbage collection and cache clearing before a new architecture is loaded.

2. **Dependency Isolation (The JetPack Challenge):**
   Edge devices like NVIDIA Jetsons run on highly specific operating systems (JetPack) with delicate CUDA/TensorRT configurations. Installing dependencies for 5+ different models natively would break the host system. The Server component is fully containerized (Docker), ensuring that heavy libraries (PyTorch, Transformers, Accelerate) run in an isolated environment without polluting the Local evaluation engine.

3. **Inference Speed & Edge Simulation:**
   In a real-world scenario, the camera capturing the frames (the Edge Client) might not be the same machine executing the neural network. This architecture simulates a real production pipeline: the Local client reads the dataset and acts as the edge sensor, rapidly dispatching images via REST API to the backend Server, which performs the heavy mathematical inference and returns the text evaluation.

## Repository Structure
* **`/server`**: The containerized backend. Contains the FastAPI application, Dockerfile, requirements, and the logic for dynamic model loading/unloading on the GPU.
* **`/local`**: The client-side logic. Contains the benchmarking scripts, dataset iterators, NLP evaluation logic (using MiniLM for semantic matching), visualization tools (Seaborn/Matplotlib), and the Jetson orchestrator.
* **`/dataset`**: *(Not included in version control)* The image dataset used for benchmarking, categorized into folders indicating ground truth.

## General Requirements
* Python 3.10+
* Docker and NVIDIA Container Toolkit (for GPU acceleration)
* Network connection (for fetching Hugging Face weights and Client-Server communication)
