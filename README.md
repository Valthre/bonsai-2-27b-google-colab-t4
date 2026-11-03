# Bonsai 2 27B on Free Google Colab (Tesla T4)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Valthre/bonsai-2-27b-google-colab-t4/blob/main/bonsai_2_27b_colab_t4_optimized_EN-US.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Model](https://img.shields.io/badge/Model-Bonsai--2--27B--PTQ1__0-blueviolet)](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
[![Hardware](https://img.shields.io/badge/GPU-Tesla%20T4%20(15GB)-green)](https://www.nvidia.com/en-us/data-center/tesla-t4/)
[![Speed](https://img.shields.io/badge/Speed-30.24%20tok%2Fs-brightgreen)](#-empirical-benchmarks)

A production-grade, zero-OOM Google Colab pipeline to run **PrismML's Ternary Bonsai 2 27B** reasoning model on a **100% free Tesla T4 GPU (15.3 GB)** at **~30 tokens/second**, complete with OpenAI-compatible API exposure via Cloudflare Tunnel.

---

## Overview

Running 27B-class language models on budget hardware has traditionally been constrained by VRAM limitations (requiring >54 GB in FP16 or >17 GB in 4-bit quantizations). 

**Bonsai 2 27B** breaks this barrier using **online Hadamard-transformed ternary quantization** (~1.75 bits per weight), compressing the full 27-billion-parameter base (Qwen3.8-27B) down to just **5.95 GB** while preserving **~96.3% of its original FP16 reasoning intelligence**.

This repository provides an automated, crash-proof pipeline specifically engineered to overcome Google Colab free-tier constraints (12.7 GB System RAM, CUDA 13 driver / CUDA 12 runtime discrepancies, and internal proxy port conflicts).

---

## Empirical Benchmarks (Measured on Tesla T4)

*Benchmark performed on Google Colab Free Tier (Tesla T4 15.3 GB, 2 vCPUs, 12.7 GB RAM).*

| Metric | Measured Value | Theoretical Headroom / Context |
| :--- | :---: | :--- |
| **Generation Throughput** | **30.24 tok/s** | Sustained rate across a 1,674-token reasoning prompt |
| **VRAM Footprint** | **6.4 GB / 15.0 GB** | **8.6 GB free** (allows context expansion up to 32k tokens) |
| **Host System RAM** | **2.6 GB / 12.7 GB** | Zero risk of Colab VM eviction / OOM container kill |
| **Model Weight Size** | **5.95 GB** (`PTQ1_0`) | 88.9% reduction compared to original FP16 (54 GB) |
| **Reasoning Retention** | **96.3%** (GSM8k) | Preserves full `<think>` chain-of-thought trace |

---

## Key Architectural Fixes Over Existing Implementations

1. **Zero-RAM Sequential Streaming (`curl -fL`):**  
   Replaces Python-based downloads (`huggingface_hub` XET chunk reassembly) that previously exhausted Colab's 12.7 GB System RAM, streaming model weights directly to disk with `<10 MB` RAM footprint.
2. **Dynamic CUDA 12/13 & cuBLAS Runtime Linker:**  
   Resolves the silent fallback to CPU caused by Colab's default CUDA 13 environment lacking `libcudart.so.12` and `libcublas.so.12` shared libraries required by the PrismML inference engine.
3. **Colab-Safe Port Binding (8081):**  
   Avoids port 8080 and eliminates destructive `fuser -k` commands that disconnect Google Colab's internal frontend-backend WebSocket proxies.
4. **Hardware-Optimized Quantized KV Cache:**  
   Uses 8-bit key/value cache quantization (`-ctk q8_0 -ctv q8_0`) and Flash Attention (`-fa on`), reducing KV memory costs to only ~32 KB per token.

---

## Quickstart Guide

### 1. Launch in Google Colab
Click the badge below to open the notebook in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Valthre/bonsai-2-27b-google-colab-t4/blob/main/bonsai_2_27b_colab_t4_optimized_EN-US.ipynb)

*Ensure your runtime is configured to use a GPU: `Runtime > Change runtime type > T4 GPU`.*

### 2. Execution Pipeline (4 Modular Cells)

- **Cell 1 — Environment Initialization:** Validates GPU availability and creates clean workspace directories.
- **Cell 2 — Engine & Model Download:** Fetches PrismML's custom `llama-server` binary with Hadamard acceleration, standalone `cloudflared`, and streams the `PTQ1_0` weights (~5.95 GB).
- **Cell 3 — Server Boot & Public Tunnel:** Starts the inference engine on port 8081 with automated CUDA library linking and generates a public HTTPS endpoint (`https://*.trycloudflare.com`).
- **Cell 4 — Interactive Streaming Chat:** An interactive in-notebook chat client with live token/second metrics and automatic demultiplexing of reasoning traces (`<think>`).

---

## API Client Integration (OpenAI-Compatible)

Once **Cell 3** outputs your public URL, you can interface with Bonsai 2 27B using standard OpenAI SDKs or third-party frontends (e.g., SillyTavern, OpenWebUI, LibreChat, Cursor, Continue.dev):

### Python (OpenAI Official SDK)
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://YOUR-SUBDOMAIN.trycloudflare.com/v1",
    api_key="not-needed"  # Local server does not require authentication
)

response = client.chat.completions.create(
    model="bonsai-2-27b",
    messages=[
        {"role": "system", "content": "You are a precise technical AI assistant."},
        {"role": "user", "content": "Explain how orthogonal Hadamard transforms mitigate outlier activations in ternary quantization."}
    ],
    temperature=1.0,
    top_p=0.95,
    extra_body={"chat_template_kwargs": {"reasoning_effort": "medium"}}
)

print(response.choices[0].message.content)
```

### cURL
```bash
curl https://YOUR-SUBDOMAIN.trycloudflare.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bonsai-2-27b",
    "messages": [{"role": "user", "content": "Write a quick Python script to calculate Fibonacci numbers."}],
    "temperature": 1.0,
    "top_p": 0.95
  }'
```

---

## Upstream References & Credits

- **Model Weights & Card:** [`prism-ml/Ternary-Bonsai-2-27B-gguf`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) by [PrismML](https://github.com/PrismML-Eng).
- **Inference Fork & Kernels:** [`PrismML-Eng/llama.cpp`](https://github.com/PrismML-Eng/llama.cpp) (branch `prism`).
- **Launcher Reference:** [`PrismML-Eng/Bonsai-demo`](https://github.com/PrismML-Eng/Bonsai-demo).
- **Base Architecture:** Qwen3.8-27B by the [Qwen Team / Alibaba Cloud](https://github.com/QwenLM).
- **Core Engine:** [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) by Georgi Gerganov and contributors.
- **Tunneling Infrastructure:** [Cloudflare cloudflared](https://github.com/cloudflare/cloudflared).

---

## License & Compliance

This repository and its customized Colab notebook are distributed under the **[MIT License](LICENSE)**. 

All upstream weights and binaries are utilized strictly in accordance with their respective permissive licenses:
- **Bonsai 2 GGUF Weights:** Apache 2.0
- **PrismML llama.cpp fork:** MIT License
- **cloudflared:** Apache 2.0
- **Qwen Base Weights:** Apache 2.0

> *Engineered & debugged with the help of an AI.*