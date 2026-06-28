# Running nano-vllm on Turing GPUs (RTX 20-series) via WSL2

A companion guide to the main [README](README.md) for running nano-vllm on **NVIDIA
Turing GPUs** — compute capability 7.5 / `sm75`, e.g. RTX 2060 / 2070 / 2080 — which
are **not supported by FlashAttention-2**. The `turing-sdpa-fallback` branch adds a
PyTorch SDPA attention fallback so nano-vllm runs (correctly, if slower) on these cards.

## Why this is needed

nano-vllm's attention layer calls FlashAttention-2 (`flash_attn_varlen_func` and
`flash_attn_with_kvcache`). FlashAttention-2 kernels only run on Ampere (`sm80`) or
newer; on Turing every call raises:

```
RuntimeError: FlashAttention only supports Ampere GPUs or newer.
```

On Windows it's also impractical natively: `triton` (a hard dependency) ships no
Windows wheels, and neither does `flash-attn`. So the supported path is **WSL2**.

## What the patch changes

- **`nanovllm/layers/attention.py`** — adds `flash_attn_supported()` (returns `False`
  when the GPU's compute capability is `< 8.0`, or when `flash-attn` isn't importable)
  and a `scaled_dot_product_attention` fallback that reproduces all three attention
  modes nano-vllm needs:
  - plain causal prefill,
  - prefix-cache prefill (bottom-right causal alignment),
  - paged decode (gathers K/V from the block table).

  It keeps the **same paged KV-cache layout and the triton store kernel** — only the
  attention math is swapped. The FlashAttention path is untouched on Ampere+.
- **`nanovllm/engine/model_runner.py`** — forces `enforce_eager` when FlashAttention
  isn't usable, because CUDA graphs can't capture the fallback's per-sequence gather.

**Trade-off:** correct output, but slower — no FlashAttention and no CUDA graphs.
Expect roughly single- to low-double-digit tokens/sec on an RTX 2070 SUPER.

## Setup (Windows 11 + WSL2)

### 1. Install WSL2 + Ubuntu

In an **Administrator** PowerShell:

```powershell
wsl --install -d Ubuntu
```

Reboot if prompted, then create your Linux user when Ubuntu first launches. Confirm the
GPU is visible inside Linux (needs a recent NVIDIA driver on Windows — no driver install
inside Linux):

```bash
nvidia-smi   # should list your Turing GPU
```

### 2. Python 3.12 toolchain

Recent Ubuntu releases ship a Python that's too new for the ML wheels (torch / triton /
flash-attn). Install Python 3.12 with [uv](https://github.com/astral-sh/uv):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
uv python install 3.12
```

Also install a host C compiler (triton JIT-compiles a small CUDA helper at runtime):

```bash
sudo apt-get update && sudo apt-get install -y build-essential git
```

### 3. Clone this fork and create a venv

```bash
git clone https://github.com/<your-username>/nano-vllm.git
cd nano-vllm
git checkout turing-sdpa-fallback
uv venv --python 3.12
```

### 4. Install dependencies

Install PyTorch first (the CUDA 12.4 build pulls in a compatible `triton`):

```bash
uv pip install torch==2.6.0
```

`flash-attn` is declared as a dependency but is **never called at runtime on Turing**.
To satisfy the dependency without compiling it from source (which needs the full CUDA
toolkit), install a matching **prebuilt Linux wheel** — pick the one matching your
Python / torch / ABI from the
[flash-attn releases](https://github.com/Dao-AILab/flash-attention/releases). For
Python 3.12 + torch 2.6 + cxx11abi=False:

```bash
uv pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
```

Then install nano-vllm itself (editable, so code edits apply without reinstalling):

```bash
uv pip install -e .
```

### 5. Download a small model and run

8 GB of VRAM comfortably fits a sub-1B model. Download one locally (nano-vllm loads from
a local directory):

```bash
uv pip install huggingface_hub
uv run hf download Qwen/Qwen3-0.6B --local-dir ./models/Qwen3-0.6B
```

Minimal run script (`run_turing.py`):

```python
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

path = "./models/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(path)
# enforce_eager is auto-forced on Turing, but setting it is harmless
llm = LLM(path, enforce_eager=True, tensor_parallel_size=1, max_model_len=2048)

prompts = [tok.apply_chat_template(
    [{"role": "user", "content": "list the first 5 prime numbers"}],
    tokenize=False, add_generation_prompt=True)]

for out in llm.generate(prompts, SamplingParams(temperature=0.6, max_tokens=64)):
    print(out["text"])
```

```bash
.venv/bin/python run_turing.py
```

You should see coherent generation. The first run is a few seconds slower while triton
JIT-compiles its kernel.

## Verifying the fallback is active

```python
from nanovllm.layers.attention import USE_FLASH_ATTN
print("USE_FLASH_ATTN:", USE_FLASH_ATTN)   # False on Turing => SDPA fallback in use
```

## Caveats

- **Speed:** SDPA + eager mode, no CUDA graphs. Fine for learning and experimentation,
  not for serving.
- **VRAM:** 8 GB fits small models (≲ ~1.5B params). Keep `max_model_len` modest.
- **Eager mode** is enabled automatically on Turing; CUDA-graph capture is skipped.
- This patch lives on the `turing-sdpa-fallback` branch only — it is not part of upstream
  nano-vllm. Re-apply it if you rebase onto a newer upstream.

## What you still learn

nano-vllm uses FlashAttention as a black-box kernel, so swapping it for SDPA costs you
nothing conceptually — the FlashAttention algorithm lives in the compiled `flash-attn`
library, not in nano-vllm. Everything nano-vllm actually teaches stays intact: the
scheduler, paged KV cache + block manager, prefix caching, continuous batching, the
sampler, weight loading, and tensor parallelism.
