# nano-vllm — Handover / Onboarding Notes

Context doc for continuing work on this machine. Goal: **run, test, and learn nano-vllm end to end.**

## 1. What's been done

- Installed nano-vllm in **WSL2 Ubuntu** (native Windows is impractical: no Windows wheels for `triton`/`flash-attn`).
- GPU is an **RTX 2070 SUPER (Turing, sm75, 8 GB)**. FlashAttention-2 does NOT run on Turing (`only supports Ampere GPUs or newer`).
- **Patched the attention layer** to fall back to PyTorch SDPA on Turing — see section 5. nano-vllm now runs end to end (smoke test passes, coherent Qwen3-0.6B output, ~13 tok/s prefill / ~8 tok/s decode).
- Forked to **github.com/longjichen1/nano-vllm**, branch `turing-sdpa-fallback` (code patch + `README_TURING.md`). The clone here has a `fork` git remote pointing to it.

## 2. Environment & layout (everything under /root, default WSL user = root)

| What | Path |
|---|---|
| Code (patched clone) | `/root/nano-vllm/src` |
| Python 3.12 venv (uv) | `/root/nano-vllm/.venv` |
| Model | `/root/models/Qwen3-0.6B` |
| Run script | `/root/nano-vllm/example_turing.py` |
| VS Code workspace | open `/root/nano-vllm` (WSL-remote) |

Stack: torch 2.6.0+cu124, triton 3.2.0, flash-attn 2.7.4.post1 (installed but unused on Turing), transformers. Needs `build-essential` (gcc) for triton's JIT.

## 3. How to run / edit

- **VS Code (WSL-remote):** open `/root/nano-vllm`. Press `F5` to run `example_turing.py`, or use the terminal alias `run-nanovllm`. `justMyCode:false` is set so you can step into `nanovllm` library code while debugging.
- **Terminal:** `cd /root/nano-vllm/src && /root/nano-vllm/.venv/bin/python /root/nano-vllm/example_turing.py`
- Editable install → code edits apply immediately, no reinstall.
- To try prompts: edit the `raw_prompts` list in `example_turing.py`.

## 4. Learning path (codebase is ~1,200 lines; read in dependency order)

**Start:** `nanovllm/engine/llm_engine.py` — read `generate()` then `step()`. The spine is
`step(): schedule() -> run(model) -> postprocess()`.

Then:
1. `sampling_params.py` (temperature/max_tokens/ignore_eos) -> `engine/sequence.py` (the `Sequence` struct: token_ids, block_table, prefill/decode state, the counters).
2. **Heart (the real vLLM ideas):** `engine/scheduler.py` (`schedule()` = prefill-first batching, chunked prefill, continuous batching, preemption; `postprocess()` = append token, EOS/max_tokens, free blocks) and `engine/block_manager.py` (paged KV-cache allocator + **prefix caching** via xxhash block hashing + ref-counting).
3. **GPU bridge:** `engine/model_runner.py` (`prepare_prefill`/`prepare_decode` build cu_seqlens / slot_mapping / block_tables; `allocate_kv_cache` memory-profiles; `capture_cudagraph`) + `utils/context.py`.
4. **Model/layers last:** `models/qwen3.py`, `layers/attention.py` (the patched file), `layers/linear.py` (tensor-parallel), then rotary/layernorm/activation/embed_head/sampler, `utils/loader.py`.

**Mental model:** prefill = whole prompt at once (compute-bound, many tokens/seq); decode = 1 token/seq/step (bandwidth-bound). Paged KV cache makes batching memory-efficient. scheduler + block_manager exist to keep the GPU busy juggling many sequences in limited VRAM.

**Best hands-on exercise:** add to `LLMEngine.step()` after `schedule()`:
```python
print(f"[step] prefill={is_prefill} nseqs={len(seqs)} "
      f"sched={[s.num_scheduled_tokens for s in seqs]} "
      f"cached={[s.num_cached_tokens for s in seqs]} "
      f"blocks={[len(s.block_table) for s in seqs]} "
      f"waiting={len(self.scheduler.waiting)} running={len(self.scheduler.running)}")
```
Run it and watch one big prefill step, then many single-token decode steps, and the block_table grow. That trace IS the engine.

## 5. The Turing patch (what changed vs upstream)

- `nanovllm/layers/attention.py`: adds `flash_attn_supported()` (False when device capability < 8.0 OR flash-attn unimportable) and a `scaled_dot_product_attention` fallback reproducing all three modes — plain causal prefill (`is_causal=True`), prefix-cache prefill (explicit bottom-right mask), paged decode (`Lq==1`, gather K/V from the block_table, no mask). Keeps the same paged KV-cache layout and the triton store kernel. Flash path unchanged on Ampere+.
- `nanovllm/engine/model_runner.py`: forces `enforce_eager` when flash-attn isn't usable (CUDA graphs can't capture the per-sequence gather loop).

## 6. Current understanding & gaps (as of handover)

Solid: the control plane — `step()` loop, waiting/running queues, Sequence lifecycle, prefill-vs-decode, the counters, sampling (Gumbel trick + per-seq temperature broadcasting), causal masking across the three cases.

Biggest remaining gaps, in priority order:
1. **`block_manager.py` / paged KV cache + prefix caching** — not yet read. The core idea; highest leverage.
2. **`model_runner.prepare_prefill/decode`** — the GPU bridge (how slot_mapping / cu_seqlens / block_tables get built). Understood conceptually (packed 1-D tensors), not concretely.
3. KV-cache **write** path (`store_kvcache` + slot_mapping).
4. Model internals (`qwen3.py`): RoPE, GQA, RMSNorm.
5. Tensor parallelism (`linear.py`), CUDA graphs (disabled on Turing).

Recommended next step: guided `block_manager.py` walkthrough — set a breakpoint in `schedule()`, run two **identical** prompts, watch prefix caching skip recomputation on the second.
