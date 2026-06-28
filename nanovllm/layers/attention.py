import torch
from torch import nn
import torch.nn.functional as F
import triton
import triton.language as tl

from nanovllm.utils.context import get_context

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    _FLASH_ATTN_IMPORTED = True
except ImportError:
    flash_attn_varlen_func = flash_attn_with_kvcache = None
    _FLASH_ATTN_IMPORTED = False


def flash_attn_supported() -> bool:
    """FlashAttention-2 kernels only run on Ampere (sm80) GPUs or newer.

    On older GPUs (e.g. Turing / sm75, like the RTX 20-series) flash-attn imports
    fine but every kernel raises 'FlashAttention only supports Ampere GPUs or newer'
    at runtime, so we fall back to a PyTorch SDPA implementation instead.
    """
    if not _FLASH_ATTN_IMPORTED or not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8


USE_FLASH_ATTN = flash_attn_supported()


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


def sdpa_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    """Single-sequence attention via torch SDPA (Turing-compatible flash-attn replacement).

    q: (Lq, num_heads, head_dim)
    k, v: (Lk, num_kv_heads, head_dim)
    returns: (Lq, num_heads, head_dim)

    Handles the three cases nano-vllm needs, matching flash-attn's bottom-right
    causal alignment (query token i lines up with the LAST keys, not the first):
      * decode      (Lq == 1):  attend to all Lk cached keys, no mask
      * full prefill(Lq == Lk): standard causal
      * prefix-cache prefill (1 < Lq < Lk): causal offset by (Lk - Lq)
    """
    Lq, num_heads, _ = q.shape
    Lk, num_kv_heads, _ = k.shape
    if num_heads != num_kv_heads:                       # GQA: expand kv heads to match q heads
        rep = num_heads // num_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    q = q.transpose(0, 1)                               # (H, Lq, D)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)
    attn_mask = None
    is_causal = False
    if Lq == 1:
        pass                                            # decode: full attention over the cache
    elif Lq == Lk:
        is_causal = True                                # plain causal prefill
    else:
        # bottom-right causal: query i (0..Lq-1) may attend key j iff j <= (Lk - Lq) + i
        qpos = torch.arange(Lq, device=q.device).unsqueeze(1)
        kpos = torch.arange(Lk, device=q.device).unsqueeze(0)
        attn_mask = kpos <= (Lk - Lq) + qpos            # bool (Lq, Lk), True == attend
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=scale)
    return o.transpose(0, 1).contiguous()               # (Lq, H, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def _gather_kv(self, block_table: torch.Tensor, length: int):
        """Gather a contiguous [0, length) slice of K/V for one sequence out of the
        paged cache, following its block_table. Mirrors the slot math in store_kvcache."""
        k_flat = self.k_cache.view(-1, self.num_kv_heads, self.head_dim)
        v_flat = self.v_cache.view(-1, self.num_kv_heads, self.head_dim)
        block_size = self.k_cache.size(1)
        pos = torch.arange(length, device=k_flat.device)
        blocks = block_table[pos // block_size].to(torch.long)
        slots = blocks * block_size + (pos % block_size)
        return k_flat.index_select(0, slots), v_flat.index_select(0, slots)

    def _forward_sdpa(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, context):
        outputs = []
        if context.is_prefill:
            cu_q = context.cu_seqlens_q.tolist()
            cu_k = context.cu_seqlens_k.tolist()
            use_cache = context.block_tables is not None    # prefix cache: read K/V from paged cache
            for i in range(len(cu_q) - 1):
                qi = q[cu_q[i]:cu_q[i + 1]]
                if use_cache:
                    ki, vi = self._gather_kv(context.block_tables[i], cu_k[i + 1] - cu_k[i])
                else:
                    ki = k[cu_k[i]:cu_k[i + 1]]
                    vi = v[cu_k[i]:cu_k[i + 1]]
                outputs.append(sdpa_attention(qi, ki, vi, self.scale))
        else:                                               # decode: one query token per sequence
            context_lens = context.context_lens.tolist()
            for i in range(q.size(0)):
                ki, vi = self._gather_kv(context.block_tables[i], context_lens[i])
                outputs.append(sdpa_attention(q[i:i + 1], ki, vi, self.scale))
        return torch.cat(outputs, dim=0)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if not USE_FLASH_ATTN:
            return self._forward_sdpa(q, k, v, context)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True)
        return o
