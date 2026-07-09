import torch
from nanovllm.layers.attention import USE_FLASH_ATTN
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

print("=== nano-vllm smoke test ===")
print("GPU:", torch.cuda.get_device_name(0))
print("compute capability:", torch.cuda.get_device_capability(0))
print("USE_FLASH_ATTN:", USE_FLASH_ATTN, "(False => using SDPA Turing fallback)")

path = "/root/models/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(path)
llm = LLM(path, enforce_eager=True, tensor_parallel_size=1, max_model_len=2048)

sampling_params = SamplingParams(temperature=0.6, max_tokens=64)
raw_prompts = [
    "introduce yourself in one sentence",
    "list the first 5 prime numbers",
]
prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": p}],
        tokenize=False,
        add_generation_prompt=True,
    )
    for p in raw_prompts
]

outputs = llm.generate(prompts, sampling_params)

for rp, output in zip(raw_prompts, outputs):
    print("\n----------")
    print("Prompt:", rp)
    print("Completion:", output["text"])

print("\n=== SMOKE TEST PASSED ===")
