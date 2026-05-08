# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB5 — Merge + Deploy + GGUF
#
# **Stack:** Unsloth `merge_and_unload` + `save_pretrained_gguf(quantization='Q4_K_M')`
# + llama-cpp-python smoke test.
# Maps to deck §7.1 lab brief: "merge adapter, quantize GGUF, serve với vLLM".
#
# > **Mục tiêu:** export the SFT+DPO adapter as a deployable GGUF Q4_K_M file
# > (~1.5 GB on 3B / ~4 GB on 7B), then smoke-test it through llama-cpp-python.
# > Final cell shows the optional vLLM serving command (BigGPU only).

# %% [markdown]
# ## 0. Setup

# %%
import os
import json
from pathlib import Path

COMPUTE_TIER = os.environ.get("COMPUTE_TIER", "T4").upper()
BASE_MODEL = (
    "unsloth/Qwen2.5-3B-bnb-4bit" if COMPUTE_TIER == "T4"
    else "unsloth/Qwen2.5-7B-bnb-4bit"
)
MAX_LEN = 512 if COMPUTE_TIER == "T4" else 1024

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DPO_PATH = REPO_ROOT / "adapters" / "dpo"
MERGED_PATH = REPO_ROOT / "adapters" / "merged-fp16"
GGUF_DIR = REPO_ROOT / "gguf"
MERGED_PATH.mkdir(parents=True, exist_ok=True)
GGUF_DIR.mkdir(parents=True, exist_ok=True)

assert DPO_PATH.exists(), "NB3 must run first"

print(f"COMPUTE_TIER:    {COMPUTE_TIER}")
print(f"DPO adapter:     {DPO_PATH}")
print(f"merged output:   {MERGED_PATH}")
print(f"GGUF output:     {GGUF_DIR}")

# %%
import torch

assert torch.cuda.is_available()

# %% [markdown]
# ## 1. Load DPO model + merge adapter

# %%
# CRITICAL: load FP16 base (not bnb-4bit). Merging on bnb-4bit produces packed
# NF4 storage that GGUF converter refuses ("Quant method bitsandbytes not supported").
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

FP16_BASE = "Qwen/Qwen2.5-3B" if COMPUTE_TIER == "T4" else "Qwen/Qwen2.5-7B"

base = AutoModelForCausalLM.from_pretrained(
    FP16_BASE,
    torch_dtype=torch.float16,
    device_map="cuda:0",
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(FP16_BASE)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.chat_template is None:
    try:
        from unsloth.chat_templates import get_chat_template
        tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    except Exception:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        )

# Stage 1: apply + merge SFT
SFT_PATH = REPO_ROOT / "adapters" / "sft-mini"
model = PeftModel.from_pretrained(base, str(SFT_PATH))
model = model.merge_and_unload()
print("SFT merged into FP16 base")

# Stage 2: apply DPO (will be merged in §2)
model = PeftModel.from_pretrained(model, str(DPO_PATH))
print(f"DPO adapter loaded from {DPO_PATH}")

# %% [markdown]
# > **Note:** The DPO adapter trained in NB3 stacks on top of SFT. To get a fully
# > aligned merged model, we apply both adapters before merging. Unsloth's
# > `save_pretrained_merged` handles the SFT + DPO + base merge in one shot.

# %% [markdown]
# ## 2. Save merged FP16 weights
#
# `save_pretrained_merged(method="merged_16bit")` produces a HuggingFace-format
# directory you can either upload to HF Hub directly OR feed into the GGUF
# converter in step 3.

# %%
merged = model.merge_and_unload()
merged.save_pretrained(
    str(MERGED_PATH),
    safe_serialization=True,
    save_original_format=False,
)
tokenizer.save_pretrained(str(MERGED_PATH))
print(f"Saved merged FP16 to {MERGED_PATH}")

# Verify weight dtype on disk — must be float16/bfloat16, NOT uint8
from safetensors import safe_open
shards = sorted(MERGED_PATH.glob("model*.safetensors"))
with safe_open(shards[0], framework="pt") as f:
    for k in [k for k in f.keys() if "weight" in k][:3]:
        t = f.get_tensor(k)
        print(f"  {k}  dtype={t.dtype}  shape={tuple(t.shape)}")

import gc
del model, merged, base
gc.collect()
torch.cuda.empty_cache()

# %% [markdown]
# ## 3. Quantize to GGUF Q4_K_M
#
# Q4_K_M is the sweet spot: ~4× compression vs FP16, minimal quality loss.
# Unsloth wraps llama.cpp's `quantize` binary — first run downloads + compiles
# llama.cpp (~3 min) then quantizes (~30 s).

# %%
# Direct subprocess: bypass Unsloth's GGUF wrapper entirely (it tries to reload
# merged dir with bnb which re-quantizes). Call llama.cpp converter + quantize bin.
import subprocess, os
from pathlib import Path

TMP_DIR = Path("/tmp/gguf-work")
TMP_DIR.mkdir(exist_ok=True)
GGUF_DIR.mkdir(parents=True, exist_ok=True)

f16_path = TMP_DIR / "merged.F16.gguf"
q4_path = GGUF_DIR / "lab22-dpo-Q4_K_M.gguf"

print("Step 1: HF FP16 -> GGUF F16")
result = subprocess.run([
    "/usr/bin/python3",
    "/root/.unsloth/llama.cpp/unsloth_convert_hf_to_gguf.py",
    "--outfile", str(f16_path),
    "--outtype", "f16",
    str(MERGED_PATH),
], capture_output=True, text=True)
if result.returncode != 0:
    print("STDERR:", result.stderr[-2000:])
    raise RuntimeError(f"hf->gguf failed: rc={result.returncode}")
print(f"  F16 GGUF: {f16_path.stat().st_size / 1e6:.1f} MB")

print("\nStep 2: GGUF F16 -> Q4_K_M")
quantize_bin = next(
    (p for p in [
        "/root/.unsloth/llama.cpp/build/bin/llama-quantize",
        "/root/.unsloth/llama.cpp/llama-quantize",
        "/root/.unsloth/llama.cpp/build/llama-quantize",
    ] if Path(p).exists()),
    None,
)
assert quantize_bin, "llama-quantize binary not found"

result = subprocess.run(
    [quantize_bin, str(f16_path), str(q4_path), "Q4_K_M"],
    capture_output=True, text=True,
)
if result.returncode != 0:
    print("STDERR:", result.stderr[-2000:])
    raise RuntimeError(f"quantize failed: rc={result.returncode}")
print(f"  Q4_K_M GGUF: {q4_path.stat().st_size / 1e6:.1f} MB")

f16_path.unlink()
print(f"\nFinal GGUF in {GGUF_DIR}:")
for p in sorted(GGUF_DIR.glob("*.gguf")):
    print(f"  {p.name}  {p.stat().st_size / 1e6:.1f} MB")

# %% [markdown]
# ### 3a. Optional — additional quantization tiers (for the +3 rigor add-on)

# %%
# Uncomment if you want Q5_K_M + Q8_0 too (~2× total disk space).
# Each adds ~30s for an extra GGUF file.
#
# model.save_pretrained_gguf(str(GGUF_DIR), tokenizer, quantization_method="q5_k_m")
# model.save_pretrained_gguf(str(GGUF_DIR), tokenizer, quantization_method="q8_0")

# %%
import os

print("GGUF files:")
for p in sorted(GGUF_DIR.iterdir()):
    if p.suffix == ".gguf":
        size_mb = p.stat().st_size / 1e6
        print(f"  {p.name:50s}  {size_mb:>8.1f} MB")

del model
gc.collect()
torch.cuda.empty_cache()

# %% [markdown]
# ## 4. Smoke test with llama-cpp-python

# %%
from llama_cpp import Llama

# Find the Q4_K_M GGUF
gguf_files = list(GGUF_DIR.glob("*Q4_K_M*.gguf")) + list(GGUF_DIR.glob("*q4_k_m*.gguf"))
assert gguf_files, "No Q4_K_M GGUF found — step 3 may have failed"
gguf_path = gguf_files[0]
print(f"Loading: {gguf_path.name}")

# n_gpu_layers=-1 offloads all layers to GPU if compiled with CUDA/Metal/Vulkan
llm = Llama(
    model_path=str(gguf_path),
    n_ctx=MAX_LEN,
    n_gpu_layers=-1,           # all layers on GPU; falls back to CPU if no GPU compile
    verbose=False,
)
print("Loaded.")

# %% [markdown]
# ### 4a. Smoke prompt + response (deliverable: `06-gguf-smoke.png`)

# %%
SMOKE_PROMPT = "Giải thích ngắn gọn (3 câu) cách thuật toán Bubble sort hoạt động."

response = llm.create_chat_completion(
    messages=[{"role": "user", "content": SMOKE_PROMPT}],
    max_tokens=200,
    temperature=0.0,
)

print(f"PROMPT:\n  {SMOKE_PROMPT}\n")
print(f"RESPONSE (Q4_K_M GGUF, llama-cpp-python):\n  {response['choices'][0]['message']['content']}")
print(f"\nTokens used: {response['usage']}")

# %% [markdown]
# ## 5. Optional — vLLM serving (BigGPU only)
#
# vLLM provides production-grade OpenAI-compatible serving. **Requires CUDA GPU
# with ≥ 16 GB VRAM** and `vllm` installed (see `requirements-biggpu.txt`).
# On T4 tier this cell will OOM. Skip on T4.
#
# Run in a SEPARATE terminal (NOT in the notebook — vLLM blocks until killed):
#
# ```bash
# pip install vllm                         # once
# vllm serve adapters/merged-fp16 \
#   --port 8000 \
#   --max-model-len 1024 \
#   --gpu-memory-utilization 0.9
# ```
#
# Then test:
#
# ```bash
# curl http://localhost:8000/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -d '{"model": "merged-fp16", "messages": [{"role": "user", "content": "Hello"}]}'
# ```
#
# **Why not in the notebook?** vLLM's process model doesn't play nicely with
# Jupyter — it expects to own the GPU + a long-running HTTP server. Run it as
# a sidecar process. The deck mentions vLLM as the deploy target; for actual
# production you'd containerize this command. For the lab, llama-cpp-python in
# step 4 is the graded artifact.

# %% [markdown]
# ## 6. Save deployment metadata

# %%
deploy_meta = {
    "compute_tier": COMPUTE_TIER,
    "base_model": BASE_MODEL,
    "merged_path": str(MERGED_PATH),
    "gguf_path": str(gguf_path),
    "gguf_size_mb": round(gguf_path.stat().st_size / 1e6, 1),
    "quantization": "q4_k_m",
    "smoke_prompt": SMOKE_PROMPT,
    "smoke_response": response["choices"][0]["message"]["content"],
}
(REPO_ROOT / "data" / "eval" / "deploy_meta.json").parent.mkdir(parents=True, exist_ok=True)
(REPO_ROOT / "data" / "eval" / "deploy_meta.json").write_text(
    json.dumps(deploy_meta, ensure_ascii=False, indent=2)
)
print("Saved data/eval/deploy_meta.json")

# %% [markdown]
# ## 7. Submission checklist
#
# Bạn vừa hoàn thành core lab. Trước khi submit:
#
# 1. **Run** `make verify` — gatekeeper sẽ list missing artifacts.
# 2. **Take screenshots** vào `submission/screenshots/` (xem `submission/screenshots/README.md`).
# 3. **Fill** `submission/REFLECTION.md` — đặc biệt là § 3 (reward curves analysis,
#    cross-reference deck §3.4) và § 6 (single change that mattered most).
# 4. **(Optional)** Pick a rigor add-on từ rubric.md (β-sweep, HF push, GGUF
#    release, W&B link, cross-judge).
# 5. **(Optional)** Pick a `BONUS-CHALLENGE.md` provocation cho creative bonus.
#
# Push public repo + paste URL vào VinUni LMS Day-22 box.
#
# Câu hỏi cuối để brainstorm trước khi đóng laptop:
#
# > **The deck says:** "DPO + 30 min A100 + 2k UltraFeedback → 3.2 → 4.1 helpfulness."
# > **You measured:** _<your win-rate from NB4>_.
# > **Why might they differ?** Dataset (English vs VN), base model (Qwen2.5-3B vs
# > deck's unspecified base), judge bias, sample size (8 prompts vs deck's full eval).
# > Đó chính là § 6 trong REFLECTION — what 1 change would close the gap.
