import os
import sys
import time

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

# Ensure Triton cache/temp dirs are writable before importing torch/triton
triton_cache = os.environ.get("TRITON_CACHE_DIR")
if not triton_cache:
    triton_cache = os.path.join(repo_root, ".triton_cache")
    os.environ["TRITON_CACHE_DIR"] = triton_cache
os.makedirs(triton_cache, exist_ok=True)
tmp_dir = os.path.join(triton_cache, "tmp")
os.makedirs(tmp_dir, exist_ok=True)
os.environ["TEMP"] = tmp_dir
os.environ["TMP"] = tmp_dir

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.int8_triton import apply_quantized_weights
from core.model import resolve_int8_weights


def main():
    default_model_dir = os.path.abspath(os.path.join(repo_root, "..", "promptenhancer"))

    model_dir = os.environ.get("PE_MODEL_DIR", default_model_dir)
    raw_weights = os.environ.get("PE_INT8_WEIGHTS", "").strip() or None
    int8_weights = resolve_int8_weights(model_dir, raw_weights)
    use_triton = os.environ.get("PE_USE_TRITON", "1") != "0"
    variant = os.environ.get("PE_INT8_VARIANT", "optimized")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[1/4] Loading model from: {model_dir}")
    print(f"      Variant={variant}  Weights={int8_weights}")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ).eval()

    print(f"[2/4] Applying INT8 weights: {int8_weights}")
    apply_quantized_weights(model, int8_weights, device=device, use_triton=use_triton)
    model = model.to(device)

    print("[3/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)

    prompt = "A serene mountain lake at sunrise, cinematic lighting, ultra-detailed."
    sys_prompt = "You are a prompt enhancer. Rewrite the user prompt with richer detail and tags."
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    if not isinstance(inputs, torch.Tensor):
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
    else:
        input_ids = inputs.to(device)
        attention_mask = None

    print("[4/4] Generating...")
    t0 = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=128,
            do_sample=False,
            use_cache=True,
        )
    elapsed = time.perf_counter() - t0

    generated_ids = outputs[0][input_ids.shape[-1]:]
    decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print("=== Output ===")
    print(decoded_text)
    gen_tokens = int(generated_ids.shape[-1])
    tps = gen_tokens / elapsed if elapsed > 0 else 0.0
    print(f"=== Perf === elapsed={elapsed:.3f}s tokens={gen_tokens} tok/s={tps:.2f}")


if __name__ == "__main__":
    main()
