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

from core.model import HunyuanPromptEnhancer, resolve_int8_weights


def main():
    default_model_dir = os.path.abspath(os.path.join(repo_root, "..", "promptenhancer"))

    model_dir = os.environ.get("PE_MODEL_DIR", default_model_dir)
    raw_weights = os.environ.get("PE_INT8_WEIGHTS", "").strip() or None
    int8_weights = resolve_int8_weights(model_dir, raw_weights)
    use_triton = os.environ.get("PE_USE_TRITON", "1") != "0"
    variant = os.environ.get("PE_INT8_VARIANT", "optimized")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[1/3] Loading model from: {model_dir}")
    print(f"      Variant={variant}  Weights={int8_weights}")
    enhancer = HunyuanPromptEnhancer(
        model_dir,
        device_map=device,
        force_int8=True,
        attn_backend="auto",
        quant_backend="triton_int8",
        quantized_safetensors=int8_weights,
        use_triton=use_triton,
    )

    prompt = "A serene mountain lake at sunrise, cinematic lighting, ultra-detailed."
    sys_prompt = "You are a prompt enhancer. Rewrite the user prompt with richer detail and tags."
    run_config = {
        "temperature": 0.0,
        "top_p": 0.9,
        "top_k": 5,
        "max_new_tokens": 128,
        "seed": 0,
        "enable_thinking": False,
        "use_cache": True,
        "use_torch_compile": False,
        "compile_mode": "reduce-overhead",
    }

    print("[2/3] Generating...")
    t0 = time.perf_counter()
    decoded_text = enhancer.predict(prompt, sys_prompt, run_config)
    elapsed = time.perf_counter() - t0

    print("=== Output ===")
    print(decoded_text)
    print(f"=== Perf === elapsed={elapsed:.3f}s output_chars={len(decoded_text)}")


if __name__ == "__main__":
    main()
