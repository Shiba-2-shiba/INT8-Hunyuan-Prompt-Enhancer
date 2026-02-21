import os
import sys
import time
from unittest.mock import patch

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

# Ensure Triton cache/temp dirs are writable before importing nodes
triton_cache = os.environ.get("TRITON_CACHE_DIR")
if not triton_cache:
    triton_cache = os.path.join(repo_root, ".triton_cache")
    os.environ["TRITON_CACHE_DIR"] = triton_cache
os.makedirs(triton_cache, exist_ok=True)
tmp_dir = os.path.join(triton_cache, "tmp")
os.makedirs(tmp_dir, exist_ok=True)
os.environ["TEMP"] = tmp_dir
os.environ["TMP"] = tmp_dir


def main():
    model_dir = os.environ.get(
        "PE_MODEL_DIR",
        os.path.abspath(os.path.join(repo_root, "models", "reprompt-INT8-shiba")),
    )
    from core.model import resolve_int8_weights
    int8_weights = resolve_int8_weights(model_dir, os.environ.get("PE_INT8_WEIGHTS", "").strip() or None)
    variant = os.environ.get("PE_INT8_VARIANT", "optimized")

    from nodes import INT8_Hunyuan_PromptEnhancer

    # Patch config to use local model directory (no download)
    conf = {
        "int8": {
            "repo_id": "local",
            "local_dir": model_dir,
        }
    }

    node = INT8_Hunyuan_PromptEnhancer()
    print(f"[config] Variant={variant} Weights={int8_weights}")
    with patch("core.config.load_config", return_value=conf):
        t0 = time.perf_counter()
        out = node.run(
            text="A calm river in a misty forest at dawn.",
            style_policy="illustration (Tag List)",
            temperature=0.0,
            top_p=0.9,
            top_k=5,
            max_new_tokens=128,
            seed=0,
            enable_thinking=False,
            device_map="cuda:0",
            attn_backend="auto",
            custom_sys_prompt="",
        )
        elapsed = time.perf_counter() - t0

    print("=== Node Output ===")
    print(out[0])
    print(f"=== Perf === elapsed={elapsed:.3f}s output_chars={len(out[0])}")


if __name__ == "__main__":
    main()
