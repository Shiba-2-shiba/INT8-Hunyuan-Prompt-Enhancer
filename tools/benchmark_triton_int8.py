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

from core.int8_triton import apply_quantized_weights, get_int8_stats, reset_int8_stats
from core.model import resolve_int8_weights


def _sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _load_pipeline(model_dir, weights, use_triton, device):
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ).eval()
    apply_quantized_weights(model, weights, device=device, use_triton=use_triton)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    return model, tokenizer


def _prepare_inputs(tokenizer, prompt, device, enable_thinking=False):
    messages = [
        {"role": "system", "content": "You are a prompt enhancer. Rewrite with richer detail and tags."},
        {"role": "user", "content": prompt},
    ]
    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=bool(enable_thinking),
        )
    except TypeError:
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
    return input_ids, attention_mask


def _benchmark_mode(model, input_ids, attention_mask, max_new_tokens, runs, warmup):
    def _run_once():
        if max_new_tokens <= 0:
            return 0.0, 0.0, 0

        _sync_cuda()
        t0_prefill = time.perf_counter()
        with torch.inference_mode():
            prefill_out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
        _sync_cuda()
        prefill_elapsed = time.perf_counter() - t0_prefill

        # First generated token comes from prefill logits.
        next_token = torch.argmax(prefill_out.logits[:, -1, :], dim=-1, keepdim=True)
        past_key_values = prefill_out.past_key_values
        generated = 1

        if attention_mask is None:
            running_attention_mask = torch.ones(
                (input_ids.shape[0], input_ids.shape[1]),
                dtype=torch.long,
                device=input_ids.device,
            )
        else:
            running_attention_mask = attention_mask

        t0_decode = time.perf_counter()
        with torch.inference_mode():
            while generated < max_new_tokens:
                # Extend mask by one token for current decode step.
                one = torch.ones(
                    (running_attention_mask.shape[0], 1),
                    dtype=running_attention_mask.dtype,
                    device=running_attention_mask.device,
                )
                running_attention_mask = torch.cat((running_attention_mask, one), dim=-1)
                dec_out = model(
                    input_ids=next_token,
                    attention_mask=running_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                next_token = torch.argmax(dec_out.logits[:, -1, :], dim=-1, keepdim=True)
                past_key_values = dec_out.past_key_values
                generated += 1

        _sync_cuda()
        decode_elapsed = time.perf_counter() - t0_decode
        return prefill_elapsed, decode_elapsed, generated

    for _ in range(max(0, warmup)):
        _run_once()

    prefill_times = []
    decode_times = []
    token_counts = []
    for i in range(runs):
        prefill_elapsed, decode_elapsed, tokens = _run_once()
        total_elapsed = prefill_elapsed + decode_elapsed
        prefill_times.append(prefill_elapsed)
        decode_times.append(decode_elapsed)
        token_counts.append(tokens)
        decode_tps = tokens / decode_elapsed if decode_elapsed > 0 else 0.0
        total_tps = tokens / total_elapsed if total_elapsed > 0 else 0.0
        print(
            f"[run {i+1}/{runs}] prefill={prefill_elapsed:.3f}s decode={decode_elapsed:.3f}s "
            f"total={total_elapsed:.3f}s tokens={tokens} decode_tok/s={decode_tps:.2f} total_tok/s={total_tps:.2f}"
        )

    total_prefill = sum(prefill_times)
    total_decode = sum(decode_times)
    total_time = total_prefill + total_decode
    total_tokens = sum(token_counts)
    avg_prefill = total_prefill / len(prefill_times) if prefill_times else 0.0
    avg_decode = total_decode / len(decode_times) if decode_times else 0.0
    avg_time = total_time / len(prefill_times) if prefill_times else 0.0
    avg_decode_tps = total_tokens / total_decode if total_decode > 0 else 0.0
    avg_total_tps = total_tokens / total_time if total_time > 0 else 0.0
    return avg_prefill, avg_decode, avg_time, avg_decode_tps, avg_total_tps


def _print_int8_stats(mode_name):
    stats = get_int8_stats()
    attempts = int(stats.get("triton_attempts", 0))
    success = int(stats.get("triton_success", 0))
    fallback_dim = int(stats.get("fallback_dim_mismatch", 0))
    fallback_gemm = int(stats.get("fallback_gemm_error", 0))
    fallback_dequant = int(stats.get("fallback_dequant", 0))
    total_fallback = max(0, attempts - success)
    fallback_rate = (100.0 * total_fallback / attempts) if attempts > 0 else 0.0
    print(
        f"[int8:{mode_name}] attempts={attempts} success={success} "
        f"fallback={total_fallback} fallback_rate={fallback_rate:.2f}% "
        f"dequant={fallback_dequant} dim_mismatch={fallback_dim} gemm_error={fallback_gemm} "
        f"int8_layers={int(stats.get('int8_layers_replaced', 0))}"
    )


def main():
    model_dir = os.environ.get(
        "PE_MODEL_DIR",
        os.path.abspath(os.path.join(repo_root, "..", "promptenhancer")),
    )
    weights = resolve_int8_weights(model_dir, os.environ.get("PE_INT8_WEIGHTS", "").strip() or None)
    use_triton = os.environ.get("PE_USE_TRITON", "1") != "0"
    compare_triton = os.environ.get("PE_COMPARE_TRITON", "1") != "0"
    enable_thinking = os.environ.get("PE_ENABLE_THINKING", "0") == "1"
    runs = int(os.environ.get("PE_BENCH_RUNS", "3"))
    warmup = int(os.environ.get("PE_BENCH_WARMUP", "1"))
    max_new_tokens = int(os.environ.get("PE_MAX_NEW_TOKENS", "128"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] model={model_dir}")
    print(f"[load] weights={weights} device={device}")
    print(
        f"[config] compare_triton={compare_triton} enable_thinking={enable_thinking} "
        f"runs={runs} warmup={warmup} max_new_tokens={max_new_tokens}"
    )

    prompt = os.environ.get("PE_BENCH_PROMPT", "A serene mountain lake at sunrise, cinematic lighting.")
    modes = [("triton_on", True), ("triton_off", False)] if compare_triton else [("single", use_triton)]
    results = []
    for mode_name, mode_use_triton in modes:
        print(f"[mode:{mode_name}] use_triton={mode_use_triton}")
        reset_int8_stats()
        model, tokenizer = _load_pipeline(model_dir, weights, mode_use_triton, device)
        input_ids, attention_mask = _prepare_inputs(tokenizer, prompt, device, enable_thinking=enable_thinking)
        avg_prefill, avg_decode, avg_time, avg_decode_tps, avg_total_tps = _benchmark_mode(
            model,
            input_ids,
            attention_mask,
            max_new_tokens=max_new_tokens,
            runs=runs,
            warmup=warmup,
        )
        print(
            f"[summary:{mode_name}] avg_prefill={avg_prefill:.3f}s avg_decode={avg_decode:.3f}s "
            f"avg_total={avg_time:.3f}s decode_tok/s={avg_decode_tps:.2f} total_tok/s={avg_total_tps:.2f}"
        )
        _print_int8_stats(mode_name)
        results.append((mode_name, avg_prefill, avg_decode, avg_time, avg_decode_tps, avg_total_tps))
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if compare_triton and len(results) == 2:
        _, p_on, d_on, t_on, dtps_on, ttps_on = results[0]
        _, p_off, d_off, t_off, dtps_off, ttps_off = results[1]
        speedup_prefill = (p_off / p_on) if p_on > 0 else 0.0
        speedup_decode = (d_off / d_on) if d_on > 0 else 0.0
        speedup_total = (t_off / t_on) if t_on > 0 else 0.0
        speedup_decode_tps = (dtps_on / dtps_off) if dtps_off > 0 else 0.0
        speedup_total_tps = (ttps_on / ttps_off) if ttps_off > 0 else 0.0
        print(
            f"[compare] triton_on_vs_off: prefill_speedup={speedup_prefill:.2f}x "
            f"decode_speedup={speedup_decode:.2f}x total_speedup={speedup_total:.2f}x "
            f"decode_tok/s_speedup={speedup_decode_tps:.2f}x total_tok/s_speedup={speedup_total_tps:.2f}x"
        )


if __name__ == "__main__":
    main()
