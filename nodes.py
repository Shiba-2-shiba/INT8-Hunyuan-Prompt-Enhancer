# This file is part of a derivative work based on Tencent Hunyuan.
# See LICENSE.txt and NOTICE.txt for details.

import logging
try:
    from .core import config, assets, tokenizer_patch, cache, prompts, postprocess
except ImportError:
    from core import config, assets, tokenizer_patch, cache, prompts, postprocess

# --- ComfyUI Node Definition ---

DEFAULT_MODEL_VARIANTS = [
    "INT8 (Standard)",
    "INT8 (Heretic)",
]


def _load_model_entries():
    conf = config.load_config()
    if "models" in conf:
        return conf["models"]
    if "int8" in conf:
        return {DEFAULT_MODEL_VARIANTS[0]: conf["int8"]}
    return {}


def _get_default_model_variant(model_entries):
    for variant_name, variant_conf in model_entries.items():
        if variant_conf.get("default"):
            return variant_name
    return next(iter(model_entries), DEFAULT_MODEL_VARIANTS[0])


def _get_model_variants():
    try:
        model_entries = _load_model_entries()
        variants = list(model_entries.keys())
        if variants:
            return variants
    except Exception:
        logging.exception("Failed to load model variants from ckpts.yaml")
    return list(DEFAULT_MODEL_VARIANTS)


def _resolve_model_conf(selected_variant):
    model_entries = _load_model_entries()
    if not model_entries:
        raise KeyError("No model definitions found in ckpts.yaml")
    variant_name = selected_variant if selected_variant in model_entries else _get_default_model_variant(model_entries)
    return model_entries[variant_name]

class INT8_Hunyuan_PromptEnhancer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "placeholder": "Enter your prompt here..."}),
                "model_variant": (_get_model_variants(),),
                "style_policy": (["illustration (Tag List)", "photography (Detailed)"],),
                "temperature": ("FLOAT", {"default": 0.0, "step": 0.01, "min": 0, "max": 2.0}),
                "top_p": ("FLOAT", {"default": 0.9, "step": 0.01, "min": 0, "max": 1.0}),
                "top_k": ("INT", {"default": 5, "min": 1, "max": 100, "step": 1}),
                "max_new_tokens": ("INT", {"default": 512, "step": 1, "min": 128, "max": 4096}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "enable_thinking": ("BOOLEAN", {"default": False, "label": "Enable CoT Reasoning"}),
                "device_map": (["cuda:0", "auto", "cpu"],),
                "attn_backend": (["auto", "sdpa", "flash_attention_2"],),
            },
            "optional": {
                 # Allowed to override system prompt if absolutely needed, but hidden by default in basic use
                "custom_sys_prompt": ("STRING", {"multiline": True, "default": "", "placeholder": "Override System Prompt (Optional)"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("enhanced_prompt",)
    FUNCTION = "run"
    CATEGORY = "Hunyuan"

    def run(self, text, style_policy, temperature, top_p, top_k=5, max_new_tokens=512, seed=0, 
            enable_thinking=False, device_map="auto", attn_backend="auto", model_variant=None, custom_sys_prompt=""):
        
        # VRAM Management: Unload other models to free up space
        try:
            import comfy.model_management as mm
            mm.soft_empty_cache()
        except ImportError:
            pass
        
        # Ensure torch is available for device checks
        import torch

        # 1. Model Setup
        conf = _resolve_model_conf(model_variant)
        policy = config.load_policy() # Load policy (new in Phase 2)
        
        # Initialize Metrics (Phase 3)
        try:
            from .core import metrics
        except ImportError:
            from core import metrics
        metrics.collector.configure(policy)

        # Auto-download if missing
        ckpt_path = assets.ensure_model(conf)

        # Patch Tokenizer
        tokenizer_patch.patch_tokenizer(ckpt_path)

        normalized_device_map = {"": 0} if device_map == "cuda:0" else device_map
        
        # Quant backend is fixed to Triton; INT8 weights are resolved automatically.
        quant_backend = "triton_int8"
        quant_path = None

        enhancer = cache.get_enhancer(
            ckpt_path,
            device_map=normalized_device_map,
            force_int8=True,
            attn_backend=attn_backend,
            quant_backend=quant_backend,
            quantized_safetensors=quant_path,
            use_triton=True,
        )

        # Ensure model is on the correct device (if offloaded previously)
        if hasattr(enhancer, 'model'):
            # If device_map is auto, it handles itself usually, but if we offloaded to cpu manually:
            # We need to move it back to CUDA if implied by device_map
            target_device = "cuda" if device_map in ["cuda:0", "auto"] and torch.cuda.is_available() else "cpu"
            if str(enhancer.model.device) == 'cpu' and target_device == 'cuda':
                enhancer.model.to(target_device)

        # 2. Prompt Strategy Selection
        if custom_sys_prompt and custom_sys_prompt.strip():
            sys_prompt = custom_sys_prompt
        elif "illustration" in style_policy:
            sys_prompt = prompts.SYS_PROMPT_ILLUST_OPTIMIZED
        else:
            sys_prompt = prompts.SYS_PROMPT_PHOTO_OPTIMIZED

        # 3. Execution
        
        # Dynamic Token Budget for Thinking Mode
        # If thinking is enabled, the model needs more tokens for the thought process within the same generation call.
        # We double the user's limit (up to a safe hard cap) to prevent cut-offs.
        final_max_tokens = max_new_tokens
        if enable_thinking:
            final_max_tokens = min(max_new_tokens * 2, 8192)

        run_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_new_tokens": final_max_tokens,
            "seed": seed,
            "enable_thinking": enable_thinking,
            "use_cache": True,
            "use_torch_compile": False, # Disabled by default for stability
            "compile_mode": "reduce-overhead"
        }

        new_prompt = enhancer.predict(text, sys_prompt, run_config)

        # 4. Post-Processing Enforcement (The "Safety Valve")
        if "illustration" in style_policy:
            # Force strip photo terms unless user originally asked for them
            # Fetch banned words from policy, safely fallback if key missing
            banned_patterns = policy.get('banned_words', {}).get('illustration', [])
            new_prompt = postprocess.strip_unwanted_photo_style(
                new_prompt, 
                text, 
                banned_patterns,
                collector=metrics.collector
            )

        # VRAM Management: Offload this model to let others use VRAM
        try:
            enhancer.offload()
            import comfy.model_management as mm
            mm.soft_empty_cache()
        except Exception:
            pass

        return (new_prompt,)

NODE_CLASS_MAPPINGS = {
    "INT8_Hunyuan_PromptEnhancer": INT8_Hunyuan_PromptEnhancer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "INT8_Hunyuan_PromptEnhancer": "INT8 Hunyuan Prompt Enhancer",
}
