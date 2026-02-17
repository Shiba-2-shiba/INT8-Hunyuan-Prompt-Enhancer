import logging
try:
    from .core import config, assets, tokenizer_patch, cache, prompts, postprocess
except ImportError:
    from core import config, assets, tokenizer_patch, cache, prompts, postprocess

# --- ComfyUI Node Definition ---

class INT8_Hunyuan_PromptEnhancer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "placeholder": "Enter your prompt here..."}),
                "style_policy": (["illustration (Tag List)", "photography (Detailed)"],),
                "temperature": ("FLOAT", {"default": 0.7, "step": 0.01, "min": 0, "max": 2.0}),
                "top_p": ("FLOAT", {"default": 0.9, "step": 0.01, "min": 0, "max": 1.0}),
                "max_new_tokens": ("INT", {"default": 512, "step": 1, "min": 128, "max": 4096}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "enable_thinking": ("BOOLEAN", {"default": True, "label": "Enable CoT Reasoning"}),
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
    CATEGORY = "XX/Hunyuan"

    def run(self, text, style_policy, temperature, top_p, max_new_tokens, seed, 
            enable_thinking, device_map, attn_backend, custom_sys_prompt=""):
        
        # 1. Model Setup
        conf = config.load_config()['int8']
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
        
        enhancer = cache.get_enhancer(
            ckpt_path, 
            device_map=normalized_device_map, 
            force_int8=True, 
            attn_backend=attn_backend
        )

        # 2. Prompt Strategy Selection
        if custom_sys_prompt and custom_sys_prompt.strip():
            sys_prompt = custom_sys_prompt
        elif "illustration" in style_policy:
            sys_prompt = prompts.SYS_PROMPT_ILLUST_OPTIMIZED
        else:
            sys_prompt = prompts.SYS_PROMPT_PHOTO_OPTIMIZED

        # 3. Execution
        run_config = {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
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

        return (new_prompt,)

NODE_CLASS_MAPPINGS = {
    "INT8_Hunyuan_PromptEnhancer": INT8_Hunyuan_PromptEnhancer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "INT8_Hunyuan_PromptEnhancer": "INT8 Hunyuan Prompt Enhancer",
}
