# This file is part of a derivative work based on Tencent Hunyuan.
# See LICENSE.txt and NOTICE.txt for details.

import torch

import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from .postprocess import _extract_reprompt, replace_single_quotes

def _triton_available() -> bool:
    try:
        import triton  # noqa: F401
        return True
    except Exception:
        return False

class HunyuanPromptEnhancer:
    def __init__(self, models_root_path, device_map="auto", force_int8=True, attn_backend="auto"):
        self.logger = logging.getLogger("HunyuanEnhancer")
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)

        self._compiled = False
        
        # Determine Dtype
        self.dtype = torch.float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
        elif not torch.cuda.is_available():
            self.dtype = torch.float32

        # Quantization Setup
        quant_cfg = None
        if force_int8:
            quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
            self.logger.info("INT8 Quantization Enabled")

        self.logger.info(f"Loading Model from: {models_root_path}")
        
        model_kwargs = dict(
            device_map=device_map,
            torch_dtype=self.dtype,
            quantization_config=quant_cfg,
            trust_remote_code=True,
        )
        
        # Attention Backend
        if attn_backend and attn_backend != "auto":
            model_kwargs["attn_implementation"] = attn_backend

        # Load Model
        try:
            self.model = AutoModelForCausalLM.from_pretrained(models_root_path, **model_kwargs).eval()
        except TypeError:
            # Fallback if attn_implementation isn't supported
            model_kwargs.pop("attn_implementation", None)
            self.model = AutoModelForCausalLM.from_pretrained(models_root_path, **model_kwargs).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(models_root_path, trust_remote_code=True)

    @torch.inference_mode()
    def predict(self, prompt_text, sys_prompt, config):
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt_text},
        ]

        # Apply Template
        try:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=config['enable_thinking'], # Hunyuan specific
            )
        except TypeError:
            # Fallback for older tokenizers
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )

        # Handle Dict vs Tensor inputs
        if not isinstance(inputs, torch.Tensor):
            # BatchEncoding or dict
            input_ids = inputs["input_ids"].to(self.model.device)
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.model.device)
        else:
            input_ids = inputs.to(self.model.device)
            attention_mask = None

        # Compile if requested
        if config['use_torch_compile'] and not self._compiled and hasattr(torch, "compile") and _triton_available():
            self.model = torch.compile(self.model, mode=config['compile_mode'])
            self._compiled = True

        # Seeding
        if config['seed'] is not None:
            torch.manual_seed(int(config['seed']))

        # Generate
        gen_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": config['max_new_tokens'],
            "do_sample": config['temperature'] > 0,
            "temperature": config['temperature'] if config['temperature'] > 0 else None,
            "top_p": config['top_p'] if config['temperature'] > 0 else None,
            "top_k": config.get('top_k', 50) if config['temperature'] > 0 else None,
            "use_cache": config['use_cache']
        }
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask

        outputs = self.model.generate(**gen_kwargs)
        
        # Decode
        generated_ids = outputs[0][input_ids.shape[-1]:]
        decoded_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Post-process
        reprompt, metadata = _extract_reprompt(decoded_text)
        reprompt = replace_single_quotes(reprompt)
        
        # Log Metrics
        from . import metrics
        if metrics.collector.enabled:
            metrics.collector.log("extraction", {
                "success": bool(reprompt),
                "method": metadata.get("method", "unknown"),
                "raw_length": metadata.get("raw_length", 0),
                "extracted_length": len(reprompt)
            })

        return reprompt
        return reprompt
    
    def offload(self):
        """Offload model to CPU and clear cache."""
        if hasattr(self, 'model'):
            self.model.to('cpu')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
