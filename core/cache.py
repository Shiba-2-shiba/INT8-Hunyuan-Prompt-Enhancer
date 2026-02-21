# This file is part of a derivative work based on Tencent Hunyuan.
# See LICENSE.txt and NOTICE.txt for details.

import torch
import threading
from .model import HunyuanPromptEnhancer

# --- Global Cache ---
_MODEL_CACHE = {}
_CACHE_LOCK = threading.Lock()

def get_enhancer(ckpt_path, device_map, force_int8, attn_backend, quant_backend="bitsandbytes", quantized_safetensors=None, use_triton=True):
    key = (ckpt_path, str(device_map), force_int8, attn_backend, quant_backend, str(quantized_safetensors), use_triton)
    with _CACHE_LOCK:
        if key not in _MODEL_CACHE:
            _MODEL_CACHE[key] = HunyuanPromptEnhancer(
                ckpt_path,
                device_map,
                force_int8,
                attn_backend,
                quant_backend=quant_backend,
                quantized_safetensors=quantized_safetensors,
                use_triton=use_triton,
            )
        return _MODEL_CACHE[key]
