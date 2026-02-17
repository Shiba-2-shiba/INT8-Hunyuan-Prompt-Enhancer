import threading
from .model import HunyuanPromptEnhancer

# --- Global Cache ---
_MODEL_CACHE = {}
_CACHE_LOCK = threading.Lock()

def get_enhancer(ckpt_path, device_map, force_int8, attn_backend):
    key = (ckpt_path, str(device_map), force_int8, attn_backend)
    with _CACHE_LOCK:
        if key not in _MODEL_CACHE:
            _MODEL_CACHE[key] = HunyuanPromptEnhancer(ckpt_path, device_map, force_int8, attn_backend)
        return _MODEL_CACHE[key]
