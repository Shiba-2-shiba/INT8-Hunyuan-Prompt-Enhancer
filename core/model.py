# This file is part of a derivative work based on Tencent Hunyuan.
# See LICENSE.txt and NOTICE.txt for details.

import os
import json
import torch
import time

import logging
from typing import Optional
from safetensors.torch import safe_open

# Set HF cache env before importing transformers modules that snapshot these paths at import time.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HF_HOME = os.environ.get("HF_HOME", "").strip() or os.path.join(_REPO_ROOT, ".hf_home")
os.environ["HF_HOME"] = _HF_HOME
os.environ["HF_MODULES_CACHE"] = os.environ.get("HF_MODULES_CACHE", "").strip() or os.path.join(_HF_HOME, "modules")
os.environ["TRANSFORMERS_CACHE"] = os.environ.get("TRANSFORMERS_CACHE", "").strip() or os.path.join(_HF_HOME, "transformers")
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
os.makedirs(os.environ["HF_MODULES_CACHE"], exist_ok=True)
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from .postprocess import _extract_reprompt, replace_single_quotes
from .int8_triton import apply_quantized_weights

def _triton_available() -> bool:
    try:
        import triton  # noqa: F401
        return True
    except Exception:
        return False


def _has_complete_base_model_dir(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False
    if not os.path.exists(os.path.join(path, "config.json")):
        return False
    if os.path.exists(os.path.join(path, "model.safetensors")):
        return True
    index_path = os.path.join(path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return False
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            idx = json.load(f)
        weight_map = idx.get("weight_map", {})
        shard_files = set(weight_map.values())
        if not shard_files:
            return False
        return all(os.path.exists(os.path.join(path, shard)) for shard in shard_files)
    except Exception:
        return False


def _looks_like_quantized_safetensors(path: str) -> bool:
    """
    Detect if model.safetensors contains INT8 metadata (e.g., comfy_quant/weight_scale).
    Used to avoid treating a quantized-only repo as the base model.
    """
    base_path = os.path.join(path, "model.safetensors")
    if not os.path.exists(base_path):
        return False
    try:
        with safe_open(base_path, framework="pt") as f:
            for k in f.keys():
                if k.endswith(".comfy_quant") or k.endswith(".weight_scale") or k.endswith(".scale_weight"):
                    return True
    except Exception:
        return False
    return False


def _resolve_base_model_dir(preferred_path: str) -> str:
    if _has_complete_base_model_dir(preferred_path) and not _looks_like_quantized_safetensors(preferred_path):
        return preferred_path

    candidates = []
    env_base = os.environ.get("PE_BASE_MODEL_DIR", "").strip()
    if env_base:
        candidates.append(env_base)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates.append(os.path.abspath(os.path.join(repo_root, "..", "promptenhancer")))
    candidates.append(os.path.abspath(os.path.join(preferred_path, "..", "..", "..", "promptenhancer")))

    for c in candidates:
        if _has_complete_base_model_dir(c):
            return c
    return preferred_path


def _ensure_hf_runtime_env() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hf_home = os.environ.get("HF_HOME", "").strip()
    if not hf_home:
        hf_home = os.path.join(repo_root, ".hf_home")
        os.environ["HF_HOME"] = hf_home
    modules_cache = os.environ.get("HF_MODULES_CACHE", "").strip()
    if not modules_cache:
        modules_cache = os.path.join(hf_home, "modules")
        os.environ["HF_MODULES_CACHE"] = modules_cache
    transformers_cache = os.environ.get("TRANSFORMERS_CACHE", "").strip()
    if not transformers_cache:
        transformers_cache = os.path.join(hf_home, "transformers")
        os.environ["TRANSFORMERS_CACHE"] = transformers_cache
    os.makedirs(hf_home, exist_ok=True)
    os.makedirs(modules_cache, exist_ok=True)
    os.makedirs(transformers_cache, exist_ok=True)


def _dispatch_or_move_model(model: torch.nn.Module, device_map):
    """
    Best-effort device placement for INT8 Triton path.

    We avoid passing state_dict to from_pretrained (new Transformers restriction),
    so we move/dispatch after load.
    """
    if device_map is None:
        return model

    # String device_map: "auto", "cuda:0", "cpu"
    if isinstance(device_map, str):
        if device_map == "auto":
            target = "cuda" if torch.cuda.is_available() else "cpu"
            return model.to(target)
        if device_map.startswith("cuda") and torch.cuda.is_available():
            return model.to("cuda")
        if device_map == "cpu":
            return model.to("cpu")
        # Fallback for unexpected strings
        target = "cuda" if torch.cuda.is_available() else "cpu"
        return model.to(target)

    # Dict device_map (accelerate-style)
    if isinstance(device_map, dict):
        try:
            from accelerate import dispatch_model
            return dispatch_model(model, device_map=device_map)
        except Exception:
            target = "cuda" if torch.cuda.is_available() else "cpu"
            return model.to(target)

    return model


def _load_missing_base_weights(model_dir: str, missing_keys: list[str]) -> dict:
    """
    Load only missing weights from the base model safetensors.
    """
    if not missing_keys:
        return {}

    missing_set = set(missing_keys)
    base_path = os.path.join(model_dir, "model.safetensors")
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    out: dict = {}

    if os.path.exists(base_path):
        with safe_open(base_path, framework="pt") as f:
            for k in f.keys():
                if k in missing_set:
                    out[k] = f.get_tensor(k)
        return out

    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            idx = json.load(f)
        weight_map = idx.get("weight_map", {})
        shard_to_keys: dict[str, list[str]] = {}
        for k in missing_set:
            shard = weight_map.get(k, None)
            if shard is None:
                continue
            shard_to_keys.setdefault(shard, []).append(k)
        for shard, keys in shard_to_keys.items():
            shard_path = os.path.join(model_dir, shard)
            if not os.path.exists(shard_path):
                continue
            with safe_open(shard_path, framework="pt") as f:
                for k in keys:
                    if k in f.keys():
                        out[k] = f.get_tensor(k)
        return out

    return {}


class HunyuanPromptEnhancer:
    def __init__(
        self,
        models_root_path,
        device_map="auto",
        force_int8=True,
        attn_backend="auto",
        quant_backend="bitsandbytes",
        quantized_safetensors=None,
        use_triton=True,
    ):
        _ensure_hf_runtime_env()
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
        if quant_backend == "bitsandbytes" and force_int8:
            quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
            self.logger.info("INT8 Quantization Enabled (bitsandbytes)")
        elif quant_backend == "triton_int8":
            self.logger.info("INT8 Quantization Enabled (triton)")

        model_load_path = _resolve_base_model_dir(models_root_path)
        if os.path.abspath(model_load_path) != os.path.abspath(models_root_path):
            self.logger.warning(
                "Base model shards are missing in %s; falling back to %s. "
                "You can override via PE_BASE_MODEL_DIR.",
                models_root_path,
                model_load_path,
            )

        self.logger.info(f"Loading Model from: {model_load_path}")
        
        env_device_map = os.environ.get("PE_DEVICE_MAP", "").strip()
        if env_device_map:
            device_map = env_device_map

        model_kwargs = dict(
            device_map=device_map,
            dtype=self.dtype,
            quantization_config=quant_cfg,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        # Attention Backend
        if attn_backend and attn_backend != "auto":
            model_kwargs["attn_implementation"] = attn_backend

        # Load Model
        force_old_int8 = os.environ.get("PE_FORCE_OLD_INT8", "0").strip() == "1"

        if quant_backend == "triton_int8" and force_old_int8:
            # Old path: load base model then replace with INT8 weights (VRAM heavy)
            try:
                self.model = AutoModelForCausalLM.from_pretrained(model_load_path, **model_kwargs).eval()
            except TypeError:
                model_kwargs.pop("attn_implementation", None)
                self.model = AutoModelForCausalLM.from_pretrained(model_load_path, **model_kwargs).eval()

            tokenizer_path = models_root_path if os.path.exists(os.path.join(models_root_path, "tokenizer_config.json")) else model_load_path
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

            quantized_safetensors = resolve_int8_weights(models_root_path, quantized_safetensors)
            self.logger.info(f"Using INT8 weights: {quantized_safetensors}")

            def _infer_target_device(dm):
                if not torch.cuda.is_available():
                    return "cpu"
                if dm == "auto":
                    return "cuda"
                if isinstance(dm, str) and dm.startswith("cuda"):
                    return "cuda"
                if isinstance(dm, dict):
                    for v in dm.values():
                        if v == 0:
                            return "cuda"
                        if isinstance(v, str) and v.startswith("cuda"):
                            return "cuda"
                return "cpu"

            target_device = _infer_target_device(device_map)
            apply_quantized_weights(
                self.model,
                quantized_safetensors,
                device=target_device,
                use_triton=use_triton,
            )
            return

        if quant_backend == "triton_int8":
            quantized_safetensors = resolve_int8_weights(models_root_path, quantized_safetensors)
            self.logger.info(f"Using INT8 weights: {quantized_safetensors}")
            cpu_kwargs = dict(model_kwargs)
            cpu_kwargs["device_map"] = "cpu"
            cpu_kwargs["low_cpu_mem_usage"] = True
            cpu_kwargs["local_files_only"] = True
            try:
                self.model = AutoModelForCausalLM.from_pretrained(model_load_path, **cpu_kwargs).eval()
            except TypeError:
                cpu_kwargs.pop("attn_implementation", None)
                self.model = AutoModelForCausalLM.from_pretrained(model_load_path, **cpu_kwargs).eval()

            t1 = time.perf_counter()
            missing, unexpected = apply_quantized_weights(
                self.model,
                quantized_safetensors,
                device="cpu",
                use_triton=use_triton,
            )
            self.logger.info("INT8 weights applied in %.2fs", time.perf_counter() - t1)
            if missing:
                self.logger.info("Missing keys after INT8 apply: %d", len(missing))

                def _is_expected_int8_missing(k: str) -> bool:
                    return (
                        k.endswith(".weight_int8")
                        or k.endswith(".weight_scale")
                        or k.endswith(".scale_weight")
                        or k.endswith(".comfy_quant")
                        or ".int8_impl." in k
                        or "._int8_impl." in k
                    )

                missing_non_int8 = [k for k in missing if not _is_expected_int8_missing(k)]
                if missing_non_int8:
                    self.logger.info("Missing non-INT8 keys: %d", len(missing_non_int8))
                    self.logger.info("Missing non-INT8 keys are being filled from base model.")
                    if "lm_head.weight" in missing_non_int8:
                        try:
                            self.model.lm_head.weight = self.model.get_input_embeddings().weight
                        except Exception:
                            pass
                    t_fill = time.perf_counter()
                    fill_state = _load_missing_base_weights(model_load_path, missing_non_int8)
                    if fill_state:
                        self.model.load_state_dict(fill_state, strict=False)
                        self.logger.info(
                            "Filled %d missing keys from base model in %.2fs",
                            len(fill_state),
                            time.perf_counter() - t_fill,
                        )
            if unexpected:
                self.logger.info("Unexpected keys after INT8 apply: %d", len(unexpected))
            if hasattr(self.model, "tie_weights"):
                try:
                    self.model.tie_weights()
                except Exception:
                    pass
            t2 = time.perf_counter()
            self.model = _dispatch_or_move_model(self.model, device_map)
            self.logger.info("Model dispatched in %.2fs", time.perf_counter() - t2)
        else:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(model_load_path, **model_kwargs).eval()
            except TypeError:
                # Fallback if attn_implementation isn't supported
                model_kwargs.pop("attn_implementation", None)
                self.model = AutoModelForCausalLM.from_pretrained(model_load_path, **model_kwargs).eval()

        tokenizer_path = models_root_path if os.path.exists(os.path.join(models_root_path, "tokenizer_config.json")) else model_load_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        # INT8 weights are loaded during model construction when using triton_int8
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
    
    def offload(self):
        """Offload model to CPU and clear cache."""
        if hasattr(self, 'model'):
            self.model.to('cpu')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def resolve_int8_weights(models_root_path: str, explicit_path: Optional[str] = None) -> str:
    """
    Resolve INT8 safetensors path for Triton backend.

    Priority:
    1) explicit_path argument
    2) PE_INT8_WEIGHTS env var
    3) variant-based defaults in model dir (PE_INT8_VARIANT=optimized|high|auto)
    """
    if explicit_path and str(explicit_path).strip():
        candidate = str(explicit_path).strip()
        if os.path.exists(candidate):
            return candidate
        raise FileNotFoundError(f"Explicit quantized_safetensors not found: {candidate}")

    env_path = os.environ.get("PE_INT8_WEIGHTS", "").strip()
    if env_path:
        if os.path.exists(env_path):
            return env_path
        raise FileNotFoundError(f"PE_INT8_WEIGHTS does not exist: {env_path}")

    variant = os.environ.get("PE_INT8_VARIANT", "optimized").strip().lower()
    candidates_by_variant = {
        "optimized": [
            "model.safetensors",
            "HunyuanImage-2.1-reprompt-INT8-optimized.safetensors",
            "promptenhancer_merged_simple_int8_tensorwise.safetensors",
        ],
        "high": [
            "model.safetensors",
            "promptenhancer_int8_high.safetensors",
        ],
        "auto": [
            "model.safetensors",
            "HunyuanImage-2.1-reprompt-INT8-optimized.safetensors",
            "promptenhancer_int8_high.safetensors",
            "promptenhancer_merged_simple_int8_tensorwise.safetensors",
        ],
    }
    candidate_names = candidates_by_variant.get(variant, candidates_by_variant["optimized"])
    for name in candidate_names:
        candidate = os.path.join(models_root_path, name)
        if os.path.exists(candidate):
            return candidate

    checked = ", ".join(candidate_names)
    raise FileNotFoundError(
        f"quantized_safetensors not provided and no default found (variant={variant}). "
        f"Checked: {checked}"
    )
