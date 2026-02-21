"""
INT8 Triton backend helpers for Hunyuan Prompt Enhancer.

This module provides:
- Int8Linear: a lightweight INT8 Linear layer with optional Triton kernel path
- apply_quantized_weights: load INT8 safetensors (convert_to_quant output) into a HF model

It is designed to work with convert_to_quant outputs that include:
- <name>.weight (int8)
- <name>.weight_scale (scalar or 2D)
- <name>.comfy_quant (JSON metadata: format, group_size)
"""

from __future__ import annotations

import json
import logging
import os
import importlib.util
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Set
from contextlib import contextmanager
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import safe_open

logger = logging.getLogger("HunyuanEnhancer.INT8")


_INT8_STATS = defaultdict(int)


def reset_int8_stats() -> None:
    _INT8_STATS.clear()


def get_int8_stats() -> Dict[str, int]:
    return dict(_INT8_STATS)


def _ensure_triton_env():
    """
    Ensure Triton cache/temp dirs are writable (important on Windows sandboxes).
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = os.environ.get("TRITON_CACHE_DIR")
    if not cache_dir:
        cache_dir = os.path.join(repo_root, ".triton_cache")
        os.environ["TRITON_CACHE_DIR"] = cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    # Triton may use tempfile under TEMP/TMP for compilation artifacts.
    temp_dir = os.path.join(cache_dir, "tmp")
    os.makedirs(temp_dir, exist_ok=True)
    os.environ["TEMP"] = temp_dir
    os.environ["TMP"] = temp_dir


# Set env early to avoid Triton using unwritable system temp on Windows.
_ensure_triton_env()


def _triton_available() -> bool:
    try:
        import triton  # noqa: F401
        return True
    except Exception:
        return False


_KERNELS_CACHE = None


def _load_int8_kernels() -> Optional[object]:
    """
    Load Triton kernels from this repo if vendored, otherwise from sibling checkout.
    """
    global _KERNELS_CACHE
    if _KERNELS_CACHE is not None:
        return _KERNELS_CACHE

    _ensure_triton_env()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # 1) Prefer vendored kernels: <repo_root>/kernels/int8_kernels.py
    vendored_path = os.path.join(repo_root, "kernels", "int8_kernels.py")
    if os.path.exists(vendored_path):
        kernels_path = vendored_path
    else:
        # 2) Fallback to sibling repo: ../ComfyUI-QuantOps/kernels/int8_kernels.py
        kernels_path = os.path.abspath(os.path.join(repo_root, "..", "ComfyUI-QuantOps", "kernels", "int8_kernels.py"))
        if not os.path.exists(kernels_path):
            _KERNELS_CACHE = None
            return None

    try:
        spec = importlib.util.spec_from_file_location("comfyui_quantops_int8_kernels", kernels_path)
        if spec is None or spec.loader is None:
            _KERNELS_CACHE = None
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _KERNELS_CACHE = module
        return module
    except Exception as e:
        logger.warning(f"Failed to load Triton kernels from {kernels_path}: {e}")
        _KERNELS_CACHE = None
        return None


def _act_quant_pytorch(x: torch.Tensor, block_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Block-wise activation quantization (PyTorch fallback)."""
    if not x.is_contiguous():
        x = x.contiguous()
    K = x.shape[-1]
    if K % block_size != 0:
        raise ValueError(f"Last dim K={K} must be divisible by block_size={block_size}")
    batch_shape = x.shape[:-1]
    x_blocked = x.reshape(*batch_shape, K // block_size, block_size)
    amax = x_blocked.abs().amax(dim=-1)
    scale = torch.maximum(amax / 127.0, torch.tensor(1e-8, device=amax.device, dtype=amax.dtype))
    q = (x_blocked / scale.unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8)
    q = q.reshape(x.shape)
    return q, scale.to(torch.float32)


@dataclass
class Int8Meta:
    format: str = "int8_tensorwise"
    block_size: int = 128


class Int8Linear(nn.Module):
    """
    Minimal INT8 Linear layer with optional Triton backend.
    """

    def __init__(
        self,
        weight_int8: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: Optional[torch.Tensor],
        block_size: int = 128,
        quant_format: str = "int8_tensorwise",
        use_triton: bool = True,
    ):
        super().__init__()
        self.block_size = int(block_size)
        self.quant_format = quant_format
        self.use_triton = bool(use_triton) and _triton_available() and _load_int8_kernels() is not None

        # Register buffers so .to(device) moves them.
        self.register_buffer("weight_int8", weight_int8.to(torch.int8), persistent=True)
        self.register_buffer("weight_scale", weight_scale.to(torch.float32), persistent=True)
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    @staticmethod
    def _expand_tensorwise_scale(scale: torch.Tensor, weight: torch.Tensor, block_size: int) -> torch.Tensor:
        if scale.dim() == 0:
            N, K = weight.shape
            return scale.expand(N // block_size, K // block_size).contiguous()
        return scale

    def _int8_gemm(self, a_int8, a_scale, b_int8, b_scale, bias):
        kernels = _load_int8_kernels()
        if kernels is None:
            raise RuntimeError("Triton kernels not available")
        if bias is not None:
            return kernels.int8_addmm(a_int8, a_scale, b_int8, b_scale, bias)
        return kernels.int8_gemm(a_int8, a_scale, b_int8, b_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _INT8_STATS["forward_calls"] += 1
        # Fast path: Triton INT8 matmul with dynamic activation quant
        if self.use_triton and x.is_cuda:
            _INT8_STATS["triton_enabled_calls"] += 1
            K = x.shape[-1]
            if K % self.block_size != 0:
                # Fallback to dequant if dimension mismatch
                _INT8_STATS["fallback_dim_mismatch"] += 1
                return F.linear(x, self.dequantize_weight(x.dtype), self.bias)

            # Quantize activations
            _INT8_STATS["triton_attempts"] += 1
            try:
                kernels = _load_int8_kernels()
                if kernels is not None:
                    a_int8, a_scale = kernels.act_quant(x.contiguous(), block_size=self.block_size)
                else:
                    a_int8, a_scale = _act_quant_pytorch(x, self.block_size)
            except Exception:
                _INT8_STATS["fallback_quant_error"] += 1
                a_int8, a_scale = _act_quant_pytorch(x, self.block_size)

            weight_scale = self._expand_tensorwise_scale(self.weight_scale, self.weight_int8, self.block_size)

            try:
                out = self._int8_gemm(
                    a_int8,
                    a_scale,
                    self.weight_int8.contiguous(),
                    weight_scale.contiguous(),
                    self.bias,
                )
                # int8_kernels returns float32 by default
                _INT8_STATS["triton_success"] += 1
                return out.to(x.dtype)
            except Exception as e:
                _INT8_STATS["fallback_gemm_error"] += 1
                logger.warning(f"Triton INT8 matmul failed, fallback to dequant: {e}")

        # Fallback: dequantize and use standard matmul
        if self.use_triton and x.is_cuda:
            _INT8_STATS["fallback_dequant"] += 1
        return F.linear(x, self.dequantize_weight(x.dtype), self.bias)

    def dequantize_weight(self, out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
        scale = self.weight_scale
        w = self.weight_int8
        if scale.dim() == 0:
            return w.to(out_dtype) * scale.to(out_dtype)

        # Blockwise dequant (N,K) with scale (N//B, K//B)
        N, K = w.shape
        B = self.block_size
        if K % B != 0 or N % B != 0:
            return w.to(out_dtype) * scale.to(out_dtype)
        w_blocked = w.reshape(N // B, B, K // B, B).permute(0, 2, 1, 3)
        scale_b = scale.unsqueeze(-1).unsqueeze(-1).to(out_dtype)
        deq = w_blocked.to(out_dtype) * scale_b
        return deq.permute(0, 2, 1, 3).reshape(N, K)


class Int8LinearProxy(nn.Linear):
    """
    Drop-in replacement for nn.Linear that can consume INT8 weights at load time.

    This avoids post-load module replacement and prevents GPU memory spikes.
    """

    DEFAULT_USE_TRITON = True
    DEFAULT_BLOCK_SIZE = 128

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._int8_impl: Optional[Int8Linear] = None
        self._int8_ready = False
        self._int8_use_triton = bool(Int8LinearProxy.DEFAULT_USE_TRITON)
        self._int8_default_block_size = int(Int8LinearProxy.DEFAULT_BLOCK_SIZE)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        weight_key = prefix + "weight"
        weight_tensor = state_dict.get(weight_key, None)

        if weight_tensor is not None and weight_tensor.dtype == torch.int8:
            # Consume INT8 tensors directly
            state_dict.pop(weight_key, None)
            scale = state_dict.pop(prefix + "weight_scale", None)
            if scale is None:
                scale = state_dict.pop(prefix + "scale_weight", None)
            bias = state_dict.pop(prefix + "bias", None)
            comfy_quant = state_dict.pop(prefix + "comfy_quant", None)

            if scale is None:
                error_msgs.append(f"Missing weight_scale for INT8 layer: {prefix}")
                return

            meta = _parse_comfy_quant(comfy_quant) if comfy_quant is not None else Int8Meta()
            block_size = meta.block_size or self._int8_default_block_size
            quant_format = meta.format

            if scale.dim() == 0 and quant_format == "int8_tensorwise":
                scale = Int8Linear._expand_tensorwise_scale(scale, weight_tensor, block_size)

            # Build INT8 implementation and drop float params to avoid GPU copies
            self._int8_impl = Int8Linear(
                weight_int8=weight_tensor,
                weight_scale=scale,
                bias=bias,
                block_size=block_size,
                quant_format=quant_format,
                use_triton=self._int8_use_triton,
            )
            self._int8_ready = True
            if "weight" in self._parameters:
                del self._parameters["weight"]
            if "bias" in self._parameters:
                del self._parameters["bias"]
            self.int8_impl = self._int8_impl
            return

        # Fallback to normal float loading
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._int8_ready and self._int8_impl is not None:
            return self._int8_impl(x)
        return super().forward(x)


@contextmanager
def patched_linear_for_int8_loading(use_triton: bool = True, default_block_size: int = 128):
    """
    Temporarily patch torch.nn.Linear so model construction uses Int8LinearProxy.
    """
    orig_linear = nn.Linear

    Int8LinearProxy.DEFAULT_USE_TRITON = bool(use_triton)
    Int8LinearProxy.DEFAULT_BLOCK_SIZE = int(default_block_size)
    nn.Linear = Int8LinearProxy  # type: ignore
    try:
        yield
    finally:
        nn.Linear = orig_linear  # type: ignore


def load_int8_state_dict(safetensors_path: str, force_scale_float32: bool = True) -> Dict[str, torch.Tensor]:
    """
    Load an INT8 safetensors file into a CPU state_dict.

    If force_scale_float32 is True, cast weight_scale/scale_weight tensors to float32.
    """
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"Quantized safetensors not found: {safetensors_path}")

    state_dict: Dict[str, torch.Tensor] = {}
    with safe_open(safetensors_path, framework="pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if force_scale_float32 and (k.endswith(".weight_scale") or k.endswith(".scale_weight")):
                t = t.to(torch.float32)
            state_dict[k] = t
    return state_dict


def _parse_comfy_quant(tensor: torch.Tensor) -> Int8Meta:
    try:
        data = bytes(tensor.tolist()).decode("utf-8")
        meta = json.loads(data)
    except Exception:
        return Int8Meta()

    fmt = meta.get("format", "int8_tensorwise")
    block_size = meta.get("group_size", 128)
    return Int8Meta(format=fmt, block_size=block_size)


def _resolve_parent(model: nn.Module, name: str) -> Tuple[nn.Module, str]:
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def apply_quantized_weights(
    model: nn.Module,
    safetensors_path: str,
    device: Optional[str] = None,
    use_triton: bool = True,
    default_block_size: int = 128,
) -> Tuple[list[str], list[str]]:
    """
    Load quantized weights from a safetensors file and patch INT8 Linear layers.
    """
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"Quantized safetensors not found: {safetensors_path}")

    logger.info(f"Loading INT8 safetensors: {safetensors_path}")
    _INT8_STATS["apply_calls"] += 1

    with safe_open(safetensors_path, framework="pt") as f:
        keys = set(f.keys())

        # Identify INT8 modules by weight dtype
        int8_modules: Set[str] = set()
        for k in keys:
            if k.endswith(".weight"):
                t = f.get_tensor(k)
                if t.dtype == torch.int8:
                    int8_modules.add(k[:-len(".weight")])

        # Replace Linear modules with INT8 versions
        replaced = 0
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            weight_key = f"{name}.weight"
            if weight_key not in keys:
                continue
            if name not in int8_modules:
                continue

            w = f.get_tensor(weight_key)
            bias = f.get_tensor(f"{name}.bias") if f"{name}.bias" in keys else None
            scale = f.get_tensor(f"{name}.weight_scale") if f"{name}.weight_scale" in keys else None
            if scale is None:
                raise RuntimeError(f"Missing weight_scale for INT8 layer: {name}")

            meta = _parse_comfy_quant(f.get_tensor(f"{name}.comfy_quant")) if f"{name}.comfy_quant" in keys else Int8Meta()
            block_size = meta.block_size or default_block_size
            quant_format = meta.format

            # Expand tensorwise scale to blockwise matrix if needed
            if scale.dim() == 0 and quant_format == "int8_tensorwise":
                scale = Int8Linear._expand_tensorwise_scale(scale, w, block_size)

            new_mod = Int8Linear(
                weight_int8=w,
                weight_scale=scale,
                bias=bias,
                block_size=block_size,
                quant_format=quant_format,
                use_triton=use_triton,
            )
            if device is not None:
                new_mod = new_mod.to(device)

            parent, child = _resolve_parent(model, name)
            setattr(parent, child, new_mod)
            replaced += 1

        # Load all non-INT8 tensors (float weights, norms, embeddings, etc.)
        float_state: Dict[str, torch.Tensor] = {}
        for k in keys:
            if k.endswith(".weight_scale") or k.endswith(".comfy_quant"):
                continue
            if any(k.startswith(prefix + ".") for prefix in int8_modules):
                # Skip INT8 module tensors (weight/bias handled above)
                continue
            t = f.get_tensor(k)
            if t.dtype == torch.int8:
                continue
            float_state[k] = t

    missing, unexpected = model.load_state_dict(float_state, strict=False)
    _INT8_STATS["int8_layers_replaced"] += replaced
    if missing:
        logger.debug(f"Missing keys after load (expected for INT8): {len(missing)}")
    if unexpected:
        logger.info(f"Unexpected keys after load: {len(unexpected)}")
    return missing, unexpected
