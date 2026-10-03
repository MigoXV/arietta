"""固定 W8A8 数值规则：动态逐 token 激活、逐输出通道权重、真实整数 GEMM。"""

from __future__ import annotations

import re
from contextlib import nullcontext
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

_LINEAR_PATH = re.compile(r"encoder\.layers\.\d+\.(attn\.(Wqkv|Wo)|mlp\.(Wi|Wo))")
_SCHEME = {
    "schema_version": 1,
    "scheme": "dynamic_w8a8",
    "kernel": "torch._int_mm",
    "activation_granularity": "per_token",
    "weight_granularity": "per_output_channel",
    "quant_min": -127,
    "quant_max": 127,
    "padding_multiple": 32,
    "rotary_compute_dtype": "float32",
    "sdpa_backend": "math",
}


def attention_context(decision: nn.Module):
    if getattr(decision, "arietta_sdpa_backend", None) == "math":
        return sdpa_kernel([SDPBackend.MATH])
    return nullcontext()


def rotary_pos_emb_fp32(query, key, cos, sin):
    """低精度输入在 FP32 中旋转，结果回到原精度。"""
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)

    def rotate(x):
        first, second = x.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    return tuple(
        (value.float() * cos + rotate(value.float()) * sin).to(value.dtype)
        for value in (query, key)
    )


def _legacy_sdpa_forward(
    module,
    hidden_states,
    attention_mask=None,
    sliding_window_mask=None,
    position_ids=None,
    output_attentions=False,
    **kwargs,
):
    """4.57 SDPA 接口，RoPE 乘加遵循训练侧 5.18 的 FP32 规则。"""
    if output_attentions:
        raise ValueError("dynamic_w8a8 legacy SDPA does not return attention weights")
    batch = hidden_states.shape[0]
    qkv = module.Wqkv(hidden_states).view(
        batch, -1, 3, module.num_heads, module.head_dim
    )
    # 5.18 builds positions from the embedding/residual stream (often FP32 under
    # autocast), while 4.57 uses BF16 QKV and rounds cos/sin before rotation.
    cos, sin = module.rotary_emb(hidden_states, position_ids=position_ids)
    query, key, value = qkv.transpose(3, 1).unbind(dim=2)
    query, key = rotary_pos_emb_fp32(query, key, cos, sin)
    mask = attention_mask if module.local_attention == (-1, -1) else sliding_window_mask
    if mask is not None:
        allowed = mask if mask.dtype == torch.bool else mask == 0
        mask = None if bool(allowed.all()) else allowed
    output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=mask,
        dropout_p=module.attention_dropout if module.training else 0.0,
    )
    output = output.transpose(1, 2).contiguous().view(batch, -1, module.all_head_size)
    return (module.out_drop(module.Wo(output)),)


def align_rotary_precision(decision: nn.Module) -> None:
    # Dynamic INT8 can amplify small attention rounding differences. Pin the
    # same SDPA backend for training and deployment, including the floating head.
    decision.arietta_sdpa_backend = "math"
    for layer in decision.encoder.layers:
        attention = layer.attn
        if hasattr(attention, "rotary_emb"):
            if attention.config._attn_implementation != "sdpa":
                raise ValueError("dynamic_w8a8 legacy ModernBERT requires SDPA")
            attention.forward = MethodType(_legacy_sdpa_forward, attention)


def quantize_rows(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    value = value.float()
    scale = value.detach().abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127
    quantized = (value / scale).round().clamp(-127, 127).to(torch.int8)
    return quantized, scale


def fake_quantize_rows(value: torch.Tensor) -> torch.Tensor:
    """动态范围覆盖整行，使用 STE；scale 不参与梯度。"""
    value = value.float()
    quantized, scale = quantize_rows(value)
    return value + (quantized.float() * scale - value).detach()


def effective_linear(module: nn.Module) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(module, nn.Linear):
        return module.weight, module.bias
    # QAT 必须看到合并后的权重，不能分别量化 base 和未量化的 LoRA 残差。
    from peft.tuners.lora.layer import Linear

    if not isinstance(module, Linear):
        raise ValueError(
            f"QAT requires nn.Linear or PEFT Linear, got {type(module).__name__}"
        )
    base = module.get_base_layer()
    weight = base.weight
    if module.disable_adapters or module.merged:
        return weight, base.bias
    for adapter in module.active_adapters:
        if module.use_dora.get(adapter, False) or module.lora_bias.get(adapter, False):
            raise ValueError("QAT supports plain LoRA without DoRA or adapter bias")
        dropout = module.lora_dropout[adapter]
        if isinstance(dropout, nn.Dropout) and dropout.p != 0:
            raise ValueError(
                "QAT + LoRA requires dropout=0 for merged-weight equivalence"
            )
        weight = weight + module.get_delta_weight(adapter)
    return weight, base.bias


class QATLinear(nn.Module):
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self.register_buffer("qat_batches", torch.zeros((), dtype=torch.long))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output_dtype = (
            torch.get_autocast_dtype(value.device.type)
            if torch.is_autocast_enabled(value.device.type)
            else value.dtype
        )
        with torch.autocast(value.device.type, enabled=False):
            weight, bias = effective_linear(self.inner)
            if self.training:
                self.qat_batches.add_(1)
            output = F.linear(
                fake_quantize_rows(value),
                fake_quantize_rows(weight),
                bias.float() if bias is not None else None,
            )
        return output.to(output_dtype)


class Int8Linear(nn.Module):
    """INT8 常驻权重；CPU/CUDA 都执行 s8 × s8 → s32，不回退浮点 GEMM。"""

    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        k, n = ((in_features + 31) // 32 * 32, (out_features + 31) // 32 * 32)
        self.register_buffer("qweight", torch.zeros(n, k, dtype=torch.int8))
        self.register_buffer("weight_scales", torch.zeros(n, 1, dtype=torch.float32))
        self.register_buffer("bias", torch.zeros(out_features) if bias else None)

    @classmethod
    def from_float(cls, source: nn.Linear) -> Int8Linear:
        result = cls(
            source.in_features, source.out_features, source.bias is not None
        ).to(source.weight.device)
        quantized, scales = quantize_rows(source.weight)
        result.qweight[: source.out_features, : source.in_features].copy_(quantized)
        result.weight_scales[: source.out_features].copy_(scales)
        if source.bias is not None:
            result.bias.copy_(source.bias.detach().float())
        return result

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.device.type not in {"cpu", "cuda"} or not value.is_floating_point():
            raise ValueError(
                "dynamic_w8a8 requires floating-point input on CPU or CUDA"
            )
        if value.shape[-1] != self.in_features:
            raise ValueError("dynamic_w8a8 input feature count mismatch")
        output_dtype = (
            torch.get_autocast_dtype(value.device.type)
            if torch.is_autocast_enabled(value.device.type)
            else value.dtype
        )
        rows = value.reshape(-1, self.in_features)
        if rows.shape[0] == 0:
            return value.new_empty(
                (*value.shape[:-1], self.out_features), dtype=output_dtype
            )
        with torch.autocast(value.device.type, enabled=False):
            quantized, scales = quantize_rows(rows)
            # Torch 2.8 CUDA _int_mm 要求 M >= 32；K/N 统一填充到 32 的倍数。
            m = (
                max(32, (rows.shape[0] + 31) // 32 * 32)
                if value.is_cuda
                else rows.shape[0]
            )
            quantized = F.pad(
                quantized,
                (0, self.qweight.shape[1] - self.in_features, 0, m - rows.shape[0]),
            ).contiguous()
            accum = torch._int_mm(quantized, self.qweight.T)
            output = accum[: rows.shape[0], : self.out_features].float()
            output = output * scales * self.weight_scales[: self.out_features].T
            if self.bias is not None:
                output = output + self.bias
        return output.to(output_dtype).reshape(*value.shape[:-1], self.out_features)


def quantization_metadata(modules: list[str]) -> dict:
    return {**_SCHEME, "modules": sorted(modules)}


def validate_quantization_metadata(metadata: dict) -> list[str]:
    if not isinstance(metadata, dict) or set(metadata) != {*_SCHEME, "modules"}:
        raise ValueError("unsupported quantization metadata fields")
    if any(
        type(metadata[name]) is not type(expected) or metadata[name] != expected
        for name, expected in _SCHEME.items()
    ):
        raise ValueError("unsupported quantization scheme or schema_version")
    modules = metadata["modules"]
    if (
        not isinstance(modules, list)
        or not modules
        or any(
            not isinstance(name, str) or not _LINEAR_PATH.fullmatch(name)
            for name in modules
        )
        or len(set(modules)) != len(modules)
    ):
        raise ValueError(
            "quantization modules must be unique supported encoder linear paths"
        )
    return modules


def install_int8_modules(decision: nn.Module, metadata: dict) -> None:
    for name in validate_quantization_metadata(metadata):
        source = decision.get_submodule(name)
        if not isinstance(source, nn.Linear):
            raise ValueError(
                f"{name}: expected floating-point Linear before INT8 loading"
            )
        owner, _, attribute = name.rpartition(".")
        decision.get_submodule(owner).__setattr__(
            attribute,
            Int8Linear(
                source.in_features, source.out_features, source.bias is not None
            ),
        )
    align_rotary_precision(decision)


def validate_int8_weights(decision: nn.Module) -> None:
    for name, module in decision.named_modules():
        if not isinstance(module, Int8Linear):
            continue
        if (
            module.qweight.dtype != torch.int8
            or module.weight_scales.dtype != torch.float32
        ):
            raise ValueError(f"{name}: INT8 weights and FP32 scales are required")
        if (module.qweight == -128).any() or not torch.isfinite(
            module.weight_scales
        ).all():
            raise ValueError(f"{name}: invalid quantized weight or scale")
        if not (module.weight_scales[: module.out_features] > 0).all():
            raise ValueError(f"{name}: weight scales must be positive")
        if module.bias is not None and not torch.isfinite(module.bias).all():
            raise ValueError(f"{name}: nonfinite quantized bias")


def validate_quantized_state(
    state: dict[str, torch.Tensor], metadata: dict, prefix: str = ""
) -> None:
    for name in validate_quantization_metadata(metadata):
        for suffix, dtype in (
            ("qweight", torch.int8),
            ("weight_scales", torch.float32),
        ):
            key = f"{prefix}{name}.{suffix}"
            if key not in state or state[key].dtype != dtype:
                raise ValueError(f"{key}: missing or invalid quantized tensor dtype")


class Int8QAT:
    """可注入训练策略；未配置时现有浮点训练、导出完全沿用原路径。"""

    def __init__(self, exclude_modules: list[str] | None = None):
        self.exclude_modules = list(exclude_modules or [])
        if any(not isinstance(name, str) for name in self.exclude_modules):
            raise ValueError("QAT exclude_modules must contain strings")
        if len(set(self.exclude_modules)) != len(self.exclude_modules):
            raise ValueError("QAT exclude_modules must be unique")

    def signature(self) -> dict:
        return {
            "schema_version": 1,
            "scheme": "dynamic_w8a8",
            "sdpa_backend": "math",
            "exclude_modules": self.exclude_modules,
        }

    @classmethod
    def from_signature(cls, signature: dict) -> Int8QAT:
        if (
            not isinstance(signature, dict)
            or set(signature)
            != {"schema_version", "scheme", "sdpa_backend", "exclude_modules"}
            or type(signature["schema_version"]) is not int
            or signature["schema_version"] != 1
            or signature["scheme"] != "dynamic_w8a8"
            or signature["sdpa_backend"] != "math"
            or not isinstance(signature["exclude_modules"], list)
        ):
            raise ValueError("unsupported checkpoint QAT configuration")
        return cls(signature["exclude_modules"])

    def prepare(self, model: nn.Module) -> None:
        if getattr(model.config, "arietta_quantization", None):
            raise ValueError("QAT requires a floating-point source repository")
        candidates = {
            name: module
            for name, module in model.decision.named_modules()
            if _LINEAR_PATH.fullmatch(name)
        }
        if set(self.exclude_modules) - set(candidates):
            raise ValueError(
                "QAT exclude_modules contains unsupported or missing paths"
            )
        selected = {
            name: module
            for name, module in candidates.items()
            if name not in self.exclude_modules
        }
        if not selected:
            raise ValueError("QAT selected no encoder linear modules")
        for name, source in selected.items():
            effective_linear(source)
            owner, _, attribute = name.rpartition(".")
            setattr(model.decision.get_submodule(owner), attribute, QATLinear(source))
        align_rotary_precision(model.decision)


def convert_qat(model: nn.Module) -> nn.Module:
    selected = {
        name: module
        for name, module in model.decision.named_modules()
        if isinstance(module, QATLinear)
    }
    if not selected:
        return model
    for name, module in selected.items():
        if not isinstance(module.inner, nn.Linear):
            raise ValueError("merge LoRA before converting QAT to INT8")
        owner, _, attribute = name.rpartition(".")
        setattr(
            model.decision.get_submodule(owner),
            attribute,
            Int8Linear.from_float(module.inner),
        )
    model.config.arietta_quantization = quantization_metadata(list(selected))
    validate_int8_weights(model.decision)
    return model
