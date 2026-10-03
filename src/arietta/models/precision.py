from __future__ import annotations
import copy
import torch

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def move_model(model, device, dtype):
    buffers = {
        name: value
        for name, value in model.named_buffers()
        if value.is_floating_point()
    }
    model.to(device=device, dtype=dtype)
    for name, value in buffers.items():
        owner, _, attr = name.rpartition(".")
        setattr(model.get_submodule(owner), attr, value.to(device=device))
    return model


def deployment_copy(model, dtype):
    result = copy.deepcopy(model)
    if hasattr(result, "merge_and_unload"):
        result = result.merge_and_unload(safe_merge=True)
    result.decision.temperature.fill_(1)
    result = move_model(result, next(result.parameters()).device, DTYPES[dtype])
    result.requires_grad_(False).eval()
    return result
