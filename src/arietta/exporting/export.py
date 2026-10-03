from __future__ import annotations
import hashlib
import json
import shutil
from pathlib import Path
import torch
from safetensors.torch import save_file
from arietta.models.modeling import load_pretrained, load_tokenizer
from arietta.models.precision import deployment_copy, DTYPES
from arietta.criterions.choice import ChoiceCriterion


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fresh_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"output directory must be empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def save_deployment(model, source, destination, dtype, format="service"):
    if dtype not in DTYPES or format not in {"service", "hf"}:
        raise ValueError("unsupported dtype or export format")
    source, destination = Path(source), Path(destination)
    fresh_directory(destination)
    model = deployment_copy(model, dtype).cpu()
    tokenizer = load_tokenizer(source)
    if (source / "LICENSE").is_file():
        shutil.copy2(source / "LICENSE", destination / "LICENSE")
    (destination / "README.md").write_text(
        f"# Arietta 导出模型\n\n部署精度：{dtype}；格式：{format}。\n\n"
        "温度已重置为 1，需用独立 calibration split 重新校准。"
        "模型来源与检查点摘要见 export.json；工程冒烟权重不能视为领域训练成果。\n"
    )
    if format == "hf":
        model.save_pretrained(destination, safe_serialization=True)
        tokenizer.save_pretrained(destination / "tokenizer")
        # HF wrapper configs originating from native models also need the root tokenizer.
        tokenizer.save_pretrained(destination)
    else:
        tokenizer.save_pretrained(destination)
        native = getattr(model.config, "native_config", None)
        token_cfg = (
            native["tokenizer"]
            if native
            else {
                key: getattr(tokenizer, key)
                for key in (
                    "pad_token",
                    "cls_token",
                    "sep_token",
                    "mask_token",
                    "unk_token",
                )
            }
        )
        config = {
            "format_version": 1,
            "encoder": model.decision.encoder.config.to_dict(),
            "decision_head": {
                "layers": len(model.decision.head.layers),
                "num_actions": model.decision.act_head[-1].out_features,
            },
            "input_limits": {
                k: model.config.agent_config[k] for k in ("max_len", "head_max_len")
            },
            "tokenizer": token_cfg,
            "calibration": {
                "temperature": [1.0, 1.0, 1.0],
                "temperature_by_options": {},
            },
            "deployment_dtype": dtype,
        }
        config["encoder"]["dtype"] = str(DTYPES[dtype]).removeprefix("torch.")
        (destination / "config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False)
        )
        save_file(
            {
                k: v.detach().contiguous()
                for k, v in model.decision.state_dict().items()
            },
            str(destination / "model.safetensors"),
        )
    # Fail at delivery time if the repository cannot be read strictly.
    reloaded = load_pretrained(str(destination), dtype=DTYPES[dtype])
    for key, value in model.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[key], value, rtol=0, atol=0)
    return destination


def export_checkpoint(
    checkpoint_path: Path,
    destination: Path,
    format: str = "service",
    dtype: str | None = None,
):
    from arietta.tasks.task import LoraFinetuneTask, FullFinetuneTask, ScratchTask

    if format not in {"service", "hf", "adapter"}:
        raise ValueError(f"unsupported export format: {format}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    classes = {
        cls.__name__: cls for cls in (LoraFinetuneTask, FullFinetuneTask, ScratchTask)
    }
    cls = classes.get(checkpoint.get("task_class"))
    if cls is None:
        raise ValueError("unsupported checkpoint task_class")
    if format == "adapter" and cls is not LoraFinetuneTask:
        raise ValueError("adapter export requires LoRA checkpoint")
    optimizer_states = checkpoint.get("optimizer_states")
    if optimizer_states is not None and not any(
        float(state.get("step", 0)) > 0
        for optimizer in optimizer_states
        for state in optimizer.get("state", {}).values()
    ):
        raise ValueError(
            "checkpoint has no optimizer updates; refusing an overflow-only run"
        )
    task = cls.load_from_checkpoint(
        str(checkpoint_path),
        map_location="cpu",
        criterion=ChoiceCriterion(),
        strict=True,
    )
    dtype = dtype or task.deployment_dtype
    if dtype != task.deployment_dtype:
        raise ValueError(
            "export dtype must match the precision validated during training; use a separate run"
        )
    source = Path(
        task.hparams.get("pretrained_model_path", task.hparams.get("model_config_path"))
    )
    if format == "adapter":
        fresh_directory(destination)
        task.model.save_pretrained(destination, safe_serialization=True)
        load_tokenizer(source).save_pretrained(destination)
    else:
        save_deployment(task.model, source, destination, dtype, format)
    (destination / "export.json").write_text(
        json.dumps(
            {
                "format": format,
                "dtype": dtype,
                "task_class": cls.__name__,
                "checkpoint_sha256": digest_file(checkpoint_path),
                "source": str(source),
                "source_fingerprint": task.source_fingerprint,
                "calibration": "reset; recalibration required on held-out calibration split",
                "adapter_parameter_dtype": "fp32 training/master parameters"
                if format == "adapter"
                else None,
            },
            indent=2,
        )
    )
    return destination
