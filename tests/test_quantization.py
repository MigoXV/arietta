from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

from lightning.pytorch import Trainer
from lightning.pytorch.cli import LightningCLI
import pytest
from safetensors.torch import load_file, save_file
import torch
from torch import nn

from arietta.criterions.choice import ChoiceCriterion
from arietta.exporting.export import export_checkpoint, save_deployment
from arietta.evaluation.evaluate import evaluate
from arietta.models.modeling import load_pretrained
from arietta.models.precision import deployment_copy
from arietta.models.processing import MODEL_KEYS
from arietta.models.quantization import (
    Int8Linear,
    Int8QAT,
    QATLinear,
    fake_quantize_rows,
    rotary_pos_emb_fp32,
    align_rotary_precision,
    attention_context,
)
from arietta.tasks.data import DecisionDataModule
from arietta.tasks.sft_dataset import SPLITS, build_tiny_sft
from arietta.tasks.task import FullFinetuneTask, LoraFinetuneTask, ScratchTask


def test_fake_quantization_values_and_ste():
    values = torch.tensor([[0.0, 0.0, 0.0], [-0.9, 0.2, 0.8]], requires_grad=True)
    result = fake_quantize_rows(values)
    scale = values.detach().abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127
    expected = (values.detach() / scale).round().clamp(-127, 127) * scale
    torch.testing.assert_close(result, expected)
    result.sum().backward()
    torch.testing.assert_close(values.grad, torch.ones_like(values))


def test_qat_respects_mixed_precision_output():
    source = nn.Linear(64, 96)
    values = torch.randn(7, 64, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = QATLinear(source)(values)
        actual = Int8Linear.from_float(source)(values)
    assert output.dtype == actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, output, atol=0.008, rtol=0.008)
    output.float().sum().backward()
    assert source.weight.grad is not None and values.grad is not None


def test_qat_attention_backend_is_scoped(repository):
    def flags():
        return tuple(
            enabled()
            for enabled in (
                torch.backends.cuda.math_sdp_enabled,
                torch.backends.cuda.flash_sdp_enabled,
                torch.backends.cuda.mem_efficient_sdp_enabled,
                torch.backends.cuda.cudnn_sdp_enabled,
            )
        )

    before = flags()
    task = FullFinetuneTask(str(repository), ChoiceCriterion(), quantization=Int8QAT())
    with attention_context(task.model.decision):
        assert flags() == (True, False, False, False)
    assert flags() == before
    ordinary = FullFinetuneTask(str(repository), ChoiceCriterion())
    with attention_context(ordinary.model.decision):
        assert flags() == before


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rotary_precision_matches_complex_rotation(dtype):
    generator = torch.Generator().manual_seed(42)
    query = torch.randn(2, 3, 7, 16, generator=generator).to(dtype)
    key = torch.randn(2, 3, 7, 16, generator=generator).to(dtype)
    angles = torch.randn(2, 7, 8, generator=generator)
    cos = angles.cos().repeat(1, 1, 2).to(dtype)
    sin = angles.sin().repeat(1, 1, 2).to(dtype)
    rotation = torch.complex(cos[..., :8].float(), sin[..., :8].float())[:, None]
    actual = rotary_pos_emb_fp32(query, key, cos, sin)
    for source, rotated in zip((query, key), actual):
        complex_values = torch.complex(source[..., :8].float(), source[..., 8:].float())
        expected = complex_values * rotation
        expected = torch.cat((expected.real, expected.imag), -1).to(dtype)
        torch.testing.assert_close(rotated, expected, atol=0, rtol=0)
        first, second = source.chunk(2, -1)
        legacy = source * cos[:, None] + torch.cat((-second, first), -1) * sin[:, None]
        assert not torch.equal(legacy, rotated)


@pytest.mark.parametrize("padded", [False, True])
def test_legacy_sdpa_adapter_matches_training_attention(padded):
    from transformers import ModernBertConfig
    from transformers.models.modernbert.modeling_modernbert import ModernBertAttention

    cfg = ModernBertConfig(
        hidden_size=16,
        num_attention_heads=2,
        num_hidden_layers=1,
        reference_compile=False,
    )
    cfg._attn_implementation = "sdpa"
    reference = ModernBertAttention(cfg, layer_idx=0).eval()
    legacy = nn.Module()
    legacy.Wqkv, legacy.Wo = reference.Wqkv, reference.Wo
    legacy.num_heads, legacy.head_dim, legacy.all_head_size = 2, 8, 16
    legacy.local_attention = (-1, -1)
    legacy.attention_dropout = 0.0
    legacy.out_drop, legacy.config = nn.Identity(), cfg
    angles = torch.randn(1, 5, 4)
    cos, sin = angles.cos().repeat(1, 1, 2), angles.sin().repeat(1, 1, 2)
    calls = []

    def rotary(value, position_ids):
        calls.append(value.dtype)
        return cos.to(value.dtype), sin.to(value.dtype)

    legacy.rotary_emb = rotary
    align_rotary_precision(
        SimpleNamespace(encoder=SimpleNamespace(layers=[SimpleNamespace(attn=legacy)]))
    )
    hidden = torch.randn(1, 5, 16)
    mask = torch.zeros(1, 1, 5, 5)
    if padded:
        mask[..., -1] = -torch.finfo(torch.float32).max
    expected_mask = mask == 0 if padded else None
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        expected = reference(
            hidden, position_embeddings=(cos, sin), attention_mask=expected_mask
        )[0]
        actual = legacy(
            hidden, attention_mask=mask, position_ids=torch.arange(5)[None]
        )[0]
    assert calls == [torch.float32]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "shape,out_features", [((1, 33), 17), ((2, 7, 64), 96), ((0, 64), 32)]
)
def test_real_int8_linear_matches_fake_quantization(shape, out_features):
    source = nn.Linear(shape[-1], out_features)
    values = torch.randn(*shape)
    qat = QATLinear(source)
    with torch.inference_mode():
        expected = qat(values)
        packed = Int8Linear.from_float(source)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as profile:
            actual = packed(values)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    assert packed.qweight.dtype == torch.int8
    assert packed.weight_scales.dtype == torch.float32
    assert not any(isinstance(module, nn.Linear) for module in packed.modules())
    if values.numel():
        assert "aten::_int_mm" in {event.key for event in profile.key_averages()}
        assert "aten::linear" not in {event.key for event in profile.key_averages()}
        alone = packed(values.reshape(-1, shape[-1])[:1])
        torch.testing.assert_close(
            alone, actual.reshape(-1, out_features)[:1], atol=1e-6, rtol=1e-6
        )


@pytest.mark.parametrize("cls", [FullFinetuneTask, LoraFinetuneTask, ScratchTask])
def test_qat_fit_resume_convert_export(cls, repository, tmp_path):
    if cls is ScratchTask:
        (repository / "model.safetensors").unlink()
    source = Path(__file__).resolve().parents[1] / "examples/tiny-sft/cases.jsonl"
    data = tmp_path / "data"
    build_tiny_sft(source, data)
    dm = DecisionDataModule(
        str(data),
        str(repository),
        required_splits=list(SPLITS),
        batch_size=6,
        max_len=128,
        head_max_len=96,
    )
    kwargs = {"dropout": 0.0, "rank": 2} if cls is LoraFinetuneTask else {}
    task = cls(str(repository), ChoiceCriterion(), quantization=Int8QAT(), **kwargs)
    scorer_module = task.model.decision.scorer
    if cls is LoraFinetuneTask:
        scorer_module = scorer_module.modules_to_save["default"]
    scorer = scorer_module[1].weight.detach().clone()
    act_head = task.model.decision.act_head[0].weight.detach().clone()
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(task, dm)
    assert trainer.global_step == 2
    assert not torch.equal(scorer, scorer_module[1].weight)
    if cls is LoraFinetuneTask:
        assert any(
            bool(parameter.detach().abs().max() > 0)
            for name, parameter in task.model.named_parameters()
            if ".lora_B." in name
        )
    torch.testing.assert_close(
        act_head, task.model.decision.act_head[0].weight, rtol=0, atol=0
    )
    assert "val_deploy_loss" in trainer.callback_metrics
    assert all(
        int(module.qat_batches) > 0
        for module in task.model.modules()
        if isinstance(module, QATLinear)
    )
    checkpoint = tmp_path / "last.ckpt"
    trainer.save_checkpoint(checkpoint)
    saved = torch.load(checkpoint, weights_only=True)
    assert "quantization" not in saved["hyper_parameters"]
    with pytest.raises(ValueError, match="QAT configuration mismatch"):
        cls.load_from_checkpoint(
            checkpoint, criterion=ChoiceCriterion(), quantization=None
        )
    restored = cls.load_from_checkpoint(checkpoint, criterion=ChoiceCriterion())
    assert restored.quantization.signature() == task.quantization.signature()
    assert restored.state_dict().keys() == task.state_dict().keys()
    for key in task.state_dict():
        torch.testing.assert_close(
            restored.state_dict()[key], task.state_dict()[key], rtol=0, atol=0
        )
    resumed = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        limit_train_batches=2,
        limit_val_batches=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    resumed.fit(restored, dm, ckpt_path=checkpoint)
    assert resumed.global_step == 4
    deployed = deployment_copy(task.model, "fp32")
    assert not any(isinstance(module, QATLinear) for module in deployed.modules())
    assert (
        len([module for module in deployed.modules() if isinstance(module, Int8Linear)])
        == 4
    )
    batch = next(iter(dm.val_dataloader()))
    with torch.inference_mode():
        expected = deployed(**{key: batch[key] for key in MODEL_KEYS}).logits
        fake = task.model.eval()(**{key: batch[key] for key in MODEL_KEYS}).logits
    torch.testing.assert_close(expected, fake, atol=2e-5, rtol=2e-4)
    for format in ("service", "hf"):
        destination = tmp_path / format
        export_checkpoint(checkpoint, destination, format)
        loaded = load_pretrained(str(destination)).eval()
        with torch.inference_mode():
            actual = loaded(**{key: batch[key] for key in MODEL_KEYS}).logits
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert (
            sum(
                tensor.dtype == torch.int8
                for tensor in load_file(destination / "model.safetensors").values()
            )
            == 4
        )
        assert not any(
            "qat_batches" in key or ".inner." in key for key in loaded.state_dict()
        )
        if format == "service":
            report = evaluate(
                str(destination),
                str(data / "data/validation-00000-of-00001.parquet"),
                "validation",
                state_serialization="verbatim",
            )
            assert len(report["rows"]) == 12
            assert report["quantization"]["sdpa_backend"] == "math"
            assert all(math.isfinite(m["nll"]) for m in report["metrics"].values())
    if cls is LoraFinetuneTask:
        with pytest.raises(ValueError, match="merged and converted"):
            export_checkpoint(checkpoint, tmp_path / "adapter", "adapter")


def test_qat_lora_rejects_dropout_and_unsupported_exclusions(repository):
    with pytest.raises(ValueError, match="dropout=0"):
        LoraFinetuneTask(str(repository), ChoiceCriterion(), quantization=Int8QAT())
    with pytest.raises(ValueError, match="unsupported or missing"):
        FullFinetuneTask(
            str(repository),
            ChoiceCriterion(),
            quantization=Int8QAT(["decision.scorer"]),
        )


def test_qat_exclusion_and_low_precision_buffers(repository, tmp_path):
    task = FullFinetuneTask(
        str(repository),
        ChoiceCriterion(),
        quantization=Int8QAT(["encoder.layers.0.attn.Wo"]),
    )
    for dtype in ("fp16", "bf16"):
        deployed = deployment_copy(task.model, dtype)
        assert isinstance(deployed.decision.encoder.layers[0].attn.Wo, nn.Linear)
        packed = [m for m in deployed.modules() if isinstance(m, Int8Linear)]
        assert len(packed) == 3
        assert all(m.qweight.dtype == torch.int8 for m in packed)
        assert all(m.weight_scales.dtype == torch.float32 for m in packed)
        destination = tmp_path / dtype
        save_deployment(task.model, repository, destination, dtype)
        loaded = load_pretrained(str(destination))
        for key, value in deployed.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[key], value, atol=0, rtol=0)
        with pytest.raises(ValueError, match="floating-point pretrained"):
            FullFinetuneTask(str(destination), ChoiceCriterion())


@pytest.mark.parametrize("corruption", ["dtype", "scale", "metadata"])
def test_quantized_repository_rejects_invalid_assets(repository, tmp_path, corruption):
    task = FullFinetuneTask(str(repository), ChoiceCriterion(), quantization=Int8QAT())
    destination = tmp_path / "quantized"
    save_deployment(task.model, repository, destination, "fp32")
    if corruption == "metadata":
        config = json.loads((destination / "config.json").read_text())
        config["quantization"]["schema_version"] = 99
        (destination / "config.json").write_text(json.dumps(config))
    else:
        tensors = load_file(destination / "model.safetensors")
        key = next(
            key
            for key in tensors
            if key.endswith("qweight" if corruption == "dtype" else "weight_scales")
        )
        tensors[key] = (
            tensors[key].float()
            if corruption == "dtype"
            else torch.zeros_like(tensors[key])
        )
        save_file(tensors, destination / "model.safetensors")
    with pytest.raises(ValueError):
        load_pretrained(str(destination))


def test_qat_native_cli(repository, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["arietta-train"])
    cli = LightningCLI(
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=None,
        run=False,
        args=[
            "--config",
            str(Path(__file__).resolve().parents[1] / "examples/qat/lora-bf16.yaml"),
            f"--model.init_args.pretrained_model_path={repository}",
            f"--data.init_args.tokenizer_path={repository}",
            "--trainer.accelerator=cpu",
            "--trainer.devices=1",
            "--trainer.precision=32-true",
            "--model.init_args.deployment_dtype=fp32",
            "--trainer.logger=false",
            "--trainer.callbacks=[]",
            f"--trainer.default_root_dir={tmp_path}",
        ],
    )
    assert isinstance(cli.model.quantization, Int8QAT)
    assert cli.model.hparams.dropout == 0
    assert any(isinstance(module, QATLinear) for module in cli.model.modules())
