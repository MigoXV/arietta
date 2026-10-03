from pathlib import Path
import pytest
import torch
from arietta.models.processing import (
    normalize_question,
    normalize_target,
    checked_sequence,
)
from arietta.models.modeling import load_pretrained, load_tokenizer
from arietta.models.precision import deployment_copy
from arietta.tasks.data import DecisionDataModule
from arietta.tasks.task import LoraFinetuneTask
from arietta.criterions.choice import ChoiceCriterion
from arietta.exporting.export import save_deployment
from arietta.evaluation.evaluate import evaluate, calibrate


@pytest.mark.parametrize(
    "target",
    [
        {"probability": -0.1},
        {"probability": float("inf")},
        {"distribution": [1.0]},
        {"distribution": [True, False]},
        {"other": 0.5},
    ],
)
def test_invalid_soft_targets(target):
    q = normalize_question({"type": "noul", "instructions": "a"})
    with pytest.raises(ValueError):
        normalize_target({"target": target}, q)


def test_noul_probability_and_mutual_exclusion():
    q = normalize_question({"type": "noul", "instructions": "a"})
    assert normalize_target({"target": {"probability": 0.8}}, q) == (
        [0.19999999999999996, 0.8],
        -1,
    )
    with pytest.raises(ValueError):
        normalize_target({"label": "a", "target": {"probability": 0.8}}, q)


def test_data_entry_once_and_split_leak(repository, data_files, monkeypatch):
    import arietta.tasks.data as module

    original = module.load_dataset
    calls = []

    def counted(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(module, "load_dataset", counted)
    dm = DecisionDataModule("json", str(repository), data_files, batch_size=2)
    dm.setup()
    dm.setup()
    assert len(calls) == 1
    assert calls[0] == dict(
        path="json",
        name=None,
        data_dir=None,
        data_files=data_files,
        cache_dir=None,
        revision=None,
        streaming=False,
    )
    p = Path(data_files["validation"])
    p.write_text(p.read_text().replace("validationg0", "traing0"))
    with pytest.raises(ValueError, match="cross-split"):
        DecisionDataModule("json", str(repository), data_files).setup()


def test_no_silent_truncation(repository):
    tok = load_tokenizer(repository)
    q = normalize_question(
        {"type": "choice", "instructions": "a", "criteria": ["a", "b"]}
    )
    with pytest.raises(ValueError, match="state_token_budget"):
        checked_sequence(tok, "a " * 200, q, {"max_len": 128, "head_max_len": 96})
    with pytest.raises(ValueError, match="reserved MASK"):
        checked_sequence(tok, "[MASK]", q, {"max_len": 128, "head_max_len": 96})


def test_lora_frozen_policy_and_clone(repository):
    task = LoraFinetuneTask(str(repository), ChoiceCriterion(), rank=2)
    assert not any(p.requires_grad for p in task.model.decision.act_head.parameters())
    assert "ModulesToSave" not in type(task.model.decision.act_head).__name__
    cloned = deployment_copy(task.model, "bf16")
    assert all(p.dtype == torch.bfloat16 for p in cloned.parameters())
    assert any("lora_" in n for n, _ in task.model.named_parameters())
    assert not any("lora_" in n for n, _ in cloned.named_parameters())


@pytest.mark.parametrize("format", ["service", "hf"])
def test_native_strict_keys(repository, tmp_path, format):
    from safetensors.torch import load_file, save_file

    out = tmp_path / "native"
    save_deployment(load_pretrained(str(repository)), repository, out, "fp16", format)
    weights = load_file(out / "model.safetensors")
    weights.pop(("decision." if format == "hf" else "") + "scorer.1.bias")
    save_file(weights, out / "model.safetensors")
    with pytest.raises(RuntimeError, match="Missing key"):
        load_pretrained(str(out))


def test_evaluate_and_calibrate(repository, data_files):
    result = evaluate(
        str(repository),
        data_files["validation"],
        "validation",
        state_serialization="verbatim",
    )
    assert result["dtype"] == "fp32" and result["model_fingerprint"]
    assert calibrate(result["rows"]) > 0


def test_scratch_uses_master_parameters_even_with_low_precision_metadata(
    repository, tmp_path
):
    from arietta.tasks.task import ScratchTask

    out = tmp_path / "scratch"
    save_deployment(load_pretrained(str(repository)), repository, out, "bf16")
    (out / "model.safetensors").unlink()
    task = ScratchTask(str(out), ChoiceCriterion(), deployment_dtype="bf16")
    assert all(p.dtype == torch.float32 for p in task.parameters())


def test_all_skipped_optimizer_steps_are_not_training_success():
    from types import SimpleNamespace
    from arietta.tasks.task import DecisionTask

    task = SimpleNamespace(
        trainer=SimpleNamespace(optimizers=[SimpleNamespace(state={})])
    )
    with pytest.raises(RuntimeError, match="No optimizer update"):
        DecisionTask.on_train_end(task)


def test_export_refuses_overflow_only_checkpoint(tmp_path):
    from arietta.exporting.export import export_checkpoint

    checkpoint = tmp_path / "skipped.ckpt"
    torch.save(
        {
            "task_class": "FullFinetuneTask",
            "global_step": 2,
            "optimizer_states": [{"state": {}}],
        },
        checkpoint,
    )
    with pytest.raises(ValueError, match="no optimizer updates"):
        export_checkpoint(checkpoint, tmp_path / "export")
