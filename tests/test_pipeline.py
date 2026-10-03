import inspect
import json
import pytest
import torch
from lightning.pytorch import Trainer
from arietta.models.modeling import load_pretrained
from arietta.models.processing import normalize_question, normalize_target, MODEL_KEYS
from arietta.tasks.data import DecisionDataModule
from arietta.tasks.task import FullFinetuneTask, LoraFinetuneTask, ScratchTask
from arietta.criterions.choice import ChoiceCriterion
from arietta.exporting.export import export_checkpoint, save_deployment
from arietta.configs.run_config import save_immutable


def test_labels():
    for kind, criteria, label, distribution in [
        ("choice", ["a", "b"], "a", [0.3, 0.7]),
        ("score", ["low", "high"], 1, [0.2, 0.8]),
        ("noul", None, True, [0.4, 0.6]),
    ]:
        q = normalize_question(dict(type=kind, instructions="a", criteria=criteria))
        target, index = normalize_target({"target": {"label": label}}, q)
        assert index in (0, 1) and sum(target) == 1
        assert normalize_target({"target": {"distribution": distribution}}, q) == (
            distribution,
            -1,
        )
        for bad in (
            {"distribution": [0.2, 0.2]},
            {"distribution": [float("nan"), 1]},
            {"label": label, "distribution": distribution},
        ):
            with pytest.raises(ValueError):
                normalize_target({"target": bad}, q)
    with pytest.raises(ValueError):
        normalize_question(
            dict(type="choice", instructions="a", criteria=list(map(str, range(17))))
        )
    with pytest.raises(ValueError):
        normalize_target(
            {"target": {"label": True}},
            normalize_question(
                dict(type="score", instructions="a", criteria=["a", "b"])
            ),
        )


def test_soft_mask():
    loss = ChoiceCriterion()(
        torch.tensor([[0.0, 0.0, 100.0]]),
        torch.tensor([[0.25, 0.75, 0.0]]),
        torch.tensor([[True, True, False]]),
    )
    torch.testing.assert_close(loss, torch.tensor(2.0).log())


@pytest.mark.parametrize("cls", [FullFinetuneTask, LoraFinetuneTask, ScratchTask])
def test_fit_resume_export(cls, repository, data_files, tmp_path):
    if cls is ScratchTask:
        (repository / "model.safetensors").unlink()
    for split, path in data_files.items():
        rows = []
        for i, (kind, criteria, label) in enumerate(
            [
                ("choice", {"a": None, "b": None}, "a"),
                ("score", ["a", "b"], 1),
                ("noul", None, True),
            ]
        ):
            for soft in (False, True):
                rows.append(
                    dict(
                        id=f"{split}{i}{soft}",
                        group_id=split,
                        family_id=split,
                        task=kind,
                        split=split,
                        state="a b",
                        question=json.dumps(
                            dict(type=kind, instructions="a", criteria=criteria)
                        ),
                        target=json.dumps(
                            {"distribution": [0.2, 0.8]} if soft else {"label": label}
                        ),
                    )
                )
        from pathlib import Path

        Path(path).write_text("\n".join(json.dumps(r) for r in rows))
    dm = DecisionDataModule(
        "json", str(repository), data_files, batch_size=6, max_len=128, head_max_len=96
    )
    task = cls(str(repository), ChoiceCriterion())
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(task, dm)
    assert "val_deploy_loss" in trainer.callback_metrics
    assert not any("_deploy_model" in k for k in task.state_dict())
    path = tmp_path / "last.ckpt"
    trainer.save_checkpoint(path)
    restored = cls.load_from_checkpoint(path, criterion=ChoiceCriterion())
    trainer2 = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    trainer2.fit(restored, dm, ckpt_path=path)
    assert trainer2.global_step == 2
    for fmt in ("hf", "service"):
        export_checkpoint(path, tmp_path / fmt, fmt)
        model = load_pretrained(str(tmp_path / fmt), dtype=torch.float32).eval()
        original = task.model
        if hasattr(original, "merge_and_unload"):
            original = original.merge_and_unload()
        original.eval()
        batch = next(iter(dm.val_dataloader()))
        with torch.no_grad():
            torch.testing.assert_close(
                original(**{k: batch[k] for k in MODEL_KEYS}).logits,
                model(**{k: batch[k] for k in MODEL_KEYS}).logits,
            )
    if cls is LoraFinetuneTask:
        export_checkpoint(path, tmp_path / "adapter", "adapter")


def test_native_dtype_roundtrip(repository, tmp_path):
    from safetensors import safe_open

    model = load_pretrained(str(repository), dtype=torch.float32)
    for dtype in ("fp16", "bf16"):
        out = tmp_path / dtype
        save_deployment(model, repository, out, dtype)
        reloaded = load_pretrained(str(out))
        assert (
            next(reloaded.parameters()).dtype
            == {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
        )
        with safe_open(out / "model.safetensors", framework="pt") as f:
            assert f.get_tensor("temperature").dtype == torch.float32
        assert all(
            b.dtype == torch.float32
            for n, b in reloaded.named_buffers()
            if "inv_freq" in n
        )


def test_source_and_drift(repository, tmp_path):
    for cls in (FullFinetuneTask, LoraFinetuneTask):
        assert "model_config_path" not in inspect.signature(cls).parameters
    with pytest.raises(ValueError, match="must not contain weights"):
        ScratchTask(str(repository), ChoiceCriterion())
    path = tmp_path / "resolved.yaml"
    save_immutable(path, "model: a\nckpt_path: null\n")
    save_immutable(path, "model: a\nckpt_path: last.ckpt\n")
    with pytest.raises(RuntimeError):
        save_immutable(path, "model: b\n")
