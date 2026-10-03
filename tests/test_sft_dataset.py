from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

from datasets import (
    DatasetDict,
    get_dataset_config_names,
    get_dataset_split_names,
    load_dataset,
)
from huggingface_hub import DatasetCard
from lightning.pytorch.cli import LightningCLI
import pytest
import torch

from arietta.models.processing import MODEL_KEYS
from arietta.tasks.data import DecisionDataModule
from arietta.tasks.sft_dataset import (
    FEATURES,
    QUESTIONS,
    SPLITS,
    build_tiny_sft,
    validate_cases,
)
from arietta.tasks.task import LoraFinetuneTask

SOURCE = Path(__file__).resolve().parents[1] / "examples/tiny-sft/cases.jsonl"
CONFIG = SOURCE.with_name("lora-bf16.yaml")


@pytest.fixture
def cases():
    return list(load_dataset("json", data_files=str(SOURCE), split="train"))


def test_source_boundary_labels_and_grounding(cases):
    validate_cases(cases)
    by_id = {case["id"]: case for case in cases}
    expected = {
        "permission-03": ("support", 1, False),  # 有条件承诺
        "permission-04": ("undecided", 0, False),  # 转述他人
        "release-02": ("oppose", 2, True),  # 反对但接受具体任务
        "release-04": ("undecided", 0, True),  # 未表态但承诺执行
        "onboarding-03": ("oppose", 1, False),  # 能力不等于承诺
        "notification-04": ("undecided", 0, False),  # 撤回旧立场
    }
    for case_id, labels in expected.items():
        actual = by_id[case_id]["labels"]
        assert tuple(actual[name] for name in QUESTIONS) == labels


@pytest.mark.parametrize("field", ["evidence", "consistency", "family", "duplicate"])
def test_reject_invalid_annotation_and_split_leaks(cases, field):
    if field == "evidence":
        cases[0]["review"]["stance"]["evidence"] = "原文不存在的证据"
    elif field == "consistency":
        cases[0]["labels"]["certainty"] = 0
    elif field == "family":
        cases[-1]["family_id"] = cases[0]["family_id"]
    else:
        cases[-1]["utterance"] = cases[0]["utterance"]
    with pytest.raises(ValueError):
        validate_cases(cases)


def test_build_load_all_types_without_annotation_leak(
    repository, tmp_path, monkeypatch
):
    root = tmp_path / "dataset"
    manifest = build_tiny_sft(SOURCE, root)
    assert manifest["cases"] == 32 and manifest["rows"] == 96
    assert {s: manifest["splits"][s]["rows"] for s in SPLITS} == {
        "train": 60,
        "validation": 12,
        "calibration": 12,
        "test": 12,
    }
    path = str(root.resolve())
    native = load_dataset(path)
    assert isinstance(native, DatasetDict)
    assert set(native) == set(SPLITS)
    assert get_dataset_config_names(path) == ["default"]
    assert get_dataset_split_names(path) == list(SPLITS)
    assert load_dataset(path, split="train[:3]")["id"] == native["train"][:3]["id"]
    card = DatasetCard.load(root / "README.md")
    config = card.data.to_dict()["configs"][0]
    assert config["config_name"] == "default" and config["default"] is True
    assert {item["split"] for item in config["data_files"]} == set(SPLITS)
    infos = card.data.to_dict()["dataset_info"]["splits"]
    for info in infos:
        assert info["num_examples"] == len(native[info["name"]])
        assert info["num_bytes"] == native[info["name"]].data.nbytes
    assert {p.name for p in root.iterdir()} == {"README.md", "data"}
    assert (root.with_name(root.name + "-audit") / "review.md").is_file()

    import arietta.tasks.data as module

    calls = []
    original = module.load_dataset

    def counted(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(module, "load_dataset", counted)
    dm = DecisionDataModule(
        path,
        str(repository),
        dataset_name="default",
        required_splits=list(SPLITS),
        batch_size=3,
    )
    dm.setup()
    dm.setup()
    assert len(calls) == 1
    assert calls[0]["path"] == path and calls[0]["name"] == "default"
    assert calls[0]["data_files"] is None
    group_splits, family_splits = {}, {}
    sources = {
        case["id"]: case
        for case in load_dataset("json", data_files=str(SOURCE), split="train")
    }
    for split in SPLITS:
        assert dm.dataset[split].features == FEATURES
        assert list(dm.dataset[split]) == list(native[split])
        for row in dm.dataset[split]:
            assert set(row) == {
                "id",
                "group_id",
                "family_id",
                "task",
                "split",
                "state",
                "question",
                "target",
            }
            assert set(json.loads(row["target"])) == {"label"}
            case = sources[row["group_id"]]
            task = row["id"].rsplit("/", 1)[1]
            assert json.loads(row["target"])["label"] == case["labels"][task]
            assert (
                row["state"] == f"议题：{case['proposal']}\n发言：{case['utterance']}"
            )
            assert json.loads(row["question"]) == QUESTIONS[task]
            for field, seen in (
                ("group_id", group_splits),
                ("family_id", family_splits),
            ):
                assert row[field] not in seen or seen[row[field]] == split
                seen[row[field]] = split
        batch = next(iter(dm.loader(split)))
        assert set(batch["qtype"].tolist()) == {0, 1, 2}
        assert set(batch["labels"].tolist()) != {-1}
    rows = list(dm.dataset["train"].select([0, 1, 2]))
    first = dm.collator(rows)
    changed = deepcopy(rows)
    for row, label in zip(changed, ["oppose", 0, False]):
        row["target"] = json.dumps({"label": label})
        row["review"] = "这段标注解释不应该进模型 [MASK]"
    second = dm.collator(changed)
    for key in MODEL_KEYS:
        torch.testing.assert_close(first[key], second[key])
    assert not torch.equal(first["targets"], second["targets"])
    assert build_tiny_sft(SOURCE, root) == manifest  # 确定性重建


def test_parquet_can_be_evaluated_through_existing_entry(repository, tmp_path):
    from arietta.evaluation.evaluate import evaluate

    root = tmp_path / "dataset"
    manifest = build_tiny_sft(SOURCE, root)
    result = evaluate(
        str(repository),
        str(root / manifest["splits"]["test"]["path"]),
        "test",
        state_serialization="verbatim",
    )
    assert len(result["rows"]) == 12
    assert {row["question_type"] for row in result["rows"]} == {
        "choice",
        "score",
        "noul",
    }
    assert all(
        result["metrics"][kind]["hard_count"] == 4
        for kind in ("choice", "score", "noul")
    )
    assert result["dataset_sha256"] == manifest["splits"]["test"]["sha256"]


def test_dataset_version_rejects_silent_label_replacement(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    build_tiny_sft(SOURCE, root)
    source = tmp_path / "changed.jsonl"
    source.write_bytes(SOURCE.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="new destination"):
        build_tiny_sft(source, root)
    criteria = dict(reversed(list(QUESTIONS["stance"]["criteria"].items())))
    monkeypatch.setitem(QUESTIONS["stance"], "criteria", criteria)
    with pytest.raises(ValueError, match="new destination"):
        build_tiny_sft(SOURCE, root)


def test_tiny_sft_config_uses_native_cli(repository, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["arietta-train"])
    cli = LightningCLI(
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=None,
        run=False,
        args=[
            "--config",
            str(CONFIG),
            f"--model.init_args.pretrained_model_path={repository}",
            f"--data.init_args.tokenizer_path={repository}",
            "--trainer.accelerator=cpu",
            "--trainer.devices=1",
            "--trainer.precision=32-true",
            "--trainer.logger=false",
            "--trainer.callbacks=[]",
            f"--trainer.default_root_dir={tmp_path}",
        ],
    )
    assert isinstance(cli.model, LoraFinetuneTask)
    assert isinstance(cli.datamodule, DecisionDataModule)
    assert cli.trainer.max_epochs == 5
    assert cli.trainer.limit_train_batches == 1.0
    assert cli.model.hparams.deployment_dtype == "bf16"
    assert cli.datamodule.hparams.dataset_path == "data-bin/tiny-sft-v2"
    assert cli.datamodule.hparams.dataset_name == "default"
    assert cli.datamodule.hparams.data_files is None
    assert cli.datamodule.hparams.required_splits == list(SPLITS)
