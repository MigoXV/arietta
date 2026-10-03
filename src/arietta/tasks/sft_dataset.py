from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from datasets import load_dataset

from arietta.models.processing import normalize_question, normalize_target

VERSION = "tiny-sft-v1"
SPLITS = ("train", "validation", "calibration", "test")
QUESTIONS = {
    "stance": {
        "type": "choice",
        "instructions": (
            "判断发言者本人对当前议题的最新立场。明确或暂定支持归support，"
            "明确或暂定反对归oppose；仅提问、转述他人、未决定或撤回旧立场归undecided。"
        ),
        "criteria": {
            "support": "本人支持或倾向支持",
            "oppose": "本人反对或倾向反对",
            "undecided": "本人未表态或尚未决定",
        },
    },
    "certainty": {
        "type": "score",
        "instructions": (
            "判断本人对当前议题的立场确定程度。未表态或未决定为0，"
            "暂定倾向或有条件立场为1，明确无条件支持或反对为2。"
            "任务执行承诺不代表议题立场确定。"
        ),
        "criteria": ["未表态或未决定", "暂定倾向或仍有条件", "明确无条件支持或反对"],
    },
    "commitment": {
        "type": "noul",
        "instructions": (
            "本人是否已无条件确认亲自执行具体任务？明确接受或承诺才为true。"
            "只表达立场、能力、建议、转述他人、否认承诺或条件未满足均为false。"
            "任务可以与议题立场独立。"
        ),
        "criteria": {"false": "没有已确认的个人任务", "true": "已确认亲自执行具体任务"},
    },
}
TASKS = {
    "stance": "speaker_stance",
    "certainty": "stance_certainty",
    "commitment": "execution_commitment",
}


def validate_cases(cases: Iterable[dict]) -> None:
    """检查标注结构与证据定位；语义正确性仍需按标注规范逐条审阅。"""
    ids, utterances, families = set(), set(), {}
    coverage = {split: {task: set() for task in QUESTIONS} for split in SPLITS}
    for case in cases:
        case_id, split = case["id"], case["split"]
        if not isinstance(case_id, str) or not case_id or case_id in ids:
            raise ValueError(f"duplicate or missing id: {case_id}")
        ids.add(case_id)
        if split not in coverage:
            raise ValueError(f"{case_id}: invalid split {split}")
        for field in ("family_id", "proposal", "utterance"):
            if not isinstance(case[field], str) or not case[field].strip():
                raise ValueError(f"{case_id}: missing {field}")
        text = "".join(case["utterance"].split())
        if text in utterances:
            raise ValueError(f"{case_id}: duplicate utterance")
        utterances.add(text)
        family = case["family_id"]
        if family in families and families[family] != split:
            raise ValueError(f"{case_id}: cross-split family {family}")
        families[family] = split
        labels, review = case["labels"], case["review"]
        if set(labels) != set(QUESTIONS) or set(review) != set(QUESTIONS):
            raise ValueError(f"{case_id}: all three labels and reviews are required")
        for name, question in QUESTIONS.items():
            normalize_target(
                {"target": {"label": labels[name]}}, normalize_question(question)
            )
            annotation = review[name]
            evidence, reason = annotation["evidence"], annotation["reason"]
            if (
                not isinstance(evidence, str)
                or not evidence.strip()
                or evidence not in case["utterance"]
                or not isinstance(reason, str)
                or not reason.strip()
            ):
                raise ValueError(
                    f"{case_id}/{name}: missing reason or ungrounded evidence"
                )
            coverage[split][name].add(labels[name])
        if (labels["stance"] == "undecided") != (labels["certainty"] == 0):
            raise ValueError(
                f"{case_id}: stance and certainty contradict annotation rules"
            )
    expected = {
        "stance": {"support", "oppose", "undecided"},
        "certainty": {0, 1, 2},
        "commitment": {False, True},
    }
    for split, tasks in coverage.items():
        for name, labels in tasks.items():
            if labels != expected[name]:
                raise ValueError(f"{split}/{name}: incomplete label coverage")


def expand_cases(batch: dict[str, list]) -> dict[str, list]:
    rows = {
        key: []
        for key in (
            "id",
            "group_id",
            "family_id",
            "task",
            "split",
            "state",
            "question",
            "target",
        )
    }
    for i, case_id in enumerate(batch["id"]):
        for name, question in QUESTIONS.items():
            row = {
                "id": f"{case_id}/{name}",
                "group_id": case_id,
                "family_id": batch["family_id"][i],
                "task": TASKS[name],
                "split": batch["split"][i],
                "state": f"议题：{batch['proposal'][i]}\n发言：{batch['utterance'][i]}",
                "question": json.dumps(question, ensure_ascii=False),
                "target": json.dumps(
                    {"label": batch["labels"][i][name]}, ensure_ascii=False
                ),
            }
            for key, value in row.items():
                rows[key].append(value)
    return rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_tiny_sft(source: Path, destination: Path) -> dict:
    cases = load_dataset("json", data_files=str(source), split="train")
    validate_cases(cases)
    source_hash = sha256(source)
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["source_sha256"] != source_hash or json.dumps(
            previous["questions"], ensure_ascii=False
        ) != json.dumps(QUESTIONS, ensure_ascii=False):
            raise ValueError("dataset changed; use a new destination/version")
    expanded = cases.map(expand_cases, batched=True, remove_columns=cases.column_names)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": VERSION,
        "provenance": "synthetic; manually authored utterances and explicit labels; assistant reviewed",
        "source_sha256": source_hash,
        "questions": QUESTIONS,
        "cases": len(cases),
        "rows": len(expanded),
        "splits": {},
    }
    for split in SPLITS:
        selected = expanded.filter(lambda row: row["split"] == split)
        output = destination / f"{split}.jsonl"
        selected.to_json(str(output), force_ascii=False)
        labels = {
            name: dict(
                sorted(
                    Counter(
                        str(case["labels"][name])
                        for case in cases
                        if case["split"] == split
                    ).items()
                )
            )
            for name in QUESTIONS
        }
        manifest["splits"][split] = {
            "rows": len(selected),
            "cases": len(selected) // len(QUESTIONS),
            "labels": labels,
            "sha256": sha256(output),
        }
    cases.to_json(str(destination / "annotations.jsonl"), force_ascii=False)
    review = [
        "# 小型 SFT 标注审阅表",
        "",
        "合成发言，人工指定硬标签；证据与解释不进入模型输入。",
        "",
    ]
    for case in cases:
        review.extend(
            [
                f"## {case['id']} · {case['split']}",
                "",
                f"议题：{case['proposal']}",
                "",
                case["utterance"],
                "",
            ]
        )
        for name in QUESTIONS:
            annotation = case["review"][name]
            label = json.dumps(case["labels"][name], ensure_ascii=False)
            review.append(
                f"- {name} = {label}；原文：{annotation['evidence']}；依据：{annotation['reason']}"
            )
        review.append("")
    (destination / "review.md").write_text("\n".join(review), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
