from __future__ import annotations
import json
import torch
from .state_serialization import dataset_state
from .reference import build_sequence, render_options, serialize_state


MODEL_KEYS = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")


def normalize_question(question):
    question = json.loads(question) if isinstance(question, str) else dict(question)
    if set(question) - {"type", "instructions", "criteria"}:
        raise ValueError("unknown question fields")
    kind, criteria = question.get("type"), question.get("criteria")
    if (
        not isinstance(question.get("instructions"), str)
        or not 1 <= len(question["instructions"]) <= 8192
    ):
        raise ValueError("instructions must contain 1..8192 characters")
    if kind == "choice":
        if isinstance(criteria, list):
            if any(not isinstance(k, str) for k in criteria) or len(
                set(criteria)
            ) != len(criteria):
                raise ValueError("choice options must be unique strings")
            criteria = dict.fromkeys(criteria)
        if (
            not isinstance(criteria, dict)
            or not 2 <= len(criteria) <= 16
            or any(
                not isinstance(k, str)
                or not k
                or v is not None
                and not isinstance(v, str)
                for k, v in criteria.items()
            )
        ):
            raise ValueError("choice needs 2..16 string options")
    elif kind == "score":
        if (
            not isinstance(criteria, list)
            or not 2 <= len(criteria) <= 16
            or any(not isinstance(v, str) for v in criteria)
        ):
            raise ValueError("score needs 2..16 ordered descriptions")
    elif kind == "noul":
        if criteria is not None and (
            not isinstance(criteria, dict)
            or set(criteria) - {"false", "true"}
            or any(v is not None and not isinstance(v, str) for v in criteria.values())
        ):
            raise ValueError("noul criteria only allows false/true descriptions")
    else:
        raise ValueError("question.type must be choice, score or noul")
    question["criteria"] = criteria
    return question


def option_keys(q):
    return (
        list(q["criteria"])
        if q["type"] == "choice"
        else list(range(len(q["criteria"])))
        if q["type"] == "score"
        else [False, True]
    )


def normalize_target(row, q):
    import math

    keys = option_keys(q)
    target = row.get("target")
    if target is not None and row.get("label") is not None:
        raise ValueError("target and legacy label are mutually exclusive")
    if target is None:
        if q["type"] != "choice" or row.get("label") is None:
            raise ValueError("target is required; legacy label is choice-only")
        target = {"label": row["label"]}
    target = json.loads(target) if isinstance(target, str) else target
    if not isinstance(target, dict) or len(target) != 1:
        raise ValueError(
            "target must have exactly one of label, distribution, probability"
        )
    if "label" in target:
        value = target["label"]
        expected = {"choice": str, "score": int, "noul": bool}[q["type"]]
        if type(value) is not expected or value not in keys:
            raise ValueError("label not in criteria or wrong label type")
        index = keys.index(value)
        return [float(i == index) for i in range(len(keys))], index
    if "probability" in target and q["type"] == "noul":
        p = target["probability"]
        if type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("probability must be finite in [0,1]")
        values = [1 - p, p]
    elif "distribution" in target:
        values = target["distribution"]
    else:
        raise ValueError("unsupported target field")
    if (
        not isinstance(values, list)
        or len(values) != len(keys)
        or any(
            type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
            for v in values
        )
        or not math.isclose(sum(values), 1, abs_tol=1e-6)
    ):
        raise ValueError(
            "distribution must be finite, normalized and match option order"
        )
    return values, -1


def checked_sequence(tok, state, question, config):
    question = normalize_question(question)
    q = {
        "t": question["type"],
        "ins": question["instructions"],
        "crit": question["criteria"],
    }
    if (
        tok.mask_token in serialize_state(state)
        or tok.mask_token in q["ins"]
        or any(tok.mask_token in text for text in render_options(q))
    ):
        raise ValueError("reserved MASK marker in input")
    head = tok(f"{q['t']} question: {q['ins']}", add_special_tokens=False)["input_ids"]
    options = [
        tok(" " + text, add_special_tokens=False)["input_ids"]
        for text in render_options(q)
    ]
    option_len = sum(1 + len(ids) for ids in options)
    budget = config["head_max_len"] - option_len
    if (
        any(len(ids) > 48 for ids in options)
        or budget < 16
        or len(head) > max(8, budget)
    ):
        raise ValueError("question_head_exceeded")
    state_len = len(tok(serialize_state(state), add_special_tokens=False)["input_ids"])
    if 4 + len(head) + option_len + state_len > config["max_len"]:
        raise ValueError("state_token_budget_exceeded")
    return build_sequence(tok, state, q, config["max_len"], config["head_max_len"])


class DecisionCollator:
    def __init__(self, tokenizer, config):
        self.tokenizer, self.config = tokenizer, config

    def __call__(self, rows):
        encoded = []
        for row in rows:
            q = normalize_question(row["question"])
            ids, markers = checked_sequence(
                self.tokenizer,
                dataset_state(
                    row["state"], self.config.get("state_serialization", "verbatim")
                ),
                q,
                self.config,
            )
            distribution, label = normalize_target(row, q)
            encoded.append(
                (
                    ids,
                    markers,
                    label,
                    distribution,
                    {"choice": 0, "score": 1, "noul": 2}[q["type"]],
                )
            )
        n, length, options = (
            len(rows),
            max(len(x[0]) for x in encoded),
            max(len(x[1]) for x in encoded),
        )
        result = {
            "input_ids": torch.full(
                (n, length), self.tokenizer.pad_token_id, dtype=torch.long
            ),
            "attention_mask": torch.zeros(n, length, dtype=torch.long),
            "marker_pos": torch.zeros(n, options, dtype=torch.long),
            "marker_mask": torch.zeros(n, options, dtype=torch.bool),
            "qtype": torch.tensor([x[4] for x in encoded]),
            "targets": torch.zeros(n, options, dtype=torch.float32),
            "labels": torch.tensor([x[2] for x in encoded]),
            "tasks": [row["task"] for row in rows],
            "ids": [row["id"] for row in rows],
            "groups": [row["group_id"] for row in rows],
            "option_keys": [
                option_keys(normalize_question(row["question"])) for row in rows
            ],
        }
        for i, (ids, markers, _, distribution, _) in enumerate(encoded):
            result["targets"][i, : len(distribution)] = torch.tensor(distribution)
            result["input_ids"][i, : len(ids)] = torch.tensor(ids)
            result["attention_mask"][i, : len(ids)] = 1
            result["marker_pos"][i, : len(markers)] = torch.tensor(markers)
            result["marker_mask"][i, : len(markers)] = True
        return result
