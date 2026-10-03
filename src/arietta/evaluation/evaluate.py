from __future__ import annotations
import hashlib
import math
from pathlib import Path
from time import perf_counter
import torch
from arietta.models.state_serialization import StateSerialization
from arietta.models.modeling import load_pretrained
from arietta.models.identity import repository_fingerprint
from arietta.models.processing import MODEL_KEYS
from arietta.models.precision import DTYPES, move_model
from arietta.tasks.data import DecisionDataModule
from arietta.tasks.task import classification_metrics


@torch.inference_mode()
def evaluate(
    pretrained_model_path: str,
    data_file: str,
    split: str,
    device: str = "cpu",
    batch_size: int = 8,
    temperature: float = 1.0,
    dtype: str = "fp32",
    state_serialization: StateSerialization = "online_json",
):
    if state_serialization not in {"verbatim", "online_json"}:
        raise ValueError("state_serialization must be verbatim or online_json")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive")
    model_fingerprint = repository_fingerprint(pretrained_model_path)
    dataset_sha256 = hashlib.sha256(Path(data_file).read_bytes()).hexdigest()
    if dtype not in DTYPES or device == "cpu" and dtype == "fp16":
        raise ValueError("unsupported device/dtype")
    model = move_model(
        load_pretrained(pretrained_model_path, dtype=DTYPES[dtype]),
        torch.device(device),
        DTYPES[dtype],
    ).eval()
    cfg = model.config.agent_config
    dm = DecisionDataModule(
        "parquet" if Path(data_file).suffix.lower() == ".parquet" else "json",
        pretrained_model_path,
        {split: data_file},
        required_splits=[split],
        batch_size=batch_size,
        state_serialization=state_serialization,
        max_len=cfg["max_len"],
        head_max_len=cfg["head_max_len"],
    )
    dm.setup()
    records, rows = [], []
    started = perf_counter()
    for batch in dm.loader(split):
        with torch.autocast(
            device_type=torch.device(device).type,
            dtype=DTYPES[dtype],
            enabled=dtype != "fp32",
        ):
            logits = (
                model(**{key: batch[key].to(device) for key in MODEL_KEYS})
                .logits.float()
                .cpu()
            )
        probabilities = (logits / temperature).softmax(-1)
        for i, pred in enumerate(probabilities.argmax(-1).tolist()):
            keys, truth, task = (
                batch["option_keys"][i],
                batch["labels"][i].item(),
                batch["tasks"][i],
            )
            if truth >= 0:
                records.append(
                    (
                        ("choice", "score", "noul")[batch["qtype"][i]],
                        keys[truth],
                        keys[pred],
                    )
                )
            rows.append(
                {
                    "id": batch["ids"][i],
                    "group_id": batch["groups"][i],
                    "task": task,
                    "label": keys[truth] if truth >= 0 else None,
                    "target": batch["targets"][i, : len(keys)].tolist(),
                    "question_type": ("choice", "score", "noul")[batch["qtype"][i]],
                    "prediction": keys[pred],
                    "logits": dict(zip(keys, logits[i, : len(keys)].tolist())),
                    "probabilities": dict(
                        zip(keys, probabilities[i, : len(keys)].tolist())
                    ),
                }
            )
    metrics = classification_metrics(records)
    for kind in ("choice", "score", "noul"):
        group = [r for r in rows if r["question_type"] == kind]
        if not group:
            continue
        values = metrics.setdefault(kind, {})
        values["hard_count"] = sum(r["label"] is not None for r in group)
        values["soft_count"] = len(group) - values["hard_count"]
        values["nll"] = sum(
            -sum(
                t * math.log(max(p, 1e-30))
                for t, p in zip(r["target"], r["probabilities"].values())
            )
            for r in group
        ) / len(group)
        values["brier"] = sum(
            sum((t - p) ** 2 for t, p in zip(r["target"], r["probabilities"].values()))
            for r in group
        ) / len(group)
    return {
        "model": pretrained_model_path,
        "model_fingerprint": model_fingerprint,
        "dataset_sha256": dataset_sha256,
        "split": split,
        "temperature": temperature,
        "dtype": dtype,
        "state_serialization": state_serialization,
        "elapsed_seconds": perf_counter() - started,
        "metrics": metrics,
        "rows": rows,
    }


def calibrate(records):
    if not records:
        raise ValueError("calibration records are empty")
    # Single temperature fitted on calibration only; choice ordering is preserved.
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature], max_iter=50, line_search_fn="strong_wolfe"
    )
    examples = [
        (
            torch.tensor(list(row["logits"].values())),
            torch.tensor(row["target"]),
        )
        for row in records
    ]

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.clamp(-4, 4).exp()
        loss = torch.stack(
            [
                -(label * (logits / temperature).log_softmax(-1)).sum()
                for logits, label in examples
            ]
        ).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().clamp(-4, 4).exp())
