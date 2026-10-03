from __future__ import annotations
import logging
import math
from numbers import Real
from datasets import load_dataset
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, WeightedRandomSampler
from arietta.models.modeling import load_tokenizer
from arietta.models.processing import (
    DecisionCollator,
    checked_sequence,
    normalize_question,
    normalize_target,
)

from arietta.models.state_serialization import (
    StateSerialization,
    dataset_state,
)

logger = logging.getLogger(__name__)


class DecisionDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_path: str,
        tokenizer_path: str,
        data_files: dict[str, str],
        dataset_name: str | None = None,
        data_dir: str | None = None,
        cache_dir: str | None = None,
        revision: str | None = None,
        streaming: bool = False,
        batch_size: int = 8,
        num_workers: int = 0,
        max_len: int = 1024,
        head_max_len: int = 256,
        limit: int | None = None,
        required_splits: list[str] | None = None,
        sample_weight_column: str | None = None,
        state_serialization: StateSerialization = "verbatim",
    ):
        super().__init__()
        self.save_hyperparameters()
        if state_serialization not in {"verbatim", "online_json"}:
            raise ValueError("state_serialization must be verbatim or online_json")
        if streaming:
            raise ValueError(
                "streaming is unsupported: preflight validation requires finite datasets"
            )
        if batch_size < 1 or num_workers < 0 or (limit is not None and limit < 1):
            raise ValueError("invalid batch_size, num_workers or limit")
        if sample_weight_column is not None and not sample_weight_column.strip():
            raise ValueError("sample_weight_column must be a nonempty column name")
        self.dataset = None

    def setup(self, stage=None):
        if self.dataset is not None:
            return
        p = self.hparams
        dataset = load_dataset(
            path=p.dataset_path,
            name=p.dataset_name,
            data_dir=p.data_dir,
            data_files=dict(p.data_files),
            cache_dir=p.cache_dir,
            revision=p.revision,
            streaming=p.streaming,
        )
        tokenizer = load_tokenizer(p.tokenizer_path)
        config = {"max_len": p.max_len, "head_max_len": p.head_max_len}
        config["state_serialization"] = p.state_serialization
        self.collator = DecisionCollator(tokenizer, config)
        required = p.required_splits or ["train", "validation", "test"]
        seen_ids, seen_groups, seen_families = set(), {}, {}
        for split in required:
            if split not in dataset:
                raise ValueError(f"Dataset split {split} is missing")
            needed = {
                "id",
                "group_id",
                "family_id",
                "task",
                "state",
                "question",
                "split",
            }
            if split == "train" and p.sample_weight_column is not None:
                needed.add(p.sample_weight_column)
            missing = needed - set(dataset[split].column_names)
            if missing:
                raise ValueError(
                    f"Dataset split {split} is missing required columns: {sorted(missing)}"
                )
            if len(dataset[split]) == 0:
                raise ValueError(f"Dataset split {split} is empty")
            for row in dataset[split]:
                if split == "train" and p.sample_weight_column is not None:
                    weight = row[p.sample_weight_column]
                    if (
                        isinstance(weight, bool)
                        or not isinstance(weight, Real)
                        or not math.isfinite(weight)
                        or weight <= 0
                    ):
                        raise ValueError(
                            f"train/{row['id']}: {p.sample_weight_column} must be finite and positive"
                        )
                if (
                    row["split"] != split
                    or not isinstance(row["task"], str)
                    or not row["task"]
                ):
                    raise ValueError(f"invalid split or task at {row['id']}")
                if row["id"] in seen_ids:
                    raise ValueError(f"duplicate id: {row['id']}")
                seen_ids.add(row["id"])
                for field, seen in (
                    ("group_id", seen_groups),
                    ("family_id", seen_families),
                ):
                    value = row[field]
                    if not value or value in seen and seen[value] != split:
                        raise ValueError(f"cross-split or missing {field}: {value}")
                    seen[value] = split
                q = normalize_question(row["question"])
                normalize_target(row, q)
                try:
                    checked_sequence(
                        tokenizer,
                        dataset_state(row["state"], p.state_serialization),
                        q,
                        config,
                    )
                except ValueError as error:
                    raise ValueError(f"{split}/{row['id']}: {error}") from error
            if p.limit is not None:
                dataset[split] = dataset[split].select(
                    range(min(p.limit, len(dataset[split])))
                )
            logger.info(
                "validated split=%s rows=%d fingerprint=%s",
                split,
                len(dataset[split]),
                dataset[split]._fingerprint,
            )
        self.dataset = dataset

    def loader(self, split: str, shuffle: bool = False):
        p = self.hparams
        sampler = None
        if split == "train" and shuffle and p.sample_weight_column is not None:
            sampler = WeightedRandomSampler(
                self.dataset[split][p.sample_weight_column],
                num_samples=len(self.dataset[split]),
                replacement=True,
            )
        return DataLoader(
            self.dataset[split],
            batch_size=p.batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=p.num_workers,
            collate_fn=self.collator,
            pin_memory=True,
            persistent_workers=p.num_workers > 0,
        )

    def train_dataloader(self):
        return self.loader("train", True)

    def val_dataloader(self):
        return self.loader("validation")

    def test_dataloader(self):
        return self.loader("test")
