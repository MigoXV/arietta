import json
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, ModernBertConfig
from arietta.models import LayaConfig, LayaModel, register_models


@pytest.fixture(autouse=True)
def small_threads():
    torch.set_num_threads(1)


@pytest.fixture
def repository(tmp_path):
    register_models()
    root = tmp_path / "base"
    ecfg = ModernBertConfig(
        vocab_size=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        cls_token_id=1,
        sep_token_id=2,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=1,
        num_attention_heads=1,
        max_position_embeddings=128,
        reference_compile=False,
    )
    cfg = LayaConfig(
        encoder_config=ecfg.to_dict(),
        agent_config={
            "head_layers": 1,
            "act_costs": {"escalate": 0.5},
            "max_len": 128,
            "head_max_len": 96,
        },
    )
    LayaModel(cfg).save_pretrained(root)
    tokenizer = Tokenizer(
        WordLevel(
            {
                "[PAD]": 0,
                "[CLS]": 1,
                "[SEP]": 2,
                "[MASK]": 3,
                "[UNK]": 4,
                "a": 5,
                "b": 6,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        unk_token="[UNK]",
    )
    tok.save_pretrained(root / "tokenizer")
    return root


@pytest.fixture
def data_files(tmp_path):
    files = {}
    for split in ("train", "validation", "test"):
        rows = [
            {
                "id": f"{split}{i}",
                "group_id": f"{split}g{i}",
                "family_id": f"{split}f",
                "task": "owner",
                "split": split,
                "state": "a b",
                "question": json.dumps(
                    {
                        "type": "choice",
                        "instructions": "a",
                        "criteria": {"yes": "a", "no": "b"},
                    }
                ),
                "label": "yes" if i == 0 else "no",
                "evidence_ids": [],
                "provenance": "{}",
            }
            for i in range(2)
        ]
        path = tmp_path / f"{split}.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows))
        files[split] = str(path)
    return files
