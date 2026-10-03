from __future__ import annotations

import json
import torch
from safetensors.torch import load_file
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from pathlib import Path

from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from .reference import DecisionModel
from .quantization import (
    install_int8_modules,
    validate_int8_weights,
    validate_quantized_state,
    attention_context,
)


class LayaConfig(PretrainedConfig):
    model_type = "arietta_laya"

    def __init__(self, encoder_config=None, agent_config=None, **kwargs):
        super().__init__(**kwargs)
        self.encoder_config = encoder_config or {}
        self.agent_config = agent_config or {
            "head_layers": 2,
            "act_costs": {"escalate": 0.5},
            "max_len": 1024,
            "head_max_len": 256,
        }


class LayaModel(PreTrainedModel):
    config_class = LayaConfig
    base_model_prefix = "decision"

    def __init__(self, config):
        super().__init__(config)
        ecfg = dict(config.encoder_config)
        model_type = ecfg.pop("model_type")
        encoder_config = AutoConfig.for_model(model_type, **ecfg)
        encoder_config.reference_compile = False
        encoder = AutoModel.from_config(
            encoder_config, dtype=torch.float32, attn_implementation="sdpa"
        )
        self.decision = DecisionModel(
            encoder,
            config.agent_config["head_layers"],
            config.agent_config.get(
                "num_actions",
                len(config.agent_config.get("act_costs", {"escalate": 0.5})) + 1,
            ),
        )
        self.post_init()
        metadata = getattr(config, "arietta_quantization", None)
        if metadata:
            install_int8_modules(self.decision, metadata)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, *args, config=None, dtype="auto", **kwargs
    ):
        # Native repositories deliberately have no HF wrapper prefix. Validate every key/shape.
        config = config or read_config(pretrained_model_name_or_path)
        native = (
            json.loads(
                (Path(pretrained_model_name_or_path) / "config.json").read_text()
            ).get("format_version")
            == 1
        )
        state = load_file(
            str(Path(pretrained_model_name_or_path) / "model.safetensors")
        )
        key = (
            "" if native else "decision."
        ) + "encoder.embeddings.tok_embeddings.weight"
        target = state[key].dtype if dtype == "auto" else dtype
        from .precision import move_model

        model = move_model(cls(config), torch.device("cpu"), target)
        metadata = getattr(config, "arietta_quantization", None)
        if metadata:
            validate_quantized_state(state, metadata, "" if native else "decision.")
        (model.decision if native else model).load_state_dict(state, strict=True)
        validate_int8_weights(model.decision)
        return model

    def forward(
        self, input_ids, attention_mask, marker_pos, marker_mask, qtype, **kwargs
    ):
        with attention_context(self.decision):
            logits, _ = self.decision(
                input_ids, attention_mask, marker_pos, marker_mask, qtype
            )
        return SequenceClassifierOutput(logits=logits)


def register_models():
    AutoConfig.register(LayaConfig.model_type, LayaConfig, exist_ok=True)
    AutoModel.register(LayaConfig, LayaModel, exist_ok=True)


def read_config(path):
    raw = json.loads((Path(path) / "config.json").read_text())
    if raw.get("format_version") == 1:
        return LayaConfig(
            encoder_config=raw["encoder"],
            agent_config={
                "head_layers": raw["decision_head"]["layers"],
                "num_actions": raw["decision_head"]["num_actions"],
                **raw["input_limits"],
            },
            native_config=raw,
            arietta_quantization=raw.get("quantization"),
        )
    if raw.get("model_type") != LayaConfig.model_type:
        raise ValueError("expected format_version=1 or arietta_laya model repository")
    return LayaConfig.from_dict(raw)


def validate_repository(path: str, scratch: bool = False):
    root = Path(path)
    for name in ("config.json",):
        if not (root / name).is_file():
            raise ValueError(f"incomplete model repository: {root / name}")
    if not (
        (root / "tokenizer.json").is_file()
        or (root / "tokenizer/tokenizer.json").is_file()
    ):
        raise ValueError(f"missing tokenizer: {root}")
    weights = (
        list(root.glob("*.safetensors"))
        + list(root.glob("*.bin"))
        + list(root.glob("*.index.json"))
    )
    if scratch and weights:
        raise ValueError(f"model_config_path must not contain weights: {weights}")
    if not scratch and not (root / "model.safetensors").is_file():
        raise ValueError(f"pretrained_model_path requires full weights: {root}")
    read_config(root)
    return root


def load_tokenizer(path):
    root = Path(path)
    if (root / "config.json").is_file():
        config = read_config(root)
        native = getattr(config, "native_config", None)
        if native:
            return PreTrainedTokenizerFast(
                tokenizer_file=str(root / "tokenizer.json"), **native["tokenizer"]
            )
    if (root / "tokenizer").is_dir():
        root = root / "tokenizer"
    return AutoTokenizer.from_pretrained(root, local_files_only=True)


def load_pretrained(pretrained_model_path: str, dtype="auto"):
    register_models()
    validate_repository(pretrained_model_path)
    config = read_config(pretrained_model_path)
    return AutoModel.from_pretrained(
        pretrained_model_path, config=config, dtype=dtype, local_files_only=True
    )
