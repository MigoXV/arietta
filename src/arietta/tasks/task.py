from __future__ import annotations
import re
import torch
from arietta.models.precision import DTYPES, deployment_copy
from arietta.models.quantization import Int8QAT
from arietta.models.modeling import read_config
from lightning.pytorch import LightningModule
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModel
from arietta.models.modeling import (
    load_pretrained,
    register_models,
    validate_repository,
)
from arietta.models.processing import MODEL_KEYS
from arietta.models.identity import repository_fingerprint
from arietta.criterions.choice import ChoiceCriterion


def classification_metrics(records):
    result = {}
    for task in sorted({row[0] for row in records}):
        pairs = [(truth, pred) for category, truth, pred in records if category == task]
        labels = sorted({x for pair in pairs for x in pair})
        f1 = []
        for label in labels:
            tp = sum(t == p == label for t, p in pairs)
            fp = sum(t != label and p == label for t, p in pairs)
            fn = sum(t == label and p != label for t, p in pairs)
            f1.append(2 * tp / max(1, 2 * tp + fp + fn))
        result[task] = {
            "accuracy": sum(t == p for t, p in pairs) / len(pairs),
            "macro_f1": sum(f1) / len(f1),
            "count": len(pairs),
        }
    return result


class DecisionTask(LightningModule):
    def __init__(
        self,
        lr: float,
        head_lr: float,
        weight_decay: float,
        warmup_ratio: float,
        criterion: ChoiceCriterion,
        quantization: Int8QAT | None = None,
    ):
        super().__init__()
        self.lr, self.head_lr, self.weight_decay, self.warmup_ratio = (
            lr,
            head_lr,
            weight_decay,
            warmup_ratio,
        )
        self.criterion = criterion
        self.quantization = quantization
        self.records = []
        object.__setattr__(self, "_deploy_model", None)

    def forward(self, **batch):
        return self.model(**{key: batch[key] for key in MODEL_KEYS})

    def training_step(self, batch, batch_idx):
        loss = self.criterion(
            self(**batch).logits, batch["targets"], batch["marker_mask"]
        )
        self.log("train_loss", loss, batch_size=len(batch["labels"]))
        return loss

    def on_train_end(self):
        # GradScaler can skip every step while Lightning's global_step still advances.
        states = [
            state
            for optimizer in self.trainer.optimizers
            for state in optimizer.state.values()
        ]
        if not any(float(state.get("step", 0)) > 0 for state in states):
            raise RuntimeError(
                "No optimizer update occurred; inspect FP16 overflow and GradScaler initial scale"
            )

    def validation_step(self, batch, batch_idx):
        output = self(**batch)
        loss = self.criterion(output.logits, batch["targets"], batch["marker_mask"])
        self.log("val_loss", loss, batch_size=len(batch["labels"]))
        dtype = DTYPES[self.deployment_dtype]
        with torch.autocast(
            device_type=self.device.type, dtype=dtype, enabled=dtype != torch.float32
        ):
            deployed = self._deploy_model(
                **{k: batch[k] for k in MODEL_KEYS}
            ).logits.float()
        log_probs = deployed.log_softmax(-1)
        probs = log_probs.exp()
        for i, (kind, label, keys) in enumerate(
            zip(batch["qtype"].tolist(), batch["labels"].tolist(), batch["option_keys"])
        ):
            count = len(keys)
            p, target = probs[i, :count], batch["targets"][i, :count]
            self.records.append(
                {
                    "kind": kind,
                    "label": label,
                    "truth": keys[label] if label >= 0 else None,
                    "pred": keys[int(p.argmax())],
                    "nll": float(-(target * log_probs[i, :count]).sum()),
                    "brier": float((p - target).square().sum()),
                    "error": float((p * torch.arange(count, device=p.device)).sum())
                    - label
                    if label >= 0
                    else None,
                }
            )
        return loss

    def on_fit_start(self):
        expected = {"fp16": "16-mixed", "bf16": "bf16-mixed", "fp32": "32-true"}[
            self.deployment_dtype
        ]
        if str(self.trainer.precision) != expected:
            raise ValueError(
                f"deployment_dtype={self.deployment_dtype} requires Trainer precision={expected}"
            )
        if self.trainer.world_size != 1:
            raise ValueError(
                "Arietta deployment validation currently requires a single device"
            )

    def on_validation_epoch_start(self):
        self.records.clear()
        # Bypass nn.Module registration: deployment weights must never enter checkpoints.
        object.__setattr__(
            self, "_deploy_model", deployment_copy(self.model, self.deployment_dtype)
        )

    def on_validation_epoch_end(self):
        losses = []
        for kind, name in enumerate(("choice", "score", "noul")):
            rows = [r for r in self.records if r["kind"] == kind]
            if not rows:
                continue
            hard = [r for r in rows if r["label"] >= 0]
            values = {
                "nll": sum(r["nll"] for r in rows) / len(rows),
                "brier": sum(r["brier"] for r in rows) / len(rows),
                "hard_count": len(hard),
                "soft_count": len(rows) - len(hard),
            }
            losses.append(values["nll"])
            if hard:
                values.update(
                    classification_metrics(
                        [(name, r["truth"], r["pred"]) for r in hard]
                    )[name]
                )
                if name == "score":
                    values["mae"] = sum(abs(r["error"]) for r in hard) / len(hard)
                    values["rmse"] = (
                        sum(r["error"] ** 2 for r in hard) / len(hard)
                    ) ** 0.5
            for metric, value in values.items():
                self.log(f"val_deploy_{name}_{metric}", float(value))
        if losses:
            self.log("val_deploy_loss", sum(losses) / len(losses))
        self.records.clear()
        object.__setattr__(self, "_deploy_model", None)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_start(self):
        self.on_validation_epoch_start()

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    def configure_optimizers(self):
        heads, encoder = [], []
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                (encoder if ".encoder." in name else heads).append(parameter)
        optimizer = AdamW(
            [{"params": encoder, "lr": self.lr}, {"params": heads, "lr": self.head_lr}],
            weight_decay=self.weight_decay,
        )
        steps = int(self.trainer.estimated_stepping_batches)
        if steps <= 0:
            return optimizer
        warmup = int(steps * self.warmup_ratio)

        def scale(step):
            if step < warmup:
                return (step + 1) / max(1, warmup)
            return max(0.0, (steps - step) / max(1, steps - warmup))

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": LambdaLR(optimizer, scale),
                "interval": "step",
            },
        }

    def on_save_checkpoint(self, checkpoint):
        checkpoint["task_class"] = type(self).__name__
        checkpoint["source_fingerprint"] = self.source_fingerprint
        checkpoint["qat_signature"] = (
            self.quantization.signature() if self.quantization else None
        )

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        map_location=None,
        strict=None,
        weights_only=True,
        **kwargs,
    ):
        # Store only primitive QAT metadata. mmap avoids eagerly reading the large
        # tensor payload twice when Lightning performs its regular strict restore.
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=weights_only, mmap=True
        )
        if "quantization" not in kwargs and checkpoint.get("qat_signature") is not None:
            kwargs["quantization"] = Int8QAT.from_signature(checkpoint["qat_signature"])
        return super().load_from_checkpoint(
            checkpoint_path,
            map_location=map_location,
            strict=strict,
            weights_only=weights_only,
            **kwargs,
        )

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get("task_class") != type(self).__name__:
            raise ValueError("checkpoint task_class mismatch")
        if checkpoint.get("source_fingerprint") != self.source_fingerprint:
            raise ValueError("checkpoint source fingerprint mismatch")
        saved = checkpoint.get("hyper_parameters", {})
        expected_qat = self.quantization.signature() if self.quantization else None
        if checkpoint.get("qat_signature") != expected_qat:
            raise ValueError("checkpoint QAT configuration mismatch")
        if saved.get("deployment_dtype") != self.deployment_dtype:
            raise ValueError("checkpoint deployment precision mismatch")
        source = (
            "model_config_path"
            if type(self).__name__ == "ScratchTask"
            else "pretrained_model_path"
        )
        other = (
            "pretrained_model_path"
            if source == "model_config_path"
            else "model_config_path"
        )
        if (
            source not in saved
            or other in saved
            or saved[source] != self.hparams[source]
        ):
            raise ValueError("checkpoint source fields mismatch")


class FullFinetuneTask(DecisionTask):
    def __init__(
        self,
        pretrained_model_path: str,
        criterion: ChoiceCriterion,
        lr: float = 2e-5,
        head_lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.05,
        deployment_dtype: str = "fp32",
        quantization: Int8QAT | None = None,
    ):
        super().__init__(
            lr, head_lr, weight_decay, warmup_ratio, criterion, quantization
        )
        if deployment_dtype not in DTYPES:
            raise ValueError("deployment_dtype must be fp16, bf16 or fp32")
        self.deployment_dtype = deployment_dtype
        self.save_hyperparameters(ignore=["criterion", "quantization"])
        self.source_fingerprint = repository_fingerprint(pretrained_model_path)
        self.model = load_pretrained(pretrained_model_path, dtype=torch.float32)
        if getattr(self.model.config, "arietta_quantization", None):
            raise ValueError("finetuning requires floating-point pretrained weights")
        self.model.decision.temperature.fill_(1)
        # Action policy receives no supervised loss; keep its pretraining weights unchanged.
        for p in self.model.decision.act_head.parameters():
            p.requires_grad = False
        if quantization:
            quantization.prepare(self.model)


class LoraFinetuneTask(DecisionTask):
    def __init__(
        self,
        pretrained_model_path: str,
        criterion: ChoiceCriterion,
        lr: float = 2e-4,
        head_lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.05,
        deployment_dtype: str = "fp32",
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
        quantization: Int8QAT | None = None,
    ):
        super().__init__(
            lr, head_lr, weight_decay, warmup_ratio, criterion, quantization
        )
        if quantization is not None and dropout != 0:
            raise ValueError("QAT + LoRA requires dropout=0")
        if deployment_dtype not in DTYPES:
            raise ValueError("deployment_dtype must be fp16, bf16 or fp32")
        self.deployment_dtype = deployment_dtype
        self.save_hyperparameters(ignore=["criterion", "quantization"])
        self.source_fingerprint = repository_fingerprint(pretrained_model_path)
        model = load_pretrained(pretrained_model_path, dtype=torch.float32)
        if getattr(model.config, "arietta_quantization", None):
            raise ValueError("finetuning requires floating-point pretrained weights")
        model.decision.temperature.fill_(1)
        targets = [
            name
            for name, module in model.named_modules()
            if re.fullmatch(r"decision.encoder.layers.\d+.attn.(Wqkv|Wo)", name)
        ]
        if not targets:
            raise ValueError("encoder has no supported attention Wqkv/Wo LoRA targets")
        self.model = get_peft_model(
            model,
            LoraConfig(
                r=rank,
                lora_alpha=alpha,
                lora_dropout=dropout,
                target_modules=targets,
                modules_to_save=[
                    "decision.head",
                    "decision.scorer",
                    "decision.type_emb",
                ],
                bias="none",
            ),
        )
        if quantization:
            quantization.prepare(self.model)


class ScratchTask(DecisionTask):
    def __init__(
        self,
        model_config_path: str,
        criterion: ChoiceCriterion,
        lr: float = 2e-4,
        head_lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.05,
        deployment_dtype: str = "fp32",
        quantization: Int8QAT | None = None,
    ):
        super().__init__(
            lr, head_lr, weight_decay, warmup_ratio, criterion, quantization
        )
        if deployment_dtype not in DTYPES:
            raise ValueError("deployment_dtype must be fp16, bf16 or fp32")
        self.deployment_dtype = deployment_dtype
        self.save_hyperparameters(ignore=["criterion", "quantization"])
        register_models()
        validate_repository(model_config_path, scratch=True)
        self.source_fingerprint = repository_fingerprint(model_config_path)
        self.model = AutoModel.from_config(read_config(model_config_path))
        if getattr(self.model.config, "arietta_quantization", None):
            raise ValueError("scratch training requires a floating-point model config")
        for p in self.model.decision.act_head.parameters():
            p.requires_grad = False
        if quantization:
            quantization.prepare(self.model)
