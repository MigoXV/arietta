from __future__ import annotations
import os
import json
import importlib.metadata
import subprocess
from pathlib import Path
import uuid
import yaml
from lightning.pytorch.cli import SaveConfigCallback


def normalized(text):
    value = yaml.safe_load(text)
    # Invocation-only: resuming the same immutable run changes just this field.
    value.pop("ckpt_path", None)
    data = value.get("data")
    if (
        isinstance(data, dict)
        and data.get("class_path") == "arietta.tasks.data.DecisionDataModule"
    ):
        # Historical runs predate this explicit policy and used verbatim input.
        data.setdefault("init_args", {}).setdefault("state_serialization", "verbatim")
    return value


def save_immutable(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if normalized(path.read_text()) != normalized(text):
            raise RuntimeError(f"resolved configuration drift: {path}")
    else:
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
        temporary.write_text(text)
        os.replace(temporary, path)
    invocation = path.parent / "invocations"
    invocation.mkdir(exist_ok=True)
    (invocation / f"{uuid.uuid4().hex}.yaml").write_text(text)


class DriftSafeSaveConfigCallback(SaveConfigCallback):
    def setup(self, trainer, pl_module, stage):
        if self.already_saved:
            return
        error = None
        if trainer.is_global_zero:
            try:
                manifest = {
                    "source_fingerprint": pl_module.source_fingerprint,
                    "data": {
                        split: ds._fingerprint
                        for split, ds in trainer.datamodule.dataset.items()
                    },
                    "runtime": {
                        name: importlib.metadata.version(name)
                        for name in (
                            "torch",
                            "transformers",
                            "peft",
                            "lightning",
                            "datasets",
                        )
                    },
                }
                save_manifest(Path(trainer.log_dir) / "manifest.json", manifest)
                save_immutable(
                    Path(trainer.log_dir) / "resolved.yaml",
                    self.parser.dump(self.config, skip_none=False, format="yaml"),
                )
            except Exception as exc:
                error = str(exc)
        error = trainer.strategy.broadcast(error)
        if error:
            raise RuntimeError(error)
        self.already_saved = True


def save_manifest(path: Path, manifest):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise RuntimeError(f"model/data/runtime fingerprint drift: {path}")
    else:
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
        temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        os.replace(temporary, path)
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
        (path.parent / "code-revision.txt").write_text(revision + "\n")
