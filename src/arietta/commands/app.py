from __future__ import annotations
import json
import logging
from pathlib import Path
import typer
from arietta.models.state_serialization import StateSerialization

logging.basicConfig(level=logging.INFO)
app = typer.Typer(no_args_is_help=True)


@app.command()
def build_tiny_sft(
    source: Path = typer.Option(Path("examples/tiny-sft/cases.jsonl")),
    destination: Path = typer.Option(
        Path("data-bin/tiny-sft-v2"), envvar="ARIETTA_DATASET_OUTPUT"
    ),
):
    from arietta.tasks.sft_dataset import build_tiny_sft as build

    result = build(source, destination)
    typer.echo(
        json.dumps(
            {
                "destination": str(destination),
                "audit_directory": str(
                    destination.with_name(destination.name + "-audit")
                ),
                "cases": result["cases"],
                "rows": result["rows"],
            },
            ensure_ascii=False,
        )
    )


@app.command()
def export(
    checkpoint: Path = typer.Argument(...),
    destination: Path = typer.Argument(...),
    format: str = typer.Option("service"),
    dtype: str | None = typer.Option(None),
    threads: int = typer.Option(4, envvar="ARIETTA_CPU_THREADS"),
):
    import torch
    from arietta.exporting.export import export_checkpoint

    torch.set_num_threads(threads)
    typer.echo(str(export_checkpoint(checkpoint, destination, format, dtype)))


@app.command()
def evaluate(
    pretrained_model_path: str = typer.Option(..., envvar="ARIETTA_MODEL"),
    data_file: str = typer.Option(...),
    split: str = typer.Option("validation"),
    device: str = typer.Option("cpu", envvar="ARIETTA_DEVICE"),
    batch_size: int = typer.Option(8),
    temperature: float = typer.Option(1.0),
    dtype: str = typer.Option("fp32", envvar="ARIETTA_DTYPE"),
    state_serialization: StateSerialization = typer.Option("online_json"),
    output: Path = typer.Option(...),
    threads: int = typer.Option(4, envvar="ARIETTA_CPU_THREADS"),
):
    import torch
    from arietta.evaluation.evaluate import evaluate as run

    torch.set_num_threads(threads)
    result = run(
        pretrained_model_path,
        data_file,
        split,
        device,
        batch_size,
        temperature,
        state_serialization=state_serialization,
        dtype=dtype,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    typer.echo(json.dumps(result["metrics"], ensure_ascii=False))


@app.command()
def calibrate(
    predictions: Path = typer.Argument(...), output: Path = typer.Option(...)
):
    from arietta.evaluation.evaluate import calibrate as fit

    data = json.loads(predictions.read_text())
    if data["split"] != "calibration":
        raise ValueError("temperature must be fit on calibration split")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "temperature": fit(data["rows"]),
                "model": data["model"],
                "model_fingerprint": data["model_fingerprint"],
                "dtype": data["dtype"],
                "dataset_sha256": data.get("dataset_sha256"),
                "source": str(predictions),
                "state_serialization": data.get("state_serialization", "verbatim"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    app()
