"""显式跨仓库验收；由 Arietta Poetry 环境执行，GPU0 串行。"""

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import torch
from arietta.criterions.choice import ChoiceCriterion
from arietta.tasks.task import FullFinetuneTask, LoraFinetuneTask
from arietta.models.precision import deployment_copy, DTYPES
from arietta.models.processing import DecisionCollator, MODEL_KEYS
from arietta.models.modeling import load_tokenizer, load_pretrained
from arietta.exporting.export import export_checkpoint, digest_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--inference-repo", type=Path, default=Path("/workspace/opusi/laya")
    )
    parser.add_argument(
        "--runtime-environment", choices=["inference", "arietta"], default="inference"
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cls = {"FullFinetuneTask": FullFinetuneTask, "LoraFinetuneTask": LoraFinetuneTask}[
        checkpoint["task_class"]
    ]
    assert checkpoint["global_step"] >= 2
    assert all(
        torch.isfinite(v).all()
        for v in checkpoint["state_dict"].values()
        if v.is_floating_point()
    )
    task = cls.load_from_checkpoint(
        args.checkpoint, criterion=ChoiceCriterion(), map_location="cpu"
    )
    source = task.hparams.pretrained_model_path
    baseline = load_pretrained(source, dtype=torch.float32)
    model = deployment_copy(task.model, "fp32")
    delta = float(
        (
            model.decision.scorer[1].weight.float()
            - baseline.decision.scorer[1].weight.detach()
        )
        .abs()
        .max()
    )
    assert delta > 0, "trained scorer has not changed"
    del model
    # Use exactly the production validation/export path; keep the FP32 audit separate.
    model = deployment_copy(task.model, task.deployment_dtype)
    assert torch.equal(
        model.decision.act_head[0].weight,
        baseline.decision.act_head[0].weight.to(DTYPES[task.deployment_dtype]),
    ), "frozen act_head changed"
    del baseline, task, checkpoint
    if (args.destination / "export.json").exists():
        assert json.loads((args.destination / "export.json").read_text())[
            "checkpoint_sha256"
        ] == digest_file(args.checkpoint)
    else:
        export_checkpoint(args.checkpoint, args.destination, "service")
    dtype = json.loads((args.destination / "export.json").read_text())["dtype"]
    model.cuda().eval()
    rows = [
        json.loads(line)
        for line in Path("examples/fixtures/validation.jsonl").read_text().splitlines()
    ][::2]
    collator = DecisionCollator(load_tokenizer(source), model.config.agent_config)
    expected = []
    for row in rows:
        batch = collator([row])
        with torch.inference_mode(), torch.autocast("cuda", dtype=DTYPES[dtype]):
            p = (
                model(**{k: batch[k].cuda() for k in MODEL_KEYS})
                .logits.float()
                .softmax(-1)[0]
                .cpu()
                .tolist()
            )
            expected.append(p)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    request = {
        "state": rows[0]["state"],
        "questions": {str(i): json.loads(r["question"]) for i, r in enumerate(rows)},
    }
    script = """
import json,sys,torch,transformers
from laya.config import Config
from laya.contracts import DecisionRequest
from laya.runtime import Runtime
torch.set_num_threads(4)
r=Runtime(Config(model_dir=sys.argv[1],device='cuda:0',dtype=sys.argv[2],_env_file=None))
result=r.infer(DecisionRequest.model_validate_json(sys.stdin.read()))
result["model"]["transformers_version"]=transformers.__version__
print(json.dumps(result))
"""
    output = subprocess.run(
        [
            "poetry",
            "run",
            "python",
            "-c",
            script,
            str(args.destination.resolve()),
            dtype,
        ],
        cwd=args.inference_repo
        if args.runtime_environment == "inference"
        else Path(__file__).resolve().parents[1],
        input=json.dumps(request),
        text=True,
        capture_output=True,
        check=False,
        env={
            **{
                k: v
                for k, v in os.environ.items()
                if k not in {"VIRTUAL_ENV", "POETRY_ACTIVE"}
            },
            **(
                {"PYTHONPATH": str(args.inference_repo / "src")}
                if args.runtime_environment == "arietta"
                else {}
            ),
        },
    )
    if output.returncode:
        raise RuntimeError(output.stderr)
    actual = json.loads(output.stdout.strip().splitlines()[-1])
    errors = []
    tol = {"fp16": 1e-4, "bf16": 1e-3}[dtype]
    for i, probs in enumerate(expected):
        answer = actual["answers"][str(i)]
        errors.extend(
            abs(a - b) for a, b in zip(probs, answer["probabilities"].values())
        )
        if i == 0:
            assert (
                answer["choice"]
                == list(request["questions"]["0"]["criteria"])[
                    max(range(len(probs)), key=probs.__getitem__)
                ]
            )
        if i == 1:
            errors.append(
                abs(sum(j * p for j, p in enumerate(probs)) - answer["score"])
                / (len(probs) - 1)
            )
        if i == 2:
            errors.append(abs(probs[1] - answer["noul"]))
    (args.destination / f"parity-debug-{args.runtime_environment}.json").write_text(
        json.dumps({"expected": expected, "actual": actual, "errors": errors}, indent=2)
    )
    assert max(errors) <= tol, (max(errors), tol)
    report = {
        "runtime_environment": args.runtime_environment,
        "runtime_source": str(args.inference_repo.resolve()),
        "checkpoint": str(args.checkpoint),
        "dtype": dtype,
        "max_probability_or_normalized_score_error": max(errors),
        "tolerance": tol,
        "scorer_max_delta": delta,
        "finite_checkpoint": True,
        "act_head_unchanged": True,
        "choice_equal": True,
        "runtime": actual["model"],
    }
    (args.destination / f"parity-{args.runtime_environment}.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
