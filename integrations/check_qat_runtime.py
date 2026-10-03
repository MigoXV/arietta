"""QAT 工程验收：真实权重更新、导出、GPU INT8、独立推理与 HTTP 生命周期。"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import subprocess

import torch
import typer

from arietta.criterions.choice import ChoiceCriterion
from arietta.exporting.export import digest_file, export_checkpoint
from arietta.models.modeling import load_pretrained
from arietta.models.precision import deployment_copy
from arietta.models.processing import MODEL_KEYS
from arietta.models.quantization import Int8Linear
from arietta.tasks.data import DecisionDataModule
from arietta.tasks.task import FullFinetuneTask, LoraFinetuneTask

app = typer.Typer()
ROOT = Path(__file__).resolve().parents[1]


def http_lifecycle(python: Path, snapshot: Path, model: Path, requests, env):
    # httpx is provided by the existing inference environment, not a new dependency.
    script = r"""
import concurrent.futures,json,os,socket,subprocess,sys,time
import httpx
payload=json.load(sys.stdin)
with socket.socket() as listener:
    listener.bind(("127.0.0.1",0))
    port=listener.getsockname()[1]
process=subprocess.Popen([sys.executable,"-m","laya.commands.app","serve",
    "--model-dir",payload["model"],"--dtype","bf16","--runner","eager",
    "--device","cuda:0","--max-batch-size","4","--host","127.0.0.1",
    "--port",str(port)],stdout=sys.stderr,stderr=sys.stderr)
worker=None
try:
    with httpx.Client(base_url=f"http://127.0.0.1:{port}",timeout=30) as client:
        deadline=time.monotonic()+180
        while time.monotonic()<deadline:
            assert process.poll() is None,"service exited during startup"
            try:
                if client.get("/health/ready",timeout=1).status_code==200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
        else:
            raise RuntimeError("service startup timeout")
        info=client.get("/v1/info").json()
        worker=info["worker_pid"]
        assert info["quantization"]["scheme"]=="dynamic_w8a8"
        assert info["dtype"]=="bf16" and info["runner"]=="eager"
        def submit(request):
            response=client.post("/v1/decisions",json=request)
            response.raise_for_status()
            answer=response.json()["answers"]["q"]
            assert abs(sum(answer["probabilities"].values())-1)<1e-5
            return answer
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            answers=list(pool.map(submit,payload["requests"]))
        assert client.post("/v1/decisions",json={"state":"x","questions":{}}).status_code==422
finally:
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
if worker:
    try:
        os.kill(worker,0)
    except ProcessLookupError:
        pass
    else:
        raise RuntimeError("worker remains after HTTP shutdown")
print(json.dumps({"ready":True,"concurrent_requests":len(answers),
    "invalid_request_status":422,"worker_exited":True,"answers":answers,"model":info}))
"""
    result = subprocess.run(
        [str(python), "-c", script],
        cwd=snapshot,
        env=env,
        input=json.dumps({"model": str(model), "requests": requests}),
        text=True,
        capture_output=True,
        timeout=240,
    )
    if result.returncode:
        raise RuntimeError(result.stderr[-10000:])
    return json.loads(result.stdout.strip().splitlines()[-1])


@app.command()
def main(
    checkpoint: Path,
    destination: Path,
    snapshot: Path = typer.Option(ROOT / "tmp-workspace/laya-qat-validation"),
    runtime_python: Path = typer.Option(Path("/workspace/opusi/laya/.venv/bin/python")),
    dataset_path: str = typer.Option(str(ROOT / "data-bin/tiny-sft-v2")),
    output: Path = typer.Option(...),
):
    torch.set_num_threads(4)
    checkpoint, destination, snapshot = (
        path.resolve() for path in (checkpoint, destination, snapshot)
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    assert saved["global_step"] >= 2 and saved["qat_signature"]
    assert all(
        torch.isfinite(v).all()
        for v in saved["state_dict"].values()
        if v.is_floating_point()
    )
    cls = {
        "FullFinetuneTask": FullFinetuneTask,
        "LoraFinetuneTask": LoraFinetuneTask,
    }[saved["task_class"]]
    task = cls.load_from_checkpoint(
        checkpoint, criterion=ChoiceCriterion(), map_location="cpu"
    )
    deployed = deployment_copy(task.model, "bf16")
    baseline = load_pretrained(task.hparams.pretrained_model_path)
    scorer = task.model.decision.scorer
    if cls is LoraFinetuneTask:
        scorer = scorer.modules_to_save["default"]
    delta = float(
        (
            scorer[1].weight.detach().float()
            - baseline.decision.scorer[1].weight.detach().float()
        )
        .abs()
        .max()
    )
    assert delta > 0
    torch.testing.assert_close(
        deployed.decision.act_head[0].weight,
        baseline.decision.act_head[0].weight.bfloat16(),
        atol=0,
        rtol=0,
    )
    del baseline, task, saved
    if destination.exists():
        assert json.loads((destination / "export.json").read_text())[
            "checkpoint_sha256"
        ] == digest_file(checkpoint)
    else:
        export_checkpoint(checkpoint, destination, "service")
    reloaded = load_pretrained(str(destination))
    for name, value in deployed.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[name], value, atol=0, rtol=0)
    int8_count = sum(isinstance(m, Int8Linear) for m in reloaded.modules())
    del reloaded
    dm = DecisionDataModule(
        dataset_path,
        str(destination),
        required_splits=["validation"],
        state_serialization="verbatim",
    )
    dm.setup()
    rows = list(dm.dataset["validation"])
    requests = [
        {"state": row["state"], "questions": {"q": json.loads(row["question"])}}
        for row in rows
    ]
    deployed.cuda().eval()
    expected_single = []
    for row in rows:
        batch = dm.collator([row])
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            probs = (
                deployed(**{k: batch[k].cuda() for k in MODEL_KEYS})
                .logits.float()
                .softmax(-1)
                .cpu()
            )
        expected_single.append(probs[0, : len(batch["option_keys"][0])].tolist())
    deployed.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    env = {
        **os.environ,
        "PYTHONPATH": str(snapshot / "src"),
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    probes = []
    for size in (1, 2, 4):
        process = subprocess.run(
            [str(runtime_python), str(ROOT / "integrations/qat_runtime_probe.py")],
            cwd=snapshot,
            env=env,
            input=json.dumps(
                {"model": str(destination), "requests": requests, "batch_size": size}
            ),
            text=True,
            capture_output=True,
            timeout=180,
        )
        if process.returncode:
            raise RuntimeError(process.stderr[-10000:])
        result = json.loads(process.stdout.strip().splitlines()[-1])
        # Runtime buckets by length. Compare exactly those batches rather than
        # changing padding and SDPA dispatch along with the inference implementation.
        deployed.cuda()
        references = [None] * len(rows)
        errors = []
        for capture in result["raw_batches"]:
            inputs = {
                key: torch.tensor(
                    capture["inputs"][key],
                    dtype=torch.bool if key == "marker_mask" else torch.long,
                    device="cuda:0",
                )
                for key in MODEL_KEYS
            }
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                reference = deployed(**inputs).logits.float().softmax(-1).cpu()
            for i, index in enumerate(capture["request_indices"]):
                actual = result["results"][index]["answers"]["q"]["probabilities"]
                probs = reference[i, : len(actual)].tolist()
                references[index] = probs
                errors.extend(abs(a - b) for a, b in zip(probs, actual.values()))
        deployed.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        assert len(result["results"]) == len(rows)
        if max(errors) > 1e-3:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "batch_size": size,
                        "expected": references,
                        "actual": result,
                        "max_probability_error": max(errors),
                    },
                    indent=2,
                )
            )
        assert max(errors) <= 1e-3, (size, max(errors))
        assert result["int8_linear_count"] == int8_count
        batch_drift = max(
            abs(a - b)
            for single, batched in zip(expected_single, references)
            for a, b in zip(single, batched)
        )
        assert batch_drift <= 1e-3, ("batch probability drift", size, batch_drift)
        typer.echo(
            json.dumps(
                {
                    "batch_size": size,
                    "max_probability_error": max(errors),
                    "batch_vs_single_max_probability_drift": batch_drift,
                    "int_mm_calls": result["int_mm_calls"],
                }
            ),
            err=True,
        )
        probes.append(
            {
                "batch_size": size,
                "max_probability_error": max(errors),
                "batch_vs_single_max_probability_drift": batch_drift,
                **result,
            }
        )
    del deployed
    http = http_lifecycle(runtime_python, snapshot, destination, requests[:6], env)
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest_file(checkpoint),
        "model": str(destination),
        "snapshot": str(snapshot),
        "runtime_python": str(runtime_python),
        "scorer_max_training_delta": delta,
        "act_head_unchanged": True,
        "strict_reload_equal": True,
        "int8_linear_count": int8_count,
        "probability_tolerance": 1e-3,
        "probes": probes,
        "http": http,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    typer.echo(
        json.dumps(
            {
                "report": str(output),
                "int8_linear_count": int8_count,
                "max_probability_error": max(
                    p["max_probability_error"] for p in probes
                ),
                "http_worker_exited": http["worker_exited"],
            }
        )
    )


if __name__ == "__main__":
    app()
