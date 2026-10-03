"""由指定的推理 Python 执行；标准输入只包含无标注模型请求。"""

from __future__ import annotations

import json
import sys

import torch
import transformers

from laya.config import Config
from laya.contracts import DecisionRequest
from laya.quantization import Int8Linear
from laya.runtime import Runtime


def main():
    payload = json.load(sys.stdin)
    config = Config(
        model_dir=payload["model"],
        device="cuda:0",
        dtype="bf16",
        runner="eager",
        max_batch_size=payload["batch_size"],
        _env_file=None,
    )
    torch.cuda.reset_peak_memory_stats()
    runtime = Runtime(config)
    try:
        requests = [DecisionRequest.model_validate(row) for row in payload["requests"]]
        lookup = {
            (tuple(item["ids"]), item["qtype"]): index
            for index, request in enumerate(requests)
            for _, _, item in runtime.prepare(request)
        }
        captures = []

        def observe(batch, logits, acts):
            indices = []
            for ids, mask, kind in zip(
                batch["input_ids"], batch["attention_mask"], batch["qtype"]
            ):
                indices.append(lookup[(tuple(ids[mask.bool()].tolist()), int(kind))])
            captures.append(
                {
                    "request_indices": indices,
                    "inputs": {
                        key: batch[key].tolist()
                        for key in (
                            "input_ids",
                            "attention_mask",
                            "marker_pos",
                            "marker_mask",
                            "qtype",
                        )
                    },
                    "logits": logits.float().cpu().tolist(),
                }
            )

        runtime.runner.raw_observer = observe
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as profile:
            results = runtime.infer_many(requests)
        int_mm_calls = sum(
            event.count
            for event in profile.key_averages()
            if event.key == "aten::_int_mm"
        )
        assert int_mm_calls > 0
        packed = [
            module
            for module in runtime.runner.model.modules()
            if isinstance(module, Int8Linear)
        ]
        assert len(packed) == len(runtime.info["quantization"]["modules"])
        assert all(m.qweight.dtype == torch.int8 for m in packed)
        assert all(m.qweight.device.type == "cuda" for m in packed)
        print(
            json.dumps(
                {
                    "results": results,
                    "raw_batches": captures,
                    "batches": runtime.last_batches,
                    "model": runtime.info,
                    "transformers_version": transformers.__version__,
                    "int_mm_calls": int_mm_calls,
                    "int8_linear_count": len(packed),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                }
            )
        )
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
