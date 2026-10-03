# 可选 W8A8 量化感知训练

在已有 Scratch、Full、LoRA Task 上配置 `quantization` 即启用 QAT；省略时沿用浮点训练。训练入口仍为原生 LightningCLI，数据仍由 `datasets.load_dataset()` 加载。

```yaml
quantization:
  class_path: arietta.models.quantization.Int8QAT
  init_args:
    exclude_modules: []
```

## 数值与精度

首版支持动态、对称 W8A8：权重按输出通道、激活按 token 的最后一维计算 `max(abs(x)).clamp_min(1e-8) / 127`，四舍五入后限制到 `[-127, 127]`。无需静态激活校准。训练使用 STE 假量化及 FP32 主参数，线性层的输出遵循 Trainer 的 autocast 精度；推理转换为真实 INT8 权重和 FP32 scale，调用 `torch._int_mm` 完成整数乘法。

模型元数据还固定 RoPE 的 FP32 计算规则。对现有 Transformers 4.57 的 ModernBERT SDPA，兼容层从残差输入生成位置编码、在 FP32 中旋转 Q/K 后返回原精度，并将遮罩规范为布尔张量或 None，与训练环境 5.18 一致；只绑定当前量化模型的 attention 方法，不修改已安装库或浮点模型。

QAT 路径固定使用 SDPA math 后端，覆盖编码器及浮点决策头。自动选择的 fused 后端在本次动态 INT8 模型上产生了较大的 batch 概率漂移；math 在当前小样本上将该漂移降到既定 `1e-3` 门槛内。此策略同时用于训练、部署验证及推理，进入和退出 forward 时恢复原后端设置，不影响普通浮点模型。它仍使用真实 INT8 线性层，但不据此承诺推理加速；恢复 fused SDPA 前需重新做稳定性和性能验收。

`deployment_dtype=bf16` 表示其余浮点层和中间输出使用 BF16。它不会把已打包权重转回 BF16，也不会把 scale 或 RoPE 缓冲区降成 BF16。每次验证都复制当前模型、合并 LoRA、转换 INT8，并按部署精度计算 `val_deploy_loss`。训练 checkpoint 保留可恢复的浮点参数及优化器状态。

量化范围是 ModernBERT 编码器每层的 `attn.Wqkv`、`attn.Wo`、`mlp.Wi`、`mlp.Wo`；嵌入、归一化、决策头及动作头保留浮点。`exclude_modules` 可排除完整路径，例如 `encoder.layers.0.attn.Wo`；不存在的路径、重复路径或排除全部层会报错。动作头仍冻结。

LoRA 必须设置 `dropout: 0.0`，训练对 `base + LoRA delta` 的有效权重整体做假量化；导出先合并 LoRA 再量化，避免训练与部署使用不同规则。QAT checkpoint 不支持独立 `adapter` 导出，请使用 `service` 或 `hf`。已量化部署资产不能直接作为新一轮浮点微调的来源；继续训练应恢复 checkpoint。

## 运行

两份示例使用本地官方权重和 `tiny-sft-v2` 数据集，仅训练两个优化器步骤、验证十二条记录，用于工程验收。

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 poetry run python -m arietta.commands.train fit --config examples/qat/lora-bf16.yaml
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 poetry run python -m arietta.commands.train fit --config examples/qat/full-bf16.yaml
poetry run arietta export outputs/checkpoints/qat-lora-bf16/last.ckpt model-bin/qat-lora-bf16 --format service
poetry run arietta export outputs/checkpoints/qat-full-bf16/last.ckpt model-bin/qat-full-bf16 --format service
# HF 包装格式支持 AutoModel 加载，先调用 arietta.models.register_models()。
poetry run arietta export outputs/checkpoints/qat-lora-bf16/last.ckpt model-bin/qat-lora-bf16-hf --format hf
```

完整训练需使用新 run，修改 `max_epochs`、移除 batch 数限制，并设置新的 logger version、checkpoint 目录和 default_root_dir；不得覆盖已有 resolved config。恢复时保持原配置，仅添加 `--ckpt_path`。QAT 版本及排除列表存为普通 checkpoint 字段，支持 Torch 安全加载；恢复时配置不一致会拒绝。

## 部署边界

导出为根目录的 config/tokenizer/model.safetensors；`service` 采用官方 `format_version=1`，增加 `quantization` 元数据；`hf` 使用已注册的 `arietta_laya` 格式和 `arietta_quantization` 元数据。文件中的权重为 INT8 张量，scale 为 FP32，未量化层按该 run 的部署精度保存。加载严格校验方案版本、模块路径、权重 dtype、shape 和有效 scale。

当前安装的 Torch 2.8.0 同时支持 CPU 和 CUDA 的整数 GEMM。CUDA 路径将 M 至少补到 32，K/N 补到 32 的倍数，计算后裁去 padding。`torch._int_mm` 是 Torch 内部算子，升级 Torch 时应重新验收；不提供浮点乘法回退。本次不安装 TorchAO，也不修改任何依赖版本。

推理仓库同期有独立的 INT8 工作，因此通过固定源码快照交付兼容补丁，不改动它的活动分支。首版兼容 `runner=eager`，CUDA Graph/compile 会明确拒绝，需单独适配和验收。

```bash
# 创建一次；在 Arietta 根目录执行。
mkdir -p tmp-workspace/laya-qat-validation
git -C /workspace/opusi/laya archive 402b9be916e07350aabe4ea97ce1f3f03467c3d7 src/laya tests pyproject.toml poetry.lock README.md | tar -x -C tmp-workspace/laya-qat-validation
git apply --directory=tmp-workspace/laya-qat-validation integrations/laya-qat-int8.patch
cp src/arietta/models/quantization.py tmp-workspace/laya-qat-validation/src/laya/quantization.py

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 poetry run python integrations/check_qat_runtime.py outputs/checkpoints/qat-lora-bf16/last.ckpt model-bin/qat-lora-bf16 --output outputs/qat-lora-acceptance.json
# Full 使用对应的 checkpoint、model-bin 和输出路径，串行执行。
```

脚本核对真实参数更新、冻结动作头、严格重载、batch 1/2/4 的概率、GPU 上整数算子调用，以及 HTTP ready、并发请求、无效输入和 worker 退出。同一实际分桶批次的训练/推理概率误差与 batch 相对单条推理的漂移分别检查，两项门槛均为 `1e-3`。仅无标注的 state/question 进入推理进程。默认使用推理仓库现有 Python 环境，也可通过 `--runtime-python` 显式指定另一个已安装环境。

这些检查证明训练到部署的工程链路可用。两步训练及小样本结果不能证明 QAT 提升质量，真实整数计算也不保证延迟下降；正式精度、速度对照应使用固定数据和运行配置另行完成。验收记录见 [QAT 验收](../../docs/qat-validation.md)。
