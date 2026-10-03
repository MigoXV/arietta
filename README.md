# Arietta

Laya 通用决策模型训练框架。直接读取 `/workspace/model-bin/MigoXV/laya-multilingual` 的 `format_version=1` 资产，支持 choice、score、noul，以及 Scratch、Full、LoRA 三种训练入口。历史会议实验继续保留在 `/workspace/opus/meetnote-decision-train`；本项目不搬动历史数据、权重与运行记录。

## 环境

Python 3.10，使用 Poetry。当前 Torch 为已安装的 2.8.0+cu128，不由 Poetry 管理或替换。

```bash
poetry install
# Torch 已安装；以下运行依赖通过 pip 管理，升级前自行检查兼容性。
poetry run pip install -c runtime-constraints.txt 'lightning==2.6.5' 'peft==0.19.1' 'datasets==4.8.4'
poetry run pytest -q
```

不要使用 `poetry sync` 清理 pip 管理的训练依赖。`poetry.lock` 不包含 Torch。

## 训练与恢复

生产入口是原生 LightningCLI；Typer `arietta` 只提供导出、评估和校准工具。示例是工程冒烟配置，每次只训练两步，不能作为质量验证。

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 poetry run python -m arietta.commands.train fit --config examples/lora-fp16.yaml
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 poetry run python -m arietta.commands.train fit --config examples/lora-bf16.yaml
# Full 配置分别为 examples/full-fp16.yaml、examples/full-bf16.yaml
# 仅 ckpt_path 可以在同一 run 中改变；恢复 optimizer/scheduler/global_step。
CUDA_VISIBLE_DEVICES=0 poetry run python -m arietta.commands.train fit --config examples/lora-fp16.yaml --ckpt_path outputs/checkpoints/lora-fp16/last.ckpt
```

Full/LoRA 只接受 `pretrained_model_path`；Scratch 只接受没有权重的 `model_config_path`。任务、数据、criterion、logger、callbacks 都在 YAML 的 `class_path/init_args` 中声明。更改训练轮数等配置需新建 run，不能覆盖原 run 的 `resolved.yaml`。模型、数据和运行库指纹发生漂移时拒绝恢复。

## 精度与检查点

官方权重的参数是 FP16，配置中的历史 `encoder.dtype=float32` 不能代表真实权重精度。加载器以 safetensors 张量为准，原生权重严格检查 key/shape。训练允许 FP32 主参数与优化器状态，但前向分别使用 `16-mixed`、`bf16-mixed`；损失和概率运算使用 FP32。

每次验证复制当前模型，合并 LoRA，再转换为目标部署精度。复制品不注册到训练模块，不会进入 checkpoint。RoPE 等浮点缓冲区保持原精度。`val_deploy_loss` 是当前验证集中各问题类型 NLL 的等权平均，用于选择 best；latest 独立保存。硬标签报告 accuracy/macro-F1、score 的 MAE/RMSE，所有标签报告 NLL/Brier，并分别记录 hard/soft 数量。当前只支持单设备训练和验证。

Full-FP16 示例通过原生 Lightning MixedPrecision 插件将 GradScaler 初始 scale 设为 128（此时 trainer.precision 留空，由插件声明 16-mixed）。最初两步曾被默认 scale 的溢出保护全部跳过，失败记录保留在 outputs/checkpoints/full-fp16；正式通过的冒烟 run 为 full-fp16-safe。训练结束与导出都会拒绝没有发生优化器更新的运行。

动作策略头 `act_head` 冻结，不进行强化学习训练。训练后温度重置为 1；之前的校准结果不可沿用。

## 数据

所有训练数据仅通过 `datasets.load_dataset()` 加载。数据列为 `id/group_id/family_id/task/split/state/question/target`。question 和 target 建议存为 JSON 字符串，避免 Arrow 自动展开动态字段。不同 split 不可复用样本、group 或 family。task 是非空业务分类字符串，不限定会议任务。

- choice：criteria 为 2–16 个键及说明，或唯一字符串列表；`target={"label":"语义键"}`。
- score：criteria 为 2–16 个有序等级说明；`target={"label":1}`，等级从 0 开始。
- noul：criteria 省略或包含 false/true 的说明；`target={"label":true}`。
- 任意类型支持 `target={"distribution":[0.2,0.8]}`，顺序与选项一致；noul 额外支持 `target={"probability":0.8}`。
- 硬标签与软标签互斥；旧 choice 的字符串 `label` 列仍可使用。

`state_serialization=verbatim` 原样使用文本；`online_json` 解析 JSON 对象/数组，保持插入顺序并复用在线序列化。超长问题、选项和状态明确拒绝，不截断。examples/fixtures 只是可公开的合成工程样本。

## 导出、评估与校准

```bash
poetry run arietta export outputs/checkpoints/lora-fp16/last.ckpt model-bin/lora-fp16 --format service
poetry run arietta export outputs/checkpoints/lora-bf16/last.ckpt model-bin/lora-bf16 --format service
poetry run arietta evaluate --pretrained-model-path model-bin/lora-fp16 --data-file examples/fixtures/calibration.jsonl --split calibration --state-serialization verbatim --device cuda:0 --dtype fp16 --output outputs/calibration-fp16.json
poetry run arietta calibrate outputs/calibration-fp16.json --output outputs/temperature-fp16.json
```

默认导出精度来自 checkpoint；禁止指定未经该 run 验证的另一精度。`service` 导出根目录 config/tokenizer/model.safetensors，可直接交给 `/workspace/opusi/laya`。`hf` 为注册的 `arietta_laya` HF 包装格式，需先调用 `register_models()`；`adapter` 仅适用于 LoRA，保存 FP32 训练适配器，不冒充低精度独立部署模型。导出前后逐张量核对。

校准只允许 calibration split，结果绑定模型指纹、数据摘要、精度和序列化方式；不同训练或导出精度需重新校准。当前温度工具输出供审阅，不自动改写模型资产。

Laya 推理侧 BF16 最小补丁在 `integrations/laya-bf16.patch`，已应用到当前工作区；该仓库原有未提交改动保留。训练与推理实现之间的同精度验证脚本见 integrations。

## 目录与边界

`models` 管模型/输入编码，`tasks` 管数据与训练步骤，`criterions` 管损失，`configs` 管不可变运行记录，`exporting` 管交付，`evaluation` 管离线评估。`data-bin/model-bin/outputs/tmp-workspace` 均忽略 Git。原始上游实现和许可证见 THIRD_PARTY.md。

## 跨仓库验收

推理工作区同期有其他任务修改，因此数值验收固定在已提交版本与本次补丁上。建立快照（目录只创建一次）：

```bash
mkdir -p tmp-workspace/laya-validation
git -C /workspace/opusi/laya archive 5877eb7 src/laya tests pyproject.toml poetry.lock | tar -x -C tmp-workspace/laya-validation
git apply --directory=tmp-workspace/laya-validation integrations/laya-bf16.patch
```

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 poetry run python integrations/check_runtime.py outputs/checkpoints/lora-fp16/last.ckpt model-bin/lora-fp16 --runtime-environment arietta --inference-repo /workspace/opus/arietta/tmp-workspace/laya-validation
# LoRA-BF16 和 Full-BF16 分别替换为 lora-bf16、full-bf16，串行执行。
# Full-FP16 使用显式 loss scale 配置的新 run：
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 poetry run python integrations/check_runtime.py outputs/checkpoints/full-fp16-safe/last.ckpt model-bin/full-fp16 --runtime-environment arietta --inference-repo /workspace/opus/arietta/tmp-workspace/laya-validation
```

`--runtime-environment arietta` 使用固定训练环境执行现有 Laya 源码，隔离运行库差异。省略该参数则使用推理仓库自己的环境并保持相同严格门槛。验收期间推理环境发生外部版本变更，跨版本差异详见报告。

脚本检查检查点张量有限、分类头实际更新、动作头不变、导出重新加载，并使用推理仓库独立 Poetry 环境运行相同输入。FP16 最大概率/归一化 score 误差门槛为 1e-4，BF16 为 1e-3，choice 必须一致。已有导出只在 checkpoint 摘要匹配时复用。实际结果见 [验收记录](docs/validation.md)。
