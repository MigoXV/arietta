# Arietta 工程验收记录

日期：2026-10-03。范围为训练代码迁移和工程闭环，未开展正式领域训练、语义质量对照或强化学习。

## 环境与来源

训练使用既有 Python 3.10 / Torch 2.8.0+cu128，Lightning 2.6.5、PEFT 0.19.1、Transformers 5.18.0、datasets 4.8.4。比对安装前后清单，Torch、torchaudio、triton、NVIDIA CUDA 包均未改变。训练依赖由 Poetry 外的 pip 管理，并提供 runtime-constraints.txt 防止意外升级。

源资产 `/workspace/model-bin/MigoXV/laya-multilingual/model.safetensors` 包含 169 个 F16 张量和 1 个 F32 temperature 缓冲区；旧配置中的 float32 字样并不代表参数精度。原始资产和旧训练仓库均未修改。

GPU0 上已有用户 Laya 服务，保持运行。训练与模型验收作业串行使用 GPU0；本次没有测吞吐或延迟，因此不做性能结论。

## 自动化验证

- Arietta：22 项 CPU 测试，覆盖三种 Task 的实际 fit/resume/export、三类问题与软硬标签、原生/HF 严格加载、FP16/BF16 导出张量、RoPE 缓冲区精度、数据入口与泄漏拒绝、不可变配置、长度拒绝和校准。
- Laya：独立补丁测试覆盖 BF16 配置、eager autocast、动作头精度与新版 RoPE 参数向旧字段映射。真实 HTTP 生命周期测试需显式开启。
- 四个真实模型运行均在 GPU0 训练两步并进行全验证，best 和 latest 分开保存。每组验证数据仅 6 条合成样本，硬/软标签各 3 条，不能用来判断泛化质量。
- 训练后保留 FP32 主参数/优化器状态，验证副本与交付参数为对应 FP16 或 BF16。所有浮点缓冲区保持原精度。部署验证副本不进入检查点。

训练验证 NLL 的原始摘要见 [smoke-metrics.json](smoke-metrics.json)。这些数值仅用于确认损失、指标与检查点链路正常，不用于比较哪种精度更好。

## 验收中发现并修复的问题

1. LoRA 的 `modules_to_save=["head", ...]` 会匹配到 `act_head`，现改为完整模块名。动作头保持冻结，单独核对导出权重未改变。
2. 子进程继承 Poetry 的 VIRTUAL_ENV 会错误使用训练环境，跨仓库验收已显式选择运行环境。
3. 验收期间，外部安装进程将 Laya 环境改为 Torch 2.9.1 / Transformers 4.57.6 / tokenizers 0.22.2。本任务未发起、回退或覆盖此次安装。
4. Transformers 4.x 忽略新版 `rope_parameters`，静默采用旧字段默认值。已为推理加载补充 global/local RoPE theta 映射。首组 FP16 跨版本误差从约 0.135 降到 0.000233。

## 数值门槛与运行环境

同一运行库环境下，训练侧部署副本与实际 Laya Runtime 代码比较，FP16 的最大概率/归一化 score 误差门槛为 1e-4，BF16 为 1e-3；choice 必须相同。使用 `--runtime-environment arietta` 将固定的 Laya 源码快照放入子进程 PYTHONPATH，双方共享固定的 Torch/Transformers 版本。

默认 `--runtime-environment inference` 使用 Laya 独立环境，也执行相同严格门槛。已测首组 FP16 跨版本最大误差 0.0002326965，未通过 1e-4，不视为通过，也未放宽门槛。不同运行库版本带来的这部分差异仍需后续专项评估；不能把它算作训练或导出精度损失。

5. 首次 Full-FP16 的默认 GradScaler 初始 scale=65536 导致两次更新全部被跳过，global_step 仍为 2。此运行不计入通过。原生 MixedPrecision 插件显式配置 init_scale=128 后，full-fp16-safe 成功更新两次。训练结束与导出现在都会拒绝零优化器更新的检查点。

最终通过的四组优化器审计见 [checkpoint-audit.json](checkpoint-audit.json)：实际 optimizer step 都为 2，全部浮点优化器状态有限。参数级验收还检查分类头更新及冻结动作头不变。

同期 Laya 工作区另有任务持续修改 SDPA 后端和 RoPE 前向；中途出现的 math 后端结果不能与 auto 后端混作一组对照。最终数值验收固定在 Laya 提交 `5877eb7` 加 `integrations/laya-bf16.patch` 的隔离快照，两个进程同时固定 Torch 2.8.0 / Transformers 5.18.0。快照位于 tmp-workspace/laya-validation，不改变用户的活动仓库。

最终模型级数值结果和实际 HTTP 生命周期结果记录于下方。

## 最终数值验收

| 运行 | 部署精度 | 最大概率/归一化 score 误差 | choice |
| --- | --- | --- | --- |
| LoRA-FP16 | FP16 | 2.98e-8 | 一致 |
| LoRA-BF16 | BF16 | 3.73e-9 | 一致 |
| Full-FP16（safe） | FP16 | 2.98e-8 | 一致 |
| Full-BF16 | BF16 | 2.98e-8 | 一致 |

四组推理代码 SHA-256 均为 `2fab376f16c0ea75dc76dc597a58fff24e2ab8fe3f7e70599cc0f366b99994c6`。完整模型指纹、参数更新幅度和冻结策略头检查见 [runtime-parity.json](runtime-parity.json)。导出目录为 `model-bin/{lora,full}-{fp16,bf16}`，每个约 647 MiB；训练 checkpoint 另存 outputs/checkpoints，二者不混用。

必须区分：以上证明迁移、训练、恢复、部署精度交付与固定运行库下推理对齐成功；不能据此认定微调提高了真实任务准确率，也不能认定活动推理工作区未来版本仍维持这些数值。跨版本与持续修改中的工作区需重新验收。

## BF16 HTTP 生命周期

使用固定 Laya 代码快照、Laya 当前独立环境（Torch 2.9.1 / Transformers 4.57.6），加载 `model-bin/lora-bf16`，在 GPU0 与随机空闲端口运行 HTTP/Worker 验收：启动就绪、info 中 BF16/autocast、真实请求返回有限概率、关闭后 worker PID 消失均通过。该测试文件 4 项通过（22.90 秒）；此处的耗时是测试执行时间，不是推理性能数据。

命令：

```bash
cd /workspace/opusi/laya
PYTHONPATH=/workspace/opus/arietta/tmp-workspace/laya-validation/src \
ARIETTA_BF16_E2E_MODEL=/workspace/opus/arietta/model-bin/lora-bf16 \
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
poetry run pytest -q /workspace/opus/arietta/tmp-workspace/laya-validation/tests/test_bf16_arietta.py
```

这项生命周期测试证明当前运行库能加载导出模型并通过服务执行，不替代上面的严格数值对照。既有活动仓库的其他修改没有归入 Arietta 提交；本次推理修改以独立补丁交付。
