# QAT 工程验收

日期：2026-10-03。范围为可选动态 W8A8 QAT 的训练、恢复、部署精度验证、模型导出、加载、离线评估及推理服务闭环。此次使用两步训练验收工程能力，不用于判断 QAT 是否提高领域质量，也未测正式性能收益。

## 环境与实现

训练环境保留 Torch 2.8.0+cu128、Transformers 5.18.0、Lightning 2.6.5、PEFT 0.19.1、datasets 4.8.4。独立推理环境保留 Torch 2.8.0+cu128、Transformers 4.57.6。本次没有安装 TorchAO、修改依赖版本或修改 Poetry lock。

Full、LoRA、Scratch 共享可注入的 `Int8QAT` 策略。权重逐输出通道、激活逐 token 动态对称量化到 `[-127,127]`，训练使用 STE；导出为 INT8 权重及 FP32 scale，推理执行真实 `torch._int_mm`，不回退到浮点 GEMM。BF16 用于剩余浮点参数和 autocast 输出，RoPE 及量化 scale 保留 FP32。LoRA 的有效合并权重整体参与假量化，dropout 必须为 0。

QAT 路径固定 SDPA math 后端。验收初期，自动选择的 fused 后端在 batch 1 与 batch 2/4 间出现约 0.22 的最大概率漂移；同一实际批次两种实现的输出却能对齐。因此此次将后端选择写入训练签名和模型元数据，同时用于训练、验证与部署，并另外检查 batch 数值稳定性。该选择有性能取舍，恢复 fused SDPA 必须另做验收。

Transformers 4.57 与 5.18 的 RoPE 存在精度差异：前者从 BF16 QKV 生成 cos/sin，后者从 FP32 残差流生成。兼容代码统一位置编码输入、FP32 旋转计算和 SDPA 的 bool/None 遮罩；不修改已安装的 Transformers。修正后，同一输入的嵌入、22 个编码层、最终归一化逐元素一致。

## 自动化和真实权重

- Arietta 的 50 项 CPU 回归通过，默认不依赖网络、GPU或大模型。包括三类 Task 的实际参数更新、LoRA B 更新、checkpoint 安全恢复及继续训练、配置漂移拒绝、导出/重载、FP16/BF16 缓冲区精度、损坏资产拒绝、离线评估和原生 LightningCLI 配置。
- Laya 固定快照的相关回归通过；真实 HTTP 生命周期由专门脚本在 GPU0 上执行。
- 官方 FP16 来源 `/workspace/model-bin/MigoXV/laya-multilingual` 上分别完成 Full-BF16-QAT 和 LoRA-BF16-QAT，两次实际优化器更新；88 个 QAT 层均参与两次训练前向，浮点状态有限，动作头冻结。
- 数据来自标准 HF 仓库 `data-bin/tiny-sft-v2`：train 60、validation 12、calibration 12、test 12；本次训练仅取两个 batch。参考标注仅用于训练损失、验证及审计，不传入推理请求。

完整优化器步骤、训练签名、指标见 [qat-smoke-metrics.json](qat-smoke-metrics.json)。小样本 accuracy/NLL 只能用于确认指标链路工作，不作为量化精度或泛化提升的证据。

## 推理验收门槛

通过 `integrations/check_qat_runtime.py` 执行。固定 Laya 源码提交 `402b9be916e07350aabe4ea97ce1f3f03467c3d7`，应用 `integrations/laya-qat-int8.patch` 并复制共用量化模块到隔离快照；当前活动推理仓库继续由其原任务维护。

每组模型使用全部十二条 validation 请求、choice/score/noul 三类问题和 batch 1/2/4。推理进程只接收 state/question。捕获实际分桶后的 tensor batch，在训练环境部署模型中重放同一形状，训练/推理概率误差门槛为 `1e-3`；batch 相对单条推理的漂移也须小于 `1e-3`。二者分别报告，避免把分桶、padding、库实现的影响混在一起。

HTTP 验收使用随机本地端口，检查 ready、真实量化信息、六个并发请求、非法请求返回 422、正常关闭和 worker PID 消失。所有临时服务均在测试后关闭。

最终数值和服务摘要见 [qat-runtime-parity.json](qat-runtime-parity.json)。每次运行的完整原始报告保留于 `outputs/qat-{lora,full}-acceptance.json`，含整数算子调用次数、实际 batch 和运行库信息。Torch 的 peak allocated 是测试进程分配峰值，不等同于 nvidia-smi 总显存或性能基准。

| 运行 | 最大训练/推理概率误差 | 最大 batch 概率漂移 | HTTP |
| --- | --- | --- | --- |
| LoRA-BF16-QAT | 5.96e-8 | 0.000741661 | 六个并发请求通过，worker 已退出 |
| Full-BF16-QAT | 5.96e-8 | 0.000569656 | 六个并发请求通过，worker 已退出 |

每组模型的 batch 1/2/4 整数算子调用次数分别为 1056、616、352；这些是十二条请求的执行计数。Laya 相关 CPU 回归为 19 项通过、1 项显式 GPU 生命周期测试跳过；上表的真实 GPU HTTP 验收通过专门脚本完成。

## 交付

原生部署资产位于 `model-bin/qat-{lora,full}-bf16`，保持官方 `format_version=1`，增加量化元数据。标准 HF 包装模型位于 `model-bin/qat-{lora,full}-bf16-hf`，`model_type=arietta_laya`，先调用 `register_models()` 即可经 AutoModel 加载。两种格式均包含 config、tokenizer、safetensors，导出前后逐张量一致；训练 checkpoint 独立保存。

推理接入交付为固定快照和可应用补丁，复现命令见 [QAT 使用说明](../examples/qat/README.md)。当前仅验收 eager，CUDA Graph/compile 会明确拒绝。模型温度重置为 1，需要独立 calibration split 重新校准。正式 QAT 质量、长输入与目标业务分布、fused 后端及性能对照留待后续实验，不能用本次工程结果替代。
