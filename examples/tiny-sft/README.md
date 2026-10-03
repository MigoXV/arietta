# 小型 SFT 数据集

这是 Laya 的中文决策监督数据，使用合成发言和逐条指定的硬标签。不是对话生成 SFT，也不是原有 `examples/fixtures` 的流程样例。32 条独立发言各有 choice、score、noul 三项监督，共 96 条训练格式记录。

| 划分 | 发言数 | 监督记录数 | 场景 |
| --- | ---: | ---: | --- |
| train | 20 | 60 | 权限、版本发布、回归测试、新人导师、通知 |
| validation | 4 | 12 | 会议室预约 |
| calibration | 4 | 12 | 文档入口 |
| test | 4 | 12 | 工作坊组织 |

同一发言的三个问题在同一划分，场景 family 也不跨划分；所有划分均覆盖三种立场、三个确定程度等级及承诺的真假。训练不使用 calibration/test 的标签，温度只可在 calibration 上拟合。

## 标签规则

- `stance` / choice：本人对议题的**最新立场**。支持或倾向支持为 `support`；反对或倾向反对为 `oppose`；仅提问、引用他人、尚未决定、撤回旧意见后未决定为 `undecided`。暂定倾向归其方向，不标为未表态。
- `certainty` / score：**议题立场的确定程度**，不是承诺强度或对事实的确信程度。0 为未表态/未决定；1 为暂定倾向/最终立场仍有条件；2 为明确无条件支持或反对。与 stance 的语义一致：undecided 对应 0，support/oppose 对应 1 或 2。
- `commitment` / noul：本人是否**已经无条件确认亲自执行具体任务**。明确接受任务或承诺执行为 `true`；表达支持、能够做、建议、考虑帮忙、条件尚未满足的未来承诺、引用他人承诺、否认接受为 `false`。任务可以与议题立场独立，因此反对方案或尚未决定也可能是 true。

只用原文能确定的硬标签，不编造概率。不把“没有证据”解释成“人一定不会做”，false 仅表示没有已确认的承诺。合成发言刻意把判断依据写明确，避免需要推测语气或常识的歧义。

例如：“是否本周上线我还没决定……发布说明我已经答应负责，我会在明天中午前写完。”标为 `undecided / 0 / true`；“我暂时倾向支持……如果试点通过，我才负责配置；现在还没有接下配置任务。”标为 `support / 1 / false`。

## 生成与审阅

在 Arietta 根目录执行：

```bash
poetry run arietta build-tiny-sft
```

源文件 [cases.jsonl](cases.jsonl) 记录每条发言的标签、原文证据和解释，由助手编写并逐条复核；标签不来自模型预测或自动关键词推断。未经过第二位独立人工标注者复核。构建时检查标签类型、证据确实位于原文、立场与确定程度一致、划分隔离、重复文本和各标签覆盖。程序检查不等于自动证明语义正确，语义依据供逐条审阅。

输出采用 [Hugging Face 官方数据集仓库布局](https://huggingface.co/docs/hub/en/datasets-manual-configuration)，不需要自定义加载脚本：

```text
data-bin/tiny-sft-v2/
  README.md
  data/
    train-00000-of-00001.parquet
    validation-00000-of-00001.parquet
    calibration-00000-of-00001.parquet
    test-00000-of-00001.parquet
data-bin/tiny-sft-v2-audit/
  annotations.jsonl
  review.md
  manifest.json
  preflight.json
```

README 是中文数据集卡，YAML front matter 包含默认配置 default、四个 split 的文件映射及 dataset_info（Features、样本数和字节数）。数据由 HF Dataset.to_parquet 写入，Parquet 内含 Hugging Face schema 元数据。Features 显式声明八个 string 列：`id/group_id/family_id/task/split/state/question/target`；question/target 为 JSON 字符串，解码 target 后只有 label，保留 choice 字符串、score 整数、noul 布尔值的原始语义类型。

完整标注、解释、审阅表和 manifest 存放在相邻的 audit 目录，**不属于 DatasetDict，不进入模型输入**。manifest 记录版本、Features、来源、问题定义、划分规模、标签分布和 SHA256；源标注、问题或格式改变后需选择新输出目录。旧 `tiny-sft-v1` 的 JSONL 产物保留为历史记录，当前配置使用 v2。

## 直接使用 Hugging Face Datasets

```python
from datasets import load_dataset

dataset = load_dataset("/workspace/opus/arietta/data-bin/tiny-sft-v2")
# DatasetDict: train=60, validation=12, calibration=12, test=12
print(dataset["train"].features)
train = load_dataset("/workspace/opus/arietta/data-bin/tiny-sft-v2", split="train")
```

不需要设置 data_files，split 由仓库元数据识别。也可指定 `name="default"`，或使用 `split="train[:3]"`。这是可直接供 load_dataset 使用的仓库格式，训练不依赖 save_to_disk/load_from_disk 的缓存快照格式。

`state` 仅包含议题及发言原文；模型输入为 state 加问题/选项，标签单独送入损失。构建与训练数据读取都使用 `datasets.load_dataset()`，无需下载公开数据。数据产物忽略 Git，可从已提交的源标注确定性重建。

## 训练配置

```bash
poetry run arietta build-tiny-sft
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 poetry run python -m arietta.commands.train fit --config examples/tiny-sft/lora-bf16.yaml
```

配置使用现有官方 FP16 资产初始化 LoRA，BF16 混合精度训练，按 BF16 部署副本的验证损失选择 best。训练 5 轮，每轮完整使用 60 条记录，batch_size=4，共 75 个更新步；没有沿用两步冒烟限制。该配置是小数据调试起点，超参数未做质量调优。

训练后使用 test 做一次独立评估，不能按 test 结果挑 checkpoint；需要校准则先用 calibration。训练配置使用 dataset_path 指向仓库，dataset_name=default，不手工列出 data_files。本次数据源格式变化使用新的 run-002；新实验需更换 logger.version、default_root_dir 和两个 checkpoint.dirpath，遵守不可变 run 配置规则。

现有 evaluate 入口支持单个 JSONL 或 Parquet 文件，例如：

```bash
poetry run arietta evaluate --pretrained-model-path model-bin/tiny-sft-lora-bf16 --data-file data-bin/tiny-sft-v2/data/test-00000-of-00001.parquet --split test --state-serialization verbatim --device cuda:0 --dtype bf16 --output outputs/tiny-sft-test.json
```

此示例需要先训练并导出对应模型；本次没有生成该模型权重。

数据量适合验证 SFT 训练、保存和评估流程；不足以证明真实会议任务的泛化效果。本次交付只构建与校验数据，不代表已经训练或提升精度。

## 本次校验结果

已直接 `load_dataset` 加载整个 v2 仓库，并与 v1 的 96 条记录逐字段对比，原文、问题、标签和划分一致。使用 `/workspace/model-bin/MigoXV/laya-multilingual` 的实际 tokenizer，经 DecisionDataModule 从仓库加载全部记录并遍历所有 batch；每个划分都包含 choice、score、noul，所有目标都是硬标签。最长输入 145 tokens，未发生截断。结果保存在 `data-bin/tiny-sft-v2-audit/preflight.json`。

```bash
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 poetry run pytest -q
poetry run ruff check src/arietta/tasks/sft_dataset.py src/arietta/tasks/data.py src/arietta/commands/app.py src/arietta/evaluation/evaluate.py tests/test_sft_dataset.py
```

全仓库 31 项 CPU 测试通过，其中 9 项覆盖小数据集及配置；Ruff 检查通过。测试包含完整 DatasetDict 目录加载、默认配置和 split 识别、Features 与样本/字节数、split 切片加载、DataModule 单次加载及 data_files=None 映射、逐条标签保留、标注隔离、确定性重建、数据版本防覆盖、Parquet 评估，以及原生 LightningCLI 解析训练配置。没有启动官方模型的 GPU 训练。
