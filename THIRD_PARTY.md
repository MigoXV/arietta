# 第三方来源

`src/arietta/models/reference.py` 来源于 `/workspace/opusi/laya/src/laya/reference.py`，上游为 convaiinnovations/laya 的 `rl_common.py`（Apache-2.0）。上游仓库：https://github.com/NandhaKishorM/laya ，资产快照 `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`。许可证保存在 LICENSES/Apache-2.0.txt。

保留原始编码、决策头、动作头与概率数学；沿用推理项目的动作头输入 dtype 修正。Arietta 自己负责数据验证、模型注册、混合精度训练、部署验证与导出，不执行模型仓库中的远程 Python 代码。

训练工程基础迁移自 `/workspace/opus/meetnote-decision-train` 的 `9d56be6`，旧仓库的会议专用对照和拒答评估仍留在原处。预训练资产以 `/workspace/model-bin/MigoXV/laya-multilingual` 的新版原生格式为准；权重不纳入 Git。
