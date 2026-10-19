# 训练优化（实验分支）

本分支只记录训练优化实验；完整项目文档见 `main`。

- `--mode search`：逐维度 TF-IDF + LR 超参数搜索，使用分层 macro-F1 交叉验证。
- 默认搜索空间仅保留字符 `ngram_range=(1,2)`；`(1,3)` 会产生过多特征，在 CPU 上过慢。
- 始终交叉验证 tuned 配置；只有 CV macro-F1 更高时才会替换。
- 改动：`ml/train.py`、`tests/unit/test_train.py`。
- 运行：`python -m ml.train --data-version v2 --mode search --search-iter 8 --cv-folds 3`

| 指标（官方 ASAP 划分） | Tuned | Search | 变化 |
|---|---:|---:|---:|
| 平均 macro-F1 | 0.7672 | 0.7824 | +0.0152（约 +2.0%） |

四个维度均无回退，但提升幅度有限，未达到合并门槛。该实验暂不合并，生产模型仍为 `tuned`。

验证：`ruff` 通过；`pytest` 39 项通过。
