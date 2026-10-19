# 方面级情感分析：训练优化

本分支（`training-optimize`）只关注情感分析模型训练流程的优化；应用、部署和完整项目说明保留在 `main` 分支。

## 优化内容

生产模型由四个二分类 TF-IDF + LogisticRegression 管道组成（每个维度判断正面/负面，训练时丢弃“中性”类）。本分支优化训练与模型选择流程：

- `--mode search`：对每个维度分别做分层 `RandomizedSearchCV`，以 macro-F1 为评分标准。
- 搜索空间：字符 n-gram 范围、`min_df`、`max_df`、正则化强度 `C`、`class_weight`。
- 防止退化：同时交叉验证原有 `tuned` 配置；只有搜索结果的平均 CV macro-F1 更高时才替换。
- 可复现参数：`--seed`、`--cv-folds`、`--search-iter`、`--n-jobs`。
- 路径可配置：`--data-dir`、`--model-path`、`--demo-path`。
- MLflow 记录：每个维度的最佳参数、CV macro-F1、训练/评估样本数、留出集指标和每类 F1。

```bash
dvc pull data
python -m ml.train --data-version v2 --mode search --search-iter 8 --cv-folds 3
```

与固定模式对比：

```bash
python -m ml.train --data-version v2 --mode baseline
python -m ml.train --data-version v2 --mode tuned
```

## 性能对比

v2 测试集上的逐维度 macro-F1（正面/负面）。本次验证使用官方 ASAP train/dev/test 划分（公开镜像：`nd-ball/DSPN`），通过 `ml/data_pipeline.py` 生成 v2；三种模式使用同一划分和随机种子。tuned 结果复现了项目公布的 0.738 / 0.784 / 0.786 / 0.760，说明数据划分可比。

| 维度 | Baseline（优化前） | Tuned（上一版） | Search（本分支） | 相对 tuned |
|---|---:|---:|---:|---:|
| food | 0.6758 | 0.7384 | 0.7654 | +0.0270 |
| service | 0.7413 | 0.7840 | 0.8005 | +0.0165 |
| price | 0.7148 | 0.7864 | 0.8141 | +0.0277 |
| ambience | 0.6851 | 0.7598 | 0.7852 | +0.0254 |
| **平均** | **0.7043** | **0.7672** | **0.7913** | **+0.0242** |

search 在四个维度上均有提升：平均 +0.0242（相对 tuned 约 +3.1%，相对 baseline 约 +12.4%）。验证参数：`--search-iter 4 --cv-folds 2 --n-jobs 1`（默认命令为 8/3）。

搜索模型产物在临时目录生成；本分支保留训练代码和实测结果，已提交的 `models/model.joblib` 仍是 `tuned` 模型。要刷新模型，运行 search 命令并提交生成的 `models/model.joblib`（数据由 DVC 托管；本次因 Azure 远端 403，使用公开 ASAP 镜像）。

验证：`ruff check app ml tests scripts` 通过；`pytest` 39 项测试通过；已在 ASAP 划分上完整跑通 baseline / tuned / search，并记录到 MLflow。
