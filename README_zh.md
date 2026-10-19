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

v2 测试集上的逐维度 macro-F1（正面/负面）：

| 维度 | Baseline（优化前） | Tuned（上一版） | Search（本分支） |
|---|---:|---:|---:|
| food | 0.4219 | 0.6609 | 待测量* |
| service | 0.5577 | 0.7039 | 待测量* |
| price | 0.5333 | 0.8518 | 待测量* |
| ambience | 0.6065 | 0.7164 | 待测量* |

\* `search` 列需要 DVC 托管的真实数据。当前 Azure 账号访问 `efreiwut-projdvc` 返回 `AuthorizationPermissionMismatch`（403），因此已提交的 `models/model.joblib` 仍是 `tuned` 模型。取得权限后运行搜索命令并填入留出集结果；仅在各维度均提升时提交新模型。

Baseline/Tuned 数据来自 `ml/README_experiments.md`。根 README 后续一次运行给出的 tuned 结果为 0.738 / 0.784 / 0.786 / 0.760；上表使用同一组实验日志中的 baseline/tuned，保证优化前后可比。

验证：`ruff check app ml tests scripts` 通过；`pytest` 39 项测试通过；已用合成数据完成 MLflow 端到端训练冒烟测试。
