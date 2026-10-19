# 方面级情感分析：训练优化

本分支（`training-optimize`）只关注情感分析模型训练流程的优化；应用、部署和完整项目说明保留在 `main` 分支。

## 优化内容

生产模型由四个二分类 TF-IDF + LogisticRegression 管道组成（每个维度判断正面/负面，训练时丢弃“中性”类）。本分支的优化集中在训练与模型选择流程：

- **逐维度超参数搜索**（`--mode search`）：对 food / service / price / ambience 分别做分层 `RandomizedSearchCV`，以 macro-F1 为评分标准。
- **搜索空间**：字符 n-gram 范围、`min_df`、`max_df`、正则化强度 `C`、`class_weight`。
- **防止退化**：同时交叉验证原有 `tuned` 配置；只有搜索结果的平均 CV macro-F1 更高时才替换。
- **可复现参数**：`--seed`、`--cv-folds`、`--search-iter`、`--n-jobs`。
- **路径可配置**：`--data-dir`、`--model-path`、`--demo-path`。
- **更完整的 MLflow 记录**：每个维度的最佳参数、CV macro-F1、训练/评估样本数、留出集指标和每类 F1。

## 运行

```bash
dvc pull data
python -m ml.train --data-version v2 --mode search --search-iter 8 --cv-folds 3
```

与固定模式对比：

```bash
python -m ml.train --data-version v2 --mode baseline
python -m ml.train --data-version v2 --mode tuned
```

选中的管道保存到 `models/model.joblib`；每个维度还会通过 MLflow 记录并注册。可用 `mlflow ui` 或 DagsHub 对比实验。

## 优化指标

每个维度的 macro-F1（正面/负面）。准确率和每类 F1 也会记录用于分析，但由于类别不均衡，模型选择以 macro-F1 为准。

## 验证

- `ruff check app ml tests scripts` 通过。
- `pytest`：39 项测试全部通过，包含新增的训练/调参辅助函数测试。
- 已用合成数据完成 MLflow 端到端训练冒烟测试。

## 当前限制

真实数据由 DVC 托管在 `efreiwut-projdvc`。当前环境访问该远端返回 `AuthorizationPermissionMismatch`（403），因此已提交的 `models/model.joblib` 仍是之前的 `tuned` 模型。取得权限后运行上面的搜索命令，并仅在各维度留出集 macro-F1 均提升时提交新模型。

详细说明：`ml/README_experiments.md`、`ml/registry/current_model.md`。
