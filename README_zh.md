# 方面级情感分析

[![CI](https://img.shields.io/github/actions/workflow/status/LosproeOcl/WUT-DevOps-Course/ci.yml?branch=main&label=CI&style=for-the-badge&logo=github)](https://github.com/LosproeOcl/WUT-DevOps-Course/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/LosproeOcl/WUT-DevOps-Course?style=for-the-badge&logo=github)](https://github.com/LosproeOcl/WUT-DevOps-Course/releases)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-python%3A3.12--slim-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://hub.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.x-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![DagsHub](https://img.shields.io/badge/DagsHub-实验%20%26%20注册表-2E77BC?style=for-the-badge&logo=dagshub&logoColor=white)](https://dagshub.com/losproeocl/WUT-DevOps-Course)
[![DVC](https://img.shields.io/badge/DVC-3.x-13ADC7?style=for-the-badge&logo=dvc&logoColor=white)](https://dvc.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE)
[![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/LosproeOcl/WUT-DevOps-Course/badges/coverage.json&style=for-the-badge&logo=pytest&logoColor=white)](https://github.com/LosproeOcl/WUT-DevOps-Course/actions/workflows/ci.yml)

> [English README](README.md)

一个 Web 应用：输入一条中文餐厅评论，对 **口味 / 服务 / 价格 / 环境** 四个维度分别判断情感（正面 / 负面 / 中性）。

这是 DevOps & MLOps 课程的期末项目，最后做成了一个端到端的 MLOps 小流水线：DVC 管数据、MLflow 管实验、Docker + GitHub Actions 做 CI/CD，FastAPI 应用带中英双语界面、三个可选模型、CSV 批量导入导出、以及保存结果的增删改查。

**相关链接：**
- **在线演示** — http://efrei-DevOps.losproeocl.cn
- [DagsHub MLflow 实验](https://dagshub.com/losproeocl/WUT-DevOps-Course.mlflow) — 运行、指标与参数
- [DagsHub 模型注册表](https://dagshub.com/losproeocl/WUT-DevOps-Course.mlflow/#/models) — 已注册模型与 stage
- [DagsHub 仓库](https://dagshub.com/losproeocl/WUT-DevOps-Course) — 数据 + 代码镜像（DVC）
- [GitHub releases](https://github.com/LosproeOcl/WUT-DevOps-Course/releases)
- [部署指南](DEPLOYMENT_zh.md)

## 应用功能

- 两个页面：`/`（分析）和 `/history`（历史记录）。
- 输入一条评论 → 四个维度各给标签 + 置信度 + 概率条，以及整体星级评分（1-5）和逐词解释（SHAP，TF-IDF 模型）。
- 三个模型可选：TF-IDF + LR、BERT、RoBERTa。
- 上传 CSV → 逐条批量分析、结果入库、下载带原列 + 情感/置信度/概率列的 CSV。
- 保存的结果存 SQLite，在 `/history` 页查看、编辑、单条/批量删除、导出 CSV。
- 界面支持中英文切换。

## 目录结构

```
app/                 FastAPI 应用（API + 提供构建后的 SPA）
  main.py            路由 + 静态 SPA 服务
  model.py           模型加载与推理（tfidf / bert / roberta + 星级）
  database.py        保存记录的 SQLite 层
  schemas.py         Pydantic 模型
frontend/            Vue 3 + Vite 单页应用（前后端分离，仿 Typecho）
  src/               组件、视图、i18n、API 客户端
ml/                  训练与数据代码
  data_pipeline.py   ASAP CSV -> (text, aspect, polarity) 行，v1/v2
  train.py           TF-IDF + LR，日志进 MLflow
  train_lstm.py      BiLSTM（PyTorch）
  train_bert.py      BERT/RoBERTa 微调
  aspects.py         维度映射 + 分词
data/                raw/ processed/（DVC）、demo.csv
models/              训练权重（tfidf 入库；bert/roberta/lstm 已忽略）
tests/               pytest 单元 + 接口测试
.github/workflows/ci.yml
Dockerfile           多阶段（前端构建 + 后端）
Dockerfile.full      全量部署（CPU torch + transformers + 全部模型）
```

## 分支

本仓库以独立分支承载多个课程项目，本项目在 `main`。

| 分支 | 内容 |
|------|------|
| `main` | 本项目 — 方面级情感分析 |
| `course-log` | 课程材料（笔记、练习指南、课件）— 原 `main` |
| `cars-crud` | 练习项目 — 汽车增删改查 |
| `git-playground` | 练习项目 — Git |

## 开发历程

选题受硬件约束：本机仅具备 CPU，初期不计划租赁算力；单维度情感分类过于基础，故定为多维度的方面级情感分析，以在作业期限内保证交付完整性。餐厅评论场景适配交互演示与测试集预留的需要。

开发前先修复本机环境。五个 Python 安装共享同一 user site-packages，一次中断的 pip 升级遗留 `~umpy`、`~ip`、`~crypt` 目录，`import sklearn` 报 `numpy.core.multiarray failed to import`。统一至 `C:\Python312`、清理残留并重装 numpy，此后全程使用项目独立 `.venv`。

数据更换两次。初始为英文 SemEval-2014；开发中明确"四维度输出 + 中英双语"的定位，标注由 aspect-term 迁移至 aspect-category，并引入中文语料。MAMS 托管于 Google Drive，获取不便；改采美团点评 ASAP——中文、GitHub 托管、约 46,850 条、18 个 category 归并为四个维度，规模较 SemEval 高一个量级。中文采用字符级切分，规避分词词典依赖。

模型迭代最多。基线为 TF-IDF + LR，置信度约 0.45，根因是 neutral 类标签分布混杂；收敛为二分类、对低置信度预测标注 neutral 后，macro-F1 与置信度同步回升。深度模型依赖 GPU，本机不具备；CPU 训练的 BiLSTM 因样本量不足未达预期，最终以 5 元租赁 AutoDL 的 RTX 3080 Ti 完成 BERT 与 RoBERTa 微调。二者在多维度混合句子上优于基线，但未全面领先，且 CPU 推理开销高，三个模型均保留、运行时切换，TF-IDF 为默认。

DVC 远端迁移一次。初期无 Azure 账号，以本地目录占位；取得学生账号后创建 Storage Account 与容器，远端切换至 Azure Blob。首次 push 大文件遭遇 socket 超时，系国内至 Azure 的间歇性延迟，重试完成。

实验跟踪遵循课程 MLflow 指南，先本地、后 DagsHub；MLflow 3.x 的 `CreateLoggedModel` 端点未被 DagsHub 2.x 服务端实现，客户端锁定 `mlflow<3`。

应用由单一 `/predict` 端点扩展至四维输出、双语界面、模型切换、CSV 批量导入导出、记录 CRUD，最终拆分分析页与历史页。期间一处隐蔽故障：Python 三引号字符串内 `'\n'` 在生成 JS 中成为字面换行，脚本整体失效，以 `String.fromCharCode` 规避。

CI/CD 采用 GitHub Actions：PR 触发 ruff、pytest、Docker 构建，push 至 staging/main 发布镜像。镜像基于 `python:3.12-slim`，仅含运行时依赖，torch/transformers 懒加载。

星级评分任务后置补充。ASAP 自带 `star` 列，故另设一条预测 1-5 星的头，与四个维度并行输出。首个 TF-IDF + LR 星级模型精确率 55.9%；首版 BERT 在 128 长度下反而回退至 54.2%，统计语料发现评论平均 320 字，即 99.8% 的输入被截断。改为 384 长度并加线性 warmup 后 BERT 提升至 58.1%，配合 label smoothing 的 RoBERTa（chinese-wwm-ext）达 59.2% / MAE 0.446，现为默认星级模型。随后在 TF-IDF 模型上引入 SHAP（`LinearExplainer`），使每个维度能以影响最大的字词自证判断。

前端重写两次。初为内嵌于 FastAPI 模块的服务端渲染 HTML，界面状态增多后迁出为独立的 Vue 3 + Vite 单页应用。解耦从 `app/main.py` 移除了约 430 行 HTML/CSS/JS，仅剩纯 REST API，生产环境直接托管构建产物。视觉沿用 Typecho 默认主题——衬线正文、细边框、低饱和蓝主色；输入框增加聚焦光环、清空按钮与字数统计，结果卡片重设计为带标签的概率条形图与关键词 chips。

## 环境配置

Python 3.12，项目独立 venv：

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements-ml.txt -r requirements-dev.txt
```

| 文件 | 用途 |
|------|------|
| `requirements.txt` | 运行时（fastapi、uvicorn、joblib、scikit-learn、numpy、pandas、python-multipart） |
| `requirements-ml.txt` | 训练 + 深度模型推理（mlflow<3、matplotlib、dagshub、torch、transformers） |
| `requirements-dev.txt` | pytest、httpx、ruff |

环境变量（token 不进 git）：

| 变量 | 含义 |
|------|------|
| `DAGSHUB_REPO_OWNER` / `DAGSHUB_REPO_NAME` | DagsHub 仓库，如 `losproeocl` / `WUT-DevOps-Course` |
| `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD` | DagsHub 用户名 / access token |
| `REVIEWS_DB_PATH` | 可选；保存记录的 SQLite 路径 |
| `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN` | GitHub Actions 发布镜像用的 secret |

## 数据

ASAP（Meituan-Dianping/asap），约 46,850 条中文评论，18 个方面类别归并成四个：

| 维度 | ASAP 类别 |
|------|-----------|
| 口味 | Food#Portion、Food#Taste、Food#Appearance、Food#Recommend |
| 服务 | Service#Queue、Service#Hospitality、Service#Parking、Service#Timely |
| 价格 | Price#Level、Price#Cost_effective、Price#Discount |
| 环境 | Ambience#Decoration、Ambience#Noise、Ambience#Space、Ambience#Sanitary |

情感取值 `1`/`0`/`-1`（`-2` = 未提及，丢弃）。DVC 跟踪 `data/raw` 和 `data/processed`，远端是 Azure Blob：

```bash
dvc remote add -d myazure azure://efreiwut-projdvc
dvc remote modify myazure account_name efreiwut2026dvc
dvc remote modify --local myazure connection_string "<密钥>"   # .dvc/config.local，已 gitignore
dvc push
```

两个版本：`v1`（原始）和 `v2`（字符分词 + 去重）。

## 模型与训练

每个维度做二分类（正面 vs 负面）；训练时丢掉"中性"类，低置信度的预测标为中性。

| 模型 | 训练位置 | macro-F1（口味/服务/价格/环境） |
|------|----------|------------------------------|
| TF-IDF + LR（tuned） | CPU，`ml/train.py` | 0.738 / 0.784 / 0.786 / 0.760 |
| TF-IDF + LR（search，逐维度调参） | CPU，`ml/train.py --mode search` | 重跑后刷新 |
| BiLSTM | CPU，`ml/train_lstm.py` | 这份数据上偏弱 |
| BERT（bert-base-chinese，3 epoch） | AutoDL GPU，`ml/train_bert.py` | 0.758 / 0.806 / 0.758 / 0.824 |
| RoBERTa（chinese-wwm-ext，3 epoch） | AutoDL GPU | 0.765 / 0.803 / 0.763 / 0.789 |

另有独立的**星级评分模型**对 1-5 星做分类（用 ASAP 的 `star` 列）：

| 星级模型 | 位置 | 准确率 | MAE | ±1 星内 |
|----------|------|--------|-----|---------|
| TF-IDF + LR | CPU，`ml/train_rating.py` | 0.559 | 0.512 | — |
| BERT（bert-base-chinese，4 epoch） | AutoDL GPU | 0.581 | 0.466 | 0.957 |
| RoBERTa（chinese-wwm-ext，5 epoch） | AutoDL GPU，`ml/train_rating_bert.py` | **0.592** | **0.446** | 0.964 |

```bash
python -m ml.data_pipeline              # 生成 v1/v2 CSV
python -m ml.train --data-version v2 --mode tuned   # TF-IDF + LR（固定 tuned 配置）
python -m ml.train --data-version v2 --mode search --search-iter 8 --cv-folds 3  # 逐维度交叉验证调参
python -m ml.train_lstm --data-version v2           # BiLSTM
python -m ml.train_bert --data-dir data --epochs 3  # BERT/RoBERTa（建议 GPU）
python -m ml.train_rating                           # 星级评分
python -m ml.train_rating_bert --model-path ...     # 星级评分（GPU，RoBERTa）
python -m ml.register_rating_roberta                # 注册星级 RoBERTa 到 MLflow
```

DagsHub 上注册：`absa_restaurant`（统一 TF-IDF+LR，Production）、`absa_restaurant_rating`（TF-IDF 星级，Production）、`absa_restaurant_rating_roberta`（RoBERTa 星级，Production）、`absa_restaurant_bert`（Staging）、`absa_restaurant_roberta` 与 `absa_restaurant_lstm`（Archived）。stage 迁移在 `ml/stage_models.py`；基于注册表的部署由 `ml/serve_model.py` 演示（`mlflow.pyfunc.load_model("models:/absa_restaurant/Production")`）。

### 置信度与「中性」类

每个维度在训练时为二分类（正面 vs 负面）；模型以胜出类别的概率作为**置信度**输出。置信度低于 `UNCERTAIN_THRESHOLD = 0.55` 的预测会被降级为**中性**，避免对摇摆不定的正负判断强行下结论。该阈值在界面中通过每个维度旁的概率条形图直观呈现。

星级评分（1-5）是独立的五分类任务，其「置信度」为预测星级的 softmax 概率。RoBERTa 星级模型在 ±1 星范围内与真实评分一致的比例达 96.4%，对 1-5 的有序刻度而言，这是比精确命中更有意义的评估指标。

## 课程要求对照 — 配置指示

`project-instructions.pdf` 的每项要求，附复现步骤。

### 应用

**训练/微调模型。** TF-IDF + LR 由 `ml/train.py` 训练；BERT/RoBERTa 由 `ml/train_bert.py` 微调（建议 GPU）。

```bash
python -m ml.data_pipeline
python -m ml.train --data-version v2 --mode search --search-iter 8 --cv-folds 3
python -m ml.train_bert --data-dir data --epochs 3
```

**数据集多版本。** `data/processed/{train,val,test}_v1.csv`（原始）与 `_v2.csv`（字符分词 + 去重），用 `python -m ml.data_pipeline` 重新生成。

**测试。** `tests/` 下 39 个 pytest 测试（单元 + 接口），覆盖空/空白输入、未知模型、CSV 导入导出、CRUD、批量删除，以及逐维度训练/调参辅助函数。运行 `python -m pytest`。

### DevOps

**版本控制。** 分支模型 `main` ← `staging` ← `dev` ← `feature/*`；交付在 `main`（课程材料在 `course-log`）。

```bash
git switch -c feature/xxx
git push -u origin feature/xxx   # 再依次向 dev、staging、main 发起 PR
```

**Docker。** `Dockerfile`（`python:3.12-slim`）配 `.dockerignore`，排除 `tests/`、`ml/`、`scripts/`、`data/raw`、`data/processed`、`mlruns/`、`.dvc/` 与大模型权重。

```bash
docker build -t absa-app .
docker run -p 8000:8000 absa-app
```

**CI/CD。** `.github/workflows/ci.yml` 在 PR/push 到受保护分支时运行 `ruff check app ml tests scripts`、`python -m pytest`，再 `docker build -t absa-app:ci .`。在 GitHub → Settings → Branches 开启分支保护，使 CI 失败时 PR 无法合并；在 Settings → Secrets → Actions 配置 `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN` 用于发布镜像。

### MLOps

**DVC。**

```bash
dvc init
dvc remote add -d myazure azure://efreiwut-projdvc
dvc remote modify myazure account_name efreiwut2026dvc
dvc remote modify --local myazure connection_string "<密钥>"   # .dvc/config.local，已 gitignore
dvc add data/raw data/processed
dvc push
```

`ml/README_data.md` 记录两个版本及 v1 → v2 的变化。

**MLflow。** 设置 `DAGSHUB_REPO_OWNER` / `DAGSHUB_REPO_NAME` 与 `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD` 后运行训练——`ml/train.py` 记录 git SHA、数据校验和、超参数、指标与模型并注册。`ml/README_experiments.md` 记录 baseline 与 tuned 的对比；`mlflow ui` 查看本地历史。

**模型部署。** 用 `ml/stage_models.py` 做 stage 迁移（Staging/Production/Archived），再从注册表加载：

```bash
python -m ml.stage_models
python -m ml.serve_model --model models:/absa_restaurant/Production --text "食物很好吃"
# 或用 CLI：
mlflow models serve -m models:/absa_restaurant/Production -p 5003
```

### 目录结构

```
app/  ml/  data/  models/  scripts/  tests/
requirements.txt  requirements-ml.txt  requirements-dev.txt
Dockerfile  .dockerignore  .github/workflows/ci.yml
README.md  README_zh.md  DEPLOYMENT.md
```

## 踩坑速查

- `numpy.core.multiarray failed to import` → 多个 Python 共用 user site + `~` 残留；统一到 C:\Python312 并删掉。
- 置信度 ~0.45 → "中性"是大杂烩类；改二分类。
- 数据量限制模型 → 换 ASAP（大十倍）。
- `CreateLoggedModel` → `INVALID_PARAMETER_VALUE` → 锁 `mlflow<3`（DagsHub 服务器是 2.x）。
- AutoDL 上 `process() takes no keyword arguments` → 用 `wget` 从 hf-mirror 下模型，从本地路径加载。
- 中文下 TF-IDF 特征为空 → `token_pattern=r"\S+"`（默认会过滤单个 CJK 字符）。
- JS 按钮失灵 → 三引号 HTML 里的 `'\n'` 变成真实换行；用 `String.fromCharCode(10)`。
- 400 MB 权重 → gitignore（LSTM 走 DVC；TF-IDF ~30 MB 直接提交）。
- 5 epoch 过拟合 → 3 个最合适。
- Azure 推送超时 → 间歇性，重试。

## 运行

前端（Vue 3 + Vite）与后端（FastAPI）解耦。开发时分别启动；生产时构建 SPA 后由 FastAPI 托管。

```bash
# 1. 后端（API，:8000）
python -m uvicorn app.main:app --reload

# 2. 前端开发服务器（Vite，:5173，代理 /api -> :8000）
cd frontend && npm install && npm run dev
# http://127.0.0.1:5173

# 生产：构建 SPA，再由 FastAPI 在 :8000 提供服务
cd frontend && npm run build
python -m uvicorn app.main:app        # http://127.0.0.1:8000

docker build -t absa-app .            # 多阶段：构建前端 + 后端
docker run -p 8000:8000 absa-app

# 全量部署（全部模型：BERT/RoBERTa + RoBERTa 星级，镜像约 4 GB）
docker build -f Dockerfile.full -t absa-app:full .
docker run -d --name absa-full --restart unless-stopped -p 8000:8000 absa-app:full
```

生产服务器配置（swap、反向代理、冒烟测试）见 [DEPLOYMENT.md](DEPLOYMENT.md)。

测试：`python -m pytest`；lint：`ruff check app ml tests scripts`。
