# DevOps & MLOps 课程记录

- 机器：Windows 11（Build 10.0.26200）
- 本地用户：LAPTOP-IREQL4UJ\86181
- 记录分支：`course-log`

## 一、环境

| 组件 | 版本 | 状态 |
|---|---|---|
| Python | 3.12.10（`.venv`） | ✅ |
| Git | 2.56.0.windows.2 | ✅ |
| Docker Desktop | 4.95.0（Engine 29.8.2，WSL2 后端） | ✅ |
| JupyterLab | 4.6.4 | ✅ |
| MLflow | 3.17.0 | ✅ |
| DVC | 3.67.1（含 Azure 支持） | ✅ |
| Flask | 3.1.3 | ✅ |
| FastAPI / uvicorn | 0.136.3 / 0.48.0 | ✅ |
| scikit-learn | 1.9.1 | ✅ |
| GitHub | LosproeOcl（gh CLI 2.102.0，本地登录 qywkq） | ✅ |
| DagsHub | losproeocl | ✅ |
| Azure CLI | 2.91.0 | ✅ |
| Azure 存储（DVC 远端） | `wut2026` / 容器 `dvc` | 待授权 |

> DVC 数据存放在 Azure Blob。当前 Azure 账号可以登录，但还没有 `wut2026` 的 Blob 读取权限；
> 完成授权或配置 SAS token 后即可执行 `dvc pull`。

## 二、快速开始

```powershell
# 1. 创建并激活虚拟环境（Python 3.12）
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. 安装固定版本依赖
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3. 环境自检
powershell -ExecutionPolicy Bypass -File Config\scripts\check-env.ps1
```

DVC 数据（任选一种凭据方式）：

```powershell
# 方式 A：Azure 账号已具备存储读取权限
az login --use-device-code
.\.venv\Scripts\dvc.exe pull

# 方式 B：使用课程提供的 SAS token（只保存在本地，勿提交）
.\.venv\Scripts\dvc.exe remote modify --local myazure sas_token "<SAS>"
.\.venv\Scripts\dvc.exe pull
```

WSL2 / Docker 首次配置（需要管理员权限，完成后重启）：

```powershell
powershell -ExecutionPolicy Bypass -File Config\scripts\install-wsl.ps1
```

## 三、目录结构

```
.
├── LearningDoc/         # 学习文档（docx / pdf / 构建脚本）
├── Config/              # 环境配置（logs/、scripts/）
├── Note/                # 课堂笔记
├── PracticeIntro/       # 练习指南（PDF，含中文版）
├── PowerPoint/          # 课件
├── Quiz_recap/          # 测验回顾
└── data/                # DVC 数据指针（实际数据在 Azure Blob）
```

> `cars-crud/`、`git-playground/`、`finalProject/` 位于独立分支；当前 `course-log` 分支不包含这些目录。

## 四、期末项目

**方面级情感分析**（`finalProject/`，分支 `main`）

- 输入中文餐厅评论，输出口味 / 服务 / 价格 / 环境四个维度的正负情感
- FastAPI 应用：分析 + 历史两个页面，中英双语界面
- 三个可选模型：TF-IDF + LR（默认）、BERT、RoBERTa
- CSV 批量导入导出、保存记录 CRUD（SQLite）
- DVC 数据版本管理（Azure Blob 远端）、MLflow 实验跟踪与模型注册（DagsHub）
- Docker + GitHub Actions CI/CD
- Release：<https://github.com/LosproeOcl/WUT-DevOps-Course/releases/tag/v1.0.0>
- 详见 `finalProject/README.md`、`finalProject/README_zh.md`

## 五、版本清单

| 组件 | 版本 |
|---|---|
| Python | 3.12.10 |
| Git | 2.56.0.windows.2 |
| Docker CLI / Engine | 29.8.2 |
| Docker Desktop | 4.95.0 |
| Docker Compose | v5.6.0 |
| WSL | 3.0.1.0（内核 6.18.40.1-1） |
| JupyterLab | 4.6.4 |
| MLflow | 3.17.0 |
| DVC | 3.67.1 |
| Flask | 3.1.3 |
| FastAPI | 0.136.3 |
| uvicorn | 0.48.0 |
| scikit-learn | 1.9.1 |
| GitHub CLI (gh) | 2.102.0 |
| Azure CLI | 2.91.0 |

## 六、日志与脚本

- 日志目录 `Config/logs/`：pip / Docker / WSL 安装与自检日志
- `Config/scripts/check-env.ps1`：环境自检，优先使用仓库内的 `.venv`
- `Config/scripts/install-wsl.ps1`：启用 WSL2 / 虚拟机平台并以管理员身份安装，日志自动写入 `Config/logs/`
- `requirements.txt`：Python 依赖固定版本，便于复现实验环境
