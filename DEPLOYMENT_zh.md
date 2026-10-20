# 部署

> [English](DEPLOYMENT.md)

## 环境

应用以 Docker 镜像形式部署。CI 流水线发布两个环境：

| 环境 | Git 分支 | 镜像标签 | 触发方式 |
|------|----------|----------|----------|
| Staging | `staging` | `absa-app:staging` | push 到 `staging` |
| Production | `main` | `absa-app:latest` | push 到 `main` |

## CI 流水线（`.github/workflows/ci.yml`）

每个到 `dev`/`staging`/`main` 的 pull request 都会执行：

1. **Lint** — `ruff check app ml tests scripts`
2. **测试** — `python -m pytest`
3. **Docker 构建** — `docker build -t absa-app:ci .`

**CI 失败则 PR 不能合并**（需在仓库开启分支保护）。

push 到 `staging`/`main` 时，流水线还会用仓库 secret（`DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN`）构建镜像并发布到 Docker Hub。

## 需要的仓库 secret

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

在 **Settings -> Secrets and variables -> Actions** 中设置。

## 环境变量

应用无必填环境变量（它直接提供已提交的 `models/model.joblib` 和 `data/demo.csv`）。可选覆盖：

- `MLFLOW_TRACKING_URI` — 仅供训练脚本使用，应用不使用。

## 本地部署

```bash
docker build -t absa-app .
docker run -p 8000:8000 absa-app
# 打开 http://127.0.0.1:8000
```

## 全量部署（全部模型）

默认镜像只包含 TF-IDF 模型（体积小、CPU 友好）。要提供 BERT/RoBERTa 方面模型和 RoBERTa 星级模型（约 1.2 GB 权重），请用 `Dockerfile.full` 构建，它会安装 CPU 版 torch + transformers。大权重**不打进镜像**，而是在运行时从宿主机 `models/` 目录挂载，镜像因此只有约 2 GB，能塞进小 VPS 的磁盘预算。

```bash
docker build -f Dockerfile.full -t absa-app:full .
docker run -d --name absa-full --restart unless-stopped -p 8000:8000 \
  -v /opt/absa/models:/app/models absa-app:full
```

宿主机的 `models/` 目录需包含被 gitignore 的大权重（`models/bert/`、`models/roberta/`、`models/rating_bert/`）以及已提交的 `model.joblib` / `rating.joblib`。

运行时全部模型加载约占 2 GB 内存，因此在内存较小（≤4 GB）的 VPS 上需要加 swap：

```bash
fallocate -l 4G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab
```

## 生产服务器

项目运行在 <http://efrei-DevOps.losproeocl.cn>，部署于一台 2 核 / 4 GB 的 Ubuntu 24.04 VPS（149.104.21.193），以全量 Docker 镜像 + nginx 反向代理的形式运行。

部署步骤（在服务器上以 root 执行）：

```bash
# 1. 加 swap（见上文）

# 2. 克隆仓库并上传被 gitignore 的模型权重
git clone --depth 1 https://github.com/LosproeOcl/WUT-DevOps-Course.git /opt/absa
#    然后把 models/bert、models/roberta、models/rating_bert scp 到 /opt/absa/models/

# 3. 构建全量镜像（模型挂载，不打进镜像）
cd /opt/absa
docker build -f Dockerfile.full -t absa-app:full .

# 4. 在空闲端口（8000）运行应用，挂载 models 目录
docker run -d --name absa-full --restart unless-stopped -p 8000:8000 \
  -v /opt/absa/models:/app/models absa-app:full
```

### 反向代理

域名由 nginx 反向代理到 `127.0.0.1:8000`。两个 Web 服务器会争抢 80 端口（系统 nginx 与 1Panel 的 OpenResty 容器）；解决方法是只保留一个。本服务器以 OpenResty（1Panel）为前端服务器，因此停用了系统 nginx，并把站点迁移到 OpenResty 的 `conf.d`：

```nginx
server {
    listen 80;
    server_name efrei-DevOps.losproeocl.cn;
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_redirect off;
    }
}
```

把这段写入 `/opt/1panel/www/conf.d/efrei-DevOps.losproeocl.cn.conf` 并重载 OpenResty（`docker exec 1Panel-openresty-<id> nginx -s reload`）。

### 冒烟测试

```bash
curl -s http://127.0.0.1:8000/health          # {"status":"ok","model_loaded":true}
curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"食物非常好吃","model":"tfidf"}'   # 返回星级 + 4 个维度
```

## 备注

- Docker 镜像是 `python:3.12-slim` 运行时镜像，只包含服务依赖（见 `requirements.txt`）；训练专用依赖（MLflow、matplotlib）在 `requirements-ml.txt` 中，不装进镜像。
- `.dockerignore` 排除了 `tests/`、`scripts/`、原始/处理后数据、`mlruns/` 以及大模型权重（`models/bert/`、`models/roberta/`、`models/rating_bert/`、`models/lstm.joblib`）。`ml/` 会被拷进镜像，因为 `app/model.py` 在运行时导入了 `ml/aspects.py`。
