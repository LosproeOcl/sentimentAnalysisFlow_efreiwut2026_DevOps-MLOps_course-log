# Deployment

> [中文版](DEPLOYMENT_zh.md)

## Environments

The app is deployed as a Docker image. Two environments are published from the
CI pipeline:

| Environment | Git branch | Image tag | Trigger |
|-------------|------------|-----------|---------|
| Staging | `staging` | `absa-app:staging` | push to `staging` |
| Production | `main` | `absa-app:latest` | push to `main` |

## CI pipeline (`.github/workflows/ci.yml`)

Every pull request to `dev`/`staging`/`main` runs:

1. **Lint** — `ruff check app ml tests scripts`
2. **Tests** — `python -m pytest`
3. **Docker build** — `docker build -t absa-app:ci .`

A PR **cannot be merged if CI fails** (enable branch protection on the repo).

On push to `staging`/`main`, the pipeline additionally builds and publishes the
Docker image to Docker Hub using repository secrets
(`DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN`).

## Required repository secrets

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

Set these under **Settings -> Secrets and variables -> Actions**.

## Environment variables

The app reads no mandatory env vars (it serves the committed `models/model.joblib`
and `data/demo.csv`). Optional overrides:

- `MLFLOW_TRACKING_URI` — only used by the training script, not the app.

## Local deployment

```bash
docker build -t absa-app .
docker run -p 8000:8000 absa-app
# open http://127.0.0.1:8000
```

## Full deployment (all models)

The default image ships only the TF-IDF models (small, CPU-friendly). To serve
the BERT/RoBERTa aspect models and the RoBERTa rating model (~1.2 GB of weights),
build `Dockerfile.full`, which installs CPU-only torch + transformers. The large
weights are **not baked into the image** — they are mounted at runtime from the
host `models/` directory, keeping the image ~2 GB so it builds within a small
VPS's disk budget.

```bash
docker build -f Dockerfile.full -t absa-app:full .
docker run -d --name absa-full --restart unless-stopped -p 8000:8000 \
  -v /opt/absa/models:/app/models absa-app:full
```

The `models/` directory on the host must contain the gitignored large weights
(`models/bert/`, `models/roberta/`, `models/rating_bert/`) plus the committed
`model.joblib` / `rating.joblib`.

At runtime it loads ~2 GB RAM when all models are in use, so add swap on small
VPSes (4 GB RAM or less):

```bash
fallocate -l 4G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab
```

## Production server

The project runs at <http://efrei-DevOps.losproeocl.cn> on a 2-core / 4 GB Ubuntu
24.04 VPS (149.104.21.193), deployed as the full Docker image behind nginx.

Deployment steps (run as root on the server):

```bash
# 1. swap (see above)

# 2. clone + upload the gitignored model weights
git clone --depth 1 https://github.com/LosproeOcl/WUT-DevOps-Course.git /opt/absa
#    then scp models/bert, models/roberta, models/rating_bert into /opt/absa/models/

# 3. build the full image (models are mounted, not baked in)
cd /opt/absa
docker build -f Dockerfile.full -t absa-app:full .

# 4. run the app on a free port (8000), mounting the models dir
docker run -d --name absa-full --restart unless-stopped -p 8000:8000 \
  -v /opt/absa/models:/app/models absa-app:full
```

### Reverse proxy

The domain is proxied to `127.0.0.1:8000` by nginx. Two web servers can conflict
on port 80 (the system nginx and 1Panel's OpenResty container); the fix is to
consolidate on one. On this server OpenResty (1Panel) is the front server, so the
system nginx was disabled and the site migrated into OpenResty's `conf.d`:

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

Write this to `/opt/1panel/www/conf.d/efrei-DevOps.losproeocl.cn.conf` and reload
OpenResty (`docker exec 1Panel-openresty-<id> nginx -s reload`).

### Smoke test

```bash
curl -s http://127.0.0.1:8000/health          # {"status":"ok","model_loaded":true}
curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"食物非常好吃","model":"tfidf"}'   # returns rating + 4 aspects
```

## Notes

- The Docker image is a `python:3.12-slim` runtime image containing only the
  serving dependencies (see `requirements.txt`); training-only deps (MLflow,
  matplotlib) live in `requirements-ml.txt` and are not installed in the image.
- `.dockerignore` excludes `tests/`, `scripts/`, raw/processed data, `mlruns/`,
  and the large model weights (`models/bert/`, `models/roberta/`,
  `models/rating_bert/`, `models/lstm.joblib`). `ml/` is copied into the image
  because `app/model.py` imports `ml/aspects.py` at runtime.
