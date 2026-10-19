# Aspect-Based Sentiment Analysis

[![CI](https://img.shields.io/github/actions/workflow/status/LosproeOcl/WUT-DevOps-Course/ci.yml?branch=main&label=CI&style=for-the-badge&logo=github)](https://github.com/LosproeOcl/WUT-DevOps-Course/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/LosproeOcl/WUT-DevOps-Course?style=for-the-badge&logo=github)](https://github.com/LosproeOcl/WUT-DevOps-Course/releases)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-python%3A3.12--slim-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://hub.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.x-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![DagsHub](https://img.shields.io/badge/DagsHub-experiments%20%26%20registry-2E77BC?style=for-the-badge&logo=dagshub&logoColor=white)](https://dagshub.com/losproeocl/WUT-DevOps-Course)
[![DVC](https://img.shields.io/badge/DVC-3.x-13ADC7?style=for-the-badge&logo=dvc&logoColor=white)](https://dvc.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE)
[![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/LosproeOcl/WUT-DevOps-Course/badges/coverage.json&style=for-the-badge&logo=pytest&logoColor=white)](https://github.com/LosproeOcl/WUT-DevOps-Course/actions/workflows/ci.yml)

> [中文版 README](README_zh.md)

A web app that takes a Chinese restaurant review and classifies its sentiment across four aspects — **taste / service / price / ambience** — each marked positive, negative, or neutral.

This is the final project for a DevOps & MLOps course. It ended up as a small end-to-end MLOps pipeline: DVC for data, MLflow for experiments, Docker + GitHub Actions for CI/CD, and a FastAPI app with a bilingual UI, three selectable models, CSV batch import/export, and CRUD for saved results.

**Related links:**
- **Live demo** — http://efrei-DevOps.losproeocl.cn
- [DagsHub MLflow experiments](https://dagshub.com/losproeocl/WUT-DevOps-Course.mlflow) — runs, metrics, and parameters
- [DagsHub model registry](https://dagshub.com/losproeocl/WUT-DevOps-Course.mlflow/#/models) — registered models and stages
- [DagsHub repository](https://dagshub.com/losproeocl/WUT-DevOps-Course) — data + code mirror (DVC)
- [GitHub releases](https://github.com/LosproeOcl/WUT-DevOps-Course/releases)
- [Deployment guide](DEPLOYMENT.md)

## What the app does

- Two pages: `/` (analyze) and `/history` (saved records).
- One review in → four aspects with label + confidence + probability bars, an overall star rating (1-5), and token-level explanations (SHAP) for the TF-IDF model.
- Three models to choose from: TF-IDF + LR, BERT, RoBERTa.
- CSV upload → batch-analyze each row, save results, download a CSV with original columns plus sentiment/confidence/probability columns.
- Saved results live in SQLite, managed on `/history`: view, edit, single/batch delete, export CSV.
- UI switches between Chinese and English.

## Repo layout

```
app/                 FastAPI app (API + serves the built SPA)
  main.py            routes + static SPA serving
  model.py           model loading + inference (tfidf / bert / roberta + rating)
  database.py        SQLite layer for saved reviews
  schemas.py         Pydantic models
frontend/            Vue 3 + Vite SPA (frontend-backend separation, Typecho-inspired)
  src/               components, views, i18n, api client
ml/                  training + data code
  data_pipeline.py   ASAP CSV -> (text, aspect, polarity) rows, v1/v2
  train.py           TF-IDF + LR, logs to MLflow
  train_lstm.py      BiLSTM (PyTorch)
  train_bert.py      BERT/RoBERTa fine-tuning
  aspects.py         aspect mapping + tokenizer
data/                raw/ processed/ (DVC), demo.csv
models/              trained weights (tfidf committed; bert/roberta/lstm gitignored)
tests/               pytest unit + API tests
.github/workflows/ci.yml
Dockerfile           multi-stage (frontend build + backend)
Dockerfile.full      full deployment (CPU torch + transformers + all models)
```

## Branches

This repository hosts several course projects on separate branches. This project lives on `main`.

| Branch | Content |
|--------|---------|
| `main` | This project — aspect-based sentiment analysis |
| `course-log` | Course materials (notes, practice guides, slides) — was `main` |
| `cars-crud` | Practice project — cars CRUD |
| `git-playground` | Practice project — Git |

## Development notes

Topic selection was constrained by hardware. The machine has only a CPU, and renting compute was not planned at the outset; single-aspect sentiment classification was too basic, so multi-aspect sentiment analysis was chosen to deliver a complete result within the assignment window. Restaurant reviews fit the need for an interactive demo with a reserved test set.

The machine was fixed before development began. Five Python installations shared one user site-packages, and an interrupted pip upgrade left `~umpy`, `~ip`, and `~crypt` directories, so `import sklearn` raised `numpy.core.multiarray failed to import`. After consolidating on `C:\Python312`, removing the leftovers, and reinstalling numpy, all work moved into a project-local `.venv`.

The dataset changed twice. Starting from English SemEval-2014, the four-aspect output and bilingual UI were settled during development, moving labels from aspect-term to aspect-category and introducing Chinese data. MAMS is hosted on Google Drive and hard to fetch; ASAP from Meituan-Dianping was adopted instead — Chinese, on GitHub, roughly 46,850 reviews, 18 categories collapsed into four aspects, an order of magnitude larger than SemEval. Chinese is segmented at the character level, which removes the dictionary dependency.

The model saw the most iterations. The TF-IDF + LR baseline reported ~0.45 confidence, caused by a noisy neutral class; collapsing to binary classification and labelling low-confidence predictions neutral lifted both macro-F1 and confidence. A deep model requires a GPU, which the machine lacks; BiLSTM trained on CPU fell short for lack of samples, so an AutoDL RTX 3080 Ti was rented for 5 yuan to fine-tune BERT and RoBERTa. The latter beat the baseline on mixed multi-aspect sentences without leading across the board, and their CPU inference cost is high, so all three models remain, switchable at runtime with TF-IDF as default.

DVC's remote moved once. With no Azure account initially, a local directory served as a placeholder; after obtaining a student account, a Storage Account and container were created and the remote switched to Azure Blob. The first push hit socket timeouts on the large files — intermittent China-to-Azure latency — and finished on retry.

Experiment tracking follows the course MLflow guide, local first, then DagsHub; MLflow 3.x's `CreateLoggedModel` endpoint is not implemented by DagsHub's 2.x server, so the client is pinned to `mlflow<3`.

The app expanded from a single `/predict` endpoint to four-aspect output, a bilingual UI, model switching, CSV batch import/export, record CRUD, and finally two pages. One subtle failure: a `'\n'` in the Python triple-quoted string became a literal newline in the generated JavaScript and broke the script, avoided with `String.fromCharCode`.

CI/CD uses GitHub Actions: pull requests run ruff, pytest, and a Docker build; pushes to staging or main publish images. The image is `python:3.12-slim` with runtime dependencies only, with torch and transformers lazy-imported.

The rating task came later. ASAP ships a `star` column, so a second head predicts the overall 1-5 score alongside the four aspects. The first TF-IDF + LR rating model reached 55.9% exact accuracy; the first BERT attempt at a 128-token max length *regressed* to 54.2%, and profiling the corpus showed the mean review is 320 characters, meaning 99.8% of inputs were being truncated. Retraining at 384 tokens with a linear warmup schedule lifted BERT to 58.1%, and RoBERTa (chinese-wwm-ext) with label smoothing reached 59.2% accuracy and MAE 0.446, now the default rating model. SHAP (`LinearExplainer`) was added on the TF-IDF model so each aspect can justify itself with the tokens that most moved the decision.

The frontend was rewritten twice. It began as server-rendered HTML embedded in the FastAPI module, then became a separate Vue 3 + Vite single-page application once the UI accumulated enough state to outgrow a template string. The decoupling removed ~430 lines of HTML/CSS/JS from `app/main.py`, leaving a pure REST API that serves the built SPA as static files in production. The design follows Typecho's default theme — serif body, thin borders, a muted blue accent — with guidance hints, a focus ring, a clear button, and a character counter added to the input, and the result cards redesigned with labeled probability bars and keyword chips.

## Environment

Python 3.12 in a project venv:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements-ml.txt -r requirements-dev.txt
```

| File | Purpose |
|------|---------|
| `requirements.txt` | runtime (fastapi, uvicorn, joblib, scikit-learn, numpy, pandas, python-multipart) |
| `requirements-ml.txt` | training + deep-model serving (mlflow<3, matplotlib, dagshub, torch, transformers) |
| `requirements-dev.txt` | pytest, httpx, ruff |

Environment variables (tokens stay out of git):

| Variable | Meaning |
|----------|---------|
| `DAGSHUB_REPO_OWNER` / `DAGSHUB_REPO_NAME` | DagsHub repo, e.g. `losproeocl` / `WUT-DevOps-Course` |
| `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD` | DagsHub username / access token |
| `REVIEWS_DB_PATH` | optional; path to the saved-reviews SQLite DB |
| `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN` | GitHub Actions secrets for image publishing |

## Data

ASAP (Meituan-Dianping/asap), ~46,850 Chinese reviews, 18 aspect categories mapped to four:

| Aspect | ASAP categories |
|--------|-----------------|
| taste | Food#Portion, Food#Taste, Food#Appearance, Food#Recommend |
| service | Service#Queue, Service#Hospitality, Service#Parking, Service#Timely |
| price | Price#Level, Price#Cost_effective, Price#Discount |
| ambience | Ambience#Decoration, Ambience#Noise, Ambience#Space, Ambience#Sanitary |

Polarity is `1`/`0`/`-1` (`-2` = not mentioned, dropped). DVC tracks `data/raw` and `data/processed`, remote on Azure Blob:

```bash
dvc remote add -d myazure azure://efreiwut-projdvc
dvc remote modify myazure account_name efreiwut2026dvc
dvc remote modify --local myazure connection_string "<secret>"   # .dvc/config.local, gitignored
dvc push
```

Two versions: `v1` (raw) and `v2` (char-tokenized + deduplicated).

## Models and training

Binary per aspect (positive vs negative); the "neutral" class is dropped at training time, and low-confidence predictions are reported as neutral.

| Model | Where | macro-F1 (taste/service/price/ambience) |
|-------|-------|------------------------------------------|
| TF-IDF + LR (tuned) | CPU, `ml/train.py` | 0.738 / 0.784 / 0.786 / 0.760 |
| TF-IDF + LR (search, per aspect) | CPU, `ml/train.py --mode search` | rerun to refresh |
| BiLSTM | CPU, `ml/train_lstm.py` | weaker on this data |
| BERT (bert-base-chinese, 3 ep) | AutoDL GPU, `ml/train_bert.py` | 0.758 / 0.806 / 0.758 / 0.824 |
| RoBERTa (chinese-wwm-ext, 3 ep) | AutoDL GPU | 0.765 / 0.803 / 0.763 / 0.789 |

A separate **star-rating predictor** classifies the overall 1-5 rating (the `star` column in ASAP):

| Rating model | Where | accuracy | MAE | within ±1 |
|--------------|-------|----------|-----|-----------|
| TF-IDF + LR | CPU, `ml/train_rating.py` | 0.559 | 0.512 | — |
| BERT (bert-base-chinese, 4 ep) | AutoDL GPU | 0.581 | 0.466 | 0.957 |
| RoBERTa (chinese-wwm-ext, 5 ep) | AutoDL GPU, `ml/train_rating_bert.py` | **0.592** | **0.446** | 0.964 |

```bash
python -m ml.data_pipeline              # build v1/v2 CSVs
python -m ml.train --data-version v2 --mode tuned   # TF-IDF + LR, fixed tuned config
python -m ml.train --data-version v2 --mode search --search-iter 8 --cv-folds 3  # per-aspect CV tuning
python -m ml.train_lstm --data-version v2           # BiLSTM
python -m ml.train_bert --data-dir data --epochs 3  # BERT/RoBERTa (GPU recommended)
python -m ml.train_rating                           # star rating
python -m ml.train_rating_bert --model-path ...     # star rating (GPU, RoBERTa)
python -m ml.register_rating_roberta                # register rating RoBERTa to MLflow
```

Registered on DagsHub: `absa_restaurant` (unified TF-IDF+LR, Production), `absa_restaurant_rating` (TF-IDF rating, Production), `absa_restaurant_rating_roberta` (RoBERTa rating, Production), `absa_restaurant_bert` (Staging), `absa_restaurant_roberta` and `absa_restaurant_lstm` (Archived). Stage transitions live in `ml/stage_models.py`; registry-based serving is demonstrated by `ml/serve_model.py` (`mlflow.pyfunc.load_model("models:/absa_restaurant/Production")`).

### Confidence and the "neutral" class

Each aspect is binary (positive vs negative) at training time; the model reports the winning class's probability as **confidence**. Predictions whose confidence falls below `UNCERTAIN_THRESHOLD = 0.55` are downgraded to **neutral** rather than committing to a weak positive/negative call. This keeps the bar for a confident judgment explicit and surfaced in the UI as the probability bars next to each aspect.

The star rating (1-5) is a separate 5-class task; its "confidence" is the softmax probability of the predicted star. The RoBERTa rating model is within one star of the true rating 96.4% of the time, which is the more meaningful agreement metric for an ordinal 1-5 scale.

## Course requirements — configuration guide

Every requirement from `project-instructions.pdf`, with the steps to reproduce it.

### The app

**Train or fine-tune a model.** TF-IDF + LR is trained by `ml/train.py`; BERT/RoBERTa are fine-tuned by `ml/train_bert.py` (GPU recommended).

```bash
python -m ml.data_pipeline
python -m ml.train --data-version v2 --mode search --search-iter 8 --cv-folds 3
python -m ml.train_bert --data-dir data --epochs 3
```

**Dataset versions.** `data/processed/{train,val,test}_v1.csv` (raw) and `_v2.csv` (char-tokenized + deduplicated). Regenerate with `python -m ml.data_pipeline`.

**Tests.** 39 pytest tests under `tests/` (unit + API), covering empty/whitespace input, unknown model, CSV import/export, CRUD, batch delete, and the per-aspect training/search helpers. Run `python -m pytest`.

### DevOps

**Version control.** Branch model `main` ← `staging` ← `dev` ← `feature/*`; the deliverable is on `main` (course materials on `course-log`).

```bash
git switch -c feature/xxx
git push -u origin feature/xxx   # then open a PR into dev, then staging, then main
```

**Docker.** `Dockerfile` (`python:3.12-slim`) with a `.dockerignore` that excludes `tests/`, `ml/`, `scripts/`, `data/raw`, `data/processed`, `mlruns/`, `.dvc/`, and the large model weights.

```bash
docker build -t absa-app .
docker run -p 8000:8000 absa-app
```

**CI/CD.** `.github/workflows/ci.yml` triggers on PR/push to protected branches and runs `ruff check app ml tests scripts`, `python -m pytest`, then `docker build -t absa-app:ci .`. Enable branch protection (GitHub → Settings → Branches) so a PR cannot merge while CI fails. Set `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN` under Settings → Secrets → Actions for image publishing.

### MLOps

**DVC.**

```bash
dvc init
dvc remote add -d myazure azure://efreiwut-projdvc
dvc remote modify myazure account_name efreiwut2026dvc
dvc remote modify --local myazure connection_string "<secret>"   # .dvc/config.local, gitignored
dvc add data/raw data/processed
dvc push
```

`ml/README_data.md` records the two versions and the v1 → v2 changes.

**MLflow.** Set `DAGSHUB_REPO_OWNER` / `DAGSHUB_REPO_NAME` and `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD`, then run training — `ml/train.py` logs the git SHA, data checksum, hyperparameters, metrics, and model, and registers it. `ml/README_experiments.md` documents the baseline vs tuned comparison; `mlflow ui` shows the local history.

**Model deployment.** Stage registered models with `ml/stage_models.py` (Staging/Production/Archived), then serve from the registry:

```bash
python -m ml.stage_models
python -m ml.serve_model --model models:/absa_restaurant/Production --text "食物很好吃"
# or via the CLI:
mlflow models serve -m models:/absa_restaurant/Production -p 5003
```

### Repo structure

```
app/  ml/  data/  models/  scripts/  tests/
requirements.txt  requirements-ml.txt  requirements-dev.txt
Dockerfile  .dockerignore  .github/workflows/ci.yml
README.md  README_zh.md  DEPLOYMENT.md
```

## Bugs, quick reference

- `numpy.core.multiarray failed to import` → multiple Pythons sharing user site + `~` leftovers; consolidated on C:\Python312 and deleted them.
- Confidence ~0.45 → "neutral" was a catch-all; went binary.
- Small data capped models → switched to ASAP (~10x bigger).
- `CreateLoggedModel` → `INVALID_PARAMETER_VALUE` → pin `mlflow<3` (DagsHub server is 2.x).
- `process() takes no keyword arguments` on AutoDL → `wget` the model from hf-mirror, load from a local path.
- Empty TF-IDF features for Chinese → `token_pattern=r"\S+"` (default drops single CJK chars).
- JS buttons dead → `'\n'` in the triple-quoted HTML became a real newline; use `String.fromCharCode(10)`.
- 400 MB weights → gitignored (LSTM DVC-tracked; TF-IDF ~30 MB committed).
- 5 epochs overfit → 3 is the sweet spot.
- Azure push timeout → intermittent, retry.

## Running

The frontend (Vue 3 + Vite) is decoupled from the backend (FastAPI). For development run both; for production build the SPA and let FastAPI serve it.

```bash
# 1. backend (API on :8000)
python -m uvicorn app.main:app --reload

# 2. frontend dev server (Vite on :5173, proxies /api -> :8000)
cd frontend && npm install && npm run dev
# http://127.0.0.1:5173

# production: build the SPA, then FastAPI serves it at :8000
cd frontend && npm run build
python -m uvicorn app.main:app        # http://127.0.0.1:8000

docker build -t absa-app .            # multi-stage: builds frontend + backend
docker run -p 8000:8000 absa-app

# full deployment (all models: BERT/RoBERTa + RoBERTa rating, ~4 GB image)
docker build -f Dockerfile.full -t absa-app:full .
docker run -d --name absa-full --restart unless-stopped -p 8000:8000 absa-app:full
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for the production server setup (swap, reverse proxy, smoke test).

Tests: `python -m pytest`; lint: `ruff check app ml tests scripts`.
