"""Register the fine-tuned Chinese RoBERTa aspect model to DagsHub MLflow.

End-to-end flow for the 4-aspect (food/service/price/ambience) sentiment model:

  1. Rebuild `TransformerAbsa` from a saved checkpoint (config + state_dict)
     and run a local sanity inference (no network) to gate the upload.
  2. Log params / metrics and a self-contained `pyfunc` model to a dedicated
     DagsHub MLflow experiment.
  3. Register the model, then stage it `staging -> production`.
  4. Deploy from the registry: load the `Production` model back and infer.

Credentials are read from environment variables (NOT committed anywhere):

    export DAGSHUB_TOKEN=<your token>
    export DAGSHUB_USERNAME=1064364143        # optional, default shown
    python -m ml.register_roberta_gpu \
        --checkpoint models/roberta_absa_alldata.pt \
        --tokenizer-dir models/chinese-roberta-wwm-ext

The checkpoint (~390 MB) exceeds GitHub's 100 MB per-file limit, so it is
stored in the DagsHub registry rather than git; see `ml/download_roberta_model.py`.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

# MLflow prints emojis (e.g. "\\U0001f3c3") to stdout when a run ends. On the
# Windows GBK console this raises UnicodeEncodeError and aborts end_run BEFORE
# it flushes the large artifact upload. Reconfigure to UTF-8 up front.
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow.tracking import MlflowClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EXPERIMENT = "absa_restaurant_roberta_gpu"
MODEL_NAME = "absa_restaurant_roberta_gpu"
ASPECTS = ["food", "service", "price", "ambience"]

TOKENIZER_FILES = [
    "vocab.txt", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json", "config.json",
]

# Held-out test metrics from the CUDA all-data run (train+dev, 3 epochs).
METRICS = {
    "food_macro_f1": 0.7760, "service_macro_f1": 0.8079,
    "price_macro_f1": 0.7556, "ambience_macro_f1": 0.8257, "avg_macro_f1": 0.7913,
}
PARAMS = {
    "model": "hfl/chinese-roberta-wwm-ext", "epochs": 3, "batch_size": 8,
    "grad_accum": 4, "lr": 2e-5, "weight_decay": 0.01, "max_len": 128,
    "amp": True, "use_dev_in_train": True,
    "train_size": 41790, "test_size": 4940, "source": "gpu-finetune",
}

MAX_LEN = 128
THRESHOLD = 0.55


def _config_from_ckpt(ckpt: dict):
    model_type = ckpt["config"].get("model_type", "bert")
    return AutoConfig.for_model(model_type).from_dict(ckpt["config"])


class TransformerAbsa(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = AutoModel.from_config(config)
        self.heads = nn.ModuleList([nn.Linear(config.hidden_size, 1) for _ in ASPECTS])

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return torch.stack([h(cls).squeeze(-1) for h in self.heads], dim=1)  # (B, 4)


class RoBERTaAbsaPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.tokenizer = AutoTokenizer.from_pretrained(context.artifacts["tokenizer"])
        ckpt = torch.load(context.artifacts["model"], map_location="cpu", weights_only=False)
        self.model = TransformerAbsa(_config_from_ckpt(ckpt))
        missing, unexpected = self.model.load_state_dict(ckpt["state_dict"])
        if missing or unexpected:
            raise RuntimeError(f"state_dict mismatch: missing={missing} unexpected={unexpected}")
        self.model.eval()

    @torch.no_grad()
    def predict(self, context, model_input, params=None):
        texts = model_input["text"].astype(str).tolist()
        enc = self.tokenizer(texts, max_length=MAX_LEN, truncation=True,
                             padding="max_length", return_tensors="pt")
        prob = torch.sigmoid(self.model(enc["input_ids"], enc["attention_mask"]))  # (B, 4)
        rows = []
        for text, p in zip(texts, prob.tolist()):
            row = {"text": text}
            for i, aspect in enumerate(ASPECTS):
                v = float(p[i])
                label = "positive" if v >= THRESHOLD else ("negative" if v <= 1 - THRESHOLD else "neutral")
                row[f"{aspect}_label"] = label
                row[f"{aspect}_confidence"] = round(v, 4)
            rows.append(row)
        return pd.DataFrame(rows)


def sanity_check(checkpoint: str, tokenizer_dir: str) -> None:
    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = TransformerAbsa(_config_from_ckpt(ckpt))
    missing, unexpected = model.load_state_dict(ckpt["state_dict"])
    assert not missing and not unexpected, (missing, unexpected)
    model.eval()
    texts = [
        "这家店的味道非常棒，服务也很好，就是价格有点贵。",
        "菜品一般，环境很吵，不会再来了。",
    ]
    enc = tok(texts, max_length=MAX_LEN, truncation=True, padding="max_length", return_tensors="pt")
    with torch.no_grad():
        prob = torch.sigmoid(model(enc["input_ids"], enc["attention_mask"]))
    print("[sanity] state_dict loads cleanly; reconstruction OK")
    for t, p in zip(texts, prob.tolist()):
        print("  ", t[:20], "->", {a: round(v, 3) for a, v in zip(ASPECTS, p)})


def to_stage(client: MlflowClient, version: int, stage: str) -> None:
    try:
        client.transition_model_version_stage(MODEL_NAME, version, stage=stage)
        print(f"[stage] v{version} -> {stage}")
    except Exception as exc:
        print(f"[stage] transition unsupported ({exc}); setting alias '{stage}'")
        client.set_registered_model_alias(MODEL_NAME, stage, version)


def register(client: MlflowClient, model_uri: str, run_id: str) -> int:
    try:
        client.create_registered_model(MODEL_NAME)
        print(f"[registry] created '{MODEL_NAME}'")
    except Exception:
        print(f"[registry] '{MODEL_NAME}' already exists")
    version = int(client.create_model_version(MODEL_NAME, model_uri, run_id=run_id).version)
    print(f"[register] {MODEL_NAME} version {version}")
    return version


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=os.path.join(ROOT, "models", "roberta_absa_alldata.pt"))
    p.add_argument("--tokenizer-dir", default=os.path.join(ROOT, "models", "chinese-roberta-wwm-ext"))
    p.add_argument("--owner", default=os.environ.get("DAGSHUB_REPO_OWNER", "losproeocl"))
    p.add_argument("--repo", default=os.environ.get("DAGSHUB_REPO_NAME", "WUT-DevOps-Course"))
    args = p.parse_args()

    token = os.environ.get("DAGSHUB_TOKEN")
    user = os.environ.get("DAGSHUB_USERNAME", "1064364143")
    if not token:
        raise SystemExit("DAGSHUB_TOKEN is not set; see module docstring")

    os.environ["MLFLOW_TRACKING_URI"] = f"https://dagshub.com/{args.owner}/{args.repo}.mlflow"
    os.environ["MLFLOW_TRACKING_USERNAME"] = user
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    print(f"[mlflow] version {mlflow.__version__}")
    sanity_check(args.checkpoint, args.tokenizer_dir)

    tok_dir = os.path.join(ROOT, "models", "roberta_tokenizer_artifact")
    shutil.rmtree(tok_dir, ignore_errors=True)
    os.makedirs(tok_dir)
    for f in TOKENIZER_FILES:
        shutil.copy(os.path.join(args.tokenizer_dir, f), tok_dir)

    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)
    print(f"[experiment] '{EXPERIMENT}' id={exp.experiment_id}")

    with mlflow.start_run(run_name="roberta-gpu-alldata"):
        mlflow.log_params(PARAMS)
        mlflow.log_metrics(METRICS)
        info = mlflow.pyfunc.log_model(
            artifact_path="roberta_absa",
            python_model=RoBERTaAbsaPyfunc(),
            artifacts={"model": args.checkpoint, "tokenizer": tok_dir},
        )
        run_id = mlflow.active_run().info.run_id
        print(f"[run] id={run_id}")

    version = register(client, info.model_uri, run_id)

    sample = pd.DataFrame({"text": [
        "这家店的味道非常棒，服务也很好，就是价格有点贵。",
        "菜品一般，环境很吵，不会再来了。",
    ]})

    to_stage(client, version, "staging")
    staging = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{version}")
    print(f"[staging] v{version} inference:")
    print(staging.predict(sample).to_string())

    to_stage(client, version, "production")
    prod = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{version}")
    print(f"[production] deployed from registry v{version}; inference:")
    print(prod.predict(sample).to_string())

    mv = client.get_model_version(MODEL_NAME, version)
    print(f"[registry] v{version} current_stage={mv.current_stage} aliases={mv.aliases}")
    print("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
