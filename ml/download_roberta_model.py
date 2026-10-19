"""Pull the 390 MB RoBERTa checkpoint from the DagsHub MLflow registry.

The checkpoint exceeds GitHub's 100 MB per-file limit, so it is stored in the
DagsHub model registry (uploaded by `ml/register_roberta_gpu.py`) instead of
git. This script downloads the raw ``.pt`` back for local training/loading.

Usage:
    export DAGSHUB_TOKEN=<token>
    python -m ml.download_roberta_model --dest models/roberta_absa_alldata.pt
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from mlflow.tracking import MlflowClient

MODEL_NAME = "absa_restaurant_roberta_gpu"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--owner", default=os.environ.get("DAGSHUB_REPO_OWNER", "losproeocl"))
    p.add_argument("--repo", default=os.environ.get("DAGSHUB_REPO_NAME", "WUT-DevOps-Course"))
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--version", default="2", help="model version (or a stage alias)")
    p.add_argument("--dest", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "roberta_absa_alldata.pt"))
    args = p.parse_args()

    token = os.environ.get("DAGSHUB_TOKEN")
    user = os.environ.get("DAGSHUB_USERNAME", "1064364143")
    if not token:
        raise SystemExit("DAGSHUB_TOKEN is not set; see module docstring")

    os.environ["MLFLOW_TRACKING_URI"] = f"https://dagshub.com/{args.owner}/{args.repo}.mlflow"
    os.environ["MLFLOW_TRACKING_USERNAME"] = user
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    client = MlflowClient()
    mv = client.get_model_version(args.model, args.version)
    source = mv.source  # e.g. runs:/<run_id>/roberta_absa
    run_id = source.split("/")[1]
    artifact_path = "/".join(source.split("/")[2:]) or "roberta_absa"

    tmp = tempfile.mkdtemp(prefix="roberta_download_")
    local_dir = client.download_artifacts(run_id, artifact_path, tmp)
    pts = glob.glob(os.path.join(local_dir, "**", "*.pt"), recursive=True)
    if not pts:
        raise SystemExit(f"no .pt checkpoint found under {local_dir}")

    os.makedirs(os.path.dirname(args.dest), exist_ok=True)
    shutil.copyfile(pts[0], args.dest)
    print(f"[download] {mv.name} v{mv.version} -> {args.dest} "
          f"({os.path.getsize(args.dest) / 2**20:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
