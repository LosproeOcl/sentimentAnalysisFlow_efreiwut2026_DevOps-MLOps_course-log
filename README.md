# MLflow 3 Model Registration — GPU RoBERTa Aspect Model

This branch (`fix/mlflow3-logmodel-docs`) documents and scripts the end-to-end
**MLflow 3** flow for the fine-tuned Chinese RoBERTa aspect-sentiment model:
train → log experiment → register → staging → production → deploy from the
registry.

> The ~390 MB checkpoint is **not** committed here — GitHub rejects files over
> 100 MB. The weights live in the DagsHub model registry and are pulled with
> `ml/download_roberta_model.py`.

## Model

- Backbone: `hfl/chinese-roberta-wwm-ext` (12-layer BERT-style encoder, 768-dim).
- 4 binary heads: `food` / `service` / `price` / `ambience`.
- Data: ASAP (Meituan) Chinese restaurant reviews — train 36,850 + dev 4,940,
  test 4,940.
- Training: CUDA + AMP (fp16), batch 8 × grad-accum 4, lr 2e-5, 3 epochs,
  linear warmup + decay.

Held-out test macro-F1 (positive / negative):

| food | service | price | ambience | **avg** |
|---:|---:|---:|---:|---:|
| 0.7760 | 0.8079 | 0.7556 | 0.8257 | **0.7913** |

## DagsHub MLflow

- Experiment: `absa_restaurant_roberta_gpu`
- Registered model: `absa_restaurant_roberta_gpu` (v2, `Production`)

### Register

```bash
export DAGSHUB_TOKEN=...          # DagsHub token (scope: repo)
python -m ml.register_roberta_gpu --checkpoint models/roberta_absa_alldata.pt
```

The script runs the full flow: rebuilds the model from the checkpoint, logs
params/metrics, uploads a self-contained `pyfunc` model, registers it, moves it
`staging → production`, then loads the `Production` model back from the registry
to prove deployment.

### Download the weights

```bash
export DAGSHUB_TOKEN=...
python -m ml.download_roberta_model --dest models/roberta_absa_alldata.pt
```

## MLflow 3 gotchas (why this branch exists)

- `mlflow.pyfunc.log_model(artifact_path=...)` is deprecated in favour of
  `name=...`.
- `mlflow.register_model` is deprecated; use
  `MlflowClient.create_model_version(name, source, run_id=...)`.
- Model-registry **stages** are deprecated in favour of **aliases**; both still
  work in 3.x (`transition_model_version_stage` emits a FutureWarning).
- On Windows, `mlflow` prints emoji to stdout when a run ends; the GBK console
  raises `UnicodeEncodeError` and aborts `end_run` **before it flushes the large
  artifact upload**. Fix: `sys.stdout.reconfigure(encoding="utf-8")` before any
  `mlflow` call.

## Files

| File | Purpose |
|------|---------|
| `ml/register_roberta_gpu.py` | checkpoint → MLflow run → registry → staging → production → deploy |
| `ml/download_roberta_model.py` | pull the 390 MB checkpoint from the DagsHub registry |
