# Data Versioning (DVC)

## Source

- **Dataset**: ASAP (Aspect-based Sentiment Analysis for Chinese restaurant
  reviews), from the Meituan review benchmark.
- **Format**: CSV — a `review` text column plus 15 fine-grained aspect columns
  (e.g. `Food#Taste`, `Service#Queue`, `Price#Level`, `Ambience#Noise`). Each
  aspect cell is a polarity: `1` positive / `0` neutral / `-1` negative / `-2`
  not-mentioned (dropped).

The raw data is **not committed to git**; it lives in DVC remote storage, and git
only stores the small `.dvc` pointer files.

## How data is versioned

```bash
dvc add data/raw data/processed     # creates .dvc files
git add data/raw.dvc data/processed.dvc   # commit the pointer files
dvc push                            # upload actual data to the DVC remote
```

## DVC remote

The default remote (`myazure`) is **Azure Blob Storage**:

```bash
dvc remote add -d myazure azure://efreiwut-projdvc
dvc remote modify myazure account_name efreiwut2026dvc
# auth via AZURE_STORAGE_CONNECTION_STRING / AZURE_STORAGE_KEY, or az login
```

Only the remote *name* is committed in `.dvc/config`; the data lives wherever
that remote points. A teammate can `dvc pull` to fetch the exact snapshot.

## Aspect mapping

The 15 fine-grained ASAP categories are merged into four coarse aspects
(`ml/aspects.py`):

| Aspect | Code | ASAP categories |
|--------|------|-----------------|
| 口味 (Taste/Food) | `food` | Food#Portion, Food#Taste, Food#Appearance, Food#Recommend |
| 服务 (Service) | `service` | Service#Queue, Service#Hospitality, Service#Parking, Service#Timely |
| 价格 (Price) | `price` | Price#Level, Price#Cost_effective, Price#Discount |
| 环境 (Ambience) | `ambience` | Ambience#Decoration, Ambience#Noise, Ambience#Space, Ambience#Sanitary |

`-2` (not mentioned) rows are dropped during parsing.

## Dataset versions

| Version | Path | Description |
|---------|------|-------------|
| v1 | `data/processed/{train,val,test}_v1.csv` | Long-form parse: one row per `(text, aspect)` mention, mapped to a coarse aspect. |
| v2 | `data/processed/{train,val,test}_v2.csv` | Character-tokenized Chinese text (space-separated) + deduplicated `(text, aspect, label)` rows. |

### What changed v1 -> v2

- Chinese text tokenized character-by-character (`ml/aspects.tokenize_zh`).
- Removed duplicate `(text, aspect, label)` records.

## Linking code -> data

Each MLflow run logs `train_checksum` (MD5 of the training CSV) and `data_version`,
so a model can always be traced back to the exact dataset that produced it.
