# SentinelFin

Deep-learning pipeline that monitors CFPB consumer-complaint narratives to detect
emergent harm-pattern clusters before they are officially recognized, validated
with retrospective backtesting. Built from `docs/SentinelFin_Master_Prompt.md`.

## Pipeline stages (all neural, per the master prompt)

| Stage | Technique | Module |
|---|---|---|
| Representation | Pretrained sentence Transformer (`all-MiniLM-L6-v2`) | `embed.py` |
| Baseline | Feedforward MLP classifier (neural sanity check) | `mlp_baseline.py` |
| Clustering | Deep Embedded Clustering (autoencoder + KL self-training; k-means only initializes centroids) | `dec.py` |
| Drift | LSTM Autoencoder reconstruction-error emergence scoring | `drift.py`, `trajectories.py` |
| Backtesting | Truncated-data re-runs with pre-registered cases (no leakage) | `backtest.py` |
| Reporting | Static HTML report with TF-IDF labels and optional LLM labeling | `reporting.py` |

## Quick start

```powershell
pip install -e ".[dev]"

# Create a lightweight, balanced dataset (e.g. 20,000 complaints) for training on resource-constrained hardware
sentinefin sample --size 20000 --output data/raw/complaints_small.csv

# Run full pipeline on the sampled dataset
sentinefin pipeline

# Or stage by stage
sentinefin ingest --raw data/raw/complaints_small.csv
sentinefin embed
sentinefin cluster
sentinefin drift
sentinefin report      # writes reports/sentinefin_report.html

# Phase 5 retrospective validation
sentinefin backtest
```

Smoke profile for CI (tiny models, offline encoder fallback):

```bash
SENTINEFIN_SMOKE=1 sentinefin pipeline
```

## CI/CD (GitHub Actions)

- `.github/workflows/ci.yml` — on every push/PR: ruff lint, unit tests, then the
  full smoke pipeline end-to-end (offline encoder) so the model code is always
  proven runnable.
- `.github/workflows/pipeline.yml` — weekly schedule plus manual dispatch:
  runs the smoke pipeline and uploads artifacts (metrics JSON, report HTML);
  set `run_full: true` in the dispatch input to run against the real CFPB data.

## Layout

```
src/sentinefin/     pipeline package (one module per phase)
tests/              pytest suite incl. end-to-end smoke test
data/raw|processed  regenerable datasets (gitignored)
outputs/            metrics, plots, ranked emergent clusters
reports/            pre-registered backtest cases + HTML report
docs/               master prompt, PRD, and model accuracy metrics
```

- See [`docs/MODEL_PERFORMANCE_METRICS.md`](docs/MODEL_PERFORMANCE_METRICS.md) for full accuracy benchmarks (90.8% test accuracy), loss curves, and architecture details.
- See `docs/SentinelFin_Master_Prompt.md` for the phase roadmap and acceptance criteria.
