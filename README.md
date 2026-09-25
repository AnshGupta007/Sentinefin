# SentinelFin

Deep-learning pipeline that monitors CFPB consumer-complaint narratives to detect
emergent harm-pattern clusters before they are officially recognized, validated
with retrospective backtesting.

## Pipeline stages

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

## Deep Learning Models & Comparative Benchmark

We evaluate five distinct deep learning and hybrid architectures on the CFPB financial grievances dataset:
1. **Baseline MLP (`MLPTagClassifier`)**: 3-layer deep feedforward classifier with BatchNorm, GELU, and Cosine Annealing (**91.30% Accuracy**).
2. **DEC Autoencoder (`DECModel`)**: 4-layer symmetric autoencoder with Student-$t$ soft clustering for zero-day fraud discovery (**0.0010 MSE**, **+0.742 Silhouette**).
3. **Hybrid XGBoost + BiLSTM**: 2-layer Bidirectional LSTM deep feature extractor fused with an XGBoost decision tree ensemble (**86.10% Accuracy**).
4. **FinBERT (`ProsusAI/finbert`)**: 12-layer domain-specific financial Transformer with a deep classification head (**72.80% Accuracy**).
5. **Proposed CNN-RNN**: Multi-scale 1D Convolutional filter banks ($k=3,5,7$) with Bidirectional LSTM sequential memory (**81.70% Accuracy**).

To run the unified benchmark:
```bash
python scripts/train_and_compare_models.py
```
Detailed architectural comparisons, formulas, and oral defense guides are in [`docs/MODEL_COMPARISON.md`](docs/MODEL_COMPARISON.md).

## CI/CD (GitHub Actions)

- `.github/workflows/ci.yml` — on every push/PR: ruff lint, unit tests, then the
  full smoke pipeline end-to-end (offline encoder) so the model code is always
  proven runnable.
- `.github/workflows/pipeline.yml` — weekly schedule plus manual dispatch:
  runs the smoke pipeline and uploads artifacts (metrics JSON, report HTML);
  set `run_full: true` in the dispatch input to run against the real CFPB data.

## Interactive Clustering Dashboard (React + ECharts)

A modern, hardware-accelerated 2D/3D latent manifold dashboard built with React 18, TypeScript, Tailwind CSS, and Apache ECharts.

```bash
# 1. Export the latest SentinelFin embeddings & DEC clusters to the dashboard
source .venv_linux/bin/activate
python scripts/export_frontend_data.py

# 2. Launch the frontend development server
cd frontend
npm install
npm run dev
# Dashboard is live at http://localhost:3000
```

- **Features**: Interactive UMAP/PCA coordinate toggling, Student-$t$ DEC cluster visibility filtering, point-by-point narrative inspection drawer, and live keyword filtering.

## Layout

```
src/sentinefin/     pipeline package (one module per phase)
frontend/           React 18 + TypeScript + ECharts interactive cluster studio
scripts/            dataset sampling and frontend export scripts
tests/              pytest suite incl. end-to-end smoke test
data/raw|processed  regenerable datasets (gitignored)
outputs/            metrics, plots, ranked emergent clusters
reports/            pre-registered backtest cases + static HTML report
docs/               master prompt, PRD, and model accuracy metrics
```

- **Model 1 Results**: [`docs/MODEL_1_SUPERVISED_MLP_RESULTS.md`](docs/MODEL_1_SUPERVISED_MLP_RESULTS.md) — Supervised Deep Neural Representation Classifier (90.8% accuracy, 0.856 Macro-F1).
- **Model 2 Results**: [`docs/MODEL_2_DEC_CLUSTERING_RESULTS.md`](docs/MODEL_2_DEC_CLUSTERING_RESULTS.md) — Unsupervised Deep Embedded Clustering (`DEC` Autoencoder: 0.0010 MSE loss, +0.742 Silhouette score).
- **Model Comparison**: [`docs/MODEL_COMPARISON.md`](docs/MODEL_COMPARISON.md) — Comprehensive side-by-side evaluation, theoretical objective contrast, zero-day fraud analysis, and oral defense guide.
- **Master Metrics**: [`docs/MODEL_PERFORMANCE_METRICS.md`](docs/MODEL_PERFORMANCE_METRICS.md) — Complete benchmark records across all 5 neural pipeline phases.
- **Master Prompt**: `docs/SentinelFin_Master_Prompt.md` for the phase roadmap and acceptance criteria.
