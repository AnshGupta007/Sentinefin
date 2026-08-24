# Product Requirements Document
### SentinelFin — Temporal Drift Detection for Early Warning of Emergent Consumer-Finance Harms

| Field | Value |
|---|---|
| Document title | Product Requirements Document — SentinelFin |
| Version | 1.1 (Draft for Review) |
| Date | August 23, 2026 |
| Prepared by | [Your Name] — MBA Finance, Deep Learning & NLP Capstone |
| Institution / Course | [Your Institution] — [Course Code(s)] |
| Status | Draft — for advisor / professor review |

**Revision note (v1.1):** Every core pipeline stage has been revised to use a trained or pretrained neural network rather than a classical ML algorithm, since this project is being submitted for **Deep Learning** course credit specifically. Classical clustering (HDBSCAN) is replaced with **Deep Embedded Clustering (DEC)**; statistical drift detection is replaced with an **LSTM Autoencoder**. See Section 9.2 for an explicit stage-by-stage architecture map.

---

## 1. Executive Summary

SentinelFin is a research prototype that continuously monitors embeddings of U.S. consumer financial complaint narratives to detect emergent, semantically coherent harm patterns — new scams, risky products, or systemic issues — before they are officially recognized as a complaint category or become the subject of enforcement action or news coverage. Where existing tools classify complaints into already-known categories or predict a complaint's outcome, SentinelFin is designed to answer a different question: can a genuinely new pattern be discovered early enough to matter? The system is built entirely on free, public data (the CFPB Consumer Complaint Database) and is validated through a retrospective backtesting protocol against real historical cases, so its early-warning claim is evidence-based rather than anecdotal. Every stage of the core pipeline — representation, clustering, and drift detection — is implemented as a trained or pretrained deep neural network.

## 2. Problem Statement & Background

New financial products and scams — Buy Now Pay Later, earned-wage-access, AI voice-cloning fraud, crypto-linked scams — generate waves of consumer complaints months before they become an official complaint sub-category, draw regulatory enforcement, or attract news coverage. Compliance teams, regulators, and researchers currently rely on complaint data that is only useful in hindsight: it tells you what already happened, sorted into categories that already exist. There is no widely available tool that treats the complaint stream itself as an early-warning signal for harms that have not yet been named.

This gap persists for structural reasons. It requires genuinely temporal, streaming-aware modeling rather than a static classifier; it is hard to evaluate without inventing a rigorous backtesting protocol, since there is no off-the-shelf benchmark for "was this really emergent"; and it sits between two communities that rarely intersect — concept-drift and novelty-detection methodology (mostly developed and tested on fake-news or general text streams) and consumer-finance policy research (which reads complaint data but does not typically build drift-aware deep learning systems for it).

## 3. Goals and Objectives

### 3.1 Goals
- Detect emergent, previously unseen complaint patterns earlier than static classification or manual review would.
- Produce alerts that are interpretable — grounded in specific example narratives, not just an opaque anomaly score.
- Demonstrate the early-warning claim with credible, pre-defined retrospective validation, not just a plausible demo.
- Deliver a fully reproducible research artifact built only on public data, suitable for both a Deep Learning and an NLP course submission.
- **Maximize genuine deep learning technique coverage across the pipeline** — representation, clustering, and drift detection should each be implemented as a trained or pretrained neural network, since this project is being evaluated as a Deep Learning course deliverable.

### 3.2 Non-Goals
- Replacing human analyst judgment — the system is decision-support, not decision-making.
- Operating as a live, continuously running production system within this release.
- Making definitive causal claims about why a complaint pattern emerged.
- Supporting non-U.S. regulators or non-English complaint data in this release.

## 4. Target Users & Stakeholders

| Persona | Description | Primary Goal | Current Pain Point |
|---|---|---|---|
| **Compliance / RegTech analyst** | Works at a bank, fintech, or regulator, monitoring consumer complaint volumes | Spot a new risk pattern before it escalates into a formal issue or enforcement action | Manually reviews complaints sorted into known categories — a genuinely new pattern is invisible until it is officially recognized |
| **Consumer-protection policy researcher** | Studies emerging financial-harm trends for academic or public-policy purposes | Quantify how early a harm could plausibly have been detected from public complaint data | No existing tool systematically backtests "how early could we have known" against real historical cases |
| **Academic evaluator (professor / committee)** | Assesses the capstone for Deep Learning and NLP course credit | Verify a genuine, well-scoped deep learning contribution built on reproducible, public data | Needs a clearly documented architecture where every stage is defensibly "deep learning," not classical ML wearing a DL label |

## 5. User Stories

- As a **compliance analyst**, I want to see clusters of complaint narratives that are growing unusually fast, so that I can investigate a potential emerging harm before it spreads.
- As a **policy researcher**, I want to see how far in advance the system would have flagged a known past event (e.g., BNPL complaints), so that I can trust the system's signal before relying on it for a new, unlabeled case.
- As an **academic evaluator**, I want a fully documented, reproducible pipeline where each stage's neural architecture is explicit, so that I can verify the deep learning content and the technical claims independently.

## 6. Scope

| In Scope | Out of Scope (this release) |
|---|---|
| Ingesting and filtering the public CFPB Consumer Complaint Database | Real-time / continuously running production ingestion pipeline |
| Transformer-based embedding of complaint narratives | Automated regulatory alerting or notification integrations |
| Deep Embedded Clustering (DEC) over rolling time windows | Fusion with non-CFPB sources (news, social media, other regulators) |
| LSTM-Autoencoder-based drift / novelty scoring | Multi-language or non-U.S. regulator support |
| Retrospective backtesting against real historical cases | Definitive causal claims about the cause of any complaint pattern |
| A minimal report/dashboard of currently flagged clusters | Replacing human analyst judgment — system output is advisory only |

## 7. Functional Requirements

Prioritized using MoSCoW (Must / Should / Could / Won't have), scoped to a single academic semester.

| Priority | Requirement |
|---|---|
| Must have | CFPB data ingestion & preprocessing pipeline (filter to narrative-present complaints; build rolling time-windowed panels) |
| Must have | Pretrained Transformer sentence-embedding generation for complaint narratives |
| Must have | Deep autoencoder for nonlinear dimensionality reduction, pretraining Deep Embedded Clustering (DEC) |
| Must have | Deep Embedded Clustering (DEC) — neural clustering layer, jointly fine-tuned with the encoder, adapted for incremental/streaming time windows |
| Must have | LSTM-Autoencoder-based drift / emergence scoring layer over cluster-trajectory sequences |
| Must have | Backtesting framework validated against at least two real historical cases (e.g., Buy Now Pay Later, earned-wage-access) |
| Must have | Evaluation reporting: lead-time, cluster coherence, precision@k, false-alarm rate |
| Must have | Explicit documentation mapping every pipeline stage to its deep learning architecture (Section 9.2), for Deep Learning course evaluation |
| Should have | LLM-assisted automatic labeling of flagged clusters into a human-readable one-line summary |
| Should have | A simple visualization / report of currently emergent clusters |
| Should have | Sensitivity analysis across different window sizes and drift thresholds |
| Could have | A small temporal-Transformer autoencoder trained as a comparison against the LSTM-Autoencoder |
| Could have | Interactive exploration UI (filter by product, state, or company) |
| Could have | Exportable flagged-cluster report formatted for analyst hand-off |
| Won't have (this release) | Real-time production deployment, automated alerting integrations, multi-source fusion, non-U.S. regulator support |

## 8. Non-Functional Requirements

- **Deep learning coverage** — every core pipeline stage (representation, clustering, drift detection) must be a trained or pretrained neural network. Classical ML (e.g., k-means, HDBSCAN, logistic regression, statistical change-point tests) may be used only for lightweight, clearly-labeled baselines or as a one-time initialization heuristic (e.g., k-means to seed DEC's cluster centroids) — never as the primary technique for a core stage. Required for Deep Learning course credit.
- **Reproducibility** — the full pipeline must run on public data and open-source libraries only, with no paid API or proprietary dataset dependency for core results.
- **Interpretability** — every flagged cluster must be traceable back to specific example complaint narratives, not just a numeric score.
- **Timeliness** — the pipeline should be re-runnable on the most recently available CFPB data dump without redesign.
- **Feasibility** — the full scope must be completable within a 14-week semester, including evaluation and writeup.
- **Compute budget** — must run on standard or free-tier GPU compute (e.g., a university cluster or Colab-class GPU); DEC and LSTM-AE training are lightweight enough not to require large-scale infrastructure.

## 9. System Overview & Technical Approach

### 9.1 Pipeline Overview

At a high level, the pipeline moves complaint narratives from raw text through to a ranked list of emergent-cluster alerts:

| Stage | Technique | Output |
|---|---|---|
| 1. Ingestion | CFPB bulk CSV/JSON download, filtered to narrative-present complaints, split into rolling monthly/quarterly windows | Time-windowed complaint panels |
| 2. Representation | Pretrained Transformer sentence-encoder (Sentence-BERT / FinBERT) | Dense vector per complaint |
| 3. Deep Embedded Clustering | Deep autoencoder pretraining + joint encoder/centroid fine-tuning via KL-divergence self-training clustering loss (DEC), adapted incrementally per time window | Cluster assignments and trajectories over time |
| 4. Deep drift detection | LSTM Autoencoder (or temporal-Transformer autoencoder) trained on cluster-trajectory sequences; reconstruction error drives the emergence score | Emergence / drift score per cluster |
| 5. Labeling & reporting | Optional pretrained LLM, zero-shot prompted for cluster summarization | Human-readable alert with supporting narrative examples |

### 9.2 Deep Learning Architecture Summary (for course evaluation)

| # | Pipeline Role | Deep Learning Architecture | Trained or Pretrained? |
|---|---|---|---|
| 1 | Narrative representation | Transformer sentence-encoder (Sentence-BERT / FinBERT) | Pretrained |
| 2 | Nonlinear dimensionality reduction | Deep autoencoder (encoder-decoder) | Trained (on complaint corpus) |
| 3 | Clustering | Deep Embedded Clustering (DEC) — neural clustering layer with KL-divergence self-training loss, replacing classical clustering | Trained (jointly with encoder) |
| 4 | Drift / emergence detection | LSTM Autoencoder over cluster-trajectory time series (temporal-Transformer as an optional comparison) | Trained |
| 5 | Baseline sanity-check (Phase 2 only) | Small feedforward MLP classifier | Trained |
| 6 | Cluster labeling (optional) | Pretrained LLM, zero-shot inference | Pretrained, inference-only |

*Note: a one-time k-means pass is used only to **initialize** DEC's cluster centroids before neural fine-tuning begins — this is standard practice in the DEC literature (Xie, Girshick & Farhadi, 2016) and is not the clustering mechanism itself, which is the trained neural network.*

## 10. Data Requirements

- **Source:** CFPB Consumer Complaint Database — [consumerfinance.gov/data-research/consumer-complaints](https://www.consumerfinance.gov/data-research/consumer-complaints/) (direct CSV: [files.consumerfinance.gov/ccdb/complaints.csv.zip](https://files.consumerfinance.gov/ccdb/complaints.csv.zip)).
- **Coverage:** several million complaints since 2011, updated daily, with a large opt-in subset containing free-text narratives (filter: `has_narrative = true`).
- **Governance:** complaint narratives are already scrubbed of direct personal identifiers by the CFPB prior to publication; no additional PII-handling infrastructure is required for this release.
- **Preprocessing effort:** low (~10% of project effort) — the dataset is already clean and well-documented; the main work is filtering to narrative-present rows and constructing rolling time windows.

## 11. Success Metrics / KPIs

| Metric | What It Measures | Target for This Release |
|---|---|---|
| Lead-time (months) | How far in advance a backtested historical case (e.g., BNPL) is flagged before official recognition | Positive lead-time on ≥ 2 of the historical test cases |
| Cluster coherence (NPMI, silhouette) | Whether flagged clusters are semantically tight and interpretable | Coherence scores comparable to or better than static topic-model baselines |
| Precision@k | Of the top-k flagged "emergent" clusters, how many are genuine patterns | Documented and reported, with qualitative case review for top clusters |
| False-alarm rate | How often the system flags a cluster that never became a real pattern | Explicitly measured and reported — low enough that output is analyst-trustworthy |
| DEC clustering quality | Whether the neural clustering layer outperforms its own pre-fine-tuning (autoencoder + k-means) baseline | DEC should measurably improve on the k-means-on-latent-space baseline it started from |
| LSTM-AE reconstruction gap | Separation between reconstruction error on normal vs. known-emergent trajectories | Reconstruction error clearly higher on flagged historical cases than on stable clusters |
| Reproducibility | Whether the full pipeline runs end-to-end on public data with documented steps | 100% — no proprietary data or paid API dependency required to reproduce core results |

## 12. Milestones & Roadmap

| Phase | Weeks | Milestone / Deliverable |
|---|---|---|
| 1. Data & setup | 1–2 | CFPB data downloaded and filtered; rolling time-window structure defined; environment configured |
| 2. Representation learning | 3–4 | Transformer embedding pipeline built and validated; small MLP classifier as a sanity-check baseline |
| 3. Deep Embedded Clustering | 5–7 | Autoencoder pretrained; DEC fine-tuning implemented and adapted for incremental per-window training; cluster-quality metrics validated |
| 4. Deep drift detection | 8–9 | LSTM Autoencoder trained on cluster-trajectory sequences; reconstruction-error-based emergence scoring layer built |
| 5. Backtesting | 10–11 | Retrospective validation designed and executed against ≥ 2 real historical cases |
| 6. Reporting | 12–13 | Dashboard / report of currently emergent clusters; full metrics suite reported |
| 7. Writeup | 14 | Final report drafted, including an explicit deep learning architecture map; buffer for revisions; submission |

## 13. Risks, Assumptions & Dependencies

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Backtesting doesn't convincingly demonstrate lead-time value | Medium | High | Define test cases and success criteria before running the backtest, not after; use ≥ 2 independent historical cases, not one |
| False-alarm rate too high for practical trust | Medium | Medium | Tune clustering/drift thresholds; frame output explicitly as analyst-review triage, not autonomous action |
| Per-window data volume too small for stable neural training (DEC / LSTM-AE) | Medium | Medium | Pretrain on the full historical corpus first, then fine-tune incrementally per window with a small number of gradient steps (warm-start, not training from scratch); widen the window if a given period is too sparse |
| CFPB narrative field sparsity or future policy changes reduce data volume | Low | Medium | Document as a known limitation; design pipeline to degrade gracefully with lower narrative volume |
| Compute or time constraints within the semester | Medium | Medium | Use lightweight architectures (small autoencoder, single-layer LSTM); scope embeddings to a sampled subset if needed |
| Perceived novelty overlaps with existing static classification work | Low | High | Explicitly benchmark against classification/outcome-prediction baselines to show the discovery framing is distinct |

## 14. Open Questions

- What alert threshold best balances sensitivity against false alarms for a realistic analyst workflow?
- Should the system support company- or state-level drill-down, or remain aggregate-level only for this release?
- Is there value in a future extension to non-U.S. regulators (e.g., RBI, FCA) for a comparative, cross-market view?
- Is the temporal-Transformer autoencoder comparison (Section 7, Could Have) worth the extra time versus banking that effort toward a more thorough backtest?

## 15. Future Work (Explicitly Out of Scope for This Release)

- Fusing additional public signal sources (news headlines, FTC Consumer Sentinel data) for multi-source early warning.
- Packaging the system as a live, continuously updating dashboard for a compliance or risk team.
- Cross-country replication using an analogous complaint database (e.g., RBI's banking ombudsman data, UK FCA complaint data).
- Replacing the LSTM-Autoencoder with a full temporal-Transformer / Temporal Graph Network as the drift-detection backbone, if a longer research runway becomes available.

## 16. Appendix

**Related deliverables:** `DL_NLP_Finance_Project_Proposals` (full project comparison across three candidate capstones), `DL_NLP_Finance_One_Page_Briefs` (one-page summary per project), `SentinelFin_Literature_Survey` and `SentinelFin_Literature_Survey_Round2` (34 verified papers, 2019–2026, supporting the research-gap case in Section 2), `SentinelFin_Master_Prompt` (phase-wise implementation guide for building this system with an AI coding assistant, updated to match this architecture).

**Key references for the deep learning components:** Xie, J., Girshick, R., & Farhadi, A. (2016). *Unsupervised Deep Embedding for Clustering Analysis* (the DEC method used in Section 9.2, Stage 3). Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks* (representation stage). The literature survey documents' SCStory entry (Yoon, Meng, Lee & Han, 2023) is the methodological basis for the continual/incremental fine-tuning approach used to adapt DEC to streaming time windows.
