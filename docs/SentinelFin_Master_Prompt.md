# SentinelFin — Master Prompt (Phase-Wise Build Guide)
### v1.1 — revised so every core pipeline stage is a trained or pretrained neural network (Deep Learning course submission)

This document turns the SentinelFin PRD into a set of ready-to-use prompts for an AI coding assistant (Claude Code, Cursor, or similar). It mirrors the 7-phase, 14-week roadmap from the PRD's Section 12. **What changed in this revision:** classical clustering (HDBSCAN) is replaced with **Deep Embedded Clustering (DEC)**, and statistical drift detection is replaced with an **LSTM Autoencoder** — so representation, clustering, and drift detection are now all trained or pretrained neural networks. See PRD Section 9.2 for the full architecture map.

## How to use this

1. Put `SentinelFin_PRD.md` in your project root so your assistant can read it for full context.
2. Paste the **Global Context** block below once, at the start of your coding session (or into a `CLAUDE.md` / project-instructions file if your tool supports one).
3. Work through the phases **in order**. Paste one phase's prompt, let the assistant finish, review the output against that phase's acceptance criteria, and only then move to the next.
4. Don't let the assistant skip ahead — each phase's prompt explicitly says not to, but it's worth enforcing yourself too. The backtesting phase (5) in particular depends on everything before it being solid.
5. If you're short on time, Phases 1–5 are the actual research contribution; Phase 6 can be trimmed to something very minimal without weakening the project.
6. If your professor pushes back on any step, Section 9.2 of the PRD is the one-page answer to "where's the deep learning" — every core stage maps to a named neural architecture there.

---

## Global Context
*(Paste this once at the start of your session, before Phase 1.)*

```
You are helping me build SentinelFin, a research prototype for my Deep Learning 
and NLP capstone (MBA Finance program). Full requirements are in 
SentinelFin_PRD.md — read it before starting each phase below.

Project in one line: continuously monitor embeddings of CFPB consumer-complaint 
narratives to detect emergent, semantically coherent harm-pattern clusters before 
they are officially recognized, and validate the claim with retrospective 
backtesting against real historical cases.

Constraints that apply to every phase:
- This project is being submitted for Deep Learning course credit, so every 
  core pipeline stage (representation, clustering, drift detection) must be a 
  trained or pretrained neural network. Classical ML (k-means, HDBSCAN, 
  logistic regression, statistical change-point tests) may only be used for 
  lightweight, clearly-labeled baseline comparisons, or as a one-time 
  initialization heuristic (e.g., k-means to seed cluster centroids before 
  neural fine-tuning) — never as the primary technique for a core stage.
- Python only. Use PyTorch (or TensorFlow/Keras) for all trained neural 
  components, plus sentence-transformers for pretrained embeddings. No paid 
  APIs required for core results (an optional LLM-labeling step may use one, 
  but the pipeline must degrade gracefully without it).
- Public data only — the CFPB Consumer Complaint Database 
  (https://files.consumerfinance.gov/ccdb/complaints.csv.zip). Never fabricate 
  or synthesize data to fill gaps.
- Reproducibility matters more than cleverness: write clear, commented, modular 
  code; save intermediate artifacts (embeddings, model weights, cluster 
  assignments, metrics) to disk so later phases don't have to recompute from 
  scratch.
- After each phase, write a short markdown summary of what was built, what the 
  results were, and any deviations from the plan, so I can review before we 
  move to the next phase.
- Do not skip ahead to later phases even if it seems efficient — I want to 
  review and approve each phase's output first.
```

---

## Phase 1 — Data & Setup (Weeks 1–2)

**Goal:** Get a clean, time-windowed panel of CFPB complaint narratives ready for modeling.

**Prompt:**
```
Set up the project environment and data pipeline for SentinelFin.

1. Create a project structure: /data (raw + processed), /src (pipeline code), 
   /notebooks (exploration), /outputs (metrics, figures), /reports.
2. Write a script to download the CFPB Consumer Complaint Database 
   (https://files.consumerfinance.gov/ccdb/complaints.csv.zip), or load it if 
   already downloaded locally.
3. Filter to complaints where a narrative is present (consumer_complaint_narrative 
   is non-null / has_narrative=true if using the API).
4. Parse dates and build rolling time-window panels (start with monthly windows; 
   make the window size configurable).
5. Do basic exploratory data analysis: volume over time, narrative length 
   distribution, top products/issues, narrative-availability rate over time. 
   Save 3-5 summary plots to /outputs.
6. Document any data quality issues you find (e.g., periods of missing 
   narratives, schema changes) in a short DATA_NOTES.md.

Deliverable: a reproducible script/notebook that takes the raw CFPB download and 
produces a clean, time-windowed complaint panel saved to /data/processed, plus 
DATA_NOTES.md and the EDA plots.
```

**Acceptance criteria:**
- [ ] Raw data downloads/loads without manual intervention
- [ ] Narrative-present filter applied and documented (what % of complaints have narratives, and how that's changed over time)
- [ ] Time-windowed panel structure is configurable (window size as a parameter)
- [ ] EDA plots and DATA_NOTES.md exist and have been reviewed

---

## Phase 2 — Representation Learning (Weeks 3–4)

**Goal:** Turn complaint narratives into embeddings using a pretrained Transformer, and build a small neural baseline to sanity-check against.

**Prompt:**
```
Using the time-windowed complaint panel from Phase 1 (/data/processed), build 
the embedding pipeline for SentinelFin.

1. Choose and justify a pretrained Transformer sentence-encoder (e.g., a 
   sentence-transformers model such as all-MiniLM-L6-v2 for speed, or a 
   FinBERT-style model for finance-domain adaptation — implement both and 
   compare qualitatively on a handful of example narratives).
2. Batch-embed all narrative text; save embeddings to disk keyed by complaint 
   ID (use an efficient format — e.g., a memory-mapped numpy array or parquet 
   with embeddings as list columns).
3. As a sanity-check baseline (not the main contribution), train a small 
   feedforward MLP classifier (a real neural network, not logistic 
   regression — every reported component in this project should be a deep 
   learning technique) on the embeddings to predict complaint product/issue 
   category, and report accuracy/F1. This confirms the embeddings carry 
   meaningful signal before we build the deep clustering and drift-detection 
   components on top of them.

Note: the dimensionality-reduction autoencoder is NOT built here — it belongs 
to Phase 3, where it is pretrained specifically as the first step of Deep 
Embedded Clustering.

Deliverable: an embedding pipeline (script/module) that can be re-run on new 
data, saved embeddings for the full panel, and a short report comparing the 
two embedding model choices plus the baseline MLP classifier's performance.
```

**Acceptance criteria:**
- [ ] Embeddings generated for the full narrative-present dataset and saved to disk
- [ ] Baseline classifier is a neural MLP, not logistic regression or another classical model
- [ ] Baseline classifier accuracy/F1 reported (this is NOT the main contribution — don't over-invest here)
- [ ] Pipeline is re-runnable on new/updated data without code changes

---

## Phase 3 — Deep Embedded Clustering (Weeks 5–7)

**Goal:** Replace classical clustering with an end-to-end trainable neural clustering model (Deep Embedded Clustering, DEC), adapted to run incrementally over the rolling time windows.

**Prompt:**
```
Using the embeddings from Phase 2, implement Deep Embedded Clustering (DEC) 
for SentinelFin, following the approach of Xie, Girshick & Farhadi (2016), 
adapted for streaming data.

1. Pretrain a deep autoencoder (encoder-decoder, MLP or small Transformer) on 
   the complaint embeddings to learn a compact latent representation, 
   minimizing reconstruction loss. This is the "deep autoencoder for 
   nonlinear dimensionality reduction" stage from the PRD.
2. Initialize cluster centroids in the latent space. A one-time k-means pass 
   is standard practice here purely for initialization — document clearly 
   that this is NOT the clustering mechanism itself, just a starting point 
   for the neural fine-tuning in the next step.
3. Jointly fine-tune the encoder and the cluster centroids using DEC's 
   self-training clustering loss: compute soft cluster assignments via a 
   Student's t-distribution over latent-space distances, sharpen this into an 
   auxiliary target distribution, and minimize the KL divergence between them. 
   This is what makes the clustering itself a trained neural network rather 
   than a classical algorithm.
4. Adapt this for the rolling time-window setting: for each new window, 
   warm-start from the previous window's encoder weights and centroids and 
   fine-tune for a small number of steps on the new window's data (continual 
   learning, not training from scratch each time). You can draw on the 
   memory-replay style continual-learning approach described in the SCStory 
   paper (Yoon, Meng, Lee & Han, 2023 — see the literature survey) as 
   inspiration. Track cluster identity across windows via centroid similarity 
   so we can build cluster trajectories.
5. Evaluate cluster quality: silhouette score, Davies-Bouldin index, and topic 
   coherence (NPMI over top terms per cluster via c-TF-IDF). Also report 
   whether DEC's fine-tuned clusters improve on the pre-fine-tuning 
   (autoencoder + k-means only) baseline — this demonstrates the neural 
   fine-tuning step is actually adding value.
6. Visualize a handful of clusters over time (UMAP/t-SNE projection, colored 
   by cluster, faceted by time window) to sanity-check coherence.

Deliverable: a DEC module (autoencoder pretraining + incremental neural 
fine-tuning code), the trained encoder + centroids per window, cross-window 
cluster trajectories, a cluster-quality metrics report (including the 
DEC-vs-pre-fine-tuning comparison), and exploratory visualizations. In your 
write-up, explicitly state that clustering is a trained neural network (DEC), 
not a classical algorithm.
```

**Acceptance criteria:**
- [ ] Autoencoder pretrained and saved; reconstruction loss reported
- [ ] Clustering mechanism is DEC (trained neural network via KL-divergence self-training loss), with k-means used only as a one-time centroid initializer
- [ ] Incremental/streaming adaptation implemented (warm-start + fine-tune per window, not full retraining)
- [ ] Cluster trajectories link related clusters across consecutive windows
- [ ] Cluster-quality metrics computed, including a comparison showing DEC improves on its own pre-fine-tuning baseline
- [ ] At least 3 example clusters visually/qualitatively reviewed and described in plain English

---

## Phase 4 — Deep Drift Detection (Weeks 8–9)

**Goal:** Score cluster trajectories for "emergence" using a trained neural sequence model, not a statistical test.

**Prompt:**
```
Using the cluster trajectories from Phase 3, implement an LSTM Autoencoder as 
the drift/emergence scoring layer of SentinelFin.

1. Represent each cluster's history as a time series: per-window centroid 
   vector, complaint volume, and any other useful per-window statistics, 
   concatenated into a feature vector per time step.
2. Train an LSTM Autoencoder (encoder LSTM -> latent vector -> decoder LSTM) 
   on trajectories from "settled," non-emergent clusters, so it learns what a 
   normal trajectory pattern looks like.
3. At inference, run every cluster trajectory (including brand-new, short 
   ones) through the trained LSTM-AE and use reconstruction error as the core 
   emergence signal: a trajectory the model reconstructs poorly is behaving 
   in a way the model hasn't learned to expect, i.e., a candidate emergent 
   pattern.
4. Report the emergence score with interpretable sub-components where 
   possible (e.g., how much of the reconstruction error comes from the 
   volume dimension vs. the centroid-movement dimensions) so the output isn't 
   just an opaque number.
5. Add a simple thresholding/alerting mechanism (configurable) that flags 
   clusters above an error threshold as "candidate alerts."
6. Write unit tests using synthetic toy trajectories (e.g., a hand-constructed 
   trajectory that should obviously reconstruct poorly, and one that 
   obviously shouldn't) to validate the logic before running on real data.
7. Time permitting (this is a "Could Have," don't let it delay Phase 5): train 
   a small temporal-Transformer autoencoder as a comparison model and report 
   which architecture reconstructs normal trajectories better and separates 
   known-emergent cases more clearly.

Deliverable: an LSTM-AE drift-scoring module, its output applied to the full 
real cluster-trajectory dataset (a ranked list of candidate emergent clusters 
per window, driven by reconstruction error), and the unit tests.
```

**Acceptance criteria:**
- [ ] Drift/emergence detection is a trained neural sequence model (LSTM Autoencoder), not a statistical or rule-based test
- [ ] Emergence score has interpretable sub-components (not just a black-box number)
- [ ] Threshold-based alerting is configurable
- [ ] Unit tests pass on synthetic examples
- [ ] Ranked candidate-emergent-cluster output produced for the real dataset

---

## Phase 5 — Backtesting (Weeks 10–11)

**Goal:** Prove the early-warning claim with rigorous retrospective validation. **This is the phase that makes or breaks the research contribution — be rigorous.**

**Prompt:**
```
Design and run the backtesting protocol for SentinelFin.

1. Before looking at any results, pick at least 2 real historical cases where 
   a consumer-finance harm pattern is known to have emerged and later became 
   officially recognized — for example, Buy Now Pay Later complaints, 
   earned-wage-access complaints, or virtual-currency/crypto-linked 
   complaints. For each, research and document: (a) roughly when the 
   complaint pattern first appears in the CFPB data, (b) when it was 
   officially recognized (e.g., became a named sub-issue category, was the 
   subject of a CFPB report, or drew significant news coverage) — cite your 
   sources for these dates.
2. Define success criteria BEFORE running the backtest: e.g., "the system 
   counts as succeeding on a case if it flags a cluster matching that harm 
   pattern at least N months before the official-recognition date, with the 
   flagged cluster's example narratives clearly describing the pattern."
3. Truncate the full pipeline's input data to end before each case's 
   official-recognition date, re-run Phases 1-4 (including re-running DEC and 
   the LSTM-AE on the truncated data — do not reuse models trained on the 
   full dataset, or you will leak future information into the test) and 
   check whether the system would have flagged the relevant cluster, and how 
   many months early.
4. Report results honestly, including any cases where the system did NOT 
   succeed — this is a research report, not a sales pitch. Discuss why a 
   failure happened if one occurs.
5. Also report the false-alarm rate: of all clusters the system flagged as 
   "candidate emergent" across the full historical dataset, roughly what 
   fraction do NOT correspond to any known real pattern (acknowledge this is 
   necessarily an approximate, manually-reviewed estimate).

Deliverable: a backtesting report (markdown or notebook) documenting the 
pre-registered test cases and success criteria, the lead-time results for 
each case, and an honest discussion of false-alarm rate and any failure cases.
```

**Acceptance criteria:**
- [ ] Test cases and success criteria were defined and written down BEFORE seeing results
- [ ] At least 2 historical cases tested, with cited sources for recognition dates
- [ ] DEC and the LSTM-AE were re-trained/re-fine-tuned on truncated data for each case, not reused from the full-data run (no data leakage)
- [ ] Lead-time reported in months for each case (positive, or negative/failed — reported honestly either way)
- [ ] False-alarm rate estimated and reported

---

## Phase 6 — Reporting (Weeks 12–13)

**Goal:** Build a minimal, clear dashboard/report of currently emergent clusters, using the most recent available data.

**Prompt:**
```
Build the reporting layer of SentinelFin.

1. Run the full pipeline (Phases 1-4: ingestion, embeddings, DEC, LSTM-AE) on 
   the most recent available CFPB data to produce a current list of candidate 
   emergent clusters.
2. Optionally, add LLM-assisted labeling: for each flagged cluster, generate 
   a one-line, human-readable summary label from a sample of its narratives 
   (if using an API-based LLM, make sure the pipeline still works — with 
   reduced polish — if no API key is available, e.g., falling back to 
   top-TF-IDF-term labeling).
3. Build a simple, clear report — this can be a static HTML page, a Jupyter 
   notebook rendered to HTML, or a lightweight Streamlit/Gradio app — showing: 
   the ranked list of current candidate emergent clusters, each with its 
   LSTM-AE reconstruction-error breakdown, example narratives, and 
   volume-over-time chart.
4. Keep this genuinely minimal — the PRD explicitly scopes an interactive UI 
   as "Could Have," not "Must Have." A clean static report is a complete 
   deliverable; don't over-invest in UI polish at the expense of Phase 5's 
   rigor.

Deliverable: a report/dashboard artifact (HTML, notebook, or lightweight app) 
showing the current ranked candidate emergent clusters with supporting evidence.
```

**Acceptance criteria:**
- [ ] Report shows ranked current candidate clusters with interpretable scores
- [ ] Each flagged cluster has example narratives a human can actually read to judge it
- [ ] Pipeline degrades gracefully if the optional LLM-labeling step is unavailable

---

## Phase 7 — Writeup (Week 14)

**Goal:** Turn the working system and results into the final report/paper draft.

**Prompt:**
```
Help me draft the final report for SentinelFin, structured for both my Deep 
Learning and NLP course submissions.

1. Pull together: the problem statement and research gap (from the PRD and 
   literature survey documents), the methodology (Phases 1-4), the 
   backtesting results (Phase 5) as the central evidence for the research 
   claim, and the current-state report (Phase 6) as a demonstration.
2. Write it in standard research-report structure: Abstract, Introduction & 
   Related Work (cite the literature survey documents), Methodology, 
   Experiments & Results, Discussion & Limitations, Conclusion & Future Work.
3. In the Methodology section, include a clear architecture map (a table or 
   diagram) showing every pipeline stage and its deep learning technique — 
   Transformer embeddings, deep autoencoder, Deep Embedded Clustering, and 
   LSTM Autoencoder — mirroring PRD Section 9.2. This is the single most 
   important section for Deep Learning course evaluation, so make it 
   unambiguous.
4. Be explicit and honest about limitations — reviewers and professors 
   respond well to a clear-eyed limitations section, not overclaiming.
5. Include the key figures: the cluster-quality metrics (including the DEC 
   vs. pre-fine-tuning comparison), the backtesting lead-time results, and at 
   least one example of a currently-flagged cluster with its supporting 
   narratives.

Deliverable: a complete report draft (markdown, ready to convert to your 
submission format) plus a short one-paragraph abstract suitable for a course 
cover page.
```

**Acceptance criteria:**
- [ ] All sections present and internally consistent with actual results (not aspirational claims)
- [ ] Methodology section includes an explicit, unambiguous deep learning architecture map
- [ ] Limitations section is honest, not just a formality
- [ ] Ready for a final read-through/edit pass before submission
