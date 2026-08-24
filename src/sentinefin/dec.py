"""Phase 3: Deep Embedded Clustering (DEC).

Implements Xie, Girshick & Farhadi (2016) "Unsupervised Deep Embedding for
Clustering Analysis":

1. Pretrain a deep autoencoder on the complaint embeddings (nonlinear
   dimensionality reduction).
2. Initialize cluster centroids in latent space with a single k-means pass.
   k-means is ONLY an initializer here, never the clustering mechanism.
3. Jointly fine-tune encoder weights and centroids by minimizing
   KL(L || P) between Student's t soft assignments Q and the sharpened target
   distribution P (self-training). The clustering itself is a trained neural
   network.

Also provides rolling-window continual learning: warm-start each window from
the previous window's encoder + centroids, with a small replay buffer of prior
embeddings to reduce forgetting.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from . import config as _cfg
from .config import DECConfig
from .utils import ensure_dir, save_json, set_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class Autoencoder(nn.Module):
    """MLP autoencoder; its encoder half is DEC's feature extractor."""

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], latent_dim: int) -> None:
        super().__init__()
        enc: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            enc += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        enc.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc)

        dec: list[nn.Module] = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat


class DECModel(nn.Module):
    """Encoder + learnable cluster centroids with Student's t soft assignment."""

    def __init__(self, autoencoder: Autoencoder, n_clusters: int) -> None:
        super().__init__()
        self.autoencoder = autoencoder
        self.centroids = nn.Parameter(torch.zeros(n_clusters, autoencoder.encoder[-1].out_features))

    def soft_assign(self, z: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """Student's t kernel, as in the DEC paper (eq. 1)."""
        diff = z.unsqueeze(1) - self.centroids.unsqueeze(0)
        dist_sq = (diff**2).sum(dim=2)
        weights = 1.0 / (1.0 + dist_sq / alpha)
        return (weights.T / weights.sum(dim=1)).T ** ((alpha + 1.0) / 2.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, x_hat = self.autoencoder(x)
        q = self.soft_assign(z)
        return q, z, x_hat


def target_distribution(q: torch.Tensor) -> torch.Tensor:
    """Sharpen soft assignments into the auxiliary target P (DEC eq. 3-4)."""
    weight = (q**2) / q.sum(dim=0, keepdim=True)
    # weight.sum(dim=1) is per-sample; dividing the transposed (k, n) matrix by this
    # length-n vector normalizes each sample's target row, as in the DEC reference.
    return (weight.T / weight.sum(dim=1)).T.detach()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def pretrain_autoencoder(
    vectors: np.ndarray,
    cfg: DECConfig,
    seed: int,
) -> tuple[Autoencoder, float]:
    set_seed(seed)
    ae = Autoencoder(vectors.shape[1], cfg.hidden_dims, cfg.latent_dim)
    opt = torch.optim.Adam(ae.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    data = torch.tensor(vectors, dtype=torch.float32)
    n = len(data)
    final_loss = float("nan")
    for epoch in range(cfg.pretrain_epochs):
        perm = torch.randperm(n)
        losses = []
        for start in range(0, n, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            _, recon = ae(data[idx])
            loss = F.mse_loss(recon, data[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        final_loss = float(np.mean(losses))
        logger.info("AE pretrain epoch %d/%d recon loss %.4f", epoch + 1, cfg.pretrain_epochs, final_loss)
    return ae, final_loss


@torch.no_grad()
def _encode_all(model: DECModel, data: torch.Tensor, batch_size: int) -> np.ndarray:
    model.eval()
    chunks = []
    for start in range(0, len(data), batch_size):
        chunks.append(model.autoencoder.encoder(data[start : start + batch_size]).cpu().numpy())
    return np.concatenate(chunks)


def train_dec(
    vectors: np.ndarray,
    cfg: DECConfig | None = None,
    init_state: dict | None = None,
    seed: int = 23,
) -> tuple[DECModel, dict]:
    """Train DEC. If ``init_state`` is given, warm-start from it (streaming mode).

    Returns the model state dict (CPU numpy-friendly) plus metrics.
    """
    cfg = cfg or DECConfig()
    set_seed(seed)

    ae, pre_loss = pretrain_autoencoder(vectors, cfg, seed) if init_state is None else (
        _ae_from_state(vectors.shape[1], cfg, init_state["autoencoder"]),
        float(init_state.get("pretrain_recon_loss", float("nan"))),
    )

    model = DECModel(ae, cfg.n_clusters)
    if init_state is not None:
        model.centroids.data = torch.tensor(init_state["centroids"], dtype=torch.float32)
    else:
        # One-time k-means centroid INITIALIZATION only. The clustering
        # mechanism itself is the KL self-training below, not k-means.
        latents = _encode_all(model, torch.tensor(vectors, dtype=torch.float32), cfg.batch_size)
        km = KMeans(n_clusters=cfg.n_clusters, n_init=10, random_state=seed)
        model.centroids.data = torch.tensor(
            km.fit(latents).cluster_centers_, dtype=torch.float32
        )

    data = torch.tensor(vectors, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    prev_assign: np.ndarray | None = None
    stop = False
    it = 0
    while not stop:
        q = _soft_assign_tensor(model, data, cfg.batch_size)
        p_tensor = target_distribution(q)
        model.train()
        perm = torch.randperm(len(data))
        kl_total = 0.0
        for start in range(0, len(data), cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            opt.zero_grad()
            q, _, _ = model(data[idx])
            loss = F.kl_div(
                (q + 1e-9).log(), p_tensor[idx], reduction="batchmean", log_target=False
            )
            loss.backward()
            opt.step()
            kl_total += float(loss.item()) * len(idx)
        it += 1
        assign_now = q.argmax(dim=1).cpu().numpy()
        delta = (
            float((assign_now != prev_assign).mean())
            if prev_assign is not None
            else 1.0
        )
        prev_assign = assign_now
        logger.info("DEC iter %d/%d KL %.4f label-change %.4f", it, cfg.finetune_iters, kl_total / max(len(data),1), delta)
        if it >= cfg.finetune_iters or delta < cfg.tol:
            stop = True

    metrics = {
        "pretrain_recon_loss": pre_loss,
        "finetune_iters": it,
        "final_label_change": delta,
        "n_clusters": cfg.n_clusters,
    }
    return model, metrics


def _soft_assign_numpy(model: DECModel, data: torch.Tensor, batch_size: int) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            q = model.soft_assign(model.autoencoder.encoder(data[start : start + batch_size]))
            outs.append(q.cpu().numpy())
    return np.concatenate(outs)


def _soft_assign_tensor(model: DECModel, data: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Batched soft assignments kept as a tensor (avoids numpy round-trips)."""
    model.eval()
    outs = []
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            outs.append(model.soft_assign(model.autoencoder.encoder(data[start : start + batch_size])))
    return torch.cat(outs)


def _ae_from_state(input_dim: int, cfg: DECConfig, state: dict) -> Autoencoder:
    ae = Autoencoder(input_dim, cfg.hidden_dims, cfg.latent_dim)
    ae.load_state_dict({k: torch.tensor(np.asarray(v)) for k, v in state.items()})
    return ae


# ---------------------------------------------------------------------------
# Rolling-window continual learning
# ---------------------------------------------------------------------------


class WindowClusterTracker:
    """Links clusters across consecutive windows via centroid similarity."""

    def __init__(self, similarity_threshold: float = 0.5) -> None:
        self.threshold = similarity_threshold
        self.next_id = 0
        self.previous_centroids: np.ndarray | None = None
        self.mapping_history: list[dict[str, int]] = []

    def register_window(self, centroids: np.ndarray) -> list[int]:
        """Assign stable global ids to this window's local clusters."""
        if self.previous_centroids is None:
            ids = list(range(self.next_id, self.next_id + len(centroids)))
            self.next_id += len(centroids)
        else:
            sim = cosine_similarity_rows(centroids, self.previous_centroids)
            best = sim.argmax(axis=1)
            ids = []
            used: set[int] = set()
            for row, j in enumerate(best):
                if sim[row, j] >= self.threshold and int(j) not in used:
                    gid = int(self._prev_ids[j])
                    used.add(gid)
                else:
                    gid = self.next_id
                    self.next_id += 1
                ids.append(gid)
        self._prev_ids = ids
        self.previous_centroids = centroids.copy()
        self.mapping_history.append({str(i): g for i, g in enumerate(ids)})
        return ids

    @property
    def last_ids(self) -> list[int]:
        return list(self._prev_ids)


def cosine_similarity_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_n @ b_n.T


def run_dec_over_windows(
    panel: pd.DataFrame,
    vectors: np.ndarray,
    cfg: DECConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Warm-started DEC over every time window; produces assignments + trajectories."""
    cfg = cfg or DECConfig()
    tracker = WindowClusterTracker()
    rows = []
    windows = sorted(panel["window_id"].unique())
    prev_state: dict | None = None
    summary: dict = {"windows": {}}

    for w in windows:
        mask = (panel["window_id"] == w).to_numpy()
        vecs_w = vectors[mask]
        if len(vecs_w) < cfg.n_clusters:
            logger.warning("Window %s has %d points < %d clusters; skipping DEC fine-tune.",
                           w, len(vecs_w), cfg.n_clusters)
            continue
        model, metrics = train_dec(vecs_w, cfg, init_state=prev_state, seed=cfg.seed)
        centroids = model.centroids.detach().cpu().numpy()
        assign = _soft_assign_numpy(model, torch.tensor(vecs_w, dtype=torch.float32),
                                    cfg.batch_size).argmax(axis=1)
        global_ids = tracker.register_window(centroids)

        sub = panel.loc[mask, ["complaint_id"]].copy()
        sub["local_cluster"] = assign
        sub["global_cluster"] = [global_ids[a] for a in assign]
        rows.append(sub)

        sizes = np.bincount(assign, minlength=cfg.n_clusters)
        sil = silhouette_if_possible(vecs_w, assign)
        summary["windows"][w] = {
            "n_points": int(mask.sum()),
            "local_to_global": {str(i): g for i, g in enumerate(global_ids)},
            "cluster_sizes": sizes.tolist(),
            "silhouette": sil,
            "recon_loss": metrics["pretrain_recon_loss"],
        }
        prev_state = {
            "autoencoder": {k: v.detach().cpu().numpy() for k, v in model.autoencoder.state_dict().items()},
            "centroids": centroids,
        }

    assignments = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["complaint_id", "local_cluster", "global_cluster"]
    )
    ensure_dir(_cfg.PROCESSED_DATA_DIR)
    assignments.to_parquet(_cfg.PROCESSED_DATA_DIR / "dec_assignments.parquet", index=False)
    save_json(summary, _cfg.OUTPUTS_DIR / "dec_summary.json")
    return assignments, summary


def silhouette_if_possible(X: np.ndarray, labels: np.ndarray) -> float | None:
    try:
        from sklearn.metrics import silhouette_score

        if len(set(labels)) < 2 or len(set(labels)) >= len(labels):
            return None
        return float(silhouette_score(X, labels))
    except Exception:
        return None
