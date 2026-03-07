"""
GraphDARTS: Graph-based Differentiable Architecture Search for 
Acoustic Emission Signal Clustering in Structural Health Monitoring.

This script implements the full pipeline:
  1. Load pre-extracted wavelet features from the ORION-AE dataset
  2. Build a low-rank similarity graph (LRR-based adjacency)
  3. Perform graph diffusion to enrich node embeddings
  4. Train a DARTS-based MLP (GraphDARTS) with MCR² loss
  5. Project embeddings via truncated SVD (99% energy retention)
  6. Cluster with Time-Series K-Means and evaluate

Dataset: ORION-AE benchmark (campaigns B–F)
Preprocessing: db45 wavelet, level-14, sqtwolog thresholding,
               frequency band 30–1100 Hz (Kharrat et al. / Ramasso et al.)

Author : Dr. Oualid Laiadi et al.
Affiliation: CRTI, Algiers, Algeria
"""

# ============================================================
# Standard library & third-party imports
# ============================================================
import os
import time
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import psutil
import scipy.sparse as sp

from scipy.io import loadmat
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
)
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance


# ============================================================
# Reproducibility helper
# ============================================================

def set_seed(seed: int = 42) -> None:
    """Fix all random-number generators for reproducible results."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


# ============================================================
# Feature normalisation utilities
# ============================================================

def center_datas(datas: torch.Tensor) -> torch.Tensor:
    """
    Zero-mean and unit-norm normalisation per sample.

    Each row is shifted by its own mean, then divided by its L2 norm,
    so that all embeddings lie on the unit hypersphere.
    """
    datas = datas - datas.mean(1, keepdim=True)
    datas = datas / torch.norm(datas, dim=2, keepdim=True)
    return datas


def scale_each_unitary(datas: torch.Tensor, p: int = 2) -> torch.Tensor:
    """
    Normalise each sample to unit Lp-norm.

    Parameters
    ----------
    datas : torch.Tensor  – shape (batch, n_samples, n_features)
    p     : int           – norm order (default 2 → Euclidean)
    """
    norms = datas.norm(dim=p, keepdim=True)
    return datas / norms


def normalize_matrix(W: torch.Tensor) -> torch.Tensor:
    """
    Symmetric Laplacian-style normalisation: D^{-1/2} W D^{-1/2}.

    The diagonal is zeroed first so self-loops do not bias the degree.
    A small ε = 1e-4 is added for numerical stability.
    """
    W.fill_diagonal_(0)
    isqrt_diag = 1.0 / torch.sqrt(1e-4 + torch.sum(W, dim=-1, keepdim=True))
    W = W * isqrt_diag * isqrt_diag.T
    return W


# ============================================================
# Low-rank similarity graph construction
# ============================================================

def compute_W_gram(ndatas: torch.Tensor, lowrank: int = 15) -> torch.Tensor:
    """
    Build a low-rank similarity matrix via the Gram matrix.

    Steps
    -----
    1. Scale input to unit norm.
    2. Compute the Gram matrix G = X X^T.
    3. Eigendecompose G and keep the top-`lowrank` eigenvectors.
    4. Reconstruct a low-rank positive semi-definite W = V Λ^{1/2} (V Λ^{1/2})^T.
    5. Apply symmetric normalisation.

    Parameters
    ----------
    ndatas  : torch.Tensor – raw feature matrix
    lowrank : int          – number of dominant eigenvectors to retain

    Returns
    -------
    W_norm : torch.Tensor – normalised (n × n) similarity matrix
    """
    ndatas = torch.tensor(ndatas, dtype=torch.float32)

    # Ensure 2-D layout (n_samples × n_features)
    if ndatas.dim() == 1:
        ndatas = ndatas.unsqueeze(0)
    elif ndatas.dim() == 3:
        ndatas = ndatas.squeeze(0)

    ndatas = scale_each_unitary(ndatas.unsqueeze(0)).squeeze(0)

    if ndatas.dim() != 2:
        raise ValueError(
            f"Expected 2-D tensor after processing, got shape {ndatas.shape}"
        )

    # Gram matrix and low-rank reconstruction
    gram = torch.mm(ndatas, ndatas.T)
    eigenvals, eigenvecs = torch.linalg.eigh(gram)

    top_k = min(lowrank, eigenvals.shape[0])
    eigenvecs = eigenvecs[:, -top_k:]
    eigenvals = eigenvals[-top_k:]

    W = torch.mm(eigenvecs * eigenvals.sqrt(), eigenvecs.T)
    W = torch.abs(W)

    return normalize_matrix(W)


def build_graph(features: np.ndarray, lowrank: int = 15):
    """
    Build the graph adjacency matrices used during training.

    Returns a pair (W1, W) where:
      - W1 is a normalised all-ones matrix (uniform adjacency / graph Laplacian term)
      - W  is the low-rank feature-based similarity matrix

    The two matrices are used in the MCR² loss:
        L = log|I + α/m · Z^T Z| – log|I + α/m · Z^T P Z|
    where P comes from W1 and Z = encoder output.
    """
    features_t = torch.tensor(features, dtype=torch.float32)
    features_t = scale_each_unitary(features_t.unsqueeze(0))

    W = compute_W_gram(features_t).unsqueeze(0)

    # Uniform adjacency (used as the compressive term)
    W1 = torch.ones_like(W)
    for i in range(W1.shape[0]):
        W1[i].fill_diagonal_(0)
    isqrt_diag = 1.0 / torch.sqrt(1e-4 + torch.sum(W1, dim=-1, keepdim=True))
    W1 = W1 * isqrt_diag * torch.transpose(isqrt_diag, dim0=2, dim1=1)

    return W1.squeeze(0), W.squeeze(0)


# ============================================================
# Graph diffusion embedding
# ============================================================

def graph_diffusion(feature: torch.Tensor,
                    adj: torch.Tensor,
                    K: int = 8) -> torch.Tensor:
    """
    Augment node features through K steps of graph diffusion.

    At each step k, the feature matrix is propagated as:
        X^{(k)} = A · X^{(k-1)}
    and all intermediate representations are averaged:
        emb = (1/K) Σ_{k=1}^{K} X^{(k)}

    This is equivalent to a truncated graph polynomial filter and
    captures multi-hop neighbourhood information without learning
    additional parameters.

    Parameters
    ----------
    feature : torch.Tensor – node feature matrix  (n × d)
    adj     : torch.Tensor – normalised adjacency  (n × n)
    K       : int          – number of diffusion steps

    Returns
    -------
    emb : torch.Tensor – enriched node embeddings (n × d)
    """
    emb = feature.clone()
    x = feature
    for _ in range(K):
        x = torch.mm(adj, x)
        emb = emb + x
    return emb / K


# ============================================================
# MCR² loss functions
# ============================================================

def compute_discrimn_loss(W: torch.Tensor, eps: float = 0.001) -> torch.Tensor:
    """
    Discriminative term of the MCR² objective (expansion loss).

    Measures the log-determinant of the total covariance:
        R(Z) = (1/2) log|I + (p / m·ε) Z^T Z|

    Parameters
    ----------
    W   : torch.Tensor – encoder output Z  (m × p)
    eps : float        – coding rate precision parameter

    Returns
    -------
    Scalar log-determinant value.
    """
    m, p = W.shape
    I = torch.eye(p)
    scalar = p / (m * eps)
    logdet = torch.logdet(I + scalar * W.T.matmul(W))
    return logdet / 2.0


def compute_compress_loss(W: torch.Tensor,
                          P: torch.Tensor,
                          eps: float = 0.01) -> torch.Tensor:
    """
    Compressive term of the MCR² objective.

    Measures the within-class coding rate given membership matrix P:
        R_c(Z|P) = (1/2) log|I + ε · Z^T P Z|

    Parameters
    ----------
    W   : torch.Tensor – encoder output Z         (m × p)
    P   : torch.Tensor – soft membership matrix   (m × m)
    eps : float        – coding rate precision parameter

    Returns
    -------
    Scalar log-determinant value.
    """
    m, p = W.shape
    I = torch.eye(p)
    Scatter = I + eps * W.T.matmul(P).matmul(W)
    return torch.logdet(Scatter)



def load_adj_scatter(num_nodes):
    """
    Load adjacency scatter matrix with float32 precision
    
    Args:
        num_nodes: Number of nodes in the graph
        
    Returns:
        numpy.ndarray: Adjacency matrix in float32 format
    """
    # Create identity matrix and ones matrix
    e = np.ones((num_nodes, num_nodes), dtype=np.float32)
    
    # Calculate adjacency matrix
    adj_neg = (sp.eye(num_nodes, dtype=np.float32).toarray() - e/float(num_nodes))
    
    # Ensure float32 type
    adj_neg = adj_neg.astype(np.float32)
    
    return adj_neg


# ============================================================
# DARTS MLP encoder
# ============================================================

class GraphDARTS(nn.Module):
    """
    Differentiable Architecture Search (DARTS) MLP encoder.

    The architecture is parameterised by continuous relaxations of:
      - Number of active hidden layers (up to `max_num_layers`)
      - Hidden-layer width (chosen from `hidden_choices`)
      - Activation function (chosen from `activation_choices`)
      - Dropout rate (chosen from `dropout_rates`)
      - Skip-connection strength per layer
      - Output embedding dimension (chosen from `output_choices`)

    During the forward pass each discrete choice is replaced by a
    weighted mixture (softmax weights → α parameters), so the whole
    pipeline is end-to-end differentiable.  After training the
    argmax of each α selects the final discrete architecture.

    Parameters
    ----------
    input_size        : int  – dimension of input features
    output_choices    : list – candidate output embedding dimensions
    max_num_layers    : int  – maximum depth of the MLP supernet
    num_choices       : int  – number of width candidates
    hidden_choices    : list – candidate hidden-layer widths
    activation_choices: list – candidate activation functions
    dropout_rates     : list – candidate dropout rates
    skip_connections  : bool – whether to include learnable skip gates
    seed              : int  – random seed for weight initialisation
    """

    def __init__(
        self,
        input_size: int,
        output_choices: list,
        max_num_layers: int,
        num_choices: int,
        hidden_choices: list,
        activation_choices: list,
        dropout_rates: list,
        skip_connections: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        set_seed(seed)

        self.output_choices = output_choices
        self.dropout_rates = dropout_rates
        self.max_num_layers = max_num_layers
        self.hidden_choices = hidden_choices
        self.activation_choices = activation_choices
        self.skip_connections = skip_connections

        # ── Architecture parameters (α) ──────────────────────────
        gen = lambda s: torch.Generator().manual_seed(s)
        self.alpha_hidden     = nn.Parameter(1e-3 * torch.randn(max_num_layers, num_choices,          generator=gen(seed)))
        self.alpha_activation = nn.Parameter(1e-3 * torch.randn(max_num_layers, len(activation_choices), generator=gen(seed + 1)))
        self.alpha_dropout    = nn.Parameter(1e-3 * torch.randn(max_num_layers, len(dropout_rates),   generator=gen(seed + 2)))
        self.alpha_num_layers = nn.Parameter(1e-3 * torch.randn(max_num_layers,                       generator=gen(seed + 4)))
        self.alpha_output_size = nn.Parameter(1e-3 * torch.randn(len(output_choices),                 generator=gen(seed + 5)))

        if skip_connections:
            self.alpha_skip = nn.Parameter(1e-3 * torch.randn(max_num_layers, generator=gen(seed + 3)))
        else:
            self.alpha_skip = None

        # ── Learnable optimiser hyperparameters ──────────────────
        # These are stored as Parameters so they could in principle
        # be meta-learned; .item() is used when constructing the
        # optimiser to read their scalar value at initialisation.
        self.lr                  = nn.Parameter(torch.tensor(0.0001))
        self.weight_decay        = nn.Parameter(torch.tensor(5e-5))
        self.momentum            = nn.Parameter(torch.tensor(0.9))
        self.batch_norm_momentum = nn.Parameter(torch.tensor(0.1))

        # ── Network weights (w) ──────────────────────────────────
        self.hidden_layers = nn.ModuleList()
        self.projections    = nn.ModuleList()
        self.batch_norms    = nn.ModuleList()
        self.dropouts       = nn.ModuleList()

        current_input_size = input_size
        common_output_size = hidden_choices[-1]  # projection target width

        for _ in range(max_num_layers):
            # One linear sub-layer per width candidate
            layers_per_layer = nn.ModuleList()
            projs_per_layer  = nn.ModuleList()
            for hidden_size in hidden_choices:
                layer = nn.Linear(current_input_size, hidden_size)
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
                layers_per_layer.append(layer)

                proj = nn.Linear(hidden_size, common_output_size)
                nn.init.xavier_uniform_(proj.weight)
                nn.init.zeros_(proj.bias)
                projs_per_layer.append(proj)

            self.hidden_layers.append(layers_per_layer)
            self.projections.append(projs_per_layer)
            self.batch_norms.append(nn.BatchNorm1d(common_output_size))
            self.dropouts.append(
                nn.ModuleList([nn.Dropout(p=r) for r in dropout_rates])
            )
            current_input_size = common_output_size

        # One output projection per output-size candidate
        self.output_layers = nn.ModuleList()
        for size in output_choices:
            out_layer = nn.Linear(common_output_size, size)
            nn.init.xavier_uniform_(out_layer.weight)
            nn.init.zeros_(out_layer.bias)
            self.output_layers.append(out_layer)

    # ── Parameter accessors ──────────────────────────────────────

    def get_arch_parameters(self) -> list:
        """Return only architecture parameters (α)."""
        params = [
            self.alpha_hidden, self.alpha_activation,
            self.alpha_dropout, self.alpha_num_layers,
            self.alpha_output_size,
        ]
        if self.skip_connections:
            params.append(self.alpha_skip)
        return params

    def get_network_parameters(self) -> list:
        """Return only network weights (w), excluding α."""
        return [p for name, p in self.named_parameters() if "alpha" not in name]

    # ── Forward pass ─────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mixed-operation forward pass.

        Active layers and output head are determined by argmax of the
        respective α; all other choices are combined as a weighted sum
        using softmax(α) weights, keeping the graph differentiable.
        """
        layer_weights  = F.softmax(self.alpha_num_layers,  dim=-1)
        output_weights = F.softmax(self.alpha_output_size, dim=-1)

        active_layers     = torch.argmax(layer_weights) + 1   # at least 1 layer
        selected_out_idx  = torch.argmax(output_weights)

        arch_w    = F.softmax(self.alpha_hidden[:active_layers],     dim=-1)
        act_w     = F.softmax(self.alpha_activation[:active_layers], dim=-1)
        dropout_w = F.softmax(self.alpha_dropout[:active_layers],    dim=-1)

        for i in range(active_layers):
            # Weighted sum over width candidates (after projecting to common size)
            h = sum(
                w * self.projections[i][j](self.hidden_layers[i][j](x))
                for j, w in enumerate(arch_w[i])
            )
            h = self.batch_norms[i](h)

            # Weighted sum over activation functions
            h = sum(w * act(h) for w, act in zip(act_w[i], self.activation_choices))

            # Weighted sum over dropout rates
            h = sum(w * drop(h) for w, drop in zip(dropout_w[i], self.dropouts[i]))

            # Learnable skip gate (σ(α_skip) blends residual and new h)
            if self.skip_connections and i > 0:
                skip_w = torch.sigmoid(self.alpha_skip[i])
                h = skip_w * x + (1 - skip_w) * h

            x = h

        return self.output_layers[selected_out_idx](x)

    # ── Architecture inspection ──────────────────────────────────

    def get_current_architecture(self) -> dict:
        """Return the discrete architecture selected by current α values."""
        with torch.no_grad():
            active_layers   = torch.argmax(F.softmax(self.alpha_num_layers,   dim=-1)).item() + 1
            selected_output = self.output_choices[torch.argmax(F.softmax(self.alpha_output_size, dim=-1)).item()]
            arch_w    = F.softmax(self.alpha_hidden,     dim=-1)
            act_w     = F.softmax(self.alpha_activation, dim=-1)
            dropout_w = F.softmax(self.alpha_dropout,    dim=-1)

            config = {"num_layers": active_layers, "output_size": selected_output, "layer_configs": []}
            for i in range(active_layers):
                layer_cfg = {
                    "hidden_size": self.hidden_choices[torch.argmax(arch_w[i]).item()],
                    "activation":  self.activation_choices[torch.argmax(act_w[i]).item()].__name__,
                    "dropout":     self.dropout_rates[torch.argmax(dropout_w[i]).item()],
                }
                if self.skip_connections:
                    layer_cfg["skip_connection"] = torch.sigmoid(self.alpha_skip[i]).item()
                config["layer_configs"].append(layer_cfg)

        return config


# ============================================================
# Sliding-window segmentation
# ============================================================

def create_sliding_windows(data: np.ndarray,
                            labels: np.ndarray,
                            window_size: int = 10,
                            step_size: int = 1):
    """
    Convert a sequence of feature vectors into sliding windows.

    Each window of length `window_size` becomes one time-series sample
    for the Time-Series K-Means clustering step.  The label assigned
    to each window is the label of the *last* observation in the window,
    following the convention used in the original ORION-AE evaluation.

    Parameters
    ----------
    data        : np.ndarray  (n_samples, n_features)
    labels      : np.ndarray  (n_samples,)
    window_size : int
    step_size   : int

    Returns
    -------
    X : np.ndarray  (n_windows, window_size, n_features)
    Y : np.ndarray  (n_windows,)
    """
    X, Y = [], []
    for start in range(0, len(data) - window_size + 1, step_size):
        end = start + window_size
        X.append(data[start:end])
        Y.append(labels[end - 1])
    return np.array(X), np.array(Y)


# ============================================================
# Onset-detection evaluation (recall / precision)
# ============================================================

def load_and_preprocess_mat(data_path: str):
    """
    Load the ORION-AE .mat file and extract the relevant arrays.

    Returns
    -------
    X          : np.ndarray – feature matrix  (n, d)  (first 8 columns removed)
    labels     : np.ndarray – ground-truth labels
    timestamps : np.ndarray – absolute timestamps (column 5 in the mat file)
    list_features, les_duree : raw MATLAB arrays (metadata)
    """
    raw = loadmat(data_path)
    X          = raw["P3"]
    labels     = raw["labels3"]
    list_feats = raw["listFeatures"]
    les_duree  = raw["lesduree"]
    timestamps = X[:, 4].copy()   # column index 4 → absolute time
    X          = X[:, 8:]         # drop the first 8 descriptor columns
    print(f"Data shape:      {X.shape}")
    print(f"Labels shape:    {labels.shape}")
    print(f"Timestamps:      {timestamps.shape}")
    print(f"Unique labels:   {np.unique(labels)}")
    return X, labels, timestamps, list_feats, les_duree


def find_onsets(labels: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """
    Return the timestamps where the label sequence changes value.

    These correspond to the physical instants at which the bolt
    loosening / tightening state transitions occur in the ORION-AE
    experiment.
    """
    labels     = labels.ravel()
    timestamps = timestamps.ravel()
    changes    = np.where(np.diff(labels) != 0)[0]
    if len(changes) == 0:
        print("Warning: no label changes detected.")
        return np.array([])
    return timestamps[changes]


def compute_onset_metrics(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           t1: float = 0.5,
                           t2: float = 0.5) -> dict:
    """
    Tolerance-based precision / recall for onset detection.

    A predicted onset is a true positive if it falls within [t_true - t1,
    t_true + t2] of at least one ground-truth onset.

    Parameters
    ----------
    y_true : np.ndarray – ground-truth onset times (seconds)
    y_pred : np.ndarray – predicted onset times   (seconds)
    t1     : float      – left  tolerance (default 0.5 s)
    t2     : float      – right tolerance (default 0.5 s)

    Returns
    -------
    dict with keys: precision, recall, f1, entropy, TP, FP, FN
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        print("Warning: empty input arrays.")
        return dict(precision=0, recall=0, f1=0, entropy=0, TP=0, FP=0, FN=0)

    eval_onsets = np.zeros(len(y_true))
    FP = 0

    for pred in y_pred:
        matches = np.where((pred >= y_true - t1) & (pred <= y_true + t2))[0]
        if len(matches) == 0:
            FP += 1
        else:
            eval_onsets[matches] += 1

    FN = int(np.sum(eval_onsets == 0))
    TP = int(np.sum(eval_onsets > 0))
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    p       = eval_onsets[eval_onsets > 0] / np.sum(eval_onsets)
    entropy = -np.sum(p * np.log2(p)) / np.log2(len(p)) if len(p) > 1 else 0

    return dict(precision=precision, recall=recall, f1=f1,
                entropy=entropy, TP=TP, FP=FP, FN=FN)


def evaluate_onsets(data_path: str, y_pred: np.ndarray) -> None:
    """
    Load the raw .mat file, derive ground-truth onsets from label changes,
    convert predicted cluster sequence to onsets, and print the metrics.
    """
    X, labels, timestamps, _, _ = load_and_preprocess_mat(data_path)
    true_onsets = find_onsets(labels, timestamps)
    pred_onsets = find_onsets(y_pred, timestamps)
    print(f"\nTrue onsets: {len(true_onsets)}  |  Pred onsets: {len(pred_onsets)}")
    metrics = compute_onset_metrics(true_onsets, pred_onsets)
    print("Onset metrics (0.5 s tolerance):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")


# ============================================================
# SVD dimensionality reduction with energy threshold
# ============================================================

def svd_reduce(data: np.ndarray, energy_threshold: float = 0.99) -> np.ndarray:
    """
    Project embeddings onto the principal subspace that captures
    `energy_threshold` fraction of the total spectral energy.

    The number of retained components n_k is the smallest integer such that:
        Σ_{i=1}^{n_k} σ_i² / Σ_i σ_i²  ≥  energy_threshold

    Using singular values of X instead of eigenvalues of X^T X is
    numerically more stable and directly gives the same subspace.

    Parameters
    ----------
    data             : np.ndarray  (n_samples, n_features)
    energy_threshold : float       – fraction of energy to retain (default 0.99)

    Returns
    -------
    reduced : np.ndarray  (n_samples, n_components)
    """
    data_t = torch.tensor(data, dtype=torch.float32).unsqueeze(0)   # (1, n, d)
    data_t = scale_each_unitary(data_t)                               # unit-norm rows

    # SVD of X (not X^T X): singular values σ carry the same energy information
    _, S, V = torch.svd(data_t)   # S: (1, min(n,d))  V: (1, d, d)
    S = S.squeeze(0)               # (min(n,d),)

    # Determine how many components are needed for `energy_threshold` energy
    energy_cumsum = torch.cumsum(S, dim=0) / S.sum()
    n_components  = int(torch.where(energy_cumsum >= energy_threshold)[0][0].item()) + 1
    print(f"SVD: retaining {n_components} components "
          f"({energy_threshold*100:.0f}% of spectral energy).")

    # Project
    data_reduced = data_t.matmul(V[:, :, :n_components])  # (1, n, n_components)
    data_reduced = center_datas(data_reduced)              # zero-mean, unit-norm
    return data_reduced.squeeze(0).cpu().numpy()


# ============================================================
# Campaign configuration
# ============================================================

# Select campaign: B, C, D, E, or F
CAMPAIGN = "B"

CAMPAIGN_CONFIG = {
    "B": {"input_size": 10_866, "data_path": "mesure_B_DESSERRAGE_TH2_db45_14_sqtwolog_30_80_1100.mat", "n_clusters": 7, "window_size": 40},
    "C": {"input_size":  9_461, "data_path": "mesure_C_DESSERRAGE_TH2_db45_14_sqtwolog_30_80_1100.mat", "n_clusters": 6, "window_size": 40},
    "D": {"input_size":  9_285, "data_path": "mesure_D_DESSERRAGE_TH2_db45_14_sqtwolog_30_80_1100.mat", "n_clusters": 7, "window_size": 40},
    "E": {"input_size": 15_628, "data_path": "mesure_E_DESSERRAGE_TH2_db45_14_sqtwolog_30_80_1100.mat", "n_clusters": 7, "window_size": 40},
    "F": {"input_size": 17_810, "data_path": "mesure_F_DESSERRAGE_TH2_db45_14_sqtwolog_30_80_1100.mat", "n_clusters": 7, "window_size": 40},
}

cfg       = CAMPAIGN_CONFIG[CAMPAIGN]
DATA_PATH = cfg["data_path"]
N_CLUSTERS   = cfg["n_clusters"]
WINDOW_SIZE  = cfg["window_size"]
DEVICE       = "cpu"
LOW_RANK     = 10    # rank of the graph approximation (LRR)
DIFFUSION_K  = 8     # graph diffusion steps
EPOCHS       = 10
SEED         = 42


# ============================================================
# Step 1 – Load pre-extracted wavelet features
# ============================================================
# Features were extracted using db45 wavelet, decomposition level 14,
# sqtwolog threshold, frequency band 30–1100 Hz (Kharrat et al. 2019,
# Ramasso et al. ORION-AE benchmark).

feature = np.genfromtxt(
    f"Features/dataCompain{CAMPAIGN}_features.csv", delimiter=","
)
labels = np.genfromtxt(
    f"Features/dataCompain{CAMPAIGN}_labels.csv", delimiter=","
)
print(f"Feature matrix: {feature.shape}   Labels: {labels.shape}")

feature_t = torch.tensor(feature, dtype=torch.float32).to(DEVICE)
labels_t  = torch.tensor(labels,  dtype=torch.float32).to(DEVICE)
raw_data, raw_labels = feature_t.clone(), labels_t.clone()


# ============================================================
# Step 2 – Build low-rank similarity graph
# ============================================================
# Adjacency W is derived from the Gram matrix of the scaled features,
# kept at rank `LOW_RANK` to suppress noise.  W1 is a normalised
# all-ones matrix used as the compressive membership matrix in MCR².

print("\n[Step 2] Building similarity graph ...")
lap_normalized, adj_normalized = build_graph(feature_t.numpy(), LOW_RANK)
adj_normalized = adj_normalized.to(DEVICE)
lap_normalized = lap_normalized.to(DEVICE)

# Scatter adjacency from utils (used as alternate membership matrix)
adj_scatter = torch.from_numpy(
    load_adj_scatter(cfg["input_size"])
).float().to(DEVICE)


# ============================================================
# Step 3 – Graph diffusion embedding
# ============================================================
# Propagate node features over DIFFUSION_K hops and average the
# intermediate representations.  This integrates multi-scale
# neighbourhood context into each node's representation without
# any additional trainable parameters.

print("\n[Step 3] Graph diffusion embedding ...")
emb = graph_diffusion(feature_t, adj_normalized, K=DIFFUSION_K)
emb = torch.tensor(emb, dtype=torch.float32)


# ============================================================
# Step 4 – Initialise GraphDARTS encoder
# ============================================================

hidden_choices = [2 ** n for n in range(3, 12)]    # 8 ... 2048
output_choices = [2 ** n for n in range(4, 10)]    # 16 ... 512
activation_choices = [
    F.relu, F.leaky_relu, F.elu, F.selu, F.gelu,
    F.tanh, F.sigmoid,
    lambda x: F.relu6(x),
    lambda x: F.celu(x),
    lambda x: F.softplus(x),
]
dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

model = GraphDARTS(
    input_size        = 32,           # dimension of diffused embeddings
    output_choices    = output_choices,
    max_num_layers    = 6,
    num_choices       = len(hidden_choices),
    hidden_choices    = hidden_choices,
    activation_choices= activation_choices,
    dropout_rates     = dropout_rates,
    skip_connections  = True,
    seed              = SEED,
).to(DEVICE)

optimizer = optim.Adam(
    model.parameters(),
    lr           = model.lr.item(),
    weight_decay = model.weight_decay.item(),
)


# ============================================================
# Step 5 – Train GraphDARTS with MCR² loss
# ============================================================
# The MCR² loss maximises between-class coding rate while
# minimising within-class coding rate:
#     L = R_c(Z | A_scatter) – R_c(Z | A_lap)
# where Z is the encoder output, A_scatter is a scatter adjacency
# (within-class membership), and A_lap is the graph Laplacian term.

print("\n[Step 5] Training GraphDARTS ...")

# Profile time and memory usage during training
process   = psutil.Process(os.getpid())
mem_start = process.memory_info().rss / 1024 ** 2
t_start   = time.perf_counter()

model.train()
alpha = 0.0018
log_alpha = math.log(1 + alpha)

for epoch in range(1, EPOCHS + 1):
    optimizer.zero_grad()
    out = model(emb)

    # MCR² objective: compressive loss difference
    loss = log_alpha * (
        compute_compress_loss(out, lap_normalized, alpha)
        - compute_compress_loss(out, adj_scatter,  alpha)
    )

    print(f"  Epoch {epoch:3d}/{EPOCHS}  loss = {loss.item():.6f}")
    loss.backward()
    optimizer.step()

t_end   = time.perf_counter()
mem_end = process.memory_info().rss / 1024 ** 2
print(f"\nTraining time : {t_end - t_start:.4f} s")
print(f"Memory delta  : {mem_end - mem_start:.4f} MB")

# Print selected discrete architecture
arch = model.get_current_architecture()
print(f"\nSelected architecture  ({arch['num_layers']} layers, "
      f"output_dim={arch['output_size']}):")
for i, lc in enumerate(arch["layer_configs"]):
    print(f"  Layer {i+1}: width={lc['hidden_size']:4d}  "
          f"act={lc['activation']}  dropout={lc['dropout']:.2f}  "
          f"skip={lc.get('skip_connection', float('nan')):.4f}")

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
arch_params      = sum(p.numel() for p in model.get_arch_parameters())
net_params       = sum(p.numel() for p in model.get_network_parameters())
print(f"\nTotal parameters   : {total_params:,}")
print(f"Trainable          : {trainable_params:,}")
print(f"Architecture (α)   : {arch_params:,}")
print(f"Network weights (w): {net_params:,}")


# ============================================================
# Step 6 – Extract embeddings and reduce dimensionality (SVD 99%)
# ============================================================
# After training, pass the diffused features through the frozen encoder
# to obtain compact embeddings.  We then reduce their dimension by
# projecting onto the principal subspace that retains 99% of the total
# spectral energy, computed via SVD.

print("\n[Step 6] Extracting embeddings and applying SVD ...")

model.eval()
lap_normalized = adj_normalized = None   # free graph memory

with torch.no_grad():
    embeddings_raw = model(emb).cpu().numpy()

embeddings_raw_t = torch.tensor(embeddings_raw, dtype=torch.float32)
emb_for_tsne     = embeddings_raw.copy()                # save pre-reduction copy
emb_labels       = labels_t.cpu().numpy()

# SVD-based reduction: keep components covering 99% of energy
data_reduced = svd_reduce(embeddings_raw, energy_threshold=0.99)


# ============================================================
# Step 7 – Sliding-window segmentation
# ============================================================
# Time-Series K-Means (TSKMeans) expects 3-D input (n_windows,
# window_size, n_features).  Each window label is the label of
# its last observation.

print("\n[Step 7] Creating sliding windows ...")
STEP_SIZE = 1
data_windowed, labels_windowed = create_sliding_windows(
    data_reduced, labels_t.cpu().numpy(), WINDOW_SIZE, STEP_SIZE
)
print(f"Windowed data : {data_windowed.shape}   Labels: {labels_windowed.shape}")


# ============================================================
# Step 8 – Time-Series K-Means clustering
# ============================================================

print(f"\n[Step 8] Fitting TSKMeans (K={N_CLUSTERS}) ...")

ts_kmeans = TimeSeriesKMeans(
    n_clusters   = N_CLUSTERS,
    metric       = "euclidean",
    max_iter     = 60,
    random_state = SEED,
)

t0 = time.time()
ts_kmeans.fit(data_windowed)
t1 = time.time()
cluster_labels = ts_kmeans.predict(data_windowed)
t2 = time.time()

print(f"  Fit time     : {t1 - t0:.4f} s")
print(f"  Predict time : {t2 - t1:.4f} s")

y_pred = cluster_labels
y_true = labels_windowed


# ============================================================
# Step 9 – Clustering evaluation
# ============================================================

print("\n[Step 9] Evaluation metrics ...")

data_flat = data_windowed.reshape(data_windowed.shape[0], -1)

silhouette   = silhouette_score(data_flat, y_pred)
db_index     = davies_bouldin_score(data_flat, y_pred)
ch_index     = calinski_harabasz_score(data_flat, y_pred)
ari          = adjusted_rand_score(y_true, y_pred)
nmi          = normalized_mutual_info_score(y_true, y_pred)
homogeneity  = homogeneity_score(y_true, y_pred)
completeness = completeness_score(y_true, y_pred)
v_measure    = v_measure_score(y_true, y_pred)

print(f"  Silhouette Score       : {silhouette:.4f}")
print(f"  Davies-Bouldin Index   : {db_index:.4f}")
print(f"  Calinski-Harabasz      : {ch_index:.4f}")
print(f"  Adjusted Rand Index    : {ari:.4f}")
print(f"  Normalised MI          : {nmi:.4f}")
print(f"  Homogeneity            : {homogeneity:.4f}")
print(f"  Completeness           : {completeness:.4f}")
print(f"  V-Measure              : {v_measure:.4f}")


# ============================================================
# Step 10 – Onset-detection evaluation
# ============================================================

print("\n[Step 10] Onset detection ...")
evaluate_onsets(DATA_PATH, y_pred)


# ============================================================
# Step 11 – t-SNE visualisations
# ============================================================

print("\n[Step 11] Generating t-SNE plots ...")

tsne = TSNE(n_components=2, random_state=SEED)

# Embedded features (after encoder)
reduced_emb = tsne.fit_transform(emb_for_tsne)
plt.figure(figsize=(8, 6))
plt.scatter(reduced_emb[:, 0], reduced_emb[:, 1], c=emb_labels, cmap="viridis", s=4)
plt.colorbar()
plt.title(f"t-SNE – Embedded Features  (Campaign {CAMPAIGN})")
plt.xlabel("Dim 1"); plt.ylabel("Dim 2")
plt.tight_layout()
plt.savefig(f"tsne_{CAMPAIGN}_embedded.pdf", bbox_inches="tight")
plt.show()
