# model_multi_real.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------
# Normalization + Encoder
# -------------------------------------------------------

class FeatureNormalizer(nn.Module):
    """
    Simple per-dimension (x - mu)/sigma.
    mu, sigma are tensors of shape [d_in].
    """
    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor):
        super().__init__()
        self.register_buffer("mu", mu)
        self.register_buffer("sigma", sigma)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.mu) / (self.sigma + 1e-6)


def mlp(in_dim: int, out_dim: int, hidden: int = 128,
        depth: int = 3, act=nn.ReLU) -> nn.Module:
    layers = []
    d = in_dim
    for _ in range(depth - 1):
        layers.append(nn.Linear(d, hidden))
        layers.append(act())
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class FeatureEncoder(nn.Module):
    """
    X_normed -> H embeddings.
    """
    def __init__(self, d_in: int, d_hid: int = 128):
        super().__init__()
        self.net = mlp(d_in, d_hid, hidden=128, depth=3)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)  # [N, d_hid]


class SelfExcitationNet(nn.Module):
    """
    Per-node diagonal self-excitation weights w_self[i,k].
    """
    def __init__(self, d_hid: int, K: int):
        super().__init__()
        self.fc = nn.Linear(d_hid, K)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        return self.fc(H)  # [N,K]


# -------------------------------------------------------
# Multi-scale spatial kernel (per event type)
# -------------------------------------------------------

class MultiScaleSpatialKernel(nn.Module):
    """
    For each event type k, we build an adjacency A_k:

        S_k = H W_k H^T   (learned similarity)
        A_dist_k = exp(-dist / length_k) * 1{dist <= max_dist_k}
        A_k = sparsify_kNN( softplus(S_k) ⊙ A_dist_k )

    Returns: list of K adjacency matrices, each [N,N].
    """
    def __init__(self,
                 d_hid: int,
                 K: int,
                 length_scales,
                 max_dist,
                 k_tops):
        super().__init__()
        self.K = K
        if length_scales is None:
            # default ~hundreds of meters in projected coords;
            # if K>3, just repeat a generic scale
            default_ls = [0.003, 0.001, 0.002]
            if K <= len(default_ls):
                length_scales = default_ls[:K]
            else:
                length_scales = default_ls + [0.002] * (K - len(default_ls))

        if max_dist is None:
            # default cutoff radius
            max_dist = [0.01] * K

        if k_tops is None:
            # default number of nearest neighbors per node per type
            k_tops = [20] * K
        # ------------------------------------------------

        self.length_scales = list(length_scales)
        self.max_dist = list(max_dist)
        self.k_tops = list(k_tops)

        assert len(self.length_scales) == K
        assert len(self.max_dist) == K
        assert len(self.k_tops) == K


        # one W per type
        self.W = nn.ParameterList([
            nn.Parameter(torch.randn(d_hid, d_hid) * 0.1) for _ in range(K)
        ])

        # cache for distance matrix (per coords shape)
        self.register_buffer("_cached_coords", torch.empty(0), persistent=False)
        self.register_buffer("_cached_D", torch.empty(0), persistent=False)

    @staticmethod
    def _compute_distance_matrix(coords: torch.Tensor) -> torch.Tensor:
        """
        coords: [N,2] with columns [lon, lat].
        For Lee County scale, Euclidean in degrees is adequate.
        """
        dx = coords[:, 0:1] - coords[:, 0:1].T
        dy = coords[:, 1:2] - coords[:, 1:2].T
        return torch.sqrt(dx * dx + dy * dy)

    def _get_D(self, coords: torch.Tensor) -> torch.Tensor:
        if (self._cached_coords.numel() == 0 or
            self._cached_coords.shape != coords.shape or
            (self._cached_coords - coords).abs().sum() > 1e-6):
            # re-compute
            D = self._compute_distance_matrix(coords)
            self._cached_coords = coords.detach().clone()
            self._cached_D = D
        return self._cached_D

    def forward(self, H: torch.Tensor, coords: torch.Tensor):
        """
        H: [N, d_hid]
        coords: [N, 2]
        returns: list of length K, each A_k [N,N]
        """
        N = H.shape[0]
        D = self._get_D(coords)

        A_list = []
        for k in range(self.K):
            # learned similarity
            Wk = self.W[k]
            S = H @ Wk @ H.T  # [N,N]
            S = torch.tanh(S / 4.0)
            S = S - torch.diag(torch.diag(S))
            A_learn = F.softplus(S)

            # spatial prior
            length = self.length_scales[k]
            dmax = self.max_dist[k]
            mask = (D <= dmax)
            A_dist = torch.exp(-D / length) * mask.float()

            A_prior = A_dist * A_learn

            # sparsify with k-NN
            kt = min(self.k_tops[k], max(N - 1, 1))
            if kt <= 0:
                A_sparse = A_prior * 0.0
            else:
                top_idx = torch.topk(A_prior, k=kt, dim=1).indices
                A_sparse = torch.zeros_like(A_prior)
                A_sparse.scatter_(1, top_idx, 1.0)
                A_sparse = A_sparse * A_prior

            A_list.append(A_sparse)

        return A_list  # list of [N,N]


# -------------------------------------------------------
# Coupled multi-type Hawkes model for REAL data
# -------------------------------------------------------

class MultiScaleCoupledHawkesReal(nn.Module):
    def __init__(
        self,
        d_in: int,
        K: int,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        d_hid: int = 128,
        length_scales=None,
        max_dist=None,
        k_tops=None,
    ):
        super().__init__()
        self.K = K
        self.norm = FeatureNormalizer(mu=mu, sigma=sigma)
        self.encoder = FeatureEncoder(d_in, d_hid)
        self.self_net = SelfExcitationNet(d_hid, K)
        self.spatial = MultiScaleSpatialKernel(
            d_hid=d_hid,
            K=K,
            length_scales=length_scales,
            max_dist=max_dist,
            k_tops=k_tops,
        )

        # NEW: baseline logit from features
        self.baseline_net = nn.Linear(d_hid, K)

        # Baseline bias per event type
        self.b = nn.Parameter(torch.zeros(K))

        # Coupling weights across event types: self-history & neighbor-history
        self.W_self = nn.Parameter(0.1 * torch.eye(K))  
        self.W_nei  = nn.Parameter(0.1 * torch.eye(K))
        self.alpha_nei = nn.Parameter(torch.ones(K) * 0.1)


    def build_structures(self, X_g: torch.Tensor, coords_g: torch.Tensor):
        Xn = self.norm(X_g)
        H_g = self.encoder(Xn)
        w_self_node = self.self_net(H_g)
        A_list = self.spatial(H_g, coords_g)

        # NEW: per-node baseline logits
        baseline_node = self.baseline_net(H_g)  # [N_g, K]

        return H_g, w_self_node, A_list, baseline_node


    def step_intensity(self,
                       R_self: torch.Tensor,
                       R_nei: torch.Tensor,
                       w_self_node: torch.Tensor,
                       baseline_node: torch.Tensor) -> torch.Tensor:
        """
        Compute logits λ_{i,k}(t) given histories & node-specific params.
        R_self, R_nei: [N,K]
        w_self_node: [N,K]
        baseline_node: [N,K]
        """
        N, K = R_self.shape

        # Optional: force self-excitation non-negative
        w_self_node = torch.clamp(w_self_node, min=0.0)

        diag_term = w_self_node * R_self  # [N,K]

        cross_self = R_self @ self.W_self.T   # [N,K]
        cross_nei  = R_nei  @ self.W_nei.T    # [N,K]
        cross_nei  = cross_nei * self.alpha_nei.view(1, K)

        logits = self.b.view(1, K) + baseline_node + diag_term + cross_self + cross_nei
        logits = torch.clamp(logits, -20.0, 20.0)
        return logits

