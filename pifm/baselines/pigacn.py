"""
PIGACN and GCN baselines.

PIGACN follows Wu et al. (2025), IEEE Trans. Netw. Sci. Eng.: the attention
logits are modulated elementwise by the nodal admittance matrix, so attention
follows electrical rather than purely topological proximity.

    1. Embedding: Dense(feat_dim -> d_model) + ReLU
    2. n_layers PIGACN layers:
           X^(l) = sigma(A_aten_head X^(l-1) W^(l-1)) + X^(l-1)
       with A_aten_head = SoftMax(QK^T / sqrt(dk) * Y) V
       and Y the nodal admittance matrix
    3. Output: two 1x1 convolutional layers, then a linear projection to
       (Vm, Va)

`GCN` is a two-layer graph convolutional network without the admittance
modulation, used as the topology-only control.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DataNormalizer:
    """Data normalization for input/output."""

    def __init__(self, eps=1e-5):
        self.eps = eps
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

    def fit(self, data_list):
        xs = torch.cat([d.x for d in data_list], dim=0)
        ys = torch.cat([d.y for d in data_list], dim=0)
        self.x_mean = xs.mean(dim=0, keepdim=True)
        self.x_std = xs.std(dim=0, keepdim=True) + self.eps
        self.y_mean = ys.mean(dim=0, keepdim=True)
        self.y_std = ys.std(dim=0, keepdim=True) + self.eps

    def normalize_x(self, x):
        if self.x_mean is None:
            return x
        return (x - self.x_mean.to(x.device)) / self.x_std.to(x.device)

    def denormalize_y(self, y):
        if self.y_mean is None:
            return y
        return y * self.y_std.to(y.device) + self.y_mean.to(y.device)

    def normalize_y(self, y):
        if self.y_mean is None:
            return y
        return (y - self.y_mean.to(y.device)) / self.y_std.to(y.device)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (Eq. 12-15 from paper).

    Args:
        d_model: input/output dimension
        n_heads: number of attention heads
        d_k: dimension per head (d_model // n_heads)
    """

    def __init__(self, d_model=10, n_heads=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = np.sqrt(self.d_k)

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, ybus):
        """
        Args:
            x: (N, d_model) node features
            ybus: (N, N) nodal admittance matrix (physics information)

        Returns:
            out: (N, d_model) attended features
        """
        N = x.shape[0]

        # Linear projections
        Q = self.W_q(x)  # (N, d_model)
        K = self.W_k(x)  # (N, d_model)
        V = self.W_v(x)  # (N, d_model)

        # Reshape for multi-head: (N, n_heads, d_k)
        Q = Q.view(N, self.n_heads, self.d_k)
        K = K.view(N, self.n_heads, self.d_k)
        V = V.view(N, self.n_heads, self.d_k)

        # Attention scores: QK^T / sqrt(dk)  → (N, N, n_heads)
        attn_scores = torch.einsum('ihd,jhd->ijh', Q, K) / self.scale

        # === PHYSICS-INFORMED (Eq. 17): Y ⊙ (QK^T/√dk) ===
        # ybus must be (N, N) or broadcastable
        if ybus is not None and ybus.shape[0] == N:
            ybus_expanded = ybus.unsqueeze(-1)  # (N, N, 1)
            attn_scores = attn_scores * ybus_expanded  # Hadamard product

        # Softmax over keys (dim=1)
        attn_weights = F.softmax(attn_scores, dim=1)  # (N, N, n_heads)

        # Weighted sum: attn_weights × V  → (N, n_heads, d_k)
        out = torch.einsum('ijh,jhd->ihd', attn_weights, V)

        # Concatenate heads and project
        out = out.reshape(N, self.d_model)  # (N, d_model)
        out = self.W_o(out)
        return out


class PIGACNLayer(nn.Module):
    """
    Physics-Informed Graph Attention Convolutional Layer (Eq. 17-18).

    X^(l) = sigma(Ã_aten_head X^(l-1) W) + X^(l-1)
    """

    def __init__(self, d_model=10, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.W = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, ybus):
        """
        Args:
            x: (N, d_model) input features
            ybus: (N, N) admittance matrix for physics-informed attention

        Returns:
            out: (N, d_model) output features
        """
        # Graph attention with physics: A_aten_head × X × W
        attended = self.attention(x, ybus)  # (N, d_model)
        transformed = self.W(attended)  # (N, d_model)

        # Residual connection (Eq. 18)
        out = self.norm(F.relu(transformed) + x)
        return out


class PIGACN(nn.Module):
    """
    Physics-Informed Graph Attention Convolutional Network.

    Architecture:
    - 1 dense embedding layer: input channels -> d_model
    - n_layers PIGACN layers (default d_model=10, n_heads=2, 3 layers)
    - 2 convolutional layers (1x1 kernels) -> linear projection to the
      output state
    """

    def __init__(self, in_channels=4, out_channels=2, d_model=10, n_heads=2, n_layers=3):
        super().__init__()
        self.d_model = d_model

        # Embedding: input → d_model (Eq. 11)
        self.embedding = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.ReLU(),
        )

        # PIGACN layers
        self.pigacn_layers = nn.ModuleList([
            PIGACNLayer(d_model, n_heads) for _ in range(n_layers)
        ])

        # Output: 2-layer CNN (channels 4→1, kernel 1×1)
        # Input: (1, d_model, N) → Conv2d
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(4, 1, kernel_size=(1, 1)),
        )

        # Final linear for output dimension
        self.output_proj = nn.Linear(d_model, out_channels)

    def forward(self, x, edge_index=None, edge_attr=None, obs_mask=None):
        """
        Args:
            x: (N, F_in) measurement features
            edge_index: (2, E) graph connectivity (for building Ybus)
            edge_attr: (E, 1) normalized Ybus values (for building Ybus matrix)
            obs_mask: (N,) observation mask (ignored in this architecture)

        Returns:
            out: (N, F_out) estimated states (Vm, Va)
        """
        N = x.shape[0]

        # Build Ybus matrix from edge_attr if available
        ybus_matrix = None
        if edge_index is not None and edge_attr is not None:
            ybus_matrix = torch.zeros(N, N, device=x.device)
            row, col = edge_index
            weights = edge_attr.squeeze(-1)
            ybus_matrix[row, col] = weights
            # Symmetrize
            ybus_matrix = (ybus_matrix + ybus_matrix.t()) / 2

        # Embedding
        h = self.embedding(x)  # (N, d_model)

        # PIGACN layers
        for layer in self.pigacn_layers:
            h = layer(h, ybus_matrix)

        # CNN output mapping
        h_cnn = h.unsqueeze(0).unsqueeze(0)  # (1, 1, N, d_model)
        h_cnn = self.cnn(h_cnn)  # (1, 1, N, d_model)
        h_cnn = h_cnn.squeeze(0).squeeze(0)  # (N, d_model)

        # Project to output dim
        out = self.output_proj(h_cnn)  # (N, 2)

        return out


class GCN(nn.Module):
    """GCN baseline: two graph convolutional layers."""

    def __init__(self, in_dim=4, out_dim=2):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.gcn1 = GCNConv(in_dim, 64)
        self.gcn2 = GCNConv(64, out_dim)

    def forward(self, x, edge_index, edge_attr=None, obs_mask=None):
        ew = edge_attr.squeeze(-1) if edge_attr is not None else None
        h = F.relu(self.gcn1(x, edge_index, ew))
        return self.gcn2(h, edge_index, ew)
