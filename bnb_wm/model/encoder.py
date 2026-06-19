"""
encoder.py — Bipartite GNN encoder for B&B nodes.

Represents a B&B node as a bipartite constraint-variable graph and
produces per-variable embeddings h_vars and a graph-level embedding z.

Node types:
    0 = variable node   (19-dim features)
    1 = constraint node  (5-dim features)

Edge features (per constraint-variable edge, 3-dim):
    0 = constraint coefficient A_{ij}
    1 = normalised coefficient  A_{ij} / (|RHS_i| + 1e-8)
    2 = sign of coefficient     sign(A_{ij})

Architecture changes vs. original:
    SAGEConv  -> GATv2Conv  (attention-weighted message passing)
    global_mean_pool -> CrossAttentionPool  ([CLS]-style learned readout)
    Edge features injected into every GATv2 message
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class CrossAttentionPool(nn.Module):
    """
    Replaces global_mean_pool with a learned [CLS]-token readout.

    A single learnable query vector attends over all variable-node
    embeddings in each graph, producing a graph-level vector z that
    emphasises structurally important variables (e.g. fractional,
    high-objective) rather than averaging everything equally.

    Input  : h_vars [total_vars, H], batch_vec [total_vars]
    Output : z      [batch_size,  H]
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, hidden_dim))
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = hidden_dim ** -0.5

    def forward(self, h_vars: torch.Tensor, batch_vec: torch.Tensor) -> torch.Tensor:
        batch_size = int(batch_vec.max().item()) + 1
        keys   = self.W_k(h_vars)   # [total_vars, H]
        values = self.W_v(h_vars)   # [total_vars, H]

        out = []
        for b in range(batch_size):
            mask = batch_vec == b
            k = keys[mask]           # [n_vars_b, H]
            v = values[mask]         # [n_vars_b, H]
            attn = (self.query @ k.T) * self.scale   # [1, n_vars_b]
            attn = F.softmax(attn, dim=-1)
            out.append((attn @ v).squeeze(0))        # [H]

        return torch.stack(out, dim=0)               # [batch_size, H]


class BipartiteGNN(nn.Module):
    """
    Two-pass bipartite message passing with GATv2Conv and edge features.

    Pass A: constraints -> variables  (GATv2Conv with edge_attr)
    Pass B: variables  -> constraints (GATv2Conv with edge_attr)

    Both passes run each layer before moving to the next, with residual
    connections and LayerNorm on each update.

    Graph-level pooling uses CrossAttentionPool instead of global_mean_pool.

    Args:
        var_dim    : dimension of raw variable features (default 19)
        con_dim    : dimension of raw constraint features (default 5)
        edge_dim   : dimension of edge features (default 3)
        hidden_dim : internal embedding dimension (default 128)
        n_layers   : number of bipartite message-passing rounds (default 3)
        n_heads    : number of GAT attention heads (default 4)
    """

    def __init__(
        self,
        var_dim: int = 19,
        con_dim: int = 5,
        edge_dim: int = 3,
        hidden_dim: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Input projections
        self.var_proj = nn.Linear(var_dim, hidden_dim)
        self.con_proj = nn.Linear(con_dim, hidden_dim)

        # GATv2Conv layers — each head outputs hidden_dim // n_heads features,
        # concat=True means output dim = hidden_dim again after concatenation.
        head_dim = hidden_dim // n_heads
        self.conv_c2v = nn.ModuleList([
            GATv2Conv(hidden_dim, head_dim, heads=n_heads,
                      edge_dim=edge_dim, concat=True, add_self_loops=False)
            for _ in range(n_layers)
        ])
        self.conv_v2c = nn.ModuleList([
            GATv2Conv(hidden_dim, head_dim, heads=n_heads,
                      edge_dim=edge_dim, concat=True, add_self_loops=False)
            for _ in range(n_layers)
        ])

        self.norm_var = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(n_layers)]
        )
        self.norm_con = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(n_layers)]
        )

        # Graph-level readout
        self.pool = CrossAttentionPool(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_type: torch.Tensor,
        batch_vec: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ):
        """
        Args:
            x          : [N, 19]  combined node feature matrix
            edge_index : [2, E]   edges (any direction; split internally)
            node_type  : [N]      0=variable, 1=constraint
            batch_vec  : [N]      batch assignment
            edge_attr  : [E, 3]   per-edge features (optional; zeros if None)

        Returns:
            h_vars : [num_vars, hidden_dim]
            z      : [batch_size, hidden_dim]
        """
        var_mask = node_type == 0
        con_mask = node_type == 1

        h_v = F.relu(self.var_proj(x[var_mask]))
        h_c = F.relu(self.con_proj(x[con_mask][:, :5]))

        h = torch.zeros(x.size(0), self.hidden_dim, device=x.device, dtype=h_v.dtype)
        h[var_mask] = h_v
        h[con_mask] = h_c

        # Split edges
        src, dst = edge_index
        c2v_mask = (node_type[src] == 1) & (node_type[dst] == 0)
        v2c_mask = (node_type[src] == 0) & (node_type[dst] == 1)
        edge_c2v = edge_index[:, c2v_mask]
        edge_v2c = edge_index[:, v2c_mask]

        # Build edge features (zeros when not provided)
        if edge_attr is None:
            edge_attr = torch.zeros(edge_index.size(1), 3,
                                    device=x.device, dtype=h_v.dtype)
        attr_c2v = edge_attr[c2v_mask]
        attr_v2c = edge_attr[v2c_mask]

        for i in range(self.n_layers):
            # Constraints -> variables
            upd_v = self.conv_c2v[i](h, edge_c2v, edge_attr=attr_c2v)[var_mask]
            upd_v = self.norm_var[i](F.relu(upd_v)).to(h.dtype)

            # Variables -> constraints
            upd_c = self.conv_v2c[i](h, edge_v2c, edge_attr=attr_v2c)[con_mask]
            upd_c = self.norm_con[i](F.relu(upd_c)).to(h.dtype)

            # Residual update — in-place scatter to avoid h.clone()
            h = h.index_put((var_mask.nonzero(as_tuple=True)[0],),
                            h[var_mask] + upd_v)
            h = h.index_put((con_mask.nonzero(as_tuple=True)[0],),
                            h[con_mask] + upd_c)

        h_vars = h[var_mask]
        z = self.pool(h_vars, batch_vec[var_mask])

        return h_vars, z
