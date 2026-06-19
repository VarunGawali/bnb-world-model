"""
encoder.py — Bipartite GNN encoder for B&B nodes.

Represents a B&B node as a bipartite constraint-variable graph and
produces per-variable embeddings h_vars and a graph-level embedding z.

Node types:
    0 = variable node  (19-dim features)
    1 = constraint node (5-dim features)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool


class BipartiteGNN(nn.Module):
    """
    Two-pass bipartite message passing:
        Pass A: constraints -> variables
        Pass B: variables  -> constraints

    Both passes run each layer before moving to the next layer,
    with residual connections and LayerNorm on each update.
    """

    def __init__(
        self,
        var_dim: int = 19,
        con_dim: int = 5,
        hidden_dim: int = 128,
        n_layers: int = 3,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Input projections
        self.var_proj = nn.Linear(var_dim, hidden_dim)
        self.con_proj = nn.Linear(con_dim, hidden_dim)

        # Bipartite message-passing layers
        self.conv_c2v = nn.ModuleList(
            [SAGEConv(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.conv_v2c = nn.ModuleList(
            [SAGEConv(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )

        self.norm_var = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(n_layers)]
        )
        self.norm_con = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(n_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_type: torch.Tensor,
        batch_vec: torch.Tensor,
    ):
        """
        Args:
            x          : [N, 19]  combined node feature matrix (vars padded to 19)
            edge_index : [2, E]   constraint->variable edges
            node_type  : [N]      0=variable, 1=constraint
            batch_vec  : [N]      batch assignment vector

        Returns:
            h_vars : [num_vars, hidden_dim]   per-variable embeddings
            z      : [batch_size, hidden_dim] graph-level embedding
        """
        var_mask = node_type == 0
        con_mask = node_type == 1

        # Split raw features and project
        x_var = x[var_mask]            # [num_vars, 19]
        x_con = x[con_mask][:, :5]     # [num_cons,  5]

        h_v = F.relu(self.var_proj(x_var))
        h_c = F.relu(self.con_proj(x_con))

        # Assemble combined hidden tensor
        h = torch.zeros(
            x.size(0), self.hidden_dim, device=x.device, dtype=h_v.dtype
        )
        h[var_mask] = h_v
        h[con_mask] = h_c

        # Split edges by direction
        src, dst = edge_index
        c2v_mask = (node_type[src] == 1) & (node_type[dst] == 0)
        v2c_mask = (node_type[src] == 0) & (node_type[dst] == 1)
        edge_c2v = edge_index[:, c2v_mask]
        edge_v2c = edge_index[:, v2c_mask]

        # Message passing
        for i in range(self.n_layers):
            upd_v = self.norm_var[i](
                F.relu(self.conv_c2v[i](h, edge_c2v)[var_mask])
            ).to(h.dtype)

            upd_c = self.norm_con[i](
                F.relu(self.conv_v2c[i](h, edge_v2c)[con_mask])
            ).to(h.dtype)

            h_new = h.clone()
            h_new[var_mask] = h[var_mask] + upd_v
            h_new[con_mask] = h[con_mask] + upd_c
            h = h_new

        h_vars = h[var_mask]
        z = global_mean_pool(h_vars, batch_vec[var_mask])

        return h_vars, z
