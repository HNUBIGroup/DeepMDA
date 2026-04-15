
import torch
import torch.nn as nn
from torch_geometric.utils import add_self_loops, degree

class LightGCN(nn.Module):

    def __init__(self, num_nodes, emb_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(num_nodes, emb_dim))

    def forward(self, edge_index):

        N = self.embedding.shape[0]
        device = self.embedding.device

        edge_index, _ = add_self_loops(edge_index, num_nodes=N)

        row, col = edge_index
        deg = degree(row, num_nodes=N, dtype=torch.float)        # [N]
        deg_inv_sqrt = deg.pow(-0.5)                            # [N]
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]            # [E]

        adj_t = torch.sparse_coo_tensor(
            indices=torch.stack([col, row]), values=norm, size=(N, N)
        ).to(device)

        out = torch.sparse.mm(adj_t, self.embedding)            # [N, emb_dim]
        return out