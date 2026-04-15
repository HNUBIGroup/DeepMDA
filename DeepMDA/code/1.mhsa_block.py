import torch.nn as nn

class MHSAblock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.mha  = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, n_view, d]
        x_t   = x.transpose(0, 1)                      # [n_view, B, d]
        out, _ = self.mha(x_t, x_t, x_t)               # 自注意力
        out   = self.drop(out)
        x     = self.norm(x_t + out)                   # 残差
        return x.transpose(0, 1).mean(dim=1), _        # 返回 [B, d]