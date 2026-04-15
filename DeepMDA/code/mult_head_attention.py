import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.n_head = num_heads
        self.d_k    = hidden_size // num_heads
        assert hidden_size % num_heads == 0

        self.drop   = nn.Dropout(dropout)


        self.w_q = nn.Linear(in_size, hidden_size)
        self.w_k = nn.Linear(in_size, hidden_size)
        self.w_v = nn.Linear(in_size, hidden_size)


        self.fc  = nn.Linear(hidden_size, hidden_size)


        self.ln  = nn.LayerNorm(hidden_size)

    def forward(self, z):

        B, n_view, d = z.shape


        q = self.w_q(z)   # [B, n_view, hidden]
        k = self.w_k(z)
        v = self.w_v(z)


        q = q.view(B, n_view, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(B, n_view, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(B, n_view, self.n_head, self.d_k).transpose(1, 2)


        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)  # [B, n_head, n_view, n_view]
        attn   = torch.softmax(scores, dim=-1)
        attn   = self.drop(attn)

        out = torch.matmul(attn, v)  # [B, n_head, n_view, d_k]


        out = out.transpose(1, 2).contiguous().view(B, n_view, -1)  # [B, n_view, hidden]


        out = self.fc(out)              # [B, n_view, hidden]
        out = self.ln(out + z)


        return out.mean(dim=1), attn.mean(dim=1)