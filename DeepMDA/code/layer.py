import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F
from parms_setting import settings
from lightgcn_layer import LightGCN
from mult_head_attention import MultiHeadSelfAttention   # 替换原来的 Attention
args = settings()


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta

class DeepMDA(nn.Module):
    def __init__(self, feature, hidden2, decoder1):
        super().__init__()

        self.light_m_s = LightGCN(args.miRNA_number, hidden2)   # miRNA sequence
        self.light_m_r = LightGCN(args.miRNA_number, hidden2)   # miRNA-MDA
        self.light_d_f = LightGCN(args.drug_number,   hidden2)  # drug structure
        self.light_d_g = LightGCN(args.drug_number,   hidden2)  # drug-gene
        self.light_d_m = LightGCN(args.drug_number,   hidden2)  # drug-MDA

        self.attention_m = MultiHeadSelfAttention(hidden2, hidden2, num_heads=4, dropout=0.1)
        self.attention_d = MultiHeadSelfAttention(hidden2, hidden2, num_heads=4, dropout=0.1)

        self.decoder1 = nn.Linear(hidden2 * 4, decoder1)
        self.decoder2 = nn.Linear(decoder1, 1)

    def forward(self, data, idx):
        z_m_s = self.light_m_s(data['mm_s']['edges'].cuda())
        z_m_r = self.light_m_r(data['mm_r']['edges'].cuda())

        z_d_f = self.light_d_f(data['dd_f']['edges'].cuda())
        z_d_g = self.light_d_g(data['dd_g']['edges'].cuda())
        z_d_m = self.light_d_m(data['dd_m']['edges'].cuda())

        x_m = torch.stack([z_m_s, z_m_r], dim=1)
        x_m, att_m = self.attention_m(x_m)

        y_d = torch.stack([z_d_f, z_d_g, z_d_m], dim=1)
        y_d, att_d = self.attention_d(y_d)

        entity1 = x_m[idx[0]]
        entity2 = y_d[idx[1]]
        add = entity1 + entity2
        product = entity1 * entity2
        concatenate = torch.cat((entity1, entity2), dim=1)
        feature = torch.cat((add, product, concatenate), dim=1)

        log = F.relu(self.decoder1(feature))
        return self.decoder2(log)












