import torch
import torch.nn as nn

class CalibratedDecoder(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc = nn.Linear(in_features, 1)
        self.T  = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # 关键修复：强制fc层参数和T都移到x的设备
        self.fc = self.fc.to(x.device)
        T = self.T.to(x.device)
        return self.fc(x) / T